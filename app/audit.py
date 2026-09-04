"""
Veri seti kalite denetimi.

Kötü etiket, eksik etiketten daha zararlıdır: model yanlışı öğrenir. Bu modül
eğitime vermeden önce veri setini tarar ve tıklanabilir bulgular üretir.
Qt bilmez, tamamen test edilebilir.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import yolo_io

ERROR = "hata"
WARNING = "uyarı"
INFO = "bilgi"

_SEVERITY_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}


@dataclass
class Finding:
    """Tek bir denetim bulgusu."""
    severity: str
    category: str
    message: str
    image: str = ""          # "" ise veri seti geneli
    box_index: int = -1      # -1 ise belirli bir kutuya bağlı değil

    @property
    def image_name(self) -> str:
        return os.path.basename(self.image) if self.image else "(veri seti)"


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """İki normalize kutunun (xc, yc, w, h) kesişim/birleşim oranı."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def audit(image_paths: Sequence[str], label_dir: str,
          class_names: Sequence[str],
          min_relative_size: float = 0.01,
          duplicate_iou: float = 0.90,
          max_aspect_ratio: float = 20.0,
          border_epsilon: float = 0.002,
          imbalance_factor: float = 10.0) -> List[Finding]:
    """Veri setini tarar; bulguları ağırlıklarına göre sıralı döndürür."""
    findings: List[Finding] = []
    class_count = len(class_names)
    per_class = {}
    referenced_labels = set()
    unlabeled = 0

    for image_path in image_paths:
        txt = yolo_io.find_existing_label(image_path, label_dir)
        if txt:
            referenced_labels.add(os.path.normcase(os.path.abspath(txt)))
        rows = yolo_io.load_raw(txt) if txt else []

        if not rows:
            unlabeled += 1
            continue

        for index, (class_id, xc, yc, w, h) in enumerate(rows):
            per_class[class_id] = per_class.get(class_id, 0) + 1

            if class_id >= class_count:
                findings.append(Finding(
                    ERROR, "tanımsız-sınıf",
                    f"{class_id} numaralı sınıf classes.txt'te yok.",
                    image_path, index))

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0) or w > 1.0 or h > 1.0:
                findings.append(Finding(
                    ERROR, "aralık-dışı",
                    "Koordinatlar 0-1 aralığının dışında; dosya bozulmuş olabilir.",
                    image_path, index))

            if w < min_relative_size or h < min_relative_size:
                findings.append(Finding(
                    WARNING, "çok-küçük-kutu",
                    f"Kutu çok küçük (%{w * 100:.1f} × %{h * 100:.1f}). "
                    f"Kazara tıklama olabilir.",
                    image_path, index))

            longer, shorter = (w, h) if w >= h else (h, w)
            if shorter > 0 and longer / shorter > max_aspect_ratio:
                findings.append(Finding(
                    WARNING, "aşırı-oran",
                    f"En-boy oranı aşırı ({longer / shorter:.0f}:1). "
                    f"Kenar yanlışlıkla sürüklenmiş olabilir.",
                    image_path, index))

            x1, y1 = xc - w / 2, yc - h / 2
            x2, y2 = xc + w / 2, yc + h / 2
            if (x1 <= border_epsilon or y1 <= border_epsilon
                    or x2 >= 1.0 - border_epsilon or y2 >= 1.0 - border_epsilon):
                findings.append(Finding(
                    INFO, "kenara-yapışık",
                    "Kutu resmin kenarına değiyor; nesne kadrajdan taşmış olabilir.",
                    image_path, index))

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i][0] != rows[j][0]:
                    continue
                overlap = iou(rows[i][1:], rows[j][1:])
                if overlap >= duplicate_iou:
                    findings.append(Finding(
                        WARNING, "yinelenen-kutu",
                        f"{i + 1}. ve {j + 1}. kutular neredeyse aynı "
                        f"(IoU {overlap:.2f}); aynı nesne iki kez etiketlenmiş.",
                        image_path, j))

    # -------------------------------------------------- veri seti geneli
    if unlabeled:
        findings.append(Finding(
            INFO, "etiketsiz-resim",
            f"{unlabeled} resimde hiç kutu yok. Bunlar arka plan örneği sayılır; "
            f"kasıtlı değilse etiketlemeyi tamamlayın."))

    orphans = _orphan_label_files(label_dir, referenced_labels)
    if orphans:
        findings.append(Finding(
            WARNING, "yetim-etiket",
            f"{len(orphans)} etiket dosyasının karşılığı olan resim yok: "
            f"{', '.join(os.path.basename(p) for p in orphans[:5])}"
            f"{'...' if len(orphans) > 5 else ''}"))

    counts = [per_class.get(i, 0) for i in range(class_count)]
    used = [c for c in counts if c > 0]
    if len(used) > 1 and max(used) / min(used) >= imbalance_factor:
        biggest = counts.index(max(used))
        smallest = counts.index(min(used))
        findings.append(Finding(
            WARNING, "sınıf-dengesizliği",
            f"'{class_names[biggest]}' sınıfı '{class_names[smallest]}' sınıfından "
            f"{max(used) / min(used):.0f} kat fazla. Model seyrek sınıfı "
            f"öğrenmekte zorlanır."))

    unused = [class_names[i] for i in range(class_count) if counts[i] == 0]
    if unused:
        findings.append(Finding(
            INFO, "kullanılmayan-sınıf",
            f"Hiç kullanılmayan sınıflar: {', '.join(unused)}. "
            f"data.yaml'a yine de yazılırlar."))

    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9),
                                 f.category, f.image_name, f.box_index))
    return findings


def _orphan_label_files(label_dir: str, referenced: set) -> List[str]:
    """Resmi olmayan .txt dosyaları."""
    if not label_dir or not os.path.isdir(label_dir):
        return []
    orphans = []
    try:
        entries = os.listdir(label_dir)
    except OSError:
        return []
    for name in entries:
        if not name.lower().endswith(".txt") or name.lower() == "classes.txt":
            continue
        path = os.path.join(label_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.normcase(os.path.abspath(path)) not in referenced:
            orphans.append(path)
    return sorted(orphans)


def summarize(findings: Sequence[Finding]) -> dict:
    """Ağırlığa göre sayım — başlıkta göstermek için."""
    result = {ERROR: 0, WARNING: 0, INFO: 0}
    for finding in findings:
        result[finding.severity] = result.get(finding.severity, 0) + 1
    return result


def filter_by(findings: Sequence[Finding], severity: Optional[str] = None,
              category: Optional[str] = None) -> List[Finding]:
    return [f for f in findings
            if (severity is None or f.severity == severity)
            and (category is None or f.category == category)]
