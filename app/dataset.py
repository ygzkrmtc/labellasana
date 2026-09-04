"""
Veri seti operasyonları: bölme (split), istatistik ve YOLOv8 dışa aktarma.

Bu modül de bilerek Qt'den bağımsızdır; bölme mantığı ve istatistikler
arayüz olmadan test edilebilir.
"""

import os
import shutil
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import yolo_io

SPLITS = ("train", "val", "test")

# Göreli alana göre nesne büyüklüğü eşikleri (COCO'nun küçük/orta/büyük
# ayrımının normalize karşılığı: 32² ve 96² piksel ≈ 640² resimde 0.0025 / 0.0225)
SMALL_AREA = 0.0025
MEDIUM_AREA = 0.0225

HEATMAP_GRID = 12


# ------------------------------------------------------------------ bölme
def image_signature(txt_path: str) -> tuple:
    """Bir resmi, içerdiği sınıf kümesiyle temsil eder (katmanlı bölme için)."""
    return tuple(sorted({row[0] for row in yolo_io.load_raw(txt_path)}))


def stratified_split(items: Sequence[Tuple[str, tuple]],
                     ratios: Tuple[float, float, float],
                     seed: int = 42) -> Dict[str, List[str]]:
    """
    Resimleri sınıf bileşimlerini koruyarak train/val/test'e böler.

    items: (resim_yolu, sınıf_imzası) çiftleri
    ratios: (train, val, test) — toplamları 1 olmak zorunda değil, normalize edilir

    Aynı `seed` her zaman aynı bölmeyi üretir; deney tekrarlanabilirliği için şart.
    """
    import random

    total = float(sum(ratios))
    if total <= 0:
        raise ValueError("Bölme oranlarının toplamı sıfır olamaz.")
    train_r, val_r, test_r = (r / total for r in ratios)

    groups: Dict[tuple, List[str]] = defaultdict(list)
    for path, signature in items:
        groups[signature].append(path)

    result: Dict[str, List[str]] = {name: [] for name in SPLITS}
    rng = random.Random(seed)

    # Grupları deterministik sırada gez ki sonuç işletim sisteminden bağımsız olsun
    for signature in sorted(groups, key=lambda s: (len(s), s)):
        paths = sorted(groups[signature])
        rng.shuffle(paths)
        n = len(paths)
        n_train = min(int(round(n * train_r)), n)
        n_val = min(int(round(n * val_r)), n - n_train)
        n_test = n - n_train - n_val
        if test_r <= 0 and n_test > 0:        # test istenmiyorsa artığı doğrulamaya ver
            n_val += n_test
            n_test = 0
        if val_r <= 0 and n_val > 0:
            n_train += n_val
            n_val = 0
        result["train"].extend(paths[:n_train])
        result["val"].extend(paths[n_train:n_train + n_val])
        result["test"].extend(paths[n_train + n_val:])

    for name in SPLITS:
        result[name].sort()
    return result


# ------------------------------------------------------------------ istatistik
def collect_stats(image_paths: Sequence[str], label_dir: str,
                  class_count: int) -> dict:
    """Veri setinin sağlık raporunu çıkarır."""
    per_class = Counter()
    size_buckets = Counter()
    boxes_per_image: List[int] = []
    heatmap = [[0] * HEATMAP_GRID for _ in range(HEATMAP_GRID)]
    area_sum = defaultdict(float)

    labeled = empty = missing = 0
    unknown_ids = set()

    for path in image_paths:
        txt = yolo_io.find_existing_label(path, label_dir)
        if not txt:
            missing += 1
            boxes_per_image.append(0)
            continue
        rows = yolo_io.load_raw(txt)
        if rows:
            labeled += 1
        else:
            empty += 1
        boxes_per_image.append(len(rows))

        for cid, xc, yc, w, h in rows:
            per_class[cid] += 1
            if cid >= class_count:
                unknown_ids.add(cid)
            area = w * h
            area_sum[cid] += area
            if area < SMALL_AREA:
                size_buckets["küçük"] += 1
            elif area < MEDIUM_AREA:
                size_buckets["orta"] += 1
            else:
                size_buckets["büyük"] += 1

            col = min(HEATMAP_GRID - 1, max(0, int(xc * HEATMAP_GRID)))
            row = min(HEATMAP_GRID - 1, max(0, int(yc * HEATMAP_GRID)))
            heatmap[row][col] += 1

    total_boxes = sum(per_class.values())
    mean_area = {cid: (area_sum[cid] / per_class[cid]) for cid in per_class if per_class[cid]}

    return {
        "images": len(image_paths),
        "labeled": labeled,
        "empty": empty,
        "missing": missing,
        "total_boxes": total_boxes,
        "per_class": per_class,
        "mean_area": mean_area,
        "size_buckets": size_buckets,
        "boxes_per_image": boxes_per_image,
        "heatmap": heatmap,
        "unknown_ids": sorted(unknown_ids),
    }


def imbalance_ratio(per_class: Counter) -> float:
    """En kalabalık sınıfın en seyrek sınıfa oranı. 1.0 = kusursuz denge."""
    if not per_class:
        return 0.0
    values = [v for v in per_class.values() if v > 0]
    if not values:
        return 0.0
    return max(values) / min(values)


# ------------------------------------------------------------------ dışa aktarma
def export_dataset(split: Dict[str, List[str]], label_dir: str, out_dir: str,
                   class_names: Sequence[str], link_mode: str = "copy",
                   progress: Optional[Callable[[int, int, str], bool]] = None) -> dict:
    """
    YOLOv8'in beklediği klasör yapısını ve data.yaml dosyasını üretir.

    link_mode: "copy" (güvenli, yer kaplar) veya "hardlink" (aynı diskte yer
    kaplamaz; başarısız olursa sessizce kopyalamaya düşer).
    progress: (yapılan, toplam, dosya_adı) -> devam edilsin mi? False = iptal.
    """
    out_dir = os.path.abspath(out_dir)
    for name in SPLITS:
        os.makedirs(os.path.join(out_dir, "images", name), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", name), exist_ok=True)

    total = sum(len(v) for v in split.values())
    done = 0
    copied = Counter()
    empty_labels = 0

    for name in SPLITS:
        for image_path in split[name]:
            filename = os.path.basename(image_path)
            stem = os.path.splitext(filename)[0]
            image_target = os.path.join(out_dir, "images", name, filename)
            label_target = os.path.join(out_dir, "labels", name, stem + ".txt")

            _place_file(image_path, image_target, link_mode)

            source_label = yolo_io.find_existing_label(image_path, label_dir)
            if source_label and os.path.isfile(source_label):
                _place_file(source_label, label_target, "copy")
                if os.path.getsize(source_label) == 0:
                    empty_labels += 1
            else:
                # Etiketsiz resim = arka plan örneği; YOLO boş .txt bekler
                with open(label_target, "w", encoding="utf-8"):
                    pass
                empty_labels += 1

            copied[name] += 1
            done += 1
            if progress is not None and not progress(done, total, filename):
                return {"cancelled": True, "counts": copied, "out_dir": out_dir}

    yaml_path = os.path.join(out_dir, "data.yaml")
    write_data_yaml(yaml_path, out_dir, class_names,
                    include_test=bool(split.get("test")))

    return {
        "cancelled": False,
        "counts": copied,
        "total": total,
        "empty_labels": empty_labels,
        "out_dir": out_dir,
        "yaml": yaml_path,
    }


def _place_file(source: str, target: str, mode: str) -> None:
    if os.path.exists(target):
        try:
            os.remove(target)
        except OSError:
            pass
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except (OSError, NotImplementedError, AttributeError):
            pass          # farklı disk / dosya sistemi -> kopyalamaya düş
    shutil.copy2(source, target)


def write_data_yaml(path: str, root: str, class_names: Sequence[str],
                    include_test: bool = True) -> None:
    """Ultralytics'in okuduğu data.yaml (elle yazılır, PyYAML bağımlılığı yok)."""
    lines = [
        "# YOLOv8 veri seti tanımı - Etiketleme Aracı tarafından üretildi",
        f"path: {root.replace(os.sep, '/')}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines += ["", f"nc: {len(class_names)}", "names:"]
    for index, name in enumerate(class_names):
        lines.append(f"  {index}: {name}")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
