"""
YOLOv8 etiket dosyası okuma/yazma katmanı (Qt bağımsız, birim testi yapılabilir).

Format:
    <class_id> <x_center> <y_center> <width> <height>
Tüm koordinatlar resim genişliğine/yüksekliğine bölünerek 0-1 aralığına
normalize edilir. Satır başına bir kutu, ondalık ayırıcı nokta.
"""

import os
from typing import Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif")

# (class_id, x1, y1, x2, y2) piksel uzayında
PixelBox = Tuple[int, float, float, float, float]


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# ---------------------------------------------------------------- matematik
def to_yolo(x1: float, y1: float, x2: float, y2: float,
            img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Piksel köşe koordinatlarını normalize merkez/boyut formatına çevirir."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Geçersiz resim boyutu.")
    left, right = (x1, x2) if x1 <= x2 else (x2, x1)
    top, bottom = (y1, y2) if y1 <= y2 else (y2, y1)

    # Kutu resmin dışına taşmışsa kırp
    left = max(0.0, min(float(img_w), left))
    right = max(0.0, min(float(img_w), right))
    top = max(0.0, min(float(img_h), top))
    bottom = max(0.0, min(float(img_h), bottom))

    xc = ((left + right) / 2.0) / img_w
    yc = ((top + bottom) / 2.0) / img_h
    w = (right - left) / img_w
    h = (bottom - top) / img_h
    return _clamp01(xc), _clamp01(yc), _clamp01(w), _clamp01(h)


def from_yolo(xc: float, yc: float, w: float, h: float,
              img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Normalize YOLO değerlerini piksel köşe koordinatlarına geri çevirir."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Geçersiz resim boyutu.")
    cx, cy = xc * img_w, yc * img_h
    bw, bh = w * img_w, h * img_h
    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    x2 = min(float(img_w), cx + bw / 2.0)
    y2 = min(float(img_h), cy + bh / 2.0)
    return x1, y1, x2, y2


# ---------------------------------------------------------------- yol yardımcıları
def is_image_file(path: str) -> bool:
    return path.lower().endswith(IMAGE_EXTS)


def label_path_for(image_path: str, label_dir: str) -> str:
    """resim.jpg -> <label_dir>/resim.txt"""
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(label_dir, base + ".txt")


def find_existing_label(image_path: str, label_dir: str) -> Optional[str]:
    """
    Önce ayrılmış etiket klasörüne, bulunamazsa resmin yanına bakar.
    (labelImg ile üretilmiş eski veri setleriyle uyum için.)
    """
    primary = label_path_for(image_path, label_dir)
    if os.path.isfile(primary):
        return primary
    sibling = os.path.splitext(image_path)[0] + ".txt"
    if os.path.isfile(sibling):
        return sibling
    return None


def has_labels(image_path: str, label_dir: str) -> bool:
    p = find_existing_label(image_path, label_dir)
    if not p:
        return False
    try:
        return os.path.getsize(p) > 0
    except OSError:
        return False


def list_images(folder: str) -> List[str]:
    try:
        entries = os.listdir(folder)
    except OSError:
        return []
    files = [os.path.join(folder, f) for f in entries if is_image_file(f)]
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda p: os.path.basename(p).lower())
    return files


# ---------------------------------------------------------------- etiket G/Ç
def save_labels(txt_path: str, boxes: Sequence[PixelBox],
                img_w: int, img_h: int, decimals: int = 6) -> None:
    """
    Kutuları YOLO formatında yazar. Kutu yoksa boş dosya yazılır; bu, resmin
    'arka plan örneği olarak gözden geçirildiği' anlamına gelir (YOLO standardı).
    """
    os.makedirs(os.path.dirname(os.path.abspath(txt_path)) or ".", exist_ok=True)
    lines = []
    for class_id, x1, y1, x2, y2 in boxes:
        xc, yc, w, h = to_yolo(x1, y1, x2, y2, img_w, img_h)
        if w <= 0 or h <= 0:
            continue  # dejenere kutuyu yazma
        lines.append(f"{int(class_id)} {xc:.{decimals}f} {yc:.{decimals}f} "
                     f"{w:.{decimals}f} {h:.{decimals}f}")
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def load_raw(txt_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO dosyasını normalize haliyle okur: (class_id, xc, yc, w, h).
    Bozuk satırlar sessizce atlanır, dosyanın tamamı çöpe gitmez.
    """
    rows: List[Tuple[int, float, float, float, float]] = []
    if not txt_path or not os.path.isfile(txt_path):
        return rows
    try:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError:
        return rows

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", ".").split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            xc, yc, w, h = (float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        if cid < 0 or w <= 0 or h <= 0:
            continue
        rows.append((cid, xc, yc, w, h))
    return rows


def load_labels(txt_path: str, img_w: int, img_h: int) -> List[PixelBox]:
    """YOLO dosyasını okuyup piksel koordinatlarına çevirir."""
    result: List[PixelBox] = []
    for cid, xc, yc, w, h in load_raw(txt_path):
        x1, y1, x2, y2 = from_yolo(xc, yc, w, h, img_w, img_h)
        if x2 - x1 < 1e-6 or y2 - y1 < 1e-6:
            continue
        result.append((cid, x1, y1, x2, y2))
    return result


# ---------------------------------------------------------------- yedekleme
BACKUP_DIRNAME = ".backup"


def backup_label(txt_path: str, keep: int = 5) -> Optional[str]:
    """
    Üzerine yazmadan önce mevcut etiket dosyasının kopyasını
    <klasör>/.backup/ altına zaman damgasıyla alır. Dosya başına son `keep`
    sürüm tutulur. Kayıt akışını bloklamaması için hatalar yutulur.
    """
    import shutil
    import time
    if not txt_path or not os.path.isfile(txt_path):
        return None
    folder = os.path.dirname(os.path.abspath(txt_path))
    stem = os.path.splitext(os.path.basename(txt_path))[0]
    backup_dir = os.path.join(folder, BACKUP_DIRNAME)
    try:
        os.makedirs(backup_dir, exist_ok=True)
        target = os.path.join(backup_dir, f"{stem}.{time.strftime('%Y%m%d-%H%M%S')}.txt")
        shutil.copy2(txt_path, target)
    except OSError:
        return None

    try:
        existing = sorted(f for f in os.listdir(backup_dir)
                          if f.startswith(stem + ".") and f.endswith(".txt"))
        for old in existing[:-keep]:
            try:
                os.remove(os.path.join(backup_dir, old))
            except OSError:
                pass
    except OSError:
        pass
    return target


def list_backups(txt_path: str) -> List[str]:
    """Bir etiket dosyasının yedeklerini yeniden eskiye doğru listeler."""
    if not txt_path:
        return []
    folder = os.path.dirname(os.path.abspath(txt_path))
    stem = os.path.splitext(os.path.basename(txt_path))[0]
    backup_dir = os.path.join(folder, BACKUP_DIRNAME)
    if not os.path.isdir(backup_dir):
        return []
    try:
        names = [f for f in os.listdir(backup_dir)
                 if f.startswith(stem + ".") and f.endswith(".txt")]
    except OSError:
        return []
    names.sort(reverse=True)
    return [os.path.join(backup_dir, n) for n in names]


# ---------------------------------------------------------------- classes.txt
def save_classes(path: str, names: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for n in names:
            f.write(f"{n}\n")


def load_classes(path: str) -> List[str]:
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return []


def reassign_label_files(label_dir: str, old_id: int, new_id: int) -> int:
    """
    Diskteki tüm etiket dosyalarında bir sınıfın kutularını başka sınıfa taşır.
    (Sınıf silinirken 'kutuları başka sınıfa taşı' seçeneği için.)
    Dönüş: değiştirilen dosya sayısı.
    """
    if not os.path.isdir(label_dir) or old_id == new_id:
        return 0
    changed = 0
    for filename in os.listdir(label_dir):
        if not filename.lower().endswith(".txt") or filename.lower() == "classes.txt":
            continue
        path = os.path.join(label_dir, filename)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError:
            continue
        out, touched = [], False
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    if int(float(parts[0])) == old_id:
                        parts[0] = str(new_id)
                        touched = True
                        out.append(" ".join(parts))
                        continue
                except ValueError:
                    pass
            out.append(line)
        if touched:
            try:
                with open(path, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write("\n".join(out) + ("\n" if out else ""))
                changed += 1
            except OSError:
                pass
    return changed


def remap_label_files(label_dir: str, mapping) -> int:
    """
    Bir sınıf silindiğinde diskteki tüm .txt dosyalarındaki id'leri düzeltir.
    mapping: {eski_id: yeni_id veya None}. None olan satırlar silinir.
    Dönüş: güncellenen dosya sayısı.
    """
    if not os.path.isdir(label_dir):
        return 0
    changed = 0
    for fn in os.listdir(label_dir):
        if not fn.lower().endswith(".txt") or fn.lower() == "classes.txt":
            continue
        p = os.path.join(label_dir, fn)
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        out, touched = [], False
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                out.append(line)
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                out.append(line)
                continue
            new_id = mapping.get(cid, cid)
            if new_id is None:
                touched = True
                continue
            if new_id != cid:
                touched = True
                parts[0] = str(new_id)
            out.append(" ".join(parts))
        if touched:
            try:
                with open(p, "w", encoding="utf-8", newline="\n") as f:
                    f.write("\n".join(out) + ("\n" if out else ""))
                changed += 1
            except OSError:
                pass
    return changed
