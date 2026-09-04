"""
Görüntü iyileştirme ve kenar tabanlı yardımcılar.

Tasarım notu: bu modül Qt bilmez, yalnızca (H, W, 3) uint8 numpy dizileriyle
çalışır. Böylece hem test edilebilir hem de arayüzden bağımsızdır.

Bağımlılıklar isteğe bağlıdır:
  * numpy yoksa tüm modül devre dışı kalır (HAS_NUMPY False)
  * OpenCV yoksa CLAHE yerine küresel histogram eşitleme kullanılır,
    kenar bulma ve manyetik kutu saf numpy ile çalışmayı sürdürür.
"""

from typing import Optional, Tuple

# İçe aktarma hatası yutulmaz, kaydedilir (bkz. inference.py'deki aynı gerekçe)
IMPORT_ERRORS = {}

try:
    import numpy as np
    HAS_NUMPY = True
except Exception as _exc:                              # pragma: no cover
    np = None
    HAS_NUMPY = False
    IMPORT_ERRORS["numpy"] = f"{type(_exc).__name__}: {_exc}"

try:
    import cv2
    HAS_CV2 = True
except Exception as _exc:                              # pragma: no cover
    cv2 = None
    HAS_CV2 = False
    IMPORT_ERRORS["opencv-python-headless"] = f"{type(_exc).__name__}: {_exc}"


def available() -> bool:
    return HAS_NUMPY


def backend_name() -> str:
    if not HAS_NUMPY:
        return "yok"
    return "OpenCV + numpy" if HAS_CV2 else "numpy"


# ------------------------------------------------------------------ temel
def build_lut(brightness: int = 0, contrast: int = 0, gamma: float = 1.0):
    """
    256 girişli arama tablosu üretir.

    brightness / contrast: -100 .. 100 (0 = değişiklik yok)
    gamma: 0.2 .. 3.0 (1.0 = değişiklik yok). Gama > 1 karanlık bölgeleri açar.
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy gerekli")
    x = np.arange(256, dtype=np.float32)

    gamma = max(0.05, float(gamma))
    if abs(gamma - 1.0) > 1e-6:
        x = 255.0 * np.power(x / 255.0, 1.0 / gamma)

    contrast = max(-100, min(100, int(contrast)))
    if contrast != 0:
        # Klasik kontrast düzeltme katsayısı
        factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
        x = factor * (x - 128.0) + 128.0

    brightness = max(-100, min(100, int(brightness)))
    if brightness != 0:
        x = x + brightness * 1.28

    return np.clip(x, 0, 255).astype(np.uint8)


def apply_lut(array, lut):
    """LUT'u RGB dizisine uygular."""
    return lut[array]


def to_gray(array):
    """(H, W, 3) -> (H, W) gri tonlama (ITU-R BT.601)."""
    if array.ndim == 2:
        return array
    return (0.299 * array[..., 0] + 0.587 * array[..., 1]
            + 0.114 * array[..., 2]).astype(np.uint8)


def _equalize_gray(gray):
    """Küresel histogram eşitleme (CLAHE yoksa yedek)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    nonzero = cdf > 0
    if not nonzero.any():
        return gray
    cdf_min = cdf[nonzero][0]
    denom = max(1.0, gray.size - cdf_min)
    lut = np.clip(np.round((cdf - cdf_min) / denom * 255.0), 0, 255).astype(np.uint8)
    return lut[gray]


def clahe(array, clip_limit: float = 2.0, tiles: int = 8):
    """
    Yerel kontrast eşitleme. Karanlık kabin içi fotoğraflarda nesne sınırlarını
    ortaya çıkarır. OpenCV varsa gerçek CLAHE, yoksa küresel eşitleme uygulanır.
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy gerekli")
    if array.ndim == 2:
        return _clahe_gray(array, clip_limit, tiles)

    if HAS_CV2:
        lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
        operator = cv2.createCLAHE(clipLimit=float(clip_limit),
                                   tileGridSize=(int(tiles), int(tiles)))
        lab[..., 0] = operator.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Yedek: parlaklık kanalını eşitle, rengi oranla koru
    gray = to_gray(array)
    equalized = _equalize_gray(gray)
    scale = np.where(gray == 0, 1.0, equalized.astype(np.float32)
                     / np.maximum(gray.astype(np.float32), 1.0))
    result = array.astype(np.float32) * scale[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def _clahe_gray(gray, clip_limit: float, tiles: int):
    if HAS_CV2:
        operator = cv2.createCLAHE(clipLimit=float(clip_limit),
                                   tileGridSize=(int(tiles), int(tiles)))
        return operator.apply(gray)
    return _equalize_gray(gray)


def enhance(array, brightness: int = 0, contrast: int = 0, gamma: float = 1.0,
            use_clahe: bool = False, clip_limit: float = 2.0):
    """Tüm iyileştirme zincirini uygular. Girdi dizisi değiştirilmez."""
    if not HAS_NUMPY:
        raise RuntimeError("numpy gerekli")
    result = array
    if use_clahe:
        result = clahe(result, clip_limit)
    if brightness or contrast or abs(gamma - 1.0) > 1e-6:
        result = apply_lut(result, build_lut(brightness, contrast, gamma))
    return result


def is_identity(brightness: int, contrast: int, gamma: float, use_clahe: bool) -> bool:
    """Hiçbir şey değişmiyorsa pahalı dönüşümü hiç yapma."""
    return (not use_clahe and brightness == 0 and contrast == 0
            and abs(gamma - 1.0) < 1e-6)


# ------------------------------------------------------------------ kenarlar
def edge_map(array, low: int = 80, high: int = 160):
    """
    Kenar haritası (0-255 tek kanal). OpenCV varsa Canny, yoksa gradyan büyüklüğü.
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy gerekli")
    gray = to_gray(array)
    if HAS_CV2:
        return cv2.Canny(gray, int(low), int(high))
    gy, gx = np.gradient(gray.astype(np.float32))
    magnitude = np.hypot(gx, gy)
    peak = magnitude.max()
    if peak <= 0:
        return np.zeros_like(gray)
    normalized = (magnitude / peak * 255.0)
    threshold = float(low)
    normalized[normalized < threshold] = 0.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ manyetik kutu
def snap_box(array, x1: float, y1: float, x2: float, y2: float,
             search: float = 0.15, min_size: int = 4
             ) -> Optional[Tuple[float, float, float, float]]:
    """
    Kaba çizilmiş kutunun kenarlarını, çevresindeki en güçlü gradyan çizgisine
    oturtur. Kutu kenarı ±(search × kenar uzunluğu) bandında aranır.

    Dönüş: yeni (x1, y1, x2, y2) veya iyileştirme yapılamadıysa None.
    """
    if not HAS_NUMPY:
        return None
    height, width = array.shape[:2]

    left, right = sorted((int(round(x1)), int(round(x2))))
    top, bottom = sorted((int(round(y1)), int(round(y2))))
    left = max(0, min(width - 1, left))
    right = max(0, min(width, right))
    top = max(0, min(height - 1, top))
    bottom = max(0, min(height, bottom))

    box_w, box_h = right - left, bottom - top
    if box_w < min_size * 2 or box_h < min_size * 2:
        return None

    margin_x = max(2, int(box_w * search))
    margin_y = max(2, int(box_h * search))

    region_x1 = max(0, left - margin_x)
    region_x2 = min(width, right + margin_x)
    region_y1 = max(0, top - margin_y)
    region_y2 = min(height, bottom + margin_y)
    region = array[region_y1:region_y2, region_x1:region_x2]
    if region.size == 0:
        return None

    gray = to_gray(region).astype(np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return None
    grad_y, grad_x = np.gradient(gray)
    column_energy = np.abs(grad_x).sum(axis=0)
    row_energy = np.abs(grad_y).sum(axis=1)

    new_left = region_x1 + _strongest(column_energy, left - region_x1, margin_x)
    new_right = region_x1 + _strongest(column_energy, right - region_x1, margin_x)
    new_top = region_y1 + _strongest(row_energy, top - region_y1, margin_y)
    new_bottom = region_y1 + _strongest(row_energy, bottom - region_y1, margin_y)

    if new_right - new_left < min_size or new_bottom - new_top < min_size:
        return None
    return (float(new_left), float(new_top), float(new_right), float(new_bottom))


def _strongest(profile, center: int, radius: int) -> int:
    """Verilen indeksin ±radius komşuluğundaki en güçlü kenarın indeksi."""
    length = len(profile)
    if length == 0:
        return max(0, center)
    center = max(0, min(length - 1, center))
    low = max(0, center - radius)
    high = min(length, center + radius + 1)
    if high <= low:
        return center
    window = profile[low:high]
    if float(window.max()) <= 0.0:
        return center
    return low + int(np.argmax(window))
