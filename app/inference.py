"""
ONNX üzerinden YOLOv8 çıkarımı (model destekli ön-etiketleme).

Neden ONNX? `ultralytics` + `torch` ikilisi paketlenmiş uygulamaya ~2 GB ekler.
`onnxruntime` ~50 MB'dır ve tek başına yeterlidir. Kullanıcı modelini
`yolo export model=best.pt format=onnx` ile bir kez dışa aktarır.

Bu modül Qt bilmez; girdi (H, W, 3) uint8 RGB numpy dizisidir.
"""

import ast
import os
from typing import List, Optional, Sequence, Tuple

# İçe aktarma hataları YUTULMAZ, kaydedilir: onnxruntime Windows'ta eksik
# Visual C++ çalışma zamanı yüzünden ImportError dışında hatalar da verebilir.
# Gerçek hata metni olmadan kullanıcı "kurulu ama çalışmıyor" ile baş başa kalır.
IMPORT_ERRORS = {}

try:
    import numpy as np
    HAS_NUMPY = True
except Exception as _exc:                              # pragma: no cover
    np = None
    HAS_NUMPY = False
    IMPORT_ERRORS["numpy"] = f"{type(_exc).__name__}: {_exc}"

try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception as _exc:                              # pragma: no cover
    ort = None
    HAS_ORT = False
    IMPORT_ERRORS["onnxruntime"] = f"{type(_exc).__name__}: {_exc}"

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except Exception:                                      # pragma: no cover
    _cv2 = None
    _HAS_CV2 = False

# (class_id, x1, y1, x2, y2, score) — orijinal resim pikselinde
Detection = Tuple[int, float, float, float, float, float]


def available() -> bool:
    return HAS_ORT and HAS_NUMPY


def unavailable_reason() -> str:
    if not HAS_NUMPY:
        return IMPORT_ERRORS.get("numpy", "numpy kurulu değil (pip install numpy)")
    if not HAS_ORT:
        return IMPORT_ERRORS.get("onnxruntime",
                                 "onnxruntime kurulu değil (pip install onnxruntime)")
    return ""


def missing_packages() -> List[str]:
    missing = []
    if not HAS_NUMPY:
        missing.append("numpy")
    if not HAS_ORT:
        missing.append("onnxruntime")
    return missing


# ------------------------------------------------------------------ ön işleme
def letterbox(image, size: int = 640, color: int = 114):
    """
    Resmi en-boy oranını bozmadan `size`x`size` tuvale oturtur.

    Dönüş: (tuval, oran, pad_x, pad_y). Kutuları geri çevirmek için bu üç
    değer gerekir — burada yapılan hata, kutuların resmin yanlış yerine
    düşmesi olarak görünür, bu yüzden ayrı bir fonksiyon ve ayrı testi var.
    """
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Geçersiz resim boyutu")
    ratio = min(size / width, size / height)
    new_w = max(1, int(round(width * ratio)))
    new_h = max(1, int(round(height * ratio)))
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0

    if _HAS_CV2:
        resized = _cv2.resize(image, (new_w, new_h), interpolation=_cv2.INTER_LINEAR)
    else:
        ys = (np.arange(new_h) / ratio).astype(np.int32).clip(0, height - 1)
        xs = (np.arange(new_w) / ratio).astype(np.int32).clip(0, width - 1)
        resized = image[ys][:, xs]

    canvas = np.full((size, size, image.shape[2]), color, dtype=np.uint8)
    top, left = int(round(pad_y)), int(round(pad_x))
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, ratio, float(left), float(top)


# ------------------------------------------------------------------ NMS
def nms(boxes, scores, iou_threshold: float = 0.45) -> List[int]:
    """Örtüşen kutuları eleyip tutulacakların indekslerini döndürür."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def class_aware_nms(boxes, scores, class_ids, iou_threshold: float = 0.45) -> List[int]:
    """Sınıflar birbirini bastırmasın diye kutuları sınıfa göre kaydırıp NMS uygular."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    class_ids = np.asarray(class_ids)
    span = float(np.max(boxes)) + 1.0 if boxes.size else 1.0
    offset = class_ids.astype(np.float32)[:, None] * span
    return nms(boxes + offset, scores, iou_threshold)


# ------------------------------------------------------------------ çözümleme
def decode_output(output, conf_threshold: float = 0.25,
                  num_classes: Optional[int] = None):
    """
    Ham model çıktısını (kutular, skorlar, sınıflar) üçlüsüne çevirir.

    YOLOv8 çıktısı (1, 4+nc, N) biçimindedir; bazı dışa aktarımlar (1, N, 4+nc)
    verir. Objectness sütunu taşıyan YOLOv5 biçimi (5+nc) de desteklenir.
    Kutular letterbox uzayında xyxy olarak döner.
    """
    array = np.asarray(output)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Beklenmeyen çıktı boyutu: {np.asarray(output).shape}")

    rows, cols = array.shape
    # Hangi eksen öznitelik ekseni? Sınıf sayısı biliniyorsa kesin karar verilir,
    # bilinmiyorsa "öznitelik sayısı çapa sayısından küçüktür" varsayımı kullanılır.
    transpose = None
    if num_classes is not None:
        candidates = (num_classes + 4, num_classes + 5)
        if rows in candidates and cols not in candidates:
            transpose = True
        elif cols in candidates and rows not in candidates:
            transpose = False
    if transpose is None:
        transpose = rows < cols
    if transpose:
        array = array.T
    predictions = array
    attributes = predictions.shape[1]

    v5_layout = (num_classes is not None and attributes == num_classes + 5)
    if v5_layout:
        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:] * objectness[:, None]
    else:
        class_scores = predictions[:, 4:]

    if class_scores.shape[1] == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)
    mask = confidences >= conf_threshold
    if not mask.any():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

    centers = predictions[mask, :4]
    confidences = confidences[mask]
    class_ids = class_ids[mask].astype(np.int32)

    cx, cy, w, h = centers[:, 0], centers[:, 1], centers[:, 2], centers[:, 3]
    boxes = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
    return boxes.astype(np.float32), confidences.astype(np.float32), class_ids


def scale_boxes(boxes, ratio: float, pad_x: float, pad_y: float,
                orig_w: int, orig_h: int):
    """Letterbox uzayındaki kutuları orijinal resim pikseline geri çevirir."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32)
    result = np.asarray(boxes, dtype=np.float32).copy()
    result[:, [0, 2]] -= pad_x
    result[:, [1, 3]] -= pad_y
    result /= max(ratio, 1e-9)
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, orig_w)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, orig_h)
    return result


def postprocess(output, ratio: float, pad_x: float, pad_y: float,
                orig_w: int, orig_h: int, conf_threshold: float = 0.25,
                iou_threshold: float = 0.45,
                num_classes: Optional[int] = None,
                min_size: float = 2.0) -> List[Detection]:
    """Ham çıktıdan, resim pikselinde temizlenmiş tespit listesi üretir."""
    boxes, scores, class_ids = decode_output(output, conf_threshold, num_classes)
    if len(boxes) == 0:
        return []
    boxes = scale_boxes(boxes, ratio, pad_x, pad_y, orig_w, orig_h)
    keep = class_aware_nms(boxes, scores, class_ids, iou_threshold)

    detections: List[Detection] = []
    for index in keep:
        x1, y1, x2, y2 = boxes[index]
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue
        detections.append((int(class_ids[index]), float(x1), float(y1),
                           float(x2), float(y2), float(scores[index])))
    detections.sort(key=lambda d: d[5], reverse=True)
    return detections


# ------------------------------------------------------------------ model
class OnnxDetector:
    """Yüklenmiş bir ONNX YOLOv8 modelini saran ince katman."""

    def __init__(self, model_path: str, size: Optional[int] = None):
        if not available():
            raise RuntimeError(unavailable_reason())
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        self.path = model_path
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name

        detected = None
        shape = list(model_input.shape or [])
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            detected = int(shape[2])
        self.size = int(size or detected or 640)

        self.class_names: List[str] = self._read_class_names()

    def _read_class_names(self) -> List[str]:
        """Ultralytics dışa aktarımı sınıf adlarını model üstverisine yazar."""
        try:
            metadata = self.session.get_modelmeta().custom_metadata_map or {}
        except Exception:                              # pragma: no cover
            return []
        raw = metadata.get("names")
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, dict):
            try:
                return [str(parsed[key]) for key in sorted(parsed, key=int)]
            except (ValueError, TypeError):
                return [str(value) for value in parsed.values()]
        if isinstance(parsed, (list, tuple)):
            return [str(value) for value in parsed]
        return []

    def predict(self, image, conf_threshold: float = 0.25,
                iou_threshold: float = 0.45) -> List[Detection]:
        """(H, W, 3) uint8 RGB dizi -> tespit listesi."""
        height, width = image.shape[:2]
        canvas, ratio, pad_x, pad_y = letterbox(image, self.size)
        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
        outputs = self.session.run(None, {self.input_name: blob})
        return postprocess(outputs[0], ratio, pad_x, pad_y, width, height,
                           conf_threshold, iou_threshold,
                           num_classes=len(self.class_names) or None)


def map_model_classes(model_names: Sequence[str],
                      project_names: Sequence[str]) -> dict:
    """
    Model sınıf kimliklerini projedeki sınıf kimliklerine eşler.

    Adlar birebir (büyük/küçük harf duyarsız) tutuyorsa eşleşir; tutmayanlar
    haritada yer almaz ve çağıran taraf kullanıcıya sorar.
    """
    lookup = {name.strip().lower(): index
              for index, name in enumerate(project_names)}
    mapping = {}
    for model_index, name in enumerate(model_names):
        target = lookup.get(str(name).strip().lower())
        if target is not None:
            mapping[model_index] = target
    return mapping
