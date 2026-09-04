"""
Veri modelleri.

Bu modül bilerek Qt'den bağımsız tutulmuştur; böylece iş mantığı (sınıf yönetimi,
kutu geometrisi) arayüz olmadan da test edilebilir.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# Sınıflara otomatik atanacak, birbirinden kolay ayırt edilebilen renk paleti.
PALETTE: List[str] = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def color_for_index(index: int) -> str:
    """Sınıf sırasına göre deterministik bir renk döndürür."""
    return PALETTE[index % len(PALETTE)]


def is_light_color(hex_color: str) -> bool:
    """Etiket yazısının siyah mı beyaz mı olacağına karar vermek için."""
    c = hex_color.lstrip("#")
    if len(c) != 6:
        return True
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    # ITU-R BT.601 luma
    return (0.299 * r + 0.587 * g + 0.114 * b) > 150


@dataclass
class LabelClass:
    """Tek bir etiket sınıfı."""
    name: str
    color: str


@dataclass
class BBox:
    """Piksel koordinatlarında bir sınırlayıcı kutu (sol-üst / sağ-alt)."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0

    def normalized(self) -> "BBox":
        return BBox(
            min(self.x1, self.x2), min(self.y1, self.y2),
            max(self.x1, self.x2), max(self.y1, self.y2),
            self.class_id,
        )

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


class ClassManager:
    """
    Etiket sınıflarının tek yetkili sahibi.

    Sınıf sırası = YOLO class_id. Bu yüzden silme işlemi sonrası kimliklerin
    yeniden eşlenmesi (remap) gerekir; `remove()` bunun için bir harita döndürür.
    """

    def __init__(self, names: Optional[List[str]] = None):
        self._classes: List[LabelClass] = []
        for n in (names or []):
            self.add(n, silent_duplicates=True)

    # ---- okuma -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._classes)

    def __iter__(self):
        return iter(self._classes)

    def __getitem__(self, i: int) -> LabelClass:
        return self._classes[i]

    def names(self) -> List[str]:
        return [c.name for c in self._classes]

    def index_of(self, name: str) -> int:
        for i, c in enumerate(self._classes):
            if c.name == name:
                return i
        return -1

    def name_of(self, class_id: int) -> str:
        if 0 <= class_id < len(self._classes):
            return self._classes[class_id].name
        return f"?{class_id}"

    def color_of(self, class_id: int) -> str:
        if 0 <= class_id < len(self._classes):
            return self._classes[class_id].color
        return "#9e9e9e"

    # ---- yazma -----------------------------------------------------------
    def add(self, name: str, color: Optional[str] = None,
            silent_duplicates: bool = False) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("Sınıf adı boş olamaz.")
        if " " in name:
            name = name.replace(" ", "_")  # YOLO classes.txt satır başına tek ad
        existing = self.index_of(name)
        if existing >= 0:
            if silent_duplicates:
                return existing
            raise ValueError(f"'{name}' sınıfı zaten var.")
        self._classes.append(LabelClass(name, color or color_for_index(len(self._classes))))
        return len(self._classes) - 1

    def rename(self, class_id: int, new_name: str) -> None:
        new_name = (new_name or "").strip().replace(" ", "_")
        if not new_name:
            raise ValueError("Sınıf adı boş olamaz.")
        other = self.index_of(new_name)
        if other >= 0 and other != class_id:
            raise ValueError(f"'{new_name}' sınıfı zaten var.")
        self._classes[class_id].name = new_name

    def set_color(self, class_id: int, color: str) -> None:
        self._classes[class_id].color = color

    def ensure_capacity(self, class_id: int) -> None:
        """Etiket dosyasında tanımsız bir id görülürse yer tutucu sınıf üretir."""
        while len(self._classes) <= class_id:
            self.add(f"class_{len(self._classes)}", silent_duplicates=True)

    def remove(self, class_id: int) -> Dict[int, Optional[int]]:
        """
        Sınıfı siler ve eski id -> yeni id haritasını döndürür.
        Silinen sınıf için değer None'dır (o kutular atılmalıdır).
        """
        if not (0 <= class_id < len(self._classes)):
            raise IndexError("Geçersiz sınıf.")
        del self._classes[class_id]
        mapping: Dict[int, Optional[int]] = {}
        for old in range(len(self._classes) + 1):
            if old == class_id:
                mapping[old] = None
            elif old < class_id:
                mapping[old] = old
            else:
                mapping[old] = old - 1
        return mapping

    def clear(self) -> None:
        self._classes.clear()
