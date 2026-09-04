"""
Proje dosyası (.labelproj): klasörler, sınıflar, model yolu ve tercihler.

Amaç: çift tıklayınca kaldığınız yerden devam etmek. Qt bilmez, saf JSON.
"""

import json
import os
from typing import Dict, List, Sequence

VERSION = 1
EXTENSION = ".labelproj"


def build(image_dir: str, label_dir: str,
          classes: Sequence, model_path: str = "",
          settings: Dict = None) -> dict:
    """`classes`, .name ve .color alanları olan nesnelerin dizisidir."""
    return {
        "version": VERSION,
        "image_dir": image_dir or "",
        "label_dir": label_dir or "",
        "model_path": model_path or "",
        "classes": [{"name": c.name, "color": c.color} for c in classes],
        "settings": dict(settings or {}),
    }


def save(path: str, document: dict) -> str:
    if not path.lower().endswith(EXTENSION):
        path += EXTENSION
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    return path


def load(path: str) -> dict:
    """
    Proje dosyasını okur ve eksik alanları tamamlar.

    Bozuk veya eski bir dosya yüzünden program açılmamazlık etmemeli; bu yüzden
    her alan savunmacı biçimde okunur.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Proje dosyası bozuk.")

    classes: List[dict] = []
    for entry in raw.get("classes", []) or []:
        if isinstance(entry, dict) and entry.get("name"):
            classes.append({"name": str(entry["name"]),
                            "color": str(entry.get("color", "") or "")})
        elif isinstance(entry, str) and entry.strip():
            classes.append({"name": entry.strip(), "color": ""})

    return {
        "version": int(raw.get("version", VERSION) or VERSION),
        "image_dir": str(raw.get("image_dir", "") or ""),
        "label_dir": str(raw.get("label_dir", "") or ""),
        "model_path": str(raw.get("model_path", "") or ""),
        "classes": classes,
        "settings": raw.get("settings") if isinstance(raw.get("settings"), dict) else {},
    }


def is_usable(document: dict) -> bool:
    """Proje dosyasındaki resim klasörü hâlâ duruyor mu?"""
    folder = document.get("image_dir") or ""
    return bool(folder) and os.path.isdir(folder)
