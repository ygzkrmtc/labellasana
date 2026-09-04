"""
Model tarafından üretilen etiketlerin "gözden geçirilecek" kuyruğu.

Tasarım ilkesi: onaylanmamış hiçbir model çıktısı, insan onayından geçmiş bir
etiketle aynı görünmemelidir. Toplu tahmin sonuçları diske yazılır ama ilgili
resimler bu kuyruğa alınır; kullanıcı resmi açıp onaylayana kadar listede
'?' ile işaretli kalır.

Kuyruk, etiket klasöründe `.review.json` dosyasında tutulur (Qt'siz, saf JSON).
"""

import json
import os
from typing import Iterable, Set

FILENAME = ".review.json"


def _path(label_dir: str) -> str:
    return os.path.join(label_dir, FILENAME)


def load_pending(label_dir: str) -> Set[str]:
    """Onay bekleyen resim dosya adları."""
    if not label_dir:
        return set()
    path = _path(label_dir)
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return set()
    pending = document.get("pending") if isinstance(document, dict) else None
    if not isinstance(pending, list):
        return set()
    return {str(name) for name in pending if str(name).strip()}


def save_pending(label_dir: str, names: Iterable[str]) -> bool:
    if not label_dir:
        return False
    try:
        os.makedirs(label_dir, exist_ok=True)
        with open(_path(label_dir), "w", encoding="utf-8") as handle:
            json.dump({"pending": sorted(set(names))}, handle,
                      ensure_ascii=False, indent=1)
        return True
    except OSError:
        return False


def mark(label_dir: str, image_name: str) -> Set[str]:
    pending = load_pending(label_dir)
    pending.add(os.path.basename(image_name))
    save_pending(label_dir, pending)
    return pending


def unmark(label_dir: str, image_name: str) -> Set[str]:
    pending = load_pending(label_dir)
    pending.discard(os.path.basename(image_name))
    save_pending(label_dir, pending)
    return pending


def clear(label_dir: str) -> None:
    save_pending(label_dir, [])
