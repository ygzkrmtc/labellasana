"""
Format dönüştürme: YOLO <-> COCO JSON, YOLO <-> Pascal VOC XML.

Başka bir ekipten gelen veriyi kabul edebilmek ve çıktıyı başka bir eğitim
hattına verebilmek, profesyonel araç ile ev yapımı betik arasındaki farktır.

Qt bilmez. Resim boyutlarına ihtiyaç duyan yerlerde `size_of(path) -> (w, h)`
geri çağrımı kullanılır; böylece modül görüntü kütüphanesinden de bağımsızdır.
"""

import json
import os
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Sequence, Tuple

from . import yolo_io

SizeProvider = Callable[[str], Tuple[int, int]]


# ==================================================================== COCO
def export_coco(image_paths: Sequence[str], label_dir: str,
                class_names: Sequence[str], size_of: SizeProvider,
                out_path: str) -> dict:
    """
    YOLO etiketlerini tek bir COCO JSON dosyasına yazar.

    COCO kategori kimlikleri 1'den başlar (yaygın kural); YOLO kimliği + 1.
    """
    images = []
    annotations = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        width, height = size_of(image_path)
        if width <= 0 or height <= 0:
            continue
        images.append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": int(width),
            "height": int(height),
        })
        txt = yolo_io.find_existing_label(image_path, label_dir)
        for class_id, xc, yc, w, h in (yolo_io.load_raw(txt) if txt else []):
            box_w = w * width
            box_h = h * height
            x = (xc * width) - box_w / 2.0
            y = (yc * height) - box_h / 2.0
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id) + 1,
                "bbox": [round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2)],
                "area": round(box_w * box_h, 2),
                "iscrowd": 0,
            })
            annotation_id += 1

    document = {
        "info": {"description": "Etiketleme Aracı tarafından üretildi"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": name, "supercategory": "none"}
                       for index, name in enumerate(class_names)],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    return {"images": len(images), "annotations": len(annotations),
            "categories": len(class_names), "path": out_path}


def import_coco(json_path: str, out_label_dir: str) -> dict:
    """
    COCO JSON'u YOLO .txt dosyalarına çevirir.

    Dönüş: {"class_names": [...], "images": n, "annotations": n, "skipped": n}
    Sınıf sırası COCO kategori kimliklerine göre belirlenir, böylece kimlikler
    kararlıdır ve classes.txt ile hizalı kalır.
    """
    with open(json_path, "r", encoding="utf-8") as handle:
        document = json.load(handle)

    categories = sorted(document.get("categories", []),
                        key=lambda c: c.get("id", 0))
    class_names = [str(c.get("name", f"class_{i}"))
                   for i, c in enumerate(categories)]
    category_to_index = {c.get("id"): index for index, c in enumerate(categories)}

    images = {img["id"]: img for img in document.get("images", [])}
    per_image: Dict[int, List[str]] = {}
    skipped = 0

    for annotation in document.get("annotations", []):
        image = images.get(annotation.get("image_id"))
        if image is None:
            skipped += 1
            continue
        width = float(image.get("width", 0) or 0)
        height = float(image.get("height", 0) or 0)
        bbox = annotation.get("bbox") or []
        if width <= 0 or height <= 0 or len(bbox) < 4:
            skipped += 1
            continue
        index = category_to_index.get(annotation.get("category_id"))
        if index is None:
            skipped += 1
            continue

        x, y, box_w, box_h = (float(v) for v in bbox[:4])
        if box_w <= 0 or box_h <= 0:
            skipped += 1
            continue
        xc, yc, nw, nh = yolo_io.to_yolo(x, y, x + box_w, y + box_h,
                                         int(width), int(height))
        per_image.setdefault(image["id"], []).append(
            f"{index} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

    os.makedirs(out_label_dir, exist_ok=True)
    for image_id, image in images.items():
        stem = os.path.splitext(os.path.basename(image.get("file_name", "")))[0]
        if not stem:
            continue
        lines = per_image.get(image_id, [])
        with open(os.path.join(out_label_dir, stem + ".txt"), "w",
                  encoding="utf-8", newline="\n") as handle:
            if lines:
                handle.write("\n".join(lines) + "\n")

    yolo_io.save_classes(os.path.join(out_label_dir, "classes.txt"), class_names)
    return {"class_names": class_names, "images": len(images),
            "annotations": sum(len(v) for v in per_image.values()),
            "skipped": skipped}


# ==================================================================== VOC
def export_voc(image_paths: Sequence[str], label_dir: str,
               class_names: Sequence[str], size_of: SizeProvider,
               out_dir: str) -> dict:
    """Her resim için bir Pascal VOC XML dosyası yazar (1 tabanlı koordinat)."""
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    boxes = 0

    for image_path in image_paths:
        width, height = size_of(image_path)
        if width <= 0 or height <= 0:
            continue
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = os.path.basename(
            os.path.dirname(image_path))
        ET.SubElement(root, "filename").text = os.path.basename(image_path)
        ET.SubElement(root, "path").text = image_path

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(int(width))
        ET.SubElement(size, "height").text = str(int(height))
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

        txt = yolo_io.find_existing_label(image_path, label_dir)
        for class_id, xc, yc, w, h in (yolo_io.load_raw(txt) if txt else []):
            x1, y1, x2, y2 = yolo_io.from_yolo(xc, yc, w, h, int(width), int(height))
            name = (class_names[class_id] if 0 <= class_id < len(class_names)
                    else f"class_{class_id}")
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            box = ET.SubElement(obj, "bndbox")
            ET.SubElement(box, "xmin").text = str(int(round(x1)) + 1)
            ET.SubElement(box, "ymin").text = str(int(round(y1)) + 1)
            ET.SubElement(box, "xmax").text = str(int(round(x2)) + 1)
            ET.SubElement(box, "ymax").text = str(int(round(y2)) + 1)
            boxes += 1

        stem = os.path.splitext(os.path.basename(image_path))[0]
        ET.ElementTree(root).write(os.path.join(out_dir, stem + ".xml"),
                                   encoding="utf-8", xml_declaration=True)
        written += 1

    return {"files": written, "boxes": boxes, "path": out_dir}


def import_voc(xml_dir: str, out_label_dir: str,
               class_names: Sequence[str] = ()) -> dict:
    """
    VOC XML klasörünü YOLO .txt'ye çevirir.

    `class_names` verilirse kimlikler o sıraya göre atanır; XML'de geçen yeni
    adlar listenin sonuna eklenir (mevcut kimlikler bozulmaz).
    """
    names = list(class_names)
    index_of = {name: i for i, name in enumerate(names)}
    os.makedirs(out_label_dir, exist_ok=True)

    files = boxes = skipped = 0
    for filename in sorted(os.listdir(xml_dir)):
        if not filename.lower().endswith(".xml"):
            continue
        try:
            root = ET.parse(os.path.join(xml_dir, filename)).getroot()
        except ET.ParseError:
            skipped += 1
            continue

        size = root.find("size")
        try:
            width = int(float(size.find("width").text))
            height = int(float(size.find("height").text))
        except (AttributeError, TypeError, ValueError):
            skipped += 1
            continue
        if width <= 0 or height <= 0:
            skipped += 1
            continue

        lines = []
        for obj in root.findall("object"):
            name_node = obj.find("name")
            box = obj.find("bndbox")
            if name_node is None or box is None:
                continue
            name = (name_node.text or "").strip()
            if not name:
                continue
            if name not in index_of:
                index_of[name] = len(names)
                names.append(name)
            try:
                x1 = float(box.find("xmin").text) - 1
                y1 = float(box.find("ymin").text) - 1
                x2 = float(box.find("xmax").text) - 1
                y2 = float(box.find("ymax").text) - 1
            except (AttributeError, TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            xc, yc, w, h = yolo_io.to_yolo(x1, y1, x2, y2, width, height)
            lines.append(f"{index_of[name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            boxes += 1

        stem = os.path.splitext(filename)[0]
        with open(os.path.join(out_label_dir, stem + ".txt"), "w",
                  encoding="utf-8", newline="\n") as handle:
            if lines:
                handle.write("\n".join(lines) + "\n")
        files += 1

    yolo_io.save_classes(os.path.join(out_label_dir, "classes.txt"), names)
    return {"class_names": names, "files": files, "boxes": boxes, "skipped": skipped}
