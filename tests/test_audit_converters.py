"""Kalite denetimi ve format dönüştürme testleri (Qt gerektirmez)."""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import audit, converters, yolo_io  # noqa: E402


def write_label(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(" ".join(str(v) for v in row) + "\n")


class AuditBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.labels = os.path.join(self.dir, "labels")
        os.makedirs(self.labels)
        self.images = []
        for i in range(3):
            path = os.path.join(self.dir, f"img{i}.jpg")
            open(path, "wb").close()
            self.images.append(path)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def categories(self, findings):
        return {f.category for f in findings}


class TestAudit(AuditBase):
    def test_clean_dataset_has_no_errors(self):
        for i in range(3):
            write_label(os.path.join(self.labels, f"img{i}.txt"),
                        [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.3, 0.3, 0.15, 0.15)])
        findings = audit.audit(self.images, self.labels, ["a", "b"])
        errors = audit.filter_by(findings, severity=audit.ERROR)
        self.assertEqual(errors, [])

    def test_tiny_box_detected(self):
        write_label(os.path.join(self.labels, "img0.txt"),
                    [(0, 0.5, 0.5, 0.002, 0.002)])
        findings = audit.audit(self.images, self.labels, ["a"])
        self.assertIn("çok-küçük-kutu", self.categories(findings))

    def test_unknown_class_is_error(self):
        write_label(os.path.join(self.labels, "img0.txt"), [(7, 0.5, 0.5, 0.2, 0.2)])
        findings = audit.audit(self.images, self.labels, ["a"])
        errors = audit.filter_by(findings, severity=audit.ERROR)
        self.assertTrue(any(f.category == "tanımsız-sınıf" for f in errors))

    def test_duplicate_box_detected(self):
        write_label(os.path.join(self.labels, "img0.txt"),
                    [(0, 0.5, 0.5, 0.2, 0.2), (0, 0.502, 0.5, 0.2, 0.2)])
        findings = audit.audit(self.images, self.labels, ["a"])
        self.assertIn("yinelenen-kutu", self.categories(findings))

    def test_different_classes_are_not_duplicates(self):
        write_label(os.path.join(self.labels, "img0.txt"),
                    [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.5, 0.5, 0.2, 0.2)])
        findings = audit.audit(self.images, self.labels, ["a", "b"])
        self.assertNotIn("yinelenen-kutu", self.categories(findings))

    def test_extreme_aspect_ratio(self):
        write_label(os.path.join(self.labels, "img0.txt"),
                    [(0, 0.5, 0.5, 0.9, 0.02)])
        findings = audit.audit(self.images, self.labels, ["a"])
        self.assertIn("aşırı-oran", self.categories(findings))

    def test_border_box(self):
        write_label(os.path.join(self.labels, "img0.txt"),
                    [(0, 0.1, 0.5, 0.2, 0.2)])       # x1 = 0.0
        findings = audit.audit(self.images, self.labels, ["a"])
        self.assertIn("kenara-yapışık", self.categories(findings))

    def test_unlabeled_images_reported(self):
        findings = audit.audit(self.images, self.labels, ["a"])
        self.assertIn("etiketsiz-resim", self.categories(findings))

    def test_orphan_label_reported(self):
        write_label(os.path.join(self.labels, "hayalet.txt"), [(0, 0.5, 0.5, 0.2, 0.2)])
        findings = audit.audit(self.images, self.labels, ["a"])
        self.assertIn("yetim-etiket", self.categories(findings))

    def test_classes_txt_is_not_orphan(self):
        yolo_io.save_classes(os.path.join(self.labels, "classes.txt"), ["a"])
        findings = audit.audit(self.images, self.labels, ["a"])
        self.assertNotIn("yetim-etiket", self.categories(findings))

    def test_imbalance_reported(self):
        write_label(os.path.join(self.labels, "img0.txt"),
                    [(0, 0.5, 0.5, 0.2, 0.2)] * 30 + [(1, 0.3, 0.3, 0.1, 0.1)])
        findings = audit.audit(self.images, self.labels, ["a", "b"])
        self.assertIn("sınıf-dengesizliği", self.categories(findings))

    def test_unused_class_reported(self):
        write_label(os.path.join(self.labels, "img0.txt"), [(0, 0.5, 0.5, 0.2, 0.2)])
        findings = audit.audit(self.images, self.labels, ["a", "kullanilmayan"])
        self.assertIn("kullanılmayan-sınıf", self.categories(findings))

    def test_summary_counts(self):
        write_label(os.path.join(self.labels, "img0.txt"), [(9, 0.5, 0.5, 0.2, 0.2)])
        findings = audit.audit(self.images, self.labels, ["a"])
        summary = audit.summarize(findings)
        self.assertGreaterEqual(summary[audit.ERROR], 1)

    def test_sorted_errors_first(self):
        write_label(os.path.join(self.labels, "img0.txt"),
                    [(9, 0.5, 0.5, 0.2, 0.2), (0, 0.5, 0.5, 0.001, 0.001)])
        findings = audit.audit(self.images, self.labels, ["a"])
        self.assertEqual(findings[0].severity, audit.ERROR)

    def test_iou_math(self):
        same = audit.iou((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2))
        self.assertAlmostEqual(same, 1.0, places=6)
        apart = audit.iou((0.1, 0.1, 0.1, 0.1), (0.9, 0.9, 0.1, 0.1))
        self.assertEqual(apart, 0.0)


class TestCoco(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.labels = os.path.join(self.dir, "labels")
        os.makedirs(self.labels)
        self.images = []
        for i in range(2):
            path = os.path.join(self.dir, f"img{i}.jpg")
            open(path, "wb").close()
            self.images.append(path)
        write_label(os.path.join(self.labels, "img0.txt"),
                    [(0, 0.5, 0.5, 0.25, 0.5), (1, 0.25, 0.25, 0.1, 0.1)])
        write_label(os.path.join(self.labels, "img1.txt"), [])
        self.size_of = lambda p: (400, 200)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_export_structure(self):
        out = os.path.join(self.dir, "coco.json")
        summary = converters.export_coco(self.images, self.labels,
                                         ["aku", "msda"], self.size_of, out)
        self.assertEqual(summary["images"], 2)
        self.assertEqual(summary["annotations"], 2)
        with open(out, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(len(document["categories"]), 2)
        self.assertEqual(document["categories"][0]["id"], 1)     # 1 tabanlı
        first = document["annotations"][0]
        # 0.5,0.5,0.25,0.5 -> 400x200 resimde x=150,y=50,w=100,h=100
        self.assertAlmostEqual(first["bbox"][0], 150.0, places=1)
        self.assertAlmostEqual(first["bbox"][1], 50.0, places=1)
        self.assertAlmostEqual(first["bbox"][2], 100.0, places=1)
        self.assertAlmostEqual(first["bbox"][3], 100.0, places=1)

    def test_round_trip(self):
        out = os.path.join(self.dir, "coco.json")
        converters.export_coco(self.images, self.labels, ["aku", "msda"],
                               self.size_of, out)
        back_dir = os.path.join(self.dir, "back")
        summary = converters.import_coco(out, back_dir)
        self.assertEqual(summary["class_names"], ["aku", "msda"])
        self.assertEqual(summary["annotations"], 2)

        original = yolo_io.load_raw(os.path.join(self.labels, "img0.txt"))
        restored = yolo_io.load_raw(os.path.join(back_dir, "img0.txt"))
        self.assertEqual(len(original), len(restored))
        for a, b in zip(original, restored):
            self.assertEqual(a[0], b[0])
            for x, y in zip(a[1:], b[1:]):
                self.assertAlmostEqual(x, y, places=4)

    def test_import_writes_empty_file_for_image_without_boxes(self):
        out = os.path.join(self.dir, "coco.json")
        converters.export_coco(self.images, self.labels, ["aku", "msda"],
                               self.size_of, out)
        back_dir = os.path.join(self.dir, "back")
        converters.import_coco(out, back_dir)
        empty = os.path.join(back_dir, "img1.txt")
        self.assertTrue(os.path.isfile(empty))
        self.assertEqual(os.path.getsize(empty), 0)

    def test_import_skips_broken_annotations(self):
        path = os.path.join(self.dir, "bozuk.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [
                    {"id": 1, "image_id": 99, "category_id": 1, "bbox": [0, 0, 5, 5]},
                    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 0]},
                    {"id": 3, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                ],
            }, handle)
        summary = converters.import_coco(path, os.path.join(self.dir, "b2"))
        self.assertEqual(summary["annotations"], 1)
        self.assertEqual(summary["skipped"], 2)


class TestVoc(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.labels = os.path.join(self.dir, "labels")
        os.makedirs(self.labels)
        self.images = [os.path.join(self.dir, "img0.jpg")]
        open(self.images[0], "wb").close()
        write_label(os.path.join(self.labels, "img0.txt"),
                    [(0, 0.5, 0.5, 0.25, 0.5), (1, 0.25, 0.25, 0.1, 0.1)])
        self.size_of = lambda p: (400, 200)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_export_creates_xml(self):
        out = os.path.join(self.dir, "voc")
        summary = converters.export_voc(self.images, self.labels,
                                        ["aku", "msda"], self.size_of, out)
        self.assertEqual(summary["files"], 1)
        self.assertEqual(summary["boxes"], 2)
        self.assertTrue(os.path.isfile(os.path.join(out, "img0.xml")))

    def test_round_trip(self):
        out = os.path.join(self.dir, "voc")
        converters.export_voc(self.images, self.labels, ["aku", "msda"],
                              self.size_of, out)
        back = os.path.join(self.dir, "back")
        summary = converters.import_voc(out, back, ["aku", "msda"])
        self.assertEqual(summary["boxes"], 2)
        self.assertEqual(summary["class_names"], ["aku", "msda"])

        original = yolo_io.load_raw(os.path.join(self.labels, "img0.txt"))
        restored = yolo_io.load_raw(os.path.join(back, "img0.txt"))
        self.assertEqual(len(original), len(restored))
        for a, b in zip(original, restored):
            self.assertEqual(a[0], b[0])
            for x, y in zip(a[1:], b[1:]):
                self.assertAlmostEqual(x, y, places=2)

    def test_new_class_is_appended_not_reindexed(self):
        out = os.path.join(self.dir, "voc")
        converters.export_voc(self.images, self.labels, ["aku", "msda"],
                              self.size_of, out)
        back = os.path.join(self.dir, "back")
        summary = converters.import_voc(out, back, ["baska"])
        # Mevcut "baska" 0'da kalmalı, yeni adlar sona eklenmeli
        self.assertEqual(summary["class_names"][0], "baska")
        self.assertIn("aku", summary["class_names"])

    def test_broken_xml_is_skipped(self):
        out = os.path.join(self.dir, "voc")
        os.makedirs(out)
        with open(os.path.join(out, "bozuk.xml"), "w", encoding="utf-8") as handle:
            handle.write("<annotation><size>")
        summary = converters.import_voc(out, os.path.join(self.dir, "b"), [])
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["files"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
