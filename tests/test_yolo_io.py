"""
YOLO dönüşüm matematiğinin ve sınıf yönetiminin birim testleri.
Qt gerektirmez:  python -m tests.test_yolo_io   (veya pytest)
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import yolo_io          # noqa: E402
from app.models import ClassManager  # noqa: E402


class TestNormalization(unittest.TestCase):
    def test_center_box(self):
        # 640x480 resimde tam ortada 100x80'lik kutu
        xc, yc, w, h = yolo_io.to_yolo(270, 200, 370, 280, 640, 480)
        self.assertAlmostEqual(xc, 320 / 640)
        self.assertAlmostEqual(yc, 240 / 480)
        self.assertAlmostEqual(w, 100 / 640)
        self.assertAlmostEqual(h, 80 / 480)

    def test_round_trip(self):
        for box in [(0, 0, 640, 480), (10.5, 20.25, 100.75, 200.5), (639, 479, 640, 480)]:
            xc, yc, w, h = yolo_io.to_yolo(*box, 640, 480)
            back = yolo_io.from_yolo(xc, yc, w, h, 640, 480)
            for expected, actual in zip(box, back):
                self.assertAlmostEqual(expected, actual, places=4)

    def test_reversed_corners_are_normalized(self):
        a = yolo_io.to_yolo(370, 280, 270, 200, 640, 480)
        b = yolo_io.to_yolo(270, 200, 370, 280, 640, 480)
        self.assertEqual(a, b)

    def test_out_of_bounds_is_clipped(self):
        xc, yc, w, h = yolo_io.to_yolo(-50, -50, 700, 600, 640, 480)
        self.assertAlmostEqual(w, 1.0)
        self.assertAlmostEqual(h, 1.0)
        self.assertAlmostEqual(xc, 0.5)
        self.assertAlmostEqual(yc, 0.5)
        for v in (xc, yc, w, h):
            self.assertTrue(0.0 <= v <= 1.0)


class TestFileIO(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_save_and_load(self):
        txt = os.path.join(self.dir, "labels", "resim.txt")
        boxes = [(0, 10, 20, 110, 220), (2, 300, 100, 400, 150)]
        yolo_io.save_labels(txt, boxes, 800, 600)

        with open(txt, encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("0 "))
        self.assertEqual(len(lines[0].split()), 5)

        loaded = yolo_io.load_labels(txt, 800, 600)
        self.assertEqual(len(loaded), 2)
        for original, restored in zip(boxes, loaded):
            self.assertEqual(original[0], restored[0])
            for a, b in zip(original[1:], restored[1:]):
                self.assertAlmostEqual(a, b, places=2)

    def test_empty_file_for_no_boxes(self):
        txt = os.path.join(self.dir, "bos.txt")
        yolo_io.save_labels(txt, [], 100, 100)
        self.assertTrue(os.path.isfile(txt))
        self.assertEqual(os.path.getsize(txt), 0)
        self.assertEqual(yolo_io.load_labels(txt, 100, 100), [])

    def test_corrupt_lines_are_skipped(self):
        txt = os.path.join(self.dir, "bozuk.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
            f.write("bu bir cop satiri\n")
            f.write("\n")
            f.write("1 0.1 0.1\n")
            f.write("1 0.25 0.25 0.1 0.1\n")
        loaded = yolo_io.load_labels(txt, 200, 200)
        self.assertEqual(len(loaded), 2)

    def test_classes_round_trip(self):
        path = os.path.join(self.dir, "classes.txt")
        yolo_io.save_classes(path, ["araba", "yaya", "bisiklet"])
        self.assertEqual(yolo_io.load_classes(path), ["araba", "yaya", "bisiklet"])

    def test_remap_after_class_delete(self):
        label_dir = os.path.join(self.dir, "labels")
        os.makedirs(label_dir)
        p = os.path.join(label_dir, "a.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n2 0.5 0.5 0.1 0.1\n")
        cm = ClassManager(["a", "b", "c"])
        mapping = cm.remove(1)                      # 'b' silindi
        self.assertEqual(mapping, {0: 0, 1: None, 2: 1})
        yolo_io.remap_label_files(label_dir, mapping)
        with open(p, encoding="utf-8") as f:
            ids = [ln.split()[0] for ln in f.read().splitlines() if ln.strip()]
        self.assertEqual(ids, ["0", "1"])
        self.assertEqual(cm.names(), ["a", "c"])


class TestClassManager(unittest.TestCase):
    def test_add_and_colors(self):
        cm = ClassManager()
        cm.add("araba")
        cm.add("yaya")
        self.assertEqual(cm.names(), ["araba", "yaya"])
        self.assertNotEqual(cm.color_of(0), cm.color_of(1))

    def test_duplicate_rejected(self):
        cm = ClassManager(["araba"])
        with self.assertRaises(ValueError):
            cm.add("araba")

    def test_ensure_capacity(self):
        cm = ClassManager(["araba"])
        cm.ensure_capacity(3)
        self.assertEqual(len(cm), 4)
        self.assertEqual(cm.name_of(3), "class_3")

    def test_unknown_id_is_safe(self):
        cm = ClassManager()
        self.assertEqual(cm.name_of(7), "?7")
        self.assertEqual(cm.color_of(7), "#9e9e9e")


if __name__ == "__main__":
    unittest.main(verbosity=2)
