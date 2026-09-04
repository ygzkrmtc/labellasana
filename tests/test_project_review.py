"""Proje dosyası ve gözden geçirme kuyruğu testleri (Qt gerektirmez)."""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import project, review  # noqa: E402
from app.models import ClassManager  # noqa: E402


class TestProject(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_round_trip(self):
        classes = ClassManager(["aku", "msda"])
        document = project.build(self.dir, os.path.join(self.dir, "labels"),
                                 list(classes), "model.onnx", {"conf": 0.4})
        path = project.save(os.path.join(self.dir, "p"), document)
        self.assertTrue(path.endswith(project.EXTENSION))

        loaded = project.load(path)
        self.assertEqual([c["name"] for c in loaded["classes"]], ["aku", "msda"])
        self.assertEqual(loaded["model_path"], "model.onnx")
        self.assertAlmostEqual(loaded["settings"]["conf"], 0.4)
        self.assertTrue(project.is_usable(loaded))

    def test_missing_folder_is_not_usable(self):
        document = project.build("/olmayan/klasor", "", [], "")
        self.assertFalse(project.is_usable(document))

    def test_tolerates_partial_file(self):
        path = os.path.join(self.dir, "kismi.labelproj")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"image_dir": self.dir, "classes": ["duz_metin", {"name": "x"}]},
                      handle)
        loaded = project.load(path)
        self.assertEqual([c["name"] for c in loaded["classes"]], ["duz_metin", "x"])
        self.assertEqual(loaded["settings"], {})

    def test_rejects_non_object(self):
        path = os.path.join(self.dir, "liste.labelproj")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([1, 2, 3], handle)
        with self.assertRaises(ValueError):
            project.load(path)


class TestReview(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_mark_and_unmark(self):
        self.assertEqual(review.load_pending(self.dir), set())
        review.mark(self.dir, "a.jpg")
        review.mark(self.dir, "b.jpg")
        self.assertEqual(review.load_pending(self.dir), {"a.jpg", "b.jpg"})
        review.unmark(self.dir, "a.jpg")
        self.assertEqual(review.load_pending(self.dir), {"b.jpg"})

    def test_basename_is_used(self):
        review.mark(self.dir, os.path.join("uzun", "yol", "c.jpg"))
        self.assertEqual(review.load_pending(self.dir), {"c.jpg"})

    def test_clear(self):
        review.mark(self.dir, "a.jpg")
        review.clear(self.dir)
        self.assertEqual(review.load_pending(self.dir), set())

    def test_corrupt_file_is_ignored(self):
        with open(os.path.join(self.dir, review.FILENAME), "w",
                  encoding="utf-8") as handle:
            handle.write("bu json degil")
        self.assertEqual(review.load_pending(self.dir), set())

    def test_missing_dir_is_safe(self):
        self.assertEqual(review.load_pending(""), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
