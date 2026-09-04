"""Veri seti bölme, istatistik ve dışa aktarma testleri (Qt gerektirmez)."""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import dataset, yolo_io  # noqa: E402


def write_label(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cid, xc, yc, w, h in rows:
            f.write(f"{cid} {xc} {yc} {w} {h}\n")


class TestSplit(unittest.TestCase):
    def test_ratios_and_completeness(self):
        items = [(f"img_{i}.jpg", (i % 3,)) for i in range(100)]
        split = dataset.stratified_split(items, (70, 20, 10), seed=1)
        total = sum(len(v) for v in split.values())
        self.assertEqual(total, 100)
        # Hiçbir resim iki bölmede birden olmamalı
        everything = split["train"] + split["val"] + split["test"]
        self.assertEqual(len(set(everything)), 100)
        self.assertAlmostEqual(len(split["train"]), 70, delta=3)
        self.assertAlmostEqual(len(split["val"]), 20, delta=3)
        self.assertAlmostEqual(len(split["test"]), 10, delta=3)

    def test_deterministic(self):
        items = [(f"img_{i}.jpg", (i % 2,)) for i in range(40)]
        a = dataset.stratified_split(items, (70, 20, 10), seed=7)
        b = dataset.stratified_split(items, (70, 20, 10), seed=7)
        self.assertEqual(a, b)
        c = dataset.stratified_split(items, (70, 20, 10), seed=8)
        self.assertNotEqual(a["train"], c["train"])

    def test_stratification_keeps_classes_in_train(self):
        # 30 resim: 10'u sınıf 0, 10'u sınıf 1, 10'u ikisi birden
        items = ([(f"a{i}.jpg", (0,)) for i in range(10)]
                 + [(f"b{i}.jpg", (1,)) for i in range(10)]
                 + [(f"c{i}.jpg", (0, 1)) for i in range(10)])
        split = dataset.stratified_split(items, (60, 40, 0), seed=3)
        self.assertEqual(len(split["test"]), 0)     # test payı 0 istendi
        for group in ("a", "b", "c"):
            self.assertTrue(any(p.startswith(group) for p in split["train"]),
                            f"{group} grubu eğitim setinde yok")
            self.assertTrue(any(p.startswith(group) for p in split["val"]),
                            f"{group} grubu doğrulama setinde yok")

    def test_zero_ratio_sum_rejected(self):
        with self.assertRaises(ValueError):
            dataset.stratified_split([("a.jpg", ())], (0, 0, 0))

    def test_single_image(self):
        split = dataset.stratified_split([("a.jpg", (0,))], (70, 20, 10), seed=1)
        self.assertEqual(sum(len(v) for v in split.values()), 1)


class TestStats(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.labels = os.path.join(self.dir, "labels")
        self.images = []
        for i in range(3):
            image = os.path.join(self.dir, f"img{i}.jpg")
            open(image, "wb").close()
            self.images.append(image)
        write_label(os.path.join(self.labels, "img0.txt"),
                    [(0, 0.5, 0.5, 0.4, 0.4), (1, 0.2, 0.2, 0.02, 0.02)])
        write_label(os.path.join(self.labels, "img1.txt"), [])   # boş = arka plan
        # img2 için etiket yok

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_counts(self):
        stats = dataset.collect_stats(self.images, self.labels, 2)
        self.assertEqual(stats["images"], 3)
        self.assertEqual(stats["labeled"], 1)
        self.assertEqual(stats["empty"], 1)
        self.assertEqual(stats["missing"], 1)
        self.assertEqual(stats["total_boxes"], 2)
        self.assertEqual(stats["per_class"][0], 1)
        self.assertEqual(stats["per_class"][1], 1)

    def test_size_buckets_and_heatmap(self):
        stats = dataset.collect_stats(self.images, self.labels, 2)
        self.assertEqual(stats["size_buckets"]["büyük"], 1)   # 0.4*0.4 = 0.16
        self.assertEqual(stats["size_buckets"]["küçük"], 1)   # 0.02*0.02 = 0.0004
        self.assertEqual(sum(sum(r) for r in stats["heatmap"]), 2)

    def test_unknown_ids_detected(self):
        write_label(os.path.join(self.labels, "img2.txt"), [(9, 0.5, 0.5, 0.1, 0.1)])
        stats = dataset.collect_stats(self.images, self.labels, 2)
        self.assertEqual(stats["unknown_ids"], [9])


class TestExport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.labels = os.path.join(self.dir, "labels")
        self.out = os.path.join(self.dir, "out")
        self.images = []
        for i in range(6):
            image = os.path.join(self.dir, f"img{i}.jpg")
            with open(image, "wb") as f:
                f.write(b"fake")
            self.images.append(image)
            write_label(os.path.join(self.labels, f"img{i}.txt"),
                        [(0, 0.5, 0.5, 0.2, 0.2)])

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_structure_and_yaml(self):
        items = [(p, (0,)) for p in self.images]
        split = dataset.stratified_split(items, (50, 50, 0), seed=1)
        result = dataset.export_dataset(split, self.labels, self.out,
                                        ["aku", "msda"], "copy")
        self.assertFalse(result["cancelled"])
        self.assertEqual(result["total"], 6)

        for group in ("train", "val"):
            image_dir = os.path.join(self.out, "images", group)
            label_dir = os.path.join(self.out, "labels", group)
            self.assertTrue(os.path.isdir(image_dir))
            # Her resmin yanında bir .txt olmalı
            for name in os.listdir(image_dir):
                stem = os.path.splitext(name)[0]
                self.assertTrue(os.path.isfile(os.path.join(label_dir, stem + ".txt")))

        with open(result["yaml"], encoding="utf-8") as f:
            content = f.read()
        self.assertIn("nc: 2", content)
        self.assertIn("0: aku", content)
        self.assertIn("train: images/train", content)

    def test_missing_label_becomes_empty_file(self):
        os.remove(os.path.join(self.labels, "img0.txt"))
        split = {"train": self.images, "val": [], "test": []}
        dataset.export_dataset(split, self.labels, self.out, ["aku"], "copy")
        target = os.path.join(self.out, "labels", "train", "img0.txt")
        self.assertTrue(os.path.isfile(target))
        self.assertEqual(os.path.getsize(target), 0)

    def test_cancel(self):
        split = {"train": self.images, "val": [], "test": []}
        result = dataset.export_dataset(
            split, self.labels, self.out, ["aku"], "copy",
            progress=lambda done, total, name: done < 2)
        self.assertTrue(result["cancelled"])


class TestBackup(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.txt = os.path.join(self.dir, "a.txt")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_backup_and_list(self):
        with open(self.txt, "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        created = yolo_io.backup_label(self.txt)
        self.assertIsNotNone(created)
        self.assertTrue(os.path.isfile(created))
        self.assertEqual(len(yolo_io.list_backups(self.txt)), 1)

    def test_no_backup_for_missing_file(self):
        self.assertIsNone(yolo_io.backup_label(self.txt))
        self.assertEqual(yolo_io.list_backups(self.txt), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
