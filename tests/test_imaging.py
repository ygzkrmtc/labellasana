"""Görüntü iyileştirme ve manyetik kutu testleri (Qt gerektirmez)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import imaging  # noqa: E402

if imaging.HAS_NUMPY:
    import numpy as np


@unittest.skipUnless(imaging.HAS_NUMPY, "numpy kurulu değil")
class TestLut(unittest.TestCase):
    def test_identity(self):
        lut = imaging.build_lut(0, 0, 1.0)
        self.assertTrue((lut == np.arange(256, dtype=np.uint8)).all())

    def test_brightness_increases(self):
        lut = imaging.build_lut(50, 0, 1.0)
        self.assertGreater(int(lut[128]), 128)
        self.assertEqual(int(lut[255]), 255)      # doyuma kırpılır

    def test_brightness_decreases(self):
        lut = imaging.build_lut(-50, 0, 1.0)
        self.assertLess(int(lut[128]), 128)
        self.assertEqual(int(lut[0]), 0)

    def test_gamma_brightens_shadows(self):
        lut = imaging.build_lut(0, 0, 2.0)
        self.assertGreater(int(lut[64]), 64)      # karanlık bölge açılır

    def test_contrast_spreads(self):
        lut = imaging.build_lut(0, 60, 1.0)
        self.assertLess(int(lut[64]), 64)
        self.assertGreater(int(lut[192]), 192)

    def test_monotonic_and_in_range(self):
        for params in [(0, 0, 1.0), (30, 30, 1.5), (-30, -30, 0.7), (100, 100, 3.0)]:
            lut = imaging.build_lut(*params)
            self.assertEqual(lut.dtype, np.uint8)
            self.assertTrue((np.diff(lut.astype(int)) >= 0).all(),
                            f"LUT monoton değil: {params}")

    def test_out_of_range_inputs_are_clamped(self):
        imaging.build_lut(9999, -9999, 0.0)       # patlamamalı


@unittest.skipUnless(imaging.HAS_NUMPY, "numpy kurulu değil")
class TestEnhance(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.image = rng.integers(0, 60, (40, 50, 3), dtype=np.uint8)  # karanlık resim

    def test_identity_shortcut(self):
        self.assertTrue(imaging.is_identity(0, 0, 1.0, False))
        self.assertFalse(imaging.is_identity(0, 0, 1.0, True))
        self.assertFalse(imaging.is_identity(5, 0, 1.0, False))

    def test_enhance_does_not_modify_input(self):
        original = self.image.copy()
        imaging.enhance(self.image, brightness=40, gamma=1.8)
        self.assertTrue((self.image == original).all())

    def test_enhance_brightens(self):
        result = imaging.enhance(self.image, brightness=40)
        self.assertGreater(result.mean(), self.image.mean())
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_clahe_runs_and_keeps_shape(self):
        result = imaging.clahe(self.image)
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_clahe_increases_contrast_on_low_contrast_image(self):
        # 100-120 bandına sıkışmış dokulu bir resim: tam olarak CLAHE'nin
        # açması beklenen durum (karanlık kabin fotoğrafının minyatürü).
        rng = np.random.default_rng(3)
        low_contrast = rng.integers(100, 121, (96, 96, 3), dtype=np.uint8)
        result = imaging.clahe(low_contrast, clip_limit=4.0)
        self.assertGreater(float(result.std()), float(low_contrast.std()) * 1.5,
                           "CLAHE kontrastı belirgin biçimde artırmalı")

    def test_gray_conversion(self):
        gray = imaging.to_gray(self.image)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.shape, self.image.shape[:2])


@unittest.skipUnless(imaging.HAS_NUMPY, "numpy kurulu değil")
class TestEdges(unittest.TestCase):
    def test_edges_found_on_rectangle(self):
        image = np.zeros((80, 80, 3), np.uint8)
        image[20:60, 20:60] = 255
        edges = imaging.edge_map(image)
        self.assertEqual(edges.shape, (80, 80))
        self.assertGreater(int((edges > 0).sum()), 0)

    def test_no_edges_on_flat_image(self):
        flat = np.full((40, 40, 3), 128, np.uint8)
        self.assertEqual(int((imaging.edge_map(flat) > 0).sum()), 0)


@unittest.skipUnless(imaging.HAS_NUMPY, "numpy kurulu değil")
class TestSnapBox(unittest.TestCase):
    def setUp(self):
        # Siyah zemin üzerinde 30..90 arasında beyaz dikdörtgen
        self.image = np.zeros((120, 120, 3), np.uint8)
        self.image[30:90, 30:90] = 255

    def test_snaps_outward_box_to_edges(self):
        result = imaging.snap_box(self.image, 24, 25, 96, 95)
        self.assertIsNotNone(result)
        x1, y1, x2, y2 = result
        for value, expected in ((x1, 30), (y1, 30), (x2, 90), (y2, 90)):
            self.assertLessEqual(abs(value - expected), 2,
                                 f"{value} ~ {expected} olmalıydı")

    def test_snaps_inward_box_to_edges(self):
        result = imaging.snap_box(self.image, 35, 34, 85, 86)
        self.assertIsNotNone(result)
        x1, y1, x2, y2 = result
        for value, expected in ((x1, 30), (y1, 30), (x2, 90), (y2, 90)):
            self.assertLessEqual(abs(value - expected), 3)

    def test_returns_none_for_tiny_box(self):
        self.assertIsNone(imaging.snap_box(self.image, 10, 10, 13, 13))

    def test_handles_box_at_image_border(self):
        result = imaging.snap_box(self.image, 0, 0, 40, 40)
        self.assertIsNotNone(result)          # patlamamalı
        x1, y1, x2, y2 = result
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 120)
        self.assertLessEqual(y2, 120)

    def test_flat_image_returns_original_ish(self):
        flat = np.full((100, 100, 3), 128, np.uint8)
        result = imaging.snap_box(flat, 20, 20, 60, 60)
        self.assertIsNotNone(result)
        x1, y1, x2, y2 = result
        # Kenar yoksa kutu yerinde kalmalı
        self.assertAlmostEqual(x1, 20, delta=1)
        self.assertAlmostEqual(y2, 60, delta=1)

    def test_reversed_coordinates_are_normalized(self):
        a = imaging.snap_box(self.image, 96, 95, 24, 25)
        b = imaging.snap_box(self.image, 24, 25, 96, 95)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
