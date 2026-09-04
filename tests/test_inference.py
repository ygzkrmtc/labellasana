"""
Model çıkarım matematiğinin testleri.

Bu testler onnxruntime gerektirmez: letterbox, NMS, çözümleme ve ölçek geri
alma saf numpy fonksiyonlarıdır. Kutuların resmin doğru yerine düşmesi tam da
buraya bağlı olduğu için ayrıntılı test edilir.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import inference  # noqa: E402

if inference.HAS_NUMPY:
    import numpy as np


@unittest.skipUnless(inference.HAS_NUMPY, "numpy kurulu değil")
class TestLetterbox(unittest.TestCase):
    def test_square_image_fills_canvas(self):
        image = np.zeros((100, 100, 3), np.uint8)
        canvas, ratio, pad_x, pad_y = inference.letterbox(image, 640)
        self.assertEqual(canvas.shape, (640, 640, 3))
        self.assertAlmostEqual(ratio, 6.4, places=5)
        self.assertEqual((pad_x, pad_y), (0.0, 0.0))

    def test_wide_image_is_padded_vertically(self):
        image = np.zeros((100, 200, 3), np.uint8)
        canvas, ratio, pad_x, pad_y = inference.letterbox(image, 640)
        self.assertEqual(canvas.shape, (640, 640, 3))
        self.assertAlmostEqual(ratio, 3.2, places=5)
        self.assertEqual(pad_x, 0.0)
        self.assertGreater(pad_y, 0.0)

    def test_tall_image_is_padded_horizontally(self):
        image = np.zeros((200, 100, 3), np.uint8)
        _, _, pad_x, pad_y = inference.letterbox(image, 640)
        self.assertGreater(pad_x, 0.0)
        self.assertEqual(pad_y, 0.0)

    def test_content_preserved(self):
        image = np.zeros((64, 128, 3), np.uint8)
        image[:, :] = 200
        canvas, ratio, pad_x, pad_y = inference.letterbox(image, 256)
        # Dolgu bölgesi 114, içerik 200 olmalı
        self.assertEqual(int(canvas[0, 0, 0]), 114)
        self.assertEqual(int(canvas[128, 128, 0]), 200)

    def test_invalid_size_rejected(self):
        with self.assertRaises(ValueError):
            inference.letterbox(np.zeros((0, 10, 3), np.uint8), 640)


@unittest.skipUnless(inference.HAS_NUMPY, "numpy kurulu değil")
class TestRoundTrip(unittest.TestCase):
    """letterbox + scale_boxes birbirinin tersi olmalı."""

    def test_box_returns_to_original_position(self):
        orig_w, orig_h = 626, 584
        image = np.zeros((orig_h, orig_w, 3), np.uint8)
        _, ratio, pad_x, pad_y = inference.letterbox(image, 640)

        original = np.array([[100.0, 50.0, 300.0, 250.0]], np.float32)
        # İleri dönüşüm: orijinal -> letterbox
        forward = original * ratio
        forward[:, [0, 2]] += pad_x
        forward[:, [1, 3]] += pad_y
        # Geri dönüşüm
        back = inference.scale_boxes(forward, ratio, pad_x, pad_y, orig_w, orig_h)
        for expected, actual in zip(original[0], back[0]):
            self.assertAlmostEqual(expected, actual, places=3)

    def test_scale_clips_to_image(self):
        boxes = np.array([[-50.0, -50.0, 10000.0, 10000.0]], np.float32)
        result = inference.scale_boxes(boxes, 1.0, 0.0, 0.0, 100, 80)
        self.assertEqual(list(result[0]), [0.0, 0.0, 100.0, 80.0])

    def test_empty_input(self):
        self.assertEqual(len(inference.scale_boxes([], 1.0, 0, 0, 10, 10)), 0)


@unittest.skipUnless(inference.HAS_NUMPY, "numpy kurulu değil")
class TestNms(unittest.TestCase):
    def test_suppresses_overlapping(self):
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]]
        scores = [0.9, 0.8, 0.7]
        keep = inference.nms(boxes, scores, 0.5)
        self.assertEqual(sorted(keep), [0, 2])

    def test_keeps_all_when_no_overlap(self):
        boxes = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
        keep = inference.nms(boxes, [0.9, 0.8, 0.7], 0.5)
        self.assertEqual(sorted(keep), [0, 1, 2])

    def test_highest_score_wins(self):
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        keep = inference.nms(boxes, [0.3, 0.95], 0.5)
        self.assertEqual(keep, [1])

    def test_empty(self):
        self.assertEqual(inference.nms([], [], 0.5), [])

    def test_class_aware_keeps_different_classes(self):
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        keep = inference.class_aware_nms(boxes, [0.9, 0.8], [0, 1], 0.5)
        self.assertEqual(sorted(keep), [0, 1])       # farklı sınıf, ikisi de kalır

    def test_class_aware_suppresses_same_class(self):
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        keep = inference.class_aware_nms(boxes, [0.9, 0.8], [1, 1], 0.5)
        self.assertEqual(keep, [0])


@unittest.skipUnless(inference.HAS_NUMPY, "numpy kurulu değil")
class TestDecode(unittest.TestCase):
    def _make_output(self, num_classes=3, anchors=20):
        """(1, 4+nc, anchors) biçiminde sentetik YOLOv8 çıktısı."""
        output = np.zeros((1, 4 + num_classes, anchors), np.float32)
        # 0. çapa: sınıf 1, güven 0.9, merkez (100,100), boyut 40x60
        output[0, :4, 0] = [100, 100, 40, 60]
        output[0, 4 + 1, 0] = 0.9
        # 1. çapa: sınıf 0, güven 0.1 (eşiğin altında)
        output[0, :4, 1] = [200, 200, 10, 10]
        output[0, 4 + 0, 1] = 0.1
        return output

    def test_filters_by_confidence(self):
        boxes, scores, classes = inference.decode_output(self._make_output(), 0.25)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(int(classes[0]), 1)
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)

    def test_xywh_to_xyxy(self):
        boxes, _, _ = inference.decode_output(self._make_output(), 0.25)
        x1, y1, x2, y2 = boxes[0]
        self.assertAlmostEqual(x1, 80.0, places=3)    # 100 - 40/2
        self.assertAlmostEqual(y1, 70.0, places=3)    # 100 - 60/2
        self.assertAlmostEqual(x2, 120.0, places=3)
        self.assertAlmostEqual(y2, 130.0, places=3)

    def test_transposed_layout_is_handled(self):
        output = self._make_output()
        transposed = np.transpose(output, (0, 2, 1))   # (1, anchors, 4+nc)
        a = inference.decode_output(output, 0.25)
        b = inference.decode_output(transposed, 0.25)
        self.assertTrue(np.allclose(a[0], b[0]))
        self.assertTrue(np.allclose(a[1], b[1]))

    def test_attribute_axis_resolved_by_class_count(self):
        """Çapa sayısı öznitelik sayısından azsa bile doğru eksen seçilmeli."""
        output = np.zeros((1, 7, 3), np.float32)      # 3 sınıf, sadece 3 çapa
        output[0, :4, 0] = [10, 10, 4, 4]
        output[0, 4 + 2, 0] = 0.8
        _, scores, classes = inference.decode_output(output, 0.25, num_classes=3)
        self.assertEqual(len(scores), 1)
        self.assertEqual(int(classes[0]), 2)

    def test_v5_layout_with_objectness(self):
        num_classes = 2
        output = np.zeros((1, 5 + num_classes, 30), np.float32)
        output[0, :4, 0] = [50, 50, 20, 20]
        output[0, 4, 0] = 0.8            # objectness
        output[0, 5 + 1, 0] = 0.5        # sınıf skoru -> 0.8*0.5 = 0.4
        _, scores, classes = inference.decode_output(output, 0.25, num_classes=2)
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(float(scores[0]), 0.4, places=5)
        self.assertEqual(int(classes[0]), 1)

    def test_nothing_above_threshold(self):
        boxes, scores, classes = inference.decode_output(self._make_output(), 0.99)
        self.assertEqual(len(boxes), 0)
        self.assertEqual(len(scores), 0)
        self.assertEqual(len(classes), 0)

    def test_bad_shape_rejected(self):
        with self.assertRaises(ValueError):
            inference.decode_output(np.zeros((4,), np.float32))


@unittest.skipUnless(inference.HAS_NUMPY, "numpy kurulu değil")
class TestPostprocess(unittest.TestCase):
    def test_end_to_end(self):
        output = np.zeros((1, 6, 30), np.float32)     # 2 sınıf, 30 çapa
        output[0, :4, 0] = [320, 320, 100, 100]
        output[0, 4, 0] = 0.9                          # sınıf 0
        output[0, :4, 1] = [322, 322, 100, 100]        # neredeyse aynı kutu
        output[0, 4, 1] = 0.7
        detections = inference.postprocess(output, ratio=1.0, pad_x=0, pad_y=0,
                                           orig_w=640, orig_h=640,
                                           conf_threshold=0.25, iou_threshold=0.5)
        self.assertEqual(len(detections), 1)           # NMS ikinciyi eledi
        class_id, x1, y1, x2, y2, score = detections[0]
        self.assertEqual(class_id, 0)
        self.assertAlmostEqual(score, 0.9, places=5)
        self.assertAlmostEqual(x1, 270.0, places=3)

    def test_tiny_boxes_dropped(self):
        output = np.zeros((1, 5, 30), np.float32)
        output[0, :4, 0] = [100, 100, 0.5, 0.5]
        output[0, 4, 0] = 0.9
        detections = inference.postprocess(output, 1.0, 0, 0, 640, 640, 0.25)
        self.assertEqual(detections, [])

    def test_sorted_by_score(self):
        output = np.zeros((1, 5, 30), np.float32)
        output[0, :4, 0] = [100, 100, 20, 20]
        output[0, 4, 0] = 0.4
        output[0, :4, 1] = [300, 300, 20, 20]
        output[0, 4, 1] = 0.95
        detections = inference.postprocess(output, 1.0, 0, 0, 640, 640, 0.25)
        self.assertEqual(len(detections), 2)
        self.assertGreater(detections[0][5], detections[1][5])


class TestClassMapping(unittest.TestCase):
    def test_exact_and_case_insensitive(self):
        mapping = inference.map_model_classes(["Aku", "MSDA"], ["msda", "aku"])
        self.assertEqual(mapping, {0: 1, 1: 0})

    def test_unmatched_classes_are_absent(self):
        mapping = inference.map_model_classes(["kedi"], ["aku"])
        self.assertEqual(mapping, {})

    def test_whitespace_tolerated(self):
        self.assertEqual(inference.map_model_classes([" aku "], ["aku"]), {0: 0})


class TestAvailability(unittest.TestCase):
    def test_reason_is_empty_when_available(self):
        if inference.available():
            self.assertEqual(inference.unavailable_reason(), "")
        else:
            self.assertTrue(inference.unavailable_reason())


if __name__ == "__main__":
    unittest.main(verbosity=2)
