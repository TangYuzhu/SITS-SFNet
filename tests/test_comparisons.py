"""Behavioral checks for comparison baselines independent of pretrained downloads."""

import unittest
import numpy as np
from sits_sfnet.dynamic_threshold import classify, load_calibration, ASSETS
from sits_sfnet.evaluate import summarize


class ComparisonTests(unittest.TestCase):
    def test_threshold_does_not_mutate_input(self):
        tables = load_calibration(ASSETS)
        image = np.full((8, 8, 16), 100, np.uint16)
        image[..., 2:4] = 2200
        original = image.copy()
        mask = np.ones((8, 8), bool)
        mask[0] = False
        labels, threshold = classify(image, tables, mask)
        np.testing.assert_array_equal(image, original)
        self.assertTrue((labels[0] == 0).all())
        self.assertTrue(np.isin(labels, [0, 1, 2, 3]).all())
        self.assertTrue(np.isfinite(threshold))

    def test_invalid_lookup_index_rejected(self):
        image = np.zeros((4, 4, 16), np.int16)
        image[..., 4] = -1
        with self.assertRaises(ValueError):
            classify(image, load_calibration(ASSETS), np.ones((4, 4), bool))

    def test_metrics_and_absent_class(self):
        matrix = np.zeros((5, 5), np.int64)
        matrix[1, 1] = 8
        matrix[1, 2] = 2
        matrix[2, 2] = 10
        report = summarize(matrix)
        self.assertAlmostEqual(report["pixel_accuracy"], 0.9)
        self.assertAlmostEqual(report["precision"][2], 10 / 12)
        self.assertAlmostEqual(report["recall"][1], 0.8)
        self.assertIsNone(report["iou"][4])

    def test_land_excluded_from_mean_but_ocean_errors_retained(self):
        matrix = np.zeros((5, 5), np.int64)
        matrix[2, 2] = 8
        matrix[2, 0] = 2
        report = summarize(matrix, ignored_classes=[0])
        self.assertAlmostEqual(report["mean_iou_present_classes"], 0.8)
        self.assertAlmostEqual(report["pixel_accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
