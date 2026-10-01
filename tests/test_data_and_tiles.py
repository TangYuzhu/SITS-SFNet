"""Tests of aligned cropping, normalization, leakage guards and complete tiling."""

from pathlib import Path
import unittest
import numpy as np
from sits_sfnet.data import THRESHOLDS, normalize_sample, crop_sample, check_split
from sits_sfnet.predict import predict_region


class DataTests(unittest.TestCase):
    def test_normalization_and_selection(self):
        sample = np.zeros((2, 2, 54), np.float32)
        sample[..., :48] = np.tile(THRESHOLDS[:, 0], 3)
        sample[..., 51:53] = 255
        sample[..., 53] = 4
        image, label = normalize_sample(sample, "sits_sfnet")
        self.assertTrue((image[..., :48] == 0).all())
        self.assertTrue((image[..., 51:53] == 1).all())
        self.assertTrue((label == 4).all())
        self.assertEqual(normalize_sample(sample, "unet_3ch")[0].shape[-1], 3)
        self.assertEqual(normalize_sample(sample, "unet_16ch")[0].shape[-1], 16)

    def test_aligned_crop_and_split(self):
        sample = np.broadcast_to(np.arange(600)[:, None, None], (600, 600, 54))
        crop = crop_sample(sample, np.random.default_rng(12))
        self.assertEqual(crop.shape, (256, 256, 54))
        np.testing.assert_array_equal(crop[..., 0], crop[..., -1])
        with self.assertRaises(ValueError):
            check_split([Path("202001010320.npy")], [Path("202001010330.npy")])

    def test_tiles_cover_all_edges(self):
        class ConstantModel:
            def __call__(self, batch, training=False):
                output = np.zeros((1, 256, 256, 5), np.float32)
                output[..., 2] = 1
                return output

        result = predict_region(ConstantModel(), np.zeros((600, 600, 3), np.float32))
        self.assertEqual(result.shape, (600, 600))
        self.assertTrue((result == 2).all())


if __name__ == "__main__":
    unittest.main()
