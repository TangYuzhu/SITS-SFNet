"""Optional TensorFlow check of SegNet's per-example argmax scatter."""

import importlib.util
import unittest
import numpy as np
import sits_sfnet  # noqa: F401 - select legacy Keras before importing TensorFlow


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow not installed")
class PoolingTests(unittest.TestCase):
    def test_batch_indices_and_gradient(self):
        import tensorflow as tf
        from sits_sfnet.pooling import MaxPoolingWithArgmax2D, MaxUnpooling2D

        values = tf.Variable(np.arange(32, dtype=np.float32).reshape(2, 4, 4, 1))
        with tf.GradientTape() as tape:
            pooled, indices = MaxPoolingWithArgmax2D()(values)
            restored = MaxUnpooling2D()([pooled, indices])
            loss = tf.reduce_sum(restored)
        expected = np.zeros((2, 4, 4, 1), np.float32)
        expected[:, 1::2, 1::2] = values.numpy()[:, 1::2, 1::2]
        np.testing.assert_array_equal(restored.numpy(), expected)
        gradient = np.zeros_like(expected)
        gradient[:, 1::2, 1::2] = 1
        np.testing.assert_array_equal(tape.gradient(loss, values).numpy(), gradient)


if __name__ == "__main__":
    unittest.main()
