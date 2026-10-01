"""Optional TensorFlow model smoke test, isolated per variant to bound memory use."""

import argparse
import json
import random
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sits_sfnet.registry import build_model, VARIANTS
import tensorflow as tf
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--train-step", action="store_true")
    args = parser.parse_args()
    random.seed(12)
    np.random.seed(12)
    tf.random.set_seed(12)
    model = build_model(args.variant, backbone_weights=None)
    image = np.zeros((1, *model.input_shape[1:]), np.float32)
    output = model(image, training=False).numpy()
    assert output.shape == (1, 256, 256, 5)
    assert np.isfinite(output).all()
    if args.variant != "deeplabv3plus":
        np.testing.assert_allclose(output.sum(axis=-1), 1, atol=1e-5)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "test.weights.h5"
        model.save_weights(str(path))
        restored = tf.keras.models.model_from_json(model.to_json())
        restored.load_weights(str(path))
        np.testing.assert_allclose(restored(image, training=False).numpy(), output, atol=1e-6)
        del restored
    metrics = None
    if args.train_step:
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=args.variant == "deeplabv3plus"
            ),
        )
        metrics = float(model.train_on_batch(image, np.zeros((1, 256, 256), np.uint8)))
        assert np.isfinite(metrics)
    print(
        json.dumps(
            {
                "variant": args.variant,
                "parameters": model.count_params(),
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "round_trip": "passed",
                "training_loss": metrics,
            }
        )
    )


if __name__ == "__main__":
    main()
