"""Offline MiT-B0 forward, categorical resize, serialization and optimizer check."""

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sits_sfnet.segformer import build_segformer, prepare_patch
import tensorflow as tf
import numpy as np
from transformers import TFSegformerForSemanticSegmentation


def main():
    labels = np.zeros((256, 256), np.uint8)
    labels[:, 128:] = 4
    prepared = prepare_patch(np.zeros((256, 256, 3), np.float32), labels)
    assert set(np.unique(prepared["labels"])) == {0, 4}
    batch = {name: tensor[None] for name, tensor in prepared.items()}
    model = build_segformer(random_init=True)
    result = model(**batch, training=False)
    assert tuple(result.logits.shape) == (1, 5, 128, 128)
    assert np.isfinite(result.logits.numpy()).all()
    assert np.isfinite(result.loss.numpy()).all()
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        restored = TFSegformerForSemanticSegmentation.from_pretrained(directory)
        np.testing.assert_allclose(
            restored(**batch, training=False).logits.numpy(), result.logits.numpy(), atol=1e-6
        )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.00006))
    loss = model.train_on_batch(batch)
    assert np.isfinite(loss).all()
    print(
        json.dumps(
            {
                "variant": "segformer_b0",
                "parameters": model.count_params(),
                "logits_shape": result.logits.shape.as_list(),
                "round_trip": "passed",
                "categorical_resize": "passed",
                "training_loss": np.asarray(loss).tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
