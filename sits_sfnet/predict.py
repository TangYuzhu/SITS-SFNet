"""Predict full regions using overlapping tiles and averaged probabilities."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .data import read_sample, normalize_sample
from .registry import VARIANTS


def tile_starts(length, patch_size=256, stride=128):
    if length < patch_size or not 1 <= stride <= patch_size:
        raise ValueError("Invalid image size or stride")
    return sorted(set(range(0, length - patch_size + 1, stride)) | {length - patch_size})


def predict_region(model, image, stride=128, from_logits=False):
    height, width = image.shape[:2]
    probabilities = np.zeros((height, width, 5), np.float32)
    counts = np.zeros((height, width, 1), np.float32)
    for top in tile_starts(height, stride=stride):
        for left in tile_starts(width, stride=stride):
            patch = image[top : top + 256, left : left + 256]
            output = np.asarray(model(patch[None], training=False))[0]
            if output.shape != (256, 256, 5) or not np.isfinite(output).all():
                raise ValueError("Model must return finite 256x256x5 probabilities")
            if from_logits:
                output = np.exp(output - output.max(axis=-1, keepdims=True))
                output /= output.sum(axis=-1, keepdims=True)
            probabilities[top : top + 256, left : left + 256] += output
            counts[top : top + 256, left : left + 256] += 1
    return np.argmax(probabilities / counts, axis=-1).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="Directory produced by train.py")
    source.add_argument("--checkpoint", type=Path, help="Trusted historical full-model H5")
    parser.add_argument("--variant", choices=VARIANTS, default="sits_sfnet")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import tensorflow as tf
    from . import pooling  # noqa: F401 - registers SegNet serialization classes

    if args.run:
        import json

        variant = json.loads((args.run / "run_config.json").read_text())["variant"]
        model = tf.keras.models.model_from_json((args.run / "architecture.json").read_text())
        model.load_weights(str(args.run / "final.weights.h5"))
    else:
        variant = args.variant
        import h5py

        # Passing an HDF5 handle avoids TensorFlow's native Windows path decoder.
        with h5py.File(args.checkpoint, "r") as checkpoint:
            model = tf.keras.models.load_model(checkpoint, compile=False)
    files = sorted(args.data.rglob("*.npy"))
    if not files or len({p.stem for p in files}) != len(files):
        parser.error("Require nonempty inputs with unique filenames")
    args.output.mkdir(parents=True, exist_ok=False)
    for path in files:
        image, _ = normalize_sample(read_sample(path), variant)
        if image.shape[-1] != model.input_shape[-1]:
            raise ValueError("Input channel count does not match model; check --variant")
        result = predict_region(model, image, from_logits=variant == "deeplabv3plus")
        Image.fromarray(result).save(args.output / f"{path.stem}_prediction.png")


if __name__ == "__main__":
    main()
