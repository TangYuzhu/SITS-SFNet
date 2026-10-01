"""Train a selected architecture with explicit, recorded split manifests."""

import argparse
import json
import random
from pathlib import Path
import numpy as np
from .data import read_sample, normalize_sample, read_manifest, check_split, crop_sample
from .registry import VARIANTS, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--validation-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="sits_sfnet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--allow-shared-dates", action="store_true")
    parser.add_argument(
        "--backbone-weights",
        default="imagenet",
        help="DeepLab only: imagenet, a local ResNet50 weights file, or none",
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size and learning rate must be positive")
    train_files = read_manifest(args.train_list, args.data)
    val_files = read_manifest(args.validation_list, args.data)
    check_split(train_files, val_files, args.allow_shared_dates)
    import tensorflow as tf

    # Avoid the legacy Keras seed generator's float randint bound on Python 3.12.
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build_model(
        args.variant, None if args.backbone_weights == "none" else args.backbone_weights
    )
    channels = model.input_shape[-1]

    def samples(files, training):
        order = rng.permutation(len(files)) if training else range(len(files))
        for index in order:
            sample = crop_sample(read_sample(files[index]), rng if training else None)
            image, label = normalize_sample(sample, args.variant)
            if image.shape[-1] != channels:
                raise ValueError(
                    f"Architecture expects {channels} channels; input has {image.shape[-1]}"
                )
            yield image, label

    signature = (
        tf.TensorSpec((256, 256, channels), tf.float32),
        tf.TensorSpec((256, 256), tf.uint8),
    )
    train = tf.data.Dataset.from_generator(
        lambda: samples(train_files, True), output_signature=signature
    ).batch(args.batch_size)
    validation = tf.data.Dataset.from_generator(
        lambda: samples(val_files, False), output_signature=signature
    ).batch(args.batch_size)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, files in [("train", train_files), ("validation", val_files)]:
        (args.output / f"{name}.txt").write_text(
            "\n".join(p.relative_to(args.data.resolve()).as_posix() for p in files) + "\n",
            encoding="utf-8",
        )
    settings = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    settings["tensorflow_version"] = tf.__version__
    settings["numpy_version"] = np.__version__
    (args.output / "run_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (args.output / "architecture.json").write_text(model.to_json(), encoding="utf-8")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=args.variant == "deeplabv3plus"
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    history = model.fit(
        train,
        validation_data=validation,
        epochs=args.epochs,
        callbacks=[tf.keras.callbacks.CSVLogger(str(args.output / "history.csv"))],
    )
    model.save_weights(str(args.output / "final.weights.h5"))
    (args.output / "history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
