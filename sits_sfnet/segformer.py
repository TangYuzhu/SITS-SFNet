"""Train and predict with the historical three-channel MiT-B0 SegFormer baseline."""

import argparse
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
from .data import read_sample, normalize_sample, read_manifest, check_split, crop_sample
from .predict import predict_region

CLASS_NAMES = {0: "land", 1: "sea", 2: "fog", 3: "cloud", 4: "cloud_obscured_sea_fog"}


def build_segformer(checkpoint="nvidia/mit-b0", random_init=False):
    from transformers import SegformerConfig, TFSegformerForSemanticSegmentation

    settings = dict(
        num_labels=5,
        id2label=CLASS_NAMES,
        label2id={name: index for index, name in CLASS_NAMES.items()},
    )
    if random_init:
        return TFSegformerForSemanticSegmentation(SegformerConfig(**settings))
    return TFSegformerForSemanticSegmentation.from_pretrained(
        checkpoint, ignore_mismatched_sizes=True, **settings
    )


def prepare_patch(image, label=None):
    import tensorflow as tf

    image = tf.image.resize(image, (512, 512), method="bilinear")
    image = (image - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])
    result = {"pixel_values": tf.transpose(image, (2, 0, 1))}
    if label is not None:
        # Categorical targets must retain class indices during resizing.
        label = tf.image.resize(label[..., None], (512, 512), method="nearest")
        result["labels"] = tf.cast(label[..., 0], tf.int32)
    return result


class SegformerPredictor:
    def __init__(self, model):
        self.model = model

    def __call__(self, batch, training=False):
        import tensorflow as tf

        pixels = tf.stack([prepare_patch(image)["pixel_values"] for image in batch])
        logits = self.model(pixel_values=pixels, training=training).logits
        logits = tf.image.resize(tf.transpose(logits, (0, 2, 3, 1)), (256, 256))
        return tf.nn.softmax(logits, axis=-1)


def train(args):
    import tensorflow as tf
    import transformers

    train_files = read_manifest(args.train_list, args.data)
    validation_files = read_manifest(args.validation_list, args.data)
    check_split(train_files, validation_files, args.allow_shared_dates)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build_segformer(args.checkpoint, args.random_init)

    def samples(files, training):
        order = rng.permutation(len(files)) if training else range(len(files))
        for index in order:
            sample = crop_sample(read_sample(files[index]), rng if training else None)
            image, label = normalize_sample(sample, "segformer")
            yield prepare_patch(image, label)

    signature = {
        "pixel_values": tf.TensorSpec((3, 512, 512), tf.float32),
        "labels": tf.TensorSpec((512, 512), tf.int32),
    }
    training = tf.data.Dataset.from_generator(
        lambda: samples(train_files, True), output_signature=signature
    ).batch(args.batch_size)
    validation = tf.data.Dataset.from_generator(
        lambda: samples(validation_files, False), output_signature=signature
    ).batch(args.batch_size)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, files in [("train", train_files), ("validation", validation_files)]:
        (args.output / f"{name}.txt").write_text(
            "\n".join(p.relative_to(args.data.resolve()).as_posix() for p in files) + "\n",
            encoding="utf-8",
        )
    settings = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    settings.update(
        tensorflow_version=tf.__version__, transformers_version=transformers.__version__
    )
    (args.output / "run_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))
    history = model.fit(
        training,
        validation_data=validation,
        epochs=args.epochs,
        callbacks=[tf.keras.callbacks.CSVLogger(str(args.output / "history.csv"))],
    )
    model.save_pretrained(str(args.output / "model"))
    (args.output / "history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )


def predict(args):
    from transformers import TFSegformerForSemanticSegmentation

    model = TFSegformerForSemanticSegmentation.from_pretrained(str(args.run / "model"))
    files = sorted(args.data.rglob("*.npy"))
    if not files or len({p.stem for p in files}) != len(files):
        raise ValueError("Require nonempty inputs with unique filenames")
    args.output.mkdir(parents=True, exist_ok=False)
    adapter = SegformerPredictor(model)
    for path in files:
        image, _ = normalize_sample(read_sample(path), "segformer")
        result = predict_region(adapter, image)
        Image.fromarray(result).save(args.output / f"{path.stem}_prediction.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    training = modes.add_parser("train")
    training.add_argument("--train-list", type=Path, required=True)
    training.add_argument("--validation-list", type=Path, required=True)
    training.add_argument("--checkpoint", default="nvidia/mit-b0")
    training.add_argument(
        "--random-init",
        action="store_true",
        help="Offline smoke test or explicit scratch-training experiment",
    )
    training.add_argument("--seed", type=int, default=12)
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--batch-size", type=int, default=1)
    training.add_argument("--learning-rate", type=float, default=0.00006)
    training.add_argument("--allow-shared-dates", action="store_true")
    prediction = modes.add_parser("predict")
    prediction.add_argument("--run", type=Path, required=True)
    for command in (training, prediction):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "train":
        if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
            parser.error("Epochs, batch size and learning rate must be positive")
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
