"""Source-defined baselines and checkpoint-derived ablation architectures."""

import json
from pathlib import Path

VARIANTS = (
    "sits_sfnet",
    "unet_3ch",
    "unet_16ch",
    "base",
    "base_ca",
    "base_ma",
    "base_sa",
    "checkpoint_full",
    "deeplabv3plus",
    "segnet",
)


def build_model(variant, backbone_weights="imagenet"):
    import tensorflow as tf
    from .models import build_sits_sfnet, build_unet

    if variant == "sits_sfnet":
        return build_sits_sfnet((256, 256, 53), 5)
    if variant in ("unet_3ch", "unet_16ch"):
        return build_unet((256, 256, 3 if variant == "unet_3ch" else 16), 5)
    if variant == "deeplabv3plus":
        from .deeplab import build_deeplabv3plus

        return build_deeplabv3plus(256, 5, backbone_weights)
    if variant == "segnet":
        from .segnet import build_segnet

        return build_segnet((256, 256, 3), 5)
    path = Path(__file__).resolve().parents[1] / "configs" / "architectures" / f"{variant}.json"
    if variant not in VARIANTS or not path.is_file():
        raise ValueError(f"Unknown or unavailable variant: {variant}")
    # Only repository-owned configurations exported from the local research checkpoints.
    return tf.keras.models.model_from_json(json.dumps(json.loads(path.read_text(encoding="utf-8"))))
