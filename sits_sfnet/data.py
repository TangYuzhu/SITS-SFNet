"""Validated sample loading, historical normalization, and explicit data splits."""

from pathlib import Path
import numpy as np

THRESHOLDS = np.array(
    [
        (200, 1200),
        (150, 1000),
        (50, 1000),
        (0, 1100),
        (0, 1000),
        (0, 700),
        (15300, 16300),
        (1700, 1950),
        (1500, 1900),
        (2800, 3800),
        (1900, 3600),
        (2700, 3500),
        (2000, 3000),
        (1500, 3200),
        (1500, 3200),
        (1000, 1600),
    ],
    dtype=np.float32,
)


def read_sample(path):
    sample = np.load(path, allow_pickle=False)
    if sample.shape != (600, 600, 54):
        raise ValueError(f"{path}: expected (600, 600, 54), got {sample.shape}")
    if not np.isfinite(sample).all():
        raise ValueError(f"{path}: contains nonfinite values")
    label = sample[..., -1]
    if not np.isin(label, np.arange(5)).all():
        raise ValueError(f"{path}: label must contain integer class indices 0..4")
    return sample


def normalize_sample(sample, variant):
    image = sample[..., :53].astype(np.float32)
    low = np.tile(THRESHOLDS[:, 0], 3)
    width = np.tile(THRESHOLDS[:, 1] - THRESHOLDS[:, 0], 3)
    image[..., :48] = (image[..., :48] - low) / width
    # Historical training did not clip spectral values to [0, 1].
    image[..., 51:53] /= 255.0
    if variant in ("unet_3ch", "deeplabv3plus", "segnet", "segformer"):
        image = image[..., [3, 4, 6]]
    elif variant == "unet_16ch":
        image = image[..., :16]
    return image, sample[..., -1].astype(np.uint8)


def read_manifest(path, data_root):
    entries = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not entries or len(entries) != len(set(entries)):
        raise ValueError("Manifest must be nonempty and contain unique relative paths")
    root = Path(data_root).resolve()
    files = []
    for entry in entries:
        file = (root / entry).resolve()
        if not file.is_relative_to(root) or not file.is_file() or file.suffix != ".npy":
            raise ValueError(f"Invalid manifest entry: {entry}")
        files.append(file)
    return files


def check_split(train, validation, allow_shared_dates=False):
    if set(train) & set(validation):
        raise ValueError("Training and validation manifests share files")
    if {p.stem[:12] for p in train} & {p.stem[:12] for p in validation}:
        raise ValueError("Training and validation share target timestamps")
    shared = {p.stem[:8] for p in train} & {p.stem[:8] for p in validation}
    if shared and not allow_shared_dates:
        raise ValueError(
            f"Shared observation dates: {sorted(shared)}. Use explicit override only for a documented historical split."
        )


def crop_sample(sample, rng=None):
    if rng is None:
        top = left = (600 - 256) // 2
    else:
        top, left = rng.integers(0, 600 - 256 + 1, size=2)
    return sample[top : top + 256, left : left + 256]
