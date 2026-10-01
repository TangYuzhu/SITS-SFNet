"""Historical dynamic-threshold baseline with explicit calibration and ocean mask."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

BANDS = (3, 4, 5, 7, 11, 13, 15)
ASSETS = Path(__file__).resolve().parents[1] / "assets" / "dynamic_threshold"


def load_calibration(directory):
    tables = {}
    for band in BANDS:
        table = np.loadtxt(directory / f"band{band}.csv", delimiter=",", ndmin=2)[:, 0]
        tables[band] = table
    return tables


def classify(image, tables, ocean_mask):
    if image.ndim != 3 or image.shape[-1] < 16 or not np.issubdtype(image.dtype, np.integer):
        raise ValueError("Expected integer lookup-table indices with at least 16 channels")
    if ocean_mask.shape != image.shape[:2]:
        raise ValueError("Ocean mask and image must have the same spatial shape")
    calibrated = {}
    for band in BANDS:
        indices = image[..., band - 1].astype(np.int64)
        if band in (3, 4):
            indices = np.minimum(indices, 2047)
        if indices.min() < 0 or indices.max() >= len(tables[band]):
            raise ValueError(f"Band {band}: values outside calibration lookup range")
        calibrated[band] = tables[band][indices]
        if not np.isfinite(calibrated[band]).all():
            raise ValueError(f"Band {band}: selected lookup entries contain nonfinite fill values")
    reflectance_red, reflectance_nir, reflectance_swir = [calibrated[b] for b in (3, 4, 5)]
    temperature_short, temperature_86, temperature_108, temperature_123 = [
        calibrated[b] for b in (7, 11, 13, 15)
    ]
    difference = (temperature_108 - temperature_short).astype(np.int64)
    histogram, edges = np.histogram(difference, bins=20, range=(-20, 0))
    threshold_bin = len(histogram) - 1
    for index in range(len(histogram) - 2, 0, -1):
        if histogram[index] < histogram[index + 1] and histogram[index] < histogram[index - 1]:
            threshold_bin = index
            break
    threshold = float((edges[threshold_bin] + edges[threshold_bin + 1]) / 2)
    prediction = np.zeros(image.shape[:2], np.uint8)
    prediction[difference > threshold] = 1
    # Preserve legacy IEEE division behavior, including NaN for 0/0.
    with np.errstate(divide="ignore", invalid="ignore"):
        snow_index = (reflectance_red - reflectance_swir) / (reflectance_red + reflectance_swir)
    snow = ~((reflectance_nir > 0.11) | (temperature_108 > 256) | (snow_index < 0.4))
    prediction[snow] = 1
    phase = np.zeros(image.shape[:2], np.uint8)
    phase[temperature_123 - temperature_86 > 2.5] = 1
    phase[temperature_108 < 250] = 2
    phase[temperature_86 - temperature_108 > 0] = 2
    phase[(snow_index < 0.1) & (phase == 0)] = 1
    phase[phase == 0] = 2
    prediction[(prediction == 0) & (phase == 1)] = 2
    prediction[(prediction == 0) & (phase == 2)] = 3
    prediction[~ocean_mask.astype(bool)] = 0
    return prediction, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration", type=Path, default=ASSETS)
    parser.add_argument("--mask", type=Path, default=ASSETS / "mask.png")
    args = parser.parse_args()
    tables = load_calibration(args.calibration)
    files = sorted(args.data.rglob("*.npy"))
    if not files or len({p.stem for p in files}) != len(files):
        parser.error("Require nonempty inputs with unique stems")
    args.output.mkdir(parents=True, exist_ok=False)
    report = []
    for path in files:
        image = np.load(path, allow_pickle=False)
        with Image.open(args.mask) as mask_image:
            mask = np.asarray(mask_image)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (
            np.asarray(
                Image.fromarray(mask).resize(
                    (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
                )
            )
            > 0
        )
        prediction, threshold = classify(image, tables, mask)
        Image.fromarray(prediction).save(args.output / f"{path.stem}_prediction.png")
        report.append({"sample": path.name, "dynamic_threshold": threshold})
    (args.output / "thresholds.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
