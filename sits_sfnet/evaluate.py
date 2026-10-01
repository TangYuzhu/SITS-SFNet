"""Evaluate explicitly paired class-index PNGs, without implicit splitting or resizing."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def read_labels(path):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2 or not np.isin(array, range(5)).all():
        raise ValueError(f"{path}: expected single-channel class indices 0..4")
    return array


def summarize(matrix, ignored_classes=()):
    diagonal = np.diag(matrix).astype(float)
    truth_count = matrix.sum(axis=1)
    prediction_count = matrix.sum(axis=0)

    def ratio(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator != 0,
        )

    iou = ratio(diagonal, truth_count + prediction_count - diagonal)

    def serial(values):
        return [float(value) if np.isfinite(value) else None for value in values]

    scored_iou = iou.copy()
    scored_iou[list(ignored_classes)] = np.nan
    total = matrix.sum()
    return {
        "confusion_matrix_rows_truth": matrix.tolist(),
        "precision": serial(ratio(diagonal, prediction_count)),
        "recall": serial(ratio(diagonal, truth_count)),
        "iou": serial(iou),
        "mean_iou_present_classes": float(np.nanmean(scored_iou))
        if np.isfinite(scored_iou).any()
        else None,
        "pixel_accuracy": float(diagonal.sum() / total) if total else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--list", type=Path, required=True, help="One target filename stem per line"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-suffix", default=".png")
    parser.add_argument("--ignore-class", type=int, choices=range(5), action="append", default=[0])
    parser.add_argument("--merge-overlap-into-fog", action="store_true")
    args = parser.parse_args()
    stems = [
        line.strip()
        for line in args.list.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not stems or len(set(stems)) != len(stems) or any(Path(s).name != s for s in stems):
        parser.error("Require unique filename stems without directory components")
    matrix = np.zeros((5, 5), np.int64)
    for stem in stems:
        predicted = read_labels(args.predictions / f"{stem}_prediction.png").copy()
        truth = read_labels(args.labels / f"{stem}{args.label_suffix}").copy()
        if predicted.shape != truth.shape:
            raise ValueError(f"{stem}: prediction and label shapes differ")
        valid = ~np.isin(truth, args.ignore_class)
        if args.merge_overlap_into_fog:
            predicted[predicted == 4] = 2
            truth[truth == 4] = 2
        matrix += np.bincount(
            (truth[valid].astype(int) * 5 + predicted[valid]).ravel(), minlength=25
        ).reshape(5, 5)
    result = summarize(matrix, args.ignore_class)
    result.update(
        samples=len(stems),
        ignored_truth_classes=args.ignore_class,
        merge_overlap_into_fog=args.merge_overlap_into_fog,
        evaluated_stems=stems,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
