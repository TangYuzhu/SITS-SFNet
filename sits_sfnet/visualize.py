"""Colorize class-index PNGs without changing their grid or class values."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

PALETTE = np.array(
    [[0, 0, 0], [8, 49, 73], [174, 174, 174], [137, 217, 222], [113, 162, 165]], dtype=np.uint8
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.input.glob("*.png"))
    if not files:
        parser.error("No PNG inputs found")
    args.output.mkdir(parents=True, exist_ok=False)
    for path in files:
        with Image.open(path) as image:
            label = np.asarray(image)
        if label.ndim != 2 or not np.isin(label, range(5)).all():
            raise ValueError(f"{path}: expected single-channel indices 0..4")
        Image.fromarray(PALETTE[label.astype(int)]).save(args.output / f"{path.stem}_color.png")


if __name__ == "__main__":
    main()
