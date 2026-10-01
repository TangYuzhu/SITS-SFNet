"""Compute two-image Farneback flow or inspect stored sequence motion magnitudes."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def image_pair_flow(previous_path, current_path, size=600):
    """Retain the original blue-channel selection and linear image resizing."""
    frames = []
    for path in (previous_path, current_path):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        frames.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR))
    previous, current = frames
    # Original OpenCV BGR index 0 corresponds to RGB index 2 (blue).
    flow = cv2.calcOpticalFlowFarneback(
        previous[..., 2], current[..., 2], None, 0.5, 3, 15, 5, 7, 1.5, 0
    )
    return current, flow


def arrow_image(current, flow, step=10, scale=1.0):
    """Render displacement in image coordinates: right is +x, down is +y."""
    result = current.copy()
    height, width = result.shape[:2]
    for y in range(0, height, step):
        for x in range(0, width, step):
            dx, dy = flow[y, x] * scale
            endpoint = (int(round(x + dx)), int(round(y + dy)))
            cv2.arrowedLine(result, (x, y), endpoint, (255, 255, 255), 1, tipLength=0.25)
    return result


def stored_motion(path):
    """Read the two stored magnitudes, without interpreting them as x/y flow."""
    sample = np.load(path, allow_pickle=False)
    if sample.ndim != 3 or sample.shape[-1] != 54:
        raise ValueError("Expected an H x W x 54 combined sample")
    motion = sample[..., 51:53]
    if not np.isfinite(motion).all() or np.any(motion < 0) or np.any(motion > 255):
        raise ValueError("Stored motion values must be finite and within 0..255")
    return motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pair = commands.add_parser("pair", help="Compute flow from earlier to later image")
    pair.add_argument("--previous", type=Path, required=True)
    pair.add_argument("--current", type=Path, required=True)
    pair.add_argument("--size", type=int, default=600)
    pair.add_argument("--step", type=int, default=10)
    pair.add_argument("--arrow-scale", type=float, default=1.0)
    sample = commands.add_parser("sample", help="Inspect stored motion magnitude channels")
    sample.add_argument("--input", type=Path, required=True)
    for command in (pair, sample):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pair":
        if (
            min(args.size, args.step) <= 0
            or not np.isfinite(args.arrow_scale)
            or args.arrow_scale <= 0
        ):
            parser.error("Size, step and arrow scale must be positive and finite")
        current, flow = image_pair_flow(args.previous, args.current, args.size)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        args.output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(args.output / "flow.npz", flow=flow, magnitude=magnitude, angle=angle)
        Image.fromarray(arrow_image(current, flow, args.step, args.arrow_scale)).save(
            args.output / "flow_arrows.png"
        )
        Image.fromarray((np.clip(magnitude, 0, 30) / 30 * 255).astype(np.uint8)).save(
            args.output / "magnitude.png"
        )
        report = dict(
            previous=str(args.previous),
            current=str(args.current),
            size=args.size,
            channel="blue",
            units="pixels per supplied frame pair",
            angle_units="radians",
            arrow_scale=args.arrow_scale,
            step=args.step,
        )
    else:
        motion = stored_motion(args.input)
        args.output.mkdir(parents=True, exist_ok=False)
        for index in range(2):
            Image.fromarray(motion[..., index].astype(np.uint8)).save(
                args.output / f"motion_channel_{51 + index}.png"
            )
        report = dict(
            input=str(args.input),
            channels_equal=bool(np.array_equal(motion[..., 0], motion[..., 1])),
            meaning="Stored scaled magnitudes, not vector components",
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
