"""Build three-frame samples from already prepared, aligned 16-channel ROI arrays."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def index_frames(root):
    frames = {}
    for path in sorted(root.rglob("*.npy")):
        if len(path.stem) != 12 or not path.stem.isdigit():
            continue
        datetime.strptime(path.stem, "%Y%m%d%H%M")
        if path.stem in frames:
            raise ValueError(f"Duplicate timestamp: {path.stem}")
        frames[path.stem] = path
    return frames


def build_sample(paths, label_path, cloud_path, size):
    frames = [np.load(p, allow_pickle=False) for p in paths]
    for path, frame in zip(paths, frames):
        if frame.shape != (size, size, 16):
            raise ValueError(
                f"{path.name}: expected {(size, size, 16)}, got {frame.shape}; no implicit resampling"
            )
        if not np.issubdtype(frame.dtype, np.integer) or frame.min() < 0 or frame.max() > 65535:
            raise ValueError(
                f"{path.name}: requires integer values in [0, 65535]; resolve calibration/fill values first"
            )
    with Image.open(label_path) as image:
        label = np.asarray(image.resize((size, size), Image.Resampling.NEAREST))
    with Image.open(cloud_path) as image:
        cloud = np.asarray(image)
    if label.ndim != 2 or not np.isin(label, [0, 1, 2, 3, 4]).all():
        raise ValueError("Label must be a single-channel class-index image containing 0..4")
    if cloud.shape != (size, size) or not np.isin(cloud, [0, 1, 2]).all():
        raise ValueError("Cloud prediction must be aligned and contain class indices 0..2")
    output = np.empty((size, size, 54), dtype=np.uint16)
    for i, frame in enumerate(frames):
        output[..., i * 16 : (i + 1) * 16] = frame
    output[..., 48:51] = np.eye(3, dtype=np.uint16)[cloud.astype(int)]
    # Channel index 2 is used exactly as in the original script; band order must be confirmed upstream.
    current, previous, earliest = [f[..., 2].astype(np.float32) for f in frames]
    flow1 = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 5, 7, 1.5, 0)
    flow2 = cv2.calcOpticalFlowFarneback(earliest, previous, None, 0.5, 3, 15, 5, 7, 1.5, 0)
    flow = (flow1 + flow2) / 2
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = (np.clip(magnitude, 0, 30) / 30 * 255).astype(np.uint16)
    # Preserve legacy behavior: duplicated magnitude, NOT x/y flow components.
    output[..., 51] = magnitude
    output[..., 52] = magnitude
    output[..., 53] = label
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frames", "labels", "cloud", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--size", type=int, default=600)
    args = parser.parse_args()
    for root in (args.frames, args.labels, args.cloud):
        if not root.is_dir():
            parser.error(f"Input directory does not exist: {root}")
    if args.size <= 0:
        parser.error("--size must be positive")
    frames = index_frames(args.frames)
    labels = sorted(args.labels.glob("*.png"))
    if not labels:
        parser.error("No label PNG files found")
    # A new directory prevents accidental replacement of prior results.
    args.output.mkdir(parents=True, exist_ok=False)
    report = []
    failed = False
    seen = set()
    for label in labels:
        row = {"label": label.name}
        try:
            stamp = label.name[:12]
            target = datetime.strptime(stamp, "%Y%m%d%H%M")
            if stamp in seen:
                raise ValueError(f"Duplicate label timestamp: {stamp}")
            seen.add(stamp)
            stamps = [(target - timedelta(minutes=i * 10)).strftime("%Y%m%d%H%M") for i in range(3)]
            missing = [s for s in stamps if s not in frames]
            if missing:
                row.update(status="skipped_missing_frames", missing=missing)
            else:
                cloud = args.cloud / (label.stem + "_predict.png")
                sample = build_sample([frames[s] for s in stamps], label, cloud, args.size)
                np.save(args.output / (stamp + ".npy"), sample, allow_pickle=False)
                row.update(status="saved", timestamp=stamp)
        except (ValueError, OSError) as exc:
            row.update(status="error", reason=str(exc))
            failed = True
        report.append(row)
    (args.output / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                status: sum(r["status"] == status for r in report)
                for status in ("saved", "skipped_missing_frames", "error")
            }
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
