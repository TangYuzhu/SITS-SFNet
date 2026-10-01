"""Export a date inventory without reading or changing array contents."""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from build_sequences import index_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.frames.is_dir():
        parser.error("Frame directory does not exist")
    frames = index_frames(args.frames)
    if not frames:
        parser.error("No timestamped NPY files found")
    slots = [
        (datetime(2000, 1, 1, 3) + timedelta(minutes=10 * i)).strftime("%H%M") for i in range(15)
    ]
    rows = []
    for date in sorted({s[:8] for s in frames}):
        stamps = sorted(s for s in frames if s.startswith(date))
        windows = sum(
            all(
                (datetime.strptime(s, "%Y%m%d%H%M") - timedelta(minutes=10 * i)).strftime(
                    "%Y%m%d%H%M"
                )
                in frames
                for i in (1, 2)
            )
            for s in stamps
        )
        rows.append(
            dict(
                date=date,
                frames=len(stamps),
                three_frame_windows=windows,
                missing_times=";".join(t for t in slots if date + t not in frames),
                extra_times=";".join(s[8:] for s in stamps if s[8:] not in slots),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{len(rows)} dates, {len(frames)} frames, {sum(r['three_frame_windows'] for r in rows)} candidate windows"
    )


if __name__ == "__main__":
    main()
