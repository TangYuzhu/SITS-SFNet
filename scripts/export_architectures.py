"""Extract architecture metadata from trusted local H5 checkpoints without weights."""

import argparse
import hashlib
import json
from pathlib import Path
import h5py

CHECKPOINTS = {
    "base": "Base/Base.h5",
    "base_ca": "Base+CA/Base+CA.h5",
    "base_ma": "Base+MA/Base+MA.h5",
    "base_sa": "Base+SA/Base+SA.h5",
    "checkpoint_full": "SITS_SFNet/SITS_SFNet.h5",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for variant, relative in CHECKPOINTS.items():
        path = args.checkpoints / relative
        with h5py.File(path, "r") as checkpoint:
            raw = checkpoint.attrs["model_config"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            config = json.loads(raw)
            metadata = {
                key: str(checkpoint.attrs.get(key, "unknown"))
                for key in ("keras_version", "backend")
            }
        (args.output / f"{variant}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        records.append(
            {"variant": variant, "source": relative, "sha256": digest.hexdigest(), **metadata}
        )
    (args.output / "provenance.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
