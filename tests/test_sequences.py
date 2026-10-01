"""Synthetic behavioral checks; run with python -m unittest discover -s tests."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_sequences.py"
spec = importlib.util.spec_from_file_location("build_sequences", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SequenceTests(unittest.TestCase):
    def test_channels_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for i in range(3):
                path = root / f"{i}.npy"
                np.save(path, np.full((16, 16, 16), 10 + i, np.int16))
                paths.append(path)
            label = root / "label.png"
            cloud = root / "cloud.png"
            Image.fromarray(np.full((8, 8), 4, np.uint8)).save(label)
            Image.fromarray(np.full((16, 16), 2, np.uint8)).save(cloud)
            result = module.build_sample(paths, label, cloud, 16)
            self.assertEqual(result.shape, (16, 16, 54))
            self.assertEqual(result.dtype, np.uint16)
            for i in range(3):
                self.assertTrue((result[..., i * 16 : (i + 1) * 16] == 10 + i).all())
            self.assertTrue((result[..., 48:50] == 0).all())
            self.assertTrue((result[..., 50] == 1).all())
            self.assertTrue((result[..., 51:53] == 0).all())
            self.assertTrue((result[..., 53] == 4).all())
            with self.assertRaises(ValueError):
                module.build_sample(paths, label, cloud, 32)
            np.save(paths[0], np.full((16, 16, 16), -1, np.int16))
            with self.assertRaises(ValueError):
                module.build_sample(paths, label, cloud, 16)

    def test_missing_frame_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("frames", "labels", "cloud"):
                (root / name).mkdir()
            Image.fromarray(np.zeros((16, 16), np.uint8)).save(root / "labels" / "202012280320.png")
            command = [sys.executable, str(SCRIPT)]
            for name in ("frames", "labels", "cloud", "output"):
                command.extend(["--" + name, str(root / name)])
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            report = json.loads((root / "output" / "build_report.json").read_text())
            self.assertEqual(report[0]["status"], "skipped_missing_frames")
            self.assertEqual(len(report[0]["missing"]), 3)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)


if __name__ == "__main__":
    unittest.main()
