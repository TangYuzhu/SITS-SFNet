# Environment and execution

## Environment

The released code has been tested with Python 3.12.3, TensorFlow 2.16.2, tf-keras 2.16.0, NumPy 1.26.4, h5py 3.11.0, Pillow 10.4.0, and OpenCV 4.10.0 on Windows CPU.

From the repository directory on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux, use `python3 -m venv .venv` and `.venv/bin/python`. For GPU training, use a TensorFlow-supported Linux or WSL2 environment and install the corresponding CUDA-enabled TensorFlow package.

For an exact copy of the tested Windows environment, use `requirements-lock-windows-py312.txt`.

## Inputs and splits

Training consumes prepared `(600, 600, 54)` arrays: 53 input channels and one class-index label channel. The 3-channel U-Net baseline selects zero-based indices `[3, 4, 6]` from the current frame, while the 16-channel baseline uses current-frame indices 0–15.

Create UTF-8 text files containing one relative NPY path per line, relative to `--data`. Blank lines and lines beginning with `#` are ignored. The repository does not prescribe a fixed split file; users should construct train/validation/test manifests according to the experiment protocol described in the paper. Duplicate target timestamps across splits are rejected.

## Train

```text
python -m sits_sfnet.train --data data/combined --train-list data/splits/train.txt --validation-list data/splits/validation.txt --variant sits_sfnet --output outputs/sits_sfnet --epochs 100 --batch-size 64 --learning-rate 0.001
```

Available variants include `sits_sfnet`, `unet_3ch`, `unet_16ch`, `base`, `base_ca`, `base_ma`, `base_sa`, and `checkpoint_full`. The primary model used for the released SITS-SFNet implementation is `sits_sfnet`.

Training writes the model architecture, final weights, copied split lists, run arguments, and loss/accuracy history into a new output directory. The default study configuration uses 256 × 256 crops, an initial learning rate of 0.001, sparse categorical cross-entropy, batch size 64, and 100 epochs.

Comparison-method training is documented separately in [COMPARISONS.md](COMPARISONS.md).

## Predict and colorize

```text
python -m sits_sfnet.predict --run outputs/sits_sfnet --data data/combined_test --output outputs/predictions
python -m sits_sfnet.visualize --input outputs/predictions --output outputs/color_predictions
```

Full-region prediction combines overlapping 256 × 256 tiles with stride 128 and covers the full 600 × 600 scene. Output PNGs contain class indices 0–4; optional color images use the same dimensions.

## Evaluate

```text
python -m sits_sfnet.evaluate --predictions outputs/predictions --labels data/class_labels --list data/splits/test_stems.txt --output outputs/metrics.json --ignore-class 0
```

Reference land pixels are excluded from the reported ocean-domain metrics with `--ignore-class 0`. Class 4 (cloud-obscured sea fog) remains a separate class.

## Reproducibility notes

Use the same data preprocessing, split protocol, random-seed policy, and experiment configuration as the paper when reproducing reported results. The released validation tests verify software behavior and I/O consistency; scientific metrics depend on the exact experiment data and split.
