# Environment and execution

## Isolated environment

The tested compatibility environment is Python 3.12.3, TensorFlow 2.16.2 and tf-keras 2.16.0 on Windows CPU. These are packaging versions, not a claim about the historical training environment. The named historical H5 files report Keras 2.10.0; their original Python/CUDA/cuDNN versions have not been recovered. See `VALIDATION.md` for executed checks. The package sets `TF_USE_LEGACY_KERAS=1` before its TensorFlow imports because historical models contain Keras 2 serialized operations. Start a fresh Python process; do not import TensorFlow first in a notebook. Do not substitute Keras 3 without a separate migration validation.

From the repository directory on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux, use `python3 -m venv .venv` and `.venv/bin/python` instead. Native Windows with TensorFlow newer than 2.10 does not support CUDA GPU execution; this Windows environment is for CPU verification/inference. For GPU training use Linux or WSL2 and install `tensorflow[and-cuda]==2.16.2` alongside the pinned requirements. GPU drivers/hardware must meet TensorFlow requirements; GPU execution has not been verified here. See the [official TensorFlow installation instructions](https://www.tensorflow.org/install/pip).

`requirements-lock-windows-py312.txt` captures the complete environment actually tested, including transitive dependencies and the formatter/linter. Use it instead of `requirements-training.txt` when reproducing this Windows verification environment; do not use this Windows-specific lock on Linux. Independent seeding of Python, NumPy and TensorFlow avoids a legacy-Keras helper incompatibility with Python 3.12. Trained H5 files are opened through h5py to support non-ASCII Windows paths.

## Inputs and splits

Training consumes prepared `(600, 600, 54)` arrays: 53 input channels and one class-index label channel. It does not consume raw DAT files or colored annotation images. Spectral values are normalized with the original fixed ranges without clipping; the two motion channels are divided by 255. The 3-channel U-Net baseline selects zero-based indices `[3, 4, 6]` from the current frame; the 16-channel baseline selects current-frame indices 0–15.

Create two UTF-8 text files containing one relative NPY path per line, relative to `--data`. Blank lines and lines beginning with `#` are ignored. Actual historical split manifests have not been found and no fabricated split is provided. Duplicate files/target timestamps across splits are rejected. Shared dates are rejected by default; `--allow-shared-dates` is available only to explicitly reproduce a documented historical split. Date separation alone does not guarantee independent weather events when one event spans multiple dates.

## Train

```text
python -m sits_sfnet.train --data data/combined --train-list data/splits/train.txt --validation-list data/splits/validation.txt --variant sits_sfnet --output outputs/sits_sfnet --epochs 100 --batch-size 64 --learning-rate 0.001
```

Variants: `sits_sfnet` (current Python source), `unet_3ch`, `unet_16ch`, and checkpoint-derived `base`, `base_ca`, `base_ma`, `base_sa`, `checkpoint_full`. The latter five use exact architecture JSON exported from their corresponding historical H5 files; they start with random weights. `checkpoint_full` must not be assumed identical to the latest Python model; consult the architecture audit.

The later comparison-method package also provides `deeplabv3plus` and `segnet` through this entry point, plus dedicated SegFormer and dynamic-threshold commands. See [COMPARISONS.md](COMPARISONS.md) for additional dependencies, pretrained initialization and DeepLab's logits loss.

Training writes the model architecture, final weights, copied split lists, run arguments and loss/accuracy history into a new output directory. Validation uses one fixed central crop per listed image and visits the full validation manifest. Training uses one fresh random crop per image per epoch. The author-confirmed configuration uses 256 × 256 crops, initial learning rate 0.001, sparse categorical cross-entropy, batch size 64 and 100 epochs. The shared command defaults now match these settings. A smaller batch may be selected for memory-limited checks, but differs from the reported configuration.

## Predict and colorize

```text
python -m sits_sfnet.predict --run outputs/sits_sfnet --data data/combined_test --output outputs/predictions
python -m sits_sfnet.visualize --input outputs/predictions --output outputs/color_predictions
```

Prediction uses the variant saved with the run. To inspect a trusted historical checkpoint:

```text
python -m sits_sfnet.predict --checkpoint data/checkpoints/Base/Base.h5 --variant base --data data/combined_test --output outputs/base_predictions
```

Historical checkpoint weights remain in the original research folder; they are not bundled for GitHub. Only load checkpoints from a trusted source. Full-region prediction averages overlapping 256 × 256 tile probabilities, with stride 128 and explicit coverage of right/bottom boundaries, before argmax. This deliberately replaces the old patch reconstruction and can change edge predictions. Output PNGs contain class indices 0–4; optional color images retain the same pixel dimensions and do not overlay a map.

## Changes that affect repeatability

- Explicit manifests replace unspecified filesystem-order 80/20 splits.
- Full-manifest fixed-center validation replaces one cached, randomly cropped validation batch.
- One crop per training file per epoch replaces `floor(N / 64) * 3` steps over a repeated dataset.
- Cropping now includes the final possible row/column; the historical upper bound omitted it.
- Overlap-averaged inference replaces historical center-patch reconstruction.
- Encoder calls explicitly retain `training=False`, matching the original `predict_step` usage. Encoder Dropout therefore remains inactive; enabling it is an architecture/training change.

These are documented packaging/robustness changes, not a claim to regenerate published metrics. Recover the exact historical split and training schedule before comparing a new run with reported results. Class 0 remains included in training loss/accuracy, as in the source. Reported evaluation excludes reference land pixels with `--ignore-class 0` and retains class 4 (cloud-obscured sea fog) separately.
