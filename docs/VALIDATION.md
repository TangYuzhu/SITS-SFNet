# Validation record

Executed on 2026-09-29 on Windows CPU with Python 3.12.3, TensorFlow 2.16.2, tf-keras 2.16.0, NumPy 1.26.4, h5py 3.11.0, Pillow 10.4.0 and OpenCV 4.10.0. The complete tested dependency snapshot is `requirements-lock-windows-py312.txt`. Ruff 0.11.13 was used for linting and formatting.

## Results

- Five unit tests passed: three-frame assembly/channel placement, input rejection and missing-frame handling, normalization/channel selection, aligned cropping and split guards, and full boundary coverage in tiled prediction.
- All eight release variants built, produced finite `(1, 256, 256, 5)` probabilities summing to one, and survived JSON architecture plus HDF5 weight save/load with matching outputs.
- The source-defined SITS-SFNet completed one synthetic optimizer update with finite loss.
- A complete command-line SITS-SFNet training run completed one epoch using one synthetic training scene and one separate-date validation scene. It saved weights, architecture, manifests and history. Its saved run then produced two 600 × 600 class-index PNGs; colorization produced matching 600 × 600 RGB PNGs.
- All seven historical H5 model files loaded with their existing weights and produced finite, normalized five-class outputs. See `metadata/checkpoint_validation.json`.
- The cleaned full model was compared directly with the original source functions, using identical weights and seeded nonconstant input. Both have 11,799,725 parameters; maximum absolute output difference was **0.0**. See `metadata/source_model_parity.json`.
- Python lint and formatting checks passed. No Chinese comments/docstrings remain in the packaged Python files.

| Variant | Parameters | Forward and serialization |
| --- | ---: | --- |
| `sits_sfnet` | 11,799,725 | Passed |
| `unet_3ch` | 31,034,565 | Passed |
| `unet_16ch` | 31,042,053 | Passed |
| `base` | 11,777,829 | Passed |
| `base_ca` | 11,778,125 | Passed |
| `base_ma` | 11,782,149 | Passed |
| `base_sa` | 11,790,789 | Passed |
| `checkpoint_full` | 11,790,789 | Passed |

## Repeat the checks

```text
python -m unittest discover -s tests -v
python tests/smoke_model.py --variant sits_sfnet --train-step
python tests/smoke_model.py --variant base
```

Repeat the last command for the other variants when their architecture or environment changes. `ruff check .` and `ruff format --check .` check source style. TensorFlow's legacy Keras emits upstream deprecation warnings in this compatibility environment; these did not prevent the tested workflows from completing.

## Comparison-method extension, 2026-09-29

Transformers 4.44.2 was added to the same isolated environment. Its full dependency snapshot is `requirements-lock-comparisons-windows-py312.txt`. The unit suite now contains **nine passing tests**, including batch-safe SegNet pooling/unpooling gradients, dynamic-threshold input immutability and evaluator statistics.

- DeepLabV3+ (11,853,381 parameters) passed finite-logit forward execution, JSON/weights round-trip and one optimizer update with `from_logits=True`. The smoke test used random backbone initialization; ImageNet download and pretrained fine-tuning were not executed.
- SegNet (29,459,205 parameters) passed probability normalization, JSON/weights round-trip and one optimizer update using the new serializable pooling layers.
- SegFormer B0 (3,715,941 parameters) passed 512-pixel input / 128-pixel logits checks, categorical nearest-neighbor label resizing, `save_pretrained`/reload parity and one optimizer update. An additional end-to-end command-line run trained for one epoch with one synthetic training and one validation scene and saved the model successfully. These tests used explicit random initialization, not pretrained weights.
- Both U-Net definitions were verified by AST comparison to be identical to the already tested consolidated implementation; their input normalization/channel choices remain the same.
- The threshold algorithm matched the original `threshold_test_from_files.py` on all 4,096 pixels of a seeded synthetic input, including B03/B04 upper clipping. See `metadata/comparison_threshold_parity.json`. Its CLI produced class PNGs and threshold metadata for two synthetic scenes, and the explicit evaluator completed on them.
- The six comparison source folders and retained calibration/mask assets are recorded with hashes in `metadata/comparison_sources.json`.

The saved synthetic SegFormer run also completed the command-line tiled predictor on two full scenes, producing 600 × 600 class-index PNGs. All validation artifacts remain outside the upload package. No synthetic weights or synthetic predictions are presented as research results.

## Limits (all methods)

Synthetic runs test software behavior, not sea-fog segmentation quality. No published accuracy metric was reproduced, no GPU run was performed, and no full real-data training was launched. Actual historical split files and prepared class-index labels still need to be supplied. The checkpoint naming discrepancy in `CODE_MIGRATION.md` remains unresolved. The refactored training schedule and inference blending are documented behavioral changes; source-model forward parity does not establish parity of a new end-to-end training run.

## Author-confirmed release updates, 2026-09-30

README now describes the accompanying AHI/ground_truth inventories, temporal correspondence, class palette and author-reported training settings. Shared training defaults to batch size 64. Evaluation excludes reference land by default, omits ignored classes from mean IoU, and retains errors where valid ocean pixels are predicted as land. Class 4 is cloud-obscured sea fog and remains separate. All ten unit tests and Ruff checks passed after these changes. No real-data training or data upload was performed.

## Optical-flow utilities, 2026-10-01

The two original exploratory scripts were consolidated into a standalone CLI. Both pair-image and stored-motion commands passed synthetic end-to-end checks. For a seeded image translated 3 pixels right and 2 down, median interior displacement was (2.99988, 1.99992). The flow array matched the original Farneback call exactly. Saved stored-magnitude pixels were checked. Ruff passed. These checks validate implementation and direction conventions, not meteorological motion accuracy. The existing sequence builder was not changed.
