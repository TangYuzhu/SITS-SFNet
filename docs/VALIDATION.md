# Validation record

The released code has been tested on Windows CPU with Python 3.12.3, TensorFlow 2.16.2, tf-keras 2.16.0, NumPy 1.26.4, h5py 3.11.0, Pillow 10.4.0, and OpenCV 4.10.0. The complete tested dependency snapshot is provided in `requirements-lock-windows-py312.txt`.

## Core checks

The validation suite covers:

- three-frame assembly and channel placement;
- dimension, class-index, and numeric-range validation;
- missing-frame handling;
- normalization and channel selection;
- aligned cropping and split guards;
- full-scene tiled prediction;
- model construction and serialization;
- SITS-SFNet optimizer execution;
- SegNet pooling/unpooling;
- DeepLabV3+ forward/training behavior;
- SegFormer training/prediction I/O;
- dynamic-threshold processing;
- evaluation statistics.

The primary SITS-SFNet implementation builds successfully, produces finite five-class outputs, supports weight save/load, and completes synthetic training and prediction runs. The released comparison methods have likewise been checked for executable forward, training, serialization, or prediction workflows as appropriate.

## Repeat the checks

```text
python -m unittest discover -s tests -v
python tests/smoke_model.py --variant sits_sfnet --train-step
```

For comparison methods, install `requirements-comparisons.txt` and follow [COMPARISONS.md](COMPARISONS.md).

## Scope of validation

These tests verify implementation behavior, tensor shapes, serialization, input validation, and command-line workflows. They are software validation checks rather than a substitute for reproducing scientific accuracy metrics on the complete research dataset. Reproduction of the paper's reported results requires the same data preprocessing, experiment split, and training configuration described in the manuscript.
