# Comparison experiments

The six supplied comparison folders have been consolidated into the existing package. Original files are retained in the research workspace. New source and comments are English; hard-coded paths, obsolete private APIs, dead code, duplicate model definitions and interactive plotting were removed. Model input selection remains method-specific.

| Research folder | Release implementation / variant | Input |
| --- | --- | --- |
| `DeeplabV3Plus` | `sits_sfnet/deeplab.py`, `deeplabv3plus` | Current-frame channels 4, 5, 7 |
| `SegNet` | `sits_sfnet/segnet.py`, `pooling.py`, `segnet` | Current-frame channels 4, 5, 7 |
| `SegFormer` | `sits_sfnet/segformer.py` | Current-frame channels 4, 5, 7 |
| `U_Net_3Chs` | Shared `models.py`, `unet_3ch` | Current-frame channels 4, 5, 7 |
| `U_Net_16Chs` | Shared `models.py`, `unet_16ch` | All 16 current-frame channels |
| `Threshold_Dynamic` | `sits_sfnet/dynamic_threshold.py` | Current-frame channels 3, 4, 5, 7, 11, 13, 15 and external ocean mask |

Channel numbers in this table are one-based; the selected three-channel array indices are `[3, 4, 6]`. These are spectral inputs, not ordinary RGB photographs. The U-Net model functions in the two supplied comparison folders are identical to the previously consolidated U-Net definition. Redundant copies are therefore not retained. The standalone historical U-Net prediction files still selected all 53 channels and used a SITS model filename; the shared predictor now selects channels according to the actual trained variant.

## Environment

Use the Python 3.12 environment in [RUNNING.md](RUNNING.md), then install:

```text
python -m pip install -r requirements-comparisons.txt
```

This adds `transformers==4.44.2` to the pinned TensorFlow 2.16.2 / tf-keras 2.16.0 environment. TensorFlow/Keras 3 migration is not included. The exact expanded Windows environment is captured in `requirements-lock-comparisons-windows-py312.txt`. GPU support follows the Linux/WSL2 instructions in RUNNING.md; GPU performance has not been validated.

DeepLab uses a ResNet50 ImageNet backbone by default. Its first run needs network access or a cached/local backbone weights file. Use `--backbone-weights path/to/resnet50_weights.h5` for local weights, or `--backbone-weights none` for explicit scratch initialization and offline checks. SegFormer uses the historical `nvidia/mit-b0` checkpoint, which provides TensorFlow weights; `--checkpoint` also accepts a local model directory. `--random-init` bypasses downloads and creates a random MiT-B0 model; it is not equivalent to pretrained training. No pretrained weights are redistributed in this package. Provider terms apply to downloaded pretrained models. If required, configure writable `KERAS_HOME` and `HF_HOME` directories before starting Python.

## DeepLab, SegNet and U-Net training/prediction

Use the same explicit split files as for SITS-SFNet. For each run, replace the variant and use a distinct output directory:

```text
python -m sits_sfnet.train --variant deeplabv3plus --data data/combined --train-list data/splits/train.txt --validation-list data/splits/validation.txt --output outputs/deeplabv3plus
python -m sits_sfnet.train --variant segnet --data data/combined --train-list data/splits/train.txt --validation-list data/splits/validation.txt --output outputs/segnet
python -m sits_sfnet.train --variant unet_3ch --data data/combined --train-list data/splits/train.txt --validation-list data/splits/validation.txt --output outputs/unet_3ch
python -m sits_sfnet.train --variant unet_16ch --data data/combined --train-list data/splits/train.txt --validation-list data/splits/validation.txt --output outputs/unet_16ch
python -m sits_sfnet.predict --run outputs/deeplabv3plus --data data/combined_test --output outputs/deeplab_predictions
```

All consume the existing 600 × 600 × 54 combined arrays for consistent sample alignment, although only current-frame channels are used by these methods. Training uses 256 × 256 crops and class-index targets. First-frame normalization retains the original fixed thresholds without clipping or adding ResNet-specific preprocessing. Normalization of unused historical-frame channels has no effect on these inputs. Adam defaults to 0.001. Splits, crop schedules, full-manifest validation and tiled probability averaging follow the documented shared pipeline rather than the unreliable historical filesystem-order split.

## SegFormer training/prediction

```text
python -m sits_sfnet.segformer train --data data/combined --train-list data/splits/train.txt --validation-list data/splits/validation.txt --output outputs/segformer --checkpoint nvidia/mit-b0
python -m sits_sfnet.segformer predict --run outputs/segformer --data data/combined_test --output outputs/segformer_predictions
```

The original 256 × 256 crop is enlarged to 512 × 512, normalized using means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`, and transposed to channels-first. Labels are enlarged with **nearest-neighbor** interpolation and cast to integer indices. The model emits 128 × 128 logits, which are resized to 256 × 256 for tiled prediction. Defaults: Adam 0.00006, 100 epochs, batch size 1. Trained models are saved with `save_pretrained`, together with manifests, arguments and histories; the source script did not actually save the trained SegFormer.

## Dynamic-threshold baseline

```text
python -m sits_sfnet.dynamic_threshold --data data/current_frame_indices --output outputs/dynamic_threshold
```

The supplied seven calibration CSV files and `mask.png` are retained byte-for-byte in `assets/dynamic_threshold`. Their detailed acquisition/version/redistribution provenance still needs author confirmation. The mask has 1440 × 1440 pixels and is resized by nearest neighbor to match the input grid. This presumes the same regional extent/orientation; it performs no geolocation alignment. Positive mask values indicate ocean/valid pixels; zero indicates excluded land. Inputs must be integer lookup-table indices, shaped H × W × 16 or a compatible combined array with the current frame in the first 16 channels. **Do not supply physical reflectance/temperature arrays or already normalized floating-point inputs to this lookup-based algorithm.**

The first calibration-table column is used, as in the source. B03/B04 values above 2047 are capped without modifying the caller's array. Other out-of-range or negative indices are rejected. Some supplied infrared tables contain NaN fill entries; a selected NaN entry produces an explicit error rather than an undocumented classification. This is validation, not an inferred fill-value correction.

The implemented sequence is clear-sky exclusion using the rightmost histogram minimum of the integer-truncated temperature difference; snow exclusion; then water/ice-cloud discrimination. Threshold inequalities and the full-scene histogram (including land before masking) are retained. The original later droplet/spatial/microphysical stages are only TODO comments and are **not implemented**. The output has indices 0–3 and never predicts the overlap class 4. A per-image threshold report is saved.

## Color output and explicit evaluation

```text
python -m sits_sfnet.visualize --input outputs/deeplab_predictions --output outputs/deeplab_colors
python -m sits_sfnet.evaluate --predictions outputs/deeplab_predictions --labels data/class_labels --list data/splits/test_stems.txt --output outputs/deeplab_metrics.json --ignore-class 0
```

The evaluation list contains one filename stem per line, such as `202012280320`, without extension. Reference filenames default to `<stem>.png`; use `--label-suffix` if needed. Evaluation requires class-index masks, not Photoshop color images or rendered maps, and never reconstructs labels from colors. It checks exact dimensions and evaluates the entire supplied list. It reports a five-class confusion matrix, precision, recall, IoU and pixel accuracy. Absent classes have JSON null scores; mean IoU excludes absent classes. Use `--ignore-class 0` to exclude reference land pixels, as required by this study (also the evaluator default). Class 4 denotes cloud-obscured sea fog and remains separate; do not use the optional `--merge-overlap-into-fog` policy for this protocol. Ignored classes are omitted from mean IoU, while predictions of an ignored class on valid ocean pixels still count as errors. This replaces the old `run.py` shell command and implicit evaluation slicing, not a claim to reproduce historical metric conventions.

## Corrections affecting results

1. **DeepLab loss:** its final convolution produces logits, while the historical loss assumed probabilities. Training now uses `SparseCategoricalCrossentropy(from_logits=True)`; prediction converts logits to probabilities before tile averaging. See the [Keras DeepLab example](https://keras.io/examples/vision/deeplabv3_plus/). This corrects training semantics and may change results.
2. **SegFormer labels:** the original bilinear target resize could create fractional class IDs. Nearest-neighbor interpolation now preserves categories. Input-image resizing remains bilinear. The supplied source uses [Hugging Face TF SegFormer](https://huggingface.co/docs/transformers/v4.44.2/model_doc/segformer).
3. **SegNet layers:** replaced removed `K.tf` calls and standalone-Keras mixing with public TensorFlow operations; argmax indices stay integer, and layer configurations are serializable. Max-pooling geometry and the model's convolution/decoder layout are retained.
4. **Threshold implementation:** removed deprecated `np.int`, destructive result-folder deletion and input-array mutation. Explicit mask and calibration validation replaces hidden reliance on label files. The older `threshold_test.py` differs in B03/B04 clipping and contains an unused resize return value; `threshold_test_from_files.py` is the selected canonical algorithm.

These corrections, plus the shared training/inference changes, require rerunning experiments before updating published numbers. They are not presented as reproductions of the reported comparison results.
