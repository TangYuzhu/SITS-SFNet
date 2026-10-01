# YBSF-55 Reference Annotations and SITS-SFNet Code

This repository provides the reference annotations, model code, preprocessing utilities, and evaluation tools associated with the YBSF-55 sea-fog dataset and SITS-SFNet experiments. It includes the released ground-truth annotations, three-frame sequence assembly, SITS-SFNet and baseline implementations, training and inference scripts, comparison methods, and metadata describing the observation dates.

The raw Himawari-8 observations and large intermediate AHI arrays are not redistributed here. Readers can obtain the source observations from the NICT Science Cloud and prepare the model inputs following the workflow below.

For installation, training, and inference commands, see [Running the models](docs/RUNNING.md). Validation checks for the released code are summarized in [Validation](docs/VALIDATION.md). The comparison methods used in the experiments are documented in [Comparison experiments](docs/COMPARISONS.md).

## Repository contents

- [`ground_truth.zip`](ground_truth.zip): released YBSF-55 reference annotations.
- [`metadata/observation_dates.csv`](metadata/observation_dates.csv): observation-date and frame inventory.
- [`metadata/ground_truth_alignment.csv`](metadata/ground_truth_alignment.csv): timestamp correspondence between annotations and available observation sequences.
- [`scripts/build_sequences.py`](scripts/build_sequences.py): three-frame sample assembly.
- [`sits_sfnet/`](sits_sfnet): SITS-SFNet, U-Net baselines, DeepLabV3+, SegNet, SegFormer, dynamic-threshold baseline, prediction, visualization, and evaluation code.
- [`docs/RUNNING.md`](docs/RUNNING.md): environment and execution instructions.
- [`docs/COMPARISONS.md`](docs/COMPARISONS.md): comparison-method configurations.
- [`docs/OPTICAL_FLOW.md`](docs/OPTICAL_FLOW.md): optical-flow utilities.
- [`docs/VALIDATION.md`](docs/VALIDATION.md): software validation record.

## Raw observations and study region

The source observations are Himawari-8 Advanced Himawari Imager (AHI) multispectral data in Himawari Standard Data (HSD) format. The study region covers **30°N–42°N and 117°E–129°E**. YBSF-55 contains **55 observation dates from 2016 to 2020**. Consecutive observation days may belong to the same sea-fog episode; the dataset name refers to observation dates rather than a count of independent meteorological events.

The source Himawari-8 observations were obtained through the **Science Cloud of the National Institute of Information and Communications Technology (NICT), Japan**, with the satellite data provided by the Japan Meteorological Agency (JMA). The current observation-date inventory is listed in [`metadata/observation_dates.csv`](metadata/observation_dates.csv).

Source data service:

- [NICT Science Cloud](https://sc-nc-web.nict.go.jp/)
- [JMA Himawari Standard Data User's Guide](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/hsd_sample/HS_D_users_guide_en_v13.pdf)

For each selected date, the study uses all 16 AHI bands and nominal observation times from **03:00 to 05:20 at 10-minute intervals**. Source HSD segments should be selected to cover the full study region.

## Data preparation and model input

The preprocessing workflow used for the study is summarized as follows.

1. **Radiometric preparation.** AHI observations are calibrated to reflectance or brightness temperature as appropriate for each band. Solar-elevation correction is applied to the reflective channels.
2. **Regional extraction and common grid.** All channels are mapped to the study region and prepared on a common **600 × 600** grid at approximately 2-km spatial sampling.
3. **Three-frame temporal input.** For a target time (t), the model uses observations at (t), (t-10) min, and (t-20) min. Each frame contributes 16 spectral channels.
4. **Cloud auxiliary information.** Three aligned cloud-class channels are represented by one-hot encoding of the auxiliary cloud prediction.
5. **Motion representation.** Farneback optical flow is calculated between the two consecutive frame pairs using channel index 2. The two flow fields are averaged, converted to magnitude, clipped to [0, 30], and scaled to [0, 255]. The resulting motion magnitude is stored in two auxiliary channels, following the input definition used in the study.
6. **Combined sample.** The assembled array has shape **600 × 600 × 54**: 53 model-input channels and one target-label channel.

The channel layout is:

| Channels | Content |
| --- | --- |
| 0–15 | Current frame (t) |
| 16–31 | Previous frame (t-10) min |
| 32–47 | Earliest frame (t-20) min |
| 48–50 | Three one-hot cloud channels |
| 51–52 | Motion-magnitude channels |
| 53 | Target label |

The sequence builder requires complete three-frame observations for a target timestamp; incomplete temporal windows are skipped rather than interpolated.

## Reference annotations

The released annotations are provided in [`ground_truth.zip`](ground_truth.zip). They were manually produced in Adobe Photoshop on imagery aligned with the AHI observations. Sea fog and cloud were annotated on separate layers, allowing their overlap to be retained as a distinct **cloud-obscured sea fog** category.

The five classes used by the model are:

| Class index | Category | Display color | RGB |
| --- | --- | --- | --- |
| 0 | Land | Black | (0, 0, 0) |
| 1 | Clear ocean | Dark blue | (8, 49, 73) |
| 2 | Sea fog | Gray | (174, 174, 174) |
| 3 | Cloud | Light blue | (137, 217, 222) |
| 4 | Cloud-obscured sea fog | Gray-blue | (113, 162, 165) |

The colors are used for visualization; model training and evaluation use categorical class indices. Class 4 is retained as an independent class and is not merged into sea fog.

For evaluation, reference land pixels (class 0) are excluded from the reported ocean-domain metrics. Predictions over valid ocean pixels are still evaluated normally, including false predictions of land over ocean.

## Temporal correspondence

Each target annotation at time (t) is paired with one three-frame observation sequence ending at that time:

[
(t-20 mathrm{min}, t-10 mathrm{min}, t).
]

For example, an annotation named `202012280320_groundtruth_vis.png` corresponds to the sequence ending at 03:20 and therefore uses observations at 03:00, 03:10, and 03:20.

The timestamp correspondence is recorded in [`metadata/ground_truth_alignment.csv`](metadata/ground_truth_alignment.csv). This file can be used to identify annotations with complete temporal inputs when preparing experiments.

## Training configuration

The SITS-SFNet training configuration used in the study is:

| Setting | Value |
| --- | --- |
| Input patch | 256 × 256 pixels |
| Initial learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Sparse categorical cross-entropy |
| Batch size | 64 |
| Epochs | 100 |

Training operates on 256 × 256 crops sampled from the 600 × 600 regional scenes. Class 0 is included during training, while reference land pixels are excluded when reporting the ocean-domain evaluation metrics.

See [`docs/RUNNING.md`](docs/RUNNING.md) for the current training, prediction, and visualization commands.

## Running the included scripts

Use Python 3.10 or later. The tested environment and dependency versions are documented in [`docs/RUNNING.md`](docs/RUNNING.md) and [`docs/VALIDATION.md`](docs/VALIDATION.md).

Install the core dependencies with:

```text
python -m pip install -r requirements.txt
```

Audit an observation directory:

```text
python scripts/audit_frames.py --frames data/ROI_latlon_1km --output outputs/observation_dates.csv
```

Build three-frame samples:

```text
python scripts/build_sequences.py --frames data/ROI_data --labels data/labels --cloud data/cloud_predictions --output outputs/sequences
```

Frame filenames must follow `YYYYMMDDHHMM.npy`. The sequence builder expects aligned 16-channel observation arrays, categorical label masks, and three-class cloud predictions. It validates dimensions, class indices, and numeric ranges before writing the combined samples.

Evaluate class-index predictions:

```text
python -m sits_sfnet.evaluate --predictions outputs/predictions --labels data/class_labels --list data/splits/test_stems.txt --output outputs/metrics.json --ignore-class 0
```

For the study protocol, keep **cloud-obscured sea fog** as a separate class.

## Comparison methods

The repository includes the six comparison methods used in the experiments:

- Dynamic threshold
- Three-channel U-Net
- Sixteen-channel U-Net
- SegNet
- DeepLabV3+
- SegFormer

Their input requirements, training commands, and evaluation procedures are provided in [`docs/COMPARISONS.md`](docs/COMPARISONS.md).

## Validation

The released implementation has been checked with unit tests and end-to-end smoke tests covering sequence assembly, channel placement, input validation, model construction, serialization, prediction tiling, comparison methods, and evaluation. The tested software environment and validation commands are recorded in [`docs/VALIDATION.md`](docs/VALIDATION.md).

These checks verify the released software workflow. Scientific results reported in the paper should be reproduced using the same dataset split, preprocessing settings, and experiment configuration described in the paper.

## Data availability

The repository distributes the author-created YBSF-55 reference annotations through [`ground_truth.zip`](ground_truth.zip), together with the code and metadata required to interpret the temporal samples.

The original Himawari-8 HSD observations are not redistributed. They should be obtained from the NICT/JMA source and used in accordance with the provider's terms. Large intermediate AHI arrays are likewise not hosted in this repository.

## Citation

When using the released annotations or code, please cite the associated paper and this repository:

> Y. Tang, "YBSF-55 reference annotations and SITS-SFNet code," GitHub, 2026. Available: https://github.com/TangYuzhu/SITS-SFNet.
