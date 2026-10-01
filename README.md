# YBSF-55: data preparation

This repository is a preparation-stage package for the YBSF-55 sea-fog dataset. It documents the verified processing steps and provides configurable three-frame assembly, SITS-SFNet/U-Net training, inference and checkpoint-derived ablation architectures. Raw satellite observations and the large intermediate AHI NumPy arrays are not redistributed in this repository. Readers should obtain source observations from NICT Science Cloud and follow the acquisition selection and preprocessing description below. Author-created ground-truth annotations are a separate accompanying resource; their release link will be added when available. The upstream preprocessing implementation is not bundled, so this is not an automated end-to-end reproduction package.

For model installation and commands see [Running the models](docs/RUNNING.md). See [Code migration and ablation findings](docs/CODE_MIGRATION.md) before interpreting historical checkpoints: the file named `SITS_SFNet.h5` does not match the current full-model Python definition. Executed checks are recorded in [Validation](docs/VALIDATION.md).

The six comparison methods (DeepLabV3+, SegFormer, SegNet, dynamic threshold, three-channel U-Net and 16-channel U-Net) are documented in [Comparison experiments](docs/COMPARISONS.md), including their environments, commands, retained input conventions and corrections that can affect results.

## Raw observations, acquisition, and preprocessing

### Observation source and download

The source observations are Himawari-8 Advanced Himawari Imager (AHI) multispectral data in Himawari Standard Data (HSD) format. The study region extends from **30°N to 42°N and 117°E to 129°E**. The current processed inventory contains **55 distinct observation dates during 2016–2020**. The name YBSF-55 refers to 55 observation dates, not necessarily 55 independent fog events. Consecutive observation days are treated as one fog episode under the authors' grouping convention.

According to the authors, the Himawari-8 standard data were downloaded through the **Science Cloud of the National Institute of Information and Communications Technology (NICT), Japan**, with the source attributed to JMA's Meteorological Satellite Center. The historical [NICT Science Cloud download link](https://sc-nc-web.nict.go.jp/wsdb_osndisk/shareDirDownload/03ZzRnKS?lang=en) was cited as accessed on **15 March 2020**. This historical access date predates some dataset observations and must not be interpreted as the acquisition date of the entire 2016–2020 collection. Use the provider's current access conditions when obtaining data. The historical link has not been confirmed to remain active; this README does not guarantee anonymous access or continued availability of every historical file.

To obtain the source observations:

1. Visit the [NICT Science Cloud data service](https://sc-nc-web.nict.go.jp/) and use the historical download link above if it remains available. Locate the Himawari-8 HSD archive and follow the provider's current access instructions. The repository does not include a downloader or provider credentials.
2. Select the 55 observation dates in [observation_dates.csv](metadata/observation_dates.csv), with nominal time slots **03:00–05:20 at 10-minute intervals**, and all **16 bands B01–B16**. Raw HSD filename times are UTC; confirm correspondence with the annotation timestamps before pairing derived files.
3. Obtain the segments covering **30°N–42°N, 117°E–129°E**. The inspected raw example uses **S0210 and S0310** for each band. Verify coverage using the reader's geolocation; acquire additional segments if needed rather than treating the example as a universal coverage guarantee.
4. Preserve provider filenames, decompress archives when necessary, and check that files are complete and readable. For the inspected two-segment selection, expect **32 DAT files per time slot** or **480 for 15 complete slots**. Exclude unfinished transfer files.
5. Apply the calibration, solar-angle correction, regional reprojection and common-grid preparation described below. Save aligned single-time 16-channel arrays as `YYYYMMDDHHMM.npy`, then assemble three-frame samples with the supplied script. That script starts from prepared arrays, not DAT files.

The date inventory records the authors' local holdings, including missing time slots; it is not a claim that the provider archive has the same gaps.

Example filename: `HS_H08_20201228_0300_B01_FLDK_R10_S0210.DAT`. JMA defines H08 as Himawari-8, FLDK as full disk, B01 as band 1, R10 as nominal 1.0-km resolution at the sub-satellite point, and S0210 as segment 2 of 10. HSD filename times are UTC; this does not independently establish whether derived NPY filenames were ever time-zone converted. See the [JMA HSD User's Guide, v1.3](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/hsd_sample/HS_D_users_guide_en_v13.pdf).

### Processing stages and evidence boundaries

1. **Radiometric calibration and solar correction — author-described method; implementation pending archival.** The authors report radiometric calibration of the level-0 AHI observations to reflectance and brightness temperature, followed by solar-elevation-angle correction of the visible and near-infrared observations. The exact calibration coefficients, correction formula, channel mapping, physical units and integer scaling/fill conventions must be recovered from the historical implementation. The phrase level-0 is retained from the authors' description; no separate product-level validation has been performed.
2. **Region extraction and common grid — author-described method; implementation pending archival.** All channels were converted to an equidistant latitude–longitude projection over 30°N–42°N and 117°E–129°E. The authors describe the output as nominal 2-km spatial resolution, with channels originally finer than 2 km resampled to a common size of **600 × 600**. For this 12° × 12° extent, 600 cells per axis correspond to approximately 0.02° angular spacing; nominal 2 km does not imply identical ground-distance spacing at every latitude. Exact grid registration, interpolation kernel, processing order and software versions remain to be recovered. A currently inspected intermediate file, `202012280300.npy` in `ROI_latlon_1km`, is `(1200, 1200, 16)` `int16`; its specific conversion to the final 600 × 600 arrays has not been verified in code. The packaged script therefore does not silently resample this intermediate product.
3. **Cloud predictions — required external input.** The legacy cloud classifier clips each channel to fixed ranges, normalizes it and predicts three classes. Its historical model, mask semantics and environment must be packaged and verified separately. This package accepts aligned single-channel predictions with indices 0, 1 and 2; it does not regenerate them.
4. **Three-frame assembly — implemented.** For each label timestamp `t`, read arrays at `t`, `t−10 min` and `t−20 min`, in that order. Require all three files; do not fill temporal gaps. Each frame contributes 16 channels. Labels are resized with nearest-neighbor interpolation. Spectral frames and cloud predictions must already match the requested dimensions.
5. **Motion and auxiliary channels — implemented.** Compute Farneback flow from `t−10` to `t` and from `t−20` to `t−10`, using array channel index 2. Parameters are `(0.5, 3, 15, 5, 7, 1.5, 0)`. Average the flow vectors, take their magnitude, clip to [0, 30], scale to [0, 255] and cast to `uint16`. Preserve the historical duplication of this magnitude in two channels; these are not horizontal/vertical components. Cloud indices are converted to three one-hot channels.
6. **Storage — implemented.** Store `(600, 600, 54)` `uint16` arrays by default: channels 0–15 current frame; 16–31 previous frame; 32–47 earliest frame; 48–50 cloud one-hot channels; 51–52 duplicated motion magnitude; 53 target label. Label indices are 0–4, as defined in the annotation table below. Inputs with negative values, non-integer values or values beyond `uint16` are rejected rather than silently wrapped. Resolve their physical meaning upstream.

Fifteen consecutive input frames provide thirteen candidate target windows (03:20–05:20). The current filename inventory contains 817 frames and 696 valid three-frame windows, compared with 825 frames and 715 windows for a complete 55 × 15 inventory. The eight absent frames affect nineteen windows. These counts do not verify array integrity, annotations or auxiliary-input availability, and therefore are not final training-sample counts.

| Year | Dates | Frames | Candidate three-frame windows |
| --- | ---: | ---: | ---: |
| 2016 | 10 | 149 | 127 |
| 2017 | 2 | 30 | 26 |
| 2018 | 18 | 265 | 222 |
| 2019 | 8 | 120 | 104 |
| 2020 | 17 | 253 | 217 |
| Total | 55 | 817 | 696 |

## Manual annotation and label dimensions

The reference annotations were drawn manually in Adobe Photoshop on imagery aligned with the AHI observations. Sea fog and clouds were painted on separate layers: gray denotes sea fog, light blue denotes cloud, and their gray-blue overlap denotes **cloud-obscured sea fog**, a separate category. After applying the land mask, the remaining ocean area is seawater. Colors encode categories rather than continuous image intensities. The original class indices are retained:

| Class index | Category | Display color | RGB in the packaged visualizer |
| --- | --- | --- | --- |
| 0 | Land | Black | (0, 0, 0) |
| 1 | Seawater | Dark blue | (8, 49, 73) |
| 2 | Sea fog | Gray | (174, 174, 174) |
| 3 | Cloud | Light blue | (137, 217, 222) |
| 4 | Cloud-obscured sea fog | Gray-blue | (113, 162, 165) |

The RGB values specify the packaged visualization palette; they are not a requirement that every pixel in historical rendered images exactly matches these values. No label reconstruction from color images is performed by this package.

**Evaluation excludes reference land pixels (class 0). Cloud-obscured sea fog remains a separate class and is not merged into sea fog.** Seawater remains background in the evaluation domain so that false fog/cloud predictions over seawater are counted. The shared evaluator aggregates pixel counts across the supplied images; this describes the released evaluator, not a verified historical averaging convention.

There are two distinct label sizes in the existing workflow:

- **Full-region label: 600 × 600.** The sequence-assembly script resizes the label to this size using nearest-neighbor interpolation and stores it in the final channel of the combined array.
- **Network training target: 256 × 256.** `SITS_SFNet/mode_train_sits_fog_net.py` randomly crops the same 256 × 256 spatial window from all input channels and the label. It does not resize the full scene to 256 × 256. Prediction patches are subsequently assembled into a 600 × 600 regional output.

The code establishes these dimensions, but does not record why 600 × 600 was chosen. The author-described 2-km regional product is consistent with that size; computational efficiency should not be presented as the verified historical reason.

## Dataset files and temporal correspondence

The following layout describes the authors' local observation inventory and the separate annotation resource. **The AHI directory is not an uploaded dataset archive**; retain this naming convention when preparing observations downloaded from the source:

```text
AHI/
  ROI_latlon_1km/
    2016/ ... 2020/
      YYYYMMDDHHMM.npy
ground_truth/
  YYYYMMDDHHMM_groundtruth_vis.png
```

- **Local AHI NumPy inventory (not distributed):** `AHI/ROI_latlon_1km` contains 817 single-time arrays across 55 observation dates (2016–2020), totaling approximately 37.65 GB uncompressed. The inspected array has shape `(1200, 1200, 16)` and dtype `int16`, with spatial axes followed by the 16 AHI channels. These intermediate arrays are distinct from the assembled `(600, 600, 54)` model samples described above; they do not themselves contain three time steps or a target-label channel. Stored integer values should not be assumed to be physical reflectance or temperature without the upstream encoding information.
- **Reference annotation images:** `ground_truth` contains 427 color PNG images (approximately 127 MB), named by their target timestamp. The existing exported images are 1350 × 1350 RGBA visualizations of the manual annotation categories. Their rendered dimensions are distinct from the 600 × 600 regional training labels. These color images document the reference annotations; the provided training and evaluation programs require class-index labels, not direct RGB/RGBA input.
- **One sequence, one annotation:** a target at time `t` is paired with observations at `t`, `t−10 minutes`, and `t−20 minutes`. For example, `202012280320_groundtruth_vis.png` corresponds to the sequence ending at 03:20, using 03:20, 03:10, and 03:00 observations. Each three-frame window has one target annotation, not one annotation per input frame. Missing frames are not interpolated.
- **Coverage:** the current inventory provides 696 candidate three-frame windows, not 696 annotated samples. Of the 427 annotation images, 417 match complete windows in the listed AHI inventory. The other 10 belong to 2020-06-04, an additional annotation date without matching observations in this inventory; retain these as additional annotations rather than counting them as paired YBSF-55 samples. See `metadata/ground_truth_alignment.csv` for the pairing inventory.

These counts and sizes were checked on 30 September 2026. To avoid redistributing large satellite files, neither the raw DAT observations nor the approximately 37.65 GB intermediate NumPy collection will be uploaded with this release. Their local inventory is retained to document the observation selection and temporal coverage. Readers obtain the observations from the original provider and prepare their own arrays using the workflow above.

The author-created `ground_truth` images remain a separate planned annotation release; they cannot be obtained from the satellite provider. Their download location will be added once published. No data files have been moved, deleted or uploaded during this documentation update. The `.gitignore` continues to exclude large NumPy and DAT files from ordinary Git tracking.

The supplied code covers sequence assembly, optical-flow features and model workflows. The exact historical DAT-to-array implementation and 1200-to-600 grid conversion are not included; the preprocessing description is methodological guidance, not a claim that these missing steps can already be reproduced by a provided command.

## Training configuration

The author-reported SITS-SFNet training settings are:

| Setting | Value |
| --- | --- |
| Input patch | 256 × 256 pixels, cropped from the regional sample |
| Initial learning rate | 0.001 |
| Loss | Sparse categorical cross-entropy |
| Batch size | 64 |
| Epochs | 100 |

The shared training entry point uses Adam and these learning-rate, batch-size, and epoch defaults. See `docs/RUNNING.md` for the command and documented changes to sampling and validation. Comparison-specific settings, including SegFormer, remain documented separately in `docs/COMPARISONS.md`. Excluding land from reported evaluation does not change the historical training loss, which includes class 0. Training settings describe how a model is trained; trained weight files are separate artifacts and are not bundled here.

To evaluate aligned class-index predictions and reference labels:

```text
python -m sits_sfnet.evaluate --predictions outputs/predictions --labels data/class_labels --list data/splits/test_stems.txt --output outputs/metrics.json --ignore-class 0
```

Do not enable `--merge-overlap-into-fog` for the study's separate cloud-obscured sea-fog category.

## Running the included scripts

Use Python 3.10 or later. Install dependencies with `python -m pip install -r requirements.txt`. The compatibility ranges are provisional; the smoke-test environment is recorded in `docs/PREPARATION_NOTES.md`.

```text
python scripts/audit_frames.py --frames data/ROI_latlon_1km --output outputs/observation_dates.csv
python scripts/build_sequences.py --frames data/ROI_data --labels data/labels --cloud data/cloud_predictions --output outputs/sequences
```

Frame filenames must be `YYYYMMDDHHMM.npy`; year subfolders are supported. Label PNG names must begin with the target timestamp. The matching cloud file is `<label_stem>_predict.png`. Labels and cloud inputs are class-index images, not RGB visualizations. Default dimensions are 600 × 600; supplying 1200 × 1200 inputs causes an explicit error rather than undocumented downsampling. The sequence output directory must not already exist. Each run writes `build_report.json`, including missing-frame skips and validation errors; any validation error produces a nonzero exit status.

The scripts do not modify their inputs. Large data, models, secrets and unfinished downloads are excluded by `.gitignore`. Metadata includes dates and counts only. No observation archive, credentials or pretrained model is bundled. The original observation-source link is given above; the annotation download link is pending.

## Release status

Release scope: code, documentation and observation metadata, with author-created annotations to be linked separately. Raw observations and intermediate AHI arrays are obtained/prepared by readers rather than rehosted here. Annotation access, citation details, licenses and any auxiliary-model release will be added as available. No open-source or redistribution license is asserted on behalf of the authors or data provider. This directory is staged locally; it has not been uploaded.

## Optical-flow utilities

See [Optical flow](docs/OPTICAL_FLOW.md) for two-image Farneback calculation, saved arrow visualizations and inspection of stored sequence motion magnitudes. These utilities consolidate the original optical-flow exploration scripts without changing model feature assembly.
