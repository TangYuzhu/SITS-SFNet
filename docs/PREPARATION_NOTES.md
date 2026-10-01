# Data preparation notes

This document summarizes the public data-preparation interface used by the repository.

## Source observations

The source data are Himawari-8 AHI observations obtained from the NICT Science Cloud/JMA archive. The study region is 30°N–42°N and 117°E–129°E, and the observation inventory is listed in `metadata/observation_dates.csv`.

Raw HSD observations and large intermediate AHI arrays are not redistributed in this repository. The released ground-truth annotations are provided in `ground_truth.zip`.

## Prepared observation arrays

The sequence builder starts from aligned 16-channel observation arrays named `YYYYMMDDHHMM.npy`. These arrays represent the common regional grid used by the model workflow.

The data-preparation workflow comprises radiometric calibration, reflective-channel solar-angle correction, regional extraction, common-grid resampling, and temporal alignment. The final regional model grid is 600 × 600.

## Three-frame sequence assembly

For each target timestamp (t), `scripts/build_sequences.py` reads:

- (t),
- (t-10) min,
- (t-20) min.

It then appends three one-hot cloud channels, two motion-magnitude channels, and the target class-index label to create a `(600, 600, 54)` sample.

The builder validates filenames, dimensions, class indices, numeric ranges, duplicate timestamps, and output-path safety. Temporal windows with incomplete observations are skipped and recorded in `build_report.json`.

## Released annotations

The reference annotations are distributed in `ground_truth.zip`. Their timestamp correspondence with the observation inventory is recorded in `metadata/ground_truth_alignment.csv`.

The annotation classes are:

0. Land
1. Clear ocean
2. Sea fog
3. Cloud
4. Cloud-obscured sea fog

Training and evaluation use class-index masks. Color values are provided for visualization in the repository README.

## Optical flow

The motion representation follows the Farneback optical-flow procedure documented in [OPTICAL_FLOW.md](OPTICAL_FLOW.md). The sequence builder averages the two consecutive flow fields, converts them to magnitude, clips the magnitude to [0, 30], and scales it to [0, 255].
