# Preparation record

Initially prepared on 2026-09-28; model packaging added on 2026-09-29. This package now covers the raw-data README section, inventory, three-frame assembly, model training/inference and checkpoint-derived ablation configurations. Annotations, upstream processing, cloud-model release and final data access statements remain for later preparation. See `CODE_MIGRATION.md` and `VALIDATION.md` for the model work.

## Source mapping and changes

`scripts/build_sequences.py` is a cleaned derivative of `code/conbine_all_data_into_numpy.py`. Removed machine-specific paths, unused plotting/imports and commented experiments. Added a main entry point, command-line paths, deterministic iteration, duplicate detection, safe NumPy loading, dimension/class/range validation, output-directory protection and a machine-readable run report. Original temporal ordering, optical-flow parameters, label interpolation, duplicated magnitude and uint16 output are retained. Optical-flow inputs are explicitly float32. Negative int16 values no longer silently wrap into uint16; this is an intentional validation change, not a calibration algorithm.

`scripts/audit_frames.py` is new and audits filenames, not scientific content. `metadata/observation_dates.csv` is generated from the current five year directories. No raw input has been moved or changed.

The original research scripts are retained in their working locations because other experiments may depend on them. Only cleaned deliverables are collected in this upload directory. The old root `YBSF-Readme.md` was not included: it describes 24 events and a different region/resolution and is not evidence for YBSF-55.

The legacy cloud detector was inspected but is not included as a runnable release: its mask indexing (`im[mask, :]` versus `mask > 0`), class meanings and serialized-model dependencies require verification. The current sequence builder instead accepts validated prediction PNG files. This avoids presenting a partially repaired classifier as historically equivalent.

## Outstanding release information

- Author confirmed NICT Science Cloud as the original download platform and supplied the historical URL in README. The cited access date, 2020-03-15, predates later observations; complete acquisition dates and terms remain to be confirmed. Current historical-link accessibility was not verified.
- HSD decoding/calibration, channel order, units, fill handling, geolocation, grid and resampling implementation.
- Verify derived filename time zone; JMA HSD raw filename times are UTC.
- Recover the actual 1200-to-600 processing step and dependencies.
- Final labels, cloud model/mask, split lists, data repository and DOI.
- Author-approved code license and data redistribution terms.

Author-confirmed method: radiometric calibration to reflectance/brightness temperature; solar-elevation correction of visible/near-infrared data; conversion to an equidistant latitude–longitude projection; nominal 2-km output and 600 × 600 channels, resampling finer channels. These statements are author-provided provenance, not claims that the missing implementation was recovered. The current 1200 × 1200 intermediate-to-final conversion still requires the original code.

## Verification

Local smoke-test environment: NumPy 2.5.0, OpenCV 4.12.0, Pillow 12.2.0. Synthetic checks exercise channel order, one-hot encoding, constant-frame motion, shape/range rejection, missing-frame reporting and overwrite protection. These checks do not establish scientific equivalence on historical data, which requires the missing upstream processing and real aligned inputs.

Result: `python -m unittest discover -s tests -v` passed both tests on 2026-09-28. The actual filename audit produced 55 dates, 817 frames and 696 candidate windows. Re-run tests after further packaging changes.

## Distribution decision, 2026-10-01

At the author's request, raw HSD observations and the large intermediate AHI NumPy collection will not be rehosted. README documents the original NICT source, observation selection and processing stages. Ground-truth annotations remain a separate planned resource. This supersedes the earlier plan to upload both AHI arrays and annotations. No local data were moved or deleted.
