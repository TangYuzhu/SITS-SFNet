# Optical flow

The original `code/optical_flow` folder contains two exploratory scripts. They are consolidated into `scripts/visualize_optical_flow.py`, with English names/comments, explicit paths, saved outputs and no GUI dependency. Originals remain unchanged. IDE files, unused TensorFlow/patchify imports and inactive experiments are omitted.

## Environment and commands

Install `requirements.txt`; NumPy, Pillow and OpenCV headless are sufficient. Run from the repository root:

```text
python scripts/visualize_optical_flow.py pair --previous data/earlier.png --current data/later.png --output outputs/flow_pair
python scripts/visualize_optical_flow.py sample --input data/combined/202012280320.npy --output outputs/stored_motion
```

Output directories must be new. Inputs are not modified.

## Two-image calculation

This replaces `optical_flow.py`. Both images are resized to 600 x 600 using linear interpolation (override with `--size`). The blue channel is used, preserving the original OpenCV BGR channel-0 selection; this is distinct from selecting AHI array channel index 2. Supply the earlier image as `--previous` and later image as `--current`; the old suffixes `_2` and `_0` alone do not establish timestamps.

Farneback parameters are `(0.5, 3, 15, 5, 7, 1.5, 0)`. Outputs are `flow.npz` (HxWx2 displacement, magnitude and angle in radians), `flow_arrows.png`, `magnitude.png` and `report.json`. Displacement is measured in pixels on the resized grid per supplied frame pair, not m/s. Horizontal displacement is positive rightward; vertical displacement is positive downward. Arrows use image coordinates directly. `--step` controls spacing and `--arrow-scale` controls visual enlargement only. The grayscale magnitude preview clips at 30 pixels and scales to 0..255; NPZ values are not clipped. This corrects the legacy RGB/BGR display and ambiguous vertical-arrow plotting conventions.

## Stored motion inspection

This replaces `file_test.py`. The legacy script plotted the final two input channels as if they were horizontal/vertical components after shifting by -0.5. In the recovered sequence builder, channels 51 and 52 (zero-based) contain identical scaled magnitudes, so those arrows cannot establish motion direction. The cleaned command saves each channel as a grayscale magnitude image and reports whether they are identical. It does not reconstruct direction or modify training inputs.

## Relation to the model

`scripts/build_sequences.py` remains the model's motion-feature implementation: it uses raw AHI channel index 2 for two adjacent frame pairs, averages the two flow vectors, computes magnitude, clips to 0..30, scales to 0..255 and duplicates the result in channels 51 and 52. The two-image visualization is an exploratory tool, not a replacement for that three-frame feature assembly. No change was made to the model's motion features.
