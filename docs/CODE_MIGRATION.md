# Code organization

This document summarizes the organization of the released implementation.

| Component | Release location |
| --- | --- |
| SITS-SFNet and U-Net models | `sits_sfnet/models.py` |
| Training entry point | `python -m sits_sfnet.train` |
| Prediction | `python -m sits_sfnet.predict` |
| Visualization | `python -m sits_sfnet.visualize` |
| Evaluation | `python -m sits_sfnet.evaluate` |
| SegNet pooling/unpooling | `sits_sfnet/pooling.py` |
| Comparison methods | `sits_sfnet/deeplab.py`, `sits_sfnet/segnet.py`, `sits_sfnet/segformer.py`, `sits_sfnet/dynamic_threshold.py` |
| Architecture configurations | `configs/architectures/` |

The released code consolidates data loading, normalization, cropping, tiling, model construction, and evaluation into shared modules. Machine-specific paths, obsolete imports, interactive plotting, and duplicate model definitions have been removed from the public implementation.

## Model variants

The training interface provides the primary SITS-SFNet implementation, two U-Net baselines, and several architecture variants used for ablation analysis:

| Variant | Description |
| --- | --- |
| `sits_sfnet` | Full released SITS-SFNet |
| `unet_3ch` | Three-channel U-Net baseline |
| `unet_16ch` | Sixteen-channel U-Net baseline |
| `base` | Base architecture |
| `base_ca` | Base + channel attention |
| `base_ma` | Base + motion auxiliary input |
| `base_sa` | Base + spatial/cloud auxiliary input |
| `checkpoint_full` | Archived architecture configuration retained for compatibility |

The primary model for reproducing the paper's SITS-SFNet method is `sits_sfnet`. The additional architecture files are retained to support ablation and compatibility experiments.

## Data interface

Training arrays use the final channel as the class-index target. The released code trains five classes, includes class 0 in the training loss, and uses 256 × 256 crops from 600 × 600 scenes. Class 4 denotes cloud-obscured sea fog. Reference land pixels are excluded from the reported ocean-domain evaluation metrics.

See [README.md](../README.md) for the dataset layout and [RUNNING.md](RUNNING.md) for execution commands.
