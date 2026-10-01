# Code organization and historical behavior

All cleaned files are staged here; original research files, model weights and result images remain untouched in their original locations. No repository has been published.

| Research source | Release equivalent |
| --- | --- |
| `SITS_SFNet/model.py`: U-Net, encoder, decoders, SITS-SFNet | `sits_sfnet/models.py` |
| `mode_train_sits_fog_net.py` | `python -m sits_sfnet.train --variant sits_sfnet` |
| `mode_train_unet_3ch.py` | `python -m sits_sfnet.train --variant unet_3ch` |
| `mode_train_unet_16ch.py` | `python -m sits_sfnet.train --variant unet_16ch` |
| `h8_data_predict.py` | `python -m sits_sfnet.predict` |
| `result_vis.py` | `python -m sits_sfnet.visualize` |
| `Ablation Study/*/*.h5` | `configs/architectures/*.json`, exported by `scripts/export_architectures.py` |

Model builders use descriptive `build_*` names. Dataset parsing, normalization, splitting and tiling are shared instead of repeated in three training files. All new Python comments/docstrings are English. Removed hard-coded drive paths, private TensorFlow imports, unused imports, old commented experiments, plotting during training, and inline historical score notes without provenance. The visualizer retains the class colors but intentionally produces grid-aligned color PNGs rather than map figures.

`layers.py` supplies pooling/unpooling for SegNet and is not used by SITS-SFNet or either U-Net baseline. It was omitted from the initial SITS-only pass. In the subsequent comparison-method pass, SegNet and DeepLab were extracted into dedicated modules and the SegNet layers were modernized in `sits_sfnet/pooling.py`. See `COMPARISONS.md` for their separate validation and migration details. Their originals remain available in the research directory.

## Ablation audit

No Python files exist in the supplied `Ablation Study` directory. It contains seven H5 models and 4,702 PNGs. Five named experiments were recovered from the model architecture metadata, without inferring model structure from folder names. Exact JSON configurations and checkpoint SHA-256 hashes are provided; trained weight files are not included in this GitHub source package.

All five named checkpoints declare 53 input channels, but their graphs use different subsets:

| Variant | Encoder channel attention | Auxiliary input actually used |
| --- | --- | --- |
| `base` | No | None |
| `base_ca` | Yes | None |
| `base_ma` | No | Channel 51 only (`51:-1` on 53 channels) |
| `base_sa` | No | Channels 48–50 (`48:-2`) |
| `checkpoint_full` | No | Channels 48–50 (`48:-2`) |
| `sits_sfnet` from Python | Yes | Channels 48–52 |

**The checkpoint named `SITS_SFNet.h5` has the same network structure as `Base+SA.h5`, not the architecture of the current Python full model.** Their serialized configurations differ only in the encoder input-layer name (`input_2` versus `input_3` and its references). Their whole-file hashes differ, so they are not identical checkpoint files. No claim is made that their learned weights are identical. The naming/experiment provenance must be resolved before publishing the full-model results or labeling this checkpoint as the full model.

The two backup checkpoints are not promoted to release variants: both have channel attention and select four auxiliary channels (`48:-1`), with different decoder injection patterns. They do not resolve the mismatch with the current five-auxiliary-channel Python architecture.

## Data-related findings

The training scripts consume the final NPY channel as a class-index label. They do not read colored ground-truth visualizations. No upstream DAT processing or color-to-index conversion was found in this folder. The current source trains five classes, includes class 0 in the loss, and trains on 256 × 256 crops from 600 × 600 scenes. Class 4 is author-confirmed as cloud-obscured sea fog; evaluation excludes reference land pixels. Dataset file formats and coverage are documented in README.

See `RUNNING.md` for intentional changes to splitting, crop scheduling, validation and inference reconstruction. These changes mean a new training run is not automatically a historical-result reproduction.
