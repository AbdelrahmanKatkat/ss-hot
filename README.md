# `ss-hot`

Python helper library for the HOT **seagrass semantic segmentation** model on fAIr: a UPerNet decoder on a
ConvNeXt-Tiny v1 encoder (timm `convnext_tiny.fb_in22k_ft_in1k`). This library holds the model logic; a fAIr model
pack (`fAIr-models/models/<pack>/`) stays a thin STAC / ZenML / Docker adapter. Experiments that are not part of the
model (a U-Net control, a tile-edge metric) live in the git-ignored `research/` folder.

**Python:** >= 3.12. Research behind the choices: [`docs/research.md`](docs/research.md); progress:
[`docs/progress.md`](docs/progress.md).

## Inference path

```text
chip dir (OAM-<x>-<y>-<z>.tif, georeferenced RGB GeoTIFF, 256 px)
  -> preprocess_chip()           a zoom-24 tile (the training resolution, ~0.92 cm per pixel) is used as it is: one
                                 256 px window. A coarser tile (zoom 23 or 22) is upscaled once to 0.92 cm, then cut
                                 into 256 px windows; zoom 21 and coarser are refused.
  -> onnxruntime session.run()   ONNX: normalise, UPerNet, sigmoid -> probability (N, 256, 256)
  -> stitch_windows()            back to one probability per tile
  -> decode_polygons()           threshold, opening / closing, trace, drop < 0.25 m2, fill holes < 0.25 m2, simplify
  -> polygons_to_feature_collection()   EPSG:4326 polygons: confidence, class_id, class_name, area_m2, source
```

Postprocess sizes are in ground units (`min_area_m2`, `fill_holes_m2`, `simplify_m`, `open_m`, `close_m`) and are
turned into model pixels with `model_gsd_m`. Set `threshold` to the best-F1 value of `scripts/evaluate.py` on the
validation chips.

**Window:** the ONNX graph takes a fixed 256 x 256 input (UPerNet's pyramid pooling exports only for a static size).
A chip at the training zoom is exactly one window. A coarser chip upscaled to the model resolution is larger (a zoom-22
chip becomes 1039 px), so it is cut into 256 px windows (the last ones padded with a reflection), run as one batch, and
the probabilities are stitched back into one map per chip before decoding.

`ss_hot.predict_session(session, input_images, params)` returns each chip's probability and metadata (nothing
decoded); `ss_hot.postprocess_outputs(outputs, params)` turns it into a GeoJSON `FeatureCollection`. A fAIr pack's
`predict` calls the two in turn. All `params` are optional (`PreprocessParams`, `PostprocessParams`; the fAIr alias
`confidence_threshold` is accepted). Polygons are traced per tile, so a meadow cut by a tile border comes back as two
polygons.

ONNX I/O: input `image (N, 3, 256, 256)` float32 in [0, 1], dynamic `N`; output `probability (N, 256, 256)`. The window
size is fixed because UPerNet's pyramid pooling exports only for a static size; its bins (1, 2, 4, 8) divide the 8 x 8
deepest feature map of a 256 px window.

## Install, extras and who installs what

| Extra | Contents | Who installs it |
|---|---|---|
| *(none)* | numpy, Pillow, pyproj, rasterio. **No torch.** | the CPU inference image (plus `onnxruntime` from the fAIr serving base) |
| `train` | torch, timm, onnx: fine-tuning, evaluation and ONNX export | the fAIr **GPU** training image |
| `infer` | onnxruntime | running an exported ONNX locally (`scripts/predict.py`, `scripts/evaluate.py`) |
| `test` | onnxruntime, pytest | CI |

The CUDA build of torch is chosen by the training image's wheel index, not by this metadata. A pretrained model
downloads the ConvNeXt ImageNet weights (about 110 MB) from the Hugging Face Hub through timm the first time.

```bash
uv sync                                # inference core
uv sync --extra train --extra test     # development
```

## Training data

The fAIr layout: georeferenced GeoTIFF chips `OAM-<x>-<y>-<z>.tif` and one `labels.geojson` of seagrass polygons
per split. Chips are scaled to the model resolution as at serving time (a zoom-24 chip is not resampled), and one chip
is one training sample. The prepared Timor-Leste dataset (`F:\HOTOSM\seagrass\data`, `docs/data.md`) is 256 px
zoom-24 chips: train 13,736, validation 1,960, test 1,992, split spatially with a buffer. Its test split comes from the
same four drone flights as training (other ground, same imagery), so test scores are not a measure of how the model
does on new imagery; an outside test set (for example DTU SeagrassMapper) would need a download.

Training augmentation (`TrainParams`, docs/augmentation.md): quarter turns and mirrors, brightness / contrast / gamma,
zoom-in jitter, randomised histogram matching and CutMix against another training chip, a down-and-up resample (fAIr
can send coarser chips than the training zoom; they are upscaled at serving), and blur / JPEG. No hue or saturation
changes.

```bash
python scripts/finetune.py --train-images <train>/chips --train-labels <train>/labels/labels.geojson \
    --val-images <val>/chips --val-labels <val>/labels/labels.geojson --out data/runs/upernet
python scripts/export_onnx.py --weights data/runs/upernet/seagrass_upernet.pt \
    --out data/runs/upernet/seagrass_upernet.onnx --check   # re-export a saved checkpoint, with a parity check
python scripts/evaluate.py --onnx data/runs/upernet/seagrass_upernet.onnx --images <val>/chips \
    --labels <val>/labels/labels.geojson   # threshold sweep; then the same on test
python scripts/predict.py --onnx data/runs/upernet/seagrass_upernet.onnx --chips <chips> --out predictions.geojson
```

Experiments in `research/` (not part of the package): `unet_control.py` trains a U-Net decoder on the same encoder
with the library's training loop (UPerNet must beat it); `border_band.py` scores the 16 px band along window edges
against the centre (does overlapping-window serving pay?).

## Layout

| Module | Role |
|---|---|
| `params.py` | constants, dataclasses and `parse_*` (tolerate `None` and missing keys) |
| `tiles.py` | OAM chip name -> ground resolution -> scale |
| `preprocess.py` | read RGB, scale, cut into windows, stitch back |
| `serve.py` | `predict_session`: preprocess + ONNX, no decoding |
| `decode.py`, `georef.py`, `postprocess.py` | probability -> polygons -> EPSG:4326 FeatureCollection |
| `dataset.py` | labels GeoJSON -> chip masks, scaling, `fit_tile`, `spatial_split` (no torch) |
| `augment.py` | training-time augmentation, all off by default (`TrainParams` holds the preset) |
| `model.py`, `export.py` | UPerNet + ConvNeXt-Tiny, ONNX export (`train` extra) |
| `segmentation_dataset.py`, `train.py`, `evaluate.py` | torch datasets, fine-tuning, pixel metrics and threshold sweep |

## Checks

```bash
ruff check src tests scripts
ruff format src tests scripts
uv run pytest -q
```
