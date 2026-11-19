# `ss-hot`

Python helper library for the HOT **seagrass semantic segmentation** model on fAIr: a UPerNet decoder on a
ConvNeXt-Tiny v1 encoder (timm `convnext_tiny.fb_in22k_ft_in1k`), with a U-Net decoder on the same encoder as the
control it has to beat. This library holds the model logic; a fAIr model pack (`fAIr-models/models/<pack>/`) stays a
thin STAC / ZenML / Docker adapter.

**Python:** >= 3.12. Research behind the choices: [`docs/research.md`](docs/research.md); progress:
[`docs/progress.md`](docs/progress.md).

## Inference path

```text
tile dir (OAM-<x>-<y>-<z>.tif/.png/.jpg, RGB, 256 px)
  -> preprocess_chip()           a zoom-24 tile (the training resolution, ~0.92 cm per pixel) is used as it is: one
                                 256 px window. A coarser tile (zoom 23 or 22) is upscaled once to 0.92 cm, then cut
                                 into 256 px windows; zoom 21 and coarser are refused.
  -> onnxruntime session.run()   ONNX: normalise, UPerNet, sigmoid -> probability (N, 256, 256)
  -> stitch_windows()            back to one probability per tile
  -> decode_polygons()           threshold, opening / closing, trace, drop < 0.25 m2, fill holes < 0.25 m2, simplify
  -> polygons_to_feature_collection()   EPSG:4326 polygons: confidence, class_id, class_name, area_m2, source
```

A tile without georeferencing (PNG, JPEG) is placed on the map by its `OAM-x-y-z` name. Postprocess sizes are in
ground units (`min_area_m2`, `fill_holes_m2`, `simplify_m`, `open_m`, `close_m`) and are turned into model pixels with
`model_gsd_m`. Set `threshold` to the value the training run found on validation (`history.json`).

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

Labels are either a **mask folder** (`<tile stem>.png`, 0 background, 1 seagrass) or **one GeoJSON** of seagrass
polygons over the tiles (the fAIr layout). Tiles are scaled to the model resolution as at serving time (a zoom-24 tile
is not resampled), and one tile is one training sample. The prepared Timor-Leste dataset (`F:\HOTOSM\seagrass\data`,
`docs/data.md`) is 256 px zoom-24 tiles: train 13,736, validation 1,960, test 1,992, split spatially with a buffer.
Its test split comes from the same four drone flights as training (other ground, same imagery), so test scores are not
a measure of how the model does on new imagery; an outside test set (for example DTU SeagrassMapper) would need a
download.

Training augmentation (`TrainParams`, docs/augmentation.md): quarter turns and mirrors, brightness / contrast / gamma,
zoom-in jitter, randomised histogram matching and CutMix against another training tile, a down-and-up resample (fAIr
can send coarser tiles than the training zoom; they are upscaled at serving), and blur / JPEG. No hue or saturation
changes.

```bash
python scripts/finetune.py --train-images <train>/images --train-labels <train>/masks \
    --val-images <val>/images --val-labels <val>/masks --out data/runs/upernet
python scripts/finetune.py ... --out data/runs/unet --name seagrass_unet --params '{"decoder": "unet"}'  # control
python scripts/export_onnx.py --weights data/runs/upernet/seagrass_upernet.pt \
    --out data/runs/upernet/seagrass_upernet.onnx --check   # re-export a saved checkpoint, with a parity check
python scripts/evaluate.py --onnx data/runs/upernet/seagrass_upernet.onnx --images <test>/images \
    --labels <test>/masks
python scripts/predict.py --onnx data/runs/upernet/seagrass_upernet.onnx --chips <tiles> --out predictions.geojson
```

`history.json` holds the per-epoch metrics and the F1-best threshold on validation. Evaluation reports precision,
recall, F1 and IoU, and the same for the 16 px band along each window edge (`border_*`) and for the rest (`centre_*`),
to show how much the tile edges cost.

## Layout

| Module | Role |
|---|---|
| `params.py` | constants, dataclasses and `parse_*` (tolerate `None` and missing keys) |
| `tiles.py` | OAM tile name -> footprint, ground resolution, scale |
| `preprocess.py` | read RGB (georeferencing from the file or the name), scale, cut into windows, stitch back |
| `serve.py` | `predict_session`: preprocess + ONNX, no decoding |
| `decode.py`, `georef.py`, `postprocess.py` | probability -> polygons -> EPSG:4326 FeatureCollection |
| `dataset.py` | labels (mask folder or GeoJSON), scaling, `fit_tile`, `spatial_split` (no torch) |
| `augment.py` | training-time augmentation, all off by default (`TrainParams` holds the preset) |
| `model.py`, `export.py` | UPerNet (or U-Net) + ConvNeXt-Tiny, ONNX export (`train` extra) |
| `segmentation_dataset.py`, `train.py`, `evaluate.py` | torch datasets, fine-tuning, pixel and border-band metrics |

## Checks

```bash
ruff check src tests scripts
ruff format src tests scripts
uv run pytest -q
```
