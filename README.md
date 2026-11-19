# `ss-hot`

Python helper library for the HOT **seagrass semantic segmentation** model on fAIr: a UPerNet decoder on a
ConvNeXt-Tiny v1 encoder (timm `convnext_tiny.fb_in22k_ft_in1k`, Apache-2.0). It follows the split used by `ts-hot`
and `td-hot`: this library holds the model logic, and a fAIr model pack (`fAIr-models/models/<pack>/`) stays a thin
STAC / ZenML / Docker adapter.

**Python:** >= 3.12. Research behind the choices: [`docs/research.md`](docs/research.md); progress:
[`docs/progress.md`](docs/progress.md).

## Inference path

```text
chip dir (OAM-<x>-<y>-<z>.tif/.png/.jpg, RGB)
  -> preprocess_chip()           one resize to 3 cm per pixel (scale = chip resolution from the zoom / 0.03 m),
                                 then 512 px windows (padded with a reflection of the chip)
  -> onnxruntime session.run()   ONNX: normalise, UPerNet, sigmoid -> probability (N, 512, 512)
  -> stitch_windows()            back to one probability per chip
  -> decode_polygons()           threshold, opening / closing, trace, drop < 0.25 m2, fill holes < 0.25 m2, simplify
  -> polygons_to_feature_collection()   EPSG:4326 polygons: confidence, class_id, class_name, area_m2, source
```

`ss_hot.predict_session(session, input_images, params)` returns each chip's probability and metadata (nothing
decoded); `ss_hot.postprocess_outputs(outputs, params)` turns it into a GeoJSON `FeatureCollection`. A fAIr pack's
`predict` calls the two in turn. All `params` are optional (`PreprocessParams`, `PostprocessParams`; the fAIr alias
`confidence_threshold` is accepted). Polygons are traced per chip, so a meadow cut by a chip border comes back as two
polygons.

ONNX I/O: input `image (N, 3, 512, 512)` float32 in [0, 1], dynamic `N`; output `probability (N, 512, 512)`. The window
size is fixed because UPerNet's pyramid pooling exports only for a static size.

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

Labels are either a **mask folder** (`<chip stem>.png`, 0 background, 1 seagrass, 255 ignore) or **one GeoJSON** of
seagrass polygons over georeferenced chips (the fAIr layout). Chips are scaled to the model resolution as at serving
time; chips that become small are joined with their tile-grid neighbours into one 512 px window. The prepared
Timor-Leste dataset (`F:\HOTOSM\seagrass\data`, see `docs/data.md`) needs `--source-gsd-m 0.0092` because its chip
names say zoom 22 while their x/y are on the zoom-23 grid.

```bash
python scripts/finetune.py --train-images <split>/images --train-labels <split>/masks \
    --val-images <val>/images --val-labels <val>/masks --source-gsd-m 0.0092 --out data/runs/upernet
python scripts/evaluate.py --onnx data/runs/upernet/seagrass_upernet.onnx --images <test>/images \
    --labels <test>/masks --source-gsd-m 0.0092
python scripts/predict.py --onnx data/runs/upernet/seagrass_upernet.onnx --chips <chips> --out predictions.geojson
```

## Layout

| Module | Role |
|---|---|
| `params.py` | constants, dataclasses and `parse_*` (tolerate `None` and missing keys) |
| `tiles.py` | OAM chip name -> ground resolution -> scale |
| `preprocess.py` | read RGB, scale, cut into windows, stitch back |
| `serve.py` | `predict_session`: preprocess + ONNX, no decoding |
| `decode.py`, `georef.py`, `postprocess.py` | probability -> polygons -> EPSG:4326 FeatureCollection |
| `dataset.py` | labels (mask folder or GeoJSON), scaling, mosaics, `spatial_split` (no torch) |
| `augment.py` | training-time augmentation, all off by default |
| `model.py`, `export.py` | UPerNet + ConvNeXt-Tiny, ONNX export (`train` extra) |
| `segmentation_dataset.py`, `train.py`, `evaluate.py` | torch datasets, fine-tuning, pixel metrics |

## Checks

```bash
ruff check src tests scripts
ruff format src tests scripts
uv run pytest -q
```
