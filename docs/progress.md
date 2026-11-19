# ss-hot progress

Seagrass semantic segmentation helper library for HOT fAIr (UPerNet + ConvNeXt-Tiny v1). Newest entries at the
bottom.

## 2026-11-19

### Decisions so far
- Model: UPerNet decoder + ConvNeXt-Tiny **v1** encoder (timm `convnext_tiny.fb_in22k_ft_in1k`, Apache-2.0). ConvNeXt-V2
  ImageNet weights are CC-BY-NC and are not used. Baseline to compare against: U-Net + ResNet34 / EfficientNet.
- Repository layout, packaging and public API follow the fAIr helper-library rules (the fAIr pack stays a thin
  adapter).
- Training data: local build at `F:\HOTOSM\seagrass\data` (from HF `hotosm/vhr-seagrass-segmentation`, same train split,
  extra one-chip buffer between validation and test). Train 3,434 / validation 490 / test 498 chips.

### Known data issues (before any training)
- Validation (project 55909, north-east) is under-mapped: 226 of 490 chips have no label although the imagery shows
  seagrass-like texture; seagrass pixel share is 14% in validation vs 59% train / 49% test.
- Training chips are named `OAM-x-y-22` but their x/y are on the zoom-23 grid (~4.7 m, ~0.92 cm/px per 512 px chip).

### Work log
- [x] Research agents started: preprocessing, data, postprocessing, augmentation, model (results → `docs/research.md`).
- [x] Repository scaffold: `pyproject.toml` (core numpy/Pillow/pyproj/rasterio; extras `train` = torch/timm/onnx,
  `infer`, `test`), `.gitignore`, `src/ss_hot`, `tests`, `scripts`.
- [x] Inference path (torch-free): `params`, `tiles`, `preprocess` (scale to `model_gsd_m`, cut into 512 px windows,
  stitch back), `serve` (`predict_session`), `decode`/`georef`, `postprocess` (EPSG:4326 FeatureCollection,
  RFC 7946 ring winding, geodesic `area_m2`).
- [x] Training path: `dataset` (mask-folder and GeoJSON labels, scaling as at serving, mosaics of neighbouring chips,
  `spatial_split`), `augment`, `model` (UPerNet head on timm ConvNeXt-Tiny features), `export` (ONNX, fixed 512 px
  windows, dynamic batch), `segmentation_dataset`, `train`, `evaluate`.
- [x] Scripts: `finetune.py`, `predict.py`, `evaluate.py` (thin).
- [x] Offline tests: 52 passed, 1 module skipped (`test_model.py` needs `timm`, not installed yet).
- [x] Data agent → `docs/data.md`; postprocessing agent → `docs/postprocessing.md`. Postprocess defaults updated from the
  label statistics: `min_area_px` 278 (0.25 m²), `fill_holes_px` 278, `open_px` 1, `close_px` 2, `simplify_px` 2.
- [x] Augmentation agent → `docs/augmentation.md`: hue shift dropped; `stretch` (contrast and saturation x0.5–x2),
  `water_tint` (red fades most), `lowres_prob`/`lowres_max` (down-and-up resample for the 8–32x coarser serving tiles)
  added; seagrass preset is the `TrainParams` default, `AugmentParams()` stays the identity.
- [x] Model agent → `docs/model.md`: smp's UPerNet cannot export with its (1, 2, 3, 6) bins, so the hand-written head
  stays; added `layer_decay` 0.9, epochs 40, warmup 400, monitor `val_iou` (train loss without a validation set);
  no BatchNorm on the 1 x 1 pooling branch.
- [x] `docs/research.md`: findings of the four finished agents and what the code took from them.
- [x] Tests after the changes: 55 passed, 1 module skipped (needs `timm`); ruff clean.
- [x] Preprocessing agent → `docs/preprocessing.md`: 3 cm confirmed; `MAX_SCALE` 16 → 5 (zoom 20 accepted, zoom 19
  refused), `lowres_max` 4 → 5; raw images, ImageNet normalisation in the graph. `WINDOW_PX` stays 512 (mosaics fill
  it; the proposed 256 would hold a single chip). `docs/research.md` §1 written.
- [x] Tests after the changes: 55 passed, 1 module skipped (needs `timm`); tiles doctests pass; ruff clean.
- [x] First commit pushed to `origin/main` (`AbdelrahmanKatkat/ss-hot`).
- [x] `scripts/export_onnx.py`: saved `<name>.pt` -> serving ONNX, optional `--check` (onnxruntime vs torch, limit
  1e-4; the file is not written when it fails). Argument handling checked; the export itself needs `timm`.
- [x] Correction: the first five agents analysed our own data and code; the request was a **state-of-the-art and
  available-datasets survey** per topic. The five agents were re-run with that brief (preprocessing, datasets,
  postprocessing, augmentation, models); each replaces its `docs/<topic>.md` (old versions are in the first commit).
- [x] Root cause of the misnamed chips found: the four Tasking Manager projects serve imagery with `@4x` (1024 px
  zoom-22 tiles); `tm_dataset` cut them into 512 px chips (= zoom-23 tiles) but named them with the imagery zoom (22).
  Fixed in `kits-hot/tm_dataset` (`tiles.oam_tile_id` names the chip with the zoom of the tile it covers;
  `geo.chip_bounds_from_id` uses the zoom in the id; old ids still resolve); its tests pass (33).
- [x] Dataset rebuilt as 256 px **zoom-24** tiles (0.92 cm/px) by cutting each 512 px chip in four, no resizing; every
  tile keeps its parent's split, so the spatial split and both buffers are unchanged on the ground. The 512 px layout
  is removed; the parquet files stay. Result in `F:\HOTOSM\seagrass\data`: train 13,736 / validation 1,960 / test
  1,992 tiles (4x each 512 px split), seagrass share 59.3% / 14.0% / 49.1% (unchanged); no tiles of different splits
  touch; cropped masks agree with the re-rasterized labels (worst checked tile 99.7%, 708 tiles checked).
- [x] All five survey reports written (`data.md` 52 sources, `preprocessing.md` 35, `augmentation.md` 43,
  `postprocessing.md` 57, `model.md` 35); `docs/research.md` rewritten from them, with the proposed ss-hot changes
  (§6). The model stays UPerNet + ConvNeXt-Tiny v1 (rank 1 of the shortlist, on condition it beats a U-Net control).

## 2026-11-20

### Decisions
- Non-commercial licences are acceptable for this project: ConvNeXt V2 and SegFormer weights are usable options. The
  model stays UPerNet + ConvNeXt-Tiny v1 for now.
- Serving zoom: fAIr recommends the training zoom (24) but can send coarser tiles; they are upscaled to the training
  resolution, and the `lowres` augmentation stays.
- Kept on request: `freeze_encoder`, `unfreeze_from`, `encoder_lr_factor`, `epoch_fraction`.

### Work log
- [x] ss-hot follows the 256 px zoom-24 dataset (research.md §6, "forced by the dataset fix"): `MODEL_GSD_M` 0.0092 and
  no resampling within 2%; `WINDOW_PX` 256 (ONNX `(N, 3, 256, 256)`); one tile per sample; 0/1 masks (no `IGNORE`);
  `source_gsd_m`, mosaics, `MAX_INPUT_PX`, `smooth_sigma_px` and the "mirror" orientation removed; tiles without
  georeferencing are placed by their OAM name (`tiles.tile_transform`); postprocess sizes in metres
  (`PostprocessParams`), converted to pixels for decoding (`decode.DecodeParams`, `postprocess.decode_params`).
- [x] Research-based changes: histogram matching and CutMix added, `water_tint` and `stretch` removed, zoom-in-only
  scale jitter; validation stitches window predictions as at serving.
- [x] Checks run: `ruff check` / `ruff format` clean on src, tests and scripts. No environment here has pytest and the
  library dependencies together (the project `.venv` is gone; recreating it was blocked), so the test suite was **not
  run**. Instead, with the system Python 3.10 (torch, rasterio, onnxruntime; no timm): the `tiles` doctests (4 pass),
  a smoke run of serving, postprocess, scoring, the dataset (including a real training tile) and the augmentation
  preset, and ONNX export of both decoder heads at 256 px on stand-in encoder features (largest difference vs torch
  1.2e-6 UPerNet, 1.9e-7 U-Net).
- [x] Review against the repo skills (library vs `research/`): labels are GeoJSON only (fAIr gives georeferenced
  GeoTIFF chips plus one `labels.geojson`), so the mask-folder reader and placing chips by their OAM name are removed;
  the U-Net control and the border-band metric are experiments, moved to `research/unet_control.py` and
  `research/border_band.py`; the threshold sweep is no longer inside `train()` (it is `scripts/evaluate.py` on
  validation).
- [ ] The local dataset (`F:\HOTOSM\seagrass\data`) is still `images/` (not georeferenced) + `masks/` + per-tile
  `labels/`: convert each split to the fAIr layout (`chips/` georeferenced EPSG:3857 GeoTIFFs + `labels/labels.geojson`).
- [ ] Recreate the `.venv` (`uv sync --extra test`), install `timm` (asked first), run the full test suite.

### Design notes
- The ONNX graph takes fixed 256 px windows with a dynamic batch: the UPerNet pyramid pooling exports only for a
  static size, and its scales are (1, 2, 4, 8) instead of (1, 2, 3, 6) so that they divide the 8 x 8 deepest map.
- A zoom-24 tile is one window, not resampled. A coarser tile is upscaled once and cut into windows; validation and
  evaluation stitch the window predictions back exactly as serving does.

### Open decisions
- Merge polygons across chip borders in the library (needs `shapely` in the core dependencies) or leave per-chip
  polygons. The postprocessing agent recommends merging.
- Validation set: the current one is under-mapped; the data agent recommends spatial blocks from train across all four
  projects (with `spatial_split`) and keeping the current 490 chips as an audit set only.
