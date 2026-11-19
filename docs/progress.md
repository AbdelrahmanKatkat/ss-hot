# ss-hot progress

Seagrass semantic segmentation helper library for HOT fAIr (UPerNet + ConvNeXt-Tiny v1), built on the pattern of
`fair_models/tree/ts-hot`. Newest entries at the bottom.

## 2026-11-19

### Decisions so far
- Model: UPerNet decoder + ConvNeXt-Tiny **v1** encoder (timm `convnext_tiny.fb_in22k_ft_in1k`, Apache-2.0). ConvNeXt-V2
  ImageNet weights are CC-BY-NC and are not used. Baseline to compare against: U-Net + ResNet34 / EfficientNet.
- Repository layout, packaging and public API follow `ts-hot` (helper library; the fAIr pack stays a thin adapter).
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
  stitch back), `serve` (`predict_session`), `decode`/`georef` (from ts-hot), `postprocess` (EPSG:4326 FeatureCollection,
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
- [ ] Install `timm` (asked first) and run `tests/test_model.py` (shape, freezing, ONNX parity, smoke training).

### Design notes
- The ONNX graph takes fixed 512 px windows with a dynamic batch: the UPerNet pyramid pooling exports only for a
  static size, and its scales are (1, 2, 4, 8) instead of (1, 2, 3, 6) so that they divide the 16 x 16 deepest map.
- Chips that become small at the model resolution are joined with their tile-grid neighbours into one mosaic before a
  512 px training window is cut, so windows hold real surroundings, not padding; missing neighbours are `IGNORE`.
- The prepared seagrass chips need `source_gsd_m` ≈ 0.0092 m (4.72 m / 512 px) because their names say z22 while their
  x/y are on the z23 grid.

### Open decisions
- Merge polygons across chip borders in the library (needs `shapely` in the core dependencies) or leave per-chip
  polygons as ts-hot does. The postprocessing agent recommends merging.
- Validation set: the current one is under-mapped; the data agent recommends spatial blocks from train across all four
  projects (with `spatial_split`) and keeping the current 490 chips as an audit set only.
- Rename the prepared chips to `-23` (true zoom) in the data scripts, or keep passing `source_gsd_m`.
