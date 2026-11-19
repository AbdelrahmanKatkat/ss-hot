# ss-hot research

State of the art for each part of a seagrass segmentation model on fAIr, from five research agents (2026-11-19).
Each topic has its own report with the sources; this page gathers the findings and what they mean for `ss-hot`.
Claims marked **[V]** in the reports were read in the source, **[?]** only second-hand.

| Topic | Report | Sources |
|---|---|---|
| Datasets and data practice | [`data.md`](data.md) | 52 |
| Preprocessing | [`preprocessing.md`](preprocessing.md) | 35 |
| Augmentation | [`augmentation.md`](augmentation.md) | 43 |
| Postprocessing | [`postprocessing.md`](postprocessing.md) | 57 |
| Models | [`model.md`](model.md) | 35 |

Setting the reports were asked about: RGB-only drone orthomosaics from OpenAerialMap, shallow tropical water
(Timor-Leste), 256 px web-map tiles (training tiles zoom 24, about 0.92 cm per pixel), binary segmentation, ONNX on
CPU, GeoJSON polygons for OpenStreetMap, weights and code that allow commercial use. (Update 2026-11-20:
non-commercial licences are acceptable after all, so the licence findings below no longer rule anything out.)

## 1. Datasets

- **No public dataset matches our setting** (tropical, overhead, 1–3 cm, seagrass masks, open licence) except HOT's
  own `hotosm/vhr-seagrass-segmentation` and `fAIr-guru-td-seagrass`.
- **Closest outside set:** DTU `CoastalEcology/SeagrassMapper` (2025, CC BY 4.0): drone ~3 cm and aircraft ~12.5 cm
  tiles with eelgrass masks, Denmark; RGB not confirmed. Useful as an outside test.
- **Not usable commercially:** MagicBathyNet (CC BY-NC-SA, 533 annotated patches), DeepSeagrass and SeagrassFinder
  (CC BY-NC-ND; underwater side views anyway).
- **Unlabelled imagery for pre-training:** KAUST Red Sea reef orthomosaics (CC0); Great Barrier Reef drone maps
  (CC BY 4.0, raw imagery unconfirmed); Seatizen Atlas (licence statements conflict).
- **Weak labels / checks only:** Allen Coral Atlas (5 m), Digital Earth Pacific (10 m), both CC BY 4.0; UNEP-WCMC
  needs permission for commercial use; OSM has 331 `seamark:type=seagrass` objects worldwide.
- **Practice:** split by site or by imagery, not at random (random hold-outs overstated accuracy by up to 28% in one
  drone study [?]); conservative "label only what you are sure of" annotation; point and image labels propagated
  with superpixels or DINOv2 are an established weak-label route.

## 2. Preprocessing

- Classic water corrections need what RGB drone tiles lack: Hedley deglinting needs near-infrared, the Lyzenga
  depth-invariant index needs one substrate at several depths. No evidence that either helps a deep net on RGB.
- Sun glint on RGB drone imagery is handled at capture (sun angle, wind, polarisers) or with learned glint masks
  (66–81% of glint pixels found); removal appears to need several frames, so not on one served tile.
- Underwater colour correction as preprocessing: mixed (hurt a detection benchmark, small gains on underwater
  seagrass video); not tested for aerial seagrass.
- **Normalise as the encoder was pre-trained** (ImageNet for ConvNeXt).
- Camera, date and site differences: handle with augmentation or histogram matching, not fixed corrections.
- **Train and serve at one ground resolution**; web-map tile resolution changes with latitude and a train/test
  mismatch hurts (size of the effect from a synthetic urban benchmark).
- **Tile borders:** larger inference windows, or 50% overlap with Gaussian/spline blending, reduce border artefacts
  (about 4x the model runs).
- No drone-scale evidence for extra input channels (RGB indices, depth).

## 3. Augmentation

- **No seagrass augmentation ablation exists**; every number is from another domain.
- Flips and 90° rotations: universal baseline, about 0–0.3 mIoU on their own; 8-view test-time augmentation about
  zero.
- Brightness, contrast, gamma help marine imagery more than hue or saturation; no source gives a safe hue range for
  seagrass.
- **Histogram matching** is the best-tested fix for site and sensor shift (eelgrass UAV → aircraft, F1 0.52, 68% of
  an oracle), but gains vary and it can hurt.
- **Mixing** (CutMix, Copy-Paste, ClassMix) gives the largest single gains in remote sensing (+2 to +6 mIoU).
- **Semi-supervised training** with unlabelled imagery and strong augmentation is the strongest lever (+7 to +17
  points with few labels).
- Untested: down-then-up resampling as preparation for coarser serving imagery; physical water simulation
  (attenuation, turbidity, glint) for segmentation.
- Licences: albumentations is MIT only up to 2.0.8 (successor AGPL/commercial); kornia, torchvision, TorchGeo are
  permissive.

## 4. Postprocessing

- No seagrass paper describes a probability-to-polygon pipeline; they report pixel or point metrics only.
- **Threshold:** choose it on validation (F1-optimal sweep), do not assume 0.5; temperature scaling is the cheap
  calibration.
- **Tile borders are the main error source;** overlapping windows with Gaussian weighting is the common default. TTA
  and DenseCRF give small or unclear gains for their CPU cost.
- Vector chain: remove small regions, polygonise (rasterio/GDAL), topology-preserving simplification, optional
  Chaikin smoothing. Do not square corners (a building-model default in fairpredictor).
- Across tiles: fAIr-utilities stitches masks into one raster, then merges polygons within 0.6 m;
  `shapely.coverage_simplify` keeps shared edges consistent.
- Output: RFC 7946 (counter-clockwise outer rings), 6 decimals ≈ 10 cm; an OSM way holds at most 2,000 nodes; holes
  need multipolygon relations; the documented tag is `seamark:type=seagrass` (`natural=seagrass` is unused).
- Habitat-mapping minimum mapping units are coarse (0.1 ha in a Florida guide) next to drone digitising (~0.9 m).

## 5. Models

- No benchmark matches our setting. Closest: Lian 2026 (UAV intertidal, 9-channel input): UPerNet + ConvNeXtV2-T
  94.63% mIoU vs U-Net (same encoder) 90.13% and DeepLabV3+-R101 88.88%; no RGB-only result.
- Other RGB drone seagrass studies: FPN beat U-Net (Chen & Sasaki 2021, IoU 0.85 [?]); U-Net ranked first of four
  (Jeon 2021); DeepLab F1 0.81 (Tallam 2023).
- Tiny CNNs and transformers are about equal on high-resolution aerial data (FLAIR-HUB, UPerNet head: ConvNeXtV2-T
  62.7, MiT-B2 62.7, Swin-T 62.2).
- Remote-sensing foundation models do not reliably beat a supervised U-Net (PANGAEA) and are mostly pre-trained at
  10–30 m.
- Licences: SegFormer/MiT weights are NVIDIA non-commercial; ConvNeXt V2 ImageNet weights are CC BY-NC; ConvNeXt v1
  is Apache-2.0 / MIT. Non-commercial is acceptable for this project (2026-11-20), so ConvNeXt V2-T, the encoder of
  the Lian 2026 result, is an option to try against v1.
- ONNX: UPerNet's pyramid pooling with bins (1, 2, 3, 6) fails on an 8×8 map (256 px input); bins must divide it.
- Shortlist: (1) UPerNet + ConvNeXt-Tiny v1 (current), (2) U-Net + ConvNeXt-T as the control, (3) DINOv3 ConvNeXt-T
  backbone (licence needs review), (4) UPerNet + Swin-T, (5) SAM2-UNet (research only). Keep (1) if it beats the
  U-Net control on held-out sites and its 256 px ONNX export matches PyTorch.

## 6. What this means for ss-hot

The model stays UPerNet + ConvNeXt-Tiny v1. Applied on 2026-11-20 (see `progress.md`):

**Forced by the dataset fix (256 px zoom-24 tiles, correct names):**
- The model takes 256 px tiles as they are; the ONNX input is 256 px. Our pooling bins (1, 2, 4, 8) divide the 8×8
  map, which avoids the export failure in §5.
- `model_gsd_m` is the training resolution (0.0092 m); a tile within 2% of it is not resampled. `source_gsd_m` is
  gone.
- Chip joining, 512 px windows and the 255 ignore value are gone: one tile is one sample, masks are 0/1. A scaled
  tile larger than 256 px is cropped at random for training; a smaller one is mirrored, image and mask together.
- Labels are one GeoJSON per split over georeferenced GeoTIFF chips, as fAIr provides them (no mask folders).
- Postprocess sizes are in m² / m (`min_area_m2`, `fill_holes_m2`, `simplify_m`, `open_m`, `close_m`), turned into
  model pixels with `model_gsd_m`; decoding itself stays in pixels.
- Removed: `MAX_INPUT_PX`, `smooth_sigma_px` (Gaussian smoothing of the probability before the threshold; off by
  default and no source supports it), the "mirror" orientation. Kept on request: `freeze_encoder`, `unfreeze_from`,
  `encoder_lr_factor`, `epoch_fraction`.

**Serving zoom (answered):** fAIr recommends a zoom for seagrass (the training zoom 24), but users can send coarser
tiles. Those are upscaled once to 0.92 cm (zoom 23 x 2, zoom 22 x 4; zoom 21 and coarser are refused, `MAX_SCALE`
5), so `lowres` stays in training (probability 0.3, factor up to 5).

**From the research:**
- Augmentation: quarter turns and mirrors, brightness/contrast/gamma, blur/JPEG, zoom-in jitter (1 to 1.25), and two
  new steps against a second random training tile: randomised histogram matching (probability 0.3) and CutMix
  (probability 0.5). `water_tint` and the contrast/saturation stretch are gone. The probabilities are starting values,
  not tuned.
- The serving threshold comes from the F1 sweep of `scripts/evaluate.py` on the validation chips.
- Experiments in `research/` (git-ignored, not in the package): `border_band.py` scores the 16 px band along window
  edges against the centre, to measure tile-edge error; `unet_control.py` trains a U-Net + ConvNeXt-T control with the
  library's training loop, to justify UPerNet.
- The README states that the test split shares imagery with training; an outside test (SeagrassMapper) needs a
  download.

**Later:** overlapping-window serving across neighbouring tiles (if the border metric shows it matters);
semi-supervised training on unlabelled drone imagery; merging polygons across tiles (shapely); ConvNeXt V2-T as the
next encoder to try (its licence is acceptable now).
