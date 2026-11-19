# ss-hot research

Five research agents worked in parallel with the build, one per topic. Each wrote a full report; this file collects
their findings, what the library took from them, and what is still open. Numbers marked *(est.)* are estimates or
proposals the agents could not measure; everything else they computed from the data or read from a source cited in
the full report.

| Topic | Full report | Status in the code |
|---|---|---|
| Preprocessing | [preprocessing.md](preprocessing.md) | 3 cm, `MAX_SCALE` 5, raw images, ImageNet in-graph done; window 512 kept (reason below) |
| Data | [data.md](data.md) | reader, `spatial_split`, ignore value done; validation set is an open decision |
| Postprocessing | [postprocessing.md](postprocessing.md) | defaults, ring winding, `area_m2` done; cross-chip merge open |
| Augmentation | [augmentation.md](augmentation.md) | preset in `TrainParams` done; histogram matching is phase 2 |
| Model | [model.md](model.md) | recipe done; hand-written UPerNet head kept instead of smp (reason below) |

---

## 1. Preprocessing

Findings:
- Training chips are 0.923 cm/px (all 4,422 on the zoom-23 grid although named z22). The source OAM image is
  1.128 cm/px (TiTiler `/cog/info`: 461.00 m over 40,852 px, UTM 51S), so the chips are slightly upsampled. fAIr's
  finest 256 px chip, zoom 22, is 3.69 cm/px at −8.53°; zoom 20 is 14.8 cm, zoom 19 29.5 cm.
- Shrinking the training chips to 3 cm loses only 3–5% of their variance; Pillow's bilinear resize smooths correctly
  when shrinking (a fine checkerboard averages out instead of aliasing).
- The display adjustment of `images_processed` reproduces bit for bit with Pillow 12.3.0, but its contrast pivots on
  each chip's own mean and its sharpening works at source-pixel scale, so it would mean something different on serving
  chips; it also clips up to 0.5% of pixels and does not separate seagrass consistently better.
- Raw chips have a per-channel std of only 0.024–0.051; dataset statistics would bake a ~x5 gain from one image into
  the model, and per-chip standardisation amplifies noise on plain sand or water.

Taken into the code:
- `model_gsd_m` = **0.03 m** for training and serving; one Pillow float bilinear resize in both; ImageNet mean/std
  (the timm values for this encoder) fixed inside the ONNX graph.
- **`MAX_SCALE` = 5** (was 16): zoom 20 is accepted (x4.9), zoom 19 refused (x9.8); shrinking is not limited;
  `max_input_px` 2048. `lowres_max` = 5 so the augmentation reaches the coarsest accepted zoom.
- Train and serve on raw `images`; low contrast is left to the `stretch` augmentation, not a preprocessing step.
- Masks: GeoJSON is rasterised straight onto the scaled grid; mask images are resized by area majority; padding and
  missing neighbours are `IGNORE`. The effective scale (scaled size / chip size) maps predictions back.

Not taken, and why:
- **`WINDOW_PX` 256** (proposed so that a lone 157 px chip fills 38% of a window instead of 9%): the library joins
  neighbouring chips into mosaics, so a 512 window holds 3 x 3 chips (471 px, ~85% filled) and gives the pyramid
  pooling more context; a 256 window would hold one chip. Cost per ground area is the same.
- **Ground resolution from the GeoTIFF georeferencing** (before the chip name): fAIr serving chips are named and
  georeferenced consistently, and the prepared training chips have no georeferencing, so this would not change
  anything today; `source_gsd_m` covers the training data. Worth adding if a dataset with wrong names but correct
  georeferencing appears.
- The z22-name-on-z23-grid problem cannot be detected in general (a z23 index below 2²² looks like a valid z22 tile);
  the fix belongs in the data converter: rename to `-23` and write georeferencing from the manifest bounds. Optional:
  stitch complete zoom-22 parent tiles into 1024 px chips (2,972 / 416 / 436 in train / validation / test).

Deciding experiment *(not run)*: downsample the test chips to zoom 22 / 21 / 20 resolution and compare `model_gsd_m`
0.02, 0.03 and 0.037.

## 2. Data ([data.md](data.md))

Findings:
- Provenance: HF `hotosm/vhr-seagrass-segmentation` (CC-BY-4.0 + ODbL), one OAM image, TM projects 55908–55911,
  OSM `seamark:type=seagrass`. Local split (`F:\HOTOSM\seagrass\data`): train 3,434 / validation 490 / test 498, with
  76 buffer chips; train is identical to HF, 17 HF validation chips moved into the val/test buffer.
- Seagrass pixel share: train 59.3%, validation 14.0%, test 49.1%. Under-mapping is in the **eastern half of 55909**
  only (west, train: 44.3% per chip; east, validation: 12.9%). 127 of the 132 empty train chips are also 55909.
- Validation imagery has much lower contrast (per-channel std 0.024–0.033 vs 0.041–0.053 in train), so part of the
  "empty" area may be sand or deeper water *(unverified)*.
- Labelling style differs by team: 55908 traced small patches (median 0.48 m²), 55911 drew large blobs (median
  2.98 m², 70% of its AOI covered vs 45% for 55908). Test is all 55908; 65% of train is 55910/55911.
- Two chips drifted badly between the Aug-10 export and the Oct-15 build (`OAM-7123194-4393807`, `OAM-7123195-4393807`:
  81% / 76% seagrass then, 0.2% / 0.4% now).

Recommendations and what the library does:
- Don't select checkpoints on the current validation set (it rewards under-prediction). Short term: take validation as
  spatial blocks from train across all four projects (~10%, one-chip buffer) → **`dataset.spatial_split` added**; keep
  the 490 chips as an audit set only. Long term: remap north-east 55909 in the Tasking Manager. Re-cutting along x is
  rejected (moves under-mapped chips into test).
- Support ignore = 255 in masks for training → **done** (mask values 0/1/255; padding and missing neighbours are 255).
  Don't use ignore masking for evaluation.
- No class weights or oversampling by default (seagrass is the majority class). Don't drop empty chips blindly.
- External evaluation-only sets: SeagrassMapper (CC-BY-4.0) first; MagicBathyNet (CC BY-NC 4.0) internal only.
- Reader accepts layout A (`images/` + `masks/`) and layout B (georeferenced chips + one GeoJSON) → **done**
  (`dataset.LabelSource`); train on `images`, not `images_processed`.

## 3. Postprocessing ([postprocessing.md](postprocessing.md))

Label statistics (18,542 raw OSM polygon parts): median area 1.2 m² (p5 0.09, p95 11.6, max 192), median width
0.74 m, median vertex spacing 0.25 m, only 5 holes; 26% of polygons touch a neighbour.

Taken into `PostprocessParams` (model pixels at 3 cm):

| Key | Default | On the ground |
|---|---|---|
| `threshold` (alias `confidence_threshold`) | 0.5 until a sweep | — |
| `open_px` | 1 | removes parts narrower than 9 cm |
| `close_px` | 2 | fills gaps narrower than 15 cm |
| `min_area_px` | 278 | 0.25 m² (drops 16% of labelled polygons but 0.7% of their area; 1 m² would drop 45%) |
| `fill_holes_px` | 278 | 0.25 m² |
| `simplify_px` | 2 | 6 cm |

Also done: RFC 7946 ring winding (exterior counter-clockwise, holes clockwise), geodesic `area_m2` per feature, empty
prediction → empty FeatureCollection.

Open:
- **Merge polygons across chip borders.** The platform (`fAIr/backend/predictions/post_run.py`) and ts-hot do not;
  label polygons cross a 256 px tile edge 1.6% / 4.5% / 8.5% of the time at z19 / z20 / z21 (4.9% / 10.6% / 19.3% of
  the area). The agent recommends merging edge-touching polygons in the library (`merge_chips`, `merge_distance_m`
  0.05 *(est.)*), which needs `shapely` in the core dependencies (probably already in the serving image via
  `fair-py-ops[serve]` *(unverified)*). Not implemented yet: a new dependency needs a decision.
- Threshold: tune on a cleaned validation subset or held-out train blocks, never on test; pick the middle of the F1
  plateau of a 0.2–0.8 sweep (`evaluate.score_*` with `DEFAULT_THRESHOLDS`, `best_threshold`).

## 4. Augmentation ([augmentation.md](augmentation.md))

Findings: the raw image is hazy and nearly grey (saturation ≈ 0.03); seagrass is only 3–12 DN darker and slightly bluer
than sand; held-out areas have about half the contrast of training areas. A round trip down-and-up loses 30% of the
texture at 7.4 cm and 46% at 15 cm, so ts-hot's blur is far too mild for the serving gap.

Taken into the code (`AugmentParams` stays the identity; the seagrass preset is the `TrainParams` default):

| Step | Key | Preset |
|---|---|---|
| quarter turns + mirror | `orientation` | `dihedral` |
| brightness / contrast / gamma | `photometric` | 0.2 |
| contrast and saturation x exp(±s) (x0.5–x2) | `stretch` | 0.7 |
| blend toward blue-green water, red fading most | `water_tint` | 0.3 |
| resize ±, crop or pad with ignore | `scale_jitter` | 0.25 |
| blur or JPEG | `degrade_prob` | 0.3 |
| down by 1–`lowres_max` and back up | `lowres_prob`, `lowres_max` | 0.5, 4.0 |

Not done, by recommendation: hue rotation (can flip the darker/bluer vs brighter/yellower cue), CutMix/Mosaic (labels
have coarse block borders), arbitrary rotations or elastic warps, any augmentation of validation/test/serving, test-time
augmentation (8 passes on CPU). Phase 2: histogram matching to unlabelled chips from other OAM images (needs that
imagery, which needs your OK). Evaluation plan: 3 seeds per added step, plus a fixed-seed shift suite on test.

## 5. Model ([model.md](model.md))

Findings:
- Versions on PyPI (2026-11-19): timm 1.0.31 (Apache-2.0), torch 2.15.0, onnx 1.23.2, onnxruntime 1.32.0,
  segmentation-models-pytorch 0.6.0 (MIT). ConvNeXt v1 `fb_in22k_ft_in1k` weights are Apache-2.0; ConvNeXt-V2 ImageNet
  weights are CC-BY-NC-4.0 and excluded.
- smp's UPerNet accepts `tu-convnext_tiny.fb_in22k_ft_in1k`, but its pyramid pooling bins are hard-coded to (1, 2, 3, 6),
  and the legacy ONNX exporter only exports adaptive pooling when the bins divide the static feature size (16 x 16 for a
  512 window); 3 and 6 fail (confirmed in the torch 2.15 source). smp's fix (PR #1270) covers PSPNet only.
- Size *(est.)*: about 37 M parameters and ~150 GFLOPs at 512² with a 256-channel decoder; ONNX about 150 MB; CPU
  onnxruntime 0.8–1.5 s per 512 window *(not measured)*. GPU memory with AMP at 512: batch 8 ≈ 8–10 GB *(est.)*.

Decisions in the code:
- **Hand-written UPerNet head on a timm ConvNeXt-Tiny feature extractor** (`model.py`, ~100 lines) instead of smp: it
  exports with pooling bins (1, 2, 4, 8) without patching smp internals, and keeps smp out of the dependencies. The
  agent preferred smp because it would also build the U-Net-ResNet34 baseline; if that baseline is wanted, smp can be
  added to the `train` extra then. The 1 x 1 pooling branch has no BatchNorm (a batch of one would fail otherwise).
- ONNX: fixed 512 px windows, dynamic batch, opset 17, legacy exporter (`dynamo=False`), parity test < 1e-4.
- Recipe (ConvNeXt / mmseg, adapted to 3,434 chips): AdamW lr 1e-4, weight decay 0.05 (none on 1-D parameters),
  stage-wise layer decay 0.9 (`layer_decay`; stage i x 0.9^(4−i), stem x 0.9^5 — mmseg's exact exponents differ by a
  constant factor at most, *unverified*), warmup 400 steps then cosine, gradient clipping 1.0, drop path 0.1, AMP
  (bf16, else fp16 with a gradient scaler), channels-last, 40 epochs, patience 8, monitor validation IoU, no freezing.
- Loss: BCE + Dice over non-ignored pixels; no focal loss (seagrass is the majority); Lovasz only as a later stage.

Still unverified (from the agent): the 4–6 mIoU margin of UPerNet-ConvNeXt over U-Net/DeepLabV3+ (abstract only, and
that study used multispectral bands); pinning the pretrained weights to a Hugging Face commit; real CPU latency.
