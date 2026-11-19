# Training-time augmentation (ss-hot v1)

Design note for seagrass segmentation (UPerNet + ConvNeXt-Tiny v1) on very-high-resolution aerial RGB. It extends
the `ts-hot` pattern (`fair_models/tree/ts-hot/src/ts_hot/augment.py`, `AugmentParams` in `params.py`): numpy +
Pillow only, image and mask changed together, mask padding = `IGNORE` (255), every step off in `AugmentParams()`.
**No albumentations**: every step below is a few lines of numpy (histogram matching is `np.interp` over CDFs), and
the library keeps its dependency footprint.

Status: proposal. Numbers marked *(design)* are starting points to ablate, not measured optima.

## 0. What the data already tells us

Measured on 80 random chips per project and split, raw `images` (not `images_processed`), every 4th pixel
(script kept outside the repo; values in 8-bit DN):

| split / project | chip mean RGB | saturation | p2–p98 luminance range | seagrass − other luminance | B/G seagrass vs other |
|---|---|---|---|---|---|
| train 55908 | 145 144 142 | 0.031 | 42 | −12 | 1.01 vs 0.97 |
| train 55909 | 146 146 147 | 0.024 | 31 | −7 | 1.02 vs 1.00 |
| train 55910 | 143 144 142 | 0.028 | 43 | −12 | 1.00 vs 0.97 |
| train 55911 | 142 143 145 | 0.029 | 37 | −8 | 1.02 vs 0.99 |
| validation 55909 | 149 150 150 | 0.026 | 21 | −3 | 1.01 vs 0.99 |
| test 55908 | 145 145 143 | 0.030 | 23 | −6 | 1.00 vs 0.97 |

Findings:

- The source image is hazy and almost grey (saturation ≈ 0.03). Mappers needed contrast ×2 and saturation ×3 to see
  seagrass (`F:\HOTOSM\seagrass\data\meta.json`, `display`). The cue the model gets is that seagrass is a little
  **darker (−3 to −12 DN, about 2–8 %)** and **slightly bluer** than sand. Augmentation must not erase that cue.
- Even inside one image, the held-out areas have about **half the contrast** of the training areas (p2–p98 range of
  21–23 vs 31–43). Contrast jitter has to reach about ×0.5.
- Labels in 55909 and 55911 are coarse block polygons, and validation is under-mapped (226 of 490 chips have no
  label; `training_layout_stats.json`). That is label noise, which augmentation cannot fix (§3).
- **Sharpness.** A box-downsample and bicubic-upsample round trip loses 10 % of the texture variance at ×2
  (1.8 cm/px), 17 % at ×4 (3.7 cm), 30 % at ×8 (7.4 cm) and 46 % at ×16 (15 cm), measured as RMSE over std. The
  imagery is sharper than serving zooms by a large margin.
- **GSD gap to serving.** For 256 px tiles at latitude 8.5° (the size `ts-hot` assumes for serving; *unverified for
  this pack*), the GSD is 7.4 / 14.8 / 29.5 cm/px at zoom 21 / 20 / 19. That is **8–32× coarser** than the training
  0.92 cm/px.

## 1. What matters, ranked

| # | Step | Why | Evidence |
|---|---|---|---|
| 1 | **Fixed model GSD** (preprocessing, not augmentation) + `scale_jitter` ±25 % | A gap of 8–32× cannot be covered by random rescaling. Deng et al. report that random scaling does not fix cross-location scale shift. Resample every input to `model_gsd_m` (the `ts-hot` design); jitter covers only the residual (latitude, native vs zoom GSD). | [Deng et al., WACV 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Deng_Scale_Aware_Adaptation_for_Land-Cover_Classification_in_Remote_Sensing_Imagery_WACV_2021_paper.pdf); `ts_hot/params.py` `MODEL_GSD_M` |
| 2 | **Low-resolution simulation** (`lowres_*`): downsample by a random factor and upsample back, image only | At serving, coarse zooms are *upsampled* to the model GSD and look blurry. The training chips are sharp at any model GSD finer than about 3 cm (table above). The `ts-hot` blur (radius 0.5–1.5 px) is too mild for a 4–6× gap. | Sharpness measurement above; Pawar et al. trained at 3 cm and applied at 12.5 cm ([Remote Sens. 17:2518](https://www.mdpi.com/2072-4292/17/14/2518)) |
| 3 | **Contrast / saturation stretch** (`stretch`) | The hazy source vs vivid other imagery, plus the internal ×0.5 contrast drop. Log-uniform contrast and saturation factors in [0.5, 2.0]. | Table above; Hakai kelp models jitter brightness, contrast and saturation ([habitat-mapper docs](https://habitat-mapper.readthedocs.io/en/stable/models/how-it-works/)) |
| 4 | **Dihedral orientation** | A nadir orthophoto has no "up". The 8 symmetries are exact (no interpolation, no padding) and free. | Hakai uses flips and rotations (same docs); `ts-hot` default |
| 5 | **Photometric** (`ts-hot` brightness/contrast/gamma ±20 %) | Exposure and tone-curve differences between OAM uploads. Applied to the whole chip, so the relative darkening cue survives. | `ts_hot/augment.py` `_photometric` |
| 6 | **Water tint / attenuation** (`water_tint`): blend toward a random blue-green water colour, red attenuated more | Depth, turbidity and tide move the image toward the water-column colour, and red fades first. Blending toward a backscatter colour is the standard image-formation model. | [Akkaynak & Treibitz, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.html) (attenuation is wavelength-dependent) |
| 7 | **Blur / JPEG** (`degrade_prob`, `ts-hot` as-is) | OAM tiles are JPEG/WebP-compressed. Compression is the last thing that happens to a tile. | `ts_hot/augment.py` `_degrade` |
| 8 | **Histogram matching to a reference bank** (phase 2) | The strongest evidence for colour shift between sources, but it needs unlabeled chips from other OAM images. | Pawar et al. 2025 (histogram matching + SegFormer, UAV 3 cm → orthophoto 12.5 cm; leave-one-out F1/IoU 0.52/0.48, about 68 % of a model trained on both domains; read from the abstract only); RHM, [Yaras et al., JSTARS 2024 / arXiv 2104.14032](https://arxiv.org/abs/2104.14032) |
| 9 | **Sun-glint speckles** (`glint_prob`): small saturated blobs/streaks, image only | Glint is a known failure mode, and bright specks already appear in 55909/55911 chips (*glint vs shells not verified*). Low priority; off by default. | [Hedley et al. 2005, IJRS 26:2107](https://doi.org/10.1080/01431160500034086) |

Not recommended:

- **Hue rotation.** In raw DN, seagrass vs sand differs mostly in brightness and a 2–4 % blue/green ratio. A hue shift
  of even ±10° can swap "bluish dark" and "yellowish bright", and the model learns colour is meaningless or, worse,
  flipped. Colour casts come only from `water_tint` (physically constrained) and histogram matching.
- **CutMix / Mosaic / copy-paste across chips.** They create artificial seagrass edges, and the labels already have
  coarse block borders (55909, 55911). They also break the spatial context (a meadow is a continuous field).
- **Arbitrary-angle rotation and elastic warps.** They add little over dihedral for nadir imagery, need interpolated
  masks, and pad corners. Hakai uses 0–45°. Revisit only if an ablation shows orientation bias.
- **Per-channel independent gain larger than ±10 %.** This is a de-facto hue shift (see above).
- **Augmenting validation or test** (see §3).

## 2. Recommended `AugmentParams`

Field names follow `ts-hot`. A scalar 0 means off. `AugmentParams()` stays the identity, so tests and other callers
are unchanged. The seagrass preset belongs in `TrainParams` defaults (as `ts-hot` copies the augment fields into
`TrainParams`) and in the pack's STAC hyperparameters. Every key the code reads is declared there.

| field | meaning | `AugmentParams()` | seagrass preset | valid range |
|---|---|---|---|---|
| `orientation` | `dihedral` / `mirror` / `none` | `none` | **`dihedral`** | enum |
| `scale_jitter` | resize by U(1−v, 1+v), crop or pad (mask pad = `IGNORE`) | 0.0 | **0.25** *(design)* | [0, 1) |
| `lowres_prob` | chance of a down/up resample of the image | 0.0 | **0.5** *(design)* | [0, 1] |
| `lowres_max` | max factor, log-uniform in [1.5, `lowres_max`] | 4.0 | **coarsest serving GSD ÷ `model_gsd_m`, capped at 8** (e.g. 6 at 5 cm) | [1.5, 16] |
| `stretch` | contrast and saturation factors, each exp(U(−v, v)), independent | 0.0 | **0.7** (→ 0.5–2.0) | [0, 1.5] |
| `photometric` | `ts-hot` brightness, contrast ×(1±v), gamma exp(±v) | 0.0 | **0.2** | [0, 0.5] |
| `water_tint` | blend weight U(0, v) toward a blue-green colour (hue 150–200°); red weight ×1.5 *(design)* | 0.0 | **0.3** *(design)* | [0, 0.6] |
| `glint_prob` | chance of adding glint speckles | 0.0 | 0.0 (0.15 in ablation) | [0, 1] |
| `degrade_prob` | `ts-hot` blur (r 0.5–1.5) or JPEG (q 40–90) | 0.0 | **0.3** | [0, 1] |
| `histmatch_prob` (phase 2) | match chip CDFs per channel to a random reference chip | 0.0 | 0.0 until a bank exists | [0, 1] |

Order (extends `ts-hot`): orientation → scale jitter → lowres → histmatch → stretch → photometric → water tint →
glint → degrade. This happens after the chip is scaled to `model_gsd_m`, as in
`ts_hot/segmentation_dataset.py`, and before model normalisation. The steps work on float32 CHW in [0, 1] and clip
after each step. The mask is touched only by orientation and scale jitter (nearest neighbour).

Dependencies on other design notes:

- If preprocessing adds a per-chip percentile stretch, much of `stretch`/`photometric` is absorbed. The steps must
  then run *before* that stretch, and both training and serving must use it.
- At 5 cm/px a 4.7 m training chip becomes about 94 px. Training crops may have to be mosaicked from neighbouring
  chips (tile x/y are in `manifest.csv`). This is a dataset/preprocessing decision, not settled here.

## 3. Label noise and held-out data

- **Never augment validation, test or serving inputs.** The `ts-hot` docstring already states this; keep it in
  `ss-hot`.
- Augmentation does not repair coarse polygons. Handle them in the loss: for example, an `IGNORE` band of a few
  pixels around label edges, or Dice + BCE as in `ts-hot`. That is a training-loss decision. Strong colour
  augmentation on noisy, low-contrast labels can push the model toward texture-only cues: watch training IoU vs
  validation IoU.
- Worst-case stacking: contrast 0.5 × JPEG q40 × water tint 0.3 can shrink a −3 DN cue below compression noise.
  Ablate the stack, not only single steps. If needed, cap the combined contrast.
- Validation (55909) is under-mapped and has a −3 DN contrast between classes. Use it for early stopping, but do not
  tune the threshold or the augmentation on it alone.
- **TTA:** optional, off by default. Dihedral TTA costs 8 forward passes, and flips-only costs 2–4, on a CPU serving
  pod. Measure the per-chip CPU latency of ConvNeXt-T/UPerNet before offering it (*not measured*). If offered, it is
  an inference parameter (`tta: none|flip|dihedral`), averaging probabilities before the threshold.

## 4. Evaluating whether a step helps

1. **Baseline:** dihedral only. Then add one step at a time in the ranked order, then the full preset. Use 3 seeds
   each and report mean ± sd of seagrass IoU/F1 (pixel) and polygon F1 after postprocessing.
2. **Clean test** (`test`, all project 55908): this measures in-domain fit only, since it comes from the same image.
3. **Shift suite (cheap proxy):** the same test chips under *fixed-seed* corruptions, not used in training: down/up
   ×4 and ×8, contrast ×0.5 and ×2, saturation ×3 (the display transform), water tint 0.3, JPEG q50, and
   `images_processed`. Report the per-corruption IoU drop. Pick steps that shrink the drop without costing more than
   about 1 IoU point on clean test.
4. **External sets:** Pawar et al. 2025 (Danish estuaries, UAV 3 cm / orthophoto 12.5 cm; *public availability and
   the "SeagrassMapper" name not verified*), and a few hand-checked OAM water scenes elsewhere with OSM
   `seamark:type=seagrass` polygons (*coverage and quality unverified*). Downloading either needs the user's approval.
5. Keep a step only if it helps the shift suite or the external set across seeds. Steps 6, 8 and 9 in §1 are the
   most speculative.

## 5. Tests the library must have (offline, tiny generated arrays)

- `AugmentParams()` returns the inputs unchanged (identity, same dtype).
- Same seed gives the same output, a different seed a different output, for every step.
- Alignment: on a synthetic image whose channel 0 equals the mask, orientation and scale jitter keep
  `image[0] == mask` wherever the mask is not `IGNORE`.
- Shape is always preserved. Mask values stay in {0, 1, 255}. Padding appears only when the factor is below 1, and
  there it is 255.
- Image-only steps (`lowres`, `stretch`, `photometric`, `water_tint`, `glint`, `degrade`, `histmatch`) never change
  the mask.
- Output is float32 in [0, 1], with no NaN, for extreme parameter values.
- `stretch` with contrast and saturation factor 1 is the identity. A grey image stays grey under saturation.
- `water_tint` moves the mean colour toward the target and lowers red more than green and blue.
- `lowres` lowers high-frequency energy (Laplacian variance) monotonically with the factor.
- The parser rejects out-of-range values, tolerates `None` and missing keys, and the preset round-trips through
  `TrainParams`.
- Importing the augmentation module pulls no torch (as in `ts-hot` `test_tuning_knobs.py`).
