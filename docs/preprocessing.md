# Preprocessing: RGB chip to model tensor

Design note for `ss-hot` v1 (UPerNet + timm `convnext_tiny.fb_in22k_ft_in1k`). It covers training and serving.
Tags: **[V]** verified here (command or file cited), **[P]** proposal, **[?]** unverified.
Scripts were run read-only on `F:\HOTOSM\seagrass\data` with the `tm_dataset` venv (Pillow 12.3.0, numpy). The
sample was 200 evenly spaced chips per split; `chip_stats.py` is kept outside the repo.

## 0. Contract in one table

| Step | Where | What |
|---|---|---|
| Read | `preprocess.read_chip` | Bands 1-3 as uint8 HWC RGB, plus transform and CRS. Other bands are ignored |
| Ground resolution | `tiles` | `chip_gsd_m`: override, else georeferencing, else the name (section 1.4) |
| Scale | `preprocess.scale_rgb` | float32 CHW in [0, 1], **one** Pillow bilinear resize by `chip_gsd_m / model_gsd_m`. Antialiased when shrinking |
| Windows | `preprocess.to_windows` | Symmetric padding at the bottom and right, cut into `WINDOW_PX` windows |
| Normalise | **inside ONNX** | ImageNet mean and std, then the network, then a sigmoid |
| Back to the chip | `stitch_windows` + `PreprocessMeta` | Crop the padding, then divide by the effective scale, then apply the chip transform |

No colour or display adjustment is applied anywhere (section 2).

## 1. Input resolution

### 1.1 Facts

| Imagery (latitude −8.53°) | Pixels | cm/px | Scale at `model_gsd_m = 0.03` | Scaled side |
|---|---|---|---|---|
| Training chip, true grid z23 | 512 | **0.923** [V] | 0.308 | 157 |
| Same chip read as "z22" from its name | 512 | 1.846 (wrong by 2×) | | |
| Source COG, native | | **1.128** [V] | 0.376 | |
| fAIr chip z22 | 256 | 3.691 | 1.23 | 315 |
| fAIr chip z21 | 256 | 7.382 | 2.46 | 630 |
| fAIr chip z20 | 256 | 14.764 | 4.92 | 1260 |
| fAIr chip z19 | 256 | 29.53 | 9.84 | refused (section 1.4) |

- **z23 grid.** The manifest `west` value of all 4,422 chips equals the west edge of the zoom-23 tile `(x, y)`. One chip
  spans 4.2915e-5° = 360/2²³ [V, `{split}/manifest.csv`]. The cause is in `tm_dataset/tiles.py::oam_tile_id`: a z22
  `@4x` tile (1024 px) is split into 512 px chips, which are indexed on grid zoom 23 but keep `z = 22` in the name.
- **Native GSD.** The TiTiler `/cog/info` of the project imagery returns bounds 461.00 m × 384.97 m for
  40,852 × 34,114 px in EPSG:32751. That is 1.128 cm/px [V]. The imagery field of
  [TM project 55908](https://tasking-manager-tm4-production-api.hotosm.org/api/v2/projects/55908/) points to
  `oin-hotosm-temp/.../6a029b68ef395bce007be43b.tif`. The training chips are therefore only slightly upsampled
  (×1.22) from the source.
- **What fAIr serves.** fAIr serves 256 × 256 chips ([`fAIr-models/docs/contributing/model.md`](../../../../fAIr_repos/fAIr-models/docs/contributing/model.md),
  "fAIr chips are 256*256"). Every pack's `fair:preview` zoom has a maximum of 22, and the recommended zoom is
  between 18 and 20 [V, `models/*/stac-item.json`]. Whether the platform ever serves zoom 23 or higher is
  unverified [?]. The finest chip the model can expect is therefore about 3.7 cm/px at this latitude, and
  3.2 cm/px at 30°.
- **Detail lost by coarsening.** I area-downsampled the raw chips and upsampled them back (100 chips per split,
  median MSE as a fraction of chip variance). The loss is 1.6–2.7 % at ×2, 2.9–4.6 % at ×3.25 (3 cm), 4.2–6.1 % at
  ×4 (3.7 cm) and 9.7–14.8 % at ×8 (7.4 cm) [V]. This agrees with the RMSE/std figures in
  [`augmentation.md`](augmentation.md) §0 (17 % at ×4, 30 % at ×8). Little of the content lies at scales finer
  than about 3 cm.
- **Published drone seagrass work.** Pawar et al. (2025) trained at about 3 cm and applied the model to 12.5 cm
  imagery ([Remote Sens. 17:2518](https://www.mdpi.com/2072-4292/17/14/2518); cited from [`data.md`](data.md), not
  re-read). A 2023 object-based study used GSDs of 8.22 mm (30 m flight) and 27.41 mm (100 m flight)
  ([Remote Sens. 15:3600](https://www.mdpi.com/2072-4292/15/14/3600); search abstract only, the page returned 403).
  Ellis et al. (2021) found accuracy fell by about 1 % per metre of flight altitude
  ([doi:10.1002/rse2.160](https://researchportal.plymouth.ac.uk/en/publications/influence-of-altitude-on-tropical-marine-habitat-classification-u/)).
  Jeon et al. (2020) used imagery at 10 cm or finer
  ([KJRS 36(2-1)](https://koreascience.kr/article/JAKO202013363975488.page)). I could not verify the brief's
  1.4–3.7 cm range end to end [?].

### 1.2 Decision [P]

**`model_gsd_m = 0.03`.**

- It is the coarsest value that still keeps z22 serving chips close to the model GSD (×1.23 at the tropics).
- It matches the published 3 cm practice.
- It discards only about 3–5 % of training-chip variance.
- At 0.92 cm, every serving chip would be upsampled 4–32× and would never look like training data.
- At 0.037 m (exactly z22), anything finer, or a higher latitude, would be downsampled, which is harmless. But
  z21 and z20 would be upsampled further than at 0.03.
- Keep 0.03 unless the resolution sweep in section 5 shows otherwise.

### 1.3 Scale limits [P]

- **Upscale limit.** `MAX_SCALE = 5.0` replaces the current 16 in `src/ss_hot/params.py`. It accepts z20 everywhere
  (4.98 at the equator) and refuses z19 below about 60° latitude. Beyond that factor the chip has no detail at the
  model scale (×8 already loses 10–15 % of the variance, and z19 is ×9.8). The low-resolution augmentation
  (`lowres_max`, [`augmentation.md`](augmentation.md)) should reach at least `MAX_SCALE`, so that every accepted
  zoom was simulated in training. This is a provisional number, to be confirmed by the sweep.
- **Downscale.** There is no limit: Pillow's bilinear resize antialiases. A 1-px checkerboard shrunk 512→157 gives
  mean 0.500 and std 0.001, while nearest-neighbour gives std 0.5 [V].
- **Input size.** Keep `max_input_px = 2048`. With `MAX_SCALE = 5` it never binds for 256 px chips (1280 px at most).
  It only bounds memory for unusually large chips.

### 1.4 Where the chip GSD comes from [P]

`chip_gsd_m(name, width, height, transform, crs, source_gsd_m)` tries three sources in order:

1. `source_gsd_m` (exists in the current draft). It is an explicit override, for research only.
2. **Georeferencing**, when `crs` is set. Transform the bounds to EPSG:4326 and compute
   `(east − west) × 40075016.686/360 × cos(lat_centre) / width`. This works for EPSG:4326, 3857 or UTM chips; the
   pack fixtures use both 4326 and 3857 [V, `models/*/tests/conftest.py`].
3. **Name** `OAM-x-y-z` (`tile_gsd_m`, as in ts-hot).

If both 2 and 3 are available and differ by more than 1 %, use 2 and log a warning.

**The z22/z23 mismatch cannot be detected from the name.** `parse_oam_tile` rejects these chips only because
x ≥ 2²² here. A z23 index below 2²² (west or north of the zoom-22 midpoint) would pass silently with a GSD 2× too
coarse. The library therefore must not guess the grid zoom, and the `chip_bounds_from_indices` heuristic stays in
`research/`.

The fix belongs to dataset onboarding in `research/`:

- Rename each chip to `OAM-x-y-23`, which is then truthful.
- Write GeoTIFF georeferencing from the manifest bounds (EPSG:4326).

Training and serving then share one code path, with the same GSD derivation. Optionally, the 743 / 104 / 109
complete z22 parents (2,972 / 416 / 436 chips in train / validation / test) can be stitched into 1024 px
`OAM-x-y-22` tiles. A stitched tile has exactly the footprint of a z22 serving chip (9.45 m) [V, manifest count].

## 2. Raw `images` or display-adjusted `images_processed`

| Split | Variant | Mean R G B | Std R G B | Fisher R G B (seagrass vs rest) | % px at 255 |
|---|---|---|---|---|---|
| train | raw | .563 .565 .566 | .051 .049 .040 | .43 .41 .27 | 0.001 |
| train | processed | .557 .565 .571 | .100 .091 .072 | .43 .40 **.003** | 0.51 |
| validation | raw | .582 .587 .584 | .033 .032 .024 | .10 .09 .003 | 0.003 |
| validation | processed | .565 .595 .576 | .074 .058 .052 | .16 .12 .26 | 0.031 |
| test | raw | .569 .567 .558 | .041 .036 .028 | .39 .39 .14 | 0.001 |
| test | processed | .581 .567 .517 | .089 .062 .065 | .46 .52 .16 | 0.165 |

Values are in [0, 1]. Fisher = (μ_seagrass − μ_other)² / (σ²_seagrass + σ²_other), per pixel against the mask [V].

**Reproducibility.** Applying Pillow `Contrast(2.0) → Color(3.0) → Sharpness(1.35)` to raw chips reproduces
`images_processed` bit for bit on all 600 sampled chips [V, Pillow 12.3.0; other versions unverified]. Reproducible
does not mean equivalent at serving, though:

- `Contrast` pivots on the chip's own mean luminance, so the result depends on the chip footprint (4.7 m in
  training, 9.4 m for a z22 serving chip).
- `Sharpness` is a 3×3 kernel at source-pixel scale, so it means something different at 0.92 cm and at 7.4 cm.
- The adjustment saturates up to 0.5 % of pixels, and it does not raise separability consistently: it removes the
  blue cue in train and adds one in validation.

**Decision [P]: train and serve on raw `images`.**

- The ×2 contrast is a near-linear per-chip change, which the first convolution can absorb.
- The held-out areas already have about half the contrast of train ([`augmentation.md`](augmentation.md) §0).
- The variation is therefore covered with photometric augmentation (contrast down to about ×0.5 and up to ×2,
  saturation jitter), not with a fixed display step.
- `images_processed` is review material only, consistent with [`data.md`](data.md).

## 3. Normalisation

| Option | For | Against |
|---|---|---|
| **ImageNet constants in the graph** (current `model.py`) | Matches the timm pretrained config (mean 0.485/0.456/0.406, std 0.229/0.224/0.225 [V, HF `config.json`]). Same as ts-hot. Train and serve cannot disagree | Raw chips are low-contrast (std 0.024–0.051), so the normalised inputs span only about ±0.2 |
| Dataset statistics | Gives more contrast to the encoder | They come from one OAM acquisition at one site: in effect a fixed ×5 gain. A normal-contrast image elsewhere would arrive 5× too hot |
| Per-chip standardisation | Invariant to haze and contrast | Depends on the footprint (as `Contrast` does), amplifies noise on uniform sand or water chips, and removes the brightness cue (seagrass is 2–8 % darker) |

**Decision [P]: ImageNet constants, fixed, inside the ONNX graph.** Low contrast is handled by full fine-tuning and
contrast augmentation. Revisit only if a measured run shows a gain on held-out data.

## 4. Size, padding, interpolation, metadata

- **Image.** One float32 bilinear resize with Pillow mode `F` per channel, the same function in training and
  serving. Do not use torch `interpolate`: it does not antialias by default.
- **Mask.**
  - Rasterise GeoJSON labels directly on the scaled grid (`new_w × new_h` over the chip bounds). No mask is resampled.
  - PNG masks use nearest-neighbour, so `IGNORE = 255` is never blended.
  - Image and mask must use the same rounded size, so the pixel-centre conventions agree.
- **Windows [P]: `WINDOW_PX = 256`, replacing 512.**
  - The UPerNet pyramid pooling needs a static size, and 256/32 = 8 is divisible by the bins (1, 2, 4, 8)
    ([`model.md`](model.md) §4).
  - A training chip (157 px) fills 38 % of a 256 window and 9 % of a 512 window.
  - A z22 chip (315 px) costs the same either way, and a z21 chip (630 px) wastes less with 256.
- **Padding.** Symmetric reflection at the bottom and right, as now. In training, padded mask pixels are `IGNORE`.
  A scaled chip larger than a window is randomly cropped to one window in training and tiled at serving.
  Seams between windows are not blended in v1 [?].
- **`PreprocessMeta`.** Keep `width`, `height`, `resized_width`, `resized_height`, `rows`, `cols`, `transform` and
  `crs`, and add `gsd_m` and `gsd_source`. Mapping back uses the **effective** scale `resized/original`
  (157/512 = 0.3066), not the requested 0.3076.

## 5. Parameters and tests

| `PreprocessParams` | Default | Range | Note |
|---|---|---|---|
| `model_gsd_m` | 0.03 | [0.005, 0.30] | Must match in training and serving; declared in the STAC |
| `max_input_px` | 2048 | [256, 8192] | Memory bound only |

The constants are not parameters: `MAX_SCALE = 5.0`, `WINDOW_PX = 256`, `IMAGE_MEAN`, `IMAGE_STD`, and
`GSD_MISMATCH_TOL = 0.01`. `source_gsd_m` is a keyword argument, not a STAC hyperparameter.

**Offline tests**

1. **GSD by name.**
   - `OAM-x-y-22` at 256 px and latitude −8.53° gives 0.03691 m.
   - z23 at 512 px gives 0.009227 m.
2. **GSD by georeferencing.** EPSG:4326 and EPSG:3857 fixtures agree with the name to within 0.1 %.
3. **Precedence.**
   - override > georeferencing > name.
   - A mismatch logs a warning and the georeferencing is used.
   - No georeferencing, no OAM name and no override raises an error.
4. **Regression.** A z23 tile named `-22` without georeferencing: the library does not silently return 0.0092 m.
   The research converter's output does return it.
5. **Scale.**
   - A scale above `MAX_SCALE` raises an error.
   - `max_input_px` caps the scale.
   - The scaled side is at least 1 px.
   - The effective scale is the one stored in the metadata.
6. **`scale_rgb`.**
   - Output is float32 CHW in [0, 1].
   - A red-only chip stays in channel 0.
   - Two runs are identical.
   - A checkerboard shrunk 4× has std below 0.01 (antialiasing guard).
7. **No display step.** `scale_rgb` at scale 1 equals `rgb / 255` exactly.
8. **Masks.**
   - A rasterised square aligns with the scaled image: IoU ≥ 0.99.
   - Nearest-neighbour keeps the values in {0, 1, 255}.
   - Padding is `IGNORE`.
9. **Windows.** Padding and stitching round-trip exactly for 157, 256, 315 and 630 px inputs.
10. **Parity.** A dataset item equals `preprocess_chip` for the same chip and parameters.
11. **ONNX** (`train` extra):
    - Normalisation is inside the graph.
    - ONNX matches torch on a 256 window.
12. **Parsers.**
    - `None` or missing keys give the defaults.
    - Booleans are rejected.
    - Values out of range raise an error.

**Acceptance experiment [P], not a unit test.** On the test split:

1. Area-downsample the chips to 3.69, 7.38 and 14.76 cm (z22, z21 and z20 at 256 px).
2. Run the full `preprocess_chip` path and report IoU at each level.
3. Do the same for candidate values `model_gsd_m` ∈ {0.02, 0.03, 0.037}.

`model_gsd_m` and `MAX_SCALE` are fixed by this sweep.
