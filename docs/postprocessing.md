# Postprocessing: seagrass probability to GeoJSON

Design note for `ss-hot` (UPerNet + ConvNeXt-Tiny v1), written 2026-11-19. Tags: **[V]** verified in code or data,
**[P]** proposal to confirm with a sweep, **[?]** unverified.

## 0. Contract and reference

- The pack's `predict(session, input_images, params)` returns a GeoJSON `FeatureCollection` that one function serves for
  live serving, batch `run_inference` and tests (`fAIr-models/docs/contributing/model.md`, "Inference flow") [V].
  The library supplies `predict_session` (preprocess + ONNX, returning probability + `PreprocessMeta` per chip) and
  `postprocess_outputs` (decode + georeference). The pack's `postprocess` calls `postprocess_outputs` and renames the properties to
  the STAC variables, as `models/deepforest_unet_tree/pipeline.py` does (`class`, `label`, `confidence`, `source`) [V].
- Reference: `fair_models/tree/ts-hot/src/ts_hot/{decode,georef,postprocess,params}.py` and `docs/Theory.md` §5.
  `ss-hot` already mirrors it (`src/ss_hot/decode.py`, `georef.py`, `postprocess.py`) [V]. All of it carries over. Three
  things change for seagrass: the defaults (§2), a cross-chip seam merge (§3) and an `area_m2` property (§4).
- The platform validates predictions against the STAC (`fair/stac/validators.py::validate_predictions_geojson`), then
  writes GeoJSON, FlatGeobuf and PMTiles (`fAIr/backend/predictions/post_run.py`). It does not merge, dissolve or clip [V].

## 1. Label statistics behind the defaults

The source is the four raw OSM exports `F:\HOTOSM\seagrass\HOTOSM\hotosm_project_5590{8,9,10,11}_seagrass_polygons_geojson\seagrass.geojson`.
They hold 18,591 features, 52 of which repeat an OSM id across projects, leaving 18,542 polygon parts. All parts are
valid. Areas were measured in UTM 51S (EPSG:32751) [V, computed with a read-only one-off script].

| Statistic | p5 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|
| Polygon area (m²) | 0.09 | 0.41 | 1.20 | 3.54 | 11.6 | 192 |
| Max inscribed circle diameter (m) | 0.23 | 0.46 | 0.74 | 1.23 | 2.34 | n/a |
| Vertex spacing (m) | 0.10 | 0.17 | 0.25 | 0.36 | 0.73 | n/a |
| Gap to nearest other polygon (m) | 0 | 0 | 0.06 | 0.13 | 0.41 | n/a |

- **Small, patchy objects.** Polygons under 0.25 m² are 16% of all polygons but hold only 0.7% of the area. Polygons
  under 1 m² are 45% of the polygons and 5.9% of the area. The total is 5.96 ha.
- **Holes are almost never mapped.** Only 5 holes appear, in 2 polygons, and they are 0.26 to 1.54 m². Volunteers map bare sand by leaving it out
  between patches, not with inner rings.
- **Instance boundaries are arbitrary.** 26% of polygons touch a neighbour and 67% have one within 10 cm.
  483 pairs overlap. One meadow is often drawn as several pieces.
- **Exterior orientation is mixed** (52% counter-clockwise), so the output must set its own ring orientation.
- **Training chips are much sharper than serving tiles.** A training chip is 4.72 m across at about 0.92 cm/px. A fAIr
  chip is 256 px (model.md), so a source pixel is 0.295, 0.148, 0.074 or 0.037 m at zoom 19, 20, 21 or 22
  (latitude 8.5°S). At `MODEL_GSD_M = 0.03` (`src/ss_hot/params.py`, not final), one model pixel is 3 cm, so one source
  pixel spans 9.8, 4.9, 2.5 or 1.2 model pixels at those zooms.

## 2. Stages and `PostprocessParams`

**The ONNX graph contains** the ImageNet normalisation, UPerNet (logits at stride 4, then bilinear upsampling to the input) and
the **sigmoid** (`export.py`, `model.py`) [V]. Keeping the sigmoid in the graph makes `threshold` and `confidence`
probabilities. The threshold, argmax or morphology must never go into the graph, so that they stay tunable without a re-export.

**Numpy runs the rest, in model pixels with no CRS (`decode.py`):** smooth, then threshold, then open, then close, then
trace (4-connectivity, holes kept as inner rings), then drop polygons under the minimum area, then fill holes, then
simplify (Douglas-Peucker), then compute confidence as the mean probability. **Map space (`georef.py`,
`postprocess.py`)** takes over from there: divide by the scale to get chip pixels, apply the affine transform into the
chip CRS, merge across seams (§3), convert to EPSG:4326, then orient and validate the rings.

Store each default in metres and derive the pixel value from `MODEL_GSD_M`, so a change of GSD keeps the ground meaning:

| Key | Default (px at 0.03 m) | Ground | Why |
|---|---|---|---|
| `threshold` (alias `confidence_threshold`) | 0.5 | n/a | Placeholder until the §5 sweep [P]. ts-hot's tuned value (0.3) does not transfer |
| `smooth_sigma_px` | 0 | n/a | Logits are already bilinear from stride 4, and serving inputs are upsampled 2.5 to 10 times. Turn smoothing on only if the sweep shows speckle |
| `open_px` | 1 | removes parts narrower than 9 cm | Narrower than 99% of labels (p1 inscribed diameter 0.16 m) [P] |
| `close_px` | 2 | fills gaps narrower than 15 cm | 67% of label gaps are 10 cm or less (digitising slack). This is about 1 to 2 source pixels at zoom 20 to 21 [P] |
| `min_area_px` | 278 | 0.25 m² | Costs 0.7% of label area (16% of label polygons by count). At zoom 19 it is about 3 source pixels. The 1,100 px (1 m²) now in `params.py` would drop 45% of label polygons [P] |
| `fill_holes_px` | 278 | 0.25 m² | Labels have almost no holes. Use the same floor as `min_area_px`: no object and no gap below 0.25 m² |
| `simplify_px` | 2.0 | 6 cm | Removes the 3 cm pixel staircase and stays below the mappers' vertex spacing (p5 = 0.10 m) [P] |

`confidence_threshold` is a compulsory STAC key (model.md, "Hyperparameters"), so the alias is required. Here it means
the per-pixel probability threshold. It does not filter polygons by score. Do not add a separate polygon-score filter
without evidence that one is needed.

The non-overlapping 512 px windows (`preprocess.py::to_windows`) create seams inside a chip. Blending or overlapping
windows belongs to the inference design, not here [?].

## 3. Chip borders: merge in the library

What the existing code does:

- The platform: no merge [V].
- ts-hot and `unet_tree_segmentation`: one polygon set per chip, no merge (ts-hot's Theory.md §5.2 leaves it as an open question) [V].
- The legacy `fairpredictor/predictor/app.py` and `hot_fair_utilities/postprocessing/polygonize.py` mosaic the masks
  before vectorising. `polygonize.py` then merges polygons within 0.6 m [V].
- `dinov3_hot/serve.py` mosaics the chips before inference [V].

Seagrass needs the merge. Label polygons cross a 256 px tile edge in these shares:

| Zoom (tile width) | Polygons crossing | Area in crossing polygons |
|---|---|---|
| 19 (75.6 m) | 1.6% | 4.9% |
| 20 (37.8 m) | 4.5% | 10.6% |
| 21 (18.9 m) | 8.5% | 19.3% |
| 22 (9.45 m) | 15.6% | 32.8% |
| Training chip (4.72 m) | 45.5% | 78.5% |

Predicted meadows, after closing, are larger than single OSM pieces, so they cross even more often. 99.5% of non-empty
training masks touch their chip edge. Without a merge, every request shows tile-grid seams and inflated polygon counts.

**Recommendation:**

1. Keep decoding per chip. That bounds memory: a 20 × 20 tile request at zoom 20 would be a 25,200² px float mosaic,
   about 2.5 GB, at 3 cm.
2. Add a vector **seam merge** in `postprocess_outputs`:
   1. Flag the polygons that touch the chip edge during decoding, before simplification.
   2. Convert them to the chips' common projected CRS.
   3. Group them by `dwithin(merge_distance_m)` with an STRtree.
   4. Union each group (`union_all(buffer(+d/2)).buffer(-d/2)`), then re-simplify.
   5. Set confidence to the area-weighted mean of the parts.

   Polygons that do not touch an edge pass through unchanged.
3. Set `merge_distance_m` to 0.05 by default (above the float and simplification slack, below the label gap median) [P].
   Add `merge_chips` (default true) to switch the merge off.
4. Add `shapely` as a core dependency of `ss-hot`. The serving image already gets it through `fair-py-ops[serve]`, then
   `geomltoolkits`, then `shapely` (from `pyproject.toml` files; not checked in a built image) [?].
5. Read ground metres in EPSG:3857 as projected metres × cos(latitude). That is negligible at 8.5°S but not at higher
   latitudes [V maths].

## 4. Output features and validity

- **One `Polygon` per feature**, with holes. Split any `MultiPolygon` produced by `make_valid` or the union into separate
  features. The validator accepts both types for the `polygon` keyword [V], but the reference packs emit `Polygon`.
- **Properties emitted by the library:**
  - `confidence`: float in [0, 1], 4 decimals.
  - `class_id`: 1.
  - `class_name`: `"seagrass"`.
  - `area_m2`: geodesic area from `pyproj.Geod(ellps="WGS84")`, 2 decimals. It adds no new dependency.

  The pack maps these to `class`, `label`, `confidence` and `area_m2`, and declares every one of them in `cube:variables`.
  The integration test fails on a missing declared variable (model.md, "Output Requirements"). Drop `source`, since a
  merged polygon spans several chips; if it is kept, declare it as an auxiliary variable.
- **Validity rules:**
  - EPSG:4326 `[lon, lat]` coordinates, with no `crs` member.
  - Closed rings of at least 4 positions, with no NaN.
  - Exterior counter-clockwise and holes clockwise, as RFC 7946 §3.1.6 requires. The pixel-to-map y-flip reverses ring
    orientation, so orient explicitly at the end.
  - `shapely.is_valid` on every feature, with `make_valid` and keeping only the polygonal parts as the fallback.
  - Rounding to 7 decimals (OSM's 1.1 cm precision) is optional, with validity checked again after rounding [P].
- **Empty input:** no pixels above the threshold gives `{"type": "FeatureCollection", "features": []}`, never `None` and
  never an error.
- **Chips without a CRS:** a PNG or JPEG chip stays in pixel coordinates, as in ts-hot. fAIr serving fetches
  georeferenced tiles (`fair/serve/base.py` with `georeference=True`) [V].

## 5. Tuning the threshold despite under-mapped validation

The validation split is 14% seagrass and test is 49% (`F:\HOTOSM\seagrass\data\training_layout_stats.json`). Even
counting only non-empty chips, validation is 26% against 49% for test. 226 of the 490 validation chips are empty
although they lie inside the project AOI, and 220 of them are within 20 m of a mapped polygon
(`label_validation_summary.json`). False positives on validation are therefore partly unmapped seagrass, and the
F1-optimal threshold there is biased upwards.

1. **Tune on a cleaned subset.** Drop the 226 empty validation chips, hand-review the rest (`label_review.pdf`) and
   keep only chips whose labels look complete. If that is not possible, hold out spatial blocks of train (59%, close to
   test) as a tuning set. Never tune on test.
2. **Sweep the whole postprocess** through the ONNX session, not only the pixel threshold: thresholds 0.2 to 0.8 in steps
   of 0.05, with `min_area_px`, `open_px` and `close_px` on. Pick the centre of the F1 plateau, not the argmax.
3. **Report three sets:** raw validation, cleaned validation and test (once, at the chosen setting).
4. **Pixel metrics:** IoU, F1, precision and recall, micro-averaged over pixels. The key names must equal
   `fair:metrics_spec` (for example `pixel_iou`).
5. **Polygon metrics:** label instances are arbitrary, so **dissolve** labels and predictions first, then report:
   - vector IoU of the dissolved areas, which is what users receive;
   - component recall: the share of dissolved label components of 0.25 m² or more that are at least 50% covered;
   - component precision: the share of predicted polygons whose area is at least 50% inside labels;
   - boundary F-score at 0.25 m (the median label vertex spacing);
   - fragmentation: polygons per hectare and median polygon area, compared with the labels.

   Whether polymetrics (the finetune-and-score skill) fits dissolved seagrass is not verified [?].

## 6. Tests (offline, tiny synthetic arrays)

1. The parameter parser handles `None`, missing keys and the `confidence_threshold` alias, rejects negative values, and
   keeps the metre-derived defaults consistent with `MODEL_GSD_M`.
2. The threshold is inclusive. An all-zero probability gives an empty FeatureCollection.
3. Opening removes a 1 px speck and a 2 px bridge. Closing fills a 3 px gap but leaves the chip edge uneroded.
4. A polygon smaller than `min_area_px` is dropped. A hole smaller than `fill_holes_px` is filled; a larger hole stays
   as a clockwise inner ring.
5. Simplifying keeps every dropped vertex within the tolerance, and a collapsing ring is dropped.
6. Confidence is the mean probability inside the polygon, excluding its holes. For merged polygons it is area-weighted.
7. Georeferencing: model pixels to chip pixels (the scale), then the affine transform, then EPSG:4326, checked against
   known corner coordinates.
8. **Seam merge:** two adjacent synthetic GeoTIFF chips with one square spanning their shared edge give one feature
   with the summed area. Two separate squares stay two features. `merge_chips=False` gives two features.
9. Every feature is a valid `Polygon` with a counter-clockwise exterior, `class_name == "seagrass"`, confidence in
   [0, 1], and `area_m2` within 1% of the analytic area.
10. The output passes `fair.stac.validators.validate_predictions_geojson` (pack test). The pack constants match the
    STAC `classification:classes`.
11. Importing `ss_hot.postprocess` does not import torch.
