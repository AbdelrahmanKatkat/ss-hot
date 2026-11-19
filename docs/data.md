# ss-hot data design

Training data, split, label quality, and what the `ss-hot` dataset reader must do. Status: design, 2026-11-19.
Numbers below were computed read-only from `F:\HOTOSM\seagrass\data` unless a line says otherwise. Items marked
**(unverified)** were not checked against data or a primary source.

## 1. Dataset card

| Item | Value |
|---|---|
| Source | HF `hotosm/vhr-seagrass-segmentation` (public, sha `05ce576e376dd90658534a2988dc9d7ea650c0b2`, 2026-10-26), built with `fAIr_repos/kits-hot/tm_dataset` (`src/tm_dataset/{splits,hf_push,dataset}.py`) |
| Area | Timor-Leste, HOT Tasking Manager projects 55908-55911 (`meta.json`), four AOIs of about 28,200 m² each |
| Imagery | One OpenAerialMap image (all four projects have the same `imagery_url`), CC-BY 4.0 |
| Labels | OSM `seamark:type=seagrass` + `area=yes` polygons, ODbL 1.0 |
| Chips | 4,498 RGB TIFF chips, 512x512, uint8, **no georeferencing** (no GeoTIFF tags) |
| Ground size | ~4.72 m per chip, ~0.92 cm/px. Names say `OAM-x-y-22`, but x/y are on the **zoom-23** grid (`tm_dataset/geo.py::chip_bounds_from_indices` derives the grid zoom from x/y; the tile service is z22 `@4x`) |
| Split | Spatial: train 3,434 / validation 490 / test 498; 76 buffer chips dropped |

The local split differs from the published one: HF has validation 507 and a 59-chip buffer; the local
`scripts/build_split.py` moved 17 validation chips to a buffer row so validation and test do not touch
(`split_summary.json`: `train_touches_heldout: 0`, `validation_touches_test: 0`). Train is identical.

Geometry of the split (tile x/y from `splits.json`): train is x 7123157-7123223; the held-out block is the
eastern strip x 7123225-7123241, with a one-chip column of buffer between them. Validation is its northern part
(y 4393807-4393837, 456 chips from 55909 + 34 from 55908), test its southern part (y 4393839-4393868, all 55908).
Projects 55910 and 55911 are train only.

### On-disk contract, layout A (local split layout, written by `scripts/make_training_layout.py`)

```
<root>/<split>/images/<tile_id>.tif            raw RGB chip (train on this)
<root>/<split>/images_processed/<tile_id>.tif  display-adjusted copy (contrast 200, saturation 300); review only
<root>/<split>/masks/<tile_id>.png             uint8 class index, 0 background, 1 seagrass (checked: values {0,1})
<root>/<split>/labels/<tile_id>.geojson        seagrass polygons clipped to the chip, CRS84
<root>/<split>/manifest.csv                    tile_id, project_id, tile_x, tile_y, tile_z, num_features, west, south, east, north
<root>/splits.json                             {"method", "buffer_chips", ..., "tiles": {tile_id: split}}
```

`<split>` is `train`, `validation` or `test`. A chip's label is found by stem: `masks/<stem>.png`. A chip with
no mask file is an error, not background (every chip has one). `tile_z` in the manifest is the name's 22; the
manifest bounds are the true chip bounds. Use `images`, not `images_processed`: serving receives raw imagery,
and the processed copy only reproduces what mappers saw in iD.

### On-disk contract, layout B (generic fAIr layout, as in ts-hot)

`ts_hot/dataset.py` and `segmentation_dataset.py` read a folder of georeferenced `OAM-x-y-z.tif` chips plus
**one** labels GeoJSON (`_find_labels_geojson`: the file, or the only `.geojson` under a folder). Labels are
cut to chips in pixel space through each chip's affine and CRS (`build_chip_labels`), rasterised after scaling
the chip to the model resolution (`scale_chip_and_labels`), with `IGNORE = 255` for pixels left out of the loss.
This is the layout the pack's training step receives, so ss-hot must support it. Two consequences:

- The seagrass chips cannot be read through layout B as they are: they have no georeference, and
  `ts_hot.tiles.parse_oam_tile` rejects `OAM-7123157-4393811-22` because x is outside the zoom-22 grid. That
  rejection is useful; ss-hot should keep it.
- Converting the local data to layout B (georeferenced GeoTIFFs named with z=23 + one merged labels GeoJSON)
  belongs in `research/`, not `src/`. It is optional if the reader also accepts layout A.

## 2. Statistics

Seagrass pixel fraction per chip, chip counts per bucket (computed from `masks/*.png`).

| Split | Chips | Pixel share | Median chip | 0 | (0,5]% | (5,25] | (25,50] | (50,75] | (75,95] | >95 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 3,434 | 59.3% | 64.3% | 132 | 46 | 193 | 709 | 1,247 | 1,014 | 93 |
| validation | 490 | 14.0% | 1.7% | 226 | 47 | 98 | 78 | 34 | 7 | 0 |
| test | 498 | 49.1% | 46.9% | 1 | 6 | 81 | 185 | 132 | 75 | 18 |

| Split / project | Chips | Mean chip fraction | Empty | ≥75% |
|---|---|---|---|---|
| train 55908 | 574 | 44.7% | 1 | 80 |
| train 55909 | 643 | 44.3% | 127 | 117 |
| train 55910 | 1,094 | 63.8% | 0 | 406 |
| train 55911 | 1,123 | 71.0% | 4 | 504 |
| validation 55908 | 34 | 29.5% | 3 | 0 |
| validation 55909 | 456 | 12.9% | 223 | 7 |
| test 55908 | 498 | 49.1% | 1 | 93 |

"Wall-to-wall" is a fair description of 55910/55911 (37-45% of chips ≥75% seagrass) but only 93 train chips
(2.7%) are >95%, so most training chips still contain both classes.

Raw OSM polygons per project (raw exports `F:\HOTOSM\seagrass\HOTOSM\hotosm_project_5590x_seagrass_polygons_geojson`,
2026-08-10, de-duplicated by OSM id, area by a local equirectangular approximation):

| Project | Polygons | Median m² | p10-p90 m² | <1 m² | ≥25 m² | AOI covered |
|---|---|---|---|---|---|---|
| 55908 | 7,021 | 0.48 | 0.08-3.79 | 71% | 75 | 45% |
| 55909 | 2,264 | 1.71 | 0.32-10.4 | 36% | 38 | 31% |
| 55910 | 6,119 | 1.92 | 0.35-6.92 | 31% | 5 | 63% |
| 55911 | 3,135 | 2.98 | 0.48-13.9 | 22% | 131 | 70% |

Clipped label features per chip: median 6 (train), 1 (validation), 4 (test). Raw RGB mean is similar across
splits (0.56-0.59), but the validation chips have much lower contrast (per-channel std 0.024-0.033 vs
0.041-0.053 in train), which says the validation area also looks different, not only that it is labelled less.

## 3. Label-quality risks

**R1. Validation is under-mapped (high impact).** 226 of 490 validation chips (46%) have no label and 273
(56%) have <5% seagrass; 268 of those are in 55909. All empty chips lie fully inside their project AOI, and
every one is empty in the raw OSM export too (`label_validation_summary.json`), so this is how the area was
mapped, not a build error. Within 55909 the eastern (validation) part has a mean chip fraction of 12.9% against
44.3% in its own western (train) part. 101 empty validation chips are 5-20 m from the nearest mapped polygon,
i.e. they sit in large unmapped gaps. Visual review (`label_review.pdf`, not re-read here) shows dark mottled
seagrass-like texture there. The lower image contrast leaves room for some genuine absence (deeper water or
bare sand) **(unverified)**.
*Effect:* on validation, correct seagrass predictions in unlabelled areas count as false positives. Precision
and IoU are depressed, and checkpoint selection, early stopping and any threshold tuned on validation favour a
model that under-predicts. Validation numbers cannot be compared with test numbers.

**R2. Under-mapping also reaches train.** 127 of the 132 empty train chips are in 55909. If they hide seagrass,
they teach "this texture is background" **(unverified; these chips were not reviewed one by one)**.

**R3. Labelling style differs by team.** 55908 traced many tiny patches (median 0.48 m², 71% under 1 m²);
55911 drew larger blobs (median 2.98 m², 131 polygons ≥25 m²) and covers 70% of its AOI against 45% for
55908. Test is 100% 55908 (fine style), while 65% of train is 55910/55911 (coarse style).
*Effect:* a model trained mostly on coarse labels merges patches and fills gaps; on the fine-style test it
will lose boundary precision. Test IoU partly measures style agreement, not only detection.

**R4. Temporal drift.** 6 train chips in 55911 differ from the 2026-08-10 export by >5 percentage points. Two
(`OAM-7123194-4393807-22`, `OAM-7123195-4393807-22`) were 81% and 76% seagrass in August and are 0.2% and 0.4%
in the build (2026-10-15): their polygons were removed or edited in OSM in between. The other four change by
7-12 points. Mean absolute mask-vs-export difference is under 0.5 points per split, so drift is local.

**R5. Single image, single site.** One OAM image, one date, one coast. Any score is in-domain; nothing in the
dataset measures transfer to other imagery, water clarity or species.

## 4. Recommendations

1. **Validation.** Do not select checkpoints on the current validation split. Options, in order of preference:
   - *Remap* the north-east of 55909 in Tasking Manager (validation tasks plus the empty train chips), then rebuild
     with the same scripts. This is the only option that fixes the labels; it needs mapper time.
   - *Short term:* take validation from **train** as whole spatial blocks spread over all four projects (about
     10% of train, ts-hot `spatial_split` style), drop train chips adjacent to them (one-chip buffer, like the
     dataset), keep test unchanged, and keep the current 490 chips as a separate **audit** set reported
     per project and by chip coverage, never used for selection.
   - *Exclude* the 268 55909 validation chips under 5%: cheap, but what remains is biased towards labelled
     chips and still contains partial under-mapping. Not recommended alone.
   - *Re-cut the held-out strip along x*: rejected. It moves under-mapped chips into test.
   - *Ignore index:* useful for training (below) but not for evaluation: masking all unlabelled pixels of the
     audit area leaves only seagrass pixels and makes precision undefined.
2. **Uncertain training areas.** Support `IGNORE = 255` in masks. A `research/` list or GeoJSON of uncertain
   areas (55909 empty train chips after review, the two drift chips until checked) becomes ignore pixels or is
   dropped by name. Do not drop all empty chips blindly: 5 of 132 are outside 55909 and are likely true negatives.
3. **Class balance.** Train is 59% seagrass, test 49%: background is the minority, not seagrass. No class
   weights or oversampling by default. Report IoU for both classes and per-project scores so a "predict
   seagrass everywhere" bias is visible.
4. **Style.** Report test results per chip-coverage bucket and optionally at a coarser tolerance (e.g.
   boundary-tolerant F1); do not tune to the fine 55908 style alone.
5. **External test set** (evaluation only, no training, no download yet):
   - *SeagrassMapper*, HF `CoastalEcology/SeagrassMapper`: CC-BY-4.0, Danish eelgrass (*Zostera marina*), UAV
     ~3 cm and aircraft 12.5 cm GSD, 512x512 PNG, binary masks 1 = eelgrass (dataset README; Pawar et al.,
     *Remote Sensing* 17(14) 2518, 2025). Licence-compatible; tests transfer to other species and water.
     Its native GSD is coarser than ours, so it only fits if the model input GSD is set accordingly.
   - *MagicBathyNet*, Zenodo 10470959: **CC BY-NC 4.0** (non-commercial), 5.9 GB, Agia Napa (Cyprus) and Puck
     Lagoon (Poland), aerial ~0.25 m/px (720 px per 180 m patch), Agia Napa classes seagrass / rock / macroalgae /
     sand / undefined (github.com/pagraf/MagicBathyNet). Good for macroalgae/rock confusion; internal evaluation
     only because of NC, and 25 cm may be too coarse for the model.
   - Best option **(unverified availability)**: a second HOT project with `seamark:type=seagrass` on different
     OAM imagery, built with `tm_dataset`.
6. **Ground resolution.** Training chips are 0.92 cm/px; fAIr serving tiles are likely much coarser **(unverified)**. The
   model input GSD is a preprocessing decision, but the reader must scale chips and masks to it (below).

## 5. What the library provides (minimal)

Follow ts-hot names and keep everything dataset-specific in `research/`.

- Constants `BACKGROUND = 0`, `SEAGRASS = 1`, `IGNORE = 255` (fixed, not parameters).
- `list_chip_paths(root)`, as in ts-hot.
- `spatial_split(names, val_ratio, seed, *, block_size=4)`, as in ts-hot, for the platform path where the step
  gets one dataset and splits it itself. Optional addition: drop train chips touching validation (one-chip
  buffer). Research runs instead pass fixed name lists from `splits.json` through `names=`.
- Label loading with two sources, chosen by the `labels` argument:
  - a **masks folder** (layout A): `<labels>/<chip stem>.png`, uint8 values in {0, 1, 255}, same size as the
    chip; anything else raises;
  - a **labels GeoJSON** (layout B): as ts-hot `build_chip_labels`, polygons only (no point discs), chips must
    be georeferenced; an optional second GeoJSON of ignore polygons for research use.
- `scale_chip_and_labels`: scale the chip exactly as at serving; masks from PNG are resized with nearest
  neighbour, GeoJSON labels are rasterised at the scaled size.
- `SegmentationDataset(images_dir, labels, pre, *, names=None, augment=False, seed=0)` returning
  `(float32 (3,H,W), uint8 (H,W))`, in the `train` extra.
- Name parsing keeps rejecting x/y outside the named zoom grid. For layout A, the scale comes from the chip
  size and the true grid zoom (or manifest bounds), not from the `-22` suffix **(decision needed: derive the
  zoom in the library, or rename chips to z23 in `research/`)**.
- Params: `val_ratio` and `split_seed` with ts-hot defaults (0.2, 42) unless a decision changes them. No
  `drop_empty`, class-weight or ignore-index parameters.

Offline tests: a tiny generated chip + PNG mask with 0/1/255, a mask with an invalid value, a missing mask,
a z22-named chip with z23 indices, and a two-polygon GeoJSON on a georeferenced fixture chip.
