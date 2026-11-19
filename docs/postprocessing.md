# Postprocessing: state of the art

2026-11-19. Scope: what the literature and open-source tools do between a per-pixel seagrass (or similar amorphous cover)
probability map from very-high-resolution RGB imagery and GeoJSON polygons for OpenStreetMap. Our data and code are not analysed here.

Tags: **[V]** read in the source (full text, official docs or abstract), **[?]** seen only in a search snippet or secondary
source, not confirmed. "Reasoning" marks an inference of this note, not a sourced claim. Numbers in brackets refer to §9.

## 1. Summary

1. **Seagrass deep-learning papers rarely describe their vector postprocess.** The drone and aerial papers read report
   pixel or point metrics (F1, IoU, OA) on rasters and say nothing about thresholds, morphology, MMU or polygonisation
   [46][47][48] [V]. Habitat-mapping programmes (non-DL) do define MMUs and post-classification filters [50][51][52] [V].
   A documented "seagrass probability to OSM polygon" pipeline was not found.
2. **Thresholds should be tuned, not assumed.** 0.5 is the common default (Microsoft footprints use >0.5 [33] [V]).
   For calibrated scores the F1-optimal threshold equals half the best attainable F1 [3] [V], so it is usually below 0.5
   when F1 < 1. Temperature scaling is the cheap standard calibration [1] [V]; it changes reported confidences, not the
   ranking of pixels (reasoning).
3. **Stitching matters more than refinement.** Errors rise towards patch borders; the recommended fix is a larger input
   at inference plus clipping 20–40 border pixels, while averaging many shifted predictions gives only small gains at
   ~15x cost [12] [V]. Overlapping windows with Gaussian or spline weighting is the default in common tooling [13][14] [V].
4. **Test-time augmentation (TTA) and CRFs give small or uncertain gains for their cost.** On ISPRS Vaihingen,
   5-scale + flip TTA added 0.22 mIoU points [11] [V]. DeepLabv3 dropped DenseCRF postprocessing [10] [V], and the
   reference DenseCRF wrapper is unmaintained [9] [?]. Neither is a clear win for CPU serving.
5. **The standard vector chain is simple and cheap:** raster sieve or minimum-area filter, then
   `rasterio.features.shapes` / GDAL polygonize, then Douglas–Peucker with topology preservation, optionally Chaikin
   smoothing [19][21][26][28][29] [V]. Building tools add orthogonalisation (fairpredictor turns it on by default [30] [V]),
   which is wrong for amorphous cover (reasoning).
6. **Cross-tile merging is solved either in raster (mosaic before polygonising) or in vector (merge within a distance).**
   HOT's fAIr-utilities does both: mosaic masks, vectorise, then merge polygons within 0.6 m [29] [V]. Independent
   simplification of neighbours breaks shared edges; coverage-aware simplification (`shapely.coverage_simplify`,
   mapshaper) avoids this [22][34] [V].
7. **Output rules are fixed by RFC 7946 and OSM:** WGS84, counter-clockwise exteriors, holes clockwise [35] [V]; OSM
   ways hold at most 2,000 nodes and holes need multipolygon relations [37][38] [V]. The documented OSM tag for seagrass is
   `seamark:type=seagrass` (331 uses); `natural=seagrass` has none [40][41] [V].
8. **Evaluation beyond pixel IoU exists but is not used for seagrass.** Boundary IoU [43], boundary F1 [44] and
   area-estimation good practice [45] are available [V]; the seagrass papers read do not report them [46][47][48] [V].

## 2. Calibration, threshold selection and uncertainty

**Methods.**
- *Fixed threshold* (0.5). Microsoft's footprint pipeline counts a pixel as building above 0.5 [33] [V].
- *Metric-optimal sweep.* Choose the threshold that maximises F1 or IoU on held-out data. Theory: for calibrated
  probabilities the F1-optimal threshold is half the optimal F1 [3] [V]; Tallam et al. cite this work in their eelgrass
  study [46] [V].
- *Calibration.* Modern networks are poorly calibrated; temperature scaling (one parameter) works well on most datasets
  [1] [V]. Local temperature scaling predicts a per-pixel temperature for segmentation [2] [V abstract]. A 2026 remote
  sensing segmentation paper applies global temperature scaling to get comparable per-class probability fields [4] [V abstract].
- *Per-image adaptive (Otsu).* Otsu assumes a bimodal histogram and can mislabel the class with the wider distribution
  [57] [?, search snippet only]. No remote sensing study comparing Otsu with a tuned fixed threshold on network
  probabilities was found.
- *Uncertainty maps.* Deep ensembles give calibrated uncertainty [6] [V]; TTA variance gives aleatoric uncertainty and
  reduces overconfident errors in medical segmentation [7] [V abstract]. A 2026 benchmark of MC dropout, sub-ensembles, TTA
  and evidential learning reports clear trade-offs between accuracy, calibration and compute, on non-remote-sensing data [5] [V abstract].

**Evidence for seagrass.** None found on threshold choice. Tallam et al. note that annotators assumed continuous beds
while the model traced patches, which created both "false" positives and missed small patches [46] [V]: label style
shifts the optimal threshold.

**CPU cost.** Threshold and temperature scaling: negligible. Ensembles and TTA: one forward pass per member or view.

**Tools.** NumPy; `uncbiag/LTS` code for local temperature scaling [2] [?].

## 3. Mask refinement

**Morphology and minimum mapping unit (MMU).**
- Opening removes specks; closing fills narrow gaps; hole filling and small-object removal enforce an MMU.
  `skimage.morphology.remove_small_objects` / `remove_small_holes` take `max_size` in pixels since 0.26 (the old
  `min_size` / `area_threshold` are deprecated) [25] [V]. `rasterio.features.sieve` and `gdal_sieve` replace regions
  below a pixel count by their largest neighbour [19] [V], [20] [?].
- Habitat programmes: the Florida Reef Tract guide recommends an MMU of 0.1 ha for all features (150 m² for patch reefs)
  and warns that seagrass vs. sediment delineation depends on the MMU: a coarse MMU yields "discontinuous seagrass", a
  fine one splits it into patches [50] [V]. The Allen Coral Atlas applies contextual object-based rules (size, depth,
  neighbours) to 5 m maps [51] [V]. A GBR seagrass dataset cleans PlanetScope maps with a 3–9 px majority filter [52] [V].
  A 2026 drone abstract uses a 20×20 px sieve after classification [53] [?].
- At drone scale, Tallam et al. digitised beds down to a 0.9 m length scale at ~3.35 cm/px [46] [V].
- Vector tools expose the same idea as `min_area` (solaris [28], fAIr-utilities 5 m² default [29], fairpredictor 3 [30],
  RAMP `remove_slivers.py -min 10.0` m² [32]) [V].

**DenseCRF.** Fully connected CRF with Gaussian edge potentials [8] [V]. DeepLabv3 reports better results than earlier
DeepLab versions without it [10] [V]. Remote sensing evidence found is older or unquantified (mixed: cleaner output,
sometimes lower accuracy) [?]. `pydensecrf` is not actively maintained; a fork `pydensecrf2` exists [9] [?].

**TTA.** Flip/rotate averaging (D4 group, 8 views) suits nadir imagery [13] [V], [17] [?]. Measured gains are small:
5 scales × flip on ISPRS Vaihingen raised mIoU from 83.27 to 83.49 and OA by 0.16 points [11] [V]; a Mars landslide
study saw D4 TTA lower mIoU slightly [18] [?]. `ttach` provides flip, D4, multi-scale wrappers and mean/gmean/tsharpen
merges [16] [V].

**Multi-scale inference and ensembling.** Covered by the HMANet result above [11] [V]; no seagrass evidence found.

**Seagrass-specific failure modes** (what refinement must fix): omissions in weak-response areas, boundary transition
zones and small scattered patches [48] [V]; false positives from macroalgae, terrestrial vegetation and shadowed water
[46] [V]. Morphology does not fix class confusion (reasoning).

**CPU cost.** Morphology, sieve: linear in pixels, cheap. TTA: ×N views. DenseCRF: iterative mean-field over all pixels,
far heavier than morphology; no CPU timing for remote sensing found [?].

## 4. Tiling and stitching

**Findings of the one systematic study** (Huang et al., Duke, INRIA building data) [12] [V]:
- Errors increase towards output patch edges, caused by zero-padding and strided layers (translational variance).
- Recommendations: (1) enlarge the network input only at inference (faster and slightly more accurate); (2) clip the
  edges of each output, 20–40 px for deep networks such as DeepLabV2; (3) averaging predictions from translated inputs
  gives small but consistent gains only with many (10+) shifts, at e.g. 15× processing.

**Overlapping windows with weighted blending.**
- MONAI `sliding_window_inference`: default overlap 0.25, `mode="constant"` (equal weights) or `"gaussian"` (less weight
  at window edges), `sigma_scale` 0.125 [14] [V].
- nnU-Net style: ≥50% overlap with Gaussian weighting [15] [?].
- Smoothly-Blend-Image-Patches: 50% overlap, second-order spline window, mirror padding, optional D4 [13] [V].

**Web-map tiles.** XYZ tiles are fixed 256 px squares, so a prediction at a tile edge has no context beyond it. The
options from the above are: fetch neighbouring tiles and predict on a mosaic (larger input), or predict with overlap and
discard borders (reasoning from [12][13]).

**CPU cost.** Overlap r multiplies forward passes by about 1/(1−r)² (50% → ~4×) (reasoning). A larger input with
border clipping costs only the margin.

## 5. Vectorisation, simplification and smoothing

**Polygonisation.** `rasterio.features.shapes` (GDAL polygonize underneath; connectivity 4 or 8, optional mask, affine
transform to map coordinates) [19] [V]; GDAL CLI default is 4-connectedness, `-8` for 8 [20] [?]. samgeo
`raster_to_vector` [27], solaris `mask_to_poly_geojson` [28] and fAIr-utilities `vectorize_mask` [29] all wrap this [V].
geomltoolkits (used by fAIr-utilities) offers potrace tracing [31] [?].

**Simplification.**
- Douglas–Peucker: `shapely.simplify(tol, preserve_topology=True)` avoids invalid output at extra cost; with
  `preserve_topology=False` holes can vanish [21] [V]. solaris and samgeo use it [27][28] [V].
- Visvalingam–Whyatt: `shapely.coverage_simplify` (Shapely ≥2.1, GEOS ≥3.12) simplifies a polygon coverage while
  keeping shared edges consistent [22] [V]. mapshaper defaults to weighted Visvalingam, simplifies shared boundaries
  identically only if vertices align, and rolls back simplification that creates intersections [34] [V].
- Independent simplification of adjacent polygons breaks shared boundaries; a 2026 parcel paper builds faces from a
  common planar graph to avoid intrusions [55] [V abstract].

**Smoothing.** Raster polygons have staircase edges. `smoothify` (Chaikin corner cutting with median of variants,
area-preserving by default, `segment_length` ≈ pixel size) targets exactly this [26] [V]; samgeo `smooth_vector` wraps
it [27] [V]. `shapelysmooth` offers Chaikin, Taubin, Catmull–Rom [?].

**Regularisation.** Right-angle regularisation (samgeo `regularize` via `buildingregulariser` [27]; fairpredictor
`orthogonalize=True` default [30]; Microsoft's footprint pipeline [33]) [V] is for buildings. Seagrass has no right
angles, so it should be off (reasoning).

**Confidence per polygon.** Microsoft averages pixel probabilities inside each polygon as the footprint confidence [33] [V].

**CPU cost.** Polygonisation and DP simplification are linear in boundary length; cheap compared with inference.
`preserve_topology=True` is slower than plain DP [21] [V]. Smoothing adds vertices; simplify afterwards (reasoning).

## 6. Merging across tiles

- **Raster mosaic first.** fAIr-utilities `polygonize` merges all mask GeoTIFFs into one raster, then vectorises [29] [V].
  Simple and seam-free, but memory grows with the request area (reasoning).
- **Vector merge.** fAIr-utilities then merges polygons within `distance_threshold` 0.6 m [29] [V]. The generic
  recipe is a spatial index, group by distance, `union_all`, re-simplify (reasoning; no paper found).
- **Shared-edge consistency.** Merge before simplifying, or simplify the merged coverage with a coverage-aware method
  [22][34] [V].
- **Instance identity.** For amorphous cover a "patch" boundary is a mapping convention: the MMU and scale decide whether
  an area is one discontinuous meadow or many patches [50] [V].
- RAMP has no documented cross-tile fusing script [32] [V].

## 7. Validity and output format

**GeoJSON (RFC 7946)** [35] [V]: WGS84 lon/lat; exterior rings counter-clockwise, holes clockwise (parsers should not
reject wrong winding); 6 decimals ≈ 10 cm (more precision costs size; 7 decimals ≈ 1 cm, reasoning); split geometries
crossing the antimeridian.

**Validity tools** [V]: `shapely.make_valid` (`linework` or `structure` method; may return a GeometryCollection) [23];
`shapely.orient_polygons` (CCW exterior by default, Shapely ≥2.1) [24]; fairpredictor exposes `make_geoms_valid=True` [30].

**Area.** `pyproj.Geod.geometry_area_perimeter` returns geodesic area in m², signed CCW-positive, holes subtracted;
orientation matters [36] [V].

**OpenStreetMap.**
- Tags: `seamark:type=seagrass` with `seamark:seagrass:category=unspecified|specified|posidonia` (seagrass split from
  weed/kelp in 2021) [40] [V]. Taginfo on 2026-11-19: `seamark:type=seagrass` 331 objects (174 ways, 3 relations),
  `wetland=seagrass` 10 ways, `natural=seagrass` 0 [41] [V]. A 2019 tagging-list thread debated `natural=seagrass` and
  `wetland=seagrass` without adoption [42] [?].
- Geometry limits: max 2,000 nodes per way, 10,000 elements per changeset [37] [V]; areas with holes must be
  multipolygon relations [38] [V]. Long simplified rings and holes therefore matter for import effort (reasoning).
- Process: imports need local community buy-in, a wiki plan and a dedicated account; humanitarian imports should
  consult HOT. The guidelines do not mention AI-generated data [39] [V].

## 8. Evaluation

**What seagrass papers report.**

| Paper | Imagery | Metrics | Postprocess described |
|---|---|---|---|
| Tallam et al. 2023 [46] | Drone RGB ~3.35 cm, DeepLab | Point-based recall 0.954, precision 0.723, F1 0.809 over >1 M random points | No |
| Pawar et al. 2025 [47] | UAV 3 cm → aerial 12.5 cm, SegFormer + domain adaptation | Per-image F1/IoU, best leave-one-out 0.52/0.48 (Oracle 0.76/0.70) | No |
| Lian et al. 2026 [48] | UAV 1.4–3.05 cm, UPerNet-ConvNeXtV2 | OA 97.45%, mIoU 94.63%, F1 97.23% | No |
| Hobley et al. 2021 [49] | Drone, semi-supervised FCNN vs OBIA | FCNN comparable to OBIA [abstract] | Not checked |
| Yamakita et al. 2026 [56] | Historic aerial to Sentinel-2, YOLO segmentation | OA ≥ 0.9; area time series [abstract] | Not stated |

All [V] for the cells shown. None reports polygon-level, boundary or area-agreement metrics.

**Available metrics.**
- *Pixel:* IoU and F1 differ systematically; Pawar et al. note IoU is harsher than F1 [47] [V].
- *Boundary:* Boundary IoU scores only pixels near the contours and is more sensitive to boundary errors on large
  objects [43] [V]. Boundary F1 (BF) from Csurka et al., who also argue for per-image scores [44] [V abstract].
- *Object/polygon:* polymetrics (used in the fAIr workflow) matches polygons by Hungarian assignment on IoU and reports
  precision, recall, F1, mIoU, mAP; object-level only [54] [?]. For amorphous cover, instance matching depends on how
  labels split meadows (reasoning, see §6).
- *Area:* Olofsson et al. give the standard for unbiased area estimation and confidence intervals from a probability
  sample, rather than pixel counting of the map [45] [V abstract].
- *MMU reporting:* the Florida guide requires mapping scale and MMU in metadata [50] [V].

## 9. Comparison table

| Method | Evidence | CPU cost | Dependency / tool | Sources |
|---|---|---|---|---|
| Fixed 0.5 threshold | Common default | Negligible | NumPy | [33] |
| F1/IoU-optimal sweep | Theory: optimum = F1*/2 if calibrated | Offline only | NumPy | [3] |
| Temperature scaling | Strong on classification; used in RS 2026 | Negligible | NumPy | [1][4] |
| Local temperature scaling | Segmentation (non-RS) | Extra small network | LTS code | [2] |
| Otsu per image | No RS comparison found | Negligible | scikit-image | [57] |
| Ensembles / TTA uncertainty | Better calibration (non-RS) | ×members / ×views | ttach | [5][6][7] |
| Opening / closing / hole fill | Standard; no gain numbers for seagrass | Linear, cheap | scikit-image, SciPy | [25] |
| Sieve / majority filter / MMU | Standard in habitat maps | Linear, cheap | rasterio, GDAL | [19][50][52] |
| DenseCRF | Dropped by DeepLabv3; RS gains unclear | High | pydensecrf (unmaintained) | [8][9][10] |
| TTA flips / D4 / multi-scale | +0.22 mIoU (Vaihingen, 10 views) | ×views | ttach | [11][16] |
| Larger input + border clip | Faster and slightly better | Margin only | Own code | [12] |
| Overlap + Gaussian/spline blend | Standard in tooling | ~1/(1−r)² passes | MONAI, Smoothly-Blend | [13][14] |
| Polygonise | Standard | Cheap | rasterio / GDAL | [19][29] |
| Douglas–Peucker (topology-preserving) | Standard | Cheap | shapely | [21][28] |
| Coverage Visvalingam | Keeps shared edges | Cheap | shapely ≥2.1, mapshaper | [22][34] |
| Chaikin smoothing | Removes staircase | Cheap, adds vertices | smoothify | [26][27] |
| Orthogonalisation | Buildings only | Cheap | fairpredictor, buildingregulariser | [27][30] |
| Mosaic then polygonise | Seam-free | Memory ∝ area | fAIr-utilities | [29] |
| Vector merge within distance | Used in fAIr (0.6 m) | Cheap with index | shapely | [29] |
| make_valid + orient | Required for RFC 7946 | Cheap | shapely ≥2.1 | [23][24][35] |
| Geodesic area | Exact on ellipsoid | Cheap | pyproj | [36] |

## 10. Gaps and open questions

1. No published seagrass study found that tunes the threshold, MMU or vector simplification, or reports polygon or
   boundary metrics. Any choice will need our own validation sweep.
2. No remote sensing study found that measures how much TTA, CRF or guided filtering helps on amorphous vegetation
   at 1–3 cm/px, or what each costs on CPU.
3. The right MMU for cm-scale seagrass mapping intended for OSM is undocumented: habitat MMUs (0.1 ha [50]) are far
   coarser than drone digitising scales (~0.9 m [46]).
4. OSM tagging for seagrass is thin (`seamark:type=seagrass`, 331 uses); no import precedent for ML-derived seagrass was
   found. Community agreement on tag and generalisation level is needed before any import [39][40][41].
5. Label style (continuous beds vs. traced patches [46]) makes pixel metrics and "optimal" thresholds depend on the
   annotators, not only on the model.
6. Polymetrics' object matching [54] is designed for discrete objects; its fitness for dissolved amorphous cover is
   unverified.
7. Unchecked: nnU-Net overlap default [15], GDAL connectivity default [20], Otsu failure-mode claim, DenseCRF remote
   sensing gains, polymetrics internals [54], geomltoolkits potrace option [31].

## 11. Sources

1. Guo, Pleiss, Sun, Weinberger. On Calibration of Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599
2. Ding, Han, Liu, Niethammer. Local Temperature Scaling for Probability Calibration. ICCV 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Ding_Local_Temperature_Scaling_for_Probability_Calibration_ICCV_2021_paper.html
3. Lipton, Elkan, Narayanaswamy. Thresholding Classifiers to Maximize F1 Score. 2014. https://arxiv.org/abs/1402.1892
4. Hauser, Dachsberger, Schmitt, Hinz. Calibrated U-Net with HELIX-Based Label Enrichment... ISPRS Annals XI-3-2026. https://isprs-annals.copernicus.org/articles/XI-3-2026/959/2026/
5. Landgraf, Hinz, Ulrich. A Critical Synthesis of Uncertainty Quantification and Foundation Models for Semantic Segmentation. ISPRS Annals XI-2-2026. https://isprs-annals.copernicus.org/articles/XI-2-2026/673/2026/
6. Lakshminarayanan, Pritzel, Blundell. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017. https://arxiv.org/abs/1612.01474
7. Wang et al. Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation. Neurocomputing 2019. https://arxiv.org/abs/1807.07356
8. Krähenbühl, Koltun. Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials. NeurIPS 2011. https://arxiv.org/abs/1210.5644
9. pydensecrf (Beyer et al.). https://github.com/lucasb-eyer/pydensecrf
10. Chen et al. Rethinking Atrous Convolution for Semantic Image Segmentation (DeepLabv3). 2017. https://arxiv.org/abs/1706.05587
11. Niu, Sun, Tian, Diao, Chen, Fu. Hybrid Multiple Attention Network for Semantic Segmentation in Aerial Images (Table X). 2020. https://arxiv.org/abs/2001.02870
12. Huang, Reichman, Collins, Bradbury, Malof. Tiling and Stitching Segmentation Output for Remote Sensing: Basic Challenges and Recommendations. 2018/2019. https://arxiv.org/abs/1805.12219
13. Chevalier (Vooban). Smoothly-Blend-Image-Patches. https://github.com/Vooban/Smoothly-Blend-Image-Patches
14. MONAI, `sliding_window_inference` source. https://github.com/Project-MONAI/MONAI/blob/dev/monai/inferers/utils.py
15. NVIDIA Deep Learning Examples, nnU-Net for PyTorch (blend/overlap options). https://catalog.ngc.nvidia.com/orgs/nvidia/dle/resources/nnunet_pyt/-/advanced
16. qubvel (P. Iakubovskii). ttach: Image Test Time Augmentation with PyTorch. https://github.com/qubvel/ttach
17. Albumentations. D4 TTA for aerial segmentation (tutorial). https://albumentations.ai/docs/examples/example-tta-d4-aerial-segmentation/
18. DualSwinFusionSeg: Multimodal Martian Landslide Segmentation. 2026. https://arxiv.org/abs/2603.14132
19. rasterio, `rasterio.features` (shapes, sieve). https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html
20. GDAL, gdal_polygonize and gdal_sieve. https://gdal.org/en/stable/programs/gdal_polygonize.html ; https://linux.die.net/man/1/gdal_sieve
21. Shapely, `simplify`. https://shapely.readthedocs.io/en/stable/reference/shapely.simplify.html
22. Shapely, `coverage_simplify`. https://shapely.readthedocs.io/en/stable/reference/shapely.coverage_simplify.html
23. Shapely, `make_valid`. https://shapely.readthedocs.io/en/stable/reference/shapely.make_valid.html
24. Shapely, `orient_polygons`. https://shapely.readthedocs.io/en/stable/reference/shapely.orient_polygons.html
25. scikit-image 0.26, `skimage.morphology`. https://scikit-image.org/docs/stable/api/skimage.morphology.html
26. Wright (DPIRD). Smoothify 0.3.3. https://pypi.org/project/smoothify/ ; https://github.com/DPIRD-DMA/Smoothify
27. Wu (opengeos). segment-geospatial (samgeo) `common` module. https://samgeo.gishub.org/common/
28. CosmiQ Works. solaris, converting model outputs to vector format. https://solaris.readthedocs.io/en/dev/tutorials/notebooks/api_mask_to_vector.html
29. HOT. fAIr-utilities `postprocessing/polygonize.py`. https://github.com/hotosm/fAIr-utilities/blob/master/hot_fair_utilities/postprocessing/polygonize.py
30. HOT. fairpredictor README. https://github.com/hotosm/fairpredictor
31. geomltoolkits (PyPI). https://pypi.org/project/geomltoolkits/
32. Dev Global Partners. RAMP, list of scripts. https://github.com/devglobalpartners/ramp-code/blob/main/docs/list-of-ramp-scripts.md
33. Microsoft. Global ML Building Footprints README. https://github.com/microsoft/GlobalMLBuildingFootprints
34. mapshaper. Simplification guide. https://mapshaper.org/docs/guides/simplification.html
35. Butler et al. RFC 7946, The GeoJSON Format. IETF 2016. https://www.rfc-editor.org/rfc/rfc7946
36. pyproj, `Geod`. https://pyproj4.github.io/pyproj/stable/api/geod.html
37. OpenStreetMap Wiki. API v0.6. https://wiki.openstreetmap.org/wiki/API_v0.6
38. OpenStreetMap Wiki. Relation:multipolygon. https://wiki.openstreetmap.org/wiki/Relation:multipolygon
39. OpenStreetMap Wiki. Import/Guidelines. https://wiki.openstreetmap.org/wiki/Import/Guidelines
40. OpenStreetMap Wiki. Seamarks/INT-1 Section J. https://wiki.openstreetmap.org/wiki/Seamarks/INT-1_Section_J
41. Taginfo, tag statistics (queried 2026-11-19 via `/api/4/tag/stats`). https://taginfo.openstreetmap.org/tags/seamark:type=seagrass ; https://taginfo.openstreetmap.org/tags/wetland=seagrass ; https://taginfo.openstreetmap.org/tags/natural=seagrass
42. OSM tagging mailing list, "eelgrass", December 2019. https://lists.openstreetmap.org/pipermail/tagging/2019-December/049757.html
43. Cheng, Girshick, Dollár, Berg, Kirillov. Boundary IoU. CVPR 2021. https://arxiv.org/abs/2103.16562 ; https://github.com/bowenc0221/boundary-iou-api
44. Csurka, Larlus, Perronnin. What is a good evaluation measure for semantic segmentation? BMVC 2013. https://bmva-archive.org.uk/bmvc/2013/Papers/paper0032/index.html
45. Olofsson et al. Good practices for estimating area and assessing accuracy of land change. RSE 148, 2014. https://doi.org/10.1016/j.rse.2014.02.015
46. Tallam et al. Application of Deep Learning for Classification of Intertidal Eelgrass from Drone-Acquired Imagery. Remote Sensing 15(9):2321, 2023. https://doi.org/10.3390/rs15092321
47. Pawar, Thomasberger, Bengtson, Pedersen, Timmermann. Eye in the Sky for Sub-Tidal Seagrass Mapping. Remote Sensing 17(14):2518, 2025. https://doi.org/10.3390/rs17142518
48. Lian et al. Intertidal Seagrass Mapping Using UAV Visible and Multispectral Imagery. Remote Sensing 18:2057, 2026. https://doi.org/10.3390/rs18122057
49. Hobley et al. Semi-Supervised Segmentation for Coastal Monitoring Seagrass Using RPA Imagery. Remote Sensing 13(9):1741, 2021. https://doi.org/10.3390/rs13091741
50. Florida Fish and Wildlife Conservation Commission (FWRI). Mapping Guide for Partners of the Florida Reef Tract. 2016. https://archive.myfwc.com/archive/Research/GIS/unifiedreefmap-mappingguide.pdf
51. Allen Coral Atlas. Science & Methods. https://allencoralatlas.org/methods/
52. Langlois, McKenzie. Seagrass meadows derived from field to spaceborne earth observation at Yule Point. PANGAEA 2022. https://doi.pangaea.de/10.1594/PANGAEA.946604
53. EGU26-18973 (abstract, UAV seagrass growth with sieve filter). 2026. https://meetingorganizer.copernicus.org/EGU26/EGU26-18973.html
54. polymetrics (PyPI). https://pypi.org/project/polymetrics/
55. Jiao, Zuo, Persello. Topologically Consistent Agricultural Parcel Vectorization... 2026. https://arxiv.org/abs/2609.07520
56. Yamakita et al. Feasibility to detect rapid change and disappearance of seagrass... ISPRS Archives L-4-W1-2026. https://isprs-archives.copernicus.org/articles/L-4-W1-2026/347/2026/
57. Understanding Important Features of Deep Learning Models for Transmission Electron Microscopy Image Segmentation (Otsu remark). 2019. https://arxiv.org/abs/1912.06077
