# Datasets and data practice: state of the art

Date: 2026-11-19. Scope: public datasets and derived products usable to train, pre-train or validate RGB
seagrass segmentation on very-high-resolution (VHR, ~1–3 cm/px) drone/aerial orthomosaics of shallow tropical
water, and the data practice (splits, label noise, imbalance, annotation, pre-training) found in recent papers.
Desk research only: nothing was downloaded, no HOT data or code was analysed.

Markers: **[V]** read on the dataset page, record or paper; **[?]** taken from a search-engine summary or a
secondary page, not confirmed on the primary page. "Licence not found" means no licence statement was found;
it does not mean the data are free to use. Numbers in brackets `[n]` refer to the Sources list.

---

## 1. Summary

**Datasets most useful to us**

1. **HOT's own data** stay the only public VHR (zoom 22), RGB, tropical, overhead seagrass segmentation set
   found: `hotosm/vhr-seagrass-segmentation` (4,498 chips, Timor-Leste, CC BY 4.0 imagery + ODbL labels) [1]
   and its sibling `hotosm/fAIr-guru-td-seagrass` (4,497 tiles) [2]. No other source matches all of RGB +
   1–3 cm + overhead + tropical + pixel/polygon labels + open licence.
2. **CoastalEcology/SeagrassMapper** (DTU, 2025): 512×512 aerial tiles, UAV ~3 cm and aircraft ~12.5 cm GSD,
   binary eelgrass masks, Denmark, **CC BY 4.0** [3][4]. Closest external match in GSD, label type and licence;
   temperate eelgrass, not tropical. Best candidate for an external out-of-domain test or extra training data.
3. **MagicBathyNet** (2024): aerial ~6.3–6.8 cm + SPOT-6 + Sentinel-2 triplets, Cyprus and Poland, 5 seabed
   classes including *Posidonia* seagrass; **CC BY-NC-SA 4.0** (non-commercial) [5]. Few annotated patches.
4. **Drone orthomosaics without seagrass labels, open licence**, usable for self-supervised pre-training or
   unlabelled domain adaptation: KAUST Red Sea reef UAV orthomosaics (**CC0**) [13]; Seatizen Atlas
   UAV/ASV imagery, Réunion and SW Indian Ocean (metadata CC BY 4.0, but text says annotated images are
   for non-commercial use) [11]; Langlois & McKenzie Great Barrier Reef seagrass drone maps (**CC BY 4.0**;
   whether the raw orthomosaics are included is unconfirmed) [6][7].
5. **Weak labels / validation**: Allen Coral Atlas v2.0 benthic map with a *Seagrass* class, 5 m, tropics
   (**CC BY 4.0**) [32]; Digital Earth Pacific seagrass extent, Sentinel-2 10 m, yearly from 2017 (**CC BY
   4.0**) [35]; UNEP-WCMC Global Distribution of Seagrasses v7.1 (UNEP-WCMC General Data License; commercial
   use needs permission) [34]. All are 170–500× coarser than our imagery: useful to pick sites, sample
   negatives and sanity-check outputs, not as pixel labels.
6. **Underwater sets** (DeepSeagrass, SeagrassFinder, Coralscapes, BenthicNet, ReefNet, NOAA PIFSC) are
   large but side-view or close-range photos: low direct value for overhead segmentation; useful as evidence
   for data practice, and Coralscapes (Apache-2.0) and NOAA PIFSC (public domain) are the cleanly licensed ones.

**Key data-practice findings**

- Recent benthic datasets split **by site, transect or lagoon**, not at random: Coralscapes (by reef site)
  [18], SeagrassFinder (hold out a whole transect) [17], Contini et al. (train one lagoon, test another) [12],
  BenthicNet (≥50 m buffer) [19], Pawar et al. (leave-one-site-out) [4]. Random hold-outs overstate CNN
  accuracy on drone imagery (Kattenborn et al., up to 28%) [44] [?].
- **Cross-site/cross-sensor shift is the main failure mode.** Pawar et al. reach F1/IoU 0.52/0.48 on an unseen
  site, about 68% of a model trained on both domains [4] [?].
- **Label noise** is handled by conservative annotation ("speculation-free": uncertain pixels left as
  background) [18], by dropping images where annotators disagree [17], by coarse cover classes rather than
  exact outlines [6][7], and by dropping classes whose weak labels are too sparse [12].
- **Partial/weak labels** are well studied: image-level labels for coarse seagrass segmentation [15], sparse
  point labels propagated to dense masks with superpixels or DINOv2 features [45][46], and underwater
  predictions projected onto UAV orthophotos as coarse masks [12].
- **Annotation tooling**: CVAT with SAM helps on salient boundaries but not on diffuse or small regions [18];
  CoralNet / CPCe / ReefCloud use random-point annotation [21][22][23].
- **Pre-training**: in-domain self-supervised pre-training (Barlow Twins on 1M seafloor images) beat ImageNet
  on a small downstream task and tied on a large one [19]. No published SSL study on unlabelled overhead
  coastal drone imagery for seagrass was found.

---

## 2. Master table

GSD = ground sampling distance. "Commercial OK?" is read from the stated licence only; it is not legal advice.

| # | Name | Year | Platform / sensor | GSD | Region | Size | Label type | Licence | Commercial OK? | URL |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | hotosm/vhr-seagrass-segmentation | 2026 | Aerial/drone tiles (TM imagery, OAM among sources), RGB | zoom 22 (GSD not stated) | Timor-Leste | 4,498 chips 512², 20,761 features | Masks + GeoJSON polygons, binary | CC BY 4.0 (imagery) + ODbL 1.0 (labels) [V] | Yes, with attribution/share-alike on labels | [1] |
| 2 | hotosm/fAIr-guru-td-seagrass | 2026 | OAM drone imagery | not stated | Timor-Leste | 4,497 tiles | Masks, binary | CC BY 4.0 + ODbL [V] | Yes (as #1) | [2] |
| 3 | CoastalEcology/SeagrassMapper | 2025 | UAV + aircraft cameras | ~3 cm / ~12.5 cm | Denmark (3 fjords + source sites) | 1K–10K tiles 512² | Binary eelgrass masks | CC BY 4.0 [V] | Yes | [3] |
| 4 | MagicBathyNet | 2024 | Aerial + SPOT-6 + Sentinel-2 (+DSM, bathymetry) | aerial 6.3–6.8 cm; S2 10 m | Agia Napa (CY), Puck Lagoon (PL) | 3,355 triplets 180×180 m; 533 annotated patches | Sparse pixel annotations, 5 classes | CC BY-NC-SA 4.0 [V] | **No** | [5] |
| 5 | Green Island seagrass (Langlois & McKenzie) | 2022 | DJI Mavic 2 Pro UAV; PlanetScope | 2.45 cm; 3.7 m | Great Barrier Reef, AU | 1 site; 6 data points (GIS files) | Mapped cover classes (0/1–15/15–50/>50%) | CC BY 4.0 [V] | Yes | [6] |
| 6 | Yule Point seagrass (Langlois & McKenzie) | 2022 | GPS, helicopter, DJI Mavic 2 Pro, PlanetScope | UAV 0.2 cm at 30 m [?] | Great Barrier Reef, AU | 1 site, 2017–2020 | Mapped cover classes | CC BY 4.0 [V] | Yes | [7] |
| 7 | Seatizen Atlas | 2024–25 | ASV, snorkel, kitesurf, paddleboard, UAV | not stated | Réunion, SW Indian Ocean | 14,492 multilabel + 1,200 instance-seg. images; 39.9 TB all versions | Image multilabel, instance masks (2 seagrass species) | CC BY 4.0 in metadata; text says non-commercial [V] | **Unclear** (conflict) | [11] |
| 8 | Contini et al., "The point is the mask" | 2025 | DJI Mavic 2 Pro UAV + ASV | 0.9 / 1.6 cm | Réunion (2 lagoons) | ~19 + 20 ha orthophotos | Coarse masks from ASV predictions; coral classes, no seagrass | Code on GitHub; data licence not found | Unclear | [12] |
| 9 | KAUST Red Sea UAV orthomosaics + DEMs | 2023 | UAV (SfM) | not stated | Saudi Red Sea | 39.7 GB, 3 reef areas | OBIA habitat maps + manual accuracy points | CC0 1.0 [V] | Yes | [13] |
| 10 | DeepSeagrass | 2021 | Handheld camera, snorkeller (~1 m depth) | close range, 4624×2600 px [?] | Moreton Bay, AU | >12,000 images [?] | Image-level morphotype (3 + background) | CC BY-NC-ND 4.0 [?] | **No** | [14] |
| 11 | Luderick-Seagrass test patches | 2023 | Underwater video | close range | SE Queensland, AU | 1.2 GB patches | Patch-level (background/fish/seagrass) | CC BY 4.0 [V] | Yes | [16] |
| 12 | SeagrassFinder | 2024 | Towed camera sled video | close range | Copenhagen, DK | 8,500 images, 6 transects | Image-level presence | CC BY-NC-ND 4.0 [V] | **No** | [17] |
| 13 | Coralscapes (v1) / CoralscapesV2 | 2025 / 2026 | Diver video frames | close range | Red Sea, 35 sites | 2,075 / 2,433 images | Dense masks, 39 classes incl. seagrass | Apache-2.0 [V] | Yes | [18] |
| 14 | BenthicNet | 2024–25 | Diver, ROV, AUV, towed, drop cameras | close range | Global | 1.3M images; 3.1M annotations on 190k images | Sparse points, CATAMI | Compilation CC BY 4.0; per-image licences vary [V] | **Per image** | [19] |
| 15 | ReefNet-1.0 | 2025 | Photo-quadrats (76 CoralNet sources + Al-Wajh) | close range | Global | ~925k annotations, 13.6 GB | Sparse points, hard coral genera | CC BY-NC-SA 4.0 [V] | **No** | [20] |
| 16 | CoralNet (platform) | ongoing | Photo-quadrats | close range | Global | ~4.4M images, ~245M points [?] | Sparse points | Per source; licence not found | Unclear | [21] |
| 17 | ReefCloud (AIMS platform) | ongoing | Photo-quadrats | close range | Indo-Pacific | not stated | Sparse points | "Not Specified" on data.gov.au [?] | Unclear | [22] |
| 18 | NOAA PIFSC benthic cover T1 (HF) | 2025 [?] | Photo-quadrats (NCRMP) | crops 224² | US Pacific Islands | 790,352 crops | 1 label/crop, 10 classes (no seagrass) | Public domain (NOAA open data) [V] | Yes | [23] |
| 19 | Seattle Aquarium benthic imagery | 2024–25 [?] | ROV | close range | Elliott Bay, US | >6,000 images [?] | Image enhancement pairs (not habitat masks) [?] | CC BY-NC 4.0 [V] | **No** | [24] |
| 20 | RapidBenthos | 2024 [?] | Diver photogrammetry orthomosaics | ~3,000 photos/plot | Great Barrier Reef, AU | 12 plots | Segmented benthic constituents; images on request | CC BY-NC-SA 4.0 [V] | **No** | [25] |
| 21 | SeagrassSpotter (Project Seagrass) | ongoing | Citizen photos + GPS points | n/a | Global | ~13,283 records (OBIS copy) [?] | Occurrence points, species | OBIS copy CC BY [?] | Probably (check) | [26] |
| 22 | Seagrass Explorer (Zooniverse) | ongoing | Citizen-classified images | n/a | SW England | not stated | Crowd labels | Licence not found | Unclear | [27] |
| 23 | Cymodocea nodosa WorldView-2 set (Mederos-Barrera) | 2025 | WorldView-2 | not stated | El Río, Canary Is. | not stated | Segmentation from field survey; "open dataset" claimed [?] | Licence not found | Unclear | [29] |
| 24 | Placentia Bay eelgrass drone maps | 2020 | UAV orthomosaics, classified | not stated | Newfoundland, CA | 3 sites | Classified rasters (model output) | CC BY-NC 4.0 [?] | **No** | [30] |
| 25 | Sea-Undistort (synthetic) | 2025 | Blender-rendered airborne through-water scenes | not stated | Synthetic | 1,200 pairs 512² | Clean/distorted pairs (glint, waves); seagrass seabeds | CC BY-NC-SA 4.0 (arXiv licence link) [?] | **No** | [28] |

Not released or unconfirmed (method papers only): Tallam et al. 2023 Morro Bay eelgrass drone data [31];
SGDenseNet training samples (9,369 field points) [37]; BenthIQ Mo'orea drone masks [?]; Remote Sensing 18
(2026) 2057 intertidal UAV set [48]; JCU UNet++ tropical seagrass drone work [10]; SAM2 seagrass study (data
source not stated) [52].

---

## 3. Per-dataset notes

**HOT datasets [1][2].** Labels are mapper-digitised OSM-style polygons (`seamark:type=seagrass`); the card
lists known caveats: imagery and mapping dates can differ, missed or misdrawn features, varying colour and
angle, and validation/test chips share imagery URLs with training, so they test new areas of the same photos,
not new imagery [1] [V]. The sibling set splits each project into 10 contiguous regions [2] [V]. Both are
single-country.

**SeagrassMapper [3][4].** Built for unsupervised domain adaptation from UAV (3 cm) to aircraft orthophotos
(12.5 cm) across three Danish fjords [V]. Suitable as an external test of generalisation and as a second
domain. Caveats: temperate *Zostera marina*, different water optics; band set not confirmed as RGB [?];
dataset viewer errored, so masks were not inspected.

**MagicBathyNet [5].** Only public set found that pairs cm-scale aerial imagery with satellite imagery and
seabed habitat labels including seagrass. Annotations are sparse (533 annotated patches; mostly Puck Lagoon,
a Baltic site) [V]. Non-commercial, share-alike. The official site (`magicbathy.eu`) could not be fetched;
licence read from the arXiv paper.

**Langlois & McKenzie GBR sets [6][7][8][9].** The closest tropical, drone, seagrass work: UAV at 100 m
(2.45 cm) at low spring tide, seagrass cover mapped with deep learning, model trained on a hand-labelled
subset of images [V]. PANGAEA holds GIS map products; whether orthomosaics and training masks are included was
not confirmed. CC BY 4.0. Images of Green Island are also on GeoNadir; GeoNadir's licence terms were not found.

**Seatizen Atlas and Contini et al. [11][12].** Large open collection of tropical shallow-reef imagery
(ASV underwater + UAV aerial) from Réunion with two seagrass species in the label set. Contini et al. derive
coarse UAV masks from underwater ASV predictions, train on one lagoon and test on another, and **dropped the
seagrass classes** because the teacher predicted too few of them (noise risk) [V]. Licence conflict
(CC BY 4.0 metadata vs "non-commercial use" text) must be resolved with the authors before use.

**KAUST UAV [13].** CC0 shallow-reef orthomosaics (39.7 GB) with OBIA habitat maps. Labels are classifier
output, not manual masks; most useful as unlabelled VHR overhead water imagery for pre-training. GSD and
class list were not stated on the record.

**Underwater sets [14]–[20].** All side-view or close-range; texture and scale differ from 1–3 cm nadir
orthomosaics. DeepSeagrass and SeagrassFinder are ND (no derivatives) and NC. Coralscapes is the cleanest
licence (Apache-2.0) and documents good practice (site split, CVAT+SAM, speculation-free labels) [18].
BenthicNet's per-image licences vary and some forbid commercial/derivative use [19].

**Satellite sets [29][38].** WorldView-2 *Cymodocea* (U-Net IoU 0.74 on seagrass [?]) and Sentinel-2 work show
the task at 0.5–10 m; useful for method comparison only.

---

## 4. Global and regional products (weak labels, validation, site selection)

| Product | Resolution / form | Coverage | Licence | Use for us | Src |
|---|---|---|---|---|---|
| Allen Coral Atlas reef habitat v2.0 | 5 m raster, PlanetScope 2018–2020; benthic classes incl. Seagrass | Tropical shallow reefs (lat ±33°) | CC BY 4.0, © Allen Coral Atlas Partnership & Vulcan [V] | Site selection, negatives, coarse sanity check | [32] |
| ACA benthic/geomorphic reference data (UQ, figshare) | Points/polygons | Global reefs | Open access; licence not found [?] | Field reference near our sites | [33] |
| Global seagrass map, Peng et al., *Nature* 2026 | 10 m, dense/sparse, Sentinel-2 (2019–20, 2023–24); 148,506 km² | Global clear shallow water | Availability and licence not found; being added to Allen Coral Atlas [?] | Best global prior once released | [36] |
| SGDenseNet (Peng et al. 2025) | Sentinel-2 model, 9,369 field samples | 4 regions + global test | Data release not found | Method only | [37] |
| Digital Earth Pacific seagrass extent | 10 m, probability + class, yearly from 2017 | Pacific (SPC) | CC BY 4.0 [V] | Weak labels / validation if sites fall in coverage (Timor-Leste coverage not checked) | [35] |
| UNEP-WCMC Global Distribution of Seagrasses v7.1 (2021) | Polygons + points, ~1:1,000,000 | Global | UNEP-WCMC General Data License; commercial use: contact UNEP-WCMC [V] | Presence prior only | [34] |
| UNEP-WCMC Modelled Global Distribution of the Seagrass Biome | Modelled raster | Global | Licence not found | Prior only | [34] |
| Mediterranean *P. oceanica* extent (Traganos et al. 2022) | Sentinel-2, 62,928 labelled pixels, 22 countries | Mediterranean | Paper CC BY 4.0; map release not confirmed [?] | Not our region | [38] |
| EMODnet / MEDISEH *Posidonia* maps | Vector maps; modelled raster ~0.0042° | Mediterranean | CC BY 3.0 / 4.0 per record [?] | Not our region | [39] |
| NOAA NCCOS shallow-water benthic habitat maps | Polygons from aerial photo interpretation; seagrass density classes | US Caribbean, Hawaii, Guam, Palau, Marshall Is., etc. | InPort "use constraints: none" for some items [?]; licence not found | Possible VHR-ish polygon labels if paired with aerial photos (1999 photos exist) | [40] |
| Chesapeake Bay SAV (VIMS) | Annual SAV bed polygons from aerial survey, 1984– | Chesapeake Bay, US | Licence not found ("access constraint: none") [?] | Temperate SAV; long time series | [41] |
| Florida seagrass statewide (FWC/FWRI) | Polygons, continuous/patchy | Florida, US | Licence not found | Temperate/subtropical polygons | [42] |
| OpenStreetMap `seamark:type=seagrass` | Nodes/ways | Global | ODbL | Only **331** objects worldwide on 2026-11-19 (taginfo) [V]: not a usable label source | [43] |

---

## 5. Data practice

### 5.1 Splits
- **By site/region is the norm in 2024–2026 benthic datasets.** Coralscapes: 27/3/5 reef sites for
  train/val/test [18] [V]. SeagrassFinder: authors recommend holding out a whole transect because frames are
  near-duplicates [17] [V]. BenthicNet: test images chosen to minimise spatial overlap, images within 50 m
  of the other partition excluded, done per ecological unit [19] [V]. Contini et al.: train Trou d'eau,
  test Saint-Leu [12] [V]. Pawar et al.: one-to-one, leave-one-site-out and all-to-one across three fjords
  [4] [?]. NOAA PIFSC card warns users to make site-aware splits to prevent leakage [23] [V].
- **Evidence that random splits mislead:** Kattenborn et al. (2022) on drone tree-species segmentation found
  random hold-outs overestimated performance by up to 28% versus spatially blocked hold-outs, more so with
  small training sets [44] [?].
- **Implication for us (from the evidence, not tested):** a split that holds out whole imagery acquisitions or
  whole sites, not only new areas of the same orthomosaic, is what the literature treats as an honest test;
  the HOT card already notes its test shares imagery with training [1].

### 5.2 Label noise and partial labelling
- **Conservative labelling:** Coralscapes labels uncertain pixels as background ("speculation-free") and
  verifies every polygon [18] [V].
- **Consensus filtering:** SeagrassFinder dropped images on which experts and non-experts disagreed [17] [V].
- **Cover classes instead of outlines:** GBR drone maps use percent-cover classes (e.g. 0, 1–15, 15–50,
  >50%) [6] [V]; the JCU UNet++ work found models do best at moderate-high cover and struggle in fragmented,
  low-cover meadows [10] [V].
- **Weak labels:** image-level labels can train coarse seagrass segmentation (Raine et al., WACV 2024) [15];
  sparse point labels can be propagated to dense masks with point-label-aware superpixels [45] or DINOv2
  features + KNN, with human-in-the-loop choice of points at 5–25 points/image [46] [?].
- **Cross-platform teacher labels:** underwater ASV predictions interpolated into rasters and upsampled to UAV
  resolution gave 50.1% mIoU on an unseen lagoon; seagrass dropped as too sparse [12] [V].
- **Date mismatch** between imagery and labels is a known noise source in HOT data [1] [V].

### 5.3 Class imbalance
- Contini et al. gave rare classes more weight [12] [V]. The 2026 intertidal UAV paper includes a sensitivity
  analysis of the loss weight [48] [?] (section heading only; numbers not read). NOAA PIFSC documents strong
  class skew (turf 337k vs mobile fauna 1.1k) [23] [V]. No seagrass-specific comparison of loss functions
  (Dice, focal, Lovász) was found.

### 5.4 Annotation tools and protocols
- **CVAT + Segment Anything:** good for salient boundaries, "largely unhelpful" for diffuse boundaries and small
  polygons, which were drawn by hand [18] [V]. A 2025 study fine-tunes SAM2 for seagrass [52] [?] (data and
  results not read).
- **Point-count platforms:** CoralNet, CPCe, ReefCloud — random points per photo-quadrat, CATAMI or custom
  label sets [19][21][22][23].
- **Field protocols for drone seagrass:** fly at low spring tide, 100 m altitude, 85% overlap, PIX4D
  orthomosaic [6] [V/?]; about 50 low-altitude spot-checks per location stabilised cover estimates [10] [V];
  hand-label a subset and let the model map the rest [9] [V].
- **Satellite labelling:** human-labelled pixel inventories (62,928 pixels, Mediterranean) [38] [?] and
  digitised points (≈4,000 seagrass / 8,000 non-seagrass, BVI) [51] [?].

### 5.5 Domain shift, synthetic data and pre-training
- **Domain adaptation:** histogram matching + SegFormer from UAV 3 cm to aircraft 12.5 cm reached ~68% of a
  model trained on both domains; target-domain contrast mattered [4] [?].
- **In-domain SSL:** Barlow Twins ResNet-50 on BenthicNet-1M beat ImageNet on a small task (77.0 vs 65.9%
  accuracy) and tied on a large one (88.1 vs 88.3%) [19] [V]. Evidence is from underwater photos.
- **SSL for UAV imagery (other domains):** FastSiam pre-training on UAV data beat end-to-end training with few
  labels (vineyards) [50] [?]; DINOv2 features transfer to UAV flood segmentation [49] [?]. No study of SSL on
  unlabelled overhead coastal drone imagery for seagrass was found.
- **Synthetic data:** Sea-Undistort renders 1,200 airborne through-water pairs with glint/waves over seabeds
  including seagrass (NC-SA) [28] [?]; Mai et al. trained DeepLabV3+ on rendered underwater scenes with a
  seagrass class, but note real-world validation is still needed [47] [?]. No synthetic overhead seagrass
  segmentation study with real-data results was found.

---

## 6. Gaps

- **No public, openly licensed, tropical, RGB, 1–3 cm overhead seagrass segmentation set** other than HOT's.
  The nearest (SeagrassMapper) is temperate; the nearest tropical ones (GBR drone maps, Seatizen) lack released
  pixel masks or have unclear licences.
- **No multi-site drone seagrass benchmark** with a site-level held-out test exists in the sources found.
- **Licences unclear or missing** for CoralNet sources, ReefCloud, NOAA NCCOS maps, Chesapeake VIMS, Florida
  FWC, GeoNadir uploads, ACA reference data and the Canary Islands WorldView-2 set. Seatizen has a licence
  conflict. These need direct confirmation before any use.
- **Nature 2026 global 10 m seagrass map**: data availability and licence not found (Nature page redirected to
  login).
- **OSM is not a source of seagrass labels** (331 objects worldwide) [43]. The HOT dataset card reports 20,761
  features from its projects [1]; how those relate to the OSM count was not checked here.
- **Not verified on primary pages:** DeepSeagrass licence (CSIRO portal blocked), MagicBathyNet site, MDPI full
  texts (403), Kattenborn 28% figure, Roboflow `seagrass_segmentation` (403; licence not found).
- **Method gaps:** no seagrass-specific study on loss choice for imbalance, on SSL with unlabelled overhead
  coastal imagery, or on synthetic overhead data validated on real orthomosaics.

---

## 7. Sources

1. hotosm/vhr-seagrass-segmentation — https://huggingface.co/datasets/hotosm/vhr-seagrass-segmentation (API: https://huggingface.co/api/datasets/hotosm/vhr-seagrass-segmentation)
2. hotosm/fAIr-guru-td-seagrass — https://huggingface.co/api/datasets/hotosm/fAIr-guru-td-seagrass
3. CoastalEcology/SeagrassMapper (DOI 10.57967/hf/5205) — https://huggingface.co/datasets/CoastalEcology/SeagrassMapper
4. Pawar et al. 2025, "Eye in the Sky for Sub-Tidal Seagrass Mapping…", Remote Sensing 17(14) 2518 — https://www.mdpi.com/2072-4292/17/14/2518 ; https://doaj.org/article/072d734ced5b4d7594290ccb21ef39b4
5. MagicBathyNet, arXiv 2405.15477 — https://arxiv.org/html/2405.15477v2 ; https://www.magicbathy.eu/magicbathynet.html
6. Langlois & McKenzie 2022, Green Island seagrass, PANGAEA — https://doi.pangaea.de/10.1594/PANGAEA.946605
7. Langlois & McKenzie 2022, Yule Point seagrass, PANGAEA — https://doi.pangaea.de/10.1594/PANGAEA.946604 ; https://b2find.eudat.eu/dataset/423edea8-c70f-5701-89cc-2a2a88acdb17
8. McKenzie, Langlois & Roelfsema 2022, Remote Sensing 14(11) 2604 — https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-02604/article_deploy/remotesensing-14-02604.pdf
9. GeoNadir, "Mapping seagrass with machine learning and drones" — https://geonadir.com/mapping-seagrass-machine-learning-drones/
10. Langlois et al., "Drone-Based Monitoring of Tropical Seagrass…" (AMSA 2026 abstract) — https://portfolio.jcu.edu.au/publications/JCU325143
11. Seatizen Atlas image dataset, Zenodo — https://zenodo.org/records/12819156 ; paper: https://archimer.ifremer.fr/doc/00932/104424
12. Contini et al. 2025, "The point is the mask", arXiv 2508.18958 — https://arxiv.org/html/2508.18958v1 ; code: https://github.com/SeatizenDOI/the-point-is-the-mask
13. Nieuwenhuis et al. 2023, KAUST UAV orthomosaics/DEMs, Zenodo — https://zenodo.org/records/7236603
14. DeepSeagrass, arXiv 2103.05226 — https://arxiv.org/abs/2103.05226 ; CSIRO record: https://researchdata.edu.au/deepseagrass/1670799
15. Raine et al., WACV 2024, "Image Labels Are All You Need for Coarse Seagrass Segmentation" — https://openaccess.thecvf.com/content/WACV2024/html/Raine_Image_Labels_Are_All_You_Need_for_Coarse_Seagrass_Segmentation_WACV_2024_paper.html
16. Global Wetlands Luderick-Seagrass test patches, Zenodo — https://zenodo.org/records/7659203 (original: doi:10.1594/PANGAEA.926930)
17. SeagrassFinder, Zenodo — https://zenodo.org/records/13904604 ; paper: https://arxiv.org/html/2412.16147v1
18. Coralscapes — https://huggingface.co/datasets/EPFL-ECEO/coralscapes ; https://huggingface.co/datasets/josauder/coralscapesV2 ; paper: https://arxiv.org/html/2503.20000v2
19. BenthicNet, arXiv 2405.05241 / Sci. Data 2025 — https://arxiv.org/html/2405.05241v3 ; https://www.frdr-dfdr.ca/repo/dataset/24c6c813-d0ff-4461-8174-3f960f1edc0d
20. ReefNet-1.0 — https://huggingface.co/datasets/ReefNet/ReefNet-1.0 ; https://arxiv.org/abs/2510.16822v1
21. CoralNet — https://coralnet.ucsd.edu/
22. ReefCloud on data.gov.au — https://data.gov.au/data/dataset/reefcloud
23. NOAA PIFSC benthic cover T1 — https://huggingface.co/datasets/NMFS-OSI/noaa-pacific-benthic-cover-t1-all
24. Seattle Aquarium benthic imagery — https://huggingface.co/datasets/Seattle-Aquarium/Seattle_Aquarium_benthic_imagery
25. RapidBenthos data, JCU — https://researchdata.edu.au/data-from-rapidbenthos-photogrammetric-reconstruction/3481305
26. SeagrassSpotter: OBIS — https://obis.org/dataset/872b3454-7114-4ca9-914b-99b2518f86ee ; https://www.projectseagrass.org/tag/open-data/ ; https://api.seagrassspotter.org/
27. Seagrass Explorer (Zooniverse) — https://www.zooniverse.org/projects/mark-dot-parry/the-community-seagrass-initiative-seagrass-explorer
28. Sea-Undistort, arXiv 2508.07760 — https://arxiv.org/abs/2508.07760
29. Mederos-Barrera et al. 2025, IEEE JSTARS, *Cymodocea nodosa* WorldView-2 — https://accedacris.ulpgc.es/handle/10553/154907
30. Placentia Bay eelgrass drone maps, CIOOS Atlantic — https://catalogue.dev.cioosatlantic.ca/en/dataset/ca-cioos_f8bd7be3-d4e5-4775-85e9-30dc7e9ae126
31. Tallam et al. 2023, Remote Sensing 15(9) 2321 — https://mdpi-res.com/d_attachment/remotesensing/remotesensing-15-02321/article_deploy/remotesensing-15-02321.pdf
32. Allen Coral Atlas reef habitat v2.0 (Earth Engine) — https://developers.google.com/earth-engine/datasets/catalog/ACA_reef_habitat_v2_0
33. ACA reference data (UQ, figshare) — https://researchdata.edu.au/benthic-geomorphic-reference-reef-mapping/3367080 ; Roelfsema et al. 2021 — https://www.frontiersin.org/articles/10.3389/fmars.2021.643381/full
34. UNEP-WCMC Global Distribution of Seagrasses v7.1 — https://sextant.ifremer.fr/geonetwork/srv/api/records/9e5e5d26-eea9-4729-a7bc-4f13223e0cf2 ; Ocean+ Habitats: https://habitats.oceanplus.org/
35. Digital Earth Pacific seagrass extent — https://registry.opendata.aws/dep-seagrass/
36. Peng, Li, Krause et al. 2026, *Nature*, doi:10.1038/s41586-026-10704-3 — https://news.asu.edu/20260624-environment-and-sustainability-first-complete-map-worlds-seagrass-offers-warnings-and-hope
37. SGDenseNet, Peng et al. 2025, ISPRS J. 220:277–294 — https://www.conservationgateway.org/collections/ocean/global-seagrass-mapping-sgdensenet-deep-learning/
38. Traganos et al. 2022, Front. Mar. Sci. — https://www.frontiersin.org/articles/10.3389/fmars.2022.871799
39. EMODnet / MEDISEH *Posidonia* — https://gis.ices.dk/geonetwork/srv/api/records/c690ac1c-015b-43e2-8aa1-0aff72eec50e ; https://emodnet.ec.europa.eu/en/map-week-seagrass-distribution
40. NOAA NCCOS benthic habitat maps — https://products.coastalscience.noaa.gov/collections/benthic/default.aspx ; https://www.fisheries.noaa.gov/inport/item/39445
41. Chesapeake Bay SAV (VIMS / CBP) — https://mobjack.vims.edu/sav/gisdata/beds10_metadata.xml ; https://gis.chesapeakebay.net/ags/rest/services/ChesapeakeProgress/cpSAV_beds_2010_2023/MapServer
42. Florida seagrass (FWC) — https://gis.myfwc.com/Data/metadata/SDE_Current/seagrass_fl_poly.htm
43. OSM taginfo `seamark:type=seagrass` — https://taginfo.openstreetmap.org/tags/seamark:type=seagrass
44. Kattenborn et al. 2022, ISPRS Open J. — https://doaj.org/article/23b9fa0bb95147349b20e4cdb872469e
45. Raine et al. 2022, Point Label Aware Superpixels — https://arxiv.org/abs/2202.13487
46. Raine et al. 2024, Human-in-the-Loop Segmentation of Multi-species Coral Imagery — https://arxiv.org/html/2404.09406v2
47. Mai et al. 2025, synthetic underwater marine-growth segmentation — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11751705/
48. "Intertidal Seagrass Mapping Using UAV Visible and Multispectral Imagery", Remote Sensing 18 (2026) 2057 — https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-02057/article_deploy/remotesensing-18-02057.pdf
49. DINOv2 features for UAV flood segmentation, arXiv 2507.04915 — https://arxiv.org/html/2507.04915v1
50. Doornbos & Babur 2025, FastSiam UAV pre-training, ISPRS Annals — https://isprs-annals.copernicus.org/articles/X-2-W2-2025/31/2025/
51. UK Hydrographic Office, BVI Sentinel-2 seagrass mapping — https://medium.com/uk-hydrographic-office/collaborative-seagrass-mapping-from-sentinel-2-satellite-imagery-fd499edc75cd
52. Huang 2025, "Advancing Seagrass semantic segmentation with SAM2 models", JHSS — https://jhss.scholasticahq.com/article/138532-advancing-seagrass-semantic-segmentation-with-sam2-models
