# Preprocessing: state of the art

2026-11-19. Scope: published practice and evidence on preprocessing very-high-resolution (VHR) RGB drone/aerial imagery for seagrass, benthic and shallow-water semantic segmentation, plus general VHR remote-sensing segmentation. This is a literature review. It does not assess our data or code.

Tags: **[V]** the claim was read in the source (abstract, docs or full text, as stated). **[?]** the claim is second-hand (a search snippet, secondary description or citing paper) or was not confirmed in the original. Numbers are quoted only with their dataset and setting. "SG" marks seagrass- or benthic-specific evidence. "GEN" marks general remote sensing or computer vision.

---

## 1. Summary

1. **The classic water-column and glint corrections do not fit RGB-only drone imagery as published.** Hedley deglinting regresses each visible band against an NIR band over deep water [2][V]. Lyzenga's depth-invariant index (DII) needs pixels of one substrate at varying depths and assumes homogeneous water [7][V]. A drone RGB orthomosaic has no NIR band and usually no depth. No study found here shows either method improving a deep network on RGB-only drone imagery. (SG)
2. **For RGB drone imagery, glint is now handled at acquisition and with learned glint masks, not radiometric correction.** Acquisition guidance exists: sun angle, cloud, wind, polarising filters [13][?][14][?]. There are also learned RGB glint detectors: a U-Net that found ~66% of glint pixels and ~99.6% of background [5][V], and FANet, with glint IoU 81.34% [6][V]. (SG)
3. **The evidence on underwater colour correction or enhancement as preprocessing is mixed.** On URPC2020, enhancement before detection generally lowered accuracy compared with raw images [11][V]. On underwater seagrass video, enhancement gave small gains [12][V]. Sea-thru needs a per-pixel range map (RGBD) [10][V]. No aerial seagrass study tests these methods. (SG/GEN)
4. **Normalisation matters most when it matches what the pretrained encoder saw.** Matching pretraining resize and normalisation gave large gains in frozen-feature remote-sensing benchmarks [15][V]. Library docs say the same and note it matters less when the whole model is trained [16][V]. Seagrass UAV papers use dataset or training-set statistics [18][?]. One reports min-max beating z-score on optical drone images, but only per site [17][?]. (GEN, SG weak)
5. **Differences between cameras, dates and sites are handled better with augmentation and domain matching than with fixed input normalisation.** Randomised histogram matching as a training augmentation matched or beat unsupervised domain adaptation on building segmentation [19][V]. Histogram matching plus SegFormer, from 3 cm UAV imagery to 12.5 cm orthophotos of seagrass, reached F1/IoU 0.52/0.48, about 68% of a model trained on both domains [20][?]. (GEN, SG)
6. **Train and serve at one fixed ground sample distance (GSD), and resample to it.** Web-Mercator tile GSD scales with cos(latitude) [29][V]. Train/test altitude mismatch causes large drops in UAV segmentation, though that evidence comes from a synthetic urban benchmark [28][?]. Scale-aware pretraining gives only modest gains (+0.9 to 1.7 mIoU on SpaceNet) [27][V]. (GEN)
7. **Use larger windows or overlapping windows with weighted blending at inference.** Predictions near patch borders are worse [22][V][25][V]. A larger input size at inference cut time and modestly improved accuracy on INRIA [22][V]. 50% overlap with Gaussian blending is the nnU-Net default [24][V]. Tile size has a significant, dataset-dependent effect [26][?]. (GEN)
8. **Extra input channels (indices, depth) have no clean evidence at drone scale.** In one UAV seagrass study, indices such as NGRDI were influential features, but it has no RGB-only ablation [18][?]. Water-column correction helped a lot at satellite scale (PlanetScope) [9][?]. That does not carry over to cm-scale RGB tiles. (SG)

---

## 2. Water-column and optical corrections

### 2.1 Sun-glint removal (Hedley, Lyzenga, Kay; learned methods)

- **What:** remove specular sun reflection from the water surface.
  - Hedley et al. (2005) fit a linear regression of each visible band against NIR over a deep-water sample showing a range of glint. For each pixel they subtract `b_i * (NIR - minNIR)` [1][?][2][V]. The ESA Sen2Coral spec states the core assumption: "the NIR reference band only contains the glint (because water is opaque in the NIR)" [2][V].
  - Lyzenga and Joyce variants follow the same NIR-reference idea [4][?].
  - Kay et al. (2009) reviewed these methods. For coastal images with pixels under ~10 m, the methods rely on the NIR signal [3][?].
- **Where (SG):** Muslim et al. (2019) compared Lyzenga, Joyce, Hedley and Goodman glint corrections on UAV imagery of a coral reef (Pulau Bidong, Malaysia). They found the Lyzenga method, with their "Strategy-2", most accurate over coral, rubble and sand classes. They also noted that few studies had tested glint correction on UAV camera data over reefs [4][?]. Which band they used as the reference was not confirmed here.
- **Learned RGB glint detection (SG):**
  - Giles et al. (2021) trained a modified U-Net on high-resolution RGB aerial imagery, with labels from object-based analysis. It identified ~99.6% of background and ~66% of glint, where glint was ~1.2% of test pixels, and was released as open source [5][V].
  - Qin et al. (WACV 2024) detect glint with FANet (glint IoU 81.34%, background IoU 99.52%). They then fill it by propagating pixels along an optical-flow field, beating DeepFillv2 and Fast Marching inpainting [6][V]. A related patent fills glint from adjacent frames with RAFT optical flow [6][?]. This suggests the method needs overlapping frames, not a single orthomosaic tile (unconfirmed).
- **Evidence that it helps segmentation:** none found for deep segmentation on RGB drone imagery. Kay et al. report that uncorrected glint caused errors of up to 30% in water-depth estimates in one case study [3][?]. That figure concerns depth, not classification.
- **RGB-only:** the NIR-based methods cannot be used as published. Learned detectors work on RGB, but they mask glint (no data) rather than recover the seabed.
- **CPU cost:** a learned detector is a second segmentation network per tile. Flow-based filling needs multiple frames, so it cannot run on a single served tile.

### 2.2 Acquisition-time mitigation (SG)

- Nahirnick et al. (2019) studied 26 eelgrass sites in British Columbia mapped by UAS with object-based analysis. Sun angle and "theoretical visibility" (tide, Secchi depth, deep edge of the meadow) drove analyst confidence most. They reported favourable conditions as sun angle < 40°, Secchi > 5 m, cloud < 10% or > 90%, and wind < 5 km/h [13][?].
- An MBARI EGU24 presentation recommends a circular polariser, survey legs flown towards and away from the sun, and a solar elevation of about 30–45° [14][?]. Note that its range only partly overlaps Nahirnick's "< 40°".
- **Implication:** for imagery already captured (as with OAM orthomosaics), conditions cannot be changed. The literature's answer is then to mask or down-weight glint-affected or turbid areas, or to accept lower confidence there.

### 2.3 Depth-invariant index (Lyzenga) and water-column correction

- **What:** a log-linearised ratio of a band pair that cancels depth for a given substrate. The ratio of attenuation coefficients Kd_i/Kd_j comes from the slope fitted on same-substrate pixels at different depths [7][V].
- **Assumptions and limits (from the review):** radiance differences within one substrate come only from depth; Kd is constant per band; the water is optically homogeneous. The method works only in clear water and "cannot be applied to very shallow waters". Red bands need shallower depths than blue or green [7][V].
- **Evidence (SG, mixed) [7][V]:**
  - Mumby et al.: +13% accuracy for detailed classes with CASI, no gain for coarse classes, significant gains with Landsat TM.
  - MSS/SPOT XS: losing one dimension cancelled the gain.
  - Zhang et al. (AVIRIS): no gain, attributed to a narrow depth range.
  - Bejarano et al.: Lyzenga-corrected maps reached 56% overall accuracy against 70% for another method.
- **With a CNN at satellite scale (SG):** a Geocarto International (2023) PlanetScope study reported CNN overall accuracy of 89.80% with a bottom-reflectance index and 87.85% with DII, against 47.2% without water-column correction [9][?]. This was seen only in a snippet, and the authors and setup were not confirmed.
- **RGB-only:** DII can in principle be formed from RGB band pairs. However, red attenuates quickly, there is no depth information, and a 2.4 m drone tile does not contain a range of depths of one substrate. No drone or deep-learning evidence was found.
- **CPU cost:** negligible (per-pixel log and linear combination), but it needs calibrated coefficients per scene.

### 2.4 Dehazing and underwater colour correction

- **Sea-thru (Akkaynak & Treibitz, CVPR 2019) (GEN-underwater):** it removes water effects with a revised image-formation model. Backscatter is estimated from the darkest pixels and their known range, so it needs RGBD input [10][V]. The range usually comes from structure-from-motion, which is not available per served tile [10][?].
- **Enhancement before a detector (GEN-underwater):** Wang et al. tested 18 enhancement algorithms × 7 detectors (126 models, plus 7 trained on raw images) on URPC2020.
  - Raw-image detectors performed best: "underwater image enhancement suppresses the performance of object detection" [11][V].
  - CLAHE, white balance, ULAP and some others came close to raw. GAN-based methods (UGAN, TUDA, TACL) degraded results most [11][V].
  - The losses were attributed to more background false positives, because enhancement made the seabed resemble targets [11][V].
- **Seagrass underwater video (SG, classification, not aerial):** SeagrassFinder used DeepWave-Net enhancement. Enhanced models scored higher in 26 of 30 initial runs. On the final test, accuracy went from 0.877 to 0.878 for DenseNet-201 (AUROC 0.955 for both) and from 0.877 to 0.902 for ViT [12][V].
- **UAV video corrections (SG):** Scilla et al. (JSTARS 2025) used colour transfer and averaging of 60 fps UAV video to reduce refraction, caustics and glint over a lagoon [33][?]. This needs video, not a single orthomosaic.
- **Dark-channel dehazing:** no aerial-seagrass segmentation evidence was found. Sea-thru describes its backscatter step as inspired by the dark channel prior [10][?].
- **RGB-only:** yes for CLAHE and white balance, and for learned enhancers. Sea-thru needs range.
- **CPU cost:** CLAHE and white balance are cheap per tile. Learned enhancers are a second network.

---

## 3. Radiometric normalisation

- **Matching the pretraining (GEN):**
  - Corley et al. (CVPRW 2024) showed that following the pretraining preprocessing (image size and normalisation) gave +32.28 overall accuracy on So2Sat (random split) and +11.16 on EuroSAT when evaluating extracted features. ImageNet pretraining "remains a competitive baseline" [15][V]. These are scene-classification benchmarks, not segmentation.
  - segmentation_models_pytorch: "Preparing your data the same way as during weights pre-training may give your better results". It says this is not needed when the whole model is trained [16][V].
- **Seagrass practice (SG):**
  - Lian et al. (Remote Sensing 2026, intertidal seagrass, UAV RGB plus multispectral) cut 256×256 tiles and standardised them with training-set channel statistics [18][?].
  - Jeon et al. (Ecological Informatics 2021) compared z-score and min-max normalisation with U-Net, SegNet, PSPNet and DeepLabv3+ on drone seagrass imagery. Min-max was best for optical images and z-score for greyscale. U-Net ranked first. They advised testing per site [17][?].
- **Per-image vs dataset statistics:** no seagrass study found isolates this. Per-image standardisation removes brightness and contrast differences between flights. It also removes real signal, such as a tile that is entirely dark seagrass, and makes a tile's values depend on its content. This is reasoning, not evidence.
- **Histogram matching / colour constancy across cameras and dates:**
  - **GEN:** randomised histogram matching (RHM) is a training augmentation. It matched or beat state-of-the-art unsupervised domain adaptation on two building-segmentation benchmarks while being simpler [19][V].
  - **Pathology analogue (GEN):** Tellez et al. compared stain-colour augmentation with stain normalisation for unseen centres [21][V]. Secondary summaries say augmentation was the key factor [21][?].
  - **SG:** Pawar et al. (2025) matched histograms of the target domain (12.5 cm airplane orthophotos) to the source (3 cm UAV) and segmented with SegFormer across three Danish estuaries. The best was Skive Fjord, leave-one-out, F1/IoU 0.52/0.48 (~68% of a model trained on both domains). Target-domain contrast strongly affected the results [20][?].
- **CLAHE as input normalisation:** no aerial seagrass evidence. In the underwater-detection study, CLAHE was among the least harmful enhancers but did not beat raw images [11][V].
- **CPU cost:** a fixed affine normalisation is free and can sit inside the ONNX graph. Histogram matching at inference needs a reference histogram and is cheap. RHM costs only at training time.

---

## 4. Spatial resolution, tiling and stitching

### 4.1 GSD and scale

- **Web-map tiles:** metres per pixel = C·cos(latitude)/2^(zoom+8) [29][V]. One zoom level therefore has a different GSD at different latitudes, and source orthomosaics differ in native GSD. Resampling to a fixed GSD (in metres) instead of a fixed zoom is the scale-consistent choice (derived).
- **Mismatch evidence (GEN):**
  - FlyAwareV2 (synthetic-to-real urban UAV): a model trained at 20 m reached 44.5% mIoU on varied heights, against 37.2% when trained at 50 m and 29.4% at 80 m [28][?].
  -   - No seagrass-specific GSD-mismatch study was found.
- **Multi-scale and scale-aware (GEN):** standard practice is to fine-tune on heavily augmented imagery to cover scales. Scale-MAE instead ties positional encoding to ground area. It gives +2.4–5.6% kNN classification on 8 datasets and +0.9–1.7 mIoU on SpaceNet building segmentation across evaluation scales [27][V]. The gains are modest.
- **Resizing for pretrained models (GEN):** in Corley et al., upsampling small benchmark images to the pretraining size helped frozen features a lot [15][V]. The exact EuroSAT 0.66→0.91 figure was seen only in a snippet [15][?].
- **SG practice:** seagrass drone work generally trains on mosaics from one flight design, with tiles of 224–256 px [18][?]. Few papers report how they handle GSD across flights.

### 4.2 Tile size and context

- Cira et al. (Remote Sensing 2024, road extraction from aerial orthophotos, 3 architectures): tile size, and its interactions with overlap and with architecture, had statistically significant effects (α = 0.05). Common sizes were 64 to 1024 px [26][?].
- The best context size depends on the dataset. In ultra-high-resolution segmentation (DeepGlobe vs Inria), too large a context rescaled to a small input loses detail [34][?].
- Reina et al. (Frontiers Neurosci 2020, MRI and satellite): tiling introduces small but meaningful differences at inference. Training on whole images or larger context helped [23][?].
- **Implication:** a 2.36 m, 256 px tile gives limited context for meadow-scale patterns. Larger windows at inference are one of the few gains confirmed across studies (4.3).

### 4.3 Overlap, padding and blended stitching (GEN)

- Huang et al. (2018): stitching patch outputs "can result in discontinuities at output boundaries". Increasing the CNN input size only at inference cut inference time and modestly improved accuracy on INRIA and a second dataset. It was part of their winning INRIA entry [22][V].
- Smoothly-Blend-Image-Patches (widely reused): 50% overlap, a second-order spline window, mirror padding at image edges, and optional D4 (8-fold) rotation/flip test-time augmentation. Its motivation is "a high error on the predictions made near the outside of the window" [25][V].
- nnU-Net (NVIDIA port): sliding-window overlap 0.5 and Gaussian blending by default [24][V].
- **CPU cost (derived):** stride = half the window means about 4× the number of forward passes of non-overlapping windows over an area. D4 test-time augmentation multiplies that by 8. A single larger window avoids the overlap cost if memory allows.

---

## 5. Input representations

- **RGB only (SG):** most drone surveys in the marine environment use RGB sensors, some with polarising filters [35][?]. Seagrass DL studies with RGB alone report useful results:
  - Tallam et al. (Remote Sensing 2023, DeepLab, Morro Bay eelgrass): recall 0.954, precision 0.723, F1 0.809 [30][V].
  - Hobley et al. (2021, Budle Bay): FCNNs were comparable to OBIA [31][?].
- **RGB-derived indices (SG):** in Lian et al. (2026), red- and green-related features and NGRDI were the most influential. The inputs mixed RGB, multispectral and indices, and no RGB-only ablation is reported [18][?]. In principle a CNN can learn fixed index arithmetic from RGB itself. No study was found that isolates indices for drone seagrass.
- **Depth / bathymetry (SG):**
  - Satellite seagrass and benthic models often use water-column corrections or depth [9][?].
  - SGDenseNet (Peng et al. 2025, Sentinel-2) reached 90% overall accuracy with regional data and 81% producer's accuracy on 1,067 global field samples. Its exact inputs (including depth) were not confirmed [32][V for the accuracy figures].
  - No drone-scale study adding SfM depth as a channel to a seagrass CNN was found. OAM tiles carry no depth.
- **Multi-temporal (GEN/SG):** no drone seagrass study using multi-date stacks as input was found. Change studies compare single-date maps instead (for example a 2024 Ecological Informatics study of a tropical meadow, 2012 vs 2022) [?]; seen only in a search snippet, so it is not listed.

---

## 6. Comparison table

| Technique | Evidence strength (for deep nets on VHR RGB seagrass) | RGB-only? | CPU-serving cost | Sources |
|---|---|---|---|---|
| Hedley/Lyzenga NIR deglint | None for this setting; established for multispectral | No (needs NIR) | Low, but not usable | [1][2][3][4] |
| Learned glint detection (mask) | Moderate for detection; no downstream segmentation gain shown | Yes | One extra network | [5][6] |
| Flow-based glint filling | Detection/fill quality only | Yes, but needs multiple frames | Not possible per tile | [6][33] |
| Acquisition rules (sun, wind, polariser) | Moderate (OBIA confidence) | Yes | None (capture time) | [13][14] |
| Lyzenga DII / water-column correction | Mixed for multispectral; large gain in one satellite CNN study | Weak (needs depth range and clear water) | Low, but needs per-scene calibration | [7][8][9] |
| Sea-thru | Strong for colour restoration; none for segmentation | Needs a range map | High (needs depth) | [10] |
| Learned / GAN enhancement | Negative for detection (URPC2020); small positive for seagrass video classification | Yes | One extra network | [11][12] |
| CLAHE / white balance | Close to raw for detection; no seagrass test | Yes | Very low | [11] |
| Pretraining-matched normalisation | Strong (GEN, frozen features) | Yes | Free (inside ONNX) | [15][16] |
| Dataset vs per-image vs min-max | Weak, site-dependent | Yes | Free | [17][18] |
| Histogram matching / RHM augmentation | Moderate (GEN); some SG domain-transfer evidence | Yes | Training only; cheap at inference | [19][20][21] |
| Fixed-GSD resampling | Indirect (mismatch hurts) | Yes | One resize | [27][28][29] |
| Larger inference windows | Moderate (GEN) | Yes | More memory, fewer passes | [22][23] |
| 50% overlap + Gaussian/spline blend | Standard practice; artefact reduction | Yes | ~4× passes | [24][25] |
| D4 test-time augmentation | Practice; variance reduction | Yes | 8× passes | [25] |
| RGB-derived indices | Weak (no ablation) | Yes | Very low | [18] |
| Depth channel | None at drone scale | No (needs depth) | n/a for OAM | [9][32] |

---

## 7. Open questions and gaps

- No published ablation was found that tests glint masking, CLAHE, white balance or learned enhancement as preprocessing for **aerial RGB seagrass segmentation**. The closest evidence is underwater detection (negative) [11] and underwater video classification (slightly positive) [12].
- No seagrass study measures the effect of **train/test GSD mismatch** or compares fixed-GSD resampling with multi-scale training.
- **Per-image vs dataset normalisation** for seagrass has only one weak, site-dependent study [17][?].
- **Tile size** effects are reported as significant but dataset-dependent [26][?]. There is no seagrass-specific test of 256 vs 512 px or of ground extent per window.
- **Cross-flight / cross-camera shift** for seagrass has one domain-transfer study, with modest absolute scores (F1 0.52) [20][?]. How well histogram-matching augmentation transfers to tropical seagrass is untested.
- **Learned glint masks** have not been tested as an input mask or loss mask for downstream seagrass segmentation.
- Several key sources (Muslim 2019, Jeon 2021, Lian 2026, Pawar 2025, Hobley 2021, Cira 2024) could not be read in full because the publisher blocked automated access. Their claims are tagged [?].

---

## 8. Sources

1. Hedley, J.D., Harborne, A.R., Mumby, P.J. (2005). Simple and robust removal of sun glint for mapping shallow-water benthos. Int. J. Remote Sensing 26(10):2107–2112. https://doi.org/10.1080/01431160500034086
2. ESA STEP / Sen2Coral. Deglint Algorithm Specification (Hedley et al. 2005 implementation). https://step.esa.int/main/wp-content/help/versions/9.0.0/snap-supported-plugins/org.esa.sen2coral.sen2coral.algorithms/deglint/DeglintAlgoSpec.html
3. Kay, S., Hedley, J.D., Lavender, S. (2009). Sun glint correction of high and low spatial resolution images of aquatic scenes: a review. Remote Sensing 1(4):697–730. https://www.mdpi.com/2072-4292/1/4/697
4. Muslim, A.M. et al. (2019). Coral Reef Mapping of UAV: A Comparison of Sun Glint Correction Methods. Remote Sensing 11(20):2422. https://www.mdpi.com/2072-4292/11/20/2422
5. Giles, A.B., Davies, J.E., Ren, K., Kelaher, B. (2021). A deep learning algorithm to detect and classify sun glint from high-resolution aerial imagery over shallow marine environments. ISPRS J. Photogramm. Remote Sens. 181:20–26. https://researchportal.scu.edu.au/esploro/outputs/journalArticle/A-deep-learning-algorithm-to-detect/991012968465902368
6. Qin, J., Li, M., Zhao, J., Zhong, J., Zhang, H. (2024). Revolutionize the Oceanic Drone RGB Imagery with Pioneering Sun Glint Detection and Removal Techniques. WACV 2024, 8326–8335. https://mlanthology.org/wacv/2024/qin2024wacv-revolutionize ; related patent US 12,277,713 B2: https://patentsgazette.uspto.gov/week15/OG/html/1533-3/US12277713-20250415.html
7. Zoffoli, M.L., Frouin, R., Kampel, M. (2014). Water Column Correction for Coral Reef Studies by Remote Sensing. Sensors 14:16881–16931. https://pmc.ncbi.nlm.nih.gov/articles/PMC4208206/
8. Lyzenga, D.R. (1978). Passive remote sensing techniques for mapping water depth and bottom features. Applied Optics 17(3):379–383; and Lyzenga (1981), Int. J. Remote Sensing 2(1):71–82. Cited via [7]; not read. https://doi.org/10.1364/AO.17.000379
9. Geocarto International (2023). PlanetScope benthic habitat classification with bathymetry retrieval and water-column correction (CNN, RF, SVM). DOAJ record: https://doaj.org/article/3ccee48d28144c259f58f625ed55412a (seen only in a snippet; authors not confirmed)
10. Akkaynak, D., Treibitz, T. (2019). Sea-thru: A Method for Removing Water From Underwater Images. CVPR 2019. https://mlanthology.org/cvpr/2019/akkaynak2019cvpr-seathru/
11. Wang, Y., Li, C. et al. (2023/2024). Is Underwater Image Enhancement All Object Detectors Need? arXiv:2311.18814; IEEE J. Oceanic Eng. 2024. https://arxiv.org/abs/2311.18814
12. Elsäßer, J., Weihl, L., Cheplygina, V., Nielsen, L.T. (2024/2025). SeagrassFinder: Deep Learning for Eelgrass Detection and Coverage Estimation in the Wild. arXiv:2412.16147. https://arxiv.org/abs/2412.16147
13. Nahirnick et al. (2019). Mapping with confidence; delineating seagrass habitats using Unoccupied Aerial Systems (UAS). Remote Sens. Ecol. Conserv. 5(2):121–135. https://doi.org/10.1002/rse2.98
14. MBARI (2024). EGU24-2934 presentation on UAV imaging over water (polariser, sun elevation). https://presentations.copernicus.org/EGU24/EGU24-2934_presentation-h765343.pdf
15. Corley, I., Robinson, C., Dodhia, R., Lavista Ferres, J.M., Najafirad, P. (2024). Revisiting pre-trained remote sensing model benchmarks: resizing and normalization matters. CVPRW 2024 (PBVS). https://arxiv.org/abs/2305.13456
16. segmentation_models_pytorch documentation, Quick start (input preprocessing). https://smp.readthedocs.io/en/latest/quickstart.html
17. Jeon, E.-I. et al. (2021). Semantic segmentation of seagrass habitat from drone imagery based on deep learning: A comparative study. Ecological Informatics, article 101430. https://doi.org/10.1016/j.ecoinf.2021.101430
18. Lian, J., Mo, Z., Liu, Z., Peng, B., Chang, M., Wang, X., Wang, W. (2026). Intertidal Seagrass Mapping Using UAV Visible and Multispectral Imagery: A Comparative Semantic Segmentation Study with Explainability Analysis. Remote Sensing 18(12):2057. https://www.mdpi.com/2072-4292/18/12/2057
19. Yaras, C., Kassaw, K., Huang, B., Bradbury, K., Malof, J.M. (2021/2024). Randomized Histogram Matching: A Simple Augmentation for Unsupervised Domain Adaptation in Overhead Imagery. arXiv:2104.14032; IEEE JSTARS 2024. https://arxiv.org/abs/2104.14032
20. Pawar, S., Thomasberger, A., Bengtson, S.H., Pedersen, M., Timmermann, K. (2025). Eye in the Sky for Sub-Tidal Seagrass Mapping: Leveraging Unsupervised Domain Adaptation with SegFormer for Multi-Source and Multi-Resolution Aerial Imagery. Remote Sensing 17(14):2518. https://www.mdpi.com/2072-4292/17/14/2518
21. Tellez, D. et al. (2019). Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. Medical Image Analysis. https://arxiv.org/abs/1902.06543
22. Huang, B., Reichman, D., Collins, L.M., Bradbury, K., Malof, J.M. (2018). Tiling and Stitching Segmentation Output for Remote Sensing: Basic Challenges and Recommendations. arXiv:1805.12219. https://arxiv.org/abs/1805.12219
23. Reina, G.A., Panchumarthy, R., Thakur, S.P., Bastidas, A., Bakas, S. (2020). Systematic Evaluation of Image Tiling Adverse Effects on Deep Learning Semantic Segmentation. Frontiers in Neuroscience 14:65. https://doi.org/10.3389/fnins.2020.00065
24. NVIDIA Deep Learning Examples, nnU-Net for PyTorch, advanced options (sliding-window overlap and blend). https://catalog.ngc.nvidia.com/orgs/nvidia/dle/resources/nnunet_pyt/22.11.0/advanced
25. Chevalier, G. / Vooban. Smoothly-Blend-Image-Patches (GitHub). https://github.com/Vooban/Smoothly-Blend-Image-Patches
26. Cira et al. (2024). Tile size and tile overlap effects on road extraction from aerial orthophotos. Remote Sensing 16(16):2954. https://doi.org/10.3390/rs16162954
27. Reed, C.J. et al. (2023). Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning. ICCV 2023. https://arxiv.org/abs/2212.14532
28. FlyAwareV2: A Multimodal Cross-Domain UAV Dataset for Urban Scene Understanding (2025). arXiv:2510.13243. https://arxiv.org/abs/2510.13243
29. OpenStreetMap Wiki. Zoom levels. https://wiki.openstreetmap.org/wiki/Zoom_levels
30. Tallam, K. et al. (2023). Application of Deep Learning for Classification of Intertidal Eelgrass from Drone-Acquired Imagery. Remote Sensing 15(9):2321. https://www.mdpi.com/2072-4292/15/9/2321
31. Hobley, B., Arosio, R., French, G., Bremner, J., Dolphin, T., Mackiewicz, M. (2021). Semi-Supervised Segmentation for Coastal Monitoring Seagrass Using RPA Imagery. Remote Sensing 13(9):1741. https://doi.org/10.3390/rs13091741
32. Peng, J., Li, J., Ingalls, T.C., Schill, S.R., Kerner, H.R., Asner, G.P. (2025). A novel deep learning algorithm for broad scale seagrass extent mapping in shallow coastal environments. ISPRS J. Photogramm. Remote Sens. 220:277–294. https://www.conservationgateway.org/collections/ocean/global-seagrass-mapping-sgdensenet-deep-learning/
33. Scilla et al. (2025). Computer Vision Corrections Enhance UAV-Based Retrievals in Shallow Waters. IEEE JSTARS. https://doaj.org/article/b22656360a6a4f48923675ff802dce01
34. Ultra-high Resolution Image Segmentation via Locality-aware Context Fusion and Alternating Local Enhancement (2021). arXiv:2109.02580. https://arxiv.org/abs/2109.02580 (finding seen only in a search snippet)
35. Sensors 23(22):9193 (2023). Drone marine-fauna detection study comparing a polariser, a control and a green band-pass filter. https://www.mdpi.com/1424-8220/23/22/9193 (statement seen only in a search snippet)
