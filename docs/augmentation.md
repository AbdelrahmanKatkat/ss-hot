# Augmentation: state of the art

2026-11-19. Scope: data augmentation for semantic segmentation of seagrass / benthic / shallow-water habitats from
very-high-resolution (about 1-3 cm/px) RGB drone or aerial imagery, with general remote-sensing (RS) and underwater
evidence where seagrass-specific evidence is missing. Literature review only; nothing here was run on our data.

Marks: **[V]** = read in the source (full text, or the abstract/page for claims stated there).
**[?]** = seen only in a search snippet or secondary summary, or not checked in the source. Numbers are as reported;
none were invented. "Domain" says where the evidence comes from.

---

## 1. Summary

1. **There is no published augmentation ablation for drone-RGB seagrass segmentation.** Seagrass papers list an
   augmentation recipe (flips, small rotations, elastic, brightness/contrast/gamma, blur, noise) but do not measure
   each part [V: Lian et al. 2026, ref 1]. Several seagrass/coral drone papers describe no augmentation at all
   [V: Contini et al. 2025, ref 3]. Every gain below is borrowed from other domains.
2. **Radiometric shift between sites is the main failure, and histogram matching is the best-tested cheap fix for
   seagrass.** UAV (3 cm) to aerial orthophoto (12.5 cm) eelgrass transfer with histogram-matching augmentation +
   SegFormer reached F1/IoU 0.52/0.48 at the best site, 68 % of an oracle trained on both domains (0.76/0.70); the
   worst site stayed near 0.07/0.05 [V: Pawar et al. 2025, ref 2]. In overhead imagery, randomized histogram matching
   is reported to match state-of-the-art unsupervised domain adaptation [V-abstract: Yaras et al. 2024, ref 10].
3. **Colour/brightness/contrast augmentation helps marine imagery more than everyday imagery, but hue and saturation
   help less.** On benthic image transects (classification), rotation, brightness, contrast and shear were the four
   best transforms; hue and saturation gave smaller gains; full flips (p=1) slightly hurt [V: Tan et al. 2022,
   ref 6]. Strong hue shifts risk changing what seagrass looks like; no source tests a hue limit for seagrass.
4. **Dihedral (D4) flips/rotations are the baseline for overhead imagery; their measured gain is small.** Every RS
   ablation found uses them as the baseline [V: refs 11, 12, 14]; when ablated alone they change mIoU by ~0-0.3 points
   [V: SiamSeg, ref 13]. D4 test-time augmentation (TTA) is principled for nadir imagery but measured gains are near
   zero or negative on a well-trained model [? ref 22].
5. **Copy-paste / object-based mixing gives the largest single-augmentation gains in RS segmentation:** +6.2 mIoU
   (37.9 to 44.1, DynamicEarthNet, U-Net) [V: ref 11]; F1 0.788 to 0.829 for buildings [V: ref 12]; ChessMix
   +1.2 to +2.7 mIoU on two of three datasets, no gain on the third [V: ref 14].
6. **With unlabelled imagery, weak-to-strong consistency (FixMatch/UniMatch) is the strongest known use of
   augmentation.** On Pascal with 92 labels, FixMatch went from 39.5 mIoU without strong perturbations to 56.7 with
   CutMix and 63.9 with CutMix + colour transforms [V: ref 17]. In RS, FixMatch gained 7-9 points on a domain-shifted
   riverbed task with 1-10 % labels [V: ref 18].
7. **Resolution robustness (our 5x coarser serving case) is poorly studied.** The only quantified case found: training
   on generated imagery at the target GSD raised tree IoU from 57.43 to 68.05 when moving 50 cm to 20 cm, and from
   50.99 to 67.80 on 8x-degraded input [V: ref 20]. No source compares plain down-up sampling augmentation against
   no augmentation for coarser test GSD in benthic imagery.
8. **Licences:** albumentations stays MIT only up to 2.0.8 (May 2025); its successor AlbumentationsX is AGPL-3.0 or
   commercial [V/?: ref 27]. kornia (Apache-2.0), torchvision (BSD-3-Clause) and TorchGeo (MIT) are permissive
   [? ref 28].

---

## 2. Geometric

**Methods.** Horizontal/vertical flip, 90-degree rotations, transpose (together the D4 group), small free rotation,
scale jitter / random resized crop, elastic or grid distortion, shear.

**Evidence.**
- Standard practice in RS segmentation: flips + 90-degree rotations are the baseline in DynamicEarthNet copy-paste
  [V: ref 11], ChessMix (also transpose, grid/perspective distortion) [V: ref 14], ColorMapGAN [V: ref 9] and
  FixMatch-for-RS (also random distorted crop) [V: ref 18].
- Seagrass: Lian et al. 2026 (UAV RGB/MS, 1.4-3.05 cm GSD, UPerNet-ConvNeXtV2) used h/v flips, rotation within
  +/-10 degrees and elastic deformation, no ablation [V: ref 1]. Huang 2025 (Looking for Seagrass, SAM2) used
  flips, rotation and scaling [V: ref 4].
- Ablation: in SiamSeg's contrastive branch (Potsdam IRRG to Vaihingen IRRG), resize or flip variants scored
  63.0-63.3 mIoU against 65.2-65.6 for the variants with colour jitter; authors: "Resize and Flip have a minimal
  effect" [V: ref 13; table layout partly garbled in extraction].
- Benthic classification: rotation was the best single transform (+8.02 % mAA at 100 degrees, 50 images/class), gain
  shrank to about +3.3 % at 200 images/class; flips at p=1 gave about -1 % [V: ref 6]. Classification, not
  segmentation.
- Scale jitter: the UniMatch weak view resizes by 0.5-2.0, crops and flips [V: ref 17]; a SegFormer RS paper uses
  0.75-1.25 [? ref 21]. Note that resize also changes GSD [? SatDINO, ref 21].

**Typical ranges.** D4 with p=0.5 each; free rotation +/-10 degrees [V: ref 1]; scale 0.5-2.0 [V: ref 17].

**Risks.** Free rotation needs mask-aware padding (ignore label). Large scale-up at 1-3 cm GSD makes sand texture
and seagrass blades look like a different GSD, which may help or hurt depending on serving GSD. Elastic distortion
on fine-boundary classes is unablated in this domain.

---

## 3. Photometric and water-specific

### 3.1 Generic colour jitter

**Evidence.**
- Marine transects (MobileNet-v2 classification): brightness best at factor 2 (+7.74 % mAA), contrast best at 5
  (+7.70 %); saturation peaked about +3.1 %, hue about +5.1 %; custom rotation+brightness+contrast policies beat
  AutoAugment ImageNet policy (+10.86 % vs +9.59 % on PAP50) [V: ref 6]. On Cityscapes the same custom policies were
  negative, so "augmentation effects are domain-dependent" [V: ref 6].
- RS domain adaptation: colour jitter was the only augmentation in the SiamSeg ablation with a clear gain
  (about +2 mIoU) [V: ref 13].
- RS semi/self-supervised: supervised baseline uses colour jitter p=0.5, strength 0.4; SimCLR p=0.8, strength 0.8
  [V: ref 18]. Authors say these augmentations "significantly improve" the baseline under domain shift; the numbers
  are only in a figure [V: ref 18].
- Counter-example: colour jitter hurt a correspondence-based self-supervised pretext task on LandCover.ai
  (val IoU 0.9605 none, 0.9434 flip, 0.8052 jitter, 0.7159 flip+jitter) [V: ref 19]. Pretext-specific; downstream
  differences were small.
- ColorMapGAN argues that colour jitter, gamma or contrast changes give "limited improvement" under large spectral
  shifts; it reports no jitter experiment [V: ref 9].

**Typical ranges.** Strength 0.4 at p=0.5 (supervised), 0.8 at p=0.8 (contrastive) [V: ref 18]; brightness/contrast
factors far beyond torchvision defaults were optimal for benthic images [V: ref 6].

**Risks.** Seagrass vs algae vs dark sand/rock is mostly a colour-and-texture decision. Hue rotations can turn
green-brown seagrass into colours of other classes; no source tests a safe hue range for seagrass. Prefer
brightness/contrast/gamma and per-channel gains (physically plausible) over large hue shifts [inference from refs
6, 7, not tested].

### 3.2 Histogram matching, colour transfer, Fourier style swaps

**Evidence.**
- Seagrass (eelgrass, Denmark): source UAV 3 cm, target aerial orthophoto 12.5 cm; histogram matching of source
  chips to random target references before SegFormer training. Leave-one-out F1/IoU: 0.32/0.31, 0.52/0.48,
  0.47/0.43 at three sites; oracle 0.76/0.70. Bhattacharyya distance between histograms dropped from 0.81 (about 68 %
  reduction). Matching a low-contrast reference to segment a higher-contrast image worked better than the reverse.
  Sparse seagrass on sand produced "extreme spectral dissimilarity" that degraded matching [V: ref 2]. The paper
  gives no source-only (no-matching) baseline, so the gain from matching itself is not isolated [V: ref 2].
- Overhead buildings: randomized histogram matching (RHM) "similar or superior" to state-of-the-art UDA and much
  better than other simple methods [V-abstract: ref 10; numbers not read].
- ColorMapGAN (Pléiades 1 m, U-Net): no adaptation vs histogram matching vs CycleGAN vs ColorMapGAN overall IoU:
  21.68 / 38.63 / 46.95 / 48.40 (Bad Ischl to Villach) and 15.03 / 11.11 / 14.22 / 42.88 (Béziers to Roanne).
  Histogram matching helped in one pair and hurt in the other; gray-world normalisation alone gave 22.6-36.7
  [V: ref 9].
- FDA swaps the low-frequency amplitude spectrum between domains, one parameter beta, no training [V-abstract:
  ref 23]. No benthic result found.
- Benthic: colour normalisation raised a seafloor detector from 56.6 % to 74.1 % on low-altitude data
  [? ref 8; metric unclear]. Colour enhancement gave no overall gain for coral substrate annotation [? ref 24].

**Typical setting.** Per-channel CDF matching to a random target or random training image, applied with some
probability during training [V: refs 2, 10].

**Risks.** Whole-image matching assumes similar class proportions; a chip that is mostly sand matched to a chip
that is mostly seagrass shifts colours wrongly (rooftops turned cyan in ref 9; sparse seagrass in ref 2).

### 3.3 Physically based water effects (attenuation, turbidity, glint, caustics, waves)

**Evidence.**
- Image formation: Akkaynak & Treibitz show that the coefficients governing backscatter differ from those governing
  direct transmission, and Sea-thru that attenuation depends on range [V-abstract: refs 25, 26]. These define what a
  physically valid "depth/turbidity" augmentation must vary: per-channel exponential attenuation with depth plus an
  additive veiling light.
- Physics-based generators exist but report no segmentation gains: AquaFuse (waterbody fusion between images,
  geometry-preserving; augmentation only named as future work) [V: ref 29]; Sea-ing Through Scattered Rays (adds
  forward scattering, BUCKET turbidity dataset) [? ref 30]; Jerlov-water-type synthesis for enhancement [? ref 31].
  These are mostly side-looking underwater imagery, not nadir through-surface imagery.
- Nadir through-water: Sea-Undistort renders 1,200 paired 512x512 Blender scenes with sun glint, waves and
  scattering, used for bathymetry/restoration, not benthic segmentation [? ref 32]. Drone glint detection/removal
  exists (U-Net glint detector; WACV 2024 pipeline) [? refs 33, 34]; UAV coral classification was more accurate after
  Lyzenga-type deglinting applied only to glint regions [V: ref 35].
- Kelp-o-Tron uses generic flips, CLAHE and blur described as simulating turbidity/illumination, not a physical model
  [? ref 36].
- Walker et al. 2019 tested physics-based corrections plus augmentation for DeepLabV3 transfer on benthic imagery
  [? ref 8; full text not reached].

**Typical ranges.** Not reported in any segmentation ablation found.

**Risks.** Synthetic glint/caustics that cover seagrass with bright patterns while the mask still says "seagrass"
may teach the model to ignore glint (desired) or to label glint as seagrass; untested. Depth attenuation applied to
the whole chip ignores that depth varies inside a chip.

---

## 4. Degradation and resolution

**Methods.** Down-then-up sampling, Gaussian/motion blur, noise, JPEG, high-order degradation chains (Real-ESRGAN
style), multi-GSD training, super-resolution or generative translation to the target GSD.

**Evidence.**
- Seagrass: Lian et al. used blur and noise injection, local contrast and edge enhancement, no ablation [V: ref 1].
  Pawar et al. cross 3 cm to 12.5 cm GSD, but the method addresses colour, not resolution [V: ref 2].
- Trees (SegFormer, 20 vs 50 cm/px): source-only 50 to 20 cm IoU 57.43; pix2pix-generated training 68.05;
  Real-ESRGAN 63.92; Latent Diffusion 65.36; Stable Diffusion SR 62.73. On 8x-degraded input 50.99 vs 67.80. Plain
  resize augmentation was not isolated; the paper's "over 50 %" claim does not match its table (about 33 %)
  [V: ref 20].
- Real-ESRGAN: a multi-step ("high-order") degradation process with ringing/overshoot artefacts to simulate real
  degradations [V-abstract: ref 37]. Widely reused as an augmentation recipe; no benthic segmentation test found.
- Scale-aware RS segmentation (SaNet, Scale Aware Adaptation) and cross-scale pretraining exist, but none directly
  measures multi-GSD augmentation at an unseen GSD [? refs 21, 38].

**Typical setting.** Not established for benthic imagery. A down-up factor range matching the expected serving GSD
(here up to about 5x) follows from the problem, not from a source.

**Risks.** At 5x coarser, seagrass blades and patch edges vanish; labels drawn at 1-3 cm may not be learnable at
5-15 cm (label/resolution mismatch, ref 21 Workman et al.). Heavy blur on all chips can reduce accuracy at native
GSD; probability should be tuned.

---

## 5. Mixing methods

**Evidence.**
- Copy-paste for semantic segmentation (instances from connected components): DynamicEarthNet, U-Net, test mIoU
  37.9 to 44.1 (N=100 pasted, no pre-paste augmentation); 42.3 with pre-paste augmentation; validation 34.1 to 36.0
  [V: ref 11].
- Object-based augmentation (paste objects with synthetic shadows on new backgrounds): F1 0.788 to 0.829 (U-Net
  ResNet-34), 0.741 to 0.802 (FPN ResNet-18) [V: ref 12].
- ChessMix: mIoU 0.601 to 0.613 (Vaihingen), 0.809 to 0.836 (Thetford), 0.616 to 0.616 (Coffee) [V: ref 14].
- Simple Copy-Paste (instance segmentation, COCO): +0.6 mask AP over previous best [V: ref 15].
- ClassMix: mixes unlabelled images along predicted class masks; state of the art on two semi-supervised benchmarks
  at the time [V-abstract: ref 16].
- CutMix alone took FixMatch on Pascal (92 labels) from 39.5 to 56.7 mIoU [V: ref 17].
- Mosaic (YOLOv4) is common in detection; no segmentation-in-RS ablation found [not checked].

**Risks.** Seagrass meadows are continuous textures, not objects; pasted seagrass patches on sand create sharp,
unnatural edges and colour mismatches (different depth/water colour). Pasting between sites with different water
colour may be unrealistic unless colour-matched first.

---

## 6. Automatic and generative augmentation

**Evidence.**
- AutoAugment/RandAugment/TrivialAugment: SmartAugment compared them for segmentation on medical, bridge-inspection
  and KITTI data; no RS dataset [? ref 39]. On benthic transects a hand-made rotation+brightness+contrast policy
  beat AutoAugment ImageNet/CIFAR policies [V: ref 6]. No RS-segmentation benchmark of automatic policies found.
- Diffusion image-mask generation (SatSynth, CVPR 2024) reports gains over training on real data alone for aerial
  segmentation; numbers not read [V-abstract: ref 40]. SAR oil-spill diffusion augmentation: SegFormer 67.46 to
  70.74 mIoU [? ref 41].
- GAN domain translation: CycleGAN vs ColorMapGAN numbers in 3.2; CycleGAN was close to no adaptation in one pair
  [V: ref 9]. RHM reported similar to ColorMapGAN/CycleGAN-class methods at much lower cost [V: ref 2 citing ref 10].
- Generative GSD translation: section 4 [V: ref 20].

**Risks.** Generated pixels may not match masks at blade-level detail; generators trained on everyday imagery do not
know seagrass. Cost and licence of diffusion models.

---

## 7. Semi- and self-supervised methods built on strong augmentation

**Evidence.**
- UniMatch (CVPR 2023): weak view = resize 0.5-2.0, crop, flip; strong view = ST++ colour transforms + CutMix;
  feature perturbation = channel dropout 0.5. FixMatch on Pascal, labels 92/183/366/732/1464: no strong perturbation
  39.5/52.7/65.5/69.2/74.6; CutMix 56.7/67.9/71.9/75.1/78.3; full 63.9/73.0/75.5/77.8/79.2 mIoU [V: ref 17]. RS
  results only for change detection [V: ref 17].
- FixMatch for RS segmentation: riverbed domain-shifted +7 points (1 % labels) and +9.1 (10 %); Chesapeake
  +1.25-4.1; Sen1Floods11 +3.6-3.8 at full labels under domain shift [V: ref 18]. Strong view = two AutoAugment-style
  ops + multi-rectangle cutout [V: ref 18].
- Seagrass: Hobley et al. 2021, semi-supervised FCNN on a drone survey (Budle Bay, UK), comparable to OBIA, better
  for sparsely labelled features [? ref 5].
- Newer RS methods (MUCA, Co2S, S5) build on UniMatch-style weak-to-strong consistency on Potsdam/LoveDA [? ref 42].

**Risks.** Needs unlabelled imagery from the target conditions (new sites/cameras), which is the real benefit;
pseudo-labels on glinty or turbid chips can reinforce errors.

---

## 8. Test-time augmentation

**Evidence.**
- D4 TTA is argued for nadir imagery (no canonical orientation); the albumentations aerial TTA notebook (10 cm tree
  cover, SegFormer) predicts gains but shows no measured numbers [V: ref 43].
- Martian landslide segmentation: 2-view flip +0.0005 mIoU, 4-view rotation -0.0007, 8-view D4 -0.0047 [? ref 22].
- Burned-area delineation: TTA improves results at higher inference cost [? ref 22].

**Typical setting.** 2 (flip), 4 (rot90) or 8 (D4) views, average probabilities after inverse transform.

**Risks.** 2-8x inference cost on large orthomosaics; gain may be zero for a model trained with D4. Multi-scale TTA
for GSD mismatch is untested in this domain.

---

## 9. Comparison table

| Augmentation | Evidence (best found) | Domain of evidence | Typical setting | Sources |
|---|---|---|---|---|
| D4 flips/rot90 | ~0 to +0.3 mIoU when ablated; universal baseline | RS UDA, RS seg | p=0.5 each | 11, 12, 13, 14 |
| Small rotation / shear | Rotation best single op, +8.02 % mAA (50/class) | Benthic classification | +/-10 deg (seagrass); larger in benthic | 1, 6 |
| Scale jitter / crop | Standard, not ablated in domain | Natural, RS | 0.5-2.0 or 0.75-1.25 | 17, 21 |
| Elastic / grid | Used, not ablated | Seagrass UAV, RS | n/a | 1, 14 |
| Brightness / contrast / gamma | +7.7 % mAA each (benthic); jitter +2 mIoU (RS UDA) | Benthic cls, RS UDA | strength 0.4 p=0.5 | 6, 13, 18 |
| Hue / saturation | Smaller gains (+3-5 % mAA) | Benthic cls | small | 6 |
| Histogram matching | Seagrass F1 up to 68 % of oracle; RS UDA mixed (+17 or -4 IoU) | Seagrass aerial, RS | random target reference | 2, 9, 10 |
| Fourier style swap (FDA) | SOTA on synthetic-to-real | Driving | small beta | 23 |
| Physical water model | No segmentation gain reported | Underwater | n/a | 25, 26, 29-32 |
| Glint / caustic synthesis | No segmentation gain reported | Nadir water rendering | n/a | 32-35 |
| Blur / noise / JPEG / down-up | Used; not isolated | Seagrass UAV, SR | n/a | 1, 37 |
| Generative GSD translation | Tree IoU 57.43 to 68.05 | Aerial trees | pix2pix | 20 |
| Copy-paste / object paste | +6.2 mIoU; F1 +0.04-0.06 | Satellite, VHR buildings | N=100 instances | 11, 12 |
| ChessMix | +0 to +2.7 mIoU | Aerial | 1,000 synthetic images | 14 |
| CutMix / ClassMix (semi-sup.) | +17 mIoU (92 labels) | Pascal | p=0.5 | 16, 17 |
| FixMatch / UniMatch | +7-9 points (RS, domain shift, 1-10 % labels) | RS seg | weak/strong views | 17, 18 |
| AutoAugment-type | Beaten by hand policy on benthic data | Benthic cls | n/a | 6, 39 |
| Diffusion image-mask | Gains reported, numbers not read | Aerial seg | n/a | 40, 41 |
| D4 TTA | ~0 or negative to small positive | RS (various) | 2-8 views | 22, 43 |

---

## 10. Gaps

- No controlled augmentation ablation for seagrass or benthic **segmentation** from nadir RGB drone imagery; the best
  marine ablation is for classification of side/downward camera transects (ref 6).
- No test of hue/saturation limits that keep seagrass colour realistic.
- No segmentation result for physically based depth-attenuation, turbidity, glint or caustic augmentation.
- No measurement of down-up sampling augmentation against a coarser test GSD (our 5x serving case) in any water
  domain; the only quantified GSD study uses generative translation on trees (ref 20).
- Seagrass histogram-matching study lacks a no-matching baseline (ref 2).
- RHM, SatSynth and several RS TTA numbers were not read in full (marked [V-abstract] or [?]).
- Conclusion: our own ablations (held-out site/date, coarser GSD test) are needed; the literature gives only priors.

---

## 11. Sources

1. Lian J., Mo Z., Liu Z., Peng B., Chang M., Wang X., Wang W. (2026). Intertidal Seagrass Mapping Using UAV Visible and Multispectral Imagery: A Comparative Semantic Segmentation Study with Explainability Analysis. Remote Sensing 18(12):2057. https://www.mdpi.com/2072-4292/18/12/2057 [V]
2. Pawar S., Thomasberger A., Bengtson S.H., Pedersen M., Timmermann K. (2025). Eye in the Sky for Sub-Tidal Seagrass Mapping: Leveraging Unsupervised Domain Adaptation with SegFormer for Multi-Source and Multi-Resolution Aerial Imagery. Remote Sensing 17(14):2518. https://doi.org/10.3390/rs17142518 (PDF: https://backend.orbit.dtu.dk/ws/files/410545772/remotesensing-17-02518.pdf) [V]
3. Contini M. et al. (2025). The point is the mask: scaling coral reef segmentation with weak supervision. arXiv:2508.18958 / ICCVW 2025. https://arxiv.org/html/2508.18958v1 [V]
4. Huang S. (2025). Advancing Seagrass semantic segmentation with SAM2 models. Journal of High School Science. https://jhss.scholasticahq.com/article/138532 [V]
5. Hobley B., Arosio R., French G., Bremner J., Dolphin T., Mackiewicz M. (2021). Semi-Supervised Segmentation for Coastal Monitoring Seagrass Using RPA Imagery. Remote Sensing 13(9):1741. https://www.doi.org/10.3390/RS13091741 [?]
6. Tan M., Langenkämper D., Nattkemper T.W. (2022). The Impact of Data Augmentations on Deep Learning-Based Marine Object Classification in Benthic Image Transects. Sensors 22(14):5383. https://pmc.ncbi.nlm.nih.gov/articles/PMC9322900/ [V]
7. Tallam K. et al. (2023). Application of Deep Learning for Classification of Intertidal Eelgrass from Drone-Acquired Imagery. Remote Sensing 15(9):2321 (DeepLab, ~3.35 cm/px, F1 0.809). https://mdpi.com/2072-4292/15/9/2321 [V]
8. Walker J., Prügel-Bennett A., Thornton B., Yamada T. (2019). The Effect of Physics-Based Corrections and Data Augmentation on Transfer Learning for Segmentation of Benthic Imagery. IEEE UT 2019; and Walker J. (2023) PhD thesis, Univ. Southampton. https://eprints.soton.ac.uk/428954 ; https://eprints.soton.ac.uk/482390/ [?]
9. Tasar O., Happy S.L., Tarabalka Y., Alliez P. (2020). ColorMapGAN: Unsupervised Domain Adaptation for Semantic Segmentation Using Color Mapping GANs. IEEE TGRS. https://arxiv.org/abs/1907.12859 [V]
10. Yaras C., Kassaw K., Huang B., Bradbury K., Malof J.M. (2024). Randomized Histogram Matching: A Simple Augmentation for Unsupervised Domain Adaptation in Overhead Imagery. IEEE JSTARS 17:1988-1998. https://arxiv.org/abs/2104.14032 [V-abstract]
11. Motoi I.M., Saraceni L., Nardi D., Ciarfuglia T.A. (2024). Evaluating the Efficacy of Cut-and-Paste Data Augmentation in Semantic Segmentation for Satellite Imagery. IGARSS 2024. https://arxiv.org/html/2404.05693v2 [V]
12. Illarionova S. et al. (2021). Object-Based Augmentation Improves Quality of Remote Sensing Semantic Segmentation. arXiv:2105.05516 (ICCVW 2021). https://ar5iv.labs.arxiv.org/html/2105.05516 [V]
13. Wang B., Deng F., Wang S., Luo W., Zhang Z., Jiang P. (2024). SiamSeg: Self-Training with Contrastive Learning for Unsupervised Domain Adaptation Semantic Segmentation in Remote Sensing. arXiv:2410.13471. https://arxiv.org/abs/2410.13471 [V]
14. Pereira M.B., dos Santos J.A. (2021). ChessMix: Spatial Context Data Augmentation for Remote Sensing Semantic Segmentation. arXiv:2108.11535. https://ar5iv.labs.arxiv.org/html/2108.11535 [V]
15. Ghiasi G. et al. (2021). Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. CVPR 2021. https://arxiv.org/abs/2012.07177 [V-abstract]
16. Olsson V., Tranheden W., Pinto J., Svensson L. (2021). ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning. WACV 2021. https://openaccess.thecvf.com/content/WACV2021/html/Olsson_ClassMix_Segmentation-Based_Data_Augmentation_for_Semi-Supervised_Learning_WACV_2021_paper.html [?]
17. Yang L., Qi L., Feng L., Zhang W., Shi Y. (2023). Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation (UniMatch). CVPR 2023. https://arxiv.org/abs/2208.09910 [V]
18. Patel C., Sharma S., Pasquarella V.J., Gulshan V. (2021/2022). Evaluating Self and Semi-Supervised Methods for Remote Sensing Segmentation Tasks. arXiv:2111.10079. https://arxiv.org/html/2111.10079 [V]
19. Sharma L., Marin A. (2026). Subimage Overlap Prediction: Task-Aligned Self-Supervised Pretraining for Semantic Segmentation in Remote Sensing Imagery. arXiv:2601.01781. https://arxiv.org/abs/2601.01781 [V]
20. dos Santos Ferreira A., Ramos A.P.M., Marcato Junior J., Gonçalves W.N. (2025). Data Augmentation and Resolution Enhancement using GANs and Diffusion Models for Tree Segmentation. arXiv:2505.15077. https://arxiv.org/html/2505.15077 [V]
21. Resolution/scale references: Workman S. et al., Handling Image and Label Resolution Mismatch in Remote Sensing, arXiv:2211.15790 https://arxiv.org/abs/2211.15790 ; SatDINO, arXiv:2508.21402 https://arxiv.org/html/2508.21402v1 ; SegFormer-based overhead segmentation, Scientific Reports 2026 https://www.nature.com/articles/s41598-026-54946-7 [?]
22. TTA reports: DualSwinFusionSeg (Martian landslides), arXiv:2603.14132 https://arxiv.org/pdf/2603.14132 ; burned-area delineation, arXiv:2512.12056 https://www.alphaxiv.org/abs/2512.12056 [?]
23. Yang Y., Soatto S. (2020). FDA: Fourier Domain Adaptation for Semantic Segmentation. CVPR 2020. https://arxiv.org/abs/2004.05498 [V-abstract]
24. Pixelwise annotation of coral reef substrates (ImageCLEFcoral 2021 working note). CEUR-WS Vol-2936 paper 119. https://ceur-ws.org/Vol-2936/paper-119.pdf [?]
25. Akkaynak D., Treibitz T. (2018). A Revised Underwater Image Formation Model. CVPR 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.html [V-abstract]
26. Akkaynak D., Treibitz T. (2019). Sea-thru: A Method for Removing Water From Underwater Images. CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf [?]
27. Albumentations licensing: AlbumentationsX dual licensing blog (2025) https://albumentations.ai/blog/2025/01-albumentationsx-dual-licensing/ ; licence page https://albumentations.ai/docs/license/ ; original repo https://github.com/albumentations-team/albumentations ; Buslaev A. et al. (2020) Albumentations: fast and flexible image augmentations, Information 11(2):125, https://arxiv.org/abs/1809.06839 [? licence details from search summary; paper V-abstract]
28. Library licences: kornia (Apache-2.0) https://pypi.org/project/kornia/ ; TorchGeo (MIT) https://lists.osgeo.org/pipermail/announce/2025-November/000661.html ; torchvision (BSD-3-Clause; some pretrained weights carry other terms) https://pypi.python.org/pypi/torchvision [?]
29. Siddique M.A.B., Wu J., Rekleitis I., Islam M.J. (2024). AquaFuse: Waterbody Fusion for Physics Guided View Synthesis of Underwater Scenes. arXiv:2411.01119. https://arxiv.org/abs/2411.01119 [V-abstract]
30. Sea-ing Through Scattered Rays: Revisiting the Image Formation Model for Realistic Underwater Image Generation (2025). arXiv:2509.15011. https://arxiv.org/html/2509.15011v2 [?]
31. π-SUB: A Physics-Informed Synthetic Underwater Benchmark Dataset (2026). arXiv:2608.10589. https://arxiv.org/pdf/2608.10589 ; Uplavikar et al., All-In-One Underwater Image Enhancement using Domain-Adversarial Learning, arXiv:1905.13342 [?]
32. Kromer M., Agrafiotis P., Demir B. (2025). Sea-Undistort (synthetic sun glint/wave/scattering pairs). arXiv:2508.07760. https://arxiv.org/abs/2508.07760 [?]
33. Giles, Davies, Ren et al. Deep-learning sun-glint detection in high-resolution aerial imagery. https://marineinfo.org/doc/publication/366349 [?]
34. Qin et al. (2024). Revolutionize the Oceanic Drone RGB Imagery with Pioneering Sun Glint Detection and Removal Techniques. WACV 2024. https://openaccess.thecvf.com/content/WACV2024/papers/Qin_Revolutionize_the_Oceanic_Drone_RGB_Imagery_With_Pioneering_Sun_Glint_WACV_2024_paper.pdf [?]
35. Muslim A.M., Chong W.S., Safuan C.D.M., Khalil I., Hossain M.S. (2019). Coral Reef Mapping of UAV: A Comparison of Sun Glint Correction Methods. Remote Sensing 11:2422. https://doi.org/10.3390/rs11202422 [V-abstract]
36. Comparative Assessment of Deep Learning Architectures for Underwater Subsurface Kelp Forest Segmentation with The Kelp-o-Tron (2026). arXiv:2608.24594. https://arxiv.org/pdf/2608.24594 [?]
37. Wang X., Xie L., Dong C., Shan Y. (2021). Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. arXiv:2107.10833. https://arxiv.org/abs/2107.10833 [V-abstract]
38. Deng X. et al. (2021). Scale Aware Adaptation for Land-Cover Classification in Remote Sensing Imagery. WACV 2021. https://arxiv.org/abs/2012.04222 ; Wang L. et al., SaNet, arXiv:2103.07935 https://arxiv.org/abs/2103.07935 [?]
39. Negassi M. et al. Smart(Sampling)Augment: Optimal and Efficient Data Augmentation for Semantic Segmentation. arXiv:2111.00487 / Algorithms 15(5):165. https://arxiv.org/abs/2111.00487 [?]
40. Toker A., Eisenberger M., Cremers D., Leal-Taixé L. (2024). SatSynth: Augmenting Image-Mask Pairs through Diffusion Models for Aerial Semantic Segmentation. CVPR 2024. https://arxiv.org/abs/2403.16605 [V-abstract]
41. Diffusion-based augmentation for SAR oil-spill segmentation (2024). arXiv:2412.08116. https://arxiv.org/pdf/2412.08116 [?]
42. RS semi-supervised follow-ups: MUCA arXiv:2501.10736 https://arxiv.org/html/2501.10736v3 ; Co2S arXiv:2512.23035 https://arxiv.org/pdf/2512.23035v2 ; S5 arXiv:2508.12409 https://arxiv.org/pdf/2508.12409v3 [?]
43. Albumentations docs: D4 TTA for aerial segmentation example. https://albumentations.ai/docs/examples/example-tta-d4-aerial-segmentation/ [V]
