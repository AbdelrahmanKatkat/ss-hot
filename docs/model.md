# Models: state of the art

2026-11-19. Literature review of models for binary seagrass segmentation from very-high-resolution (VHR) RGB drone imagery, judged against our constraints: commercial-use licence for code and weights, ONNX export, CPU serving, fine-tuning on one GPU. Our setting is 256 px web-map tiles at zoom 24 (~0.92 cm/px), ~13,700 labelled training tiles from one country, shallow tropical water. Nothing was installed, trained or measured for this review.

Markers: **[V]** = read in the source (abstract, full text, model card, licence file or API metadata); **[?]** = seen only in a search snippet or secondary summary, not confirmed. Bracketed numbers like [1] point to section 9.

> **Update 2026-11-20:** non-commercial licences are acceptable for this project (decision by the project lead). The licence findings below stay as facts, but they no longer rule a model out: ConvNeXt **V2**-T weights (the encoder of Lian et al. [1]) and SegFormer/MiT become usable options. The model stays UPerNet + ConvNeXt-Tiny v1 for now; V2-T is the natural next encoder to try against it.

## 2. Summary

### Key findings

1. **No published benchmark matches our setting.** No paper reports RGB-only, ~1 cm, tropical, subtidal seagrass segmentation with modern backbones. The closest are Lian et al. 2026 [1] (intertidal, China, 1.4-3.05 cm, 9-channel RGB+multispectral+indices, 256 px patches, 3,840 training patches) and Tahara et al. 2022 [6] (0.81 cm RGB, 256 px tiles, 980 training tiles). Published seagrass numbers use different data, classes and metrics, so they cannot be compared with each other. [V]
2. **UPerNet + ConvNeXt-Tiny is the best published seagrass result we found.** In Lian et al. 2026, UPerNet + ConvNeXtV2-Tiny (34.1 M params) reached 94.63 ± 1.28 % mIoU, against 90.13 % for U-Net-ResNet101 and 88.88 % for DeepLabV3+-ResNet101 (5 seeds, paired t-tests). In the ablation both parts helped: the UPerNet decoder added 4.9 points over U-Net with the same encoder, and the ConvNeXtV2-T encoder added 5.4 over ResNet101 with the same decoder. Caveat: the input was 9-channel, not RGB, and only CNN baselines were compared (no Swin, SegFormer or foundation model). [V] [1]
3. **Older seagrass work used U-Net, FPN, DeepLab and SegFormer, and simple models often did well.** Jeon 2021: U-Net beat PSPNet, SegNet and DeepLabv3+ [5]. Chen & Sasaki 2021: FPN (IoU 0.848) beat U-Net [4]. Tallam 2023: DeepLabV3-ResNet50, F1 0.809 [3]. Pawar 2025: SegFormer, in-domain ("oracle") F1 0.76 / IoU 0.70 on subtidal eelgrass [2]. [V]
4. **On generic VHR benchmarks, Tiny-size CNNs and transformers are close; backbone size and pretraining matter more than the family.** FLAIR-HUB aerial RGB (20 cm), UPerNet head, mIoU: ConvNeXtV2-Tiny 62.7, MiT-B2 62.7, Swin-T 62.2, HRNet32 61.2, ResNet50 58.9, all at ~25-33 M params. Base-size backbones add ~1-2 points [24]. On ADE20K, UPerNet + ConvNeXt-T (46.0/46.7) slightly beats UPerNet + Swin-T (45.8) at the same 60 M params and ~940 GFLOPs [18]. [V]
5. **Remote-sensing foundation models do not reliably beat a supervised U-Net** (PANGAEA [25]). Most were pretrained on 10-30 m Sentinel/Landsat bands, far from our 1 cm RGB. The ones relevant to VHR RGB are DINOv3 (Sat ViT-L on 0.6 m Maxar RGB, plus web-image ConvNeXt variants) [26], SAM/SAM2, Scale-MAE and RemoteCLIP. Several have non-commercial or custom licences: Scale-MAE and SatMAE are CC BY-NC; SAM3 and DINOv3 use custom, gated Meta licences. [V]
6. **Licences rule out several popular choices.** SegFormer/MiT code and weights are NVIDIA non-commercial, including the MiT encoder inside `segmentation_models_pytorch` [20][30]. ConvNeXt **V2** ImageNet weights are CC BY-NC 4.0 [19]. ConvNeXt **V1** code is MIT and its timm weights are Apache-2.0 [18][27]. This is the main reason to use v1 instead of the v2 that Lian et al. used. [V]
7. **The deployment risk is in the decoder, not the backbone.** UPerNet's pyramid pooling module (PPM) uses `AdaptiveAvgPool2d` with bins (1, 2, 3, 6) [30]. At 256 px input the stride-32 map is 8 × 8, and 3 and 6 do not divide 8. The legacy PyTorch ONNX exporter is reported to fail on adaptive pooling when the output size is not a factor of the input size [?]. smp fixed exactly this failure for PSPNet in 2026 (PR #1270) [V] [30]. Mask2Former's multi-scale deformable attention is a custom CUDA op that blocks plain ONNX export [?] [31].
8. **The UPerNet head dominates compute.** ConvNeXt-T alone is 4.5 GFLOPs at 224² [18]. UPerNet + ConvNeXt-T in mmseg is 939 GFLOPs at 2048 × 512 [18], which is ≈ 59 GFLOPs at 256² by linear scaling (our arithmetic, not measured). For comparison: SegFormer-B2 is 62.4 GFLOPs at 512² (≈ 15.6 at 256²) [20], and SegNeXt-B is 35 GFLOPs at 512² (≈ 8.8 at 256²) [21]. Lian's 34 M-param UPerNet suggests a slimmer head than mmseg's 60 M configuration [1].

### Ranked shortlist for our constraints

| Rank | Candidate | Why |
|---|---|---|
| 1 | **UPerNet + ConvNeXt-Tiny v1** (ImageNet-22k → 1k, timm Apache-2.0 / repo MIT), the current team choice | Closest architecture to the best published seagrass result [1]. Matches transformers of the same size on VHR aerial imagery [24] and ADE20K [18]. Clean licence, plain CNN ops. Risks: the evidence is for v2 on multispectral input, not v1 on RGB; the PPM's adaptive pooling at 256 px needs an ONNX check; the head is the main CPU cost. |
| 2 | **U-Net + ConvNeXt-Tiny v1 (or EfficientNet/ResNet), via smp** (MIT) | The strongest simple baseline: top-2 most often in PANGAEA [25], best in Jeon 2021 [5], and best of the ConvNeXtV2 decoders on FLAIR-HUB [24]. No adaptive pooling, so it is the easiest ONNX/CPU path. But in Lian it was ~4.9 mIoU points below UPerNet with the same encoder [1]. Train it as the control. |
| 3 | **The same decoders on a DINOv3 ConvNeXt-Tiny backbone** (29 M, self-supervised on web images, LVD-1689M) | State-of-the-art self-supervised pretraining, same CNN ops (ONNX-friendly), no change to the head. Custom DINOv3 licence: royalty-free grant, mandatory "Built with DINOv3" attribution, use restrictions, gated download [26][27]. Needs a licence review first. |
| 4 | **UPerNet + Swin-Tiny** (MIT) | The transformer alternative at the same size: 62.2 vs ConvNeXtV2-T 62.7 on FLAIR-HUB [24], 45.8 vs 46.0 on ADE20K [18]. Window attention exports to ONNX but is less CPU-friendly than convolution [?]. Worth trying only if ConvNeXt underfits. |
| 5 | **SAM2 Hiera encoder + U-Net decoder (SAM2-UNet)** (SAM2 weights Apache-2.0) | Best reported seagrass IoU on underwater LookingForSeagrass data (0.923/0.935; single author, high-school journal) [9]. The licence is fine, but the evidence is weak, the encoder is larger, and ONNX/CPU behaviour is unverified. Research option only. |

Excluded:

| Model | Reason |
|---|---|
| SegFormer/MiT | non-commercial licence [20] |
| ConvNeXt V2 ImageNet weights | CC BY-NC [19] |
| Scale-MAE, SatMAE | CC BY-NC [28] |
| Mask2Former | ONNX problem with deformable attention; more than a binary task needs [31] |
| Mamba/RS-Mamba backbones | selective-scan CUDA kernel, no standard ONNX path [?] |
| SegNeXt | Apache-2.0, but the README says "for commercial use, please contact the authors" [21] |

**Verdict on UPerNet + ConvNeXt-Tiny v1:** a well-supported, licence-clean choice and the architecture with the strongest seagrass evidence. It is not proven state of the art for our data: the gains in [1] were measured on multispectral input against weak baselines. Keep it at rank 1 on two conditions: it beats a U-Net + ConvNeXt-T control on validation tiles from held-out sites, and its ONNX export at 256 px passes a check for the PPM pooling issue.

## 3. Seagrass / benthic literature

| Paper | Year | Platform / GSD | Model | Data | Metric | Source |
|---|---|---|---|---|---|---|
| Lian et al., Intertidal seagrass mapping using UAV visible and multispectral imagery | 2026 | UAV, 1.40-3.05 cm; 9-channel input (RGB + MS + NDVI/NDWI) | UPerNet + ConvNeXtV2-T (34.1 M) vs U-Net-R101 and DeepLabV3+-R101; CE + Lovász-Softmax (λ = 0.6); AdamW 2e-4; 200 epochs; batch 24 | China; 3 classes (background, exposed, submerged seagrass); 256 px patches; 3,840 train / 1,512 val / 1,045 test / 1,043 independent-region test | mIoU: UPerNet-CNXv2-T 94.63 ± 1.28; U-Net-CNXv2-T 90.13; U-Net-R101 89.71; UPerNet-R101 89.25; DLV3+-R101 88.88 [V] | [1] |
| Pawar et al., Eye in the Sky for sub-tidal seagrass mapping | 2025 | UAV 3 cm → aircraft 12.5 cm, RGB | SegFormer (ImageNet-1k MiT) + histogram matching (domain adaptation) | 3 Danish estuaries, eelgrass; 512 px samples; oracle trained on 3,560 images | Oracle F1 0.76 / IoU 0.70; best leave-one-out F1 0.52 / IoU 0.48 (68 % of oracle) [V] | [2] |
| Tallam et al., Deep learning for intertidal eelgrass from drone imagery | 2023 | UAV ~3.35 cm RGB | DeepLabV3, ResNet50 | Morro Bay, CA; 2 years | recall 0.954, precision 0.723, F1 0.809 [V] | [3] |
| Chen & Sasaki, FPN for sub- and intertidal seagrass from UAV orthophotos | 2021 | UAV RGB (DJI Phantom 4 Pro) | FPN vs U-Net | Futtsu tidal flat, Tokyo Bay | FPN OA 0.957, F1 0.918, IoU 0.848; better than U-Net [V abstract] | [4] |
| Jeon et al., Semantic segmentation of seagrass habitat from drone imagery | 2021 | UAV RGB | U-Net, SegNet, PSPNet, DeepLabv3+ | Korea | ranking U-Net > PSPNet > SegNet > DeepLabv3+; numbers not read [V abstract via listing] | [5] |
| Tahara et al., Species-level seagrass mapping with UAV and deep learning | 2022 | UAV 0.81 cm RGB | pix2pix (cGAN) | Saroma-ko Lagoon, Japan; 256 px tiles; 980 train (3,920 with flips) | OA 0.818, kappa 0.618 (object-based SVM: 0.719) [V] | [6] |
| Hobley et al., Semi-supervised segmentation for coastal seagrass from RPA imagery | 2021 | UAV RGB | FCNN, semi-supervised (sparse labels) vs OBIA | Budle Bay, UK | FCNN on par with OBIA, better on sparsely labelled features [?] | [7] |
| Weidmann et al., A closer look at seagrass meadows (LookingForSeagrass) | 2019 | underwater camera | DeepLabv3+, FCN, dilNet, U-Net | Baltic; LookingForSeagrass | DeepLabv3+ best, mIoU 87.78 % [?] | [8] |
| Huang, Advancing seagrass segmentation with SAM2 models | 2025 | underwater (LookingForSeagrass) | SAM2 variants; SAM2-UNet (Hiera encoder); BCE + IoU loss | 2 depth subsets | SAM2-UNet IoU 0.923 / 0.935, PA 0.967 / 0.972 [V abstract]; high-school journal | [9] |
| Langlois et al., Drone-based monitoring of tropical seagrass | 2026 | UAV orthomosaics | UNet++ | 3 tropical intertidal meadows, N Australia | no IoU/F1; weak below ~10 % cover of *Halodule* [V abstract] | [10] |
| Oiry et al., Seagrass vs green macroalgae from multispectral drone imagery | 2024 | UAV, 10-band MS | neural network classifier | France, Portugal | OA 94 %; seagrass 96.4 % [V abstract] | [11] |
| Gonzalez-Perez et al., Coastal wetland classification from UAS | 2022 | UAS MS + lidar/SfM | U-Net, DeepLabv3, SVM, RF | W Florida, 17 classes | OA: U-Net 83.8-85.3 %, DeepLabv3 75.2-83.5 %; for deep learning, RGB about as good as MS [V abstract] | [17] |
| Sauder et al., Coralscapes | 2025 | underwater RGB (GoPro), 1024 × 2048 | UNet++/DLV3+ R50, SegFormer b2/b5, DINOv2 linear/DPT, ± LoRA | 2,075 images, 39 classes | mIoU: SegFormer-b5 + strong aug 57.8; SegFormer-b2 54.7; DINOv2-B linear 52.5; DLV3+ 45.7; UNet++ 42.9. LoRA hurt SegFormer [V] | [13] |
| Contini et al., The point is the mask (coral, weak supervision) | 2025 | UAV 0.9-1.6 cm | SegFormer; Dice loss; 512 px | reef zones, 5 classes | pixel acc 86.07 %, mIoU 52.23 % [V] | [14] |
| Mohamed & Nadaoka, Superpixels + CNN + ViT for benthic habitats | 2026 | orthomosaics | SLIC + ConvNeXt-B / EfficientNet-B0 / DINOv2 features + SVM | benthic classes | metrics not read [?] | [15] |
| Chiabrando et al., Posidonia from Sentinel-2 | 2026 | Sentinel-2, 10 m | U-Net, DeepLabv3 (ResNet) | Sardinia, 6 classes | OA 0.830-0.882; kappa 0.786-0.843 [V abstract] | [16] |

Takeaways:

- Every architecture family has won some seagrass study. The winner tracks the year and the strength of the baselines more than the habitat.
- Only [1] compares encoders and decoders in a controlled ablation.
- The reported failure modes are sparse, low-cover seagrass and subtidal water [4][10].
- Domain shift between sites is large: leave-one-out transfer in [2] reached only 68 % of the oracle. Holding out whole sites matters more than the choice of model.

## 4. General segmentation and remote-sensing benchmarks

- **FLAIR-HUB (IGN, 2025): aerial-only, 3 bands, 20 cm.** mIoU and params [V] [24]:

  | Encoder | UNet | DeepLabV3 | UPerNet | Params (UPerNet) |
  |---|---|---|---|---|
  | ResNet50 | 51.5 | 61.7 | 58.9 | 30.0 M |
  | ConvNeXtV2 (base-size per table) | 64.2 | 63.25 | 63.8 | ~90 M |
  | Swin-B | 64.8 | 61.3 | 64.1 | 89.4 M |
  | ConvNeXtV2-T | – | – | 62.7 | 29.8 M |
  | MiT-B2 | – | – | 62.7 | 25.6 M |
  | Swin-T | – | – | 62.2 | 29.4 M |
  | HRNet32 | – | – | 61.2 | 33.4 M |
  | Swin-L | – | – | 64.8 | 199 M |

  The paper says Swin-UPerNet won FLAIR #1. Another source credits a U-Net ensemble (MiT + ResNeXt, 64.13) [?].
- **ADE20K (natural images), UPerNet, 512² crop [V] [18]:** ConvNeXt-T 46.0 single-scale / 46.7 multi-scale, 60 M params, 939 GFLOPs at 2048 × 512; Swin-T 45.8, 60 M, 945 G. Other heads: SegFormer-B2 46.5, 27.5 M, 62.4 G at 512²; DeepLabV3+-R101 44.1, 62.7 M, 255 G [20]; SegNeXt-T/S/B 41.1/44.3/48.5 at 4/14/28 M and 7/16/35 G [21].
- **UNetFormer (Wang et al. 2022, ISPRS JPRS):** lightweight ResNet18 encoder with a transformer decoder. LoveDA 52.4 mIoU, UAVid 67.8, Vaihingen 84.1 (Swin encoder), up to 322 FPS at 512² on one GPU [V abstract] [22]. Its GeoSeg code is GPL-3.0 [V], which does not suit a permissively licensed pack.
- **OpenEarthMap (Xia et al. 2023):** 5,000 images, 0.25-0.5 m, 8 classes, 44 countries [V]. Baselines covered U-Net (VGG/ResNet/EfficientNet), DeepLabV3, HRNet, K-Net, ConvNeXt, UNetFormer, SegFormer and UPerNet-Swin. Swin-B UPerNet scored 62.2-66.1 depending on the optimiser [?]. The full table was not read (PDF too large / HTTP 403) [23].
- **LoveDA leaderboard snapshot:** the top entries are ~54-56 mIoU (U-Net-MaxViT ensemble 56.16; a billion-scale RS foundation model 54.4); the original HRNet-w32 baseline was 49.79 [?] [34]. Frozen DINOv3 backbones: Sat ViT-L 54.4, Web 7B 56.2 [V] [26].
- **CNN vs transformer for VHR imagery:** at Tiny size the gap is ≤ 0.5 mIoU on both FLAIR-HUB [24] and ADE20K [18]. The large gains come from bigger backbones and better pretraining, not from the architecture family. Hybrid and Mamba models (PyramidMamba 70.8 on OpenEarthMap [?]) report small gains and carry deployment cost (section 7).
- **Mask2Former** leads some multi-class coastal and UAV comparisons (85.43 mIoU on Greek coastal imagery, 8.6 points above SegFormer) [?] [33]. For a binary mask, its query-based head adds cost and ONNX risk for no proven gain.

## 5. Foundation models

| Model | Input bands | Pretrain data / GSD | Params | Weight licence | Commercial OK? | ONNX? | Source |
|---|---|---|---|---|---|---|---|
| ConvNeXt-T v1 (supervised IN-1k/22k) | RGB | ImageNet (natural) | 28 M | MIT (repo); timm `fb_in22k_ft_in1k` Apache-2.0 [V] | Yes | Plain conv/LN; standard [?] | [18][27] |
| ConvNeXt-T V2 (FCMAE) | RGB | ImageNet | 28.6 M | CC BY-NC 4.0 (repo, timm card) [V] | **No** | Standard ops [?] | [19][27] |
| DINOv2 ViT-B | RGB | LVD-142M web | ~86 M [?] | Apache-2.0 [V] | Yes | ViT; commonly exported [?] | [27] |
| DINOv3 ConvNeXt-T | RGB | LVD-1689M web | 29 M (README) / 27.8 M (HF) [V] | DINOv3 licence, gated [V] | Royalty-free grant with attribution and use restrictions; legal check [V] | Conv; likely [?] | [26][27] |
| DINOv3 ViT-L Sat | RGB | SAT-493M, Maxar 0.6 m | 303 M [V] | DINOv3 licence, gated [V] | as above | ViT; too heavy for CPU tiles [?] | [26][27] |
| SAM ViT-B / SAM2.1 Hiera | RGB | SA-1B (natural) | 93.7 M (SAM-B) [V] | Apache-2.0 [V] | Yes | Encoder exportable [?] | [27] |
| SAM3 | RGB | Meta data | 860 M [V] | SAM licence, gated; royalty-free; military/ITAR exclusions [V] | Custom; legal check | Heavy [?] | [27][28] |
| segment-geospatial (samgeo) | RGB | wraps SAM models | – | MIT code [V]; weights follow the SAM version | Depends on SAM version | – | [28] |
| SatlasPretrain (Swin-v2) | RGB / S2 MS | Sentinel-2 10 m; NAIP aerial 0.5-2 m [V] | Swin-T/B | README: checkpoints ODC-BY; HF card: Apache-2.0 [V] (conflict) | Probably yes, with attribution | Swin [?] | [29] |
| Clay v1 | multi-sensor, incl. RGB | Landsat, Sentinel, NAIP etc. [?] | ~ViT-L [?] | Apache-2.0 for code and weights [V] | Yes | ViT [?] | [27] |
| Prithvi-EO-2.0 | 6-band HLS | HLS 30 m [?] | 300 M (from name) | Apache-2.0 [V] | Yes | ViT [?]; resolution mismatch | [27] |
| TerraMind 1.0 | S1, S2 and others [?] | Sentinel 10 m [?] | base | Apache-2.0 [V] | Yes | Resolution mismatch | [27] |
| Copernicus-FM | Sentinel family | 10 m+ [?] | – | HF: CC BY 4.0; repo: Apache-2.0 with CC BY-NC parts [V] | Check the NC parts | – | [27][28] |
| DOFA | any bands | multi-sensor [?] | – | CC BY 4.0 [V] | Yes, with attribution | – | [27] |
| RemoteCLIP | RGB | RS image-text pairs, mixed GSD [?] | RN50 / ViT-B / ViT-L | Repo Apache-2.0; HF weights: licence not found [V] | Unclear | ViT/RN [?] | [27][28] |
| Scale-MAE | RGB | fMoW, multi-scale [?] | ViT-L [?] | CC BY-NC 4.0 [V] | **No** | – | [28] |
| SatMAE | RGB / MS | fMoW [?] | ViT-L [?] | CC BY-NC 4.0 [V] | **No** | – | [28] |
| SSL4EO-S12 | S1/S2 | Sentinel 10 m | RN50 / ViT-S | Apache-2.0 (repo) [V] | Yes | Resolution mismatch | [28] |
| Galileo, AnySat | multi-sensor | mostly 10 m+ [?] | – | MIT (repos) [V] | Yes | – | [28] |

What the benchmarks say:

- **PANGAEA (2024/25) [V] [25]:** foundation models "do not consistently outperform supervised models". A UNet trained from scratch is top-2 most often (6 of 11 datasets). On VHR RGB-like tasks:
  - xView2: UNet 58.7; Scale-MAE 60.7, DOFA 59.6, GFM-Swin 59.2, RemoteCLIP 57.4, Prithvi 49.4.
  - Five-Billion-Pixels: UNet 60.5; RemoteCLIP 69.2, Scale-MAE 67.2, GFM-Swin 67.2.

  Models pretrained at high resolution help on VHR tasks; those pretrained at 10-30 m mostly lag. With 10 % of the labels, foundation models gain relative ground.
- **DINOv3 [V] [26]:** the frozen Sat ViT-L ties the previous best on LoveDA (54.4). On GEO-Bench RGB segmentation, the 7B model trained on web images (75.9 mean) beats the 7B satellite model (75.0). High-resolution RGB segmentation does not need satellite pretraining.
- **Coralscapes [V] [13]:** DINOv2 (linear or DPT head) did not beat fine-tuned SegFormer on benthic mIoU. LoRA helped DPT but hurt SegFormer.
- **What this means for us:** at ~1 cm GSD our imagery looks more like natural photos than like satellite scenes. ImageNet or web self-supervised backbones (ConvNeXt, DINOv3-ConvNeXt, Swin) match our pretraining domain. Sentinel/Landsat foundation models (Prithvi, TerraMind, SSL4EO, Copernicus-FM) do not, because their resolution is too coarse.

## 6. Small-data regime and losses

- **Data size:** our ~13,700 tiles are more than any seagrass study found used: Lian 3,840 patches [1], Pawar ~3,560 [2], Tahara 980 [6]. We are not in a few-shot regime. The larger risk is domain shift, since all tiles come from one country. Pawar's cross-site transfer reached only 68 % of the oracle [2], and Lian evaluated on a held-out region [1]. Validate on held-out sites or orthomosaics, not random tiles. [V]
- **Pretraining:** every strong seagrass result used ImageNet-pretrained encoders [1][2][3]. PANGAEA shows supervised models stay competitive with full labels, and foundation models help more at 10 % labels [25]. An empirical study of remote-sensing pretraining reports gains mainly for scene recognition and "task discrepancies" for segmentation [?] [35].
- **Freezing and parameter-efficient fine-tuning:** on Coralscapes, full fine-tuning of SegFormer beat LoRA. DINOv2 linear heads with a near-frozen backbone matched SegFormer on accuracy but not on mIoU [13]. With more than 10k tiles, full fine-tuning of a Tiny backbone on one GPU is the default. Keep freezing for small or noisy subsets.
- **Semi- and weak supervision:** Hobley 2021 found FCNNs trained on sparse labels match OBIA on drone seagrass [?] [7]. Contini 2025 trains UAV segmentation on coarse masks derived from underwater point labels, then self-distils [14]. Both are options if label coverage is partial.
- **Losses in seagrass and benthic work:**
  - CE + Lovász-Softmax, λ = 0.6, with a sensitivity study on λ [1].
  - BCE + IoU loss [9].
  - Dice, with boundary, CE and focal losses compared only visually [14].

  For binary, imbalanced masks the usual choice is BCE (or focal) plus a region loss (Dice or Lovász hinge). Azad et al. 2023 survey 25 segmentation losses: Dice, focal and Lovász are pixel- or region-level, boundary loss is boundary-level [V abstract] [32]. No seagrass study compares losses quantitatively.
- **Augmentation:** stronger augmentation was the largest single gain for SegFormer-b5 on Coralscapes (+2.8 mIoU) [13]. Lian used flips, small rotations and standardisation with training-set statistics [1]. Histogram matching handles colour shift between sensors [2].

## 7. Deployment and ONNX notes

- **Licence chain:** check three layers separately [27][30]:
  - model code;
  - library code: smp is MIT, but its MiT encoder file is NVIDIA non-commercial, MobileOne is Apple-licensed, and some encoders and the DeepLabV3 decoder are BSD-3;
  - weights: check the timm card of each checkpoint.
- **UPerNet PPM at 256 px:** smp's UPerNet `PSPModule` uses `nn.AdaptiveAvgPool2d` with sizes (1, 2, 3, 6), followed by bilinear upsampling [V] [30]. A 256 px input gives an 8 × 8 deepest map; bins 1 and 2 divide 8, bins 3 and 6 do not. smp's PSPNet hit exactly this export failure and, in 2026, replaced adaptive pooling with an area-interpolation path during ONNX export (PR #1270) [V] [30]. I could not confirm whether smp's UPerNet has the same guard. Mitigations [?]:
  - export at a fixed 256 × 256 and test with ONNX Runtime;
  - try the dynamo exporter;
  - replace the PPM bins with fixed AvgPool kernels for the known shape.

  In every case, compare ONNX and PyTorch outputs numerically.
- **Dynamic shapes:** a fixed 256 × 256 input (our serving tile) avoids most issues. Dynamic height and width bring the adaptive-pooling problem back. The legacy exporter takes `dynamic_axes`, the dynamo exporter `dynamic_shapes` [?].
- **Mask2Former and Deformable-DETR heads:** MSDeformAttn is a custom CUDA op. Community reports describe broken or diverging ONNX exports, worked around with TensorRT plugins or pure-PyTorch fallbacks [?] [31]. Avoid for CPU ONNX serving.
- **InternImage (DCNv3) and Mamba/VMamba:** both rely on custom CUDA kernels: deformable convolution v3, and a selective scan with no CPU fallback in the reference implementation [?]. Avoid them for a torch-free ONNX CPU core.
- **Params and FLOPs at 256²:** derived by linear scaling from published figures, not measured.

  | Model | FLOPs at 256² | Note |
  |---|---|---|
  | UPerNet + ConvNeXt-T (mmseg, 60 M params) | ≈ 59 G | Lian's slimmer variant has 34.1 M params [1]; FLOPs not reported |
  | DeepLabV3+-R101 | ≈ 64 G | |
  | SegFormer-B2 | ≈ 15.6 G | excluded by licence |
  | SegNeXt-B | ≈ 8.8 G | |
  | ConvNeXt-T backbone alone | ≈ 5.9 G | |

  If CPU latency is too high, shrink the UPerNet channel width (512 → 256) or switch to a U-Net/FPN head before changing the backbone.
- **CPU latency:** no published CPU latency at 256 px was found for any candidate. It has to be measured with ONNX Runtime on the target pod.

## 8. Gaps

- Lian 2026 [1] has no RGB-only result. Part of its UPerNet gain may come from the multispectral channels and indices. RGB pretrained weights cannot load directly into a 9-channel stem, and I did not find how they initialised it.
- No study compares ConvNeXt v1 with v2 on seagrass, or compares ConvNeXt with Swin, DINOv3 or SAM2 encoders on the same seagrass data.
- No benchmark exists for tropical subtidal seagrass at ~1 cm. The tropical drone work found either reports no IoU/F1 [10] or uses no deep learning (Zanzibar [12]).
- No published CPU/ONNX latency exists for UPerNet-ConvNeXt-T at 256 px. The FLOPs here are scaled estimates.
- Not read in full: Jeon 2021 numbers [5], Chen & Sasaki baseline numbers [4], the OpenEarthMap table [23], Hobley 2021 [7], Weidmann 2019 [8], Mohamed & Nadaoka 2026 [15].
- Licence questions for a lawyer:
  - whether the DINOv3 and SAM3 licences permit HOT's use and its redistribution of fine-tuned weights;
  - the SatlasPretrain conflict between ODC-BY and Apache-2.0;
  - the licence of the RemoteCLIP weights (none stated on HF).

## 9. Sources

1. Lian J., Mo Z., Liu Z., Peng B., Chang M., Wang X., Wang W. (2026). Intertidal Seagrass Mapping Using UAV Visible and Multispectral Imagery: A Comparative Semantic Segmentation Study with Explainability Analysis. *Remote Sensing* 18:2057. https://doi.org/10.3390/rs18122057 (PDF: https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-02057/article_deploy/remotesensing-18-02057.pdf) [V full text]
2. Pawar et al. (2025). Eye in the Sky for Sub-Tidal Seagrass Mapping: Leveraging Unsupervised Domain Adaptation with SegFormer for Multi-Source and Multi-Resolution Aerial Imagery. *Remote Sensing* 17(14):2518. https://www.mdpi.com/2072-4292/17/14/2518 (PDF: https://backend.orbit.dtu.dk/ws/files/410545772/remotesensing-17-02518.pdf) [V full text]
3. Tallam K. et al. (2023). Application of Deep Learning for Classification of Intertidal Eelgrass from Drone-Acquired Imagery. *Remote Sensing* 15(9):2321. https://doi.org/10.3390/rs15092321 (https://repository.library.noaa.gov/view/noaa/65522) [V]
4. Chen J., Sasaki J. (2021). Mapping of Subtidal and Intertidal Seagrass Meadows via Application of the Feature Pyramid Network to UAV Orthophotos. *Remote Sensing* 13(23):4880. https://doi.org/10.3390/rs13234880 [V abstract]
5. Jeon E.-I. et al. (2021). Semantic segmentation of seagrass habitat from drone imagery based on deep learning: a comparative study. *Ecological Informatics*. https://doi.org/10.1016/j.ecoinf.2021.101430 [V abstract via listing]
6. Tahara S. et al. (2022). Species level mapping of a seagrass bed using an unmanned aerial vehicle and deep learning technique. *PeerJ* 10:e14017. https://pmc.ncbi.nlm.nih.gov/articles/PMC9583862/ [V]
7. Hobley B. et al. (2021). Semi-Supervised Segmentation for Coastal Monitoring Seagrass Using RPA Imagery. *Remote Sensing* 13(9):1741. https://ueaeprints.uea.ac.uk/id/eprint/80073 [?]
8. Weidmann F. et al. (2019). A Closer Look at Seagrass Meadows: Semantic Segmentation for Visual Coverage Estimation. OCEANS 2019. https://doi.org/10.1109/OCEANSE.2019.8867064 [?]
9. Huang S. (2025). Advancing Seagrass semantic segmentation with SAM2 models. *Journal of High School Science*. https://jhss.scholasticahq.com/article/138532.pdf [V abstract]
10. Langlois L., Collier C., Navarro A., Carter A. (2026). Drone-Based Monitoring of Tropical Seagrass: Reproducible Methods and Practical Applications. AMSA 2026 conference abstract. https://portfolio.jcu.edu.au/publications/JCU325143 [V abstract]
11. Oiry S. et al. (2024). Discriminating Seagrasses from Green Macroalgae in European Intertidal Areas Using High-Resolution Multispectral Drone Imagery. *Remote Sensing* 16(23):4383. https://doi.org/10.3390/rs16234383 [V abstract]
12. Hamad I.Y. et al. (2022). Drone-Based Characterization of Seagrass Habitats in the Tropical Waters of Zanzibar. *Remote Sensing* 14(3):680. https://doi.org/10.3390/rs14030680 [V abstract]
13. Sauder J., Domazetoski V., Banc-Prandi G., Perna G., Meibom A., Tuia D. (2025). The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs. arXiv:2503.20000. https://arxiv.org/html/2503.20000v2 [V]
14. Contini M. et al. (2025). The point is the mask: scaling coral reef segmentation with weak supervision. arXiv:2508.18958 (ICCVW 2025). https://arxiv.org/abs/2508.18958 [V]
15. Mohamed H., Nadaoka K. (2026). Integration of Superpixel Segmentation, Convolutional Neural Networks and Vision Transformers for Automatic Benthic Habitats Classification. *Remote Sensing* 18(11):1711. https://doi.org/10.3390/rs18111711 [?]
16. Chiabrando F. et al. (2026). Deep Learning-Based Underwater Mapping of Posidonia Oceanica from Satellite Data for Coastal Habitat Monitoring. ISPRS Archives XLIX-B4-2026. https://isprs-archives.copernicus.org/articles/XLIX-B4-2026/111/2026/ [V abstract]
17. Gonzalez-Perez A. et al. (2022). Deep and Machine Learning Image Classification of Coastal Wetlands Using Unpiloted Aircraft System Multispectral Images and Lidar Datasets. *Remote Sensing* 14(16):3937. https://doi.org/10.3390/rs14163937 [V abstract]
18. Liu Z. et al. (Meta AI, 2022). A ConvNet for the 2020s (ConvNeXt). https://arxiv.org/html/2201.03545 ; code (MIT) and ADE20K table: https://github.com/facebookresearch/ConvNeXt/tree/main/semantic_segmentation [V]
19. Woo S. et al. (Meta AI, 2023). ConvNeXt V2. https://github.com/facebookresearch/ConvNeXt-V2 ("MIT license except ImageNet pre-trained and fine-tuned models … CC-BY-NC") [V]
20. Xie E. et al. (NVIDIA, 2021). SegFormer. https://arxiv.org/html/2105.15203 ; licence: https://github.com/NVlabs/SegFormer/blob/master/LICENSE ("only may be used … non-commercially") [V]
21. Guo M.-H. et al. (2022). SegNeXt. https://github.com/Visual-Attention-Network/SegNeXt (Apache-2.0; "For commercial use, please contact the authors") [V]
22. Wang L. et al. (2022). UNetFormer. *ISPRS JPRS* 190:196-214. https://arxiv.org/abs/2109.08937 ; code GeoSeg (GPL-3.0): https://github.com/WangLibo1995/GeoSeg [V]
23. Xia J. et al. (2023). OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping. WACV 2023. https://arxiv.org/abs/2210.10732 [V abstract; table ?]
24. Garioud A. et al. (IGN, 2025). FLAIR-HUB: Large-scale Multimodal Dataset for Land Cover and Crop Mapping. arXiv:2506.07080. https://arxiv.org/html/2506.07080 (Table IX) [V]
25. Marsocci V. et al. (2024/2025). PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models. arXiv:2412.04204. https://arxiv.org/html/2412.04204v2 [V]
26. Siméoni O. et al. (Meta AI, 2025). DINOv3. arXiv:2508.10104. https://arxiv.org/html/2508.10104 ; repo: https://github.com/facebookresearch/dinov3 ; licence: https://ai.meta.com/resources/models-and-libraries/dinov3-license/ [V]
27. Hugging Face model API and cards (licence, gating, params), checked 2026-11, all under https://huggingface.co/ [V]: `timm/convnext_tiny.fb_in22k_ft_in1k`, `timm/convnextv2_tiny.fcmae_ft_in22k_in1k`, `facebook/dinov2-base`, `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`, `facebook/dinov3-vitl16-pretrain-sat493m`, `facebook/sam-vit-base`, `facebook/sam2.1-hiera-large`, `facebook/sam3`, `nvidia/mit-b2`, `openmmlab/upernet-convnext-tiny` (MIT, 60.3 M), `made-with-clay/Clay`, `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`, `ibm-esa-geospatial/TerraMind-1.0-base`, `wangyi111/Copernicus-FM`, `earthflow/DOFA`, `chendelong/RemoteCLIP`, `allenai/satlas-pretrain`.
28. GitHub licence files and READMEs [V]: SatMAE (CC BY-NC 4.0) https://github.com/sustainlab-group/SatMAE ; Scale-MAE (CC BY-NC 4.0) https://github.com/bair-climate-initiative/scale-mae ; Copernicus-FM https://github.com/zhu-xlab/Copernicus-FM ; SAM3 licence https://github.com/facebookresearch/sam3 ; SAM2 (Apache-2.0) https://github.com/facebookresearch/sam2 ; samgeo (MIT) https://github.com/opengeos/segment-geospatial ; SSL4EO-S12 https://github.com/zhu-xlab/SSL4EO-S12 ; RemoteCLIP https://github.com/ChenDelong1999/RemoteCLIP ; Galileo https://github.com/nasaharvest/galileo ; AnySat https://github.com/gastruc/AnySat ; Clay https://github.com/Clay-foundation/model
29. Allen AI. SatlasPretrain models README (checkpoints ODC-BY; 0.5-2 m aerial and Sentinel-2). https://github.com/allenai/satlaspretrain_models [V]
30. segmentation_models.pytorch [V]: licences https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/licenses/LICENSES.md ; UPerNet decoder https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/upernet/decoder.py ; PSPNet ONNX fix, PR #1270 https://github.com/qubvel-org/segmentation_models.pytorch/pull/1270
31. Cheng B. et al. (2022). Mask2Former. https://github.com/facebookresearch/Mask2Former (MIT; Deformable-DETR parts Apache-2.0) [V]; notes on converting MSDeformAttn to ONNX/TensorRT, e.g. https://github.com/lantudou/mask2former_trt [?]
32. Azad R. et al. (2023). Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook. arXiv:2312.05391. https://arxiv.org/abs/2312.05391 [V abstract]
33. Integrating AI for In-Depth Segmentation of Coastal Environments in Remote Sensing Imagery (2026). *Remote Sensing* 18(2):325. https://www.mdpi.com/2072-4292/18/2/325 [?]
34. LoveDA semantic segmentation leaderboard snapshot (aggregator). https://vercel.hyper.ai/en/sota/tasks/semantic-segmentation/benchmark/semantic-segmentation-on-loveda [?]
35. Wang D. et al. (2022). An Empirical Study of Remote Sensing Pretraining. *IEEE TGRS*. https://arxiv.org/abs/2204.02825 [?]
