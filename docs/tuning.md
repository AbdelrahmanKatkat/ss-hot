# Tuning and fine-tuning: state of the art

2026-11-21. Literature review of how to train and fine-tune our segmentation model: UPerNet decoder (PPM bins 1, 2, 4, 8; 256 channels; BatchNorm) on a timm ConvNeXt-Tiny v1 encoder (`convnext_tiny.fb_in22k_ft_in1k`, LayerNorm), binary seagrass, sigmoid, BCE + Dice. There are two jobs:

- **(a) Base training:** from ImageNet weights on ~13,700 zoom-24 tiles of 256 px (~0.92 cm/px), four drone flights, Timor-Leste.
- **(b) User fine-tuning in fAIr:** from the base model, on a few hundred chips of a new AOI (possibly another site, sensor or zoom), one GPU, served as CPU ONNX.

Nothing was installed, trained or measured for this review.

**Markers:**
- **[V]**: read in the source (abstract, paper HTML or tables, official config or code, model card or docs).
- **[?]**: seen only in a search snippet, a secondary page or from memory.
- Several table numbers were pulled out of paper HTML by a page-reading tool. Spot-check any number before it drives a decision.
- "Domain" says where the evidence comes from. Most of it is **not** aerial or seagrass.

Bracketed numbers like [1] point to section 10.

**Current knobs** (from `src/ss_hot/params.py`, `train.py`, `model.py` on 2026-11-21):
- AdamW, lr 1e-4, weight decay 0.05 (none on norms and biases), cosine with 400 warmup steps stepped per batch, gradient clip 1.0, AMP, batch 8, 40 epochs.
- Early stopping on `val_iou`, patience 8. `epoch_fraction`.
- drop_path 0.1. Decoder dropout 0.1.
- `layer_decay` 0.9: encoder stage *i* gets ×0.9^(4−*i*), the stem ×0.9^5. `encoder_lr_factor` 1.0.
- `freeze_encoder`, `unfreeze_from` (stage0..stage3).
- Note: `TrainParams.freeze_encoder` defaults to **`True`** in `params.py` (changed 2026-11-21), so the base-training run must pass `freeze_encoder=False` explicitly.

## 1. Summary

### Key findings

1. **Our base recipe already matches the official ConvNeXt-T + UPerNet ADE20K recipe on the main numbers**: AdamW 1e-4, wd 0.05, stage-wise layer decay 0.9. Three differences matter [2][3][4]:
   - The official drop_path is **0.4**; ours is 0.1.
   - The official decay also puts the whole encoder one notch further below the head. Its multipliers are head 1.0, stage3 0.729 … stage0 0.531, downsample layers 0.478. Ours are stage3 0.9 … stage0 0.656. Setting `encoder_lr_factor` = 0.81 reproduces the official values.
   - Official batch is 16 with SyncBN; ours is 8 with plain BN. The official recipe has no gradient clipping.

   [V]
2. **Base training on 13.7k tiles should be full fine-tuning.** Pretraining stops being decisive somewhere between 3.5k and 10k images on COCO [13]. Fine-tuning beat fixed features in 179/192 transfer cases [12]. A frozen encoder or frozen-then-decoder start is for small data, not for job (a). A short decoder-only warm-up (LP-FT) protects features from a random head: +10% OOD vs plain fine-tuning [6], but only shown in classification. [V]
3. **For user fine-tuning, every segmentation study we found shows the same trade-off.** Freezing more keeps the source; tuning more fits the target.
   - Continual segmentation, VOC 19+1, mIoU old/new: frozen encoder 71.9/43.3 vs full fine-tuning 67.3/20.1 [28].
   - Cityscapes→ACDC Night [32]: freezing the encoder up to layer4.1 cut forgetting from 26.1 to 18.9 mIoU, but target mIoU fell from 43.6 to 32.5.
   - Our own tree detector [83] shows the same pattern.

   Freezing is the safe default, not a free one. [V]
4. **Cheap regularisers capture most of the "robust fine-tuning" gain on segmentation.** On PASCAL-Context with Swin-T + SegFormer head, clean/OOD mIoU [25]:

   | Method | Clean | OOD |
   |---|---|---|
   | Vanilla fine-tuning | 66.03 | 44.00 |
   | Layer-wise lr decay | 72.09 | 54.47 |
   | L2-SP (pull toward the pretrained weights) | 73.47 | 56.70 |
   | FTP (best) | 73.79 | 58.45 |

   L2-SP also cut source-task forgetting about 3–5× in classification [19]. In domain-incremental segmentation, **replaying source data** is the strongest anti-forgetting tool and EWC is weak [31][32]. [V]
5. **Domain shift in segmentation lives in the early layers and the normalisation statistics.** Forgetting comes mainly from drift of the first convolutions. Re-estimating only the first BN layer's statistics recovers much of it [32]. Input-level shifts are best fixed by tuning early layers [7]. Our `unfreeze_from` can only unfreeze late stages, which suits a change in what seagrass looks like more than a change of sensor. [V]
6. **PEFT on ConvNeXt + UPerNet trails full fine-tuning on large data, and nobody has tested it in our regime.**
   - ADE20K, ConvNeXt-B: LoRA 50.5 vs full 51.4 mIoU [38]; Conv-Adapter 46.9 vs 48.7 [39].
   - In ViT and remote-sensing work PEFT forgets less. LoRA held 87.6 vs 82.1 IoU on a geographic hold-out [51], and PEFT beat full fine-tuning on 100-shot OOD [50].
   - On a coral dataset LoRA *hurt* SegFormer by 2.3–2.8 mIoU [52].
   - Only mergeable methods fit our single ONNX graph: LoRA on the MLP `Linear` layers merged before export, bias/norm/LayerScale tuning, SSF. Adapters add inference branches (+45% latency for Conv-Adapter on ConvNeXt-B [39]). [V]
7. **BatchNorm at batch 8 is mildly worse than GroupNorm, and BN statistics are a lever under shift.**
   - ImageNet top-1 error: BN 24.8 vs GN 24.0 at batch 8, BN 34.7 at batch 2 [54].
   - Fine-tuning with live BN at batch 2 cost about 6 AP in detection [54].
   - Re-estimating BN statistics on the target domain helps under shift: ImageNet-C mCE 76.7→62.2 [77]; GTA→Cityscapes 28.8→31.4 [76].
   - Only our decoder has BN, since ConvNeXt v1 uses LayerNorm. Re-estimating it is parameter-free and keeps the ONNX graph.

   [V]
8. **BCE + Dice is a defensible loss; the threshold must be re-chosen after every fine-tune.**
   - Compound losses are the most robust across tasks [62]. Overlap losses beat CE on IoU/Dice [59][63].
   - Dice makes the sigmoid overconfident [66]. Under calibration the F1-optimal threshold is F1*/2, below 0.5 [67].
   - No paper tests re-tuning the threshold after fine-tuning. That part is our inference. [V]

### Defaults in one line

- **(a) Base:** full fine-tuning; current optimiser; try drop_path 0.2–0.4, `encoder_lr_factor` 0.81 and a 1-epoch decoder warm-up.
- **(b) User:** frozen encoder (decoder only) as the default, warmup cut to tens of steps, threshold re-swept. `unfreeze_from=stage3` with `encoder_lr_factor` ≈ 0.1–0.4 stays the opt-in for more target fit.
- **Next steps, once tested:** L2-SP toward the base weights, and WiSE interpolation or decoder-BN re-estimation, all of which keep the ONNX graph.

## 2. Q1: Which parameters to train

### 2.1 Evidence

| Strategy | Evidence (domain) | Numbers | Source |
|---|---|---|---|
| Full fine-tuning vs fixed features | Classification, 16 nets × 12 datasets | Fine-tuning > fixed features in 179/192 cases; > scratch in 189/192 [V] | [12] |
| Pretraining vs scratch by data size | COCO detection / segmentation | ~10k images: pretrained 26.0 AP at 60k iterations ≈ scratch 25.9 at 220k. 3.5k images: 16.0 vs 9.3. 1k: 9.9 vs 3.5. "Pre-training does not automatically reduce overfitting" [V] | [13] |
| Pretraining with strong augmentation | COCO / PASCAL | Pretraining helps at 1/5 of COCO, hurts at full COCO with strong augmentation; helps on PASCAL segmentation [V] | [14] |
| Where feature reuse happens | Medical classification | "Serious feature reuse is restricted to the lowest couple of layers"; reusing up to Block2 ≈ full transfer [V] | [16] |
| Fine-tune vs linear, 1,000 examples | VTAB classification | ResNet-50 fine-tune 65.6 vs linear 57.3 [V] | [17] |
| Domain of source data | >2,000 transfer experiments incl. segmentation and aerial | Image domain is the most important factor; in-domain source usually beats ILSVRC [V, tables not read] | [15] |
| **LP-FT** (head first, then all) | 10 classification shift datasets | Full fine-tuning +2% in-distribution, −7% OOD vs linear probe. LP-FT +1% in-distribution, +10% OOD vs full fine-tuning. Cause: a random head distorts features early [V] | [6] |
| **Surgical fine-tuning** | Classification shifts | Input-level shift → first block best. CIFAR-C: all 79.9 vs surgical 82.8; ImageNet-C 50.7 vs 51.6. Output-level shift → last layer: CIFAR-Flip 85.9 vs 93.8. Small target sets (CIFAR-C 1,000 images) [V] | [7] |
| **Gradual unfreezing + discriminative lr** (ULMFiT) | NLP classification | Lower layer lr = upper / 2.6. IMDb error: full 6.87, full+discriminative 5.57, last-layer only 6.49, all tricks 5.00. Last-only underfits on TREC-6 (16.09) [V] | [8] |
| Freeze all but the box head (TFA) | Few-shot detection, VOC | Novel AP50 at 1/3/10 shots: full fine-tuning 15.2/29.0/45.5 vs frozen 39.8/44.7/56.0. Base AP50 ≈ 66 (full) vs ≈ 79 (frozen) [V] | [18] |
| Frozen encoder in segmentation | Continual segmentation, VOC 19+1 | mIoU old/new/all: fine-tuning 67.3/20.1/65.1; frozen encoder 71.9/43.3/70.5; feature distillation 72.6/50.6/71.6 [V] | [28] |
| Partial freezing under domain shift | DeepLabV3+-R50, Cityscapes→ACDC Night | Freeze up to layer4.1: forgetting 26.1→18.9, target mIoU 43.6→32.5. Freeze only up to layer1.0: forgetting 28.4 (worse) [V] | [32] |
| In-house: tree detector, partial unfreeze | DeepForest/RetinaNet, NEON→Freetown | `unfreeze_from=layer4` matched full unfreezing on Freetown val F1 (0.482 vs 0.480) and kept 82% vs 66% of NEON F1. All weights trainable kept 32–66% of source F1; frozen encoder kept 96% (brief) [V, local file] | [83] |

### 2.2 Layer-wise lr decay values

- **ConvNeXt official ADE20K:** `LearningRateDecayOptimizerConstructor`, `decay_type='stage_wise'`, `decay_rate` 0.9, `num_layers` 6 [V] [2][4].
  - The constructor pads `num_layers` by 2 and uses `scale = 0.9 ** (num_layers − layer_id − 1)`.
  - Downsample layers and stem get id 0, stage *i* gets *i*+1, the decode head gets the last id [V] [4].
  - Resulting multipliers (our arithmetic from the code): head 1.0, stage3 0.729, stage2 0.656, stage1 0.590, stage0 0.531, downsample layers 0.478.
  - The paper swept decay {0.8, 0.9} and lr {8e-5, 1e-4} [V] [1].
- **Ours:** stage3 0.9, stage2 0.81, stage1 0.729, stage0 0.656, stem 0.590, head 1.0.
  - Same ratio between stages, but a 0.9 step between head and stage3 where the official step is 0.729.
  - `encoder_lr_factor` = 0.81 reproduces the official multipliers, except that timm keeps each downsample inside its stage.
- **Other transformer and NLP values:**
  - ConvNeXt ImageNet-22k→1k fine-tuning: 0.8 [V] [1].
  - BEiT: 0.65 (Base) on ImageNet and ADE20K [V] [10].
  - ELECTRA: 0.8 (Base), 0.9 (Large) [V] [9].
  - ULMFiT: 1/2.6 ≈ 0.385 per layer [V] [8].
  - The DeepForest pack used 0.385 as its `encoder_lr_factor` [83].
- **Measured effect on segmentation:** layer-wise lr decay +6.06 mIoU clean (66.03→72.09) and +10.5 OOD on PASCAL-Context, Swin-T [V] [25].

### 2.3 What it means for us

- **(a) Base:**
  - Keep full fine-tuning with layer decay.
  - LP-FT applies because the UPerNet head starts random [6]. The cheap version is 1–2 epochs with `freeze_encoder=True`, then a full run from that checkpoint. Classification evidence only, so test it.
- **(b) User:**
  - Decoder-only is the safest start: the base head is already trained, so the user run starts like the second phase of LP-FT.
  - If the target is under-fitted, unfreeze stage3 at a low encoder lr. Evidence: TFA, ILT and our tree detector.
  - For a sensor or colour shift, the literature points to early layers and normalisation statistics instead [7][16][32]. `unfreeze_from` cannot express "early stages only". Decoder-BN re-estimation (section 3) is the cheap alternative.

## 3. Q2: Forgetting and robustness

| Remedy | Measured effect (domain) | Source |
|---|---|---|
| Frozen encoder | Best source retention, costs target fit (section 2) [V] | [18][28][32][83] |
| **L2-SP** (penalty ‖w − w₀‖² instead of weight decay toward 0) | Classification: MIT Indoors 79.6→84.2, Stanford Dogs 81.4→85.1. Source-task accuracy drop L2 vs L2-SP: −24.1 vs −5.3 (Indoors), −14.1 vs −4.7 (Dogs). **Segmentation:** Cityscapes ResNet-101 68.1→68.7, DeepLab 72.0→73.2; PASCAL-Context Swin-T 66.03→73.47 clean, 44.00→56.70 OOD [V] | [19][25] |
| DELTA (feature-map regularisation toward the source) | Claims to beat L2 and L2-SP; tables not read [V abstract] | [20] |
| EWC | Domain-incremental segmentation: Night forgetting 18.0 vs 26.1 baseline, Rain 21.2 vs 33.2; weaker than replay [V] | [21][32] |
| **Replay of source data** | Cityscapes→ACDC: forgetting 5.6 (Night), 2.9 (Rain) vs 26.1 / 33.2 for plain fine-tuning. "Replay is most effective in the domain-incremental setting" [V] | [31][32] |
| Augmentation as anti-forgetting | Cityscapes→ACDC: ImageNet init + AutoAlbument roughly halves forgetting (Night 19.1→15.1, Rain 22.5→10.7) [V] | [32] |
| Output / feature distillation (LwF, ILT, MiB, PLOP) | VOC 15-5 disjoint mIoU all-classes: fine-tuning 9.2, ILT 57.3, MiB 64.7, joint 77.4. Class-incremental, less like our "same class, new site" [V] | [28][29][30][33] |
| **WiSE-FT** (interpolate base and fine-tuned weights) | CLIP: +4 to +6 pp under shift, +0.8 to +3.3 pp over fine-tuning on 7 transfer sets. Zero inference cost; classification only [V] | [22] |
| Model soups | Average of a fine-tuning sweep: ViT-G 90.94% top-1. Classification [V] | [23] |
| TPGM / FTP (learned projection toward pretrained weights) | FTP best on PASCAL-Context, 73.79 clean / 58.45 OOD; TPGM 72.56 / 56.96 [V] | [24][25] |
| EMA of weights | timm `ModelEmaV2/V3` average the whole state dict, BN buffers included. Default decay 0.9999, "set relative to updates per epoch" [V]. No forgetting numbers found | [26][27] |
| Early stopping | OOD robustness appears mid-fine-tuning and disappears at convergence [V abstract]. No segmentation study of early stopping on a *source* hold-out found | [34] |

What it means for us:
- Early stopping on the user's target validation set does not measure forgetting.
- AdamW's weight decay pulls toward zero, not toward the base. In user fine-tuning, L2-SP (or a lower lr with layer decay) is the measured alternative.
- WiSE interpolation is free at serving time: a 0–1 blend of two state dicts with the same architecture, then one ONNX export.
- Replay needs the base tiles inside the user's pipeline step. Whether fAIr can provide them is a platform question (section 9).

## 4. Q3: Parameter-efficient fine-tuning

| Method | Evidence closest to us | Mergeable into one ONNX graph? | Source |
|---|---|---|---|
| **LoRA** on ConvNeXt + UPerNet | ADE20K: ConvNeXt-B full 51.4 vs LoRA 50.5 mIoU (5.9 M, ~6.7%); ConvNeXt-L 52.4 vs 50.8 [V] | Yes, after `merge_and_unload()` [V] | [35][36][38] |
| LoRA, remote sensing (ViT) | Prithvi-2.0 300M, 5 datasets: full 68.14 vs LoRA r=16 (1.8%) 68.02; frozen 51.55. Geographic hold-out (Sen1Floods11): full 90.13→82.07, LoRA 90.04→87.57. Memory 15.1 vs 11.0 GB [V] | Yes, if merged | [51] |
| LoRA, coral (SegFormer) | MiT-b2 full 54.68 vs LoRA 51.91 mIoU; MiT-b5 55.03 vs 52.78. LoRA *helped* DINOv2+DPT (Base 44.20→47.23) [V] | Yes, if merged | [52] |
| LoRA forgets less | LLM: source average after code continued pretraining, full 0.545 vs LoRA r=16 0.635; target lower (HumanEval 0.263 vs 0.224 at r=256). Higher rank learns more and forgets more [V] | — | [49] |
| PEFT in low-shot / OOD | CLIP ViT-B/16, 100-shot ImageNet, shift accuracy: full 42.5, BitFit 55.4, LayerNorm 55.9, LoRA 55.9, Adapter 56.9. WiSE lifts full fine-tuning above PEFT [V] | — | [50] |
| Conv-Adapter | ADE20K UperNet: ResNet-50 full 42.1 vs 43.0; **ConvNeXt-B full 48.7 vs 46.9**. COCO ConvNeXt-B 45.2 vs 41.9 AP. Latency 1.17→1.70 ms/img (+45%) [V] | No | [39] |
| Mona / LoRand (adapters) | Swin-L ADE20K: full 51.18, Mona 51.36, LoRand 50.93, LoRA 50.34, BitFit 48.37, NormTuning 47.89, frozen 46.84 [V] | No (extra branches) | [40][41] |
| SSF (scale and shift features) | ConvNeXt-B CIFAR-100: full 94.14, SSF 93.45, bias-only 92.80. **Swin-T dense, frozen backbone:** ADE20K full 44.5 vs SSF 38.9; COCO 43.7 vs 34.9. More robust OOD (ViT-B IN-R 51.29→56.77) [V] | Yes (folds into the preceding layer) [V] | [42] |
| BitFit / bias-only, TinyTL | BERT competitive [V]; TinyTL up to 6.5× less memory, +34.1% over last-layer tuning (CNN classification) [V] | Bias-only yes; TinyTL lite-residual no | [43][44] |
| Norm-only tuning | BN-only on random nets: 82% CIFAR-10 [V]. LayerNorm-only beat LoRA on a multimodal LLM [V] | Yes | [45][46] |
| Side-tuning | No catastrophic forgetting by construction; matches or beats fine-tuning in several settings [V abstract] | No (a second network at inference) | [47] |
| ConvLoRA + AdaBN | Brain MRI U-Net, 0.24% params; surface Dice GE1.5 0.891 vs baseline 0.759; ConvLoRA without AdaBN sometimes below the baseline, so AdaBN does much of the work [V] | Yes (LoRA), BN buffers only (AdaBN) | [48] |
| UPetu (remote sensing, ConvNeXt-B) | ~0.73% params; beats full fine-tuning on scene classification [?] | No (adapters) | [53] |

ConvNeXt specifics:
- In timm ConvNeXt v1 (`conv_mlp=False`), the block MLP is `mlp.fc1`/`mlp.fc2` as `nn.Linear` on channels-last tensors. Each block also has a LayerNorm and a per-channel LayerScale `gamma` [V] [37].
- HF PEFT LoRA supports `nn.Linear` and `nn.Conv2d`, with `modules_to_save` for a decoder [V] [36].
- LoRA on the grouped/depthwise 7×7 conv is unconfirmed [?]. Target the MLP Linears.

ONNX/CPU:
- Merged LoRA, SSF, and bias/norm/LayerScale tuning change weights only: same graph, size and latency.
- Unmerged LoRA, adapters, Mona, Conv-Adapter and side networks add ops.
- Merging gives up the "small per-site delta" storage benefit. That is irrelevant to fAIr, which ships one ONNX per model anyway.

## 5. Q4: Hyperparameters and BatchNorm

### 5.1 Official ConvNeXt-T + UPerNet ADE20K recipe [V] [2][3][4][5]

| Knob | Official | Ours | Note |
|---|---|---|---|
| Optimiser | AdamW β (0.9, 0.999), lr 1e-4, wd 0.05, no wd on 1-D params and biases | same | matches |
| Layer decay | stage-wise 0.9, `num_layers` 6 (head 1.0, stage3 0.729 … stem 0.478) | stage-wise 0.9 (head 1.0, stage3 0.9 … stem 0.590) | `encoder_lr_factor` 0.81 ≈ official |
| drop_path | **0.4** (Tiny; base config 0.2) | 0.1 | ADE20K is 150 classes and 160k iterations; no small-data source found |
| Schedule | linear warmup 1,500 iterations (ratio 1e-6), poly power 1.0 to 0 | cosine, 400 warmup steps | warmup ≈ 0.9% of training officially; ours ≈ 0.6% of 68.5k steps (our arithmetic: 13,700/8 ≈ 1,712 steps × 40) |
| Length | 160k iterations × 16 = 2.56 M samples (≈ 127 passes over 20,210 images) | ≤ 40 × 13.7k = 548k samples, 256² | ours ≈ ¼ the samples at ¼ the pixels each |
| Batch, norm | 2 per GPU × 8, SyncBN | 8 on one GPU, BN | |
| Crop | 512 | 256 | |
| Head | PPM (1, 2, 3, 6), 512 channels, dropout 0.1; auxiliary FCN head on stage 2, weight 0.4 | PPM (1, 2, 4, 8), 256 channels, dropout 0.1; no auxiliary head | |
| Gradient clip | none | 1.0 | |
| Precision | fp16 | bf16/fp16 AMP | |
| Result | 46.0 single-scale / 46.7 multi-scale mIoU [1]; mmseg reproduction 46.11 / 46.62 [5] | — | |

- Remote-sensing UPerNet recipe for comparison: AdamW 6e-5, wd 0.01, poly, warmup 1,500, 80k iterations, batch 8, 512 crops (Potsdam/iSAID) [V] [69].
- Seagrass: Lian 2026 used AdamW 2e-4, 200 epochs, batch 24 on 3,840 patches of 256 px [V] [82].
- **Small datasets and 256 px:** no source tunes drop_path or layer decay for small data or small crops. The linear lr-batch scaling rule is for SGD [V] [57]. For Adam a square-root rule is proposed [V] [58], with its exact form [?].
- **User fine-tuning step counts** (our arithmetic): 300 chips / batch 8 ≈ 38 steps per epoch. The current `warmup_steps=400` is ≈ 10.5 epochs, so a 20-epoch run would spend half its time warming up. Scale warmup to about one epoch or a few per cent of steps. The official ratio is <1% [2].

### 5.2 BatchNorm in the decoder

- **BN vs GN by batch** (ResNet-50 ImageNet top-1 error, per-GPU batch 32/16/8/4/2): BN 23.6/23.7/24.8/27.3/34.7, GN 24.1/24.2/24.0/24.2/24.1 [V] [54]. Classification; a dense head normalises over many pixels per image, and no segmentation study of that was found.
- **Fine-tuning with live BN statistics at batch 2 cost about 6 AP** in detection. Detection therefore freezes BN (statistics and affine); GN beat frozen BN by +1.1 to +2.2 box AP [V] [54]. Detectron2 ships `FrozenBatchNorm2d` and a `freeze()` helper [V] [56].
- **Rethinking "batch" in BatchNorm** [V] [55]:
  - With a batch of 2, precise (recomputed) population statistics beat the EMA statistics: error 35.5→32.0.
  - Switching to frozen BN for the last 20 of 100 epochs recovers 25.2% error.
  - Re-estimating statistics on a shifted evaluation domain helps under shift (contrast 40.8→33.1) and slightly hurts clean data (23.4→23.8).
- **Our decoder** (our reading of `model.py`):
  - BN in the PPM bins 2/4/8, the lateral, smooth and fuse convs, and the bottleneck. The 1×1 bin already has none.
  - At batch 8 and 256 px, the 2×2 bin normalises over only 8×4 = 32 values per channel. The rest see thousands.
  - `freeze()` freezes encoder parameters only, so decoder BN statistics always update in `model.train()`. During user fine-tuning they drift toward the target. That is AdaBN-like adaptation for the target, but it is source forgetting in the normalisation.
- **Options:**
  1. Keep BN as it is (baseline).
  2. Decoder BN in eval mode during user fine-tuning: affine still trains, statistics frozen. This is detection practice [54][56].
  3. Re-estimate decoder BN statistics on the user's chips after training, with no gradient (PreciseBN/AdaBN) [55][78].
  4. Use GroupNorm in the decoder. This removes the batch dependence, but it is an architecture change that needs a new base model; it cannot be swapped in later.

## 6. Q5: Loss, imbalance and threshold

| Loss | Evidence (domain) | Source |
|---|---|---|
| Lovász hinge / softmax | Binary VOC per-class IoU: CE 77.14 → Lovász hinge 80.5, fine-tuned from a CE network. Cityscapes ENet 58.29 → 63.06. Image-mIoU optimisation can hurt dataset-mIoU [V] | [59] |
| CE + Lovász-softmax λ 0.6 | Used for the best seagrass UPerNet result; λ sensitivity studied [V] | [82] |
| Focal | γ 2, α 0.25: +2.9 AP vs α-balanced CE (detection) [V] | [60] |
| Tversky α 0.3 / β 0.7 | MS lesions: DSC 53.42→56.42, sensitivity 49.85→56.85 vs Dice (medical) [V] | [61] |
| Compound losses | 20 losses on 6 medical datasets: no single best; compound losses (Dice + CE/TopK/focal/boundary) most robust [V] | [62] |
| Soft Dice / Jaccard / Lovász vs CE | Six medical tasks: metric-sensitive losses beat weighted CE when scored by Dice/IoU [V]; no significant difference between them [?] | [63] |
| Survey | 25 losses, each combined 0.5/0.5 with CE; "choice depends more on the data than the network" [V] | [64] |

Calibration and threshold:
- **Dice makes outputs overconfident.** Medical NLL, CE vs Dice: brain 0.52 vs 0.62, heart 0.36 vs 0.62, prostate 0.40 vs 0.74, while Dice scores higher on Dice. Ensembles of 5 cut NLL by about 44–66% [V] [66].
- **Temperature scaling:** ECE 16.53%→1.26% on CIFAR-100 ResNet-110 [V] [65]. For a single sigmoid it does not move which pixels pass a threshold: it is monotonic, and the threshold moves with it. It fixes probabilities, not the mask.
- **F1-optimal threshold** for calibrated scores = half the best F1, so below 0.5 for any realistic F1 [V] [67].
- **Segmentation calibration** depends on capacity, crop size and multi-scale testing; mispredictions drive overconfidence [V abstract] [68].
- **Our inference, untested:** the base model's swept threshold will not be optimal after a user fine-tune on a new site, because the class balance and the calibration change. Re-sweep on the user's validation chips.

For us: keep BCE + Dice for both jobs. Lovász hinge as a fine-tuning-stage loss is the one alternative with a binary IoU gain (+3.4 on VOC [59]). Tversky only if users value recall over precision.

## 7. Q6: Remote-sensing, aerial and drone evidence

- **ImageNet vs remote-sensing pretraining (UPerNet):** ImageNet ≥ MillionAID pretraining on segmentation [V] [69]:
  - Potsdam mF1, Swin-T: 90.60 vs 90.03.
  - iSAID mIoU: R50 61.9 vs 61.6; Swin-T 64.6 vs 64.1; ViTAEv2-S 65.3 vs 64.3.
- **Benchmarks:**
  - Matching pretraining resize and normalisation is worth up to +32 points; ImageNet stays competitive [V] [70].
  - GEO-Bench: remote-sensing pretrained models no better than timm ImageNet weights (classification). ConvNeXt degrades less than SwinV2 as training data shrinks [V] [71].
  - PANGAEA (frozen encoder + UPerNet): foundation models do not reliably beat a supervised UNet. Fine-tuning the encoder is inconsistent: DOFA 43.82→63.74 on FBP, RemoteCLIP 76.59→65.13 on Burn Scars [V] [72].
- **Cross-site shift:**
  - Seagrass, 3 cm drone → 12.5 cm aircraft, SegFormer + histogram matching: best leave-one-out F1 0.52 / IoU 0.48, 68% of an oracle [V] [73].
  - LoveDA mIoU: source-only 31.27, best UDA 41.32, oracle 46.79 (Rural→Urban); 31.74 / 38.44 / 45.67 (Urban→Rural) [V] [74].
  - Potsdam→Vaihingen (sensor change): IoU 0.17 → 0.30 with GAN adaptation [V] [75].
  - **No study reports fine-tuning curves by number of labelled target tiles** for aerial or drone segmentation. One IGARSS 2019 abstract was seen but not read [?].
- **Test-time adaptation:**
  - TENT updates normalisation affine parameters and re-estimates statistics. GTA→Cityscapes: source 28.8, BN-statistics adaptation 31.4, TENT 35.8. BN networks only [V] [76].
  - BN-statistics adaptation: ImageNet-C mCE 76.7→62.2 with as few as 32 samples [V] [77]. AdaBN is parameter-free [V] [78].
  - In segmentation, classic TTA "only brings slight improvement" and sometimes hurts [V abstract] [79].
  - Satellite DeepGlobe with synthetic corruptions: noise 35.2→48.2 (TENT) [V] [80].
  - BN makes TTA unstable at small batch; LN/GN models adapt more stably but can collapse; SAR is the LN-capable method [V] [81].
- **For us:**
  - Online TTA inside CPU ONNX serving is out: the graph is static and has no gradient.
  - The usable form is **offline**: re-estimate decoder BN statistics on the user's (even unlabelled) target tiles before export. Same graph, no labels needed.
  - ConvNeXt's LayerNorm has no running statistics, so only the decoder's BN can adapt this way.

## 8. Q7: Recommendations for ss-hot

### 8.1 Defaults

| Knob | (a) Base training | (b) User fine-tuning in fAIr | Evidence |
|---|---|---|---|
| What trains | everything (`freeze_encoder=False`) | decoder only (`freeze_encoder=True`, `unfreeze_from=none`); opt-in `stage3` | §2 [13][18][28][32][83] |
| `learning_rate` (decoder) | 1e-4 | 1e-4 (decoder only); 1e-4 with encoder factor ≤ 0.4 if stage3 is unfrozen | [2][83] |
| `encoder_lr_factor` | 0.81 (= official multipliers) to test against 1.0 | 0.1–0.4 when stages are unfrozen (pack precedents 0.1 and 0.385) | [2][4][8][83] |
| `layer_decay` | 0.9 | 0.9 | [1][2][25] |
| `weight_decay` | 0.05 | 0.05 (L2-SP as a replacement, after testing) | [2][19][25] |
| `drop_path_rate` | test 0.1 vs 0.3 (official 0.4) | keep the base value (inactive when the encoder is frozen) | [2] |
| `warmup_steps` | 400 is fine (≈ 0.6% of 68.5k steps) | ~1 epoch (≈ 40 for 300 chips at batch 8), not 400 | [2] + our arithmetic |
| `epochs` / patience | 40 / 8 | short, e.g. 15–25 / 5; no source fixes these numbers | [34] |
| `batch_size` | 8 (16 if memory allows: closer to official, BN less noisy) | 8 | [2][54] |
| Decoder BN | train mode | try eval-mode BN, or re-estimate after training | [54][55][56] |
| Loss | BCE + Dice | BCE + Dice | [62][63] |
| Threshold | sweep on validation | **re-sweep on the user's validation chips** | [66][67] |
| `max_grad_norm` | 1.0 (harmless; official none) | 1.0 | [2] |

### 8.2 Knobs to expose to fAIr users

- **Now:** `epochs`, `batch_size`, `learning_rate`, `freeze_encoder`, `unfreeze_from`, `encoder_lr_factor`, `warmup_steps` (with a small default), `early_stop_patience`, and the post-process `threshold`.
- **Keep internal:** `layer_decay`, `drop_path_rate`, `weight_decay`, `max_grad_norm`, loss weights.
- **Add only after an experiment shows a gain** (repo rule: no speculative knobs): an L2-SP weight, a WiSE blend factor, a decoder-BN mode, a replay fraction.

### 8.3 Cheap experiments (each a `research/` script)

Set-up for E4–E10: simulate a user by holding out one of the four flights, or a different-site seagrass set if one is available. Take a few hundred labelled chips from it. Score **target IoU** on its other chips and **source retention** (IoU on the base validation tiles ÷ base IoU). Use 2–3 seeds; the runs are minutes each.

| # | Job | Experiment | What it decides |
|---|---|---|---|
| E1 | a | drop_path 0.1 vs 0.3, with `epoch_fraction` for short runs | drop_path default |
| E2 | a | `encoder_lr_factor` 1.0 vs 0.81 | whether to match the official multipliers |
| E3 | a | LP-FT: 1–2 epochs decoder-only, then full run from that checkpoint (two `train()` calls) vs current | decoder warm-up |
| E4 | b | decoder-only vs `unfreeze_from` stage3 / stage2 vs all trainable (factor 0.1), target vs retention | default and opt-in |
| E5 | b | `warmup_steps` 400 vs ~1 epoch | user warmup default |
| E6 | b | decoder BN train vs eval mode; plus post-hoc BN re-estimation on target chips (no labels) | BN handling |
| E7 | b | WiSE: blend base and fine-tuned weights at α = 0, 0.25 … 1, eval only | a free retention knob |
| E8 | b | L2-SP (‖w − w_base‖², replacing wd on trained weights) at 2–3 strengths | regulariser for unfrozen runs |
| E9 | b | LoRA r = 8–16 on stage2–3 `mlp.fc1/fc2`, decoder fully trained, merged before export; and bias + norm + LayerScale-only | whether PEFT beats decoder-only here |
| E10 | b | threshold: base value vs re-swept after fine-tuning | threshold policy |
| E11 | b | replay: mix a share of base tiles (e.g. 10–25%, an untested choice) into user batches | worth asking fAIr for base-data access? |

### 8.4 What not to do

- Don't use adapters, side networks, Conv-Adapter or Mona for user fine-tuning. They add branches to the ONNX graph and CPU latency [39][40][47].
- Don't run online TTA in serving. ONNX on CPU has no gradient, and segmentation TTA gains are small or negative [79].
- Don't fine-tune everything at lr 1e-4 on a few hundred chips without layer decay and a low encoder factor. That is the 32–66% retention case [83][32].
- Don't trust target-validation early stopping to control forgetting; it doesn't measure it [34].
- Don't use timm EMA with its 0.9999 default on a ~40-step-per-epoch run; the decay must be sized to the run [27].
- Don't put LoRA on the depthwise 7×7 convs until PEFT support is confirmed [36].
- Don't swap BN for GN in the decoder of an existing base model; it needs a new base training.
- Don't keep 400 warmup steps for small user datasets.
- Don't treat 0.5 as the threshold after any fine-tune.

## 9. Gaps and uncertainties

- **No evidence from our setting.** No study fine-tunes a ConvNeXt + UPerNet segmenter on a few hundred aerial or drone chips of a new site and reports both target gain and source forgetting. The closest are FTP (Swin-T, PASCAL-Context, corruptions) [25], Cityscapes→ACDC (ResNet-50) [32] and our own tree detector [83].
- **Untested here:**
  - LP-FT, surgical fine-tuning, WiSE-FT, model soups and LoRA-forgets-less are classification, LLM or ViT results.
  - No PEFT study on ConvNeXt measures forgetting or the few-shot regime.
- **No source** for drop_path, layer decay or epochs on small datasets or 256 px crops, or for re-tuning a threshold after fine-tuning.
- **Platform questions:**
  - Can a fAIr user fine-tuning step read the base training tiles (replay, source-retention check)?
  - Does fAIr re-sweep the threshold after a user fine-tune, or keep the STAC default?
  - `freeze_encoder=True` is the library default (`params.py`); the pack's STAC must declare the same default.
- **Not read in full:** DELTA, PLOP, LwF and Mensink tables; LoRand (403); UPetu numbers; Combo loss. The IGARSS 2019 few-sample adaptation paper was not identified with certainty.
- **Uncertain specifics:** whether HF PEFT supports LoRA on grouped/depthwise Conv2d; the exact form of the Adam square-root scaling rule; some medical ECE numbers in [66].

## 10. Sources

1. Liu Z. et al. (2022). A ConvNet for the 2020s. CVPR. https://arxiv.org/abs/2201.03545 [V]
2. ConvNeXt official ADE20K config, ConvNeXt-T UPerNet. https://github.com/facebookresearch/ConvNeXt/blob/main/semantic_segmentation/configs/convnext/upernet_convnext_tiny_512_160k_ade20k_ms.py [V]
3. ConvNeXt base UPerNet model config. https://github.com/facebookresearch/ConvNeXt/blob/main/semantic_segmentation/configs/_base_/models/upernet_convnext.py [V]
4. ConvNeXt layer-decay optimiser constructor. https://github.com/facebookresearch/ConvNeXt/blob/main/semantic_segmentation/mmcv_custom/layer_decay_optimizer_constructor.py [V]
5. MMSegmentation ConvNeXt configs and results. https://github.com/open-mmlab/mmsegmentation/tree/main/configs/convnext [V]
6. Kumar A. et al. (2022). Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution. ICLR. https://arxiv.org/abs/2202.10054 [V]
7. Lee Y. et al. (2023). Surgical Fine-Tuning Improves Adaptation to Distribution Shifts. ICLR. https://arxiv.org/abs/2210.11466 [V]
8. Howard J., Ruder S. (2018). Universal Language Model Fine-tuning for Text Classification. ACL. https://arxiv.org/abs/1801.06146 [V]
9. Clark K. et al. (2020). ELECTRA. ICLR. https://arxiv.org/abs/2003.10555 [V]
10. Bao H. et al. (2022). BEiT: BERT Pre-Training of Image Transformers. ICLR. https://arxiv.org/abs/2106.08254 [V]
11. Yosinski J. et al. (2014). How transferable are features in deep neural networks? NIPS. https://arxiv.org/abs/1411.1792 [V]
12. Kornblith S., Shlens J., Le Q. (2019). Do Better ImageNet Models Transfer Better? CVPR. https://arxiv.org/abs/1805.08974 [V]
13. He K., Girshick R., Dollár P. (2019). Rethinking ImageNet Pre-training. ICCV. https://arxiv.org/abs/1811.08883 [V]
14. Zoph B. et al. (2020). Rethinking Pre-training and Self-training. NeurIPS. https://arxiv.org/abs/2006.06882 [V]
15. Mensink T. et al. (2021). Factors of Influence for Transfer Learning across Diverse Appearance Domains and Task Types. TPAMI. https://arxiv.org/abs/2103.13318 [V abstract]
16. Raghu M. et al. (2019). Transfusion: Understanding Transfer Learning for Medical Imaging. NeurIPS. https://arxiv.org/abs/1902.07208 [V]
17. Zhai X. et al. (2019). A Large-scale Study of Representation Learning with the Visual Task Adaptation Benchmark. https://arxiv.org/abs/1910.04867 [V]
18. Wang X. et al. (2020). Frustratingly Simple Few-Shot Object Detection. ICML. https://arxiv.org/abs/2003.06957 [V]
19. Li X., Grandvalet Y., Davoine F. (2018). Explicit Inductive Bias for Transfer Learning with Convolutional Networks (L2-SP). ICML. https://arxiv.org/abs/1802.01483 [V]
20. Li X. et al. (2019). DELTA: DEep Learning Transfer using Feature Map with Attention. ICLR. https://arxiv.org/abs/1901.09229 [V abstract]
21. Kirkpatrick J. et al. (2017). Overcoming catastrophic forgetting in neural networks (EWC). PNAS. https://arxiv.org/abs/1612.00796 [V abstract; volume ?]
22. Wortsman M. et al. (2022). Robust fine-tuning of zero-shot models (WiSE-FT). CVPR. https://arxiv.org/abs/2109.01903 [V]
23. Wortsman M. et al. (2022). Model soups. ICML. https://arxiv.org/abs/2203.05482 [V abstract]
24. Tian J. et al. (2023). Trainable Projected Gradient Method for Robust Fine-tuning (TPGM). CVPR. https://arxiv.org/abs/2303.10720 [V abstract]
25. Tian J. et al. (2023). Fast Trainable Projection for Robust Fine-Tuning (FTP). NeurIPS. https://arxiv.org/abs/2310.19182 [V, Table 4]
26. Tarvainen A., Valpola H. (2017). Mean teachers are better role models. NeurIPS. https://arxiv.org/abs/1703.01780 [V]
27. timm `ModelEmaV2`/`ModelEmaV3`. https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py [V]
28. Michieli U., Zanuttigh P. (2019). Incremental Learning Techniques for Semantic Segmentation (ILT). ICCVW. https://arxiv.org/abs/1907.13372 [V]
29. Cermelli F. et al. (2020). Modeling the Background for Incremental Learning in Semantic Segmentation (MiB). CVPR. https://arxiv.org/abs/2002.00718 [V]
30. Douillard A. et al. (2021). PLOP: Learning without Forgetting for Continual Semantic Segmentation. CVPR. https://arxiv.org/abs/2011.11390 [V abstract]
31. Kalb T. et al. (2021). Continual Learning for Class- and Domain-Incremental Semantic Segmentation. IEEE IV. https://arxiv.org/abs/2209.08023 [V abstract]
32. Kalb T., Beyerer J. (2023). Principles of Forgetting in Domain-Incremental Semantic Segmentation in Adverse Weather Conditions. CVPR. https://arxiv.org/abs/2303.14115 [V]
33. Li Z., Hoiem D. (2016). Learning without Forgetting. ECCV / TPAMI. https://arxiv.org/abs/1606.09282 [V abstract]
34. Andreassen A. et al. (2021). The Evolution of Out-of-Distribution Robustness Throughout Fine-Tuning. https://arxiv.org/abs/2106.15831 [V abstract; venue ?]
35. Hu E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR. https://arxiv.org/abs/2106.09685 [? not re-read; mergeability confirmed in [36]]
36. Hugging Face PEFT docs: custom models and LoRA reference. https://huggingface.co/docs/peft/developer_guides/custom_models ; https://huggingface.co/docs/peft/package_reference/lora [V]
37. timm ConvNeXt implementation. https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py [V]
38. Lou et al. (2026). Parameters as Experts: Adapting Vision Models with Dynamic Parameter Routing (ParaX). https://arxiv.org/abs/2602.06862 [V]
39. Chen H. et al. (2024). Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets. CVPRW. https://arxiv.org/abs/2208.07463 [V]
40. Yin D. et al. (2025). 5%>100%: Breaking Performance Shackles of Full Fine-Tuning on Visual Recognition Tasks (Mona). CVPR. https://arxiv.org/abs/2408.08345 [V]
41. Yin D. et al. (2023). 1% VS 100%: Parameter-Efficient Low Rank Adapter for Dense Predictions (LoRand). CVPR. https://openaccess.thecvf.com/content/CVPR2023/html/Yin_1_VS_100_Parameter-Efficient_Low_Rank_Adapter_for_Dense_Predictions_CVPR_2023_paper.html [? page 403; numbers via [40]]
42. Lian D. et al. (2022). Scaling & Shifting Your Features (SSF). NeurIPS. https://arxiv.org/abs/2210.08823 [V]
43. Ben Zaken E., Ravfogel S., Goldberg Y. (2022). BitFit. ACL. https://arxiv.org/abs/2106.10199 [V]
44. Cai H. et al. (2020). TinyTL: Reduce Activations, Not Trainable Parameters. NeurIPS. https://arxiv.org/abs/2007.11622 [V]
45. Frankle J., Schwab D., Morcos A. (2021). Training BatchNorm and Only BatchNorm. ICLR. https://arxiv.org/abs/2003.00152 [V]
46. Zhao B. et al. (2023). Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning. https://arxiv.org/abs/2312.11420 [V]
47. Zhang J. et al. (2020). Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks. ECCV. https://arxiv.org/abs/1912.13503 [V abstract]
48. Aleem S. et al. (2024). ConvLoRA and AdaBN based Domain Adaptation via Self-Training. https://arxiv.org/abs/2402.04964 [V]
49. Biderman D. et al. (2024). LoRA Learns Less and Forgets Less. TMLR. https://arxiv.org/abs/2405.09673 [V]
50. Mai Z. et al. (2025). Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning in Visual Recognition. CVPR. https://arxiv.org/abs/2409.16434 [V]
51. Marti-Escofet F. et al. (2025). Fine-tune Smarter, Not Harder: Parameter-Efficient Fine-Tuning for Geospatial Foundation Models. https://arxiv.org/abs/2504.17397 [V]
52. Sauder J. et al. (2025). The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs. https://arxiv.org/abs/2503.20000 [V]
53. Dong Z., Gu Y., Liu T. (2024). UPetu: A Unified Parameter-Efficient Fine-Tuning Framework for Remote Sensing Foundation Model. IEEE TGRS 62. https://www.aminer.org/pub/66079bf913fb2c6cf66bf8f8 [?]
54. Wu Y., He K. (2018). Group Normalization. ECCV. https://arxiv.org/abs/1803.08494 [V]
55. Wu Y., Johnson J. (2021). Rethinking "Batch" in BatchNorm. https://arxiv.org/abs/2105.07576 [V]
56. Detectron2 layers docs (`FrozenBatchNorm2d`). https://detectron2.readthedocs.io/en/latest/modules/layers.html [V]
57. Goyal P. et al. (2017). Accurate, Large Minibatch SGD. https://arxiv.org/abs/1706.02677 [V]
58. Malladi S. et al. (2022). On the SDEs and Scaling Rules for Adaptive Gradient Algorithms. https://arxiv.org/abs/2205.10287 [V abstract; exact rule ?]
59. Berman M., Rannen Triki A., Blaschko M. (2018). The Lovász-Softmax loss. CVPR. https://arxiv.org/abs/1705.08790 [V]
60. Lin T.-Y. et al. (2017). Focal Loss for Dense Object Detection. ICCV. https://arxiv.org/abs/1708.02002 [V]
61. Salehi S. et al. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. MLMI. https://arxiv.org/abs/1706.05721 [V]
62. Ma J. et al. (2021). Loss odyssey in medical image segmentation. Medical Image Analysis 71:102035. https://doi.org/10.1016/j.media.2021.102035 [V]
63. Eelbode T. et al. (2020). Optimization for Medical Image Segmentation: Theory and Practice When Evaluating With Dice Score or Jaccard Index. IEEE TMI. https://arxiv.org/abs/2010.13499 [V]
64. Azad R. et al. (2023). Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook. https://arxiv.org/abs/2312.05391 [V]
65. Guo C. et al. (2017). On Calibration of Modern Neural Networks. ICML. https://arxiv.org/abs/1706.04599 [V]
66. Mehrtash A. et al. (2020). Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation. IEEE TMI. https://arxiv.org/abs/1911.13273 [V]
67. Lipton Z., Elkan C., Naryanaswamy B. (2014). Optimal Thresholding of Classifiers to Maximize F1 Measure. ECML-PKDD. https://arxiv.org/abs/1402.1892 [V]
68. Wang D., Gong B., Wang L. (2023). On Calibrating Semantic Segmentation Models: Analyses and An Algorithm. CVPR. https://arxiv.org/abs/2212.12053 [V abstract]
69. Wang D. et al. (2022). An Empirical Study of Remote Sensing Pretraining. IEEE TGRS. https://arxiv.org/abs/2204.02825 [V]
70. Corley I. et al. (2023). Revisiting pre-trained remote sensing model benchmarks: resizing and normalization matters. https://arxiv.org/abs/2305.13456 [V; venue ?]
71. Lacoste A. et al. (2023). GEO-Bench: Toward Foundation Models for Earth Monitoring. https://arxiv.org/abs/2306.03831 [V; venue ?]
72. Marsocci V. et al. (2024). PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models. https://arxiv.org/abs/2412.04204 [V]
73. Pawar et al. (2025). Eye in the Sky for Sub-Tidal Seagrass Mapping. Remote Sensing 17(14):2518. https://doi.org/10.3390/rs17142518 [V abstract]
74. Wang J. et al. (2021). LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation. NeurIPS D&B. https://arxiv.org/abs/2110.08733 [V]
75. Benjdira B. et al. (2019). Unsupervised Domain Adaptation using GANs for Semantic Segmentation of Aerial Images. Remote Sensing 11(11):1369. https://arxiv.org/abs/1905.03198 [V]
76. Wang D. et al. (2021). Tent: Fully Test-Time Adaptation by Entropy Minimization. ICLR. https://arxiv.org/abs/2006.10726 [V]
77. Schneider S. et al. (2020). Improving robustness against common corruptions by covariate shift adaptation. NeurIPS. https://arxiv.org/abs/2006.16971 [V]
78. Li Y. et al. (2016). Revisiting Batch Normalization For Practical Domain Adaptation (AdaBN). https://arxiv.org/abs/1603.04779 [V abstract]
79. Yi C. et al. (2023). From Question to Exploration: Test-Time Adaptation in Semantic Segmentation? https://arxiv.org/abs/2310.05341 [V abstract]
80. Niloy F. et al. (2024). Source-Free Online Domain Adaptive Semantic Segmentation of Satellite Images under Image Degradation. https://arxiv.org/abs/2401.02113 [V]
81. Niu S. et al. (2023). Towards Stable Test-Time Adaptation in Dynamic Wild World (SAR). ICLR. https://arxiv.org/abs/2302.12400 [V]
82. Lian J. et al. (2026). Intertidal Seagrass Mapping Using UAV Visible and Multispectral Imagery. Remote Sensing 18:2057. https://doi.org/10.3390/rs18122057 [V, via `docs/model.md`]
83. In-house: fAIr-models `models/deepforest_tree/stac-item.json` and `README.md` (`unfreeze_from`, `encoder_lr_factor` descriptions; NEON retention) and `models/deepforest_unet_tree/stac-item.json` (`encoder_lr_factor` 0.1). Local files, read 2026-11-21 [V]; the 32–66% / 96% retention figures are from the brief.
