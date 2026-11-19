# The model: UPerNet + ConvNeXt-Tiny v1

How `ss-hot` builds, trains and exports its network under the fAIr helper-library constraints (reference
library `fair_models/tree/ts-hot`). Written 2026-11-19. Nothing here was installed, run or measured.

Marks: **[V]** read in source code, release notes or an API response (URL in Sources); **[E]** my estimate;
**[?]** not verified, to check in the first run with the `train` extra.

## 1. Decision

- **Network:** `smp.UPerNet` with the timm universal encoder `tu-convnext_tiny.fb_in22k_ft_in1k` (ConvNeXt-T v1,
  ImageNet-22k pre-trained, fine-tuned on 1k, Apache-2.0 [V]). ConvNeXt-V2 weights (`convnextv2_*`) are
  CC-BY-NC-4.0 [V] and must not be used.
- **Why:** Lian et al. 2026 (Remote Sens. 18:2057) found UPerNet-ConvNeXtV2-T best on UAV intertidal seagrass,
  ahead of U-Net and DeepLabV3+ [V, abstract only]. The 4-6 mIoU margin is the figure given in the brief; I did not
  re-read it in the full text [?]. Their inputs included multispectral bands and vegetation indices; ours are RGB
  only, so the gain may be smaller.
- **Baseline:** `smp.Unet("resnet34")` (or `"efficientnet-b2"`), built from the same package, same loss, same
  windows. Ship UPerNet only if it beats the baseline on the test split.
- **Versions** (PyPI JSON, 2026-11-19) [V]: `segmentation-models-pytorch` **0.6.0** (2026-10-26, MIT);
  `timm` **1.0.31** (2026-11-03, Apache-2.0); `torch` 2.15.0; `onnx` 1.23.2; `onnxruntime` 1.32.0.
  UPerNet first shipped in smp **0.4.0** (2025-01-08) and was rewritten incompatibly in **0.5.0** ("UperNet model
  weights trained with v0.4.0 will be not compatible") [V]. Pin the `train` extra to
  `segmentation-models-pytorch>=0.6,<0.7` and `timm>=1.0.20` (any 1.0.x with the tag works; the floor is a choice).

## 2. Building it

**Does smp UPerNet + `tu-convnext_tiny` work? Yes, cleanly [V, source of v0.6.0].**
`TimmUniversalEncoder` sees ConvNeXt's reductions `[4, 8, 16, 32]`, classes it "transformer-style", asks timm for
`out_indices=(0,1,2,3)` and inserts an empty 0-channel 1/2-scale feature. `UPerNetDecoder` uses only `features[2:]`
(1/4 to 1/32: 96, 192, 384, 768 channels), applies a `LayerNorm2d` to each (as mmseg does for ConvNeXt), then PPM on
1/32, FPN laterals, 3x3 convolutions and a 3x3 fusion at 1/4; the head is a 1x1 convolution and a x4 bilinear
upsample (`align_corners=True`). The v0.6.0 notes give `tu-convnext_tiny.fb_in22k` as the example of naming the
pretrained tag in the encoder name; `encoder_weights=True` downloads it, `None` builds random weights offline. Extra
keyword arguments (for example `drop_path_rate`) go to `timm.create_model`. The UPerNet decoder code is MIT (it is not
in smp's `licenses/LICENSES.md` exceptions) [V].

```python
smp.UPerNet(encoder_name="tu-convnext_tiny.fb_in22k_ft_in1k", encoder_weights=True or None,
            decoder_channels=256, classes=1, activation=None, drop_path_rate=cfg.drop_path_rate)
model.decoder.psp = PSPModule(768, 256, sizes=PPM_BINS)  # export-safe bins, see section 4
```

wrapped in a `SeagrassNet(nn.Module)` like ts-hot's `TreeUNet`: ImageNet mean/std as non-persistent buffers
(`(0.485,0.456,0.406)`, `(0.229,0.224,0.225)`, identical to the timm `pretrained_cfg` [V]), input RGB in [0, 1],
output one logit per pixel.

Gotchas found in the source [V]:

1. **Input divisible by 32.** `SegmentationModel.forward` raises otherwise (the check is skipped while tracing).
   `WINDOW_PX = 512` satisfies it.
2. **PPM bins are hard-coded `(1, 2, 3, 6)`** in `UPerNetDecoder`; `PSPModule(..., sizes=...)` accepts others, so
   replace `model.decoder.psp` after construction. This touches a non-public smp class: cover it with a test and
   keep the `<0.7` cap.
3. **BatchNorm on a 1x1 map.** UPerNet's PPM bin 1 has a BatchNorm over a `(B, 256, 1, 1)` tensor (smp's PSPNet
   avoids this, UPerNet does not). A training batch of 1 raises; use `drop_last=True` and `batch_size >= 4`.
4. **Pretrained download in the pod.** The weights are `timm/convnext_tiny.fb_in22k_ft_in1k`, HF commit
   `bc48a87f119bae1b06418f5340bfa97ef75d4609`, `model.safetensors` (about 115 MB, from 28.6M fp32 params [E]).
   For a pinned, reproducible build, the pack declares
   `https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k/resolve/bc48a87f.../model.safetensors` and the library
   loads it with `encoder_weights=None` plus `pretrained_cfg_overlay=dict(file=path)` or a strict state-dict load with
   `stem.N -> stem_N`, `stages.N -> stages_N` key renames (timm flattens those names in `features_only`) [?].

**Alternative, if smp were unusable:** timm `features_only=True` encoder plus a hand-written UPerNet head (LayerNorm
per level, PPM, three laterals, three 3x3 FPN convolutions, fusion, 1x1 head): about 90-120 lines [E]. Not
recommended: smp works, is tested upstream, and also builds the U-Net baseline.

## 3. Size and cost

| | Params | Compute at 512x512 | Source |
|---|---|---|---|
| ConvNeXt-T encoder (no head) | ~27.8M | ~23.5 GMACs | 28.6M / 4.5 GMACs at 224 on the HF card [V], scaled [E] |
| smp UPerNet decoder, 256 ch | ~9.2M | ~53 GMACs (fusion 3x3 at 1/4 is ~39) | counted from the layer shapes [E] |
| **smp UPerNet-ConvNeXt-T** | **~37M** | **~76 GMACs (~150 GFLOPs)** | [E] |
| mmseg/paper UPerNet-ConvNeXt-T (512 ch + aux head) | 60M | 939 GFLOPs at 2048x512, i.e. ~235 at 512x512 | ConvNeXt README and paper Table 4 caption [V]; my count of the 512-ch head agrees [E] |
| smp U-Net-ResNet34 baseline | ~24M | ~30-45 GMACs | [E] |

ONNX file ~150 MB in fp32 [E]. CPU onnxruntime latency: at an effective 100-200 GFLOP/s on a few cores, about
0.8-1.5 s per 512 window and a quarter of that per 256 window [E, nothing measured]. The decoder costs more than the
encoder; if CPU time is a problem, `decoder_channels=128` cuts the fusion convolution about fourfold before any
encoder change.

## 4. ONNX export

Contract, matching `ss_hot/serve.py` (other session): input `image` `(N, 3, WINDOW_PX, WINDOW_PX)` float32 RGB in
[0, 1]; output `probability` `(N, WINDOW_PX, WINDOW_PX)`; normalisation + network + sigmoid inside; only the batch
axis dynamic. Export like ts-hot: `torch.onnx.export(..., opset_version=17, dynamo=False,
dynamic_axes={"image": {0: "batch"}, "probability": {0: "batch"}})`, `onnx.checker`, return bytes. Note torch >= 2.9
defaults to `dynamo=True` and warns on the legacy path [V]; ts-hot passes `dynamo=False` explicitly, keep that.

- **Adaptive pooling (the blocker).** The legacy exporter turns `adaptive_avg_pool2d` into `AveragePool` only when the
  input size is static and divisible by the output size; otherwise "output size that are not factor of input size"
  or "input size not accessible" [V, `symbolic_opset9._adaptive_pool` in torch 2.15]. A 512 window gives a 16x16 map
  at 1/32: bins 3 and 6 fail. Hence **fixed spatial size + `PPM_BINS = (1, 2, 4, 8)`**, and a build-time assert that
  `WINDOW_PX // 32` is divisible by every bin (512 -> 16 and 256 -> 8 both pass). The decoder is trained from scratch,
  so changing the bins loses nothing pretrained. smp 0.6.0 fixed this for PSPNet only (`F.interpolate(mode="area")`
  under export, PR #1270) [V], not for UPerNet's `PSPModule`; I did not check whether `area` really avoids the same
  symbolic [?]. If `(1, 2, 3, 6)` is wanted, replace the pooling with a fixed pooling-matrix `einsum` (exact, ~20 lines).
- **LayerNorm:** opset 17 emits `LayerNormalization`; timm ConvNeXt's channels-last LayerNorm adds `Transpose`
  nodes, which is fine. **GELU** has an ONNX op only from opset 20; at 17 it becomes an `Erf` expression (exact).
  **Depthwise 7x7** is `Conv` with `group=C`. **Bilinear resize** with fixed sizes exports as `Resize` [V for the ops
  existing; exact node list ?].
- The encoder's empty 1/2-scale tensor is unused by the decoder and should be pruned; check the graph [?].
- **Window size is an open question for the preprocessing design.** At `MODEL_GSD_M = 0.03` a training chip
  (512 px at ~0.92 cm/px, ~4.7 m) shrinks to ~157 px, so a 512 window is mostly padding. A 256 window (bins still
  valid) would be about 4x cheaper per chip; decide with the preprocessing owner.

## 5. Training recipe

Reference (ConvNeXt `upernet_convnext_tiny_512_160k_ade20k_ms.py`) [V]: AdamW, lr 1e-4, betas (0.9, 0.999),
weight decay 0.05, stage-wise decay 0.9 with `num_layers` 6, linear warmup 1,500 iterations then linear (poly 1.0)
decay, drop path 0.4, 160k iterations at batch 16, 512 crops, AMP. The paper's appendix sweeps lr {8e-5, 1e-4},
decay {0.8, 0.9}, drop path {0.3, 0.4, 0.5} [V]. mmseg's rule gives decoder x1.0, stage *i* x0.9^(6-i) (stage3 0.73
... stage0 0.53), stem and downsampling layers x0.9^7 = 0.48, and no weight decay on 1-D parameters (norms, biases,
layer-scale `gamma`) [V].

160k x 16 is ~745 passes over our 3,434 chips; far too many for a pre-trained fine-tune. Adapted:

- **Loss:** BCE + Dice over non-`IGNORE` pixels, as ts-hot. Seagrass is 59% of training pixels, so focal loss is not
  needed; Lovasz is better as a late fine-tune than from the start. Under-mapping means "no label" is sometimes
  seagrass: the cheapest defence is in the data (ignore or drop unlabeled chips from the validation metric).
  Bootstrapped BCE or label smoothing only if a run shows the model learning the omissions; no knob until then.
- **Optimiser:** AdamW, lr 1e-4, wd 0.05, stage-wise decay 0.9, grad clip 1.0, warmup ~1 epoch, cosine to 0.
- **Batch and memory:** mmseg reports 4.23 GB for UPerNet-ConvNeXt-T at 2 x 512 with AMP (V100, 512-ch head) [V].
  For the lighter smp head at 512 with AMP: batch 4 ~5-6 GB, batch 8 ~8-10 GB, batch 16 ~16-18 GB [E]; at 256 windows
  about a quarter of the activation memory [E]. The pod's GPU memory is unknown: default 8, log
  `peak_gpu_mem` (ts-hot already does) and lower the batch if it does not fit. BatchNorm in the decoder wants >= 4.
- **Epochs:** 40 with early stopping, patience 8 [E].
- **Monitor:** validation IoU at 0.5 on validation chips that have labels (226 of 490 are unlabelled although they
  show seagrass), not validation loss. Report the all-chip number too.
- **Freezing:** none by default (LayerNorm has no running statistics, layer decay protects the early stages).
  `freeze_encoder` + `unfreeze_from="stage3"` stays as a comparison run.
- **Drop path 0.1** (sweep 0.1-0.3), **AMP** (bf16 where supported, else fp16 + `GradScaler`), **channels_last**
  for model and inputs on CUDA (expected faster depthwise convolutions [E]). GPU when visible, CPU with a warning,
  epoch callback, best weights kept in memory, temporary folders removed: as ts-hot.

## 6. Parameters

Architecture constants (not user parameters): `ENCODER_NAME = "tu-convnext_tiny.fb_in22k_ft_in1k"`,
`ENCODER_CHANNELS = (96, 192, 384, 768)`, `DECODER_CHANNELS = 256`, `PPM_BINS = (1, 2, 4, 8)`, `WINDOW_PX`,
`IMAGE_MEAN/STD`, `OPSET = 17`, `INPUT_NAME = "image"`, `OUTPUT_NAME = "probability"`,
`ENCODER_STAGES = ("stage0", ..., "stage3")` mapped to `encoder.model.stages_{i}`.

`TrainParams` (names as ts-hot; changes from the current `ss_hot/params.py` marked *):

| key | default | range | note |
|---|---|---|---|
| epochs | 40* | >= 1 | |
| batch_size | 8 | >= 2 (4+ advised) | `drop_last=True` |
| learning_rate (`lr`) | 1e-4 | > 0 | decoder rate |
| weight_decay | 0.05 | >= 0 | not on 1-D params |
| layer_decay* | 0.9 | (0, 1]; 1 = off | new; stage *i* x decay^(6-i) |
| encoder_lr_factor | 1.0 | (0, 1] | on top of layer_decay |
| warmup_steps | 400* | >= 0 | ~1 epoch at batch 8 |
| scheduler | cosine | cosine, none | |
| max_grad_norm | 1.0 | >= 0 | |
| freeze_encoder / unfreeze_from | False / none | stage0-3, none | |
| drop_path_rate | 0.1 | [0, 0.5] | |
| dice_weight | 1.0 | >= 0 | |
| monitor | val_iou* | val_iou, val_loss | labelled chips |
| early_stop_patience | 8 | >= 1 | |
| amp | True | bool | CUDA only |
| epoch_fraction, augmentation keys, device, seed | as now | | |

## 7. Tests (offline: `encoder_weights=None`, no downloads)

1. Shape: `SeagrassNet` (random ConvNeXt-T) maps `(2, 3, 256, 256)` in [0, 1] to `(2, 1, 256, 256)` logits; parameter
   count within 10% of 37M.
2. Bins: `build` rejects a window whose 1/32 size is not divisible by every bin; the PPM uses `PPM_BINS`.
3. Export parity: export to bytes, onnxruntime CPU, names/dtypes/shapes, batch 1 and 3, `max |torch - ort| < 1e-4`,
   output in [0, 1].
4. Freezing and groups: frozen stages have `requires_grad=False`; each stage's learning rate follows `layer_decay`;
   1-D parameters have weight decay 0.
5. Smoke train: 2 epochs on 4 generated chips on CPU, finite loss, callback called twice, best epoch returned, no
   folder left behind.
6. Pretrained (`-m integration`): loading the pinned safetensors leaves no missing or unexpected encoder keys.

## 8. Licences

| Component | Licence |
|---|---|
| smp code incl. UPerNet decoder, `timm_universal` | MIT [V] (exceptions: SegFormer encoder NVIDIA non-commercial, MobileOne Apple; not used) |
| timm code (ConvNeXt) | Apache-2.0 [V] |
| `convnext_tiny.fb_in22k_ft_in1k` weights | Apache-2.0 (HF card) [V]; original repo facebookresearch/ConvNeXt is MIT [V] |
| `convnextv2_*` weights | CC-BY-NC-4.0 [V]: **excluded** |
| Baseline ResNet34 (torchvision) / EfficientNet-B2 (smp `_efficientnet.py`) | BSD-3 / Apache-2.0 code [V]; ImageNet weights' own terms not checked [?] |
| torch, onnx, onnxruntime | BSD-3, Apache-2.0, MIT (well known; not re-checked) |

## Sources

- smp PyPI JSON https://pypi.org/pypi/segmentation-models-pytorch/json; timm https://pypi.org/pypi/timm/json
- smp v0.6.0 source: https://github.com/qubvel-org/segmentation_models.pytorch/tree/v0.6.0/segmentation_models_pytorch
  (`decoders/upernet/`, `encoders/timm_universal.py`, `base/model.py`, `base/heads.py`); releases
  https://github.com/qubvel-org/segmentation_models.pytorch/releases; PR #1270 /pull/1270; `licenses/LICENSES.md`
- timm ConvNeXt v1.0.31: https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/convnext.py
- HF API: https://huggingface.co/api/models/timm/convnext_tiny.fb_in22k_ft_in1k, .../timm/convnextv2_tiny.fcmae_ft_in22k_in1k
- ConvNeXt paper https://arxiv.org/abs/2201.03545; segmentation configs
  https://github.com/facebookresearch/ConvNeXt/tree/main/semantic_segmentation
- mmseg ConvNeXt table https://github.com/open-mmlab/mmsegmentation/tree/main/configs/convnext; layer decay
  https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/engine/optimizers/layer_decay_optimizer_constructor.py
- torch 2.15 exporter: https://github.com/pytorch/pytorch/blob/v2.15.0/torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py
- Lian et al. 2026, https://doi.org/10.3390/rs18122057
