"""Fine-tuning of the seagrass UPerNet (needs the `train` extra; import explicitly, not from `ss_hot`).

Recipe (see docs/research.md, model): AdamW with no weight decay on norms and biases, a linear warmup then cosine
decay stepped per batch, gradient clipping, mixed precision on CUDA, binary cross-entropy plus Dice, and the best epoch
by the monitored value (validation loss or IoU) kept. Validation chips are predicted window by window and stitched, as
at serving. ConvNeXt has no BatchNorm, so nothing has to stay in eval mode while training. The serving threshold is
chosen afterwards with a sweep on validation (`evaluate.py`).
"""

import logging
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
import torch.nn.functional as functional

from .dataset import SEAGRASS, LabelSource
from .export import build_model
from .model import encoder_stage_modules
from .params import ENCODER_STAGES, PreprocessParams, TrainParams
from .segmentation_dataset import EvalTiles, TrainTiles

log = logging.getLogger(__name__)

LOG_LINES_PER_EPOCH = 50  # step logging aims at this many lines per epoch, however large the dataset
DICE_SMOOTHING = 1.0  # keeps the Dice term defined (and zero) on a batch without seagrass


@dataclass(frozen=True)
class TrainResult:
    model: Any  # SeagrassUPerNet, best weights, CPU, eval mode
    history: dict[str, list[float]]
    best_epoch: int  # 0-based
    best_value: float  # the monitored value of the best epoch


# Called after each epoch with (epoch starting at 1, {"train_loss": ..., "val_loss": ..., "val_iou": ...}).
EpochCallback = Callable[[int, dict[str, float]], None]


def _resolve_device(requested: str) -> str:
    raw = requested.strip().lower()
    if raw == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if raw in {"auto", "cuda", "gpu"}:
        return "cuda"
    return f"cuda:{raw}" if raw.isdigit() else raw


def _ensure_visible_progress() -> None:
    """Make INFO progress visible when the host configured no logging (a bare container step); leave it alone if it did."""
    if not log.isEnabledFor(logging.INFO):
        log.setLevel(logging.INFO)
    if not logging.getLogger().handlers and not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
        log.propagate = False


def freeze(model: torch.nn.Module, *, freeze_encoder: bool, unfreeze_from: str = "none") -> None:
    """With `freeze_encoder`, freeze the encoder except the ConvNeXt stages from `unfreeze_from` on; the decoder trains."""
    if not freeze_encoder:
        return
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    if unfreeze_from != "none":
        stages = encoder_stage_modules(model)
        for name in ENCODER_STAGES[ENCODER_STAGES.index(unfreeze_from) :]:
            for param in stages[name].parameters():
                param.requires_grad_(True)


def encoder_lr_factors(model: torch.nn.Module, layer_decay: float) -> dict[int, float]:
    """Learning-rate factor of every encoder parameter (by `id`): stage i x `layer_decay ** (4 - i)`, stem ** 5."""
    stages = encoder_stage_modules(model)
    factors = {
        id(param): layer_decay**5 for param in model.encoder.parameters()
    }  # the stem and anything else
    for index, name in enumerate(ENCODER_STAGES):
        for param in stages[name].parameters():
            factors[id(param)] = layer_decay ** (len(ENCODER_STAGES) - index)
    return factors


def optimizer_groups(model: torch.nn.Module, cfg: TrainParams) -> list[dict[str, Any]]:
    """AdamW groups: the decoder at `learning_rate`; each encoder parameter at `learning_rate * encoder_lr_factor`
    times its layer-wise decay; norms and biases (1-D parameters) get no weight decay."""
    encoder = encoder_lr_factors(model, cfg.layer_decay)
    groups: dict[tuple[float, bool], list[torch.nn.Parameter]] = {}
    for param in model.parameters():
        if param.requires_grad:
            factor = cfg.encoder_lr_factor * encoder[id(param)] if id(param) in encoder else 1.0
            groups.setdefault((round(factor, 6), param.ndim <= 1), []).append(param)
    return [
        {
            "params": params,
            "lr": cfg.learning_rate * factor,
            "weight_decay": 0.0 if no_decay else cfg.weight_decay,
        }
        for (factor, no_decay), params in sorted(groups.items())
    ]


def lr_schedule(optimizer: Any, total_steps: int, warmup_steps: int, cosine: bool) -> Any:
    """Per-batch learning-rate factor: a linear warmup, then cosine decay to 0 at `total_steps` (or flat)."""
    if not cosine and warmup_steps <= 0:
        return None

    def factor(step: int) -> float:
        warm = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
        decay = 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)) if cosine else 1.0
        return warm * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def loss_parts(logits: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Binary cross-entropy and Dice loss of `(B, 1, H, W)` logits against a `(B, H, W)` 0/1 mask."""
    target = (mask == SEAGRASS).float()
    logits = logits[:, 0].float()
    bce = functional.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    overlap = (probability * target).sum()
    dice = 1.0 - (2.0 * overlap + DICE_SMOOTHING) / (probability.sum() + target.sum() + DICE_SMOOTHING)
    return {"bce": bce, "dice": dice}


def stitch_logits(logits: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """`(rows * cols, 1, w, w)` window logits -> `(1, 1, height, width)`, the tile they cover (padding cut)."""
    window = logits.shape[-1]
    rows, cols = -(-height // window), -(-width // window)
    full = logits[:, 0].reshape(rows, cols, window, window).permute(0, 2, 1, 3)
    return full.reshape(rows * window, cols * window)[:height, :width][None, None]


def _hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _autocast(device: str, enabled: bool):
    if not (enabled and device.startswith("cuda")):
        return torch.autocast("cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _train_epoch(
    model, loader, device, optimizer, scheduler, scaler, cfg: TrainParams, label: str, log_every: int
) -> float:
    model.train()
    losses: list[float] = []
    start, steps = time.time(), len(loader)
    every = log_every or max(1, steps // LOG_LINES_PER_EPOCH)
    for step, (images, masks) in enumerate(loader, 1):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with _autocast(device, cfg.amp):
            parts = loss_parts(model(images), masks)
            loss = parts["bce"] + cfg.dice_weight * parts["dice"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label} step {step}: non-finite loss; lower learning_rate or max_grad_norm")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if step % every == 0 or step == steps:
            spent = time.time() - start
            log.info(
                "%s step %d/%d loss(last %d)=%.4f mean=%.4f lr=%.2e %.2fs/step eta %s",
                label, step, steps, every, np.mean(losses[-every:]), np.mean(losses),
                optimizer.param_groups[-1]["lr"], spent / step, _hms(spent / step * (steps - step)),
            )  # fmt: skip
    return float(np.mean(losses))


@torch.no_grad()
def _evaluate_epoch(model, loader, device, cfg: TrainParams, threshold: float = 0.5) -> dict[str, float]:
    """Mean loss and the IoU / F1 of the seagrass class at `threshold` over every validation tile (stitched)."""
    model.eval()
    losses: list[float] = []
    tp = fp = fn = 0
    for windows, masks in loader:
        windows, masks = windows[0].to(device), masks.to(device)
        with _autocast(device, cfg.amp):
            logits = stitch_logits(model(windows).float(), *masks.shape[-2:])
        parts = loss_parts(logits, masks)
        losses.append(float((parts["bce"] + cfg.dice_weight * parts["dice"]).cpu()))
        predicted = torch.sigmoid(logits[:, 0]) >= threshold
        truth = masks == SEAGRASS
        tp += int((predicted & truth).sum())
        fp += int((predicted & ~truth).sum())
        fn += int((~predicted & truth).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"val_loss": float(np.mean(losses)) if losses else float("nan"), "val_iou": iou, "val_f1": f1}


def train(
    *,
    train_images_dir: str | Path,
    train_labels: str | Path | LabelSource,
    val_images_dir: str | Path | None,
    val_labels: str | Path | LabelSource | None,
    cfg: TrainParams,
    pre: PreprocessParams | None = None,
    checkpoint: str | Path | BinaryIO | None = None,
    pretrained: bool = True,
    train_names: Sequence[str] | None = None,
    val_names: Sequence[str] | None = None,
    log_every: int = 0,
    num_workers: int = 0,
    on_epoch_end: EpochCallback | None = None,
    epoch_checkpoint_dir: str | Path | None = None,
) -> TrainResult:
    """Fine-tune the seagrass UPerNet on chips and their labels GeoJSON; keep the best epoch.

    Start from ImageNet encoder weights (`pretrained`, downloaded by timm on first use) or continue a saved model
    (`checkpoint`, a `SeagrassUPerNet` state dict path or binary stream). `pre.model_gsd_m` must be the value used at
    serving time.

    Progress goes to the `ss_hot.train` logger at INFO (visible even if the host configured nothing): a start summary,
    about 50 step lines per epoch, and one line per epoch. `on_epoch_end(epoch, metrics)` lets a pipeline step forward
    the numbers to MLflow. Export the result with `ss_hot.export.export_onnx_bytes`.
    """
    _ensure_visible_progress()
    pre = pre or PreprocessParams()
    device = _resolve_device(cfg.device)
    if device == "cpu":
        log.warning("training on CPU: expect it to be slow; fAIr training is meant to run on a GPU")
    torch.manual_seed(cfg.seed)

    train_ds = TrainTiles(
        train_images_dir, train_labels, pre, names=train_names, augment=cfg.augment, seed=cfg.seed
    )
    val_loader = None
    if val_images_dir is not None and val_labels is not None:
        val_ds = EvalTiles(val_images_dir, val_labels, pre, names=val_names)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    else:
        log.warning("no validation set: the best epoch is chosen by training loss")

    model = build_model(
        pretrained=pretrained and checkpoint is None, weights=checkpoint, drop_path_rate=cfg.drop_path_rate
    ).to(device)
    if device.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)
    freeze(model, freeze_encoder=cfg.freeze_encoder, unfreeze_from=cfg.unfreeze_from)
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg), lr=cfg.learning_rate)
    steps_per_epoch = math.ceil(max(1, round(len(train_ds) * cfg.epoch_fraction)) / cfg.batch_size)
    scheduler = lr_schedule(
        optimizer, cfg.epochs * steps_per_epoch, cfg.warmup_steps, cfg.scheduler == "cosine"
    )
    use_scaler = cfg.amp and device.startswith("cuda") and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    log.info(
        "start: device=%s train_chips=%d val_chips=%s batch_size=%d steps/epoch=%d epochs=%d lr=%g "
        "weight_decay=%g warmup=%d encoder_lr_factor=%g layer_decay=%g freeze_encoder=%s unfreeze_from=%s "
        "drop_path=%g amp=%s monitor=%s params=%.1fM trainable=%.1fM model_gsd_m=%g augment=%s pretrained=%s "
        "checkpoint=%s",
        device, len(train_ds), len(val_loader.dataset) if val_loader else "none",
        cfg.batch_size, steps_per_epoch, cfg.epochs, cfg.learning_rate, cfg.weight_decay, cfg.warmup_steps,
        cfg.encoder_lr_factor, cfg.layer_decay, cfg.freeze_encoder, cfg.unfreeze_from, cfg.drop_path_rate, cfg.amp,
        cfg.monitor, sum(param.numel() for param in model.parameters()) / 1e6, trainable / 1e6, pre.model_gsd_m,
        cfg.augment, pretrained and checkpoint is None,
        checkpoint if checkpoint is None or isinstance(checkpoint, (str, Path)) else "<stream>",
    )  # fmt: skip

    monitor = cfg.monitor if val_loader is not None else "train_loss"
    higher_is_better = monitor == "val_iou"
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}
    best_state: dict[str, torch.Tensor] = {}
    best_epoch, best_value = 0, -math.inf if higher_is_better else math.inf
    patience_left = cfg.early_stop_patience
    started = time.time()
    for epoch in range(cfg.epochs):
        tag = f"epoch {epoch + 1}/{cfg.epochs}"
        epoch_started = time.time()
        train_ds.set_epoch(epoch)
        subset: Any = train_ds
        if cfg.epoch_fraction < 1.0:
            count = max(1, round(len(train_ds) * cfg.epoch_fraction))
            subset = torch.utils.data.Subset(
                train_ds, np.random.default_rng(cfg.seed + epoch).permutation(len(train_ds))[:count].tolist()
            )
        loader = torch.utils.data.DataLoader(
            subset, batch_size=cfg.batch_size, shuffle=True, drop_last=len(subset) > cfg.batch_size,
            generator=torch.Generator().manual_seed(cfg.seed + epoch), num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
        )  # fmt: skip
        metrics = {
            "train_loss": _train_epoch(
                model, loader, device, optimizer, scheduler, scaler, cfg, f"{tag} train", log_every
            )
        }
        if val_loader is not None:
            metrics.update(_evaluate_epoch(model, val_loader, device, cfg))
        for key in history:
            if key in metrics:
                history[key].append(metrics[key])
        if epoch_checkpoint_dir is not None:
            Path(epoch_checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                Path(epoch_checkpoint_dir) / f"epoch_{epoch + 1:02d}.pt",
            )
        value = metrics[monitor]
        improved = value > best_value if higher_is_better else value < best_value
        if improved:
            best_epoch, best_value, patience_left = epoch, value, cfg.early_stop_patience
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            patience_left -= 1
        vram = (
            f" peak_gpu_mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            if device.startswith("cuda")
            else ""
        )
        log.info(
            "%s done in %s: %s best %s=%.4f (epoch %d) %s patience_left=%d%s",
            tag, _hms(time.time() - epoch_started), " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
            monitor, best_value, best_epoch + 1, "NEW BEST" if improved else "no improvement", patience_left, vram,
        )  # fmt: skip
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, dict(metrics))
        if not improved and patience_left <= 0:
            log.info("early stopping after %d epochs", epoch + 1)
            break

    model.load_state_dict(best_state, strict=True)
    log.info(
        "finished in %s; keeping epoch %d (%s %.4f)",
        _hms(time.time() - started),
        best_epoch + 1,
        monitor,
        best_value,
    )
    return TrainResult(
        model=model.to("cpu").eval(), history=history, best_epoch=best_epoch, best_value=float(best_value),
    )  # fmt: skip
