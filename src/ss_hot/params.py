"""Constants, defaults and dict -> dataclass parsing for `ss-hot`.

Parsers tolerate `None` and missing keys and apply the defaults below. The encoder is a ConvNeXt-Tiny trained on
ImageNet, so the normalisation is the ImageNet one; it runs inside the exported ONNX graph.
"""

from dataclasses import dataclass
from typing import Any, Mapping

# --- encoder constants (not stored in the checkpoint) ---
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# --- scale policy ---
# Ground size of one model-input pixel. Every chip, in training and at serving, is resized to this resolution
# first, so the model always sees seagrass at one scale (see docs/research.md, preprocessing).
MODEL_GSD_M = 0.03
# A chip that would need more upscaling than this is refused: zoom 20 (x 4.9 near the equator) is accepted, zoom 19
# (x 9.8) is not (docs/preprocessing.md). The training `lowres_max` covers up to this factor.
MAX_SCALE = 5.0
# The longest side of a scaled chip never exceeds this many pixels; a coarser zoom is upscaled less instead.
MAX_INPUT_PX = 2048

# --- the exported graph ---
# The ONNX graph takes windows of this fixed size (the pyramid pooling of UPerNet needs a static size to export).
# A scaled chip is padded to a whole number of windows and its probability is stitched back (`preprocess.py`).
WINDOW_PX = 512

# How a training chip is turned before it reaches the model (see `AugmentParams`).
ORIENTATIONS = ("dihedral", "mirror", "none")

# ConvNeXt stages of the encoder, shallow to deep; `unfreeze_from` picks the first one that trains.
ENCODER_STAGES = ("stage0", "stage1", "stage2", "stage3")


@dataclass(frozen=True)
class PreprocessParams:
    """How a chip is scaled before the model. Training and serving must use the same values."""

    model_gsd_m: float = MODEL_GSD_M  # ground size of one model-input pixel, metres
    max_input_px: int = MAX_INPUT_PX  # longest side of the scaled chip; a coarser zoom is upscaled less


@dataclass(frozen=True)
class AugmentParams:
    """What changes a training chip (and its mask) before the model; validation and test chips are never changed.

    Every step is off until asked for, the orientation included.
    """

    orientation: str = "none"  # "dihedral": quarter turn + mirror; "mirror": mirror only; "none": unchanged
    photometric: float = 0.0  # brightness, contrast (x 1 +- this) and gamma (exp(+- this)); 0 = off
    stretch: float = 0.0  # contrast and saturation each x exp(+- this) (0.7: x 0.5 to x 2); 0 = off
    water_tint: float = (
        0.0  # blend toward a blue-green water colour (red fades most), weight up to this; 0 = off
    )
    scale_jitter: float = 0.0  # random resize by 1 +- this, then crop or pad back to the same size; 0 = off
    degrade_prob: float = 0.0  # chance of a Gaussian blur or a JPEG round trip
    lowres_prob: float = 0.0  # chance of a resample down by up to `lowres_max` and back up (coarser imagery)
    lowres_max: float = 5.0  # largest down-and-up factor: about the coarsest serving resolution / model_gsd_m


# The augmentation keys, in `AugmentParams` order (shared by `TrainParams` and the parsers).
AUGMENT_KEYS = tuple(AugmentParams.__dataclass_fields__)


@dataclass(frozen=True)
class PostprocessParams:
    """How the seagrass probability becomes polygons (outside the ONNX graph, so all of it stays tunable)."""

    # Defaults from the OSM label statistics (docs/postprocessing.md): at 3 cm per model pixel, polygons under
    # 0.25 m2 are 16% of the mapped polygons but 0.7% of the mapped area.
    threshold: float = 0.5  # a pixel is seagrass from this probability on (re-tune per model with a sweep)
    min_area_px: int = 278  # polygons smaller than this many model pixels are dropped (0.25 m2)
    simplify_px: float = 2.0  # Douglas-Peucker tolerance in model pixels (6 cm; 0 keeps every pixel edge)
    smooth_sigma_px: float = 0.0  # Gaussian smoothing of the probability before the threshold (0 = off)
    open_px: int = 1  # square opening radius: removes specks and bridges narrower than 3 px (9 cm)
    close_px: int = 2  # square closing radius: fills gaps narrower than 5 px (15 cm)
    fill_holes_px: int = 278  # interior holes smaller than this many model pixels are filled (0.25 m2)


@dataclass(frozen=True)
class TrainParams:
    """Fine-tuning knobs for UPerNet + ConvNeXt-Tiny (see docs/research.md, model and augmentation).

    The whole network trains; the encoder at `encoder_lr_factor` of the decoder's learning rate. With
    `freeze_encoder` only the stages from `unfreeze_from` on train in the encoder.
    """

    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0  # 0 disables clipping
    scheduler: str = "cosine"  # "cosine" | "none"; cosine decays to 0 over all steps (stepped per batch)
    warmup_steps: int = 400  # linear learning-rate ramp over the first batches, about one epoch (0 = none)
    freeze_encoder: bool = False
    unfreeze_from: str = "none"  # with freeze_encoder: still train ConvNeXt stages from this one on
    encoder_lr_factor: float = (
        1.0  # learning-rate multiplier for the whole encoder (1.0 = same as the decoder)
    )
    layer_decay: float = (
        0.9  # encoder stage i trains at x layer_decay ** (4 - i), the stem at ** 5 (1.0 = off)
    )
    drop_path_rate: float = 0.1  # stochastic depth in the ConvNeXt blocks during training
    early_stop_patience: int = 8  # epochs without a better monitored value before stopping
    monitor: str = (
        "val_iou"  # "val_iou" (higher is better) or "val_loss"; train loss without a validation set
    )
    epoch_fraction: float = 1.0  # each epoch trains on a fresh random fraction of the chips (1.0 = all)
    dice_weight: float = (
        1.0  # weight of the Dice loss next to the binary cross-entropy (0 = cross-entropy only)
    )
    amp: bool = True  # mixed precision on CUDA (ignored on CPU)
    # The seagrass augmentation preset (docs/augmentation.md); every step can be switched off with 0 / "none".
    orientation: str = "dihedral"
    photometric: float = 0.2
    stretch: float = 0.7
    water_tint: float = 0.3
    scale_jitter: float = 0.25
    degrade_prob: float = 0.3
    lowres_prob: float = 0.5
    lowres_max: float = 5.0
    device: str = "auto"  # cuda if visible, else cpu (training is meant to run on a GPU)
    seed: int = 42

    @property
    def augment(self) -> AugmentParams:
        return AugmentParams(**{key: getattr(self, key) for key in AUGMENT_KEYS})


def _get(mapping: Mapping[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    """First present non-None key wins (canonical key first, then aliases)."""
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return default


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("bool is not a valid int here")
    return int(value)


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("bool is not a valid float here")
    return float(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected bool-like value, got {value!r}")


def _check(checks: tuple[tuple[str, bool, Any], ...]) -> None:
    for name, ok, value in checks:
        if not ok:
            raise ValueError(f"{name} out of range, got {value}")


def parse_preprocess_params(params: Mapping[str, Any] | None) -> PreprocessParams:
    """`model_gsd_m`: metres per model-input pixel; `max_input_px`: cap on the longest side of the scaled chip."""
    given = params or {}
    model_gsd_m = _as_float(_get(given, ("model_gsd_m",), PreprocessParams.model_gsd_m))
    max_input_px = _as_int(_get(given, ("max_input_px",), PreprocessParams.max_input_px))
    _check((
        ("model_gsd_m", 0.002 <= model_gsd_m <= 1.0, model_gsd_m),
        ("max_input_px", 64 <= max_input_px <= 8192, max_input_px),
    ))  # fmt: skip
    return PreprocessParams(model_gsd_m=model_gsd_m, max_input_px=max_input_px)


def parse_postprocess_params(params: Mapping[str, Any] | None) -> PostprocessParams:
    """`threshold` is canonical; `confidence_threshold` (the name of the other fAIr models) is an alias."""
    given = params or {}
    threshold = _as_float(_get(given, ("threshold", "confidence_threshold"), PostprocessParams.threshold))
    min_area_px = _as_int(_get(given, ("min_area_px",), PostprocessParams.min_area_px))
    simplify_px = _as_float(_get(given, ("simplify_px",), PostprocessParams.simplify_px))
    smooth_sigma_px = _as_float(_get(given, ("smooth_sigma_px",), PostprocessParams.smooth_sigma_px))
    open_px = _as_int(_get(given, ("open_px",), PostprocessParams.open_px))
    close_px = _as_int(_get(given, ("close_px",), PostprocessParams.close_px))
    fill_holes_px = _as_int(_get(given, ("fill_holes_px",), PostprocessParams.fill_holes_px))
    _check((
        ("threshold", 0.0 <= threshold <= 1.0, threshold),
        ("min_area_px", min_area_px >= 0, min_area_px),
        ("simplify_px", simplify_px >= 0, simplify_px),
        ("smooth_sigma_px", smooth_sigma_px >= 0, smooth_sigma_px),
        ("open_px", open_px >= 0, open_px),
        ("close_px", close_px >= 0, close_px),
        ("fill_holes_px", fill_holes_px >= 0, fill_holes_px),
    ))  # fmt: skip
    return PostprocessParams(
        threshold=threshold, min_area_px=min_area_px, simplify_px=simplify_px, smooth_sigma_px=smooth_sigma_px,
        open_px=open_px, close_px=close_px, fill_holes_px=fill_holes_px,
    )  # fmt: skip


def parse_augment_params(params: Mapping[str, Any] | None, defaults: Any = None) -> AugmentParams:
    """The augmentation keys of a flat dict; `defaults` (an `AugmentParams` or `TrainParams`) fills the rest."""
    given, base = params or {}, defaults or AugmentParams()
    orientation = str(_get(given, ("orientation",), base.orientation)).strip().lower()
    values = {key: _as_float(_get(given, (key,), getattr(base, key))) for key in AUGMENT_KEYS[1:]}
    _check((
        ("orientation", orientation in ORIENTATIONS, orientation),
        ("photometric", 0.0 <= values["photometric"] < 1.0, values["photometric"]),
        ("stretch", 0.0 <= values["stretch"] <= 1.5, values["stretch"]),
        ("water_tint", 0.0 <= values["water_tint"] < 1.0, values["water_tint"]),
        ("scale_jitter", 0.0 <= values["scale_jitter"] < 1.0, values["scale_jitter"]),
        ("degrade_prob", 0.0 <= values["degrade_prob"] <= 1.0, values["degrade_prob"]),
        ("lowres_prob", 0.0 <= values["lowres_prob"] <= 1.0, values["lowres_prob"]),
        ("lowres_max", 1.0 <= values["lowres_max"] <= 8.0, values["lowres_max"]),
    ))  # fmt: skip
    return AugmentParams(orientation=orientation, **values)


def parse_train_params(params: Mapping[str, Any] | None) -> TrainParams:
    """Training knobs from a flat dict (fAIr strips the `training.` prefix)."""
    given = params or {}
    defaults = TrainParams()
    epochs = _as_int(_get(given, ("epochs",), defaults.epochs))
    batch_size = _as_int(_get(given, ("batch_size",), defaults.batch_size))
    learning_rate = _as_float(_get(given, ("learning_rate", "lr"), defaults.learning_rate))
    weight_decay = _as_float(_get(given, ("weight_decay",), defaults.weight_decay))
    max_grad_norm = _as_float(_get(given, ("max_grad_norm",), defaults.max_grad_norm))
    scheduler = str(_get(given, ("scheduler",), defaults.scheduler)).strip().lower()
    warmup_steps = _as_int(_get(given, ("warmup_steps",), defaults.warmup_steps))
    freeze_encoder = _as_bool(_get(given, ("freeze_encoder", "freeze_backbone"), defaults.freeze_encoder))
    unfreeze_from = str(_get(given, ("unfreeze_from",), defaults.unfreeze_from)).strip().lower()
    encoder_lr_factor = _as_float(_get(given, ("encoder_lr_factor",), defaults.encoder_lr_factor))
    layer_decay = _as_float(_get(given, ("layer_decay",), defaults.layer_decay))
    drop_path_rate = _as_float(_get(given, ("drop_path_rate",), defaults.drop_path_rate))
    patience = _as_int(_get(given, ("early_stop_patience",), defaults.early_stop_patience))
    monitor = str(_get(given, ("monitor",), defaults.monitor)).strip().lower()
    epoch_fraction = _as_float(_get(given, ("epoch_fraction",), defaults.epoch_fraction))
    dice_weight = _as_float(_get(given, ("dice_weight",), defaults.dice_weight))
    amp = _as_bool(_get(given, ("amp",), defaults.amp))
    augment = parse_augment_params(given, defaults)
    _check((
        ("epochs", epochs >= 1, epochs),
        ("batch_size", batch_size >= 1, batch_size),
        ("learning_rate", learning_rate > 0, learning_rate),
        ("weight_decay", weight_decay >= 0, weight_decay),
        ("max_grad_norm", max_grad_norm >= 0, max_grad_norm),
        ("scheduler", scheduler in ("cosine", "none"), scheduler),
        ("warmup_steps", warmup_steps >= 0, warmup_steps),
        ("unfreeze_from", unfreeze_from in ENCODER_STAGES + ("none",), unfreeze_from),
        ("encoder_lr_factor", 0.0 < encoder_lr_factor <= 1.0, encoder_lr_factor),
        ("layer_decay", 0.0 < layer_decay <= 1.0, layer_decay),
        ("drop_path_rate", 0.0 <= drop_path_rate < 1.0, drop_path_rate),
        ("early_stop_patience", patience >= 1, patience),
        ("monitor", monitor in ("val_loss", "val_iou"), monitor),
        ("epoch_fraction", 0.0 < epoch_fraction <= 1.0, epoch_fraction),
        ("dice_weight", dice_weight >= 0, dice_weight),
    ))  # fmt: skip
    if unfreeze_from != "none" and not freeze_encoder:
        raise ValueError("unfreeze_from needs freeze_encoder=True (otherwise the whole encoder trains)")
    return TrainParams(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
        max_grad_norm=max_grad_norm, scheduler=scheduler, warmup_steps=warmup_steps, freeze_encoder=freeze_encoder,
        unfreeze_from=unfreeze_from, encoder_lr_factor=encoder_lr_factor, layer_decay=layer_decay, drop_path_rate=drop_path_rate,
        early_stop_patience=patience, monitor=monitor, epoch_fraction=epoch_fraction, dice_weight=dice_weight,
        amp=amp, **{key: getattr(augment, key) for key in AUGMENT_KEYS},
        device=str(_get(given, ("device",), defaults.device)), seed=_as_int(_get(given, ("seed",), defaults.seed)),
    )  # fmt: skip
