"""`ss-hot`: HOT fAIr seagrass semantic segmentation helpers (UPerNet on a ConvNeXt-Tiny encoder)."""

from .augment import augment_chip_and_mask
from .dataset import BACKGROUND, SEAGRASS, LabelSource, scale_pair
from .decode import DecodeParams, decode_polygons
from .params import (
    AugmentParams,
    PostprocessParams,
    PreprocessParams,
    TrainParams,
    parse_augment_params,
    parse_postprocess_params,
    parse_preprocess_params,
    parse_train_params,
)
from .postprocess import decode_params, polygons_to_feature_collection, postprocess_outputs
from .preprocess import PreprocessMeta, preprocess_chip, preprocess_rgb, stitch_windows
from .serve import predict_chip, predict_session
from .tiles import chip_scale, tile_gsd_m

__all__ = [
    # parameters
    "PreprocessParams",
    "PostprocessParams",
    "AugmentParams",
    "TrainParams",
    "parse_preprocess_params",
    "parse_postprocess_params",
    "parse_augment_params",
    "parse_train_params",
    # tiles and dataset
    "tile_gsd_m",
    "chip_scale",
    "LabelSource",
    "scale_pair",
    "BACKGROUND",
    "SEAGRASS",
    # training-time augmentation (every step is off by default)
    "augment_chip_and_mask",
    # preprocess -> decode -> postprocess
    "PreprocessMeta",
    "preprocess_rgb",
    "preprocess_chip",
    "stitch_windows",
    "DecodeParams",
    "decode_params",
    "decode_polygons",
    "polygons_to_feature_collection",
    "postprocess_outputs",
    # inference: `predict_session` stops at the probability; `postprocess_outputs` decodes and georeferences
    "predict_chip",
    "predict_session",
]
# Optional (train extra, import explicitly): ss_hot.model, ss_hot.export, ss_hot.segmentation_dataset, ss_hot.train,
# ss_hot.evaluate.score_model
