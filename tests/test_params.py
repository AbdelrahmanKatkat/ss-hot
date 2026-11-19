import pytest

from ss_hot.params import (
    AugmentParams,
    PostprocessParams,
    PreprocessParams,
    TrainParams,
    parse_augment_params,
    parse_postprocess_params,
    parse_preprocess_params,
    parse_train_params,
)


def test_none_and_empty_give_the_defaults():
    assert parse_preprocess_params(None) == PreprocessParams()
    assert parse_postprocess_params({}) == PostprocessParams()
    assert parse_augment_params(None) == AugmentParams()
    assert parse_train_params(None) == TrainParams()


def test_aliases_and_overrides():
    assert parse_postprocess_params({"confidence_threshold": 0.7}).threshold == 0.7
    assert parse_postprocess_params({"threshold": 0.4, "confidence_threshold": 0.7}).threshold == 0.4
    cfg = parse_train_params(
        {"lr": "3e-4", "freeze_backbone": "yes", "unfreeze_from": "stage3", "batch_size": None}
    )
    assert cfg.learning_rate == 3e-4 and cfg.freeze_encoder and cfg.unfreeze_from == "stage3"
    assert cfg.batch_size == TrainParams.batch_size  # None keeps the default
    assert parse_preprocess_params({"model_gsd_m": "0.05"}).model_gsd_m == 0.05


def test_train_params_carry_the_augmentation():
    cfg = parse_train_params({"orientation": "mirror", "water_tint": 0.1, "lowres_prob": 0})
    assert cfg.augment == AugmentParams(
        orientation="mirror", photometric=0.2, stretch=0.7, water_tint=0.1, scale_jitter=0.25,
        degrade_prob=0.3, lowres_prob=0.0, lowres_max=5.0,
    )  # fmt: skip
    assert AugmentParams() == parse_augment_params({})  # the plain AugmentParams changes nothing


@pytest.mark.parametrize(
    ("parser", "given"),
    [
        (parse_postprocess_params, {"threshold": 1.5}),
        (parse_postprocess_params, {"min_area_px": -1}),
        (parse_preprocess_params, {"model_gsd_m": 0.0}),
        (parse_augment_params, {"orientation": "sideways"}),
        (parse_augment_params, {"lowres_max": 0.5}),
        (parse_augment_params, {"stretch": 2.0}),
        (parse_train_params, {"layer_decay": 0}),
        (parse_train_params, {"epochs": 0}),
        (parse_train_params, {"monitor": "accuracy"}),
        (parse_train_params, {"unfreeze_from": "stage2"}),  # needs freeze_encoder
        (parse_train_params, {"amp": "maybe"}),
        (parse_train_params, {"batch_size": True}),
    ],
)
def test_invalid_values_raise(parser, given):
    with pytest.raises(ValueError):
        parser(given)
