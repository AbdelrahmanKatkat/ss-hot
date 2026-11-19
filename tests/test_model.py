"""The torch side: model shape, freezing, ONNX export parity and a two-epoch smoke run (needs the train extra).

The model is built without pretrained weights, so nothing is downloaded.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from ss_hot.export import build_model, export_onnx_bytes  # noqa: E402
from ss_hot.model import encoder_stage_modules  # noqa: E402
from ss_hot.params import WINDOW_PX, TrainParams  # noqa: E402
from ss_hot.train import freeze, loss_parts, optimizer_groups, stitch_logits  # noqa: E402


def test_logits_have_the_input_size_and_the_stages_are_named(tmp_path):
    model = build_model()
    with torch.no_grad():
        logits = model(torch.rand(2, 3, 128, 160))
    assert logits.shape == (2, 1, 128, 160)
    assert sorted(encoder_stage_modules(model)) == ["stage0", "stage1", "stage2", "stage3"]
    torch.save(model.state_dict(), tmp_path / "m.pt")
    reloaded = build_model(weights=tmp_path / "m.pt")
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values()))


def test_freezing_keeps_only_the_later_stages_and_the_decoder_trainable():
    model = build_model()
    freeze(model, freeze_encoder=True, unfreeze_from="stage3")
    stages = encoder_stage_modules(model)
    assert not any(p.requires_grad for p in stages["stage2"].parameters())
    assert all(p.requires_grad for p in stages["stage3"].parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())
    groups = optimizer_groups(model, TrainParams(encoder_lr_factor=0.5, layer_decay=0.9))
    assert {group["weight_decay"] for group in groups} == {0.0, TrainParams.weight_decay}
    lr = TrainParams.learning_rate
    assert {round(group["lr"] / lr, 4) for group in groups} == {1.0, round(0.5 * 0.9, 4)}  # decoder, stage3


def test_loss_and_stitching():
    logits = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros((1, 4, 4), dtype=torch.uint8)
    assert loss_parts(logits, mask)["bce"].item() == pytest.approx(np.log(2), rel=1e-5)  # probability 0.5
    windows = torch.arange(4.0).view(4, 1, 1, 1).expand(4, 1, 2, 2)  # a 2 x 2 grid of 2 px windows
    stitched = stitch_logits(windows, 3, 4)
    assert stitched.shape == (1, 1, 3, 4) and stitched[0, 0, 0].tolist() == [0, 0, 1, 1]
    assert stitched[0, 0, 2].tolist() == [2, 2, 3, 3]


def test_onnx_matches_torch():
    ort = pytest.importorskip("onnxruntime")
    model = build_model()
    session = ort.InferenceSession(export_onnx_bytes(model), providers=["CPUExecutionProvider"])
    windows = np.random.default_rng(0).random((3, 3, WINDOW_PX, WINDOW_PX), dtype=np.float32)
    (onnx_probability,) = session.run(None, {"image": windows})
    with torch.no_grad():
        torch_probability = torch.sigmoid(model(torch.from_numpy(windows)))[:, 0].numpy()
    assert onnx_probability.shape == (3, WINDOW_PX, WINDOW_PX)
    assert np.abs(onnx_probability - torch_probability).max() < 1e-4


def test_a_short_training_run_finishes_and_reports_every_epoch(fair_dataset):
    from ss_hot.train import train

    images, masks, _ = fair_dataset
    seen = []
    result = train(
        train_images_dir=images, train_labels=masks, val_images_dir=images, val_labels=masks,
        cfg=TrainParams(epochs=2, batch_size=2, warmup_steps=1, device="cpu", monitor="val_iou", amp=False),
        pretrained=False, on_epoch_end=lambda epoch, metrics: seen.append((epoch, metrics)),
    )  # fmt: skip
    assert [epoch for epoch, _ in seen] == [1, 2]
    assert {"train_loss", "val_loss", "val_iou", "val_f1"} <= set(seen[0][1])
    assert len(result.history["train_loss"]) == 2 and not result.model.training
