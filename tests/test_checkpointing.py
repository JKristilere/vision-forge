import pytest
import torch
from torch import nn

from src.visionforge.utils.checkpointing import load_checkpoint, save_checkpoint


def _model():
    return nn.Linear(4, 2)


def _save(directory, is_best):
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(directory, model, optimizer, epoch=3, val_loss=0.4, best_val_loss=0.4, is_best=is_best)
    return model


def test_save_and_load_round_trip_restores_weights_and_metadata(tmp_path):
    source = _save(tmp_path, is_best=True)
    target = _model()

    checkpoint = load_checkpoint(tmp_path / "best.pth", target)

    assert torch.equal(target.weight, source.weight)
    assert checkpoint["epoch"] == 3
    assert checkpoint["val_loss"] == pytest.approx(0.4)


def test_non_best_save_writes_only_last_checkpoint(tmp_path):
    _save(tmp_path, is_best=False)

    assert (tmp_path / "last.pth").exists()
    assert not (tmp_path / "best.pth").exists()


def test_load_rejects_checkpoint_without_model_weights(tmp_path):
    path = tmp_path / "broken.pth"
    torch.save({"epoch": 1}, path)

    with pytest.raises(KeyError):
        load_checkpoint(path, _model())
