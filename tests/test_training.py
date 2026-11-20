import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.visionforge.models.cnn import CNN
from src.visionforge.training.trainer import Trainer

pytestmark = pytest.mark.integration


def _random_loader(num_samples, batch_size=16):
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_one_epoch_trains_and_writes_checkpoints(tmp_path):
    model = CNN(channels=(8, 16), hidden=16, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        nn.CrossEntropyLoss(),
        optimizer,
        scheduler=None,
        train_dataloader=_random_loader(64),
        val_dataloader=_random_loader(32),
        checkpoint_dir=tmp_path,
    )

    history = trainer.train(num_epochs=1)

    assert len(history) == 1
    assert math.isfinite(history[0]["train_loss"])
    assert math.isfinite(history[0]["val_loss"])
    assert (tmp_path / "last.pth").exists()
    assert (tmp_path / "best.pth").exists()
