from pathlib import Path

import pytest
import torch
import yaml

from src.visionforge.models.cnn import CNN

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def project_config():
    with open(DEFAULT_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def write_checkpoint(tmp_path):
    """Save a randomly initialised CNN shaped by the config and return the checkpoint path."""

    def _write(config, name="best.pth"):
        model = CNN(
            num_classes=config["num_classes"],
            channels=tuple(config["channels"]),
            hidden=config["hidden"],
            dropout=config["dropout"],
        )
        path = tmp_path / name
        torch.save({"model_state_dict": model.state_dict(), "epoch": 1, "val_loss": 0.5}, path)
        return path

    return _write
