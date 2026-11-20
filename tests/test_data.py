import torch
import yaml
from PIL import Image

from src.visionforge.data import datasets as datasets_module
from src.visionforge.data.datasets import CIFAR10DataModule
from src.visionforge.data.transforms import build_transforms


class FakeCIFAR10:
    """Replaces torchvision's CIFAR10 so tests never download the dataset."""

    def __init__(self, root, train=True, download=False, transform=None):
        self.length = 50000 if train else 10000
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = Image.new("RGB", (32, 32), color=(index % 256, 128, 64))
        if self.transform is not None:
            image = self.transform(image)
        return image, index % 10


def _config_path(tmp_path, project_config):
    config = dict(project_config, root_dir=str(tmp_path), num_workers=0, persistent_workers=False, pin_memory=False)
    path = tmp_path / "data_config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _setup_module(config_path, monkeypatch):
    monkeypatch.setattr(datasets_module.datasets, "CIFAR10", FakeCIFAR10)
    module = CIFAR10DataModule(config_path)
    module.setup()
    return module


def test_setup_splits_training_and_validation_without_overlap(tmp_path, project_config, monkeypatch):
    module = _setup_module(_config_path(tmp_path, project_config), monkeypatch)

    train_idx = set(module.train_dataset.indices)
    val_idx = set(module.validation_dataset.indices)
    assert len(train_idx) == project_config["train_size"]
    assert train_idx.isdisjoint(val_idx)
    assert len(train_idx | val_idx) == 50000


def test_split_is_reproducible_for_the_same_seed(tmp_path, project_config, monkeypatch):
    config_path = _config_path(tmp_path, project_config)
    first = _setup_module(config_path, monkeypatch)
    second = _setup_module(config_path, monkeypatch)

    assert first.train_dataset.indices == second.train_dataset.indices


def test_loaders_yield_batches_of_configured_sizes(tmp_path, project_config, monkeypatch):
    module = _setup_module(_config_path(tmp_path, project_config), monkeypatch)

    images, labels = next(iter(module.train_dataloader()))
    assert images.shape == (project_config["batch_size"], 3, 32, 32)
    assert labels.dtype == torch.long

    val_images, _ = next(iter(module.val_dataloader()))
    assert val_images.shape == (project_config["eval_batch_size"], 3, 32, 32)


def test_eval_transform_normalizes_with_configured_mean_and_std():
    mean, std = [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]
    _, eval_transform, _ = build_transforms(mean, std, image_size=32)

    tensor = eval_transform(Image.new("RGB", (32, 32), (255, 255, 255)))

    assert tensor.shape == (3, 32, 32)
    assert torch.allclose(tensor, torch.full_like(tensor, (1.0 - 0.5) / 0.25))
