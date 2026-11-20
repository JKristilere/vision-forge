import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.visionforge.data.transforms import build_transforms


class CIFAR10DataModule:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def setup(self):
        config = self.config
        preprocessing = config["preprocessing"]
        train_transform, val_transform, test_transform = build_transforms(
            preprocessing["mean"], preprocessing["std"], preprocessing["image_size"]
        )
        train_data = datasets.CIFAR10(config["root_dir"], train=True, download=True, transform=train_transform)
        val_data = datasets.CIFAR10(config["root_dir"], train=True, download=True, transform=val_transform)
        self.test_dataset = datasets.CIFAR10(
            config["root_dir"], train=False, download=True, transform=test_transform
        )

        indices = torch.randperm(
            len(train_data), generator=torch.Generator().manual_seed(config["random_seed"])
        ).tolist()
        train_size = config["train_size"]
        self.train_dataset = Subset(train_data, indices[:train_size])
        self.validation_dataset = Subset(val_data, indices[train_size:])

    def _dataloader(self, dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config["num_workers"],
            persistent_workers=self.config["persistent_workers"],
            pin_memory=self.config["pin_memory"],
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, self.config["batch_size"], shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.validation_dataset, self.config["eval_batch_size"], shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, self.config["eval_batch_size"], shuffle=False)
