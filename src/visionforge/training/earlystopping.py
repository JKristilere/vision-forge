"""
Early stopping implementation for handling validation loss during training
"""

from pathlib import Path

import yaml

config_path = Path(__file__).parent.parent.parent.parent / "configs/default.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

class EarlyStopping:
    def __init__(self, patience=config["patience"], delta=config["delta"], verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Early stopping triggered.")
        else:
            self.best_loss = val_loss
            self.no_improvement_count = 0
