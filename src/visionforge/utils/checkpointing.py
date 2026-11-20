"""Utility module for checkpointing models.

Provides functions to save and load model checkpoints, including best model checkpointing.
"""

from pathlib import Path

import torch


def save_checkpoint(directory, model, optimizer, epoch, val_loss, best_val_loss, is_best, extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        **(extra or {}),
    }
    torch.save(state, directory / "last.pth")
    if is_best:
        torch.save(state, directory / "best.pth")

def load_checkpoint(path, model, optimizer=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint