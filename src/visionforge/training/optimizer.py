"""
Optimizer utilities for VisionForge training module.
"""

import torch.optim as optim

def create_optimizer(optimizer_name, params=None, lr=0.01, weight_decay=0.0):
    if optimizer_name.lower() == "sgd":
        return optim.SGD(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        return optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(optimizer, scheduler_name, **kwargs):
    if scheduler_name.lower() == "step":
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")