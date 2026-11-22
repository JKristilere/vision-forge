"""Training script for VisionForge CNN models for running experiments.
"""

import argparse
from pathlib import Path

import mlflow
import torch
import yaml
from src.visionforge.data.datasets import CIFAR10DataModule
from torch import nn

from src.visionforge.models.cnn import CNN
from src.visionforge.training.optimizer import create_optimizer, create_scheduler
from src.visionforge.training.trainer import Trainer
from src.visionforge.utils.checkpointing import load_checkpoint
from src.visionforge.utils.logging import setup_logger
from src.visionforge.utils.tracking import log_final_artifacts, setup_mlflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()

    logger = setup_logger(__name__)

    

    config_path = Path(__file__).parent.parent / "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    cifar10 = CIFAR10DataModule(config_path=config_path)
    cifar10.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = CNN(num_classes=config["num_classes"], 
                    channels=config["channels"], 
                    hidden=config["hidden"], 
                    dropout=config["dropout"]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(optimizer_name=config["optimizer_name"], 
                                 params=cnn_model.parameters(), lr=config["lr"], 
                                 weight_decay=config["weight_decay"])
    scheduler = create_scheduler(optimizer, scheduler_name=config["scheduler_name"], 
                                 **config["scheduler_params"])

    start_epoch = 0
    best_val_loss = float('inf')
    run_id = None
    if args.resume:
        # Model must be on the target device before loading optimizer state so the state is restored onto the same device.
        ckpt = load_checkpoint(
            Path(config["checkpoint_dir"]) / "last.pth",
            cnn_model, optimizer, 
            device=device,
        )
        start_epoch = ckpt["epoch"] + 1
        # Scheduler state is not checkpointed, so replay the elapsed epochs.
        for _ in range(start_epoch):
            scheduler.step()
        best_val_loss = ckpt["best_val_loss"]
        run_id = ckpt.get("mlflow_run_id")

    mlflow_cfg = config["mlflow"]
    setup_mlflow(mlflow_cfg["tracking_uri"], mlflow_cfg["experiment_name"], run_id=run_id)
    if run_id is None:
        mlflow.log_params({k: v for k, v in config.items() if k != "mlflow"})

    trainer = Trainer(cnn_model, criterion, optimizer, scheduler,
                       cifar10.train_dataloader(), 
                       cifar10.val_dataloader(), 
                       checkpoint_dir= config["checkpoint_dir"],
                       log_to_mlflow=True)


    try:
        trainer.train(num_epochs=config["num_epochs"], 
                      start_epoch=start_epoch, 
                      best_val_loss=best_val_loss)
        log_final_artifacts(cnn_model, config["checkpoint_dir"])
    except BaseException:
        mlflow.end_run(status="FAILED")
        raise
    mlflow.end_run()

