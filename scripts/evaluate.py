"""
Evaluation script for model predictions.
"""

import json
import torch
import mlflow
from torch import nn

from src.visionforge.evaluation.error_analysis import plot_error_grid
from src.visionforge.evaluation.evaluator import *
from src.visionforge.data.datasets import CIFAR10DataModule
from src.visionforge.models.cnn import CNN
from src.visionforge.utils.checkpointing import load_checkpoint
from src.visionforge.utils.tracking import setup_mlflow, log_final_artifacts
from pathlib import Path
import yaml

from src.visionforge.training.optimizer import create_optimizer
from src.visionforge.utils.logging import setup_logger


if __name__ == "__main__":
    logger = setup_logger(__name__)
    config_path = Path(__file__).parent.parent / "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cifar10 = CIFAR10DataModule(config_path=config_path)
    cifar10.setup()
    test_loader = cifar10.test_dataloader()
    test_dataset = cifar10.test_dataset
    class_names = test_dataset.classes
    # class_names = test_dataset.

    model = CNN(num_classes=config["num_classes"]).to(device)    
    ckpt = load_checkpoint( Path(config["checkpoint_dir"]) / "best.pth", 
                           model,
                           device=device)
    
    run_id = ckpt.get("run_id", None)

    labels, preds, probs = collect_predictions(model, test_loader, device)
    metrics = evaluate_predictions(labels, preds)
    report = get_classification_report(labels, preds, target_names=class_names)

    plot_dir = Path(config["plot_path"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (plot_dir / "classification_report.txt").write_text(report)
    plot_confusion_matrix(labels, preds, target_names=class_names)
    plot_error_grid(test_dataset, labels, preds, probs, class_names=class_names, save_path=plot_dir / "error_grid.png",
                    mean=config["preprocessing"]["mean"], std=config["preprocessing"]["std"])
    
    mlflow_cfg = config["mlflow"]
    setup_mlflow(mlflow_cfg["tracking_uri"], mlflow_cfg["experiment_name"], run_id=run_id)
    try:
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifacts(str(plot_dir), artifact_path="evaluation")

    except BaseException:
        mlflow.end_run(status="FAILED")
        raise
    finally:
        mlflow.end_run()
    logger.info(f"Test accuracy: {metrics['accuracy_pct']:.2f}%")
    
   