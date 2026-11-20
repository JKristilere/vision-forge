"""
Utility functions for MLflow tracking.
"""

import mlflow
import torch
from pathlib import Path

def setup_mlflow(tracking_uri: str, experiment_name: str, run_id: str = None) -> str:
    """
    Set up MLflow tracking with the specified tracking URI, experiment name, and optional run ID.

    Args:
        tracking_uri (str): The URI of the MLflow tracking server.
        experiment_name (str): The name of the MLflow experiment.
        run_id (str, optional): The ID of the MLflow run to resume. Defaults to None.

    Returns:
        str: The ID of the active run.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)
    return mlflow.start_run(run_id=run_id).info.run_id


def log_final_artifacts(model, checkpoint_dir):
    """Log the trained model and any saved checkpoints to the active run."""
    mlflow.pytorch.log_model(
        model,
        name="model",
        serialization_format="pickle",
        pip_requirements=[f"torch=={torch.__version__.split('+')[0]}"],
    )
    for name in ("best.pth", "last.pth"):
        path = Path(checkpoint_dir) / name
        if path.exists():
            mlflow.log_artifact(str(path))

