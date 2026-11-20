"""
Evaluator for model performance on evaluation datasets.
"""
import torch
import matplotlib
matplotlib.use("Agg")  # Figures are only saved to disk; avoid GUI backends (Tk) that may be unavailable.
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                               classification_report)

def collect_predictions(model, dataloader, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect predictions and corresponding labels from the model on the given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader for the evaluation dataset.
        device (torch.device): The device to perform computation on.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the corresponding labels, the predictions, and the prediction probabilities.
    """
    model.eval()
    all_preds = []
    all_labels = []
    pred_probs_list = []
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            pred_probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
            pred_probs_list.append(pred_probs)
    return (torch.cat(all_labels).cpu(), torch.cat(all_preds).cpu(), torch.cat(pred_probs_list).cpu())


def evaluate_predictions(labels: torch.Tensor, preds: torch.Tensor) -> dict[str, float]:
    """
    Evaluate the predictions against the true labels using common classification metrics.

    Args:
        labels (torch.Tensor): The true labels.
        preds (torch.Tensor): The predicted labels.

    Returns:
        dict[str, float]: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    labels = labels.numpy()
    preds = preds.numpy()
    return {
        "accuracy_pct": accuracy_score(labels, preds) * 100,
        "precision_pct": precision_score(labels, preds, average="weighted") * 100,
        "recall_pct": recall_score(labels, preds, average="weighted") * 100,
        "f1_pct": f1_score(labels, preds, average="weighted") * 100
    }

def get_classification_report(labels: torch.Tensor, preds: torch.Tensor, target_names: list[str] | None = None) -> str:
    """
    Generate a detailed classification report for the predictions against the true labels.

    Args:
        labels (torch.Tensor): The true labels.
        preds (torch.Tensor): The predicted labels.
        target_names (list[str] | None, optional): The list of class names. Defaults to None.

    Returns:
        str: A string containing the classification report.
    """
    labels = labels.numpy()
    preds = preds.numpy()
    return classification_report(labels, preds, target_names=target_names)

def plot_confusion_matrix(labels: torch.Tensor, preds: torch.Tensor, target_names: list[str], file_name: str = "confusion_matrix.png") -> None:
    """
    Plot the confusion matrix for the predictions against the true labels.

    Args:
        labels (torch.Tensor): The true labels.
        preds (torch.Tensor): The predicted labels.
        target_names (list[str]): The list of class names.
        file_name (str, optional): The name of the confusion matrix image file. Defaults to "confusion_matrix.png".

    Returns:
        np.ndarray: The confusion matrix as a NumPy array.

    """
    path_file = Path("artifacts/evaluation") / file_name
    path_file.parent.mkdir(parents=True, exist_ok=True)

    labels = labels.numpy()
    preds = preds.numpy()
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.savefig(path_file, bbox_inches="tight")
    plt.close(fig)
    return cm