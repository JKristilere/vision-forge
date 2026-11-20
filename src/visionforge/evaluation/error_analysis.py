"""
Error analysis utilities for model predictions.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Figures are only saved to disk; avoid GUI backends (Tk) that may be unavailable.
import matplotlib.pyplot as plt
import torch


def plot_error_grid(test_dataset, labels, preds, probs, class_names, save_path, mean, std, n=36):
    """
    Plot a grid of misclassified images from the dataset.

    Args:
        test_dataset: The dataset containing the images.
        labels (torch.Tensor): The true labels.
        preds (torch.Tensor): The predicted labels.
        probs (torch.Tensor): The prediction probabilities.
        class_names (list[str]): The list of class names.
        save_path (str): The path to save the error grid image.
        n (int, optional): The number of images to display. Defaults to 36.

    Returns:
        None
    """
    # dataset must be the test set loaded with shuffle=False, so index i matches row i.
    wrong = torch.nonzero(labels != preds).view(-1)
    wrong_confidence = probs[wrong, preds[wrong]]
    order = torch.argsort(wrong_confidence, descending=True)
    picks = wrong[order][:n].tolist()

    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    fig, axes = plt.subplots(6, 6, figsize=(12, 13))
    for ax in axes.flat:
        ax.axis('off')

    for ax, i in zip(axes.flat, picks):
        image, _ = test_dataset[i]
        image  = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        true_label = labels[i].item()
        pred_label = preds[i].item()
        ax.imshow(image)
        ax.set_title(f"True: {class_names[true_label]}\n"
                     f"Pred: {class_names[pred_label]}\n"
                     f"Conf: {wrong_confidence[order][picks.index(i)]:.2f}",
                     fontsize=8
                     )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)