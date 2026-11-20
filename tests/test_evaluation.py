import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.visionforge.evaluation.evaluator import collect_predictions, evaluate_predictions


class EchoLogits(nn.Module):
    """Scales the one-hot input so the argmax always equals the true label."""

    def forward(self, x):
        return x * 10


def _oracle_loader(labels, batch_size=4):
    inputs = F.one_hot(labels, num_classes=10).float()
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size)


def test_collect_predictions_returns_labels_predictions_and_probabilities():
    labels = torch.tensor([0, 3, 7, 9, 2, 5])

    labels_out, preds, probs = collect_predictions(EchoLogits(), _oracle_loader(labels), torch.device("cpu"))

    assert torch.equal(labels_out, labels)
    assert torch.equal(preds, labels)
    assert probs.shape == (6, 10)
    assert torch.allclose(probs.sum(dim=1), torch.ones(6))


def test_evaluate_predictions_reports_accuracy_as_percentage():
    labels = torch.tensor([0, 1, 2, 2])
    preds = torch.tensor([0, 1, 1, 2])

    metrics = evaluate_predictions(labels, preds)

    assert metrics["accuracy_pct"] == pytest.approx(75.0)
