"""
Inference wrapper around the trained CNN.

The model is loaded once when the Predictor is constructed and reused for every request.
"""

import io
import time
from pathlib import Path

import torch
from PIL import Image
from src.visionforge.data.transforms import build_transforms

from src.visionforge.models.cnn import CNN
from src.visionforge.utils.checkpointing import load_checkpoint


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes cannot be decoded as an image."""


class Predictor:
    def __init__(self, config: dict, checkpoint_path: Path):
        serving = config["serving"]
        preprocessing = config["preprocessing"]

        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if serving["device"] == "auto" else serving["device"]
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.class_names = list(serving["class_names"])
        self.top_k = serving["top_k"]
        self.image_size = preprocessing["image_size"]
        self.mean = list(preprocessing["mean"])
        self.std = list(preprocessing["std"])
        _, self.transform, _ = build_transforms(self.mean, self.std, self.image_size)

        self.model = CNN(
            num_classes=len(self.class_names),
            channels=tuple(config["channels"]),
            hidden=config["hidden"],
            dropout=config["dropout"],
        ).to(self.device)
        self.checkpoint = load_checkpoint(self.checkpoint_path, self.model, device=self.device)
        self.model.eval()

    def predict_bytes(self, data: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:
            raise InvalidImageError("Uploaded file is not a readable image.") from exc

        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        batch = self.transform(image).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.inference_mode():
            probs = torch.softmax(self.model(batch), dim=1)[0]
        latency_ms = (time.perf_counter() - start) * 1000

        top_probs, top_idx = torch.topk(probs, self.top_k)
        top_k = [
            {"class_index": idx, "label": self.class_names[idx], "confidence": prob}
            for idx, prob in zip(top_idx.tolist(), top_probs.tolist())
        ]
        return {"prediction": top_k[0], "top_k": top_k, "latency_ms": latency_ms}

    def info(self) -> dict:
        return {
            "architecture": type(self.model).__name__,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_epoch": self.checkpoint.get("epoch"),
            "checkpoint_val_loss": self.checkpoint.get("val_loss"),
            "class_names": self.class_names,
            "image_size": self.image_size,
            "mean": self.mean,
            "std": self.std,
            "top_k": self.top_k,
        }
