import io

import pytest
from PIL import Image

from src.visionforge.serving.predictor import InvalidImageError, Predictor


@pytest.fixture
def predictor(project_config, write_checkpoint):
    return Predictor(project_config, write_checkpoint(project_config))


def _encode(image, fmt="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_predict_bytes_accepts_grayscale_and_rgba_images(predictor):
    for mode in ("L", "RGBA"):
        result = predictor.predict_bytes(_encode(Image.new(mode, (40, 20))))

        assert len(result["top_k"]) == predictor.top_k
        assert 0.0 <= result["prediction"]["confidence"] <= 1.0


def test_predict_bytes_rejects_non_image_bytes(predictor):
    with pytest.raises(InvalidImageError):
        predictor.predict_bytes(b"definitely not an image")


def test_top_k_labels_match_their_class_indices(predictor, project_config):
    result = predictor.predict_bytes(_encode(Image.new("RGB", (32, 32))))

    for item in result["top_k"]:
        assert item["label"] == project_config["serving"]["class_names"][item["class_index"]]


def test_same_image_gives_same_prediction(predictor):
    data = _encode(Image.new("RGB", (32, 32), color=(10, 200, 30)))

    assert predictor.predict_bytes(data)["prediction"] == predictor.predict_bytes(data)["prediction"]
