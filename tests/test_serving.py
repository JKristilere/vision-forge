import io

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from src.visionforge.serving import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch, project_config, write_checkpoint):
    project_config["serving"]["checkpoint_path"] = str(write_checkpoint(project_config))
    project_config["serving"]["max_upload_bytes"] = 1024
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(project_config))
    monkeypatch.setattr(app_module, "CONFIG_PATH", config_path)

    with TestClient(app_module.app, raise_server_exceptions=False) as test_client:
        yield test_client


def _png_bytes(size=(32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 40, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model-info").json()
    assert info["architecture"] == "CNN"
    assert len(info["class_names"]) == 10


def test_predict_returns_top_k_with_confidences(client):
    response = client.post("/predict", files={"file": ("img.png", _png_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert len(body["top_k"]) == 3
    assert body["prediction"] == body["top_k"][0]
    confidences = [item["confidence"] for item in body["top_k"]]
    assert confidences == sorted(confidences, reverse=True)
    assert 0.0 < sum(confidences) <= 1.0 + 1e-6


def test_predict_rejects_undecodable_image(client):
    response = client.post("/predict", files={"file": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 400


def test_predict_rejects_oversized_upload(client):
    response = client.post("/predict", files={"file": ("big.png", _png_bytes((512, 512)), "image/png")})
    assert response.status_code == 413


def test_predict_rejects_non_image_content_type(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_predict_rejects_empty_upload(client):
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_endpoints_return_503_when_checkpoint_is_missing(tmp_path, monkeypatch, project_config):
    project_config["serving"]["checkpoint_path"] = str(tmp_path / "missing.pth")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(project_config))
    monkeypatch.setattr(app_module, "CONFIG_PATH", config_path)

    with TestClient(app_module.app, raise_server_exceptions=False) as client:
        assert client.get("/health").status_code == 503
        response = client.post("/predict", files={"file": ("img.png", _png_bytes(), "image/png")})
        assert response.status_code == 503
