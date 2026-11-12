"""
FastAPI Microservice Integration Tests
Tests /health, /api/info, and /api/predict endpoints.
"""
import os
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from src.api import api_app


@pytest.fixture
def client():
    return TestClient(api_app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "weights_loaded" in data
    assert "uptime_seconds" in data


def test_model_info_endpoint(client):
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert "VGG19" in data["architecture"]
    assert data["num_classes"] == 2
    assert "120x120" in data["input_resolution"]


def test_predict_single_endpoint(client):
    # Create synthetic test image
    img = Image.fromarray(np.uint8(np.random.randint(0, 255, (150, 150, 3))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict",
        files={"file": ("test_mri.jpg", buf, "image/jpeg")},
        params={"include_gradcam": "true"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_mri.jpg"
    assert "label" in data
    assert "is_tumor" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert "gradcam_base64" in data
    assert data["gradcam_base64"].startswith("data:image/png;base64,")


def test_predict_invalid_filetype(client):
    fake_txt = io.BytesIO(b"not an image file")
    response = client.post(
        "/api/predict",
        files={"file": ("test.txt", fake_txt, "text/plain")}
    )
    assert response.status_code == 400
    assert "not a valid image format" in response.json()["detail"]

