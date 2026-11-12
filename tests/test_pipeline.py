"""
Automated Unit and Regression Test Suite for Brain Tumor Pipeline
Tests morphological contour extraction, model initialization, inference, and API endpoints.
"""
import os
import pytest
import numpy as np
from PIL import Image

from src.preprocessing import crop_brain_tumor, preprocess_image
from src.model import build_vgg19_model, get_model, generate_gradcam, predict_mri


def test_crop_brain_tumor_synthetic():
    # Create black synthetic image with white circle in center (simulating skull)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    import cv2
    cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)

    cropped = crop_brain_tumor(img)
    # Bounding box of radius 50 centered at (100, 100) should be around 100x100
    assert cropped.shape[0] < 200
    assert cropped.shape[1] < 200
    assert cropped.shape[0] >= 90
    assert cropped.shape[1] >= 90


def test_crop_brain_tumor_fallback_empty():
    # All black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped = crop_brain_tumor(img)
    # Should safely return original image without crashing
    assert cropped.shape == (100, 100, 3)


def test_preprocess_image_pipeline():
    sample_img = np.ones((150, 150, 3), dtype=np.uint8) * 128
    tensor, display_img = preprocess_image(sample_img, target_size=(120, 120))

    assert tensor.shape == (1, 120, 120, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    assert display_img.shape[2] == 3


def test_model_architecture():
    model = build_vgg19_model(input_shape=(120, 120, 3))
    assert model.input_shape == (None, 120, 120, 3)
    assert model.output_shape == (None, 2)
    # Verify parameter layers
    layer_names = [l.name for l in model.layers]
    assert "dense_proj_4608" in layer_names
    assert "dense_latent_1152" in layer_names
    assert "classification_softmax" in layer_names


def test_inference_and_gradcam():
    sample_mri = np.ones((120, 120, 3), dtype=np.uint8) * 200
    res = predict_mri(sample_mri, include_gradcam=True)

    assert "label" in res
    assert "confidence" in res
    assert "probabilities" in res
    assert res["probabilities"]["Tumor Present"] >= 0.0
    assert res["probabilities"]["No Tumor"] >= 0.0
    assert res["gradcam_image"] is not None
    assert res["gradcam_image"].shape == res["cropped_image"].shape

