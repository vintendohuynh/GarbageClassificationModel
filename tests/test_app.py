"""Unit tests for the Streamlit web application components in app.py."""

import numpy as np
import pytest
from PIL import Image
import tensorflow as tf

from app import (
    CATEGORY_INFO,
    get_model,
    preprocess_image,
    run_prediction,
)
from src.dataset import DEFAULT_CLASSES
from src.model import build_model


def test_category_info_contains_all_classes():
    """Verify CATEGORY_INFO dictionary covers all 6 default waste classes."""
    for cls in DEFAULT_CLASSES:
        assert cls in CATEGORY_INFO
        assert "icon" in CATEGORY_INFO[cls]
        assert "bin" in CATEGORY_INFO[cls]
        assert "tip" in CATEGORY_INFO[cls]
        assert "color" in CATEGORY_INFO[cls]


def test_preprocess_image():
    """Verify preprocessing from PIL image to normalized 4D tensor."""
    dummy_img = Image.new("RGB", (300, 200), color=(120, 160, 200))
    tensor = preprocess_image(dummy_img, target_size=(150, 150))

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 150, 150, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_run_prediction():
    """Verify run_prediction forward pass sorting and output structure."""
    model = build_model(input_shape=(64, 64, 3), num_classes=6, compile_model=False)
    dummy_tensor = np.zeros((1, 64, 64, 3), dtype=np.float32)

    results = run_prediction(model, dummy_tensor, DEFAULT_CLASSES)

    assert len(results) == 6
    # Confidences must be sorted in descending order
    confidences = [r["confidence"] for r in results]
    assert confidences == sorted(confidences, reverse=True)

    for r in results:
        assert "class" in r
        assert "index" in r
        assert "confidence" in r
        assert r["class"] in DEFAULT_CLASSES
        assert 0.0 <= r["confidence"] <= 1.0


def test_get_model_fallback():
    """Verify get_model returns a valid model even when model file does not exist."""
    model = get_model("nonexistent_model.keras")
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape == (None, 6)
