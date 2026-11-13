"""Unit tests for Grad-CAM explainability in src.explain."""

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from src.dataset import DEFAULT_CLASSES, save_classes
from src.explain import (
    create_gradcam_comparison,
    make_gradcam_heatmap,
    overlay_gradcam,
    parse_args,
)
from src.model import build_model, save_model


@pytest.fixture
def sample_model_and_tensor():
    """Builds a test model and sample input tensor."""
    model = build_model(input_shape=(64, 64, 3), num_classes=6)
    tensor = np.random.uniform(0.0, 1.0, size=(1, 64, 64, 3)).astype(np.float32)
    return model, tensor


def test_make_gradcam_heatmap(sample_model_and_tensor):
    """Verify Grad-CAM produces a valid 2D normalized heatmap."""
    model, tensor = sample_model_and_tensor
    heatmap = make_gradcam_heatmap(
        img_array=tensor,
        model=model,
        last_conv_layer_name="conv3",
    )

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_overlay_gradcam():
    """Verify overlay blending produces valid RGB PIL Image."""
    orig_img = Image.new("RGB", (120, 120), color=(100, 150, 200))
    dummy_heatmap = np.random.uniform(0.0, 1.0, size=(8, 8)).astype(np.float32)

    overlay = overlay_gradcam(orig_img, dummy_heatmap, alpha=0.5)

    assert isinstance(overlay, Image.Image)
    assert overlay.mode == "RGB"
    assert overlay.size == (120, 120)


def test_create_gradcam_comparison():
    """Verify 3-panel comparison image generation."""
    orig_img = Image.new("RGB", (100, 100), color=(60, 120, 180))
    dummy_heatmap = np.zeros((8, 8), dtype=np.float32)
    dummy_heatmap[3:5, 3:5] = 1.0
    overlay = overlay_gradcam(orig_img, dummy_heatmap, alpha=0.4)

    comp = create_gradcam_comparison(
        image_input=orig_img,
        heatmap=dummy_heatmap,
        superimposed_img=overlay,
        class_label="plastic",
        confidence=0.88,
    )

    assert isinstance(comp, Image.Image)
    assert comp.mode == "RGB"
    assert comp.size[0] > 600  # Wide enough for 3 panels


def test_parse_args_explain():
    """Verify CLI argument defaults and parsing for src.explain."""
    args = parse_args(["--image", "sample.jpg", "--output", "custom_hm.png"])
    assert args.image == "sample.jpg"
    assert args.output == "custom_hm.png"
    assert args.layer == "conv3"
    assert args.alpha == 0.4
    assert args.comparison is True
