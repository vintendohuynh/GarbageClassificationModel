"""Unit tests for src.predict."""

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from src.dataset import DEFAULT_CLASSES, save_classes
from src.model import build_model, save_model
from src.predict import GarbagePredictor


@pytest.fixture
def mock_model_and_image():
    """Creates a temporary model file and test image."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = build_model(input_shape=(64, 64, 3), num_classes=6)
        model_path = os.path.join(tmp_dir, "test_model.keras")
        save_model(model, model_path)

        classes_path = os.path.join(tmp_dir, "classes.json")
        save_classes({cls: i for i, cls in enumerate(DEFAULT_CLASSES)}, classes_path)

        img_path = os.path.join(tmp_dir, "sample.jpg")
        img = Image.new("RGB", (100, 100), color=(120, 80, 200))
        img.save(img_path)

        yield {
            "model_path": model_path,
            "classes_path": classes_path,
            "image_path": img_path,
        }


def test_preprocess_formats(mock_model_and_image):
    """Verify preprocessing works for path, PIL image, and numpy array."""
    predictor = GarbagePredictor(
        model_path=mock_model_and_image["model_path"],
        classes_path=mock_model_and_image["classes_path"],
        target_size=(64, 64),
    )

    # 1. From file path
    tensor1 = predictor.preprocess(mock_model_and_image["image_path"])
    assert tensor1.shape == (1, 64, 64, 3)
    assert tensor1.dtype == np.float32
    assert tensor1.min() >= 0.0 and tensor1.max() <= 1.0

    # 2. From PIL Image
    pil_img = Image.new("RGB", (200, 200), color=(50, 50, 50))
    tensor2 = predictor.preprocess(pil_img)
    assert tensor2.shape == (1, 64, 64, 3)

    # 3. From numpy array
    np_img = np.zeros((80, 80, 3), dtype=np.uint8)
    tensor3 = predictor.preprocess(np_img)
    assert tensor3.shape == (1, 64, 64, 3)


def test_predict_single_and_top_k(mock_model_and_image):
    """Verify single prediction output structure and top-k ordering."""
    predictor = GarbagePredictor(
        model_path=mock_model_and_image["model_path"],
        classes_path=mock_model_and_image["classes_path"],
        target_size=(64, 64),
    )

    results = predictor.predict(mock_model_and_image["image_path"], top_k=3)

    assert len(results) == 3
    for r in results:
        assert "class" in r
        assert "index" in r
        assert "confidence" in r
        assert r["class"] in DEFAULT_CLASSES
        assert 0.0 <= r["confidence"] <= 1.0

    # Ensure ranked in descending confidence order
    assert results[0]["confidence"] >= results[1]["confidence"]
    assert results[1]["confidence"] >= results[2]["confidence"]


def test_predict_batch(mock_model_and_image):
    """Verify batch prediction."""
    predictor = GarbagePredictor(
        model_path=mock_model_and_image["model_path"],
        classes_path=mock_model_and_image["classes_path"],
        target_size=(64, 64),
    )

    inputs = [mock_model_and_image["image_path"], mock_model_and_image["image_path"]]
    batch_res = predictor.predict_batch(inputs, top_k=2)

    assert len(batch_res) == 2
    assert len(batch_res[0]) == 2
    assert len(batch_res[1]) == 2


def test_predict_missing_image(mock_model_and_image):
    """Verify FileNotFoundError when input image path does not exist."""
    predictor = GarbagePredictor(
        model_path=mock_model_and_image["model_path"],
        target_size=(64, 64),
    )
    with pytest.raises(FileNotFoundError):
        predictor.predict("/path/to/missing_file.jpg")
