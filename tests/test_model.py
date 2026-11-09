"""Unit tests for src.model."""

import os
import tempfile
import pytest
import tensorflow as tf
from src.model import build_model, load_model, save_model


def test_build_model_default_architecture():
    """Verify default model architecture, layer structure, and output shape."""
    model = build_model(input_shape=(150, 150, 3), num_classes=6)

    assert model.name == "GarbageCNN"
    assert model.input_shape == (None, 150, 150, 3)
    assert model.output_shape == (None, 6)

    # Check layer types and sequence
    layer_names = [l.name for l in model.layers]
    assert "conv1" in layer_names
    assert "pool1" in layer_names
    assert "conv2" in layer_names
    assert "pool2" in layer_names
    assert "conv3" in layer_names
    assert "pool3" in layer_names
    assert "flatten" in layer_names
    assert "dense_fc" in layer_names
    assert "output_probabilities" in layer_names

    # Check activation of output layer
    output_layer = model.get_layer("output_probabilities")
    assert output_layer.activation.__name__ == "softmax"


def test_build_model_with_custom_classes_and_dropout():
    """Verify model building with custom classes and dropout rate."""
    model = build_model(
        input_shape=(128, 128, 3),
        num_classes=10,
        dropout_rate=0.3,
        compile_model=False,
    )
    assert model.output_shape == (None, 10)
    layer_names = [l.name for l in model.layers]
    assert "dropout" in layer_names


def test_save_and_load_keras_format():
    """Verify saving and loading model in modern .keras format."""
    model = build_model(input_shape=(64, 64, 3), num_classes=6)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model.keras")
        saved_path = save_model(model, model_path)
        assert os.path.isfile(saved_path)

        loaded_model = load_model(saved_path, compile_model=False)
        assert loaded_model.input_shape == (None, 64, 64, 3)
        assert loaded_model.output_shape == (None, 6)


def test_load_nonexistent_model():
    """Verify FileNotFoundError when loading a nonexistent model file."""
    with pytest.raises(FileNotFoundError):
        load_model("/path/to/nonexistent_model_file.keras")
