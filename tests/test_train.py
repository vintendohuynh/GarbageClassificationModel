"""Unit tests for src.train argument parsing and resolution logic."""

import pytest
from src.train import parse_args, resolve_data_directories


def test_parse_args_defaults():
    """Verify default CLI options in train.py."""
    args = parse_args([])
    assert args.epochs == 50
    assert args.batch_size == 32
    assert args.img_height == 150
    assert args.img_width == 150
    assert args.learning_rate == 0.001
    assert args.split_ratio == 0.8
    assert args.model_type == "custom_cnn"
    assert args.fine_tune_layers == 0
    assert args.model_path is None
    assert args.classes_path == "classes.json"
    assert args.seed == 42


def test_parse_args_custom_values():
    """Verify custom CLI argument overrides."""
    custom_args = [
        "--data-dir", "/path/to/data",
        "--epochs", "20",
        "--batch-size", "16",
        "--img-height", "128",
        "--img-width", "128",
        "--lr", "0.0005",
        "--dropout", "0.25",
        "--model-path", "custom_model.keras",
        "--early-stopping", "5",
        "--seed", "123",
    ]
    args = parse_args(custom_args)
    assert args.data_dir == "/path/to/data"
    assert args.epochs == 20
    assert args.batch_size == 16
    assert args.img_height == 128
    assert args.img_width == 128
    assert args.learning_rate == 0.0005
    assert args.dropout == 0.25
    assert args.model_path == "custom_model.keras"
    assert args.early_stopping == 5
    assert args.seed == 123


def test_resolve_data_directories_missing_inputs():
    """Verify ValueError when no directories or download flag are provided."""
    args = parse_args([])
    with pytest.raises(ValueError):
        resolve_data_directories(args)
