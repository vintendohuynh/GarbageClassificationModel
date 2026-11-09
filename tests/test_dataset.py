"""Unit tests for src.dataset."""

import os
import tempfile
import pytest
from PIL import Image
from src.dataset import (
    DEFAULT_CLASSES,
    get_data_generators,
    load_classes,
    save_classes,
    split_dataset,
)


def _create_mock_image(filepath: str, size=(50, 50)):
    """Helper to create dummy RGB image for generator testing."""
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(filepath, format="JPEG")


def test_default_classes():
    """Verify standard 6 Kaggle waste classes."""
    assert len(DEFAULT_CLASSES) == 6
    expected = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    assert DEFAULT_CLASSES == expected


def test_split_dataset_logic():
    """Verify deterministic splitting into train and val directories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        source_dir = os.path.join(tmp_dir, "source")
        train_dir = os.path.join(tmp_dir, "train")
        val_dir = os.path.join(tmp_dir, "val")

        classes = ["cardboard", "glass", "paper"]
        files_per_class = 10

        for cls in classes:
            cls_path = os.path.join(source_dir, cls)
            os.makedirs(cls_path, exist_ok=True)
            for i in range(files_per_class):
                _create_mock_image(os.path.join(cls_path, f"img_{i}.jpg"))

        summary = split_dataset(
            source_dir=source_dir,
            train_dir=train_dir,
            val_dir=val_dir,
            split_ratio=0.8,
            seed=42,
        )

        assert len(summary) == 3
        for cls in classes:
            assert summary[cls]["train"] == 8
            assert summary[cls]["val"] == 2
            assert len(os.listdir(os.path.join(train_dir, cls))) == 8
            assert len(os.listdir(os.path.join(val_dir, cls))) == 2


def test_save_and_load_classes():
    """Verify serialization and deserialization of class indices."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        classes_path = os.path.join(tmp_dir, "classes.json")
        sample_indices = {"cardboard": 0, "glass": 1, "metal": 2}

        save_classes(sample_indices, classes_path)
        assert os.path.isfile(classes_path)

        loaded = load_classes(classes_path)
        assert loaded == ["cardboard", "glass", "metal"]


def test_load_classes_fallback():
    """Verify fallback to DEFAULT_CLASSES if file does not exist."""
    fallback = load_classes("/nonexistent/classes.json")
    assert fallback == DEFAULT_CLASSES


def test_get_data_generators():
    """Verify ImageDataGenerator creation on mock directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_dir = os.path.join(tmp_dir, "train")
        val_dir = os.path.join(tmp_dir, "val")

        for d in [train_dir, val_dir]:
            for cls in ["plastic", "trash"]:
                cls_p = os.path.join(d, cls)
                os.makedirs(cls_p, exist_ok=True)
                for i in range(3):
                    _create_mock_image(os.path.join(cls_p, f"img_{i}.jpg"))

        train_gen, val_gen = get_data_generators(
            train_dir=train_dir,
            val_dir=val_dir,
            target_size=(64, 64),
            batch_size=2,
            augment=True,
        )

        assert train_gen.num_classes == 2
        assert val_gen.num_classes == 2
        assert train_gen.samples == 6
        assert val_gen.samples == 6
