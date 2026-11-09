"""Garbage Classification Model package.

Modular architecture extracted from GarbageMLModel_CS549 notebooks.
"""

from src.dataset import (
    DEFAULT_CLASSES,
    download_kaggle_dataset,
    get_data_generators,
    load_classes,
    save_classes,
    split_dataset,
)
from src.model import build_model, load_model, save_model

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_CLASSES",
    "build_model",
    "download_kaggle_dataset",
    "get_data_generators",
    "load_classes",
    "load_model",
    "save_classes",
    "save_model",
    "split_dataset",
]
