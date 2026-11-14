"""Garbage Classification Model package.

Modular architecture extracted from GarbageMLModel_CS549 notebooks.
"""

from src.benchmark import benchmark_model, count_parameters
from src.dataset import (
    DEFAULT_CLASSES,
    download_kaggle_dataset,
    get_data_generators,
    load_classes,
    save_classes,
    split_dataset,
)
from src.explain import (
    create_gradcam_comparison,
    explain_image,
    make_gradcam_heatmap,
    overlay_gradcam,
)
from src.model import (
    build_custom_cnn,
    build_model,
    build_transfer_model,
    load_model,
    save_model,
)

__version__ = "1.2.0"

__all__ = [
    "DEFAULT_CLASSES",
    "benchmark_model",
    "build_custom_cnn",
    "build_model",
    "build_transfer_model",
    "count_parameters",
    "create_gradcam_comparison",
    "download_kaggle_dataset",
    "explain_image",
    "get_data_generators",
    "load_classes",
    "load_model",
    "make_gradcam_heatmap",
    "overlay_gradcam",
    "save_classes",
    "save_model",
    "split_dataset",
]
