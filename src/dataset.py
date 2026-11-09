"""Data loading, preprocessing, dataset splitting, and augmentation generators.

Supports:
- Kagglehub dataset download for 'asdasdasasdas/garbage-classification'
- Deterministic train/validation dataset splitting (default 80/20)
- Keras ImageDataGenerator data augmentation pipeline matching the notebook
- Dynamic class label persistence and loading
"""

import json
import os
import random
import shutil
from typing import Dict, List, Optional, Tuple, Union
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 6 canonical classes from Kaggle's Garbage Classification dataset
DEFAULT_CLASSES: List[str] = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

DEFAULT_KAGGLE_DATASET = "asdasdasasdas/garbage-classification"


def download_kaggle_dataset(dataset_id: str = DEFAULT_KAGGLE_DATASET) -> str:
    """Downloads the garbage classification dataset from Kaggle via kagglehub.

    Args:
        dataset_id: Kaggle dataset identifier string.

    Returns:
        str: Local directory path containing the downloaded dataset.
    """
    try:
        import kagglehub
    except ImportError as e:
        raise ImportError(
            "kagglehub is required to download Kaggle datasets. "
            "Install it via `pip install kagglehub`."
        ) from e

    path = kagglehub.dataset_download(dataset_id)
    return str(path)


def split_dataset(
    source_dir: str,
    train_dir: str,
    val_dir: str,
    split_ratio: float = 0.8,
    seed: int = 42,
    overwrite: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Splits class folders from a source directory into train and validation sets.

    Args:
        source_dir: Root directory containing one subfolder per class.
        train_dir: Destination path for train split directory.
        val_dir: Destination path for validation split directory.
        split_ratio: Proportion of files to allocate to training (default: 0.8).
        seed: Random seed for deterministic shuffling.
        overwrite: If True, recreates train and val sets even if they already exist.
                   If False, returns existing distribution without re-copying.

    Returns:
        Dict[str, Dict[str, int]]: Summary dictionary mapping each class to file counts:
                                  {'cardboard': {'train': 322, 'val': 81}, ...}
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    # Check if train and val are already populated
    if not overwrite and os.path.exists(train_dir) and os.path.exists(val_dir):
        existing_train_classes = [
            d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
        ]
        if existing_train_classes:
            summary = {}
            for class_name in sorted(existing_train_classes):
                t_count = len(os.listdir(os.path.join(train_dir, class_name)))
                v_path = os.path.join(val_dir, class_name)
                v_count = len(os.listdir(v_path)) if os.path.exists(v_path) else 0
                summary[class_name] = {"train": t_count, "val": v_count}
            return summary

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    rng = random.Random(seed)
    summary = {}

    for class_name in sorted(os.listdir(source_dir)):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path) or class_name.startswith("."):
            continue

        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        files = [
            f
            for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f)) and not f.startswith(".")
        ]
        rng.shuffle(files)

        split_index = int(len(files) * split_ratio)
        train_files = files[:split_index]
        val_files = files[split_index:]

        for file_name in train_files:
            shutil.copy2(os.path.join(class_path, file_name), os.path.join(train_class_dir, file_name))
        for file_name in val_files:
            shutil.copy2(os.path.join(class_path, file_name), os.path.join(val_class_dir, file_name))

        summary[class_name] = {"train": len(train_files), "val": len(val_files)}

    return summary


def get_data_generators(
    train_dir: str,
    val_dir: Optional[str] = None,
    target_size: Tuple[int, int] = (150, 150),
    batch_size: int = 32,
    augment: bool = True,
):
    """Creates ImageDataGenerators for training and validation splits.

    Augmentations match the notebook baseline:
    - rescale: 1.0 / 255.0
    - shear_range: 0.2
    - zoom_range: 0.2
    - horizontal_flip: True

    Args:
        train_dir: Path to directory containing class subfolders for training.
        val_dir: Optional path to directory containing class subfolders for validation.
        target_size: Tuple of (height, width) to resize images to (default: (150, 150)).
        batch_size: Batch size for training and validation batches (default: 32).
        augment: Whether to apply data augmentation to the training generator.

    Returns:
        tuple (train_generator, val_generator) if val_dir is specified;
        otherwise returns train_generator.
    """
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    if val_dir:
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )
        return train_generator, val_generator

    return train_generator


def save_classes(class_indices: Dict[str, int], filepath: str) -> None:
    """Saves class index mapping into a standardized JSON file.

    Args:
        class_indices: Mapping from class name to numerical index (e.g., {'cardboard': 0, ...}).
        filepath: Destination file path for classes.json.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    sorted_classes = [
        cls for cls, _ in sorted(class_indices.items(), key=lambda item: item[1])
    ]
    data = {
        "classes": sorted_classes,
        "indices": class_indices,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_classes(filepath: str) -> List[str]:
    """Loads ordered class names from a classes.json file, or falls back to DEFAULT_CLASSES.

    Args:
        filepath: Path to classes.json file.

    Returns:
        List[str]: Ordered list of class label strings.
    """
    if not os.path.isfile(filepath):
        return list(DEFAULT_CLASSES)

    with open(filepath, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "classes" in data and isinstance(data["classes"], list):
            return data["classes"]
        if "indices" in data and isinstance(data["indices"], dict):
            return [cls for cls, _ in sorted(data["indices"].items(), key=lambda x: x[1])]
        return [cls for cls, _ in sorted(data.items(), key=lambda x: x[1])]
    elif isinstance(data, list):
        return data

    return list(DEFAULT_CLASSES)
