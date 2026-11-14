"""CLI trainer for Garbage Classification CNN.

Supports:
- Automatic Kaggle dataset download via kagglehub
- Train/Validation splitting with configurable ratio and random seed
- Data augmentation generators
- Dynamic class discovery and classes.json export
- Modern Keras 3 training loop with EarlyStopping, evaluation, and .keras model saving
- Optional training curve visualization
"""

import argparse
import os
import sys
import tensorflow as tf

from src.dataset import (
    DEFAULT_KAGGLE_DATASET,
    download_kaggle_dataset,
    get_data_generators,
    save_classes,
    split_dataset,
)
from src.model import build_model, save_model
from src.utils import plot_training_history


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train CNN model on Garbage Classification dataset using modern Keras."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset root directory (or source directory containing class folders).",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Explicit path to train set directory.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Explicit path to validation set directory.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Kaggle dataset (asdasdasasdas/garbage-classification) via kagglehub.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training (default: 0.8).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=150,
        help="Target image height (default: 150).",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=150,
        help="Target image width (default: 150).",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=0.001,
        help="Adam optimizer learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Optional dropout rate before dense output layer (e.g. 0.2).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "mobilenet_v2"],
        help="Architecture type: 'custom_cnn' (baseline) or 'mobilenet_v2' (transfer learning).",
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=0,
        help="Number of base layers to unfreeze for transfer learning fine-tuning (default: 0).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Filepath where trained model is saved (defaults dynamically based on model-type).",
    )
    parser.add_argument(
        "--classes-path",
        type=str,
        default="classes.json",
        help="Filepath where class mapping is saved (default: classes.json).",
    )
    parser.add_argument(
        "--plot-history",
        type=str,
        default=None,
        help="Optional path to save training accuracy and loss curve plots (e.g. history.png).",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        metavar="PATIENCE",
        help="Patience (epochs without improvement) for EarlyStopping callback.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args(args)


def resolve_data_directories(args):
    """Resolves or prepares train_dir and val_dir based on CLI options."""
    train_dir = args.train_dir
    val_dir = args.val_dir

    if train_dir and val_dir:
        return train_dir, val_dir

    # Explicit download flag requested
    if args.download:
        print(f"Downloading dataset '{DEFAULT_KAGGLE_DATASET}' via kagglehub...")
        base_path = download_kaggle_dataset(DEFAULT_KAGGLE_DATASET)
        print(f"Dataset downloaded to: {base_path}")

        # Search for nested 'Garbage classification' folder
        candidate_nested = os.path.join(base_path, "Garbage classification", "Garbage classification")
        candidate_single = os.path.join(base_path, "Garbage classification")
        if os.path.isdir(candidate_nested):
            source_dir = candidate_nested
        elif os.path.isdir(candidate_single):
            source_dir = candidate_single
        else:
            source_dir = base_path

        train_dir = os.path.join(base_path, "train")
        val_dir = os.path.join(base_path, "val")
        print(f"Splitting dataset into train ({args.split_ratio*100:.0f}%) and val...")
        split_dataset(source_dir, train_dir, val_dir, split_ratio=args.split_ratio, seed=args.seed)
        return train_dir, val_dir

    # Using provided data_dir
    if args.data_dir:
        train_cand = os.path.join(args.data_dir, "train")
        val_cand = os.path.join(args.data_dir, "val")
        if os.path.isdir(train_cand) and os.path.isdir(val_cand):
            return train_cand, val_cand

        train_dir = train_cand
        val_dir = val_cand
        print(f"Splitting dataset from {args.data_dir} into {train_dir} and {val_dir}...")
        split_dataset(args.data_dir, train_dir, val_dir, split_ratio=args.split_ratio, seed=args.seed)
        return train_dir, val_dir

    raise ValueError("Must provide either --data-dir, --train-dir and --val-dir, or --download.")


def train(args):
    """Executes the training workflow."""
    tf.keras.utils.set_random_seed(args.seed)

    train_dir, val_dir = resolve_data_directories(args)
    target_size = (args.img_height, args.img_width)

    print(f"Loading datasets:\n  Train: {train_dir}\n  Val:   {val_dir}")
    train_gen, val_gen = get_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        target_size=target_size,
        batch_size=args.batch_size,
        augment=True,
    )

    num_classes = train_gen.num_classes
    class_labels = list(train_gen.class_indices.keys())
    print(f"Found {train_gen.samples} training images and {val_gen.samples} validation images.")
    print(f"Classes ({num_classes}): {class_labels}")

    # Persist class mapping for inference pipeline
    save_classes(train_gen.class_indices, args.classes_path)
    print(f"Class labels saved to: {args.classes_path}")

    model_path = args.model_path
    if model_path is None:
        model_path = (
            "GarbageMLModel_CS549_mobilenet_v2.keras"
            if args.model_type == "mobilenet_v2"
            else "GarbageMLModel_CS549.keras"
        )

    # Build CNN model
    input_shape = (args.img_height, args.img_width, 3)
    model = build_model(
        model_type=args.model_type,
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        fine_tune_layers=args.fine_tune_layers,
        compile_model=True,
    )
    model.summary()

    callbacks = []
    if args.early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping,
                restore_best_weights=True,
                verbose=1,
            )
        )

    steps_per_epoch = max(1, train_gen.samples // args.batch_size)
    val_steps = max(1, val_gen.samples // args.batch_size)

    print(f"\nStarting training for {args.epochs} epochs (steps_per_epoch={steps_per_epoch})...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    # Evaluate on validation data
    print("\nEvaluating on validation set...")
    val_loss, val_acc = model.evaluate(val_gen, steps=val_steps, verbose=1)
    print(f"Validation Accuracy: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}")

    # Save trained model
    saved_file = save_model(model, model_path)
    print(f"Model saved successfully to: {saved_file}")

    # Save training curves if requested
    if args.plot_history:
        plot_training_history(history, save_path=args.plot_history)
        print(f"Training history plot saved to: {args.plot_history}")

    return model, history


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
