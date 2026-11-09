"""Utility functions for visualization, training metrics, and model evaluation.

Extracted from evaluation code in GarbageMLModel_CS549 notebook (Cell 18).
"""

import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path: Optional[str] = None) -> None:
    """Plots training and validation accuracy and loss curves.

    Args:
        history: Keras History object or dict containing 'accuracy', 'val_accuracy',
                 'loss', and 'val_loss'.
        save_path: Optional path to save the resulting figure as an image.
    """
    history_dict = history.history if hasattr(history, "history") else history

    epochs = range(1, len(history_dict.get("accuracy", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    if "accuracy" in history_dict:
        ax1.plot(epochs, history_dict["accuracy"], label="Training Accuracy", color="#1f77b4", lw=2)
    if "val_accuracy" in history_dict:
        ax1.plot(epochs, history_dict["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", lw=2)
    ax1.set_title("Training and Validation Accuracy", fontsize=13)
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    if "loss" in history_dict:
        ax2.plot(epochs, history_dict["loss"], label="Training Loss", color="#1f77b4", lw=2)
    if "val_loss" in history_dict:
        ax2.plot(epochs, history_dict["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2)
    ax2.set_title("Training and Validation Loss", fontsize=13)
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def evaluate_model_performance(
    model,
    val_generator,
    save_cm_path: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Evaluates model predictions on a generator and generates confusion matrix and report.

    Args:
        model: Trained tf.keras.Model.
        val_generator: Data generator yielding (batch_x, batch_y).
        save_cm_path: Optional file path to save the confusion matrix heatmap plot.

    Returns:
        tuple (confusion_matrix, classification_report_string)
    """
    try:
        import seaborn as sns
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("Note: seaborn and scikit-learn are required to generate confusion matrix plots.")
        return None, None

    val_predictions = model.predict(val_generator)
    predicted_classes = np.argmax(val_predictions, axis=1)
    true_classes = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())

    cm = confusion_matrix(true_classes, predicted_classes)
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)

    if save_cm_path:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
        )
        ax.set_title("Confusion Matrix", fontsize=13)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class", fontsize=11)
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(save_cm_path)), exist_ok=True)
        plt.savefig(save_cm_path, dpi=200)
        plt.close(fig)

    return cm, report
