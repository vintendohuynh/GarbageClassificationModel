"""Model definitions, architecture factory, and persistence for Garbage Classification.

Modern Keras implementation matching the CNN baseline from GarbageMLModel_CS549:
- 3 Convolutional Blocks: Conv2D(32) -> Conv2D(64) -> Conv2D(128)
- ReLU activations and 2x2 MaxPooling
- Flatten layer
- Dense(128, relu)
- Dense(num_classes, softmax)
"""

import os
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_classes: int = 6,
    learning_rate: float = 0.001,
    dropout_rate: Optional[float] = None,
    compile_model: bool = True,
) -> tf.keras.Model:
    """Builds and optionally compiles the Garbage Classification CNN model.

    Args:
        input_shape: Input image dimensions as (height, width, channels).
                     Default is (150, 150, 3).
        num_classes: Number of classification categories (default: 6).
        learning_rate: Optimizer learning rate for Adam (default: 0.001).
        dropout_rate: Optional dropout rate between dense layers (e.g. 0.2).
        compile_model: Whether to compile the model with Adam and categorical crossentropy.

    Returns:
        tf.keras.Model: Sequential CNN model.
    """
    model = models.Sequential(name="GarbageCNN")

    # Explicit Input layer for modern Keras 3 compatibility
    model.add(layers.Input(shape=input_shape, name="image_input"))

    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation="relu", name="conv1"))
    model.add(layers.MaxPooling2D((2, 2), name="pool1"))

    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation="relu", name="conv2"))
    model.add(layers.MaxPooling2D((2, 2), name="pool2"))

    # Third Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation="relu", name="conv3"))
    model.add(layers.MaxPooling2D((2, 2), name="pool3"))

    # Classifier Head
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(128, activation="relu", name="dense_fc"))

    if dropout_rate is not None and 0.0 < dropout_rate < 1.0:
        model.add(layers.Dropout(dropout_rate, name="dropout"))

    # Output Softmax Layer
    model.add(layers.Dense(num_classes, activation="softmax", name="output_probabilities"))

    if compile_model:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


def save_model(model: tf.keras.Model, filepath: str) -> str:
    """Saves a model to disk in modern native Keras format (.keras) or legacy (.h5).

    Args:
        model: Trained or built tf.keras.Model.
        filepath: Destination path (e.g. 'GarbageMLModel_CS549.keras').

    Returns:
        str: Absolute or normalized filepath where the model was saved.
    """
    abs_path = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    model.save(filepath)
    return filepath


def load_model(filepath: str, compile_model: bool = False) -> tf.keras.Model:
    """Loads a saved Keras model from disk (.keras or .h5).

    Args:
        filepath: Path to the model file.
        compile_model: Whether to compile the model upon loading (default False for inference).

    Returns:
        tf.keras.Model: Loaded Keras model instance.

    Raises:
        FileNotFoundError: If the specified filepath does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return tf.keras.models.load_model(filepath, compile=compile_model)
