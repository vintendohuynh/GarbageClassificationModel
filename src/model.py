"""Model definitions, architecture factory, and persistence for Garbage Classification.

Supports:
- Baseline Custom CNN (3 conv blocks with MaxPooling and Dropout)
- Transfer Learning with MobileNetV2 pre-trained on ImageNet
- Fine-tuning layer control
- Modern Keras 3 persistence (.keras & .h5)
"""

import os
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_classes: int = 6,
    learning_rate: float = 0.001,
    dropout_rate: Optional[float] = None,
    compile_model: bool = True,
) -> tf.keras.Model:
    """Builds the baseline 3-block Convolutional Neural Network.

    Architecture:
    - Input: (150, 150, 3)
    - Conv2D(32, 3x3, ReLU) + MaxPooling2D(2x2)
    - Conv2D(64, 3x3, ReLU) + MaxPooling2D(2x2)
    - Conv2D(128, 3x3, ReLU) + MaxPooling2D(2x2)
    - Flatten
    - Dense(128, ReLU) + Optional Dropout
    - Dense(num_classes, Softmax)
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


def build_transfer_model(
    backbone: str = "mobilenet_v2",
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_classes: int = 6,
    weights: Optional[str] = "imagenet",
    fine_tune_layers: int = 0,
    learning_rate: float = 0.0005,
    compile_model: bool = True,
) -> tf.keras.Model:
    """Builds a Transfer Learning model using pre-trained ImageNet backbones.

    Args:
        backbone: Pre-trained architecture identifier ('mobilenet_v2').
        input_shape: Input image dimensions (H, W, C).
        num_classes: Number of classification categories.
        weights: Weights to initialize ('imagenet' or None for random).
        fine_tune_layers: Number of top base layers to unfreeze (0 = feature extraction).
        learning_rate: Optimizer learning rate.
        compile_model: Whether to compile with Adam and categorical crossentropy.

    Returns:
        tf.keras.Model: Functional transfer learning model.
    """
    inputs = layers.Input(shape=input_shape, name="image_input")

    if backbone.lower() == "mobilenet_v2":
        # Scale inputs from [0, 1] to [-1, 1] expected by MobileNetV2
        scaled_inputs = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights=weights,
            )
    else:
        raise ValueError(f"Unsupported transfer backbone: {backbone}. Choose 'mobilenet_v2'.")

    # Freeze base model weights for feature extraction
    base_model.trainable = False

    # Optional fine-tuning: unfreeze the top N layers of the base model
    if fine_tune_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

    # Forward pass through base model
    x = base_model(scaled_inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=f"Garbage_{backbone.capitalize()}")

    if compile_model:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


def build_model(
    model_type: str = "custom_cnn",
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_classes: int = 6,
    learning_rate: Optional[float] = None,
    dropout_rate: Optional[float] = None,
    fine_tune_layers: int = 0,
    compile_model: bool = True,
) -> tf.keras.Model:
    """Unified factory dispatching to Custom CNN or Transfer Learning models.

    Args:
        model_type: Either 'custom_cnn' (baseline) or 'mobilenet_v2' (transfer).
        input_shape: Image dimensions (H, W, C).
        num_classes: Number of classification categories.
        learning_rate: Custom learning rate (defaults to 0.001 for CNN, 0.0005 for Transfer).
        dropout_rate: Dropout rate for dense layers.
        fine_tune_layers: Number of base layers to unfreeze if using transfer learning.
        compile_model: Whether to compile the model.

    Returns:
        tf.keras.Model: Configured Keras model.
    """
    if model_type == "mobilenet_v2":
        lr = learning_rate if learning_rate is not None else 0.0005
        return build_transfer_model(
            backbone="mobilenet_v2",
            input_shape=input_shape,
            num_classes=num_classes,
            fine_tune_layers=fine_tune_layers,
            learning_rate=lr,
            compile_model=compile_model,
        )

    lr = learning_rate if learning_rate is not None else 0.001
    return build_custom_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=lr,
        dropout_rate=dropout_rate,
        compile_model=compile_model,
    )


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
