"""Model Explainability with Grad-CAM (Gradient-weighted Class Activation Mapping).

Computes visual explanations for decisions made by the Garbage Classification CNN,
highlighting salient image regions (e.g., textures, edges, materials) driving predictions.
"""

import argparse
import os
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

from src.dataset import DEFAULT_CLASSES, load_classes
from src.model import load_model


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = "conv3",
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """Computes a 2D Grad-CAM activation heatmap for a given input tensor.

    Uses GradientTape to calculate the gradients of the target class score
    with respect to the activations of the final convolutional layer.

    Args:
        img_array: Preprocessed input tensor of shape (1, height, width, 3).
        model: Trained tf.keras.Model instance.
        last_conv_layer_name: Name of the target convolutional layer (default: 'conv3').
        pred_index: Numerical class index to explain. If None, uses top predicted class.

    Returns:
        np.ndarray: 2D numpy array of shape (height, width) with values normalized to [0.0, 1.0].
    """
    last_conv_layer = model.get_layer(last_conv_layer_name)
    feature_extractor = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output,
    )

    # Locate downstream classifier layers following the target conv layer
    conv_idx = next(i for i, layer in enumerate(model.layers) if layer.name == last_conv_layer_name)
    classifier_layers = model.layers[conv_idx + 1 :]

    input_data = [img_array] if isinstance(img_array, (np.ndarray, tf.Tensor)) else img_array

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(input_data)
        tape.watch(conv_outputs)
        x = conv_outputs
        for layer in classifier_layers:
            x = layer(x)
        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of the class score with respect to feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature channels by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only features that positively contributed to target class
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_gradcam(
    image_input: Union[str, Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """Superimposes a Grad-CAM heatmap onto an input image.

    Args:
        image_input: Path to image, PIL Image, or numpy array.
        heatmap: 2D numpy array normalized to [0.0, 1.0].
        alpha: Blending weight of heatmap over original image (default: 0.4).
        colormap: Matplotlib colormap name (default: 'jet').

    Returns:
        PIL.Image.Image: Blended RGB image.
    """
    if isinstance(image_input, str):
        orig_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        orig_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        orig_img = Image.fromarray(image_input.astype("uint8")).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Resize heatmap to match the original image's size
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_resized = heatmap_pil.resize(orig_img.size, resample=Image.Resampling.BILINEAR)
    heatmap_norm = np.array(heatmap_resized) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_norm)  # (H, W, 4) in [0, 1]
    colored_rgb = np.uint8(255 * colored_heatmap[:, :, :3])

    # Blend original and heatmap
    orig_arr = np.array(orig_img, dtype=np.float32)
    blended = (1.0 - alpha) * orig_arr + alpha * colored_rgb
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def create_gradcam_comparison(
    image_input: Union[str, Image.Image, np.ndarray],
    heatmap: np.ndarray,
    superimposed_img: Image.Image,
    class_label: str,
    confidence: float,
    colormap: str = "jet",
) -> Image.Image:
    """Creates a 3-panel comparison image: [Original] | [Attention Heatmap] | [Superimposed].

    Args:
        image_input: Original image.
        heatmap: 2D numpy array heatmap.
        superimposed_img: Blended overlay image.
        class_label: Predicted class name.
        confidence: Prediction confidence score (0.0 to 1.0).
        colormap: Colormap used for heatmap.

    Returns:
        PIL.Image.Image: 3-panel comparison figure.
    """
    if isinstance(image_input, str):
        orig_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        orig_img = image_input.convert("RGB")
    else:
        orig_img = Image.fromarray(image_input.astype("uint8")).convert("RGB")

    target_size = (300, 300)
    img1 = orig_img.resize(target_size)
    img3 = superimposed_img.resize(target_size)

    # Prepare colored heatmap panel
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize(target_size, resample=Image.Resampling.BILINEAR)
    cmap = plt.get_cmap(colormap)
    colored_hm = np.uint8(255 * cmap(np.array(heatmap_pil) / 255.0)[:, :, :3])
    img2 = Image.fromarray(colored_hm)

    # Create composite canvas
    margin = 20
    header = 50
    footer = 40
    width = target_size[0] * 3 + margin * 4
    height = target_size[1] + header + footer

    canvas = Image.new("RGB", (width, height), color=(248, 249, 250))
    draw = ImageDraw.Draw(canvas)

    # Paste panels
    x1 = margin
    x2 = margin * 2 + target_size[0]
    x3 = margin * 3 + target_size[0] * 2
    y_pos = header

    canvas.paste(img1, (x1, y_pos))
    canvas.paste(img2, (x2, y_pos))
    canvas.paste(img3, (x3, y_pos))

    # Add text labels
    draw.text((x1 + 10, 15), "1. Original Image", fill=(30, 41, 59))
    draw.text((x2 + 10, 15), "2. Grad-CAM Activation", fill=(30, 41, 59))
    draw.text((x3 + 10, 15), "3. Superimposed Overlay", fill=(30, 41, 59))

    footer_text = f"Prediction: {class_label.upper()} ({confidence * 100:.1f}% confidence)"
    draw.text((x1 + 10, height - 30), footer_text, fill=(15, 23, 42))

    return canvas


def explain_image(
    image_path: str,
    model_path: str = "GarbageMLModel_CS549.keras",
    classes_path: Optional[str] = "classes.json",
    layer_name: str = "conv3",
    target_class: Optional[Union[int, str]] = None,
    alpha: float = 0.4,
    target_size: Tuple[int, int] = (150, 150),
) -> Tuple[np.ndarray, Image.Image, str, float]:
    """Runs end-to-end explainability pipeline on an image."""
    model = load_model(model_path, compile_model=False)
    classes = load_classes(classes_path) if classes_path and os.path.isfile(classes_path) else list(DEFAULT_CLASSES)

    # Load & preprocess
    img = Image.open(image_path).convert("RGB")
    tensor = np.array(img.resize(target_size), dtype=np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)

    # Predict
    preds = model.predict(tensor, verbose=0)[0]
    if target_class is None:
        pred_idx = int(np.argmax(preds))
    elif isinstance(target_class, str):
        pred_idx = classes.index(target_class) if target_class in classes else int(np.argmax(preds))
    else:
        pred_idx = int(target_class)

    class_name = classes[pred_idx] if pred_idx < len(classes) else f"class_{pred_idx}"
    confidence = float(preds[pred_idx])

    # Compute heatmap & overlay
    heatmap = make_gradcam_heatmap(tensor, model, last_conv_layer_name=layer_name, pred_index=pred_idx)
    overlay = overlay_gradcam(img, heatmap, alpha=alpha)

    return heatmap, overlay, class_name, confidence


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Explain Garbage Classification CNN decisions with Grad-CAM heatmaps."
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Path to input image file to explain.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="GarbageMLModel_CS549.keras",
        help="Path to trained model (.keras or .h5, default: GarbageMLModel_CS549.keras).",
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=str,
        default="classes.json",
        help="Path to classes JSON mapping (default: classes.json).",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=str,
        default="conv3",
        help="Name of convolutional layer to visualize (default: conv3).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="gradcam_result.png",
        help="Output path to save result image (default: gradcam_result.png).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Opacity weight for heatmap overlay (default: 0.4).",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        default=True,
        help="Save 3-panel comparison figure (Original | Heatmap | Overlay).",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    print(f"Generating Grad-CAM visualization for: {args.image}")
    heatmap, overlay, class_name, confidence = explain_image(
        image_path=args.image,
        model_path=args.model,
        classes_path=args.classes,
        layer_name=args.layer,
        alpha=args.alpha,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.comparison:
        comp_img = create_gradcam_comparison(
            image_input=args.image,
            heatmap=heatmap,
            superimposed_img=overlay,
            class_label=class_name,
            confidence=confidence,
        )
        comp_img.save(args.output)
    else:
        overlay.save(args.output)

    pct = confidence * 100.0
    print(f"Predicted Class: {class_name} ({pct:.2f}% confidence)")
    print(f"Grad-CAM explanation saved successfully to: {args.output}")


if __name__ == "__main__":
    main()
