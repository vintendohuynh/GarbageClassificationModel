"""Inference pipeline and CLI for Garbage Classification CNN.

Provides:
- GarbagePredictor: programmatic inference interface
- CLI runner supporting single image, directories of images, and structured JSON output
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.dataset import DEFAULT_CLASSES, load_classes
from src.model import load_model


class GarbagePredictor:
    """Inference wrapper for garbage classification CNN models."""

    def __init__(
        self,
        model_path: str,
        classes_path: Optional[str] = None,
        target_size: Tuple[int, int] = (150, 150),
    ):
        """Initializes the predictor with a trained model and class labels.

        Args:
            model_path: Path to .keras or .h5 saved model.
            classes_path: Optional path to classes.json file.
            target_size: Target image dimensions (height, width) expected by the model.
        """
        self.model = load_model(model_path, compile_model=False)
        self.target_size = target_size

        if classes_path and os.path.isfile(classes_path):
            self.classes = load_classes(classes_path)
        else:
            self.classes = list(DEFAULT_CLASSES)

    def preprocess(self, image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocesses an input image into the normalized tensor format required by the CNN.

        Steps:
        1. Convert image to RGB (handles RGBA / grayscale / PNG / JPG).
        2. Resize image to target_size (default: 150x150).
        3. Convert to float32 and scale values by 1.0 / 255.0.
        4. Add batch dimension -> (1, height, width, 3).

        Args:
            image_input: Filepath string, PIL Image instance, or numpy array.

        Returns:
            np.ndarray: Batch tensor with shape (1, height, width, 3).
        """
        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"Input image not found: {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input.astype("uint8")).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        img = img.resize(self.target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """Classifies a single image and returns top-k predictions with confidence scores.

        Args:
            image_input: Filepath string, PIL Image instance, or numpy array.
            top_k: Number of highest-confidence predictions to return.

        Returns:
            List[Dict[str, Any]]: Ranked predictions list:
                [{"class": "cardboard", "index": 0, "confidence": 0.942}, ...]
        """
        tensor = self.preprocess(image_input)
        probabilities = self.model.predict(tensor, verbose=0)[0]

        top_k = max(1, min(top_k, len(probabilities)))
        ranked_indices = np.argsort(probabilities)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            label = self.classes[idx] if idx < len(self.classes) else f"class_{idx}"
            results.append(
                {
                    "class": label,
                    "index": int(idx),
                    "confidence": float(probabilities[idx]),
                }
            )
        return results

    def predict_batch(
        self,
        image_inputs: List[Union[str, Image.Image, np.ndarray]],
        top_k: int = 1,
    ) -> List[List[Dict[str, Any]]]:
        """Runs inference sequentially over multiple image inputs.

        Args:
            image_inputs: List of image filepaths, PIL Images, or numpy arrays.
            top_k: Number of highest-confidence predictions per image.

        Returns:
            List[List[Dict[str, Any]]]: Predictions for each input image.
        """
        return [self.predict(img, top_k=top_k) for img in image_inputs]


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run inference on images using trained Garbage Classification CNN."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i",
        "--image",
        type=str,
        help="Path to an image file.",
    )
    group.add_argument(
        "-d",
        "--dir",
        type=str,
        help="Path to a directory containing images to classify.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="GarbageMLModel_CS549.keras",
        help="Path to trained model file (.keras or .h5, default: GarbageMLModel_CS549.keras).",
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=str,
        default="classes.json",
        help="Path to classes JSON mapping (default: classes.json).",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[150, 150],
        metavar=("HEIGHT", "WIDTH"),
        help="Target image size (height width, default: 150 150).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()

    classes_path = args.classes if os.path.isfile(args.classes) else None
    predictor = GarbagePredictor(
        model_path=args.model,
        classes_path=classes_path,
        target_size=tuple(args.target_size),
    )

    if args.image:
        results = predictor.predict(args.image, top_k=args.top_k)
        if args.json:
            print(json.dumps({"image": args.image, "predictions": results}, indent=2))
        else:
            print(f"\nPredictions for: {args.image}")
            print("-" * 50)
            for rank, r in enumerate(results, start=1):
                conf_pct = r["confidence"] * 100.0
                print(f"{rank}. {r['class']:<15} {conf_pct:6.2f}% (index {r['index']})")
            print("-" * 50)

    elif args.dir:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            os.path.join(args.dir, f)
            for f in sorted(os.listdir(args.dir))
            if os.path.splitext(f.lower())[1] in valid_exts
        ]

        if not image_files:
            print(f"No valid image files found in directory: {args.dir}")
            sys.exit(1)

        batch_results = []
        for img_path in image_files:
            res = predictor.predict(img_path, top_k=args.top_k)
            batch_results.append({"image": img_path, "predictions": res})

        if args.json:
            print(json.dumps(batch_results, indent=2))
        else:
            print(f"\nClassified {len(batch_results)} images from: {args.dir}")
            print("=" * 60)
            for item in batch_results:
                top = item["predictions"][0]
                pct = top["confidence"] * 100.0
                print(f"{os.path.basename(item['image']):<25} -> {top['class']:<12} ({pct:.1f}%)")
            print("=" * 60)


if __name__ == "__main__":
    main()
