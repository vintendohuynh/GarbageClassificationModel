"""Performance benchmarking tool for Garbage Classification models.

Measures:
- Parameter count (Total, Trainable, Non-trainable)
- Memory / Disk footprint (MB)
- Inference latency (p50, p95, Mean, Std Dev)
- Throughput (Images Per Second / FPS)
"""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from src.model import build_model, load_model


def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
    """Returns parameter counts for total, trainable, and non-trainable weights."""
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_variables]))
    total = trainable + non_trainable
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
    }


def benchmark_model(
    model_or_path: Union[str, tf.keras.Model],
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_runs: int = 50,
    warmup_runs: int = 10,
) -> Dict[str, Any]:
    """Measures model parameter counts, size, and forward pass latency.

    Args:
        model_or_path: Loaded tf.keras.Model or file path to model.
        input_shape: Input image dimensions (H, W, C).
        num_runs: Number of benchmark timed forward passes.
        warmup_runs: Number of un-timed warmup passes.

    Returns:
        Dict[str, Any]: Benchmark statistics including latency, FPS, and parameters.
    """
    model_size_mb = 0.0
    if isinstance(model_or_path, str):
        if os.path.isfile(model_or_path):
            model_size_mb = os.path.getsize(model_or_path) / (1024 * 1024)
        model = load_model(model_or_path, compile_model=False)
    else:
        model = model_or_path

    # Synthetic batch of 1 image
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, *input_shape)).astype(np.float32)

    # Warmup passes to trigger graph compilation / kernel initialization
    for _ in range(warmup_runs):
        _ = model.predict(dummy_input, verbose=0)

    # Timed forward passes
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy_input, verbose=0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # Convert to milliseconds

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    params = count_parameters(model)
    if model_size_mb == 0.0:
        # Estimate size from float32 weights (4 bytes per param)
        model_size_mb = params["total_params"] * 4.0 / (1024 * 1024)

    return {
        "model_name": model.name,
        "total_params": params["total_params"],
        "trainable_params": params["trainable_params"],
        "model_size_mb": round(model_size_mb, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_fps": round(fps, 1),
    }


def format_markdown_table(benchmark_results: List[Dict[str, Any]]) -> str:
    """Formats benchmark results into a GitHub-compatible Markdown comparison table."""
    header = (
        "| Model Architecture | Total Params | Trainable Params | Disk Size | Latency (p50) | Latency (p95) | Throughput |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    rows = []
    for res in benchmark_results:
        total_p = f"{res['total_params'] / 1e6:.2f}M"
        train_p = f"{res['trainable_params'] / 1e6:.2f}M"
        size = f"{res['model_size_mb']:.1f} MB"
        p50 = f"{res['p50_latency_ms']:.1f} ms"
        p95 = f"{res['p95_latency_ms']:.1f} ms"
        fps = f"{res['throughput_fps']:.0f} FPS"
        rows.append(f"| **{res['model_name']}** | {total_p} | {train_p} | {size} | {p50} | {p95} | {fps} |")

    return header + "\n".join(rows)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Benchmark inference latency, throughput, and parameters of Garbage CNN models."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="GarbageMLModel_CS549.keras",
        help="Path to trained model file (default: GarbageMLModel_CS549.keras).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "mobilenet_v2"],
        help="Model architecture type to benchmark if building from scratch.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run side-by-side benchmark comparing Custom CNN vs MobileNetV2.",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=50,
        help="Number of timed forward passes (default: 50).",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        default=True,
        help="Print results formatted as a Markdown table.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    print("=" * 65)
    print("🚀 Garbage Classification Model Performance Benchmark")
    print(f"   Timed Runs: {args.runs} passes | Target: CPU Inference")
    print("=" * 65)

    results = []

    if args.compare:
        print("\nBenchmarking Model 1: Custom CNN (Baseline)...")
        if os.path.isfile(args.model):
            cnn_model = load_model(args.model, compile_model=False)
        else:
            cnn_model = build_model(model_type="custom_cnn", compile_model=False)
        res_cnn = benchmark_model(cnn_model, num_runs=args.runs)
        res_cnn["model_name"] = "Custom CNN (Baseline)"
        results.append(res_cnn)

        print("\nBenchmarking Model 2: MobileNetV2 (Transfer Learning)...")
        mobilenet_path = "GarbageMLModel_CS549_mobilenet_v2.keras"
        if os.path.isfile(mobilenet_path):
            mb_model = load_model(mobilenet_path, compile_model=False)
        else:
            mb_model = build_model(model_type="mobilenet_v2", compile_model=False)
        res_mb = benchmark_model(mb_model, num_runs=args.runs)
        res_mb["model_name"] = "MobileNetV2 (Transfer)"
        results.append(res_mb)

    else:
        if os.path.isfile(args.model):
            print(f"Loading model checkpoint: {args.model}")
            model = load_model(args.model, compile_model=False)
        else:
            print(f"Building {args.model_type} architecture...")
            model = build_model(model_type=args.model_type, compile_model=False)

        res = benchmark_model(model, num_runs=args.runs)
        results.append(res)

    print("\n" + format_markdown_table(results) + "\n")


if __name__ == "__main__":
    main()
