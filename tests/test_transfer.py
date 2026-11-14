"""Unit tests for MobileNetV2 transfer learning and benchmarking."""

import pytest
import tensorflow as tf

from src.benchmark import benchmark_model, count_parameters, format_markdown_table
from src.model import build_custom_cnn, build_model, build_transfer_model


def test_build_transfer_model_frozen():
    """Verify MobileNetV2 builds with frozen base layers by default."""
    model = build_transfer_model(
        backbone="mobilenet_v2",
        input_shape=(128, 128, 3),
        num_classes=6,
        weights=None,
        fine_tune_layers=0,
        compile_model=True,
    )

    assert model.name == "Garbage_Mobilenet_v2"
    assert model.output_shape == (None, 6)

    params = count_parameters(model)
    assert params["total_params"] > 2_000_000
    # In feature extraction mode, base is frozen so trainable params are only head (<250k)
    assert params["trainable_params"] < 250_000
    assert params["non_trainable_params"] > 2_000_000


def test_build_transfer_model_finetuning():
    """Verify unfreezing top layers increases trainable parameters."""
    model_frozen = build_transfer_model(
        backbone="mobilenet_v2",
        input_shape=(128, 128, 3),
        num_classes=6,
        weights=None,
        fine_tune_layers=0,
        compile_model=False,
    )
    model_tuned = build_transfer_model(
        backbone="mobilenet_v2",
        input_shape=(128, 128, 3),
        num_classes=6,
        weights=None,
        fine_tune_layers=20,
        compile_model=False,
    )

    p_frozen = count_parameters(model_frozen)
    p_tuned = count_parameters(model_tuned)

    assert p_tuned["trainable_params"] > p_frozen["trainable_params"]


def test_build_model_unified_dispatch():
    """Verify unified build_model dispatches correctly based on model_type."""
    cnn = build_model(model_type="custom_cnn", input_shape=(64, 64, 3), num_classes=6, compile_model=False)
    assert cnn.name == "GarbageCNN"

    mob = build_model(model_type="mobilenet_v2", input_shape=(128, 128, 3), num_classes=6, compile_model=False)
    assert "Mobilenet" in mob.name


def test_benchmark_model_metrics():
    """Verify latency and throughput benchmarking execution."""
    model = build_custom_cnn(input_shape=(64, 64, 3), num_classes=6, compile_model=False)
    res = benchmark_model(model, input_shape=(64, 64, 3), num_runs=5, warmup_runs=2)

    assert "total_params" in res
    assert "trainable_params" in res
    assert "model_size_mb" in res
    assert "mean_latency_ms" in res
    assert "p50_latency_ms" in res
    assert "p95_latency_ms" in res
    assert "throughput_fps" in res
    assert res["mean_latency_ms"] > 0
    assert res["throughput_fps"] > 0


def test_format_markdown_table():
    """Verify markdown table formatter generates expected GitHub markdown."""
    dummy_results = [
        {
            "model_name": "ModelA",
            "total_params": 1000000,
            "trainable_params": 500000,
            "model_size_mb": 4.5,
            "mean_latency_ms": 12.3,
            "p50_latency_ms": 11.5,
            "p95_latency_ms": 15.0,
            "throughput_fps": 81.3,
        }
    ]
    table = format_markdown_table(dummy_results)
    assert "| Model Architecture |" in table
    assert "| **ModelA** |" in table
    assert "4.5 MB" in table
