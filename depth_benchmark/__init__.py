"""Depth Benchmark - A flexible framework for evaluating monocular depth estimation models."""

__version__ = "0.1.0"

from .benchmark import DepthBenchmark, BenchmarkConfig, run_skyscenes_benchmark
from .metrics import DepthMetrics, compute_depth_metrics

__all__ = [
    "DepthBenchmark",
    "BenchmarkConfig",
    "DepthMetrics",
    "compute_depth_metrics",
    "run_skyscenes_benchmark",
    "__version__",
]
