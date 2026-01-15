#!/usr/bin/env python3
"""Evaluate depth estimation models on SkyScenes dataset.

This script evaluates depth estimation error as a function of altitude and pitch.

Usage:
    python scripts/evaluate_skyscenes.py --dataset /path/to/SkyScenes --model moge

Example:
    python scripts/evaluate_skyscenes.py \
        --dataset /home/ssavian/DATASETS/SkyScenes \
        --model moge \
        --output results/moge_skyscenes
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate depth estimation on SkyScenes"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to SkyScenes dataset root",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="moge",
        choices=["moge", "moge-vitl", "moge-2-vitl", "moge-2-vitl-normal"],
        help="Model to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Disable depth alignment (use raw metric predictions)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=1000.0,
        help="Maximum depth for evaluation (meters)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--altitudes",
        type=int,
        nargs="+",
        default=None,
        help="Filter by specific altitudes (e.g., --altitudes 15 35)",
    )
    parser.add_argument(
        "--pitches",
        type=int,
        nargs="+",
        default=None,
        help="Filter by specific pitches (e.g., --pitches 0 45 90)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Import here to avoid slow startup for --help
    from depth_benchmark.datasets.skyscenes import SkyScenesDataset
    from depth_benchmark.models import get_model
    from depth_benchmark.benchmark import DepthBenchmark, BenchmarkConfig

    print(f"=" * 60)
    print(f"SkyScenes Depth Benchmark")
    print(f"=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Align predictions: {not args.no_align}")
    print(f"Max depth: {args.max_depth}m")
    print(f"=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = SkyScenesDataset(
        root=args.dataset,
        altitudes=args.altitudes,
        pitches=args.pitches,
        max_samples=args.max_samples,
        max_depth=args.max_depth,
    )

    conditions = dataset.get_available_conditions()
    print(f"Loaded {len(dataset)} samples")
    print(f"Altitudes: {conditions['altitudes']}")
    print(f"Pitches: {conditions['pitches']}")
    print(f"Weathers: {conditions['weathers']}")
    print(f"Towns: {conditions['towns']}")

    # Load model
    print(f"\nLoading model: {args.model}...")
    model = get_model(args.model, device=args.device)
    model.load()
    print(f"Model loaded: {model}")

    # Configure benchmark
    config = BenchmarkConfig(
        max_depth=args.max_depth,
        align_prediction=not args.no_align,
        save_dir=args.output,
    )

    # Run benchmark
    print("\nRunning evaluation...")
    benchmark = DepthBenchmark(model, dataset, config)

    # Run evaluation once and get per-sample results
    overall_metrics, per_sample_df = benchmark.evaluate(progress=True)

    # Group by altitude and pitch
    results_df = per_sample_df.groupby(["altitude", "pitch"]).agg(
        {
            "abs_rel": ["mean", "std", "count"],
            "sq_rel": ["mean", "std"],
            "rmse": ["mean", "std"],
            "delta_1": ["mean", "std"],
            "gt_depth_mean": ["mean", "min", "max"],
        }
    )
    results_df.columns = ["_".join(col).strip() for col in results_df.columns.values]
    results_df = results_df.reset_index()

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS BY ALTITUDE AND PITCH")
    print("=" * 60)

    # Format for display
    display_cols = [
        "altitude",
        "pitch",
        "gt_depth_mean_mean",
        "abs_rel_mean",
        "rmse_mean",
        "delta_1_mean",
        "abs_rel_count",
    ]
    display_df = results_df[display_cols].copy()
    display_df.columns = ["Alt", "Pitch", "GT_Depth", "AbsRel", "RMSE", "δ<1.25", "N"]
    print(display_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    for key, value in overall_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path / "results_by_condition.csv", index=False)
    per_sample_df.to_csv(output_path / "results_per_sample.csv", index=False)

    # Save overall metrics
    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(output_path / "results_overall.csv", index=False)

    print(f"\nResults saved to {output_path}/")


if __name__ == "__main__":
    main()
