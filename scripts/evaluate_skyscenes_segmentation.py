#!/usr/bin/env python3
"""Evaluate segmentation models on SkyScenes dataset.

This script evaluates ground segmentation performance using IoU, precision,
recall, and F1 metrics.

Usage:
    python scripts/evaluate_skyscenes_segmentation.py \
        --dataset /path/to/SkyScenes \
        --model segformer-b5-cityscapes

Note: Segmentation ground truth only available for ClearNoon weather.
"""

import argparse
from pathlib import Path

import pandas as pd


# Cityscapes -> SkyScenes class mapping for ground-like classes
# Cityscapes: terrain=9, road=0, sidewalk=1
# SkyScenes: ground=14, road=7, sidewalk=8
CITYSCAPES_TO_SKYSCENES_GROUND = {
    0: 7,    # road -> road
    1: 8,    # sidewalk -> sidewalk
    9: 14,   # terrain -> ground
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation on SkyScenes"
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
        default="segformer-b5-cityscapes",
        help="Segmentation model to evaluate (segformer-b5-cityscapes, ransac, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/segmentation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--altitudes",
        type=int,
        nargs="+",
        default=None,
        help="Filter by specific altitudes",
    )
    parser.add_argument(
        "--pitches",
        type=int,
        nargs="+",
        default=None,
        help="Filter by specific pitches",
    )
    parser.add_argument(
        "--towns",
        type=str,
        nargs="+",
        default=None,
        help="Filter by specific towns",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default=None,
        help="Depth model for RANSAC (e.g., 'moge'). If not specified, uses GT depth.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Import here to avoid slow startup
    from depth_benchmark.datasets.skyscenes import (
        SkyScenesDataset,
        SKYSCENES_GROUND_CLASSES,
    )
    from depth_benchmark.models.segmentation_registry import get_segmentation_model
    from depth_benchmark.segmentation_benchmark import (
        SegmentationBenchmark,
        SegmentationBenchmarkConfig,
    )

    print("=" * 60)
    print("SkyScenes Ground Segmentation Benchmark")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Depth source: {args.depth_model if args.depth_model else 'GT depth'}")
    print(f"Target classes: {SKYSCENES_GROUND_CLASSES} (road + sidewalk + ground)")
    print("=" * 60)

    # Load dataset (force ClearNoon for segmentation)
    print("\nLoading dataset...")
    dataset = SkyScenesDataset(
        root=args.dataset,
        altitudes=args.altitudes,
        pitches=args.pitches,
        towns=args.towns,
        max_samples=args.max_samples,
        load_segmentation=True,  # This forces ClearNoon weather
    )

    conditions = dataset.get_available_conditions()
    print(f"Loaded {len(dataset)} samples")
    print(f"Altitudes: {conditions['altitudes']}")
    print(f"Pitches: {conditions['pitches']}")
    print(f"Towns: {conditions['towns']}")

    # Load model
    print(f"\nLoading model: {args.model}...")
    model = get_segmentation_model(args.model, device=args.device)
    model.load()
    print(f"Model loaded: {model}")

    # Configure class mapping for Cityscapes models
    # RANSAC outputs SkyScenes class IDs directly, no mapping needed
    pred_class_mapping = None
    if "cityscapes" in args.model.lower() or "segformer" in args.model.lower():
        pred_class_mapping = CITYSCAPES_TO_SKYSCENES_GROUND
        print(f"Using Cityscapes -> SkyScenes ground class mapping")

    # Load depth model if specified
    depth_model = None
    if args.depth_model:
        from depth_benchmark.models.registry import get_model
        print(f"\nLoading depth model: {args.depth_model}...")
        depth_model = get_model(args.depth_model, device=args.device)
        depth_model.load()
        print(f"Depth model loaded: {depth_model}")

    # Configure benchmark
    config = SegmentationBenchmarkConfig(
        target_classes=SKYSCENES_GROUND_CLASSES,
        target_class_name="ground",
        pred_class_mapping=pred_class_mapping,
        save_dir=args.output,
    )

    # Run benchmark
    print("\nRunning evaluation...")
    benchmark = SegmentationBenchmark(model, dataset, config, depth_model=depth_model)

    overall_metrics, per_sample_df = benchmark.evaluate(progress=True)

    # Group by altitude and pitch
    results_df = per_sample_df.groupby(["altitude", "pitch"]).agg(
        {
            "iou": ["mean", "std", "count"],
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1": ["mean", "std"],
        }
    )
    results_df.columns = ["_".join(col).strip() for col in results_df.columns.values]
    results_df = results_df.reset_index()

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS BY ALTITUDE AND PITCH")
    print("=" * 60)

    display_cols = [
        "altitude", "pitch",
        "iou_mean", "precision_mean", "recall_mean", "f1_mean", "iou_count"
    ]
    display_df = results_df[display_cols].copy()
    display_df.columns = ["Alt", "Pitch", "IoU", "Precision", "Recall", "F1", "N"]
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

    results_df.to_csv(output_path / "segmentation_by_condition.csv", index=False)
    per_sample_df.to_csv(output_path / "segmentation_per_sample.csv", index=False)

    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(output_path / "segmentation_overall.csv", index=False)

    print(f"\nResults saved to {output_path}/")


if __name__ == "__main__":
    main()
