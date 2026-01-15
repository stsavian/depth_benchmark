"""Core benchmark runner for depth estimation evaluation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets.base import BaseDepthDataset
from .metrics.depth_metrics import (
    DepthMetrics,
    MetricsAccumulator,
    align_depth_least_squares,
    align_depth_scale,
    compute_depth_metrics,
)
from .models.base import BaseDepthModel


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    # Evaluation settings
    min_depth: float = 0.001
    max_depth: float = 80.0
    align_prediction: bool = True
    alignment_method: str = "least_squares"  # least_squares, median

    # Output settings
    save_dir: Optional[str] = None
    save_predictions: bool = False
    save_visualizations: bool = False
    visualization_samples: int = 10


class DepthBenchmark:
    """Main benchmark class for evaluating depth estimation models."""

    def __init__(
        self,
        model: BaseDepthModel,
        dataset: BaseDepthDataset,
        config: Optional[BenchmarkConfig] = None,
    ):
        """Initialize benchmark.

        Args:
            model: Depth estimation model to evaluate.
            dataset: Dataset to evaluate on.
            config: Benchmark configuration.
        """
        self.model = model
        self.dataset = dataset
        self.config = config or BenchmarkConfig()

    def evaluate(
        self,
        indices: Optional[List[int]] = None,
        progress: bool = True,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Run evaluation on the dataset.

        Args:
            indices: Specific sample indices to evaluate. If None, evaluate all.
            progress: Show progress bar.

        Returns:
            Tuple of (aggregated_metrics_dict, per_sample_dataframe).
        """
        if indices is None:
            indices = list(range(len(self.dataset)))

        accumulator = MetricsAccumulator()
        per_sample_results = []

        iterator = tqdm(indices, desc="Evaluating") if progress else indices

        for idx in iterator:
            # Load sample
            sample = self.dataset[idx]
            rgb = sample["rgb"]
            gt_depth = sample["depth"].numpy()
            mask = sample["mask"].numpy()
            metadata = sample["metadata"]

            # Convert tensor to PIL for model input
            if isinstance(rgb, torch.Tensor):
                rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                rgb_pil = Image.fromarray(rgb_np)
            else:
                rgb_pil = rgb

            # Predict depth
            pred_depth = self.model.predict(rgb_pil)

            # Resize prediction if needed
            if pred_depth.shape != gt_depth.shape:
                pred_depth = np.array(
                    Image.fromarray(pred_depth).resize(
                        (gt_depth.shape[1], gt_depth.shape[0]),
                        Image.BILINEAR,
                    )
                )

            # Align prediction if not metric or if requested
            if self.config.align_prediction:
                if self.config.alignment_method == "least_squares":
                    pred_depth, scale, shift = align_depth_least_squares(
                        pred_depth, gt_depth, mask.astype(bool)
                    )
                else:
                    pred_depth, scale = align_depth_scale(
                        pred_depth, gt_depth, mask.astype(bool)
                    )
                    shift = 0.0
            else:
                scale, shift = 1.0, 0.0

            # Compute metrics
            metrics = compute_depth_metrics(
                pred_depth,
                gt_depth,
                mask.astype(bool),
                min_depth=self.config.min_depth,
                max_depth=self.config.max_depth,
            )

            accumulator.update(metrics)

            # Store per-sample results
            result = {
                "index": idx,
                **metrics.to_dict(),
                "scale": scale,
                "shift": shift,
                **metadata,
            }
            per_sample_results.append(result)

        # Aggregate results
        aggregated = accumulator.compute()
        aggregated.update(accumulator.compute_std())

        # Create DataFrame
        df = pd.DataFrame(per_sample_results)

        return aggregated, df

    def evaluate_by_condition(
        self,
        group_by: List[str],
        progress: bool = True,
    ) -> pd.DataFrame:
        """Evaluate and group results by metadata conditions.

        Args:
            group_by: List of metadata keys to group by (e.g., ["altitude", "pitch"]).
            progress: Show progress bar.

        Returns:
            DataFrame with metrics grouped by conditions.
        """
        # First run full evaluation
        _, per_sample_df = self.evaluate(progress=progress)

        # Group and aggregate
        grouped = per_sample_df.groupby(group_by).agg(
            {
                "abs_rel": ["mean", "std", "count"],
                "sq_rel": ["mean", "std"],
                "rmse": ["mean", "std"],
                "rmse_log": ["mean", "std"],
                "delta_1": ["mean", "std"],
                "delta_2": ["mean", "std"],
                "delta_3": ["mean", "std"],
                "silog": ["mean", "std"],
            }
        )

        # Flatten column names
        grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()

        return grouped


def run_skyscenes_benchmark(
    model: BaseDepthModel,
    dataset_root: str,
    output_dir: Optional[str] = None,
    align: bool = True,
    max_depth: float = 1000.0,
) -> pd.DataFrame:
    """Convenience function to run benchmark on SkyScenes dataset.

    Evaluates depth estimation performance grouped by altitude and pitch.

    Args:
        model: Depth estimation model to evaluate.
        dataset_root: Path to SkyScenes dataset root.
        output_dir: Directory to save results.
        align: Whether to align predictions to ground truth.
        max_depth: Maximum depth for evaluation.

    Returns:
        DataFrame with metrics grouped by altitude and pitch.
    """
    from .datasets.skyscenes import SkyScenesDataset

    # Load dataset
    dataset = SkyScenesDataset(
        root=dataset_root,
        max_depth=max_depth,
    )

    print(f"Loaded {len(dataset)} samples")
    print(f"Conditions: {dataset.get_available_conditions()}")

    # Configure benchmark
    config = BenchmarkConfig(
        max_depth=max_depth,
        align_prediction=align,
        save_dir=output_dir,
    )

    # Run benchmark
    benchmark = DepthBenchmark(model, dataset, config)
    results_df = benchmark.evaluate_by_condition(
        group_by=["altitude", "pitch"],
        progress=True,
    )

    # Save results
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path / "results_by_altitude_pitch.csv", index=False)
        print(f"Results saved to {output_path / 'results_by_altitude_pitch.csv'}")

    return results_df
