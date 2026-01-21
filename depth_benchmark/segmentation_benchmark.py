"""Core benchmark runner for segmentation evaluation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .datasets.base import BaseDepthDataset
from .metrics.segmentation_metrics import (
    SegmentationMetrics,
    SegmentationMetricsAccumulator,
    compute_binary_segmentation_metrics,
)
from .models.segmentation_base import BaseSegmentationModel


# Cityscapes class IDs for ground-like surfaces
CITYSCAPES_ROAD = 0
CITYSCAPES_SIDEWALK = 1
CITYSCAPES_TERRAIN = 9

# SkyScenes class IDs
SKYSCENES_ROAD = 7
SKYSCENES_SIDEWALK = 8
SKYSCENES_GROUND = 14


@dataclass
class SegmentationBenchmarkConfig:
    """Configuration for segmentation benchmark evaluation."""

    # Target classes to evaluate (SkyScenes class IDs)
    # Default: combined ground surfaces (road + sidewalk + ground)
    target_classes: Set[int] = field(default_factory=lambda: {7, 8, 14})
    target_class_name: str = "ground"

    # For models with different class mappings (e.g., Cityscapes)
    # Map model predictions to target class space
    # Key: model class ID, Value: SkyScenes class ID
    pred_class_mapping: Optional[Dict[int, int]] = None

    # Output settings
    save_dir: Optional[str] = None
    save_visualizations: bool = False
    visualization_samples: int = 10


class SegmentationBenchmark:
    """Main benchmark class for evaluating segmentation models."""

    def __init__(
        self,
        model: BaseSegmentationModel,
        dataset: BaseDepthDataset,
        config: Optional[SegmentationBenchmarkConfig] = None,
    ):
        """Initialize benchmark.

        Args:
            model: Segmentation model to evaluate.
            dataset: Dataset to evaluate on (must have load_segmentation=True).
            config: Benchmark configuration.
        """
        self.model = model
        self.dataset = dataset
        self.config = config or SegmentationBenchmarkConfig()

    def _map_predictions(self, pred: np.ndarray) -> np.ndarray:
        """Map model predictions to target class space if needed.

        Handles cases where model uses different class IDs (e.g., Cityscapes
        vs SkyScenes).
        """
        if self.config.pred_class_mapping is None:
            return pred

        mapped = np.copy(pred)
        for src_class, dst_class in self.config.pred_class_mapping.items():
            mapped[pred == src_class] = dst_class
        return mapped

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

        accumulator = SegmentationMetricsAccumulator(self.config.target_classes)
        per_sample_results = []

        iterator = tqdm(indices, desc="Evaluating") if progress else indices

        for idx in iterator:
            sample = self.dataset[idx]

            # Get RGB image
            rgb = sample["rgb"]
            if isinstance(rgb, torch.Tensor):
                rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                rgb_pil = Image.fromarray(rgb_np)
            else:
                rgb_pil = rgb

            # Get ground truth segmentation
            if "segmentation" not in sample:
                continue  # Skip samples without segmentation

            gt_seg = sample["segmentation"].numpy()
            metadata = sample["metadata"]

            # Predict segmentation
            pred_seg = self.model.predict(rgb_pil)

            # Resize prediction if needed
            if pred_seg.shape != gt_seg.shape:
                pred_seg = np.array(
                    Image.fromarray(pred_seg.astype(np.uint8)).resize(
                        (gt_seg.shape[1], gt_seg.shape[0]),
                        Image.NEAREST,  # Use nearest for class labels
                    )
                ).astype(np.int32)

            # Map predictions to target class space
            pred_seg = self._map_predictions(pred_seg)

            # Compute metrics for target classes
            metrics = compute_binary_segmentation_metrics(
                pred_seg,
                gt_seg,
                target_classes=self.config.target_classes,
            )

            accumulator.update(metrics)

            # Store per-sample results
            result = {
                "index": idx,
                **metrics.to_dict(),
                **metadata,
            }
            per_sample_results.append(result)

        # Aggregate results
        aggregated = accumulator.compute()

        # Create DataFrame
        df = pd.DataFrame(per_sample_results)

        return aggregated, df

    def evaluate_by_condition(
        self,
        group_by: List[str],
        progress: bool = True,
    ) -> pd.DataFrame:
        """Evaluate and group results by metadata conditions."""
        _, per_sample_df = self.evaluate(progress=progress)

        grouped = per_sample_df.groupby(group_by).agg(
            {
                "iou": ["mean", "std", "count"],
                "precision": ["mean", "std"],
                "recall": ["mean", "std"],
                "f1": ["mean", "std"],
                "n_gt_pixels": ["mean", "sum"],
            }
        )

        grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()

        return grouped
