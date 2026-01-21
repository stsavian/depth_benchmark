"""Semantic segmentation metrics."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np


@dataclass
class SegmentationMetrics:
    """Container for segmentation metrics.

    Metrics:
    - iou: Intersection over Union (Jaccard index)
    - precision: True positives / (True positives + False positives)
    - recall: True positives / (True positives + False negatives)
    - f1: 2 * precision * recall / (precision + recall)
    - accuracy: Pixel accuracy for the class
    """

    iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    n_gt_pixels: int = 0
    n_pred_pixels: int = 0
    n_intersection: int = 0
    n_union: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "n_gt_pixels": self.n_gt_pixels,
            "n_pred_pixels": self.n_pred_pixels,
        }

    def __str__(self) -> str:
        return (
            f"IoU={self.iou:.4f}, "
            f"Precision={self.precision:.4f}, "
            f"Recall={self.recall:.4f}, "
            f"F1={self.f1:.4f}, "
            f"Accuracy={self.accuracy:.4f}"
        )


def compute_binary_segmentation_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    target_classes: Set[int],
    mask: Optional[np.ndarray] = None,
) -> SegmentationMetrics:
    """Compute segmentation metrics for a set of classes (binary evaluation).

    Combines multiple classes into a single "positive" class for evaluation.

    Args:
        pred: Predicted segmentation mask (H, W) with class IDs.
        gt: Ground truth segmentation mask (H, W) with class IDs.
        target_classes: Set of class IDs to treat as positive (e.g., {7, 8, 14} for ground).
        mask: Optional validity mask (H, W), True for valid pixels.

    Returns:
        SegmentationMetrics for the combined target classes.
    """
    # Create binary masks for target classes
    pred_binary = np.isin(pred, list(target_classes))
    gt_binary = np.isin(gt, list(target_classes))

    # Apply validity mask if provided
    if mask is not None:
        valid = mask.astype(bool)
        pred_binary = pred_binary & valid
        gt_binary = gt_binary & valid
        total_pixels = np.sum(valid)
    else:
        total_pixels = pred.size

    # Compute intersection and union
    intersection = np.sum(pred_binary & gt_binary)
    union = np.sum(pred_binary | gt_binary)

    n_gt = np.sum(gt_binary)
    n_pred = np.sum(pred_binary)

    # Handle edge cases
    if union == 0:
        iou = 1.0 if n_gt == 0 else 0.0
    else:
        iou = intersection / union

    if n_pred == 0:
        precision = 1.0 if n_gt == 0 else 0.0
    else:
        precision = intersection / n_pred

    if n_gt == 0:
        recall = 1.0
    else:
        recall = intersection / n_gt

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # Pixel accuracy (how many pixels correctly classified)
    if mask is not None:
        correct = np.sum((pred_binary == gt_binary) & valid)
    else:
        correct = np.sum(pred_binary == gt_binary)
    accuracy = correct / total_pixels if total_pixels > 0 else 0.0

    return SegmentationMetrics(
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(accuracy),
        n_gt_pixels=int(n_gt),
        n_pred_pixels=int(n_pred),
        n_intersection=int(intersection),
        n_union=int(union),
    )


def compute_multi_class_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    class_ids: List[int],
    mask: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    """Compute IoU for multiple classes individually.

    Args:
        pred: Predicted segmentation mask (H, W).
        gt: Ground truth segmentation mask (H, W).
        class_ids: List of class IDs to evaluate.
        mask: Optional validity mask.

    Returns:
        Dictionary mapping class_id -> IoU.
    """
    results = {}
    for class_id in class_ids:
        metrics = compute_binary_segmentation_metrics(pred, gt, {class_id}, mask)
        results[class_id] = metrics.iou
    return results


class SegmentationMetricsAccumulator:
    """Accumulates segmentation metrics over multiple samples."""

    def __init__(self, target_classes: Set[int]):
        """Initialize accumulator.

        Args:
            target_classes: Set of class IDs to track (e.g., {7, 8, 14} for ground).
        """
        self.target_classes = target_classes
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self._total_intersection = 0
        self._total_union = 0
        self._total_gt_pixels = 0
        self._total_pred_pixels = 0
        self._all_ious = []
        self._all_f1s = []
        self._all_precisions = []
        self._all_recalls = []
        self._count = 0

    def update(self, metrics: SegmentationMetrics):
        """Add metrics from a single sample."""
        self._total_intersection += metrics.n_intersection
        self._total_union += metrics.n_union
        self._total_gt_pixels += metrics.n_gt_pixels
        self._total_pred_pixels += metrics.n_pred_pixels
        self._all_ious.append(metrics.iou)
        self._all_f1s.append(metrics.f1)
        self._all_precisions.append(metrics.precision)
        self._all_recalls.append(metrics.recall)
        self._count += 1

    def compute(self) -> Dict[str, float]:
        """Compute aggregated metrics.

        Returns both:
        - mean IoU (average of per-sample IoUs)
        - global IoU (total intersection / total union)
        """
        if self._count == 0:
            return {"mean_iou": 0.0, "global_iou": 0.0, "n_samples": 0}

        mean_iou = np.mean(self._all_ious)
        mean_f1 = np.mean(self._all_f1s)
        mean_precision = np.mean(self._all_precisions)
        mean_recall = np.mean(self._all_recalls)

        if self._total_union > 0:
            global_iou = self._total_intersection / self._total_union
        else:
            global_iou = 0.0

        if self._total_pred_pixels > 0:
            global_precision = self._total_intersection / self._total_pred_pixels
        else:
            global_precision = 0.0

        if self._total_gt_pixels > 0:
            global_recall = self._total_intersection / self._total_gt_pixels
        else:
            global_recall = 0.0

        return {
            "mean_iou": float(mean_iou),
            "global_iou": float(global_iou),
            "mean_f1": float(mean_f1),
            "mean_precision": float(mean_precision),
            "mean_recall": float(mean_recall),
            "global_precision": float(global_precision),
            "global_recall": float(global_recall),
            "iou_std": float(np.std(self._all_ious)) if len(self._all_ious) > 1 else 0.0,
            "n_samples": self._count,
        }
