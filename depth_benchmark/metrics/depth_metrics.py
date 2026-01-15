"""Standard depth estimation metrics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class DepthMetrics:
    """Container for depth estimation metrics.

    Standard metrics used in monocular depth estimation benchmarks:
    - abs_rel: Absolute relative error
    - sq_rel: Squared relative error
    - rmse: Root mean squared error
    - rmse_log: Root mean squared error in log space
    - delta_1: Percentage of pixels with max(pred/gt, gt/pred) < 1.25
    - delta_2: Percentage of pixels with max(pred/gt, gt/pred) < 1.25^2
    - delta_3: Percentage of pixels with max(pred/gt, gt/pred) < 1.25^3
    - silog: Scale-invariant logarithmic error
    - log10: Mean absolute log10 error
    """

    abs_rel: float = 0.0
    sq_rel: float = 0.0
    rmse: float = 0.0
    rmse_log: float = 0.0
    delta_1: float = 0.0
    delta_2: float = 0.0
    delta_3: float = 0.0
    silog: float = 0.0
    log10: float = 0.0
    n_valid: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "abs_rel": self.abs_rel,
            "sq_rel": self.sq_rel,
            "rmse": self.rmse,
            "rmse_log": self.rmse_log,
            "delta_1": self.delta_1,
            "delta_2": self.delta_2,
            "delta_3": self.delta_3,
            "silog": self.silog,
            "log10": self.log10,
            "n_valid": self.n_valid,
        }

    def __str__(self) -> str:
        return (
            f"abs_rel={self.abs_rel:.4f}, "
            f"sq_rel={self.sq_rel:.4f}, "
            f"rmse={self.rmse:.4f}, "
            f"rmse_log={self.rmse_log:.4f}, "
            f"\u03b4<1.25={self.delta_1:.4f}, "
            f"\u03b4<1.25\u00b2={self.delta_2:.4f}, "
            f"\u03b4<1.25\u00b3={self.delta_3:.4f}, "
            f"silog={self.silog:.4f}"
        )


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
    min_depth: float = 0.001,
    max_depth: float = 80.0,
) -> DepthMetrics:
    """Compute depth estimation metrics.

    Args:
        pred: Predicted depth map (H, W) in meters.
        gt: Ground truth depth map (H, W) in meters.
        mask: Optional validity mask (H, W), True/1 for valid pixels.
        min_depth: Minimum valid depth for evaluation.
        max_depth: Maximum valid depth for evaluation.

    Returns:
        DepthMetrics containing all computed metrics.
    """
    # Ensure numpy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Create validity mask
    valid_mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt)
    valid_mask &= (pred > min_depth) & (pred < max_depth) & np.isfinite(pred)

    if mask is not None:
        valid_mask &= mask.astype(bool)

    # Extract valid pixels
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    n_valid = len(pred_valid)

    if n_valid == 0:
        return DepthMetrics(n_valid=0)

    # Compute metrics
    thresh = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)

    delta_1 = np.mean(thresh < 1.25)
    delta_2 = np.mean(thresh < 1.25**2)
    delta_3 = np.mean(thresh < 1.25**3)

    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    sq_rel = np.mean(((pred_valid - gt_valid) ** 2) / gt_valid)

    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))

    # Log-space metrics
    log_pred = np.log(pred_valid)
    log_gt = np.log(gt_valid)
    log_diff = log_pred - log_gt

    rmse_log = np.sqrt(np.mean(log_diff**2))

    # Scale-invariant log error (silog)
    silog = np.sqrt(np.mean(log_diff**2) - np.mean(log_diff) ** 2) * 100

    # Log10 error
    log10_err = np.mean(np.abs(np.log10(pred_valid) - np.log10(gt_valid)))

    return DepthMetrics(
        abs_rel=float(abs_rel),
        sq_rel=float(sq_rel),
        rmse=float(rmse),
        rmse_log=float(rmse_log),
        delta_1=float(delta_1),
        delta_2=float(delta_2),
        delta_3=float(delta_3),
        silog=float(silog),
        log10=float(log10_err),
        n_valid=n_valid,
    )


def align_depth_least_squares(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """Align predicted depth to ground truth using least squares (scale + shift).

    Finds optimal scale (s) and shift (t) such that: aligned = s * pred + t
    minimizes the squared error with ground truth.

    Args:
        pred: Predicted depth map (H, W).
        gt: Ground truth depth map (H, W).
        mask: Optional validity mask.

    Returns:
        Tuple of (aligned_depth, scale, shift).
    """
    if mask is None:
        mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)

    pred_valid = pred[mask].flatten()
    gt_valid = gt[mask].flatten()

    if len(pred_valid) < 2:
        return pred, 1.0, 0.0

    # Solve least squares: [pred, 1] @ [s, t].T = gt
    A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
    result = np.linalg.lstsq(A, gt_valid, rcond=None)
    scale, shift = result[0]

    aligned = scale * pred + shift
    return aligned, float(scale), float(shift)


def align_depth_scale(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Align predicted depth to ground truth using median scaling.

    Args:
        pred: Predicted depth map (H, W).
        gt: Ground truth depth map (H, W).
        mask: Optional validity mask.

    Returns:
        Tuple of (aligned_depth, scale).
    """
    if mask is None:
        mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0)

    pred_valid = pred[mask]
    gt_valid = gt[mask]

    if len(pred_valid) == 0:
        return pred, 1.0

    scale = np.median(gt_valid) / np.median(pred_valid)
    aligned = pred * scale

    return aligned, float(scale)


class MetricsAccumulator:
    """Accumulates metrics over multiple samples for computing averages."""

    def __init__(self, metric_names: Optional[List[str]] = None):
        """Initialize accumulator.

        Args:
            metric_names: List of metric names to track. If None, uses all standard metrics.
        """
        if metric_names is None:
            metric_names = [
                "abs_rel",
                "sq_rel",
                "rmse",
                "rmse_log",
                "delta_1",
                "delta_2",
                "delta_3",
                "silog",
                "log10",
            ]

        self.metric_names = metric_names
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self._sums = {name: 0.0 for name in self.metric_names}
        self._counts = {name: 0 for name in self.metric_names}
        self._all_values = {name: [] for name in self.metric_names}

    def update(self, metrics: DepthMetrics):
        """Add metrics from a single sample."""
        if metrics.n_valid == 0:
            return

        metrics_dict = metrics.to_dict()
        for name in self.metric_names:
            if name in metrics_dict:
                value = metrics_dict[name]
                self._sums[name] += value
                self._counts[name] += 1
                self._all_values[name].append(value)

    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        result = {}
        for name in self.metric_names:
            if self._counts[name] > 0:
                result[name] = self._sums[name] / self._counts[name]
            else:
                result[name] = float("nan")
        result["n_samples"] = max(self._counts.values()) if self._counts else 0
        return result

    def compute_std(self) -> Dict[str, float]:
        """Compute standard deviation of metrics."""
        result = {}
        for name in self.metric_names:
            values = self._all_values[name]
            if len(values) > 1:
                result[f"{name}_std"] = float(np.std(values))
            else:
                result[f"{name}_std"] = 0.0
        return result
