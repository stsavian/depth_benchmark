"""RANSAC plane fitting for ground segmentation from depth."""

from typing import Dict, Optional

import numpy as np
from PIL import Image
from sklearn.linear_model import RANSACRegressor

from .segmentation_base import BaseSegmentationModel
from .segmentation_registry import register_segmentation_model


# SkyScenes ground class ID
SKYSCENES_GROUND = 14


@register_segmentation_model("ransac")
@register_segmentation_model("ransac-plane")
class RANSACPlaneSegmentation(BaseSegmentationModel):
    """Ground segmentation using RANSAC plane fitting on depth maps.

    Fits a plane to the depth map and classifies pixels close to the plane as ground.
    This is a geometry-based approach that doesn't require training data.
    """

    def __init__(
        self,
        device: str = "cpu",  # Not used, but kept for interface compatibility
        distance_threshold: float = 1.2,  # Distance threshold in meters
        sample_rate: float = 0.05,
        max_trials: int = 1000,
        invalid_depth_threshold: float = 0.99,
    ):
        """Initialize RANSAC plane segmentation.

        Args:
            device: Ignored (RANSAC runs on CPU).
            distance_threshold: Maximum distance from plane to be classified as ground (meters).
            sample_rate: Fraction of pixels to use for RANSAC fitting (for speed).
            max_trials: Maximum RANSAC iterations.
            invalid_depth_threshold: Fraction of max depth to consider invalid (sky).
        """
        super().__init__(device)
        self.distance_threshold = distance_threshold
        self.sample_rate = sample_rate
        self.max_trials = max_trials
        self.invalid_depth_threshold = invalid_depth_threshold

    def load(self, checkpoint: Optional[str] = None) -> None:
        """No loading needed for RANSAC."""
        pass

    def predict(self, image: Image.Image, depth: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict ground segmentation using RANSAC plane fitting.

        Args:
            image: PIL Image (not used, but required by interface).
            depth: Depth map as numpy array (H, W). Required for RANSAC.

        Returns:
            Segmentation mask (H, W) with ground=14, other=0.
        """
        if depth is None:
            raise ValueError("RANSAC segmentation requires depth map")

        h, w = depth.shape

        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Get valid depth pixels (exclude sky/invalid - marked as 0 in dataset)
        # Also exclude very high values close to max
        max_depth = depth.max()
        valid_mask = (depth > 0) & (depth < max_depth * self.invalid_depth_threshold)

        # Subsample for speed
        n_valid = valid_mask.sum()
        if n_valid == 0:
            return np.zeros((h, w), dtype=np.int32)

        n_sample = int(n_valid * self.sample_rate)
        n_sample = max(n_sample, 100)  # Minimum samples for fitting

        valid_indices = np.where(valid_mask.ravel())[0]
        if len(valid_indices) > n_sample:
            np.random.seed(42)
            sample_indices = np.random.choice(valid_indices, size=n_sample, replace=False)
        else:
            sample_indices = valid_indices

        # Get sampled points
        flat_x = x_coords.ravel()[sample_indices]
        flat_y = y_coords.ravel()[sample_indices]
        flat_depth = depth.ravel()[sample_indices]

        # Fit plane: depth = a*x + b*y + c
        X = np.column_stack([flat_x, flat_y])
        Y = flat_depth

        ransac = RANSACRegressor(
            residual_threshold=self.distance_threshold,
            max_trials=self.max_trials,
            random_state=42,
        )
        ransac.fit(X, Y)

        # Predict plane depth for all pixels
        X_all = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        plane_depth = ransac.predict(X_all).reshape(h, w)

        # Compute residuals (distance from plane)
        residuals = np.abs(depth - plane_depth)

        # Classify as ground if close to plane and valid depth
        ground_mask = (residuals < self.distance_threshold) & valid_mask

        # Create output segmentation
        seg = np.zeros((h, w), dtype=np.int32)
        seg[ground_mask] = SKYSCENES_GROUND

        return seg

    @property
    def num_classes(self) -> int:
        return 2  # Binary: ground vs non-ground

    @property
    def class_names(self) -> Dict[int, str]:
        return {0: "non-ground", SKYSCENES_GROUND: "ground"}

    @property
    def name(self) -> str:
        return f"RANSAC(thresh={self.distance_threshold})"
