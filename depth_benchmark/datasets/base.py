"""Base dataset class for depth estimation benchmarks."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDepthDataset(Dataset, ABC):
    """Abstract base class for depth estimation datasets.

    All custom datasets should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: Optional[Any] = None,
        depth_transform: Optional[Any] = None,
        max_samples: Optional[int] = None,
    ):
        """Initialize the dataset.

        Args:
            root: Root directory of the dataset.
            split: Dataset split ('train', 'val', 'test').
            transform: Transform to apply to RGB images.
            depth_transform: Transform to apply to depth maps.
            max_samples: Maximum number of samples to use (None = all).
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.depth_transform = depth_transform
        self.max_samples = max_samples

        # Load file lists
        self.samples = self._load_samples()

        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    @abstractmethod
    def _load_samples(self) -> list:
        """Load the list of samples for this split.

        Returns:
            List of sample identifiers (paths, indices, etc.)
        """
        pass

    @abstractmethod
    def _load_rgb(self, index: int) -> Image.Image:
        """Load RGB image for the given index.

        Args:
            index: Sample index.

        Returns:
            PIL Image in RGB format.
        """
        pass

    @abstractmethod
    def _load_depth(self, index: int) -> np.ndarray:
        """Load depth map for the given index.

        Args:
            index: Sample index.

        Returns:
            Depth map as numpy array (H, W) in meters.
        """
        pass

    def _load_mask(self, index: int) -> Optional[np.ndarray]:
        """Load validity mask for the given index.

        Override this method if your dataset has explicit validity masks.

        Args:
            index: Sample index.

        Returns:
            Binary mask as numpy array (H, W) where 1 = valid, 0 = invalid.
            Returns None if no explicit mask is available.
        """
        return None

    def get_metadata(self, index: int) -> Dict[str, Any]:
        """Get metadata for the given sample.

        Override this method to provide additional metadata.

        Args:
            index: Sample index.

        Returns:
            Dictionary containing metadata (e.g., camera intrinsics, scene info).
        """
        return {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - 'rgb': RGB image as tensor (C, H, W)
                - 'depth': Ground truth depth map as tensor (H, W)
                - 'mask': Validity mask as tensor (H, W)
                - 'metadata': Additional metadata dictionary
        """
        # Load data
        rgb = self._load_rgb(index)
        depth = self._load_depth(index)
        mask = self._load_mask(index)
        metadata = self.get_metadata(index)

        # Create validity mask from depth if not provided
        if mask is None:
            mask = (depth > 0) & np.isfinite(depth)

        # Apply transforms
        if self.transform is not None:
            rgb = self.transform(rgb)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0

        if self.depth_transform is not None:
            depth = self.depth_transform(depth)
        else:
            depth = torch.from_numpy(depth).float()

        mask = torch.from_numpy(mask.astype(np.float32))

        return {
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "metadata": metadata,
            "index": index,
        }

    @property
    def depth_range(self) -> Tuple[float, float]:
        """Return the typical depth range for this dataset.

        Override this method if your dataset has a specific depth range.

        Returns:
            Tuple of (min_depth, max_depth) in meters.
        """
        return (0.001, 80.0)

    @property
    def name(self) -> str:
        """Return the dataset name."""
        return self.__class__.__name__
