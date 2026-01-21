"""Base model interface for semantic segmentation."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
from PIL import Image


class BaseSegmentationModel(ABC):
    """Abstract base class for semantic segmentation models.

    All segmentation model wrappers should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize the model.

        Args:
            device: Device to run inference on ('cuda' or 'cpu').
        """
        self.device = device

    @abstractmethod
    def load(self, checkpoint: Optional[str] = None) -> None:
        """Load model weights.

        Args:
            checkpoint: Path to model checkpoint. If None, load default weights.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> np.ndarray:
        """Predict segmentation from a single RGB image.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Segmentation mask as numpy array (H, W) with class IDs.
        """
        pass

    def predict_batch(self, images: list) -> list:
        """Predict segmentation for a batch of images.

        Default implementation processes images one at a time.
        Override for more efficient batch processing.

        Args:
            images: List of PIL Images in RGB format.

        Returns:
            List of segmentation masks as numpy arrays.
        """
        return [self.predict(img) for img in images]

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        """Mapping from class ID to class name."""
        pass

    @property
    def name(self) -> str:
        """Return the model name."""
        return self.__class__.__name__

    def to(self, device: str) -> "BaseSegmentationModel":
        """Move model to specified device."""
        self.device = device
        return self

    def __repr__(self) -> str:
        return f"{self.name}(device={self.device}, num_classes={self.num_classes})"
