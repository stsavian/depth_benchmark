"""Base model interface for depth estimation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class BaseDepthModel(ABC):
    """Abstract base class for depth estimation models.

    All model wrappers should inherit from this class and implement
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
        """Predict depth from a single RGB image.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Depth map as numpy array (H, W) in meters (or relative depth).
        """
        pass

    def predict_batch(self, images: list) -> list:
        """Predict depth for a batch of images.

        Default implementation processes images one at a time.
        Override for more efficient batch processing.

        Args:
            images: List of PIL Images in RGB format.

        Returns:
            List of depth maps as numpy arrays.
        """
        return [self.predict(img) for img in images]

    @property
    def is_metric(self) -> bool:
        """Whether the model predicts metric depth (meters) or relative depth.

        Override this property if the model predicts metric depth.
        """
        return False

    @property
    def name(self) -> str:
        """Return the model name."""
        return self.__class__.__name__

    def to(self, device: str) -> "BaseDepthModel":
        """Move model to specified device."""
        self.device = device
        return self

    def __repr__(self) -> str:
        return f"{self.name}(device={self.device}, metric={self.is_metric})"
