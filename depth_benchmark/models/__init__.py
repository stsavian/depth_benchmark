"""Depth estimation and segmentation model wrappers."""

# Depth models
from .base import BaseDepthModel
from .registry import MODEL_REGISTRY, register_model, get_model
from .moge import MoGeWrapper

# Segmentation models
from .segmentation_base import BaseSegmentationModel
from .segmentation_registry import (
    SEGMENTATION_MODEL_REGISTRY,
    register_segmentation_model,
    get_segmentation_model,
)
from .segformer import SegFormerWrapper

__all__ = [
    # Depth
    "BaseDepthModel",
    "MODEL_REGISTRY",
    "register_model",
    "get_model",
    "MoGeWrapper",
    # Segmentation
    "BaseSegmentationModel",
    "SEGMENTATION_MODEL_REGISTRY",
    "register_segmentation_model",
    "get_segmentation_model",
    "SegFormerWrapper",
]
