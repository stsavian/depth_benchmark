"""Dataset implementations for depth benchmark."""

from .base import BaseDepthDataset
from .registry import DATASET_REGISTRY, register_dataset, get_dataset
from .skyscenes import SkyScenesDataset
from .custom import CustomDepthDataset

__all__ = [
    "BaseDepthDataset",
    "DATASET_REGISTRY",
    "register_dataset",
    "get_dataset",
    "SkyScenesDataset",
    "CustomDepthDataset",
]
