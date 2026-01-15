"""Depth estimation model wrappers."""

from .base import BaseDepthModel
from .registry import MODEL_REGISTRY, register_model, get_model
from .moge import MoGeWrapper

__all__ = [
    "BaseDepthModel",
    "MODEL_REGISTRY",
    "register_model",
    "get_model",
    "MoGeWrapper",
]
