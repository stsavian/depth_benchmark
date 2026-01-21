"""Model registry for dynamic segmentation model loading."""

from typing import Any, Dict, Type

from .segmentation_base import BaseSegmentationModel

# Global registry (separate from depth models)
SEGMENTATION_MODEL_REGISTRY: Dict[str, Type[BaseSegmentationModel]] = {}


def register_segmentation_model(name: str):
    """Decorator to register a segmentation model class.

    Args:
        name: Name to register the model under.

    Example:
        @register_segmentation_model("segformer")
        class SegFormerWrapper(BaseSegmentationModel):
            ...
    """

    def decorator(cls: Type[BaseSegmentationModel]) -> Type[BaseSegmentationModel]:
        if name in SEGMENTATION_MODEL_REGISTRY:
            raise ValueError(f"Segmentation model '{name}' is already registered.")
        SEGMENTATION_MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_segmentation_model(name: str, **kwargs: Any) -> BaseSegmentationModel:
    """Get a segmentation model instance by name.

    Args:
        name: Registered model name.
        **kwargs: Arguments to pass to the model constructor.

    Returns:
        Model instance.

    Raises:
        ValueError: If model is not registered.
    """
    if name not in SEGMENTATION_MODEL_REGISTRY:
        available = list(SEGMENTATION_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Segmentation model '{name}' not found. Available models: {available}"
        )
    return SEGMENTATION_MODEL_REGISTRY[name](**kwargs)
