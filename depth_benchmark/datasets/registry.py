"""Dataset registry for dynamic dataset loading."""

from typing import Any, Dict, Type

from .base import BaseDepthDataset

# Global registry
DATASET_REGISTRY: Dict[str, Type[BaseDepthDataset]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset class.

    Args:
        name: Name to register the dataset under.

    Example:
        @register_dataset("my_dataset")
        class MyDataset(BaseDepthDataset):
            ...
    """

    def decorator(cls: Type[BaseDepthDataset]) -> Type[BaseDepthDataset]:
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered.")
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset(name: str, **kwargs: Any) -> BaseDepthDataset:
    """Get a dataset instance by name.

    Args:
        name: Registered dataset name.
        **kwargs: Arguments to pass to the dataset constructor.

    Returns:
        Dataset instance.

    Raises:
        ValueError: If dataset is not registered.
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {available}"
        )
    return DATASET_REGISTRY[name](**kwargs)
