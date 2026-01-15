"""Model registry for dynamic model loading."""

from typing import Any, Dict, Type

from .base import BaseDepthModel

# Global registry
MODEL_REGISTRY: Dict[str, Type[BaseDepthModel]] = {}


def register_model(name: str):
    """Decorator to register a model class.

    Args:
        name: Name to register the model under.

    Example:
        @register_model("my_model")
        class MyModel(BaseDepthModel):
            ...
    """

    def decorator(cls: Type[BaseDepthModel]) -> Type[BaseDepthModel]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs: Any) -> BaseDepthModel:
    """Get a model instance by name.

    Args:
        name: Registered model name.
        **kwargs: Arguments to pass to the model constructor.

    Returns:
        Model instance.

    Raises:
        ValueError: If model is not registered.
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available models: {available}")
    return MODEL_REGISTRY[name](**kwargs)
