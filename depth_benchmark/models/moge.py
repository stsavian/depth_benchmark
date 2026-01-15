"""MoGe model wrapper for depth estimation benchmark."""

from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseDepthModel
from .registry import register_model


@register_model("moge")
@register_model("moge-vitl")
class MoGeWrapper(BaseDepthModel):
    """Wrapper for MoGe (Monocular Geometry Estimation) model.

    MoGe predicts metric depth from single images using a vision transformer.
    """

    # Available pretrained models
    PRETRAINED_MODELS = {
        "moge-vitl": "Ruicheng/moge-vitl",
        "moge-2-vitl": "Ruicheng/moge-2-vitl",
        "moge-2-vitl-normal": "Ruicheng/moge-2-vitl-normal",
        "moge-2-vitb-normal": "Ruicheng/moge-2-vitb-normal",
    }

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "moge-2-vitl",
        resolution_level: int = 9,
        use_fp16: bool = True,
    ):
        """Initialize MoGe wrapper.

        Args:
            device: Device to run inference on.
            model_name: Name of pretrained model to load.
            resolution_level: Resolution level for inference (0-9, higher = more tokens).
            use_fp16: Whether to use FP16 inference for speed.
        """
        super().__init__(device)
        self.model_name = model_name
        self.resolution_level = resolution_level
        self.use_fp16 = use_fp16
        self.model = None

    def load(self, checkpoint: Optional[str] = None) -> None:
        """Load MoGe model.

        Args:
            checkpoint: Path to checkpoint or HuggingFace model name.
                       If None, uses self.model_name.
        """
        from moge.model.v2 import MoGeModel

        if checkpoint is None:
            # Use default model name
            model_path = self.PRETRAINED_MODELS.get(
                self.model_name, self.model_name
            )
        else:
            model_path = checkpoint

        self.model = MoGeModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> np.ndarray:
        """Predict depth from RGB image.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Depth map as numpy array (H, W) in meters.
        """
        if self.model is None:
            self.load()

        # Convert PIL to tensor
        img_array = np.array(image)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model.infer(
                img_tensor,
                resolution_level=self.resolution_level,
                use_fp16=self.use_fp16,
            )

        # Extract depth map
        depth = output["depth"].cpu().numpy()

        # Handle invalid values (inf from masking)
        depth = np.where(np.isinf(depth), 0.0, depth)

        return depth

    def predict_with_metadata(self, image: Image.Image) -> dict:
        """Predict depth with additional outputs (intrinsics, normal, mask).

        Args:
            image: PIL Image in RGB format.

        Returns:
            Dictionary with depth, intrinsics, normal, mask, points.
        """
        if self.model is None:
            self.load()

        # Convert PIL to tensor
        img_array = np.array(image)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model.infer(
                img_tensor,
                resolution_level=self.resolution_level,
                use_fp16=self.use_fp16,
            )

        # Convert to numpy
        result = {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                arr = value.cpu().numpy()
                # Handle invalid values
                if key in ["depth", "points"]:
                    arr = np.where(np.isinf(arr), 0.0, arr)
                result[key] = arr
            else:
                result[key] = value

        return result

    @property
    def is_metric(self) -> bool:
        """MoGe v2 predicts metric depth."""
        return True

    @property
    def name(self) -> str:
        return f"MoGe({self.model_name})"

    def to(self, device: str) -> "MoGeWrapper":
        """Move model to specified device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self

    def predict_batch(self, images: list, batch_size: int = 4) -> list:
        """Predict depth for a batch of images efficiently.

        Args:
            images: List of PIL Images in RGB format.
            batch_size: Number of images to process at once.

        Returns:
            List of depth maps as numpy arrays.
        """
        if self.model is None:
            self.load()

        results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # Convert to tensors and stack
            tensors = []
            for img in batch_images:
                img_array = np.array(img)
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                tensors.append(img_tensor)

            # Stack into batch
            batch_tensor = torch.stack(tensors).to(self.device)

            # Run batch inference
            with torch.no_grad():
                output = self.model.infer(
                    batch_tensor,
                    resolution_level=self.resolution_level,
                    use_fp16=self.use_fp16,
                )

            # Extract depths
            depths = output["depth"].cpu().numpy()

            # Handle invalid values and add to results
            for j in range(depths.shape[0]):
                depth = depths[j]
                depth = np.where(np.isinf(depth), 0.0, depth)
                results.append(depth)

        return results
