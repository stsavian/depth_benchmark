"""SegFormer model wrapper for semantic segmentation benchmark."""

from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from .segmentation_base import BaseSegmentationModel
from .segmentation_registry import register_segmentation_model


# Cityscapes class names (19 classes)
CITYSCAPES_CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic_light",
    7: "traffic_sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}


@register_segmentation_model("segformer")
@register_segmentation_model("segformer-b0-cityscapes")
@register_segmentation_model("segformer-b5-cityscapes")
class SegFormerWrapper(BaseSegmentationModel):
    """Wrapper for SegFormer semantic segmentation model.

    Uses HuggingFace transformers implementation with Cityscapes-trained models.
    """

    PRETRAINED_MODELS = {
        "segformer": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        "segformer-b0-cityscapes": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        "segformer-b5-cityscapes": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    }

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "segformer-b5-cityscapes",
        use_fp16: bool = True,
    ):
        """Initialize SegFormer wrapper.

        Args:
            device: Device to run inference on.
            model_name: Name of pretrained model to load.
            use_fp16: Whether to use FP16 inference.
        """
        super().__init__(device)
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.model = None
        self.processor = None
        self._num_classes = None
        self._class_names = None

    def load(self, checkpoint: Optional[str] = None) -> None:
        """Load SegFormer model from HuggingFace."""
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        model_path = checkpoint or self.PRETRAINED_MODELS.get(
            self.model_name, self.model_name
        )

        print(f"Loading SegFormer from: {model_path}")
        self.processor = SegformerImageProcessor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self.model = self.model.to(self.device)

        if self.use_fp16 and self.device == "cuda":
            self.model = self.model.half()

        self.model.eval()

        # Get class info from model config
        self._num_classes = self.model.config.num_labels
        self._class_names = getattr(
            self.model.config,
            "id2label",
            {i: f"class_{i}" for i in range(self._num_classes)},
        )

    def predict(self, image: Image.Image) -> np.ndarray:
        """Predict segmentation mask from RGB image.

        Returns:
            Segmentation mask (H, W) with class IDs.
        """
        if self.model is None:
            self.load()

        original_size = image.size  # (W, H)

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.use_fp16 and self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits  # (1, num_classes, H', W')

        # Upsample to original size
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(original_size[1], original_size[0]),  # (H, W)
            mode="bilinear",
            align_corners=False,
        )

        # Get class predictions
        pred_mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

        return pred_mask.astype(np.int32)

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
            self.load()
        return self._num_classes

    @property
    def class_names(self) -> Dict[int, str]:
        if self._class_names is None:
            self.load()
        return self._class_names

    @property
    def name(self) -> str:
        return f"SegFormer({self.model_name})"
