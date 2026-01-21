"""SkyScenes dataset loader for aerial depth estimation benchmark."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base import BaseDepthDataset
from .registry import register_dataset


# SkyScenes segmentation class definitions
# RGB color -> class ID mapping
SKYSCENES_COLOR_TO_CLASS = {
    (0, 0, 0): 0,           # unlabeled
    (70, 70, 70): 1,        # building
    (190, 153, 153): 2,     # fence
    (250, 170, 160): 3,     # other
    (220, 20, 60): 4,       # pedestrian
    (153, 153, 153): 5,     # pole
    (157, 234, 50): 6,      # road_line
    (128, 64, 128): 7,      # road
    (244, 35, 232): 8,      # sidewalk
    (107, 142, 35): 9,      # vegetation
    (0, 0, 142): 10,        # vehicle
    (102, 102, 156): 11,    # wall
    (220, 220, 0): 12,      # traffic_sign
    (70, 130, 180): 13,     # sky
    (150, 100, 100): 14,    # ground
    (150, 120, 90): 15,     # bridge
    (230, 150, 140): 16,    # rail_track
    (180, 165, 180): 17,    # guard_rail
    (250, 170, 30): 18,     # traffic_light
    (110, 190, 160): 19,    # static
    (170, 120, 50): 20,     # dynamic
    (45, 60, 150): 21,      # water
    (145, 170, 100): 22,    # terrain
}

# Ground-like classes for combined ground evaluation
SKYSCENES_GROUND_CLASSES = {7, 8, 14}  # road, sidewalk, ground


@register_dataset("skyscenes")
class SkyScenesDataset(BaseDepthDataset):
    """SkyScenes dataset for aerial/drone depth estimation.

    Folder structure:
        root/
            Images/
                H_{altitude}_P_{pitch}/
                    {weather}/
                        {town}/
                            {frame_id}_{weather_suffix}.png
            Depth/
                H_{altitude}_P_{pitch}/
                    {weather}/
                        {town}/
                            {frame_id}_depth.png
            Segment/  (only for ClearNoon weather)
                H_{altitude}_P_{pitch}/
                    ClearNoon/
                        {town}/
                            {frame_id}_semantic_segmentation.png

    Altitude and pitch are encoded in folder names:
        - H_15, H_35, H_60 = altitude in meters
        - P_0, P_45, P_60, P_90 = pitch angle in degrees (0=horizontal, 90=nadir)
    """

    # Depth is stored as 8-bit grayscale (linearized from CARLA 24-bit RGB encoding)
    # Reference: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
    # https://huggingface.co/datasets/hoffman-lab/SkyScenes/discussions/1
    #
    # Empirically verified scale by comparing depth values with known drone altitudes:
    # - H_15_P_90 (nadir from 15m): median depth ~25m
    # - H_35_P_90 (nadir from 35m): median depth ~41m
    # - H_60_P_90 (nadir from 60m): median depth ~51m
    # This suggests max_depth ~100m (pixel 255 = 100m or invalid/sky)
    DEPTH_SCALE = 100.0 / 255.0  # depth_meters = pixel_value * DEPTH_SCALE (~0.39)
    DEPTH_MAX_METERS = 100.0  # Maximum depth in meters
    DEPTH_INVALID = 255  # Pixel value for invalid/sky regions

    def __init__(
        self,
        root: str,
        split: str = "test",
        altitudes: Optional[List[int]] = None,
        pitches: Optional[List[int]] = None,
        weathers: Optional[List[str]] = None,
        towns: Optional[List[str]] = None,
        transform: Optional[Any] = None,
        depth_transform: Optional[Any] = None,
        max_samples: Optional[int] = None,
        min_depth: float = 0.1,
        max_depth: float = 1000.0,
        load_segmentation: bool = False,
    ):
        """Initialize SkyScenes dataset.

        Args:
            root: Root directory containing Images/ and Depth/ folders.
            split: Not used for SkyScenes (no predefined splits).
            altitudes: List of altitudes to include (e.g., [15, 35, 60]).
            pitches: List of pitch angles to include (e.g., [0, 45, 60, 90]).
            weathers: List of weather conditions to include (e.g., ["ClearNoon"]).
            towns: List of towns to include (e.g., ["Town01"]).
            transform: Transform for RGB images.
            depth_transform: Transform for depth maps.
            max_samples: Maximum samples to load.
            min_depth: Minimum valid depth in meters.
            max_depth: Maximum valid depth in meters.
            load_segmentation: If True, also load segmentation masks.
                              Note: Segmentation only available for ClearNoon weather.
        """
        self.altitudes = altitudes
        self.pitches = pitches
        self.load_segmentation = load_segmentation
        self._min_depth = min_depth
        self._max_depth = max_depth

        # Segmentation only available for ClearNoon
        if load_segmentation:
            if weathers is not None and "ClearNoon" not in weathers:
                raise ValueError(
                    "Segmentation masks only available for ClearNoon weather. "
                    "Either set weathers=None or include 'ClearNoon'."
                )
            # Force ClearNoon only when loading segmentation
            weathers = ["ClearNoon"]

        self.weathers = weathers
        self.towns = towns

        super().__init__(root, split, transform, depth_transform, max_samples)

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Find all RGB-depth pairs with metadata."""
        images_root = self.root / "Images"
        depth_root = self.root / "Depth"

        if not images_root.exists():
            raise ValueError(f"Images directory not found: {images_root}")
        if not depth_root.exists():
            raise ValueError(f"Depth directory not found: {depth_root}")

        samples = []

        # Parse H_{alt}_P_{pitch} folders
        for hp_folder in sorted(images_root.iterdir()):
            if not hp_folder.is_dir():
                continue

            # Parse altitude and pitch from folder name
            match = re.match(r"H_(\d+)_P_(\d+)", hp_folder.name)
            if not match:
                continue

            altitude = int(match.group(1))
            pitch = int(match.group(2))

            # Filter by altitude/pitch if specified
            if self.altitudes is not None and altitude not in self.altitudes:
                continue
            if self.pitches is not None and pitch not in self.pitches:
                continue

            # Traverse weather conditions
            for weather_folder in sorted(hp_folder.iterdir()):
                if not weather_folder.is_dir():
                    continue

                weather = weather_folder.name

                if self.weathers is not None and weather not in self.weathers:
                    continue

                # Traverse towns
                for town_folder in sorted(weather_folder.iterdir()):
                    if not town_folder.is_dir():
                        continue

                    town = town_folder.name

                    if self.towns is not None and town not in self.towns:
                        continue

                    # Find all images in this folder
                    for img_file in sorted(town_folder.glob("*.png")):
                        # Extract frame ID
                        frame_match = re.match(r"(\d+)_", img_file.name)
                        if not frame_match:
                            continue

                        frame_id = frame_match.group(1)

                        # Find corresponding depth file
                        depth_file = (
                            depth_root
                            / hp_folder.name
                            / weather
                            / town
                            / f"{frame_id}_depth.png"
                        )

                        if depth_file.exists():
                            samples.append(
                                {
                                    "rgb_path": img_file,
                                    "depth_path": depth_file,
                                    "altitude": altitude,
                                    "pitch": pitch,
                                    "weather": weather,
                                    "town": town,
                                    "frame_id": frame_id,
                                }
                            )

        if len(samples) == 0:
            raise ValueError(f"No samples found in {self.root}")

        return samples

    def _load_rgb(self, index: int) -> Image.Image:
        return Image.open(self.samples[index]["rgb_path"]).convert("RGB")

    def _load_depth(self, index: int) -> np.ndarray:
        """Load depth map from 8-bit grayscale PNG.

        Converts pixel values to meters using CARLA-derived formula:
        - depth_meters = pixel_value * (1000/255)
        - pixel_value 255 = ~1000m (max range / sky, marked as 0 for invalid)
        """
        depth_path = self.samples[index]["depth_path"]
        depth_img = Image.open(depth_path)

        # Convert to numpy (keep original pixel values for threshold check)
        depth_raw = np.array(depth_img, dtype=np.float32)

        # Convert to meters using CARLA-derived scale factor
        depth = depth_raw * self.DEPTH_SCALE

        # Mark invalid regions (pixel value 255 = sky/max range) as 0
        depth[depth_raw >= self.DEPTH_INVALID] = 0.0

        return depth

    def _load_segmentation(self, index: int) -> Optional[np.ndarray]:
        """Load segmentation mask for the given index.

        Returns:
            Segmentation mask as numpy array (H, W) with class IDs.
            Returns None if segmentation not available or not requested.
        """
        if not self.load_segmentation:
            return None

        sample = self.samples[index]

        # Construct segmentation path
        # Segment/H_{alt}_P_{pitch}/ClearNoon/{town}/{frame_id}_semsegCarla_clrnoon.png
        seg_path = (
            self.root / "Segment"
            / f"H_{sample['altitude']}_P_{sample['pitch']}"
            / "ClearNoon"
            / sample['town']
            / f"{sample['frame_id']}_semsegCarla_clrnoon.png"
        )

        if not seg_path.exists():
            return None

        # Load as RGB and convert to class IDs
        seg_img = Image.open(seg_path).convert("RGB")
        seg_array = np.array(seg_img)

        # Convert RGB to class ID using color lookup
        class_mask = self._rgb_to_class_id(seg_array)

        return class_mask

    def _rgb_to_class_id(self, rgb_array: np.ndarray) -> np.ndarray:
        """Convert RGB segmentation image to class ID mask.

        Args:
            rgb_array: RGB image (H, W, 3).

        Returns:
            Class ID mask (H, W).
        """
        h, w = rgb_array.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.int32)

        for color, class_id in SKYSCENES_COLOR_TO_CLASS.items():
            mask = np.all(rgb_array == color, axis=-1)
            class_mask[mask] = class_id

        return class_mask

    def get_metadata(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {
            "rgb_path": str(sample["rgb_path"]),
            "depth_path": str(sample["depth_path"]),
            "altitude": sample["altitude"],
            "pitch": sample["pitch"],
            "weather": sample["weather"],
            "town": sample["town"],
            "frame_id": sample["frame_id"],
        }

    @property
    def depth_range(self) -> Tuple[float, float]:
        return (self._min_depth, self._max_depth)

    def get_samples_by_condition(
        self,
        altitude: Optional[int] = None,
        pitch: Optional[int] = None,
    ) -> List[int]:
        """Get indices of samples matching given conditions.

        Args:
            altitude: Filter by altitude (meters).
            pitch: Filter by pitch angle (degrees).

        Returns:
            List of sample indices matching the conditions.
        """
        indices = []
        for i, sample in enumerate(self.samples):
            if altitude is not None and sample["altitude"] != altitude:
                continue
            if pitch is not None and sample["pitch"] != pitch:
                continue
            indices.append(i)
        return indices

    def get_available_conditions(self) -> Dict[str, List]:
        """Get all unique altitude/pitch combinations in the dataset."""
        altitudes = set()
        pitches = set()
        weathers = set()
        towns = set()

        for sample in self.samples:
            altitudes.add(sample["altitude"])
            pitches.add(sample["pitch"])
            weathers.add(sample["weather"])
            towns.add(sample["town"])

        return {
            "altitudes": sorted(altitudes),
            "pitches": sorted(pitches),
            "weathers": sorted(weathers),
            "towns": sorted(towns),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - 'rgb': RGB image as tensor (C, H, W)
                - 'depth': Ground truth depth map as tensor (H, W)
                - 'mask': Validity mask as tensor (H, W)
                - 'metadata': Additional metadata dictionary
                - 'index': Sample index
                - 'segmentation': (optional) Segmentation mask as tensor (H, W)
        """
        # Call parent implementation
        result = super().__getitem__(index)

        # Add segmentation if requested
        if self.load_segmentation:
            segmentation = self._load_segmentation(index)
            if segmentation is not None:
                result["segmentation"] = torch.from_numpy(segmentation).long()

        return result
