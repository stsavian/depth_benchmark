"""SkyScenes dataset loader for aerial depth estimation benchmark."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import BaseDepthDataset
from .registry import register_dataset


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

    Altitude and pitch are encoded in folder names:
        - H_15, H_35, H_60 = altitude in meters
        - P_0, P_45, P_60, P_90 = pitch angle in degrees (0=horizontal, 90=nadir)
    """

    # Depth is stored as 8-bit grayscale
    # Based on empirical analysis of the dataset:
    # - Lower pixel values = closer distances
    # - Pixel value 255 = invalid (sky/far)
    # - Approximate scale: depth_meters ≈ pixel_value * 0.5 to 0.6
    # - Implied max depth ~150m for pixel value 255
    # Using scale factor of 0.55 as middle ground
    DEPTH_SCALE = 0.55  # depth_meters = pixel_value * DEPTH_SCALE
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
        """
        self.altitudes = altitudes
        self.pitches = pitches
        self.weathers = weathers
        self.towns = towns
        self._min_depth = min_depth
        self._max_depth = max_depth

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

        Based on empirical analysis of SkyScenes:
        - depth_meters = pixel_value * DEPTH_SCALE (~0.55)
        - pixel_value 255 = invalid (sky/far regions)
        """
        depth_path = self.samples[index]["depth_path"]
        depth_img = Image.open(depth_path)

        # Convert to numpy
        depth = np.array(depth_img, dtype=np.float32)

        # Convert to meters using empirical scale factor
        depth = depth * self.DEPTH_SCALE

        # Mark invalid regions (originally 255) as 0
        depth[depth >= self.DEPTH_INVALID * self.DEPTH_SCALE * 0.99] = 0.0

        return depth

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
