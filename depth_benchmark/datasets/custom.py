"""Custom dataset implementation for user-defined folder structures."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import BaseDepthDataset
from .registry import register_dataset


@register_dataset("custom")
class CustomDepthDataset(BaseDepthDataset):
    """Flexible dataset for custom folder structures.

    Expected folder structure:
        root/
            rgb/
                image_001.png
                image_002.png
                ...
            depth/
                image_001.png  (or .npy, .exr, .pfm)
                image_002.png
                ...

    Depth maps can be:
        - PNG: 16-bit depth in millimeters
        - NPY: Numpy array in meters
        - EXR/PFM: Float depth in meters
    """

    SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    SUPPORTED_DEPTH_EXTENSIONS = {".png", ".npy", ".exr", ".pfm"}

    def __init__(
        self,
        root: str,
        split: str = "test",
        rgb_dir: str = "rgb",
        depth_dir: str = "depth",
        depth_scale: float = 1000.0,  # Scale factor: depth_meters = depth_raw / depth_scale
        transform: Optional[Any] = None,
        depth_transform: Optional[Any] = None,
        max_samples: Optional[int] = None,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
    ):
        """Initialize custom dataset.

        Args:
            root: Root directory of the dataset.
            split: Dataset split (used as subdirectory if exists).
            rgb_dir: Subdirectory containing RGB images.
            depth_dir: Subdirectory containing depth maps.
            depth_scale: Scale factor for PNG depth maps (depth_m = raw / scale).
            transform: Transform to apply to RGB images.
            depth_transform: Transform to apply to depth maps.
            max_samples: Maximum number of samples to use.
            min_depth: Minimum valid depth in meters.
            max_depth: Maximum valid depth in meters.
        """
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.depth_scale = depth_scale
        self._min_depth = min_depth
        self._max_depth = max_depth

        super().__init__(root, split, transform, depth_transform, max_samples)

    def _load_samples(self) -> List[Tuple[Path, Path]]:
        """Find matching RGB and depth pairs."""
        # Try split-specific directory first
        split_root = self.root / self.split
        if split_root.exists():
            rgb_path = split_root / self.rgb_dir
            depth_path = split_root / self.depth_dir
        else:
            rgb_path = self.root / self.rgb_dir
            depth_path = self.root / self.depth_dir

        if not rgb_path.exists():
            raise ValueError(f"RGB directory not found: {rgb_path}")
        if not depth_path.exists():
            raise ValueError(f"Depth directory not found: {depth_path}")

        # Find all RGB images
        rgb_files = []
        for ext in self.SUPPORTED_IMAGE_EXTENSIONS:
            rgb_files.extend(rgb_path.glob(f"*{ext}"))
            rgb_files.extend(rgb_path.glob(f"*{ext.upper()}"))

        # Match with depth files
        samples = []
        for rgb_file in sorted(rgb_files):
            stem = rgb_file.stem

            # Try to find matching depth file
            depth_file = None
            for ext in self.SUPPORTED_DEPTH_EXTENSIONS:
                candidate = depth_path / f"{stem}{ext}"
                if candidate.exists():
                    depth_file = candidate
                    break

            if depth_file is not None:
                samples.append((rgb_file, depth_file))

        if len(samples) == 0:
            raise ValueError(
                f"No matching RGB-depth pairs found in {self.root}"
            )

        return samples

    def _load_rgb(self, index: int) -> Image.Image:
        rgb_path, _ = self.samples[index]
        return Image.open(rgb_path).convert("RGB")

    def _load_depth(self, index: int) -> np.ndarray:
        _, depth_path = self.samples[index]

        if depth_path.suffix == ".npy":
            depth = np.load(depth_path)
        elif depth_path.suffix == ".png":
            depth = np.array(Image.open(depth_path))
            # Handle 16-bit PNG (millimeters to meters)
            depth = depth.astype(np.float32) / self.depth_scale
        elif depth_path.suffix == ".exr":
            depth = self._load_exr(depth_path)
        elif depth_path.suffix == ".pfm":
            depth = self._load_pfm(depth_path)
        else:
            raise ValueError(f"Unsupported depth format: {depth_path.suffix}")

        return depth.astype(np.float32)

    def _load_exr(self, path: Path) -> np.ndarray:
        """Load depth from EXR file."""
        try:
            import OpenEXR
            import Imath

            exr_file = OpenEXR.InputFile(str(path))
            header = exr_file.header()
            dw = header["dataWindow"]
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

            # Read the depth channel (usually 'Y' or 'Z')
            for channel in ["Y", "Z", "R"]:
                if channel in header["channels"]:
                    pt = Imath.PixelType(Imath.PixelType.FLOAT)
                    depth_str = exr_file.channel(channel, pt)
                    depth = np.frombuffer(depth_str, dtype=np.float32)
                    return depth.reshape(size[1], size[0])

            raise ValueError("No depth channel found in EXR file")
        except ImportError:
            raise ImportError(
                "OpenEXR is required for EXR files: pip install openexr"
            )

    def _load_pfm(self, path: Path) -> np.ndarray:
        """Load depth from PFM file."""
        with open(path, "rb") as f:
            header = f.readline().decode("ascii").strip()
            if header == "Pf":
                color = False
            elif header == "PF":
                color = True
            else:
                raise ValueError(f"Invalid PFM header: {header}")

            dims = f.readline().decode("ascii").strip()
            width, height = map(int, dims.split())

            scale = float(f.readline().decode("ascii").strip())
            endian = "<" if scale < 0 else ">"
            scale = abs(scale)

            data = np.frombuffer(f.read(), endian + "f")

            if color:
                data = data.reshape((height, width, 3))
                data = data[:, :, 0]  # Take first channel
            else:
                data = data.reshape((height, width))

            # Flip vertically (PFM stores bottom-to-top)
            data = np.flipud(data)

            return data * scale

    def get_metadata(self, index: int) -> Dict[str, Any]:
        rgb_path, depth_path = self.samples[index]
        return {
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "sample_id": rgb_path.stem,
        }

    @property
    def depth_range(self) -> Tuple[float, float]:
        return (self._min_depth, self._max_depth)
