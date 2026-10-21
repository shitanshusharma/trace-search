"""Camera registry and configuration.

Manages camera metadata: id, brand, site-map position, and clock offsets.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    camera_id: str
    brand: str  # e.g., "hikvision", "cpplus"
    site_map_x: float  # Position on site map (meters or normalized coords)
    site_map_y: float
    clock_offset_seconds: float = 0.0  # Correction to apply to camera timestamps

    def __post_init__(self):
        """Validate camera config."""
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        if not self.brand:
            raise ValueError("brand cannot be empty")

    def apply_offset(self, timestamp_seconds: float) -> float:
        """Apply clock offset correction to a camera timestamp.

        Args:
            timestamp_seconds: Raw timestamp from camera (seconds since epoch)

        Returns:
            Corrected timestamp (seconds since epoch)
        """
        return timestamp_seconds + self.clock_offset_seconds

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "camera_id": self.camera_id,
            "brand": self.brand,
            "site_map_x": self.site_map_x,
            "site_map_y": self.site_map_y,
            "clock_offset_seconds": self.clock_offset_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraConfig":
        """Create from dictionary."""
        return cls(**data)


class CameraRegistry:
    """Registry of camera configurations."""

    def __init__(self):
        self._cameras: dict[str, CameraConfig] = {}

    def register(self, camera: CameraConfig) -> None:
        """Register a camera configuration.

        Args:
            camera: Camera configuration to register

        Raises:
            ValueError: If camera_id already registered
        """
        if camera.camera_id in self._cameras:
            raise ValueError(f"Camera {camera.camera_id} already registered")
        self._cameras[camera.camera_id] = camera

    def get(self, camera_id: str) -> CameraConfig | None:
        """Get camera configuration by ID.

        Args:
            camera_id: Camera identifier

        Returns:
            Camera configuration or None if not found
        """
        return self._cameras.get(camera_id)

    def list_cameras(self) -> list[CameraConfig]:
        """List all registered cameras."""
        return list(self._cameras.values())

    def count(self) -> int:
        """Return number of registered cameras."""
        return len(self._cameras)

    def save(self, path: Path) -> None:
        """Save registry to JSON file.

        Args:
            path: Path to save JSON file
        """
        data = {"cameras": [cam.to_dict() for cam in self._cameras.values()]}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CameraRegistry":
        """Load registry from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Loaded camera registry
        """
        with open(path) as f:
            data = json.load(f)

        registry = cls()
        for cam_data in data["cameras"]:
            registry.register(CameraConfig.from_dict(cam_data))
        return registry
