"""Abstract video source interface.

Defines the common interface for video sources (files, RTSP streams).
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Self

import numpy as np


@dataclass
class Frame:
    """A single decoded video frame with metadata."""

    image: np.ndarray[Any, np.dtype[Any]]  # Shape (H, W, 3), dtype uint8, BGR format
    timestamp_seconds: float  # Timestamp in camera's clock (seconds since epoch)
    frame_number: int  # Sequential frame number from source (0-indexed)
    camera_id: str  # Camera identifier

    def corrected_timestamp(self, clock_offset_seconds: float) -> float:
        """Get timestamp with clock offset correction applied.

        Args:
            clock_offset_seconds: Offset to add to timestamp

        Returns:
            Corrected timestamp (seconds since epoch)
        """
        return self.timestamp_seconds + clock_offset_seconds


class VideoSource(ABC):
    """Abstract base class for video sources."""

    def __init__(self, camera_id: str):
        """Initialize video source.

        Args:
            camera_id: Identifier for the camera
        """
        self.camera_id = camera_id
        self._is_open = False

    @abstractmethod
    def open(self) -> None:
        """Open the video source.

        Raises:
            IOError: If source cannot be opened
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the video source and release resources."""
        pass

    @abstractmethod
    def read_frame(self) -> Frame | None:
        """Read next frame from source.

        Returns:
            Frame if available, None if end of stream or error
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Get frames per second of the video source.

        Returns:
            FPS value, or 0 if unknown
        """
        pass

    @abstractmethod
    def get_frame_count(self) -> int | None:
        """Get total frame count if known (for files).

        Returns:
            Frame count or None for streams
        """
        pass

    def is_open(self) -> bool:
        """Check if source is currently open."""
        return self._is_open

    def frames(self) -> Iterator[Frame]:
        """Iterate over all frames from the source.

        Yields:
            Frame objects until end of stream

        Raises:
            RuntimeError: If source not opened
        """
        if not self._is_open:
            raise RuntimeError(f"Video source {self.camera_id} not opened")

        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def __enter__(self) -> Self:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Context manager exit."""
        self.close()
        return False
