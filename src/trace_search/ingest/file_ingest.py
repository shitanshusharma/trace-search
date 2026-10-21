"""File-based video ingestion.

Ingests video files from NVR exports (Hikvision, CP Plus, generic MP4/AVI).
Uses OpenCV (cv2) with FFmpeg LGPL backend.

IMPORTANT: Assumes constant frame rate (CFR) video. Timestamps are computed
as start_timestamp + (frame_number / fps). Variable frame rate (VFR) video
will have incorrect timestamps. For VFR support, use per-frame PTS from
FFmpeg metadata.
"""

import warnings
from pathlib import Path

try:
    import cv2
except ImportError:
    raise ImportError(
        "opencv-python-headless is required for video ingestion. "
        "Install with: pip install opencv-python-headless"
    )


from .video_source import Frame, VideoSource


class FileVideoSource(VideoSource):
    """Video source that reads from a file."""

    def __init__(
        self,
        camera_id: str,
        file_path: Path,
        start_timestamp: float | None = None,
    ):
        """Initialize file video source.

        Args:
            camera_id: Identifier for the camera
            file_path: Path to video file
            start_timestamp: Starting timestamp (seconds since epoch).
                If None, uses file modification time as approximation.

        Raises:
            FileNotFoundError: If file does not exist
        """
        super().__init__(camera_id)
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        self._start_timestamp = start_timestamp
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._current_frame_number: int = 0

    def open(self) -> None:
        """Open the video file.

        Raises:
            IOError: If file cannot be opened or is corrupted
        """
        self._cap = cv2.VideoCapture(str(self.file_path))

        if not self._cap.isOpened():
            raise OSError(
                f"Failed to open video file: {self.file_path}. "
                "File may be corrupted, unsupported format, or missing codecs."
            )

        # Read metadata
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 25.0  # Default fallback

        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine start timestamp
        if self._start_timestamp is None:
            # Use file modification time as fallback
            self._start_timestamp = self.file_path.stat().st_mtime
            warnings.warn(
                f"No start_timestamp provided for {self.file_path.name}. "
                f"Using file mtime as fallback ({self._start_timestamp}). "
                f"This may be inaccurate for copied/moved files. "
                f"Provide explicit start_timestamp or parse from filename.",
                UserWarning,
                stacklevel=3,
            )

        self._current_frame_number = 0
        self._is_open = True

    def close(self) -> None:
        """Close the video file and release resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False

    def read_frame(self) -> Frame | None:
        """Read next frame from file.

        Returns:
            Frame if available, None if end of file or error
        """
        if not self._is_open or self._cap is None:
            return None

        ret, image = self._cap.read()
        if not ret or image is None:
            return None

        # Compute timestamp: start + (frame_number / fps)
        timestamp = self._start_timestamp + (self._current_frame_number / self._fps)

        frame = Frame(
            image=image,
            timestamp_seconds=timestamp,
            frame_number=self._current_frame_number,
            camera_id=self.camera_id,
        )

        self._current_frame_number += 1
        return frame

    def get_fps(self) -> float:
        """Get frames per second of the video."""
        return self._fps

    def get_frame_count(self) -> int | None:
        """Get total frame count."""
        return self._frame_count if self._frame_count > 0 else None


class NVRExportAdapter:
    """Adapter for NVR-specific export formats.

    Handles brand-specific quirks (timestamp encoding, container formats).
    Brand-specific parsers are NOT IMPLEMENTED yet (code-only mode).
    """

    @staticmethod
    def detect_brand(file_path: Path) -> str | None:
        """Detect NVR brand from file metadata or naming.

        Args:
            file_path: Path to video file

        Returns:
            Brand name ("hikvision", "cpplus") or None if generic

        Raises:
            NotImplementedError: Always. Brand detection not implemented.
        """
        raise NotImplementedError(
            "Brand detection not implemented. "
            "Requires real Hikvision/CP Plus NVR exports to develop against. "
            "Pass brand explicitly or use generic file ingestion."
        )

    @staticmethod
    def parse_hikvision_timestamp(file_path: Path) -> float:
        """Parse Hikvision filename timestamp.

        Args:
            file_path: Path to Hikvision export file

        Returns:
            Start timestamp (seconds since epoch)

        Raises:
            NotImplementedError: Always. Hikvision parser not implemented.

        Note:
            Expected format: CH01_20261020180000_20261020180500.mp4 (YYYYMMDDHHmmss)
        """
        raise NotImplementedError(
            "Hikvision timestamp parser not implemented. "
            "Requires Hikvision NVR export samples and format specification. "
            "Provide explicit start_timestamp to FileVideoSource."
        )

    @staticmethod
    def parse_cpplus_timestamp(file_path: Path) -> float:
        """Parse CP Plus filename timestamp.

        Args:
            file_path: Path to CP Plus export file

        Returns:
            Start timestamp (seconds since epoch)

        Raises:
            NotImplementedError: Always. CP Plus parser not implemented.

        Note:
            Requires CP Plus NVR export samples to determine format.
        """
        raise NotImplementedError(
            "CP Plus timestamp parser not implemented. "
            "Requires CP Plus NVR export samples and format specification. "
            "Provide explicit start_timestamp to FileVideoSource."
        )

    @staticmethod
    def create_source(
        camera_id: str,
        file_path: Path,
        brand: str | None = None,
        start_timestamp: float | None = None,
    ) -> FileVideoSource:
        """Create a video source with brand-specific handling.

        Args:
            camera_id: Camera identifier
            file_path: Path to video file
            brand: NVR brand ("hikvision", "cpplus", or None for generic)
            start_timestamp: Override timestamp (None to use mtime fallback)

        Returns:
            Configured FileVideoSource

        Raises:
            NotImplementedError: If brand is specified (parsers not implemented)

        Note:
            Brand-specific timestamp parsing is NOT IMPLEMENTED.
            For now, always provide explicit start_timestamp or accept mtime fallback.
        """
        if brand is not None:
            raise NotImplementedError(
                f"Brand-specific adapter for '{brand}' not implemented. "
                f"Provide explicit start_timestamp or use generic ingestion."
            )

        return FileVideoSource(
            camera_id=camera_id,
            file_path=file_path,
            start_timestamp=start_timestamp,
        )
