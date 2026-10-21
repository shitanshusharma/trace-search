"""Video ingestion module.

Provides file and RTSP video ingestion with per-camera clock correction.
"""

from .camera_registry import CameraConfig, CameraRegistry
from .file_ingest import FileVideoSource, NVRExportAdapter
from .rtsp_ingest import RTSPVideoSource, redact_rtsp_credentials
from .sync import estimate_offset_from_multiple_events, estimate_offset_from_sync_event
from .throughput import StreamStats, ThroughputHarness, ThroughputResult
from .video_source import Frame, VideoSource

__all__ = [
    "CameraConfig",
    "CameraRegistry",
    "FileVideoSource",
    "NVRExportAdapter",
    "RTSPVideoSource",
    "Frame",
    "VideoSource",
    "estimate_offset_from_sync_event",
    "estimate_offset_from_multiple_events",
    "ThroughputHarness",
    "ThroughputResult",
    "StreamStats",
    "redact_rtsp_credentials",
]
