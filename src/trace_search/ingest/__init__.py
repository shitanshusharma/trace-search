"""Video ingestion module.

Provides file and RTSP video ingestion with per-camera clock correction.
Public names load on first use, so importing ffmpeg_check does not import
OpenCV. The CI tools fence does that from a venv without cv2.
"""

from importlib import import_module
from typing import Any

# (relative module, attribute). Data keeps __getattr__ under the complexity gate.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CameraConfig": (".camera_registry", "CameraConfig"),
    "CameraRegistry": (".camera_registry", "CameraRegistry"),
    "FileVideoSource": (".file_ingest", "FileVideoSource"),
    "NVRExportAdapter": (".file_ingest", "NVRExportAdapter"),
    "RTSPVideoSource": (".rtsp_ingest", "RTSPVideoSource"),
    "Frame": (".video_source", "Frame"),
    "VideoSource": (".video_source", "VideoSource"),
    "estimate_offset_from_sync_event": (".sync", "estimate_offset_from_sync_event"),
    "estimate_offset_from_multiple_events": (
        ".sync",
        "estimate_offset_from_multiple_events",
    ),
    "ThroughputHarness": (".throughput", "ThroughputHarness"),
    "ThroughputResult": (".throughput", "ThroughputResult"),
    "StreamStats": (".throughput", "StreamStats"),
    "redact_rtsp_credentials": (".rtsp_ingest", "redact_rtsp_credentials"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(export: str) -> Any:
    """Import one public name the first time a caller asks for it."""
    spec = _LAZY_EXPORTS.get(export)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {export!r}")
    module_name, attr = spec
    value = getattr(import_module(module_name, __name__), attr)
    globals()[export] = value
    return value
