"""Per-camera pipeline: ingest, detect, track, and persist.

One camera has one tracker. Frames and crops are not kept.
"""

from .camera_pipeline import RunStats, run_camera, run_cameras

__all__ = ["RunStats", "run_camera", "run_cameras"]
