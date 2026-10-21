"""Per-camera tracking contract.

Trackers share one record and one protocol. A track id is unique inside one
camera. This package does not load a model.
"""

from .bytetrack import ByteTracker
from .types import Tracker, TrackObservation

__all__ = ["ByteTracker", "TrackObservation", "Tracker"]
