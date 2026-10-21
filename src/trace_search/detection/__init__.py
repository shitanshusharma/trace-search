"""Detection contract.

Detectors share one record and one protocol. This package does not load a model.
"""

from .scripted import ScriptedDetector
from .types import DetClass, Detection, Detector

__all__ = ["DetClass", "Detection", "Detector", "ScriptedDetector"]
