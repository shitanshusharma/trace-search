"""Scripted detector for FX-CODE tests.

This module does not load a model or weights.
"""

from collections.abc import Mapping, Sequence

from trace_search.detection.types import Detection
from trace_search.ingest.video_source import Frame


class ScriptedDetector:
    """Detector for CODE_TEST and SMOKE wiring only.

    Its output is never evidence. Callers receive a new list on every
    ``detect`` call, so a mutation of that list does not change the script.
    """

    def __init__(self, script: Mapping[tuple[str, int], Sequence[Detection]]) -> None:
        """Keep a copy of the script used by ``detect``.

        Args:
            script: Detections keyed by ``(camera_id, frame_number)``.
                Later changes to the caller's mapping do not change ``detect``.
        """
        self._script = dict(script)

    def detect(self, frame: Frame) -> list[Detection]:
        """Return the scripted detections for this frame.

        The key is ``(frame.camera_id, frame.frame_number)``. Any other key
        returns an empty list.
        """
        recorded = self._script.get((frame.camera_id, frame.frame_number))
        if recorded is None:
            return []
        return list(recorded)
