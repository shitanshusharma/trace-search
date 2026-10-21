"""Scripted detector for FX-CODE tests.

This module does not load a model or weights.
"""

from collections.abc import Mapping, Sequence

from trace_search.detection.types import Detection
from trace_search.ingest.video_source import Frame


def _require_script_key(key: tuple[str, int], detection: Detection) -> None:
    """Raise when a scripted detection does not carry its script key."""
    key_camera, key_frame = key
    if detection.camera_id == key_camera and detection.frame_number == key_frame:
        return
    raise ValueError(
        "scripted detection does not match script key "
        f"(camera_id={key_camera!r}, frame_number={key_frame!r}): "
        f"detection camera_id={detection.camera_id!r} "
        f"frame_number={detection.frame_number!r}"
    )


def _require_script(key: tuple[str, int], detections: Sequence[Detection]) -> None:
    for detection in detections:
        _require_script_key(key, detection)


class ScriptedDetector:
    """Detector for CODE_TEST and SMOKE wiring only.

    Its output is never evidence. Callers receive a new list on every
    ``detect`` call, so a mutation of that list does not change the script.
    A detection whose camera_id or frame_number differs from its script key
    raises ValueError.
    """

    def __init__(self, script: Mapping[tuple[str, int], Sequence[Detection]]) -> None:
        """Keep a copy of the script used by ``detect``.

        Args:
            script: Detections keyed by ``(camera_id, frame_number)``.
                Later changes to the caller's mapping, or an append on a
                caller's sequence, do not change ``detect``.
        """
        self._script = {key: tuple(value) for key, value in script.items()}

    def detect(self, frame: Frame) -> list[Detection]:
        """Return the scripted detections for this frame.

        The key is ``(frame.camera_id, frame.frame_number)``. Any other key
        returns an empty list. A detection whose ``camera_id`` or
        ``frame_number`` differs from that key raises ``ValueError``.
        """
        key = (frame.camera_id, frame.frame_number)
        recorded = self._script.get(key)
        if recorded is None:
            return []
        found = list(recorded)
        _require_script(key, found)
        return found
