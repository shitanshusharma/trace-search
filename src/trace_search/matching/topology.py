"""Directed camera topology and transit-time windows.

A topology records which registered camera may follow which, and the inclusive
walk time from an exit timestamp to the next entry timestamp. An edge is one
direction. A two-way corridor is two edges. This module does not read a site
file and does not use a graph package.

Whitespace-only camera ids are empty. ``str.strip`` decides that, and only
that. The stored id is kept as given, so a non-blank id keeps its spaces.

A second edge with the same ``(from_camera, to_camera)`` is refused, whether
or not the windows match. The opposite direction is a different edge.

``CameraTopology`` copies the camera set at init. Later edits to the caller's
registry are not part of the topology.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from trace_search.ingest.camera_registry import CameraConfig, CameraRegistry

__all__ = ["CameraTopology", "TransitEdge"]


def _reject_empty_camera_id(label: str, camera_id: object) -> None:
    """Reject a missing id. A whitespace-only string counts as empty."""
    if isinstance(camera_id, str) and camera_id.strip() != "":
        return
    raise ValueError(f"{label} must be a non-empty camera id, got {camera_id!r}")


def _reject_self_loop(from_camera: str, to_camera: str) -> None:
    if from_camera == to_camera:
        raise ValueError(f"transit edge cannot be a self-loop on {from_camera!r}")


def _require_seconds(label: str, value: object) -> float:
    """Return a finite float >= 0. Booleans and other types are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number >= 0, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    if number < 0.0:
        raise ValueError(f"{label} must be >= 0, got {number}")
    return number


def _reject_min_above_max(min_seconds: float, max_seconds: float) -> None:
    if min_seconds > max_seconds:
        raise ValueError(f"min_seconds ({min_seconds}) must be <= max_seconds ({max_seconds})")


def _require_dict(data: object, label: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    parsed: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings")
        parsed[key] = value
    return parsed


def _require_list(data: dict[str, Any], key: str) -> list[Any]:
    if key not in data:
        raise ValueError(f"topology is missing {key!r}")
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return list(value)


def _snapshot_registry(cameras: CameraRegistry) -> CameraRegistry:
    """Copy cameras through their JSON form so later edits do not leak in."""
    snapshot = CameraRegistry()
    for camera in cameras.list_cameras():
        snapshot.register(CameraConfig.from_dict(camera.to_dict()))
    return snapshot


def _registry_from_items(items: list[Any]) -> CameraRegistry:
    registry = CameraRegistry()
    for item in items:
        registry.register(CameraConfig.from_dict(_require_dict(item, "camera")))
    return registry


def _edges_from_items(items: list[Any]) -> tuple[TransitEdge, ...]:
    return tuple(TransitEdge.from_dict(item) for item in items)


@dataclass(frozen=True)
class TransitEdge:
    """One directed walk from one camera to another.

    ``min_seconds`` and ``max_seconds`` are inclusive bounds on
    ``entry_s - exit_s``. A whitespace-only camera id is empty and is rejected.
    The stored ids are not stripped.
    """

    from_camera: str
    to_camera: str
    min_seconds: float
    max_seconds: float

    def __post_init__(self) -> None:
        """Reject an empty id, a self-loop, a bad duration, or min above max."""
        _reject_empty_camera_id("from_camera", self.from_camera)
        _reject_empty_camera_id("to_camera", self.to_camera)
        min_seconds = _require_seconds("min_seconds", self.min_seconds)
        max_seconds = _require_seconds("max_seconds", self.max_seconds)
        _reject_self_loop(self.from_camera, self.to_camera)
        _reject_min_above_max(min_seconds, max_seconds)
        object.__setattr__(self, "min_seconds", min_seconds)
        object.__setattr__(self, "max_seconds", max_seconds)

    def to_dict(self) -> dict[str, str | float]:
        """Return the JSON object for this edge."""
        return {
            "from_camera": self.from_camera,
            "to_camera": self.to_camera,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
        }

    @classmethod
    def from_dict(cls, data: object) -> TransitEdge:
        """Build an edge. Invalid values use the same checks as the constructor."""
        payload = _require_dict(data, "transit edge")
        try:
            from_camera = payload["from_camera"]
            to_camera = payload["to_camera"]
            min_seconds = payload["min_seconds"]
            max_seconds = payload["max_seconds"]
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(f"transit edge is missing {missing!r}") from exc
        return cls(from_camera, to_camera, min_seconds, max_seconds)


class CameraTopology:
    """Directed transit graph over a snapshotted camera registry.

    Duplicate ``(from_camera, to_camera)`` pairs raise ``ValueError``.
    ``(to_camera, from_camera)`` is a separate edge. ``successors`` follows
    the order of the edge sequence passed to the constructor.
    """

    def __init__(self, cameras: CameraRegistry, edges: Sequence[TransitEdge]) -> None:
        self._cameras = _snapshot_registry(cameras)
        self._by_pair: dict[tuple[str, str], TransitEdge] = {}
        self._outgoing: dict[str, list[str]] = {}
        for edge in edges:
            self._add_edge(edge)

    def _require_registered(self, camera_id: str) -> None:
        if self._cameras.get(camera_id) is None:
            raise ValueError(f"camera {camera_id!r} is not registered")

    def _reject_duplicate(self, pair: tuple[str, str]) -> None:
        if pair in self._by_pair:
            left, right = pair
            raise ValueError(f"duplicate transit edge {left!r} -> {right!r}")

    def _add_edge(self, edge: TransitEdge) -> None:
        self._require_registered(edge.from_camera)
        self._require_registered(edge.to_camera)
        pair = (edge.from_camera, edge.to_camera)
        self._reject_duplicate(pair)
        self._by_pair[pair] = edge
        self._outgoing.setdefault(edge.from_camera, []).append(edge.to_camera)

    def _directed_edge(self, from_camera: str, to_camera: str) -> TransitEdge | None:
        """Return the from -> to edge only. The reverse edge does not count."""
        return self._by_pair.get((from_camera, to_camera))

    def successors(self, camera_id: str) -> tuple[str, ...]:
        """Destination cameras in constructor order.

        A camera with no outgoing edge yields an empty tuple. An unknown id
        also yields an empty tuple.
        """
        return tuple(self._outgoing.get(camera_id, ()))

    def window(self, a: str, b: str) -> tuple[float, float] | None:
        """Inclusive ``(min_seconds, max_seconds)`` for ``a -> b``, or None."""
        edge = self._directed_edge(a, b)
        if edge is None:
            return None
        return (edge.min_seconds, edge.max_seconds)

    def allows(self, a: str, exit_s: float, b: str, entry_s: float) -> bool:
        """True only for a directed edge with min <= entry_s - exit_s <= max."""
        edge = self._directed_edge(a, b)
        if edge is None:
            return False
        delta = entry_s - exit_s
        # Inclusive on both ends. A non-finite delta compares false.
        return edge.min_seconds <= delta <= edge.max_seconds

    def to_dict(self) -> dict[str, list[dict[str, str | float]]]:
        """JSON object: camera records in registry form, then directed edges."""
        return {
            "cameras": [camera.to_dict() for camera in self._cameras.list_cameras()],
            "edges": [edge.to_dict() for edge in self._by_pair.values()],
        }

    @classmethod
    def from_dict(cls, data: object) -> CameraTopology:
        """Build a topology. Validation uses the constructors' rules."""
        payload = _require_dict(data, "topology")
        registry = _registry_from_items(_require_list(payload, "cameras"))
        edges = _edges_from_items(_require_list(payload, "edges"))
        return cls(registry, edges)
