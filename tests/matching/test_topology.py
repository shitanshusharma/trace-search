"""Camera topology: directed edges, transit windows, and JSON."""

import json
import math
from dataclasses import FrozenInstanceError, fields

import pytest

from trace_search.ingest.camera_registry import CameraConfig, CameraRegistry
from trace_search.matching.topology import CameraTopology, TransitEdge

_MIN_SECONDS = 10.0
_MAX_SECONDS = 25.0


def _camera(camera_id: str, *, x: float = 0.0, y: float = 0.0) -> CameraConfig:
    return CameraConfig(
        camera_id=camera_id,
        brand="fixed",
        site_map_x=x,
        site_map_y=y,
        clock_offset_seconds=0.0,
    )


def _registry(*camera_ids: str) -> CameraRegistry:
    registry = CameraRegistry()
    for index, camera_id in enumerate(camera_ids):
        registry.register(_camera(camera_id, x=float(index)))
    return registry


def _one_way() -> CameraTopology:
    registry = _registry("cam-a", "cam-b", "cam-c")
    edge = TransitEdge("cam-a", "cam-b", _MIN_SECONDS, _MAX_SECONDS)
    return CameraTopology(registry, (edge,))


def _edge_dict(
    from_camera: object = "cam-a",
    to_camera: object = "cam-b",
    min_seconds: object = 1.0,
    max_seconds: object = 2.0,
) -> dict[str, object]:
    return {
        "from_camera": from_camera,
        "to_camera": to_camera,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
    }


def _payload(
    edges: list[dict[str, object]],
    cameras: list[str] | None = None,
) -> dict[str, object]:
    if cameras is None:
        cameras = ["cam-a", "cam-b"]
    return {
        "cameras": [
            _camera(camera_id, x=float(index)).to_dict() for index, camera_id in enumerate(cameras)
        ],
        "edges": edges,
    }


def test_transit_edge_fields() -> None:
    assert [field.name for field in fields(TransitEdge)] == [
        "from_camera",
        "to_camera",
        "min_seconds",
        "max_seconds",
    ]


def test_transit_edge_is_frozen() -> None:
    edge = TransitEdge("cam-a", "cam-b", 1.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        edge.min_seconds = 3.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("from_camera", "to_camera"),
    [
        ("", "cam-b"),
        ("cam-a", ""),
        ("   ", "cam-b"),
        ("cam-a", "\t"),
        ("\n", "cam-b"),
        (None, "cam-b"),
        ("cam-a", 4),
    ],
    ids=[
        "from-empty",
        "to-empty",
        "from-spaces",
        "to-tab",
        "from-newline",
        "from-none",
        "to-int",
    ],
)
def test_transit_edge_rejects_empty_camera_id(from_camera: object, to_camera: object) -> None:
    """A whitespace-only id is empty. The check is not a trim of a real id."""
    with pytest.raises(ValueError, match="non-empty"):
        TransitEdge(from_camera, to_camera, 1.0, 2.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "camera_id",
    [" cam-a", "cam-a ", "cam-a\n", "\tcam-a"],
    ids=["leading-space", "trailing-space", "trailing-newline", "leading-tab"],
)
def test_transit_edge_rejects_surrounding_whitespace(camera_id: str) -> None:
    """Surrounding whitespace is refused. The id is not stripped."""
    with pytest.raises(ValueError, match="whitespace"):
        TransitEdge(camera_id, "cam-b", 1.0, 2.0)
    with pytest.raises(ValueError, match="whitespace"):
        TransitEdge("cam-b", camera_id, 1.0, 2.0)
    with pytest.raises(ValueError, match="whitespace"):
        CameraTopology.from_dict(_payload([_edge_dict(from_camera=camera_id)]))


def test_ids_differing_only_by_whitespace_cannot_form_an_edge() -> None:
    """A padded id and its plain form are not two cameras an edge may join."""
    leading = _registry("cam-a", " cam-a")
    assert leading.get("cam-a") is not None
    assert leading.get(" cam-a") is not None
    with pytest.raises(ValueError, match="whitespace"):
        CameraTopology(leading, (TransitEdge("cam-a", " cam-a", 1.0, 2.0),))

    trailing = _registry("cam-a", "cam-a ")
    assert trailing.get("cam-a ") is not None
    with pytest.raises(ValueError, match="whitespace"):
        CameraTopology(trailing, (TransitEdge("cam-a ", "cam-a", 1.0, 2.0),))


def test_transit_edge_rejects_nfkc_mismatch() -> None:
    """Fullwidth ``ｃａｍ-a`` is refused. It is not folded to ``cam-a``."""
    camera_id = "\uff43\uff41\uff4d-a"
    with pytest.raises(ValueError, match="NFKC"):
        TransitEdge(camera_id, "cam-b", 1.0, 2.0)
    with pytest.raises(ValueError, match="NFKC"):
        TransitEdge("cam-b", camera_id, 1.0, 2.0)
    with pytest.raises(ValueError, match="NFKC"):
        CameraTopology.from_dict(_payload([_edge_dict(to_camera=camera_id)]))


@pytest.mark.parametrize(
    "mark",
    ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"],
    ids=["zwsp", "zwnj", "zwj", "word-joiner", "bom"],
)
def test_transit_edge_rejects_format_characters(mark: str) -> None:
    """Category Cf is refused, including each zero-width mark."""
    camera_id = f"cam-a{mark}"
    with pytest.raises(ValueError, match="format character"):
        TransitEdge(camera_id, "cam-b", 1.0, 2.0)
    with pytest.raises(ValueError, match="format character"):
        TransitEdge("cam-a", f"cam-b{mark}", 1.0, 2.0)
    with pytest.raises(ValueError, match="format character"):
        CameraTopology.from_dict(_payload([_edge_dict(from_camera=camera_id)]))


def test_transit_edge_rejects_self_loop() -> None:
    with pytest.raises(ValueError, match="self-loop"):
        TransitEdge("cam-a", "cam-a", 1.0, 2.0)


@pytest.mark.parametrize(
    ("min_seconds", "max_seconds"),
    [(-0.1, 1.0), (0.0, -1.0), (-5.0, -1.0)],
    ids=["min-negative", "max-negative", "both-negative"],
)
def test_transit_edge_rejects_negative_seconds(min_seconds: float, max_seconds: float) -> None:
    with pytest.raises(ValueError, match=">= 0"):
        TransitEdge("cam-a", "cam-b", min_seconds, max_seconds)


@pytest.mark.parametrize(
    ("min_seconds", "max_seconds"),
    [
        (math.nan, 1.0),
        (1.0, math.nan),
        (math.inf, math.inf),
        (0.0, math.inf),
        (1.0, -math.inf),
    ],
    ids=["min-nan", "max-nan", "both-inf", "max-inf", "max-neg-inf"],
)
def test_transit_edge_rejects_non_finite_seconds(min_seconds: float, max_seconds: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        TransitEdge("cam-a", "cam-b", min_seconds, max_seconds)


def test_transit_edge_rejects_min_above_max() -> None:
    with pytest.raises(ValueError, match="min_seconds"):
        TransitEdge("cam-a", "cam-b", 3.0, 2.0)


def test_equal_bounds_are_allowed() -> None:
    edge = TransitEdge("cam-a", "cam-b", 2.0, 2.0)
    assert edge.min_seconds == 2.0
    assert edge.max_seconds == 2.0


def test_zero_seconds_is_allowed() -> None:
    edge = TransitEdge("cam-a", "cam-b", 0.0, 0.0)
    assert edge.min_seconds == 0.0
    assert edge.max_seconds == 0.0


def test_whole_numbers_are_stored_as_float() -> None:
    edge = TransitEdge("cam-a", "cam-b", 1, 2)
    assert edge.min_seconds == 1.0
    assert edge.max_seconds == 2.0
    assert isinstance(edge.min_seconds, float)
    assert isinstance(edge.max_seconds, float)


@pytest.mark.parametrize(
    ("min_seconds", "max_seconds"),
    [(True, 2.0), (1.0, False), ("1.0", 2.0), (1.0, None)],
    ids=["min-true", "max-false", "min-str", "max-none"],
)
def test_transit_edge_rejects_non_numeric_seconds(min_seconds: object, max_seconds: object) -> None:
    with pytest.raises(ValueError, match="finite number"):
        TransitEdge("cam-a", "cam-b", min_seconds, max_seconds)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("from_camera", "to_camera"),
    [
        ("cam-a", "cam-missing"),
        ("cam-missing", "cam-a"),
        ("cam-missing", "cam-other"),
    ],
    ids=["to-missing", "from-missing", "both-missing"],
)
def test_topology_rejects_unregistered_camera(from_camera: str, to_camera: str) -> None:
    edge = TransitEdge(from_camera, to_camera, 1.0, 2.0)
    with pytest.raises(ValueError, match="not registered"):
        CameraTopology(_registry("cam-a", "cam-b"), (edge,))


@pytest.mark.parametrize(
    "second_max",
    [2.0, 9.0],
    ids=["same-window", "different-window"],
)
def test_topology_rejects_duplicate_directed_edge(second_max: float) -> None:
    edges = (
        TransitEdge("cam-a", "cam-b", 1.0, 2.0),
        TransitEdge("cam-a", "cam-b", 1.0, second_max),
    )
    with pytest.raises(ValueError, match="duplicate"):
        CameraTopology(_registry("cam-a", "cam-b"), edges)


def test_reverse_edge_is_not_a_duplicate() -> None:
    topology = CameraTopology(
        _registry("cam-a", "cam-b"),
        (
            TransitEdge("cam-a", "cam-b", 1.0, 3.0),
            TransitEdge("cam-b", "cam-a", 4.0, 8.0),
        ),
    )
    assert topology.window("cam-a", "cam-b") == (1.0, 3.0)
    assert topology.window("cam-b", "cam-a") == (4.0, 8.0)


def test_successors_follow_edge_order() -> None:
    topology = CameraTopology(
        _registry("cam-a", "cam-b", "cam-c"),
        (
            TransitEdge("cam-a", "cam-c", 1.0, 2.0),
            TransitEdge("cam-a", "cam-b", 1.0, 2.0),
            TransitEdge("cam-b", "cam-c", 1.0, 2.0),
        ),
    )
    assert topology.successors("cam-a") == ("cam-c", "cam-b")
    assert topology.successors("cam-b") == ("cam-c",)
    assert topology.successors("cam-c") == ()
    assert topology.successors("cam-unknown") == ()
    assert topology.successors("cam-a") == ("cam-c", "cam-b")


def test_constructor_does_not_follow_later_edge_appends() -> None:
    edges = [TransitEdge("cam-a", "cam-c", 1.0, 2.0)]
    topology = CameraTopology(_registry("cam-a", "cam-b", "cam-c"), edges)
    edges.append(TransitEdge("cam-a", "cam-b", 1.0, 2.0))
    assert topology.successors("cam-a") == ("cam-c",)


def test_constructor_snapshots_cameras() -> None:
    original = _camera("cam-a")
    registry = CameraRegistry()
    registry.register(original)
    registry.register(_camera("cam-b", x=1.0))
    topology = CameraTopology(registry, (TransitEdge("cam-a", "cam-b", 1.0, 2.0),))
    original.brand = "changed"
    registry.register(_camera("cam-c", x=2.0))
    payload = topology.to_dict()
    assert [item["camera_id"] for item in payload["cameras"]] == ["cam-a", "cam-b"]
    assert payload["cameras"][0]["brand"] == "fixed"


def test_window_returns_bounds_for_directed_edge() -> None:
    topology = _one_way()
    assert topology.window("cam-a", "cam-b") == (_MIN_SECONDS, _MAX_SECONDS)
    assert topology.window("cam-b", "cam-a") is None
    assert topology.window("cam-a", "cam-c") is None


def test_allows_true_at_exact_min() -> None:
    topology = _one_way()
    assert topology.allows("cam-a", 100.0, "cam-b", 110.0) is True


def test_allows_true_at_exact_max() -> None:
    topology = _one_way()
    assert topology.allows("cam-a", 100.0, "cam-b", 125.0) is True


def test_allows_false_just_below_min() -> None:
    topology = _one_way()
    delta = math.nextafter(_MIN_SECONDS, -math.inf)
    assert topology.allows("cam-a", 0.0, "cam-b", delta) is False


def test_allows_false_just_above_max() -> None:
    topology = _one_way()
    delta = math.nextafter(_MAX_SECONDS, math.inf)
    assert topology.allows("cam-a", 0.0, "cam-b", delta) is False


def test_allows_true_inside_window() -> None:
    topology = _one_way()
    assert topology.allows("cam-a", 100.0, "cam-b", 112.0) is True


def test_allows_false_on_missing_edge() -> None:
    topology = _one_way()
    assert topology.allows("cam-a", 0.0, "cam-c", 15.0) is False


def test_allows_false_backwards_on_one_way_edge() -> None:
    topology = _one_way()
    assert topology.allows("cam-b", 0.0, "cam-a", 15.0) is False


def test_allows_false_when_delta_is_not_finite() -> None:
    topology = _one_way()
    assert topology.allows("cam-a", math.nan, "cam-b", _MIN_SECONDS) is False
    assert topology.allows("cam-a", 0.0, "cam-b", math.inf) is False


def test_equal_bounds_allow_only_that_delta() -> None:
    topology = CameraTopology(
        _registry("cam-a", "cam-b"),
        (TransitEdge("cam-a", "cam-b", 4.0, 4.0),),
    )
    assert topology.allows("cam-a", 0.0, "cam-b", 4.0) is True
    assert topology.allows("cam-a", 0.0, "cam-b", math.nextafter(4.0, -math.inf)) is False
    assert topology.allows("cam-a", 0.0, "cam-b", math.nextafter(4.0, math.inf)) is False


def test_two_way_corridor_is_two_edges() -> None:
    topology = CameraTopology(
        _registry("cam-a", "cam-b"),
        (
            TransitEdge("cam-a", "cam-b", 1.0, 3.0),
            TransitEdge("cam-b", "cam-a", 4.0, 8.0),
        ),
    )
    assert topology.allows("cam-a", 0.0, "cam-b", 2.0) is True
    assert topology.allows("cam-b", 0.0, "cam-a", 2.0) is False
    assert topology.allows("cam-b", 0.0, "cam-a", 5.0) is True


def test_empty_topology_round_trip() -> None:
    payload: dict[str, list[object]] = {"cameras": [], "edges": []}
    assert CameraTopology.from_dict(payload).to_dict() == payload


def test_json_round_trip_equals_input() -> None:
    payload = {
        "cameras": [
            {
                "camera_id": "cam-a",
                "brand": "hikvision",
                "site_map_x": 1.0,
                "site_map_y": 2.0,
                "clock_offset_seconds": 0.5,
            },
            {
                "camera_id": "cam-b",
                "brand": "cpplus",
                "site_map_x": 3.0,
                "site_map_y": 4.0,
                "clock_offset_seconds": 0.0,
            },
        ],
        "edges": [
            {
                "from_camera": "cam-a",
                "to_camera": "cam-b",
                "min_seconds": 1.5,
                "max_seconds": 6.0,
            },
            {
                "from_camera": "cam-b",
                "to_camera": "cam-a",
                "min_seconds": 0.0,
                "max_seconds": 2.0,
            },
        ],
    }
    restored = CameraTopology.from_dict(payload)
    assert restored.to_dict() == payload
    encoded = json.dumps(payload)
    assert CameraTopology.from_dict(json.loads(encoded)).to_dict() == payload


@pytest.mark.parametrize(
    ("edges", "cameras", "match"),
    [
        ([_edge_dict(from_camera="")], None, "non-empty"),
        ([_edge_dict(from_camera="   ")], None, "non-empty"),
        ([_edge_dict(from_camera="cam-a", to_camera="cam-a")], None, "self-loop"),
        ([_edge_dict(min_seconds=-1.0)], None, ">= 0"),
        ([_edge_dict(min_seconds=math.nan)], None, "must be finite"),
        ([_edge_dict(min_seconds=5.0, max_seconds=1.0)], None, "min_seconds"),
        ([_edge_dict(to_camera="cam-missing")], None, "not registered"),
        (
            [_edge_dict(), _edge_dict(min_seconds=3.0, max_seconds=4.0)],
            None,
            "duplicate",
        ),
    ],
    ids=[
        "empty-id",
        "whitespace-id",
        "self-loop",
        "negative",
        "non-finite",
        "min-above-max",
        "unregistered",
        "duplicate",
    ],
)
def test_from_dict_rejects_invalid_topology(
    edges: list[dict[str, object]],
    cameras: list[str] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        CameraTopology.from_dict(_payload(edges, cameras))


def test_from_dict_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        CameraTopology.from_dict([])
    with pytest.raises(ValueError, match="missing 'cameras'"):
        CameraTopology.from_dict({"edges": []})
    with pytest.raises(ValueError, match="missing 'edges'"):
        CameraTopology.from_dict({"cameras": []})
    with pytest.raises(ValueError, match="cameras must be a list"):
        CameraTopology.from_dict({"cameras": {}, "edges": []})
    with pytest.raises(ValueError, match="edges must be a list"):
        CameraTopology.from_dict({"cameras": [], "edges": "cam-a"})
    with pytest.raises(ValueError, match="keys must be strings"):
        CameraTopology.from_dict({1: [], "edges": []})
    with pytest.raises(ValueError, match="camera must be a JSON object"):
        CameraTopology.from_dict({"cameras": [["cam-a"]], "edges": []})
    with pytest.raises(ValueError, match="transit edge must be a JSON object"):
        CameraTopology.from_dict({"cameras": [], "edges": ["cam-a"]})
    with pytest.raises(ValueError, match="missing 'min_seconds'"):
        CameraTopology.from_dict(
            {
                "cameras": [_camera("cam-a").to_dict(), _camera("cam-b").to_dict()],
                "edges": [{"from_camera": "cam-a", "to_camera": "cam-b", "max_seconds": 1.0}],
            }
        )
    with pytest.raises(ValueError, match="already registered"):
        CameraTopology.from_dict(
            {
                "cameras": [_camera("cam-a").to_dict(), _camera("cam-a").to_dict()],
                "edges": [],
            }
        )
    with pytest.raises(ValueError, match="camera_id cannot be empty"):
        CameraTopology.from_dict(
            {
                "cameras": [
                    {
                        "camera_id": "",
                        "brand": "fixed",
                        "site_map_x": 0.0,
                        "site_map_y": 0.0,
                        "clock_offset_seconds": 0.0,
                    }
                ],
                "edges": [],
            }
        )


def test_from_dict_rejects_non_string_edge_key() -> None:
    with pytest.raises(ValueError, match="transit edge keys must be strings"):
        CameraTopology.from_dict({"cameras": [], "edges": [{1: "cam-a"}]})
