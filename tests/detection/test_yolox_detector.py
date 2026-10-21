"""YOLOX runtime adapter.

The ONNX graph in this file is a hand-serialised ModelProto. It does not
import the ``onnx`` package and it does not vendor ``onnx.proto``. The
wire format is the protobuf encoding of an opset-11 graph: ReduceSum of
the image, multiplied by a zero scalar, added to a constant ``(1, 8400, 85)``
head. The head is the T-3-7 person cell. The file is written only under
the pytest temp directory.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from trace_search.detection.types import DetClass
from trace_search.detection.yolox_detector import YoloxDetector
from trace_search.ingest.video_source import Frame

_ENV = "TRACE_SEARCH_YOLOX_WEIGHTS"
_COUNT = 80 * 80 + 40 * 40 + 20 * 20
_INDEX = 30 * 80 + 30
_EXPECTED_BOX = (147.5, 47.5, 157.5, 57.5)


def _varint(value: int) -> bytes:
    out = bytearray()
    rest = value
    while True:
        bits = rest & 0x7F
        rest >>= 7
        if rest:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _blob(field: int, payload: bytes) -> bytes:
    return _key(field, 2) + _varint(len(payload)) + payload


def _text(field: int, value: str) -> bytes:
    return _blob(field, value.encode())


def _vint(field: int, value: int) -> bytes:
    return _key(field, 0) + _varint(value)


def _tensor(name: str, dims: tuple[int, ...], raw: bytes) -> bytes:
    body = b"".join(_vint(1, dim) for dim in dims)
    body += _vint(2, 1)
    body += _text(8, name)
    body += _blob(9, raw)
    return body


def _value_info(name: str, dims: tuple[int, ...]) -> bytes:
    shape = b"".join(_blob(1, _vint(1, dim)) for dim in dims)
    tensor_type = _vint(1, 1) + _blob(2, shape)
    return _text(1, name) + _blob(2, _blob(1, tensor_type))


def _attr_ints(name: str, values: tuple[int, ...]) -> bytes:
    body = _text(1, name)
    body += b"".join(_vint(7, value) for value in values)
    body += _vint(20, 7)
    return body


def _attr_int(name: str, value: int) -> bytes:
    return _text(1, name) + _vint(3, value) + _vint(20, 2)


def _node(
    op_type: str,
    inputs: list[str],
    outputs: list[str],
    attrs: list[bytes] | None = None,
) -> bytes:
    body = b"".join(_text(1, item) for item in inputs)
    body += b"".join(_text(2, item) for item in outputs)
    body += _text(4, op_type)
    for attr in attrs or []:
        body += _blob(5, attr)
    return body


def _painted_head() -> np.ndarray:
    assert _COUNT == 8400
    head = np.zeros((_COUNT, 85), dtype=np.float64)
    head[_INDEX, 0] = 0.5
    head[_INDEX, 1] = 0.5
    head[_INDEX, 2] = math.log(2.0)
    head[_INDEX, 3] = math.log(2.0)
    head[_INDEX, 4] = 0.9
    head[_INDEX, 5] = 0.8
    return head


def _model_bytes() -> bytes:
    head = _painted_head().astype(np.float32).reshape(1, _COUNT, 85)
    zero = np.zeros((), dtype=np.float32)
    nodes = b"".join(
        [
            _blob(
                1,
                _node(
                    "ReduceSum",
                    ["images"],
                    ["reduced"],
                    [_attr_ints("axes", (0, 1, 2, 3)), _attr_int("keepdims", 0)],
                ),
            ),
            _blob(1, _node("Mul", ["reduced", "zero"], ["zeroed"])),
            _blob(1, _node("Add", ["head", "zeroed"], ["output"])),
        ]
    )
    graph = nodes + _text(2, "tiny")
    graph += _blob(5, _tensor("head", (1, _COUNT, 85), head.tobytes()))
    graph += _blob(5, _tensor("zero", (), zero.tobytes()))
    graph += _blob(11, _value_info("images", (1, 3, 640, 640)))
    graph += _blob(12, _value_info("output", (1, _COUNT, 85)))
    opset = _text(1, "") + _vint(2, 11)
    return _vint(1, 6) + _blob(7, graph) + _blob(8, opset)


def _frame() -> Frame:
    return Frame(
        image=np.zeros((200, 400, 3), dtype=np.uint8),
        timestamp_seconds=1.25,
        frame_number=4,
        camera_id="cam-a",
    )


def _assert_person(found: list) -> None:
    assert len(found) == 1
    detection = found[0]
    assert detection.det_class is DetClass.PERSON
    assert detection.camera_id == "cam-a"
    assert detection.frame_number == 4
    assert detection.timestamp_seconds == pytest.approx(1.25)
    assert detection.score == pytest.approx(0.72)
    assert detection.box[0] == pytest.approx(_EXPECTED_BOX[0], abs=1.0)
    assert detection.box[1] == pytest.approx(_EXPECTED_BOX[1], abs=1.0)
    assert detection.box[2] == pytest.approx(_EXPECTED_BOX[2], abs=1.0)
    assert detection.box[3] == pytest.approx(_EXPECTED_BOX[3], abs=1.0)


class _RecordingSession:
    def __init__(self, rows: np.ndarray) -> None:
        self._rows = rows
        self.feed: dict[str, np.ndarray] = {}

    def get_inputs(self) -> list[object]:
        return [type("Input", (), {"name": "images"})()]

    def run(self, _names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feed = feed
        return [self._rows]


def _record_session(
    monkeypatch: pytest.MonkeyPatch,
    rows: np.ndarray,
) -> tuple[list[list[str]], list[_RecordingSession]]:
    providers_seen: list[list[str]] = []
    sessions: list[_RecordingSession] = []

    def _build(path: str, providers: list[str] | None = None) -> _RecordingSession:
        del path
        assert providers == ["CPUExecutionProvider"]
        providers_seen.append(list(providers))
        session = _RecordingSession(rows)
        sessions.append(session)
        return session

    monkeypatch.setattr("onnxruntime.InferenceSession", _build)
    return providers_seen, sessions


def test_wrong_sha_does_not_construct_a_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 16-byte file with the wrong digest never builds InferenceSession."""
    weights = tmp_path / "weights.bin"
    payload = b"0123456789abcdef"
    weights.write_bytes(payload)
    assert weights.stat().st_size == 16
    calls = {"n": 0}

    def _boom(*_args: object, **_kwargs: object) -> None:
        calls["n"] += 1
        raise AssertionError("InferenceSession was constructed")

    monkeypatch.setattr("onnxruntime.InferenceSession", _boom)
    expected = "0" * 64
    actual = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="sha256 mismatch") as caught:
        YoloxDetector(weights, expected)
    message = str(caught.value)
    assert expected in message
    assert actual in message
    assert calls["n"] == 0


def test_missing_path_names_the_env_var(tmp_path: Path) -> None:
    missing = tmp_path / "missing-weights.bin"
    with pytest.raises(FileNotFoundError, match="TRACE_SEARCH_YOLOX_WEIGHTS") as caught:
        YoloxDetector(missing, "ab")
    assert str(missing) in str(caught.value)


def test_unset_env_names_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="TRACE_SEARCH_YOLOX_WEIGHTS"):
        YoloxDetector(expected_sha256="ab")


def test_path_inside_the_repo_is_refused() -> None:
    inside = Path("README.md").resolve()
    assert inside.is_file()
    with pytest.raises(ValueError, match="inside the repository") as caught:
        YoloxDetector(inside, "ab")
    assert str(inside) in str(caught.value)


def test_matching_sha_uses_cpu_and_t3_7_postprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stand-in session still receives the letterboxed frame and the CPU provider."""
    weights = tmp_path / "weights.bin"
    payload = b"0123456789abcdef"
    weights.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    providers, sessions = _record_session(monkeypatch, _painted_head())
    detector = YoloxDetector(weights, digest)
    assert providers == [["CPUExecutionProvider"]]
    _assert_person(detector.detect(_frame()))
    assert sessions[0].feed["images"].shape == (1, 3, 640, 640)
    assert sessions[0].feed["images"].dtype == np.float32


def test_env_var_selects_the_weights_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = tmp_path / "from-env.bin"
    payload = b"0123456789abcdef"
    weights.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setenv(_ENV, str(weights))
    providers, _sessions = _record_session(monkeypatch, _painted_head())
    YoloxDetector(expected_sha256=digest)
    assert providers == [["CPUExecutionProvider"]]


def test_tiny_graph_emits_the_t3_7_detection(tmp_path: Path) -> None:
    """The real CPU session returns the painted head and T-3-7's person box."""
    weights = tmp_path / "tiny.onnx"
    weights.write_bytes(_model_bytes())
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    detector = YoloxDetector(weights, digest)
    assert "CPUExecutionProvider" in detector._session.get_providers()
    _assert_person(detector.detect(_frame()))
