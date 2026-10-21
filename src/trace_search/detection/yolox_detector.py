"""YOLOX-S on ONNX Runtime, CPU only.

Weights stay outside this repository. Pass ``weights_path`` or set
``TRACE_SEARCH_YOLOX_WEIGHTS``. The file is hashed before a session is
built. Preprocess and postprocess are ``letterbox`` and ``to_detections``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from trace_search.detection.types import Detection
from trace_search.detection.yolox_post import letterbox, to_detections
from trace_search.ingest.video_source import Frame

_ENV = "TRACE_SEARCH_YOLOX_WEIGHTS"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def weights_sha256_from_manifest(manifest_path: Path | None = None) -> str:
    """Sha256 of the YOLOX-S COCO row in ``licenses/manifest.json``.

    The default path is resolved from the repository root, not the
    working directory.
    """
    path = _repo_root() / "licenses" / "manifest.json"
    if manifest_path is not None:
        path = manifest_path
    document = json.loads(path.read_text(encoding="utf-8"))
    for entry in document.get("entries", []):
        if entry.get("name") == "YOLOX-S COCO":
            return str(entry["sha256"])
    raise ValueError(f"YOLOX-S COCO weights row is missing from {path}.")


def _resolve_weights_path(weights_path: Path | None) -> Path:
    if weights_path is not None:
        return Path(weights_path)
    raw = os.environ.get(_ENV, "")
    if raw == "":
        raise FileNotFoundError(
            "YOLOX weights path is not set. "
            f"Pass weights_path or set {_ENV} to a file outside the repository."
        )
    return Path(raw)


def _require_file(path: Path) -> None:
    if path.is_file():
        return
    raise FileNotFoundError(
        f"YOLOX weights file not found at {path}. "
        f"Set {_ENV} to a weights file outside the repository."
    )


def _require_outside_repo(path: Path) -> None:
    root = _repo_root()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return
    raise ValueError(
        f"Refusing YOLOX weights path {resolved} inside the repository {root}. "
        f"Set {_ENV} to a path outside the repository."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(path: Path, expected_sha256: str) -> None:
    actual = _sha256(path)
    if actual == expected_sha256.lower():
        return
    raise ValueError(
        f"YOLOX weights sha256 mismatch for {path}: expected {expected_sha256}, got {actual}."
    )


def _session(path: Path) -> Any:
    import onnxruntime  # type: ignore[import-untyped]

    build: Any = onnxruntime.InferenceSession
    return build(str(path), providers=["CPUExecutionProvider"])


def _rows(output: object) -> npt.NDArray[np.float64]:
    raw = np.asarray(output, dtype=np.float64)
    if raw.ndim == 3 and int(raw.shape[0]) == 1:
        raw = raw[0]
    return np.ascontiguousarray(raw, dtype=np.float64)


class YoloxDetector:
    """``Detector`` for one YOLOX ONNX file on ``CPUExecutionProvider``."""

    _session: Any

    def __init__(self, weights_path: Path | None = None, expected_sha256: str = "") -> None:
        """Open ``weights_path`` after the sha256 check.

        Args:
            weights_path: File outside the repository. When omitted, the path
                is ``TRACE_SEARCH_YOLOX_WEIGHTS``.
            expected_sha256: Expected sha256 hex digest of that file. When
                omitted, the digest is the YOLOX-S COCO row in
                ``licenses/manifest.json``.

        Raises:
            FileNotFoundError: The path is missing. The message names the
                path and ``TRACE_SEARCH_YOLOX_WEIGHTS``.
            ValueError: The path is inside the repository, or the sha256
                does not match. A mismatch names both digests and is raised
                before a session is built.
        """
        path = _resolve_weights_path(weights_path)
        _require_file(path)
        _require_outside_repo(path)
        digest = expected_sha256 or weights_sha256_from_manifest()
        _require_sha256(path, digest)
        self._session = _session(path)

    def detect(self, frame: Frame) -> list[Detection]:
        """Letterbox ``frame``, run the session, and decode with T-3-7."""
        image = cast(npt.NDArray[np.uint8], np.asarray(frame.image))
        tensor, ratio, pad = letterbox(image)
        batched = np.expand_dims(tensor, axis=0)
        feeds = self._session.get_inputs()
        input_name = str(feeds[0].name)
        outputs = self._session.run(None, {input_name: batched})
        return to_detections(
            _rows(outputs[0]),
            ratio=ratio,
            pad=pad,
            camera_id=frame.camera_id,
            frame_number=frame.frame_number,
            timestamp_seconds=frame.timestamp_seconds,
        )
