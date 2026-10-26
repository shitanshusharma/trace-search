"""Shared ReID fixtures. Clips are generated outside the repo tree."""

from __future__ import annotations

import socket
from pathlib import Path

import cv2
import numpy as np
import pytest

from trace_search.fixtures.generator import generate_smoke_fixtures


@pytest.fixture(autouse=True)
def refuse_socket_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test under ``tests/reid`` refuses ``socket.socket.connect``."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("ReID tests refuse socket.socket.connect")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(scope="module")
def smoke_clips(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict[str, object]]]:
    """Generate the smoke set once, outside the repo tree."""
    directory = tmp_path_factory.mktemp("reid-smoke-clips")
    repo_root = Path(__file__).resolve().parents[2]
    assert not directory.resolve().is_relative_to(repo_root)
    entries = generate_smoke_fixtures(directory, base_seed=42, num_clips=5)
    assert len(entries) >= 5
    assert all(entry["use"] == "SMOKE_ONLY" for entry in entries)
    return directory, entries


def rgb_frame(clip: Path) -> np.ndarray:
    """First sampled RGB frame, matching the smoke pHash frame list."""
    cap = cv2.VideoCapture(str(clip))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        index = int(np.linspace(0, total - 1, 10, dtype=int)[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read frame {index} of {clip}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)
    finally:
        cap.release()
