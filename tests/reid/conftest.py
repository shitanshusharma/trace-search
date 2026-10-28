"""Shared ReID fixtures. Clips are generated outside the repo tree."""

from __future__ import annotations

import copy
import socket
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest


def fail_if_socket_connect_attempted(attempts: list[tuple[object, object]]) -> None:
    """Fail when ``socket.socket.connect`` ran, including a caught ``OSError``."""
    if attempts:
        pytest.fail(f"socket.socket.connect was called {len(attempts)} time(s)")


@pytest.fixture(autouse=True)
def refuse_socket_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[tuple[object, object]]]:
    """Record every ``socket.socket.connect`` and fail the test if one happened.

    The replacement raises before a network call. A test that catches that
    ``OSError`` still fails here, because the attempt was recorded.
    """
    attempts: list[tuple[object, object]] = []

    def refuse(*args: object, **kwargs: object) -> None:
        attempts.append((args, kwargs))
        raise OSError("ReID tests refuse socket.socket.connect")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    yield attempts
    fail_if_socket_connect_attempted(attempts)


@pytest.fixture(scope="module")
def smoke_clips(
    canonical_smoke_set: tuple[Path, list[dict[str, object]]],
) -> tuple[Path, list[dict[str, object]]]:
    """Reuse the shared smoke set. It is generated outside the repo tree."""
    directory, entries = canonical_smoke_set
    repo_root = Path(__file__).resolve().parents[2]
    assert not directory.resolve().is_relative_to(repo_root)
    assert len(entries) >= 5
    assert all(entry["use"] == "SMOKE_ONLY" for entry in entries)
    copied: list[dict[str, object]] = copy.deepcopy(entries)
    return directory, copied


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
