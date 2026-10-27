"""The ReID suite refuses outbound ``socket.socket.connect``."""

from __future__ import annotations

import socket

import pytest

from tests.reid.conftest import fail_if_socket_connect_attempted


def test_reid_socket_connect_is_refused(
    refuse_socket_connect: list[tuple[object, object]],
) -> None:
    """The autouse guard is active: connect raises before any network call."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="ReID tests refuse socket.socket.connect"):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()
    assert len(refuse_socket_connect) == 1
    refuse_socket_connect.clear()


def test_swallowed_socket_connect_still_fails(
    refuse_socket_connect: list[tuple[object, object]],
) -> None:
    """Catching the guard's OSError still counts as a connect attempt."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        try:
            sock.connect(("127.0.0.1", 9))
        except OSError as exc:
            assert "ReID tests refuse socket.socket.connect" in str(exc)
    finally:
        sock.close()
    assert len(refuse_socket_connect) == 1
    with pytest.raises(pytest.fail.Exception, match="socket.socket.connect"):
        fail_if_socket_connect_attempted(refuse_socket_connect)
    refuse_socket_connect.clear()
