"""The ReID suite refuses outbound ``socket.socket.connect``."""

from __future__ import annotations

import socket

import pytest


def test_reid_socket_connect_is_refused() -> None:
    """The autouse guard is active: connect raises before any network call."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="ReID tests refuse socket.socket.connect"):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()
