"""Loopback-only socket guard for the UI tests."""

from __future__ import annotations

import socket
from collections.abc import Callable

import pytest

_ALLOWED_HOSTS = frozenset({"127.0.0.1", "::1"})
_GUARD_MESSAGE = "ui socket guard allows only 127.0.0.1 and ::1"


def _host_of(address: object) -> object:
    if isinstance(address, tuple) and address:
        return address[0]
    return None


def _guarded(original: Callable[..., object]) -> Callable[..., object]:
    def connect(self: socket.socket, address: object) -> object:
        host = _host_of(address)
        if host not in _ALLOWED_HOSTS:
            raise OSError(f"{_GUARD_MESSAGE}; refused {host!r}")
        return original(self, address)

    return connect


@pytest.fixture(autouse=True)
def loopback_socket_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow ``connect`` only to 127.0.0.1 and ::1."""
    monkeypatch.setattr(socket.socket, "connect", _guarded(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded(socket.socket.connect_ex))
