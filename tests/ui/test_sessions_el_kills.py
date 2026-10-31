"""EL kill tests for PR #75. Assertions are the reviewer's; clocks sit on EPOCH."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.ui.test_sessions import (
    _OTHER,
    _SECRET,
    EPOCH,
    World,
    _cookie_header,
    _cookie_value,
    _csrf,
    _exchange,
    _get,
    _login,
    _post,
    _running,
)
from trace_search.audit import TRACE_AUDIT_HMAC_KEY_ENV, AuditLog, default_log_path, init_hmac_key
from trace_search.ui.cli import build_server
from trace_search.ui.sessions import (
    AppState,
    LoginLimiter,
    SessionStore,
    bound_host_values,
    bound_origin_values,
    csrf_matches,
)
from trace_search.ui.users import UserStore

pytest_plugins = ["tests.ui.test_sessions"]


def test_k1_refused_request_does_not_slide_idle_window(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    world.add("bea", "operator", _OTHER)
    ok = _login(world, "ada", _SECRET)
    cookie = f"session={_cookie_value(ok)}"
    world.clock.current = EPOCH + 1000
    assert _post(world, "/export", cookie, "wrong-token").status == 403
    assert _post(world, "/retention", cookie, _csrf(ok)).status == 403  # operator on admin route
    world.clock.current = EPOCH + 1800
    assert _get(world, cookie).status == 401


def test_k2_cookie_attributes_are_exact(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    ok = _login(world, "ada", _SECRET)
    header = _cookie_header(ok)
    key = _cookie_value(ok)
    parts = [p.strip() for p in header.split(";")]
    assert parts[0] == f"session={key}"
    assert sorted(p.lower() for p in parts[1:]) == ["httponly", "path=/", "samesite=strict"]
    out = _post(world, "/logout", f"session={key}", _csrf(ok))
    clear = [p.strip().lower() for p in _cookie_header(out).split(";")]
    assert clear[0] == "session="
    assert sorted(clear[1:]) == ["httponly", "max-age=0", "path=/", "samesite=strict"]


def test_k3_csrf_same_prefix_is_refused(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    ok = _login(world, "ada", _SECRET)
    token = _csrf(ok)
    flipped = token[:-1] + ("A" if token[-1] != "A" else "B")
    assert csrf_matches(token, flipped) is False
    assert csrf_matches(token, token[:8] + "x" * (len(token) - 8)) is False
    assert _post(world, "/export", f"session={_cookie_value(ok)}", flipped).status == 403


def test_k4_host_and_origin_allowlists_are_exact() -> None:
    assert bound_host_values(8765) == {"127.0.0.1:8765", "localhost:8765", "[::1]:8765"}
    assert bound_origin_values(8765) == {
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "http://[::1]:8765",
    }


def test_k5_block_runs_fifteen_minutes_from_the_fifth_failure(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    world.clock.current = EPOCH
    for _ in range(4):
        assert _login(world, "ada", _OTHER).status == 401
    world.clock.current = EPOCH + 899
    assert _login(world, "ada", _OTHER).status == 401
    world.clock.current = EPOCH + 1798
    assert _login(world, "ada", _SECRET).status == 403
    world.clock.current = EPOCH + 1799
    assert _login(world, "ada", _SECRET).status == 200


def test_k6_failed_login_does_not_initialize_a_missing_audit_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path = tmp_path / "keys" / "audit.key"
    init_hmac_key(key_path)
    monkeypatch.setenv(TRACE_AUDIT_HMAC_KEY_ENV, key_path.read_text(encoding="utf-8").strip())
    log = AuditLog(tmp_path / default_log_path())
    calls: list[int] = []
    monkeypatch.setattr(AuditLog, "initialize", lambda self: calls.append(1))
    users = UserStore(tmp_path / "users.sqlite")
    users.add_user("ada", "operator", _SECRET)
    app = AppState(
        users=users, audit=log, sessions=SessionStore(), limiter=LoginLimiter(), port=8765
    )
    w = World.__new__(World)
    w.app, w.log, w.users = app, log, users
    with pytest.raises(Exception):
        _login(w, "ada", _OTHER)
    assert calls == []
    assert not log.log_path.exists()


def test_k7_foreign_origin_is_refused_over_http(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    with _running(world.app) as httpd:
        port = int(httpd.server_address[1])
        before = len(world.log.read_all())
        status, headers, _b = _exchange(
            port,
            "POST",
            "/login",
            b"username=ada&password=" + _SECRET.encode(),
            headers={"Origin": "http://evil.example"},
        )
        assert status == 403
        assert headers.get("set-cookie") is None
        assert len(world.log.read_all()) == before
        assert len(world.app.sessions) == 0


def test_k8_cli_live_app_uses_wall_clock_and_the_verified_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    key_path = data / "keys" / "audit.key"
    init_hmac_key(key_path)
    monkeypatch.setenv(TRACE_AUDIT_HMAC_KEY_ENV, key_path.read_text(encoding="utf-8").strip())
    log = AuditLog(data / default_log_path())
    log.initialize()
    server = build_server(["--data", str(data), "--host", "127.0.0.1", "--port", "0"])
    assert not isinstance(server, int)
    try:
        app = server.app_state
        assert app is not None
        assert app.audit.log_path == log.log_path
        assert abs(app.sessions._now() - time.time()) < 60
        assert abs(app.limiter._now() - time.time()) < 60
    finally:
        server.server_close()


class _Clock:
    def __init__(self) -> None:
        self.current = EPOCH

    def __call__(self) -> float:
        return self.current


def test_k9_timeouts_and_window_hold_at_a_real_epoch() -> None:
    clock = _Clock()
    store = SessionStore(clock)
    s = store.create("ada", "operator")
    for step in range(16):
        clock.current = EPOCH + step * 1799
        assert store.resolve(s.session_key, touch=True) is not None
    clock.current = EPOCH + 28800
    assert store.resolve(s.session_key, touch=True) is None
    limiter = LoginLimiter(clock)
    clock.current = EPOCH
    for _ in range(4):
        assert limiter.decide("ada", False) == "invalid"
    clock.current = EPOCH + 10
    assert limiter.decide("ada", False) == "invalid"
    assert limiter.decide("ada", True) == "blocked"


def test_k10_blocked_attempt_still_runs_authenticate(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.add("ada", "operator", _SECRET)
    calls: list[str] = []
    real = UserStore.authenticate
    monkeypatch.setattr(
        UserStore,
        "authenticate",
        lambda self, u, p: (calls.append(u), real(self, u, p))[1],
    )
    for _ in range(5):
        _login(world, "ada", _OTHER)
    assert _login(world, "ada", _SECRET).status == 403
    assert len(calls) == 6
