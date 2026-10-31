"""Login, logout, sessions, rate limits, and the loopback Host/Origin check."""

from __future__ import annotations

import hmac
import http.client
import json
import os
import re
import secrets
import threading
import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from trace_search.audit import TRACE_AUDIT_HMAC_KEY_ENV, AuditLog, default_log_path, init_hmac_key
from trace_search.ui.cli import build_server
from trace_search.ui.pages.login import AppState
from trace_search.ui.routes import UiRequest, dispatch
from trace_search.ui.server import STATIC_DIR, make_server
from trace_search.ui.sessions import (
    ABSOLUTE_TIMEOUT_SECONDS,
    BLOCK_SECONDS,
    FAILURE_WINDOW_SECONDS,
    IDLE_TIMEOUT_SECONDS,
    MAX_FAILURES,
    SESSION_KEY_BYTES,
    LoginLimiter,
    SessionStore,
    request_is_allowed,
)
from trace_search.ui.users import UserStore

_SECRET = "session-secret-value"
_OTHER = "other-secret-value"
_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")


class Clock:
    """Injectable clock. Tests move ``current`` instead of sleeping."""

    def __init__(self) -> None:
        self.current = 0.0

    def __call__(self) -> float:
        return self.current


class World:
    """One data dir, user db, audit log, and in-memory session store."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        key_path = tmp_path / "keys" / "audit.key"
        init_hmac_key(key_path)
        monkeypatch.setenv(TRACE_AUDIT_HMAC_KEY_ENV, key_path.read_text(encoding="utf-8").strip())
        log = AuditLog(tmp_path / default_log_path())
        log.initialize()
        clock = Clock()
        users = UserStore(tmp_path / "users.sqlite")
        self.tmp_path = tmp_path
        self.log = log
        self.clock = clock
        self.users = users
        self.app = AppState(
            users=users,
            audit=log,
            sessions=SessionStore(clock),
            limiter=LoginLimiter(clock),
            port=8765,
        )

    def add(self, username: str, role: str, password: str) -> None:
        self.users.add_user(username, role, password)


def _check_log(log: AuditLog) -> None:
    raw = log.log_path.read_bytes()
    assert _SECRET.encode("utf-8") not in raw
    assert _OTHER.encode("utf-8") not in raw
    if log.head_path.is_file():
        head = log.head_path.read_bytes()
        assert _SECRET.encode("utf-8") not in head
        assert _OTHER.encode("utf-8") not in head
    outcome = log.verify()
    assert outcome.valid, outcome.error_message


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    built = World(tmp_path, monkeypatch)
    yield built
    _check_log(built.log)


def _body(username: str, password: str, *, as_json: bool = False) -> bytes:
    if as_json:
        return json.dumps({"username": username, "password": password}).encode("utf-8")
    return urllib.parse.urlencode({"username": username, "password": password}).encode("ascii")


def _login(
    world: World,
    username: str,
    password: str,
    *,
    host: str | None = None,
    origin: str | None = None,
    cookie: str | None = None,
    include_host: bool = True,
    as_json: bool = False,
    port: int | None = None,
) -> object:
    bound = world.app.port if port is None else port
    headers: dict[str, str] = {}
    if include_host:
        headers["host"] = f"127.0.0.1:{bound}" if host is None else host
    if origin is not None:
        headers["origin"] = origin
    if cookie is not None:
        headers["cookie"] = cookie
    response = dispatch(
        UiRequest(
            "POST",
            "/login",
            STATIC_DIR,
            _body(username, password, as_json=as_json),
            headers,
            world.app,
        )
    )
    assert _SECRET.encode("utf-8") not in response.body
    assert _OTHER.encode("utf-8") not in response.body
    return response


def _cookie_header(response: object) -> str | None:
    pairs = response.header_pairs
    found = [value for key, value in pairs if key == "Set-Cookie"]
    if not found:
        return None
    assert len(found) == 1
    return found[0]


def _cookie_value(response: object) -> str:
    header = _cookie_header(response)
    assert header is not None
    value = header.split(";", 1)[0]
    assert value.startswith("session=")
    return value.split("=", 1)[1]


def _flags(header: str) -> list[str]:
    return [part.strip().lower() for part in header.split(";")]


def _assert_cookie_flags(header: str) -> None:
    parts = _flags(header)
    assert "httponly" in parts
    assert "samesite=strict" in parts
    assert "path=/" in parts
    assert "secure" not in parts


def _get(world: World, cookie: str, path: str = "/search", csrf: str | None = None) -> object:
    headers = {"cookie": cookie, "host": f"127.0.0.1:{world.app.port}"}
    if csrf is not None:
        headers["x-csrf-token"] = csrf
    return dispatch(UiRequest("GET", path, STATIC_DIR, b"", headers, world.app))


def _post(
    world: World,
    path: str,
    cookie: str,
    csrf: str | None,
    *,
    host: str | None = None,
    origin: str | None = None,
) -> object:
    headers = {
        "cookie": cookie,
        "host": host if host is not None else f"127.0.0.1:{world.app.port}",
    }
    if csrf is not None:
        headers["x-csrf-token"] = csrf
    if origin is not None:
        headers["origin"] = origin
    return dispatch(UiRequest("POST", path, STATIC_DIR, b"", headers, world.app))


def _csrf(response: object) -> str:
    payload = json.loads(response.body)
    token = payload["csrf_token"]
    assert isinstance(token, str)
    return token


def _entries(world: World) -> list[object]:
    return world.log.read_all()


@contextmanager
def _running(app: AppState) -> Iterator[object]:
    httpd = make_server(0, app_state=app)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _exchange(
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    *,
    extra_host: str | None = None,
) -> tuple[int, http.client.HTTPMessage, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest(method, path, skip_host=True)
    merged = {"Host": f"127.0.0.1:{port}"}
    if headers:
        merged.update(headers)
    if body is not None:
        merged["Content-Length"] = str(len(body))
    for key, value in merged.items():
        conn.putheader(key, value)
    if extra_host is not None:
        conn.putheader("Host", extra_host)
    conn.endheaders()
    if body:
        conn.send(body)
    response = conn.getresponse()
    payload = response.read()
    status = response.status
    header_map = response.headers
    conn.close()
    return status, header_map, payload


def test_timeout_constants_are_pinned() -> None:
    assert IDLE_TIMEOUT_SECONDS == 1800
    assert ABSOLUTE_TIMEOUT_SECONDS == 28800
    assert FAILURE_WINDOW_SECONDS == 900
    assert BLOCK_SECONDS == 900
    assert MAX_FAILURES == 5
    assert SESSION_KEY_BYTES == 32


def test_enabled_role_is_operator_or_admin_only(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.sqlite")
    store.add_user("ada", "operator", _SECRET)
    store.add_user("bea", "admin", _OTHER)
    assert store.enabled_role("ada") == "operator"
    assert store.enabled_role("bea") == "admin"
    store.disable_user("ada")
    assert store.enabled_role("ada") is None
    assert store.enabled_role("missing") is None
    assert _SECRET.encode("utf-8") not in (tmp_path / "users.sqlite").read_bytes()


def test_login_sets_cookie_and_one_success_audit(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    before = len(_entries(world))
    response = _login(world, "ada", _SECRET)
    assert response.status == 200
    header = _cookie_header(response)
    assert header is not None
    _assert_cookie_flags(header)
    session_key = _cookie_value(response)
    assert _KEY_RE.fullmatch(session_key)
    assert len(session_key) == 43
    assert session_key.encode("ascii") not in response.body
    payload = json.loads(response.body)
    assert payload["ok"] is True
    assert "session" not in payload
    rows = _entries(world)
    assert len(rows) == before + 1
    entry = rows[-1]
    assert entry.action == "login"
    assert entry.operator_username == "ada"
    assert entry.details is None
    assert _get(world, f"session={session_key}").status == 501


def test_session_id_comes_from_token_urlsafe_32(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.add("ada", "operator", _SECRET)
    seen: list[int] = []
    real = secrets.token_urlsafe

    def _wrapped(nbytes: int = 32) -> str:
        seen.append(nbytes)
        return real(nbytes)

    monkeypatch.setattr("trace_search.ui.sessions.secrets.token_urlsafe", _wrapped)
    response = _login(world, "ada", _SECRET)
    assert seen == [SESSION_KEY_BYTES, SESSION_KEY_BYTES]
    assert _cookie_value(response) != ""
    assert len(_cookie_value(response)) == 43


def test_login_rotates_a_preset_cookie(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    planted = world.app.sessions.create("intruder", "operator")
    response = _login(world, "ada", _SECRET, cookie=f"session={planted.session_key}")
    issued = _cookie_value(response)
    assert issued != planted.session_key
    old = world.app.sessions.resolve(planted.session_key, touch=False)
    assert old is None
    fresh = world.app.sessions.resolve(issued, touch=False)
    assert fresh is not None
    assert fresh.username == "ada"
    garbage = _login(world, "ada", _SECRET, cookie="session=short")
    assert _cookie_value(garbage) != "short"


def test_logout_invalidates_the_server_session(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(world, "ada", _SECRET)
    session_key = _cookie_value(response)
    csrf = _csrf(response)
    cookie = f"session={session_key}"
    denied = _post(world, "/logout", cookie, None)
    assert denied.status == 403
    assert _get(world, cookie).status == 501
    wrong = _post(world, "/logout", cookie, "not-the-token")
    assert wrong.status == 403
    assert _get(world, cookie).status == 501
    gone = _post(world, "/logout", cookie, csrf)
    assert gone.status == 200
    clearing = _cookie_header(gone)
    assert clearing is not None
    _assert_cookie_flags(clearing)
    assert "max-age=0" in _flags(clearing)
    assert _get(world, cookie).status == 401
    replay = _post(world, "/logout", cookie, csrf)
    assert replay.status == 401


def test_csrf_rejects_another_sessions_token(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    world.add("bea", "admin", _OTHER)
    first = _login(world, "ada", _SECRET)
    second = _login(world, "bea", _OTHER)
    ada_cookie = f"session={_cookie_value(first)}"
    bea_csrf = _csrf(second)
    crossed = _post(world, "/logout", ada_cookie, bea_csrf)
    assert crossed.status == 403
    assert _get(world, ada_cookie).status == 501


def test_csrf_uses_compare_digest(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(world, "ada", _SECRET)
    cookie = f"session={_cookie_value(response)}"
    csrf = _csrf(response)
    calls: list[int] = []
    real_digest = hmac.compare_digest

    def _reject(left: object, right: object) -> bool:
        calls.append(1)
        assert isinstance(left, str)
        assert isinstance(right, str)
        return False

    monkeypatch.setattr("trace_search.ui.sessions.hmac.compare_digest", _reject)
    try:
        denied = _post(world, "/logout", cookie, csrf)
        assert calls
        assert denied.status == 403
        assert _get(world, cookie).status == 501
    finally:
        monkeypatch.setattr("trace_search.ui.sessions.hmac.compare_digest", real_digest)


def test_csrf_rejects_a_non_ascii_token(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(world, "ada", _SECRET)
    cookie = f"session={_cookie_value(response)}"
    denied = _post(world, "/logout", cookie, "bad-\u00e9")
    assert denied.status == 403
    assert _get(world, cookie).status == 501


def test_post_without_csrf_is_forbidden_and_does_not_run_the_handler(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(world, "ada", _SECRET)
    cookie = f"session={_cookie_value(response)}"
    csrf = _csrf(response)
    missing = _post(world, "/export", cookie, None)
    assert missing.status == 403
    assert json.loads(missing.body) == {"error": "forbidden"}
    allowed = _post(world, "/export", cookie, csrf)
    assert allowed.status == 501


def test_admin_route_requires_the_admin_role(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    world.add("bea", "admin", _OTHER)
    operator = _login(world, "ada", _SECRET)
    admin = _login(world, "bea", _OTHER)
    operator_cookie = f"session={_cookie_value(operator)}"
    admin_cookie = f"session={_cookie_value(admin)}"
    assert _post(world, "/retention", operator_cookie, _csrf(operator)).status == 403
    assert _post(world, "/retention", admin_cookie, _csrf(admin)).status == 501
    assert _get(world, admin_cookie).status == 501


def test_session_id_in_the_query_does_not_authenticate(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(world, "ada", _SECRET)
    session_key = _cookie_value(response)
    leaked = _get(world, "", path=f"/search?session={session_key}")
    assert leaked.status == 401
    assert session_key.encode("ascii") not in response.body
    assert response.header_pairs[0][0] == "Set-Cookie"


def test_idle_timeout_boundary(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    early = _login(world, "ada", _SECRET)
    early_cookie = f"session={_cookie_value(early)}"
    world.clock.current = 1799
    assert _get(world, early_cookie).status == 501
    world.clock.current = 0
    expired = _login(world, "ada", _SECRET)
    expired_cookie = f"session={_cookie_value(expired)}"
    world.clock.current = 1800
    assert _get(world, expired_cookie).status == 401


def test_idle_window_slides_and_absolute_is_not_swapped(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(world, "ada", _SECRET)
    cookie = f"session={_cookie_value(response)}"
    world.clock.current = 1799
    assert _get(world, cookie).status == 501
    world.clock.current = 1800
    assert _get(world, cookie).status == 501
    world.clock.current = 1799 + 1799
    assert _get(world, cookie).status == 501


def test_absolute_timeout_boundary(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(world, "ada", _SECRET)
    cookie = f"session={_cookie_value(response)}"
    stamp = 1700
    while stamp < 28800:
        world.clock.current = stamp
        assert _get(world, cookie).status == 501
        stamp += 1700
    world.clock.current = 28799
    assert _get(world, cookie).status == 501
    world.clock.current = 28800
    assert _get(world, cookie).status == 401


def test_failure_writes_one_detailed_audit_entry(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    before = len(_entries(world))
    response = _login(world, "ada", _OTHER)
    assert response.status == 401
    assert _cookie_header(response) is None
    assert len(world.app.sessions) == 0
    rows = _entries(world)
    assert len(rows) == before + 1
    assert rows[-1].action == "login"
    assert rows[-1].operator_username == "ada"
    assert rows[-1].details == "invalid credentials"


def test_unknown_and_disabled_users_are_failures(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    world.users.disable_user("ada")
    unknown = _login(world, "ghost", _SECRET)
    disabled = _login(world, "ada", _SECRET)
    assert unknown.status == 401
    assert disabled.status == 401
    rows = _entries(world)
    assert [row.operator_username for row in rows] == ["ghost", "ada"]
    assert [row.details for row in rows] == ["invalid credentials", "invalid credentials"]


def test_authenticate_runs_for_unknown_disabled_wrong_and_blocked(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.add("ada", "operator", _SECRET)
    world.add("ned", "operator", _OTHER)
    world.users.disable_user("ned")
    calls: list[str] = []
    real = UserStore.authenticate

    def _wrapped(self: UserStore, username: str, password: str) -> bool:
        calls.append(username)
        return real(self, username, password)

    monkeypatch.setattr(UserStore, "authenticate", _wrapped)
    assert _login(world, "ghost", _SECRET).status == 401
    assert _login(world, "ned", _OTHER).status == 401
    assert _login(world, "ada", _OTHER).status == 401
    for _ in range(4):
        assert _login(world, "ada", _OTHER).status == 401
    blocked = _login(world, "ada", _SECRET)
    assert blocked.status == 403
    assert calls[:3] == ["ghost", "ned", "ada"]
    assert calls[-1] == "ada"


def test_four_failures_do_not_block_and_five_do(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    world.clock.current = 100
    for _ in range(4):
        assert _login(world, "ada", _OTHER).status == 401
    assert _login(world, "ada", _SECRET).status == 200
    world.add("bea", "operator", _OTHER)
    world.clock.current = 300
    for _ in range(5):
        assert _login(world, "bea", _SECRET).status == 401
    blocked = _login(world, "bea", _OTHER)
    assert blocked.status == 403
    assert json.loads(blocked.body) == {"error": "blocked"}
    assert _cookie_header(blocked) is None
    rows = [row for row in _entries(world) if row.operator_username == "bea"]
    assert rows[-1].details == "blocked"
    assert rows[-1].action == "login"
    again = _login(world, "bea", _OTHER)
    assert again.status == 403
    assert _entries(world)[-1].details == "blocked"


def test_block_duration_and_window_boundaries(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    world.add("bea", "operator", _OTHER)
    world.clock.current = 0
    for _ in range(4):
        assert _login(world, "ada", _OTHER).status == 401
    world.clock.current = 899
    assert _login(world, "ada", _OTHER).status == 401
    assert _login(world, "ada", _SECRET).status == 403
    world.clock.current = 0
    for _ in range(4):
        assert _login(world, "bea", _SECRET).status == 401
    world.clock.current = 900
    assert _login(world, "bea", _SECRET).status == 401
    assert _login(world, "bea", _OTHER).status == 200


def test_block_lasts_fifteen_minutes_for_that_username_only(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    world.add("bea", "operator", _OTHER)
    world.clock.current = 100
    for _ in range(5):
        assert _login(world, "ada", _OTHER).status == 401
    world.clock.current = 999
    assert _login(world, "ada", _SECRET).status == 403
    assert _login(world, "bea", _OTHER).status == 200
    world.clock.current = 1000
    released = _login(world, "ada", _SECRET)
    assert released.status == 200
    assert _entries(world)[-1].details is None


def test_success_clears_failures_but_not_an_active_block(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    for _ in range(4):
        assert _login(world, "ada", _OTHER).status == 401
    assert _login(world, "ada", _SECRET).status == 200
    for _ in range(4):
        assert _login(world, "ada", _OTHER).status == 401
    assert _login(world, "ada", _SECRET).status == 200
    world.clock.current = 50
    for _ in range(5):
        assert _login(world, "ada", _OTHER).status == 401
    assert _login(world, "ada", _SECRET).status == 403
    assert _login(world, "ada", _SECRET).status == 403


def test_malformed_login_is_not_an_attempt(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    before = len(_entries(world))
    bodies = (
        b"user=a&secret=b",
        b"username=ada",
        b"username=ada&password=",
        b"username=+ada&password=x",
        b"username=ada&username=bea&password=x",
        b"\xff",
        b"{",
        json.dumps({"username": "ada", "password": 1}).encode("ascii"),
        json.dumps({"username": " ada", "password": _SECRET}).encode("utf-8"),
    )
    for body in bodies:
        response = dispatch(
            UiRequest(
                "POST",
                "/login",
                STATIC_DIR,
                body,
                {"host": f"127.0.0.1:{world.app.port}"},
                world.app,
            )
        )
        assert response.status == 401
        assert _SECRET.encode("utf-8") not in response.body
    assert len(_entries(world)) == before
    assert len(world.app.sessions) == 0


@pytest.mark.parametrize("as_json", [False, True])
def test_form_and_json_bodies_both_login(world: World, as_json: bool) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(world, "ada", _SECRET, as_json=as_json)
    assert response.status == 200
    assert _cookie_header(response) is not None


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1:{port}",
        "localhost:{port}",
        "[::1]:{port}",
    ],
)
def test_each_loopback_host_is_allowed(world: World, host: str) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(world, "ada", _OTHER, host=host.format(port=world.app.port))
    assert response.status == 401
    assert _entries(world)[-1].details == "invalid credentials"


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:{port}",
        "http://localhost:{port}",
        "http://[::1]:{port}",
    ],
)
def test_each_loopback_origin_is_allowed(world: World, origin: str) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(
        world,
        "ada",
        _OTHER,
        origin=origin.format(port=world.app.port),
    )
    assert response.status == 401
    assert _entries(world)[-1].details == "invalid credentials"


def test_missing_origin_is_allowed_and_empty_origin_is_refused(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    missing = _login(world, "ada", _OTHER)
    assert missing.status == 401
    assert len(_entries(world)) == 1
    refused = _login(world, "ada", _SECRET, origin="")
    assert refused.status == 403
    assert len(_entries(world)) == 1
    assert len(world.app.sessions) == 0


@pytest.mark.parametrize(
    "host",
    [
        "evil.example:{port}",
        "127.0.0.1:1",
        "127.0.0.1",
        "localhost",
        "[::1]",
        "LocalHost:{port}",
        "localhost.:{port}",
        "127.0.0.1.:{port}",
        "127.0.0.1:{port} ",
        "127.0.0.1:{port}.evil",
    ],
)
def test_foreign_host_variants_are_refused(world: World, host: str) -> None:
    world.add("ada", "operator", _SECRET)
    before = len(_entries(world))
    response = _login(world, "ada", _SECRET, host=host.format(port=world.app.port))
    assert response.status == 403
    assert json.loads(response.body) == {"error": "forbidden"}
    assert _cookie_header(response) is None
    assert len(_entries(world)) == before
    assert len(world.app.sessions) == 0


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "http://evil.example",
        "http://127.0.0.1:1",
        "http://LocalHost:{port}",
        "http://localhost.:{port}",
        "HTTP://localhost:{port}",
        "https://127.0.0.1:{port}",
        "http://127.0.0.1:{port}/",
        "http://127.0.0.1:{port}/login",
        " http://127.0.0.1:{port}",
    ],
)
def test_foreign_origin_variants_are_refused(
    world: World, monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    world.add("ada", "operator", _SECRET)
    before = len(_entries(world))
    calls = {"count": 0}
    real = UserStore.authenticate

    def _wrapped(self: UserStore, username: str, password: str) -> bool:
        calls["count"] += 1
        return real(self, username, password)

    monkeypatch.setattr(UserStore, "authenticate", _wrapped)
    response = _login(
        world,
        "ada",
        _SECRET,
        origin=origin.format(port=world.app.port),
    )
    assert response.status == 403
    assert _cookie_header(response) is None
    assert len(_entries(world)) == before
    assert len(world.app.sessions) == 0
    assert calls["count"] == 0


def test_refused_host_does_not_authenticate(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    world.add("ada", "operator", _SECRET)

    def _boom(self: UserStore, username: str, password: str) -> bool:
        raise AssertionError("authenticate called")

    monkeypatch.setattr(UserStore, "authenticate", _boom)
    response = _login(world, "ada", _SECRET, host="evil.example:9")
    assert response.status == 403
    assert len(world.app.sessions) == 0


def test_get_and_head_skip_the_host_check(world: World) -> None:
    page = dispatch(
        UiRequest(
            "GET",
            "/login",
            STATIC_DIR,
            b"",
            {"host": "evil.example:9"},
            world.app,
        )
    )
    assert page.status == 200
    head = dispatch(
        UiRequest(
            "HEAD",
            "/login",
            STATIC_DIR,
            b"",
            {"host": "evil.example:9"},
            world.app,
        )
    )
    assert head.status == 404


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def test_non_get_methods_require_a_loopback_host(world: World, method: str) -> None:
    refused = dispatch(
        UiRequest(method, "/login", STATIC_DIR, b"", {"host": "evil.example:9"}, world.app)
    )
    assert refused.status == 403
    allowed = dispatch(
        UiRequest(
            method,
            "/no-such",
            STATIC_DIR,
            b"",
            {"host": f"127.0.0.1:{world.app.port}"},
            world.app,
        )
    )
    assert allowed.status == 404


def test_missing_host_is_refused(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(world, "ada", _SECRET, include_host=False)
    assert response.status == 403
    assert len(world.app.sessions) == 0
    assert request_is_allowed(None, None, world.app.port) is False
    assert request_is_allowed(f"127.0.0.1:{world.app.port}", None, world.app.port) is True


def test_login_does_not_initialize_a_missing_audit_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path = tmp_path / "keys" / "audit.key"
    init_hmac_key(key_path)
    monkeypatch.setenv(TRACE_AUDIT_HMAC_KEY_ENV, key_path.read_text(encoding="utf-8").strip())
    log = AuditLog(tmp_path / default_log_path())
    calls = {"count": 0}

    def _refuse(self: AuditLog) -> None:
        calls["count"] += 1
        raise AssertionError("initialize called")

    monkeypatch.setattr(AuditLog, "initialize", _refuse)
    users = UserStore(tmp_path / "users.sqlite")
    users.add_user("ada", "operator", _SECRET)
    app = AppState(
        users=users,
        audit=log,
        sessions=SessionStore(),
        limiter=LoginLimiter(),
        port=8765,
    )
    world = World.__new__(World)
    world.app = app
    world.log = log
    world.users = users
    with pytest.raises(RuntimeError, match="not initialized"):
        _login(world, "ada", _SECRET)
    assert calls["count"] == 0
    assert len(app.sessions) == 0
    assert not log.head_path.is_file()
    with _running(app) as httpd:
        port = int(httpd.server_address[1])
        app.port = port
        status, _headers, body = _exchange(
            port,
            "POST",
            "/login",
            _body("ada", _SECRET),
        )
    assert status == 500
    assert _SECRET.encode("utf-8") not in body
    assert calls["count"] == 0
    assert len(app.sessions) == 0
    assert not log.head_path.is_file()


def test_verify_accepts_the_log_after_mixed_attempts(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    assert _login(world, "ada", _OTHER).status == 401
    success = _login(world, "ada", _SECRET)
    assert success.status == 200
    _check_log(world.log)
    assert world.log.verify().valid is True
    assert world.log.verify().total_entries == 2


def test_http_cookie_flags_logout_and_duplicate_host(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    with _running(world.app) as httpd:
        port = int(httpd.server_address[1])
        status, headers, body = _exchange(port, "POST", "/login", _body("ada", _SECRET))
        assert status == 200
        set_cookie = headers.get("set-cookie")
        assert set_cookie is not None
        _assert_cookie_flags(set_cookie)
        session_key = set_cookie.split(";", 1)[0].split("=", 1)[1]
        assert session_key.encode("ascii") not in body
        payload = json.loads(body)
        denied, _denied_headers, _denied_body = _exchange(
            port,
            "GET",
            f"/search?session={session_key}",
        )
        assert denied == 401
        authed, _authed_headers, _authed_body = _exchange(
            port,
            "GET",
            "/search",
            headers={"Cookie": f"session={session_key}"},
        )
        assert authed == 501
        logged_out, logout_headers, _logout_body = _exchange(
            port,
            "POST",
            "/logout",
            headers={
                "Cookie": f"session={session_key}",
                "X-CSRF-Token": payload["csrf_token"],
            },
        )
        assert logged_out == 200
        clearing = logout_headers.get("set-cookie")
        assert clearing is not None
        _assert_cookie_flags(clearing)
        replay, _replay_headers, _replay_body = _exchange(
            port,
            "GET",
            "/search",
            headers={"Cookie": f"theme=dark; session={session_key}"},
        )
        assert replay == 401
        foreign, _foreign_headers, foreign_body = _exchange(
            port,
            "POST",
            "/login",
            _body("ada", _SECRET),
            headers={"Host": "evil.example:9", "Origin": "null"},
        )
        assert foreign == 403
        assert _SECRET.encode("utf-8") not in foreign_body
        duplicate, _duplicate_headers, _duplicate_body = _exchange(
            port,
            "POST",
            "/login",
            _body("ada", _SECRET),
            extra_host="evil.example:9",
        )
        assert duplicate == 403
        reversed_hosts, _reversed_headers, _reversed_body = _exchange(
            port,
            "POST",
            "/login",
            _body("ada", _SECRET),
            headers={"Host": "evil.example:9"},
            extra_host=f"127.0.0.1:{port}",
        )
        assert reversed_hosts == 403
    _check_log(world.log)


def test_extra_cookie_pair_still_selects_the_session(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    response = _login(world, "ada", _SECRET)
    session_key = _cookie_value(response)
    allowed = _get(world, f"theme=dark; session={session_key}")
    assert allowed.status == 501


def test_server_start_creates_a_private_user_db_and_does_not_initialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}
    real_initialize = AuditLog.initialize

    def _count(self: AuditLog) -> None:
        calls["count"] += 1
        real_initialize(self)

    monkeypatch.setattr(AuditLog, "initialize", _count)
    for mask in (0o277, 0o022):
        data = tmp_path / f"data-{mask:03o}"
        data.mkdir()
        key_path = data / "keys" / "audit.key"
        init_hmac_key(key_path)
        monkeypatch.setenv(TRACE_AUDIT_HMAC_KEY_ENV, key_path.read_text(encoding="utf-8").strip())
        log = AuditLog(data / default_log_path())
        log.initialize()
        before = calls["count"]
        previous = os.umask(mask)
        try:
            result = build_server(["--data", str(data), "--host", "127.0.0.1", "--port", "0"])
        finally:
            os.umask(previous)
        try:
            assert not isinstance(result, int)
        finally:
            if not isinstance(result, int):
                result.server_close()
        db_path = data / "users.sqlite"
        mode = db_path.stat().st_mode & 0o777
        assert mode == 0o600
        assert db_path.stat().st_uid == os.getuid()
        assert calls["count"] == before
        assert log.verify().valid is True


def test_login_does_not_leak_file_descriptors(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    before = set(os.listdir("/proc/self/fd"))
    for _ in range(4):
        assert _login(world, "ada", _OTHER).status == 401
    assert _login(world, "ada", _SECRET).status == 200
    after = set(os.listdir("/proc/self/fd"))
    assert after == before


def test_logout_handler_rejects_a_missing_session_and_a_bad_csrf(world: World) -> None:
    from trace_search.ui.pages.login import handle_logout
    from trace_search.ui.routes import _from_handler

    world.add("ada", "operator", _SECRET)
    signed_in = _login(world, "ada", _SECRET)
    session_key = _cookie_value(signed_in)
    csrf = _csrf(signed_in)
    host = {"host": f"127.0.0.1:{world.app.port}"}
    missing = _from_handler(
        handle_logout(UiRequest("POST", "/logout", STATIC_DIR, b"", host, world.app))
    )
    assert missing.status == 401
    bad = _from_handler(
        handle_logout(
            UiRequest(
                "POST",
                "/logout",
                STATIC_DIR,
                b"",
                {**host, "cookie": f"session={session_key}", "x-csrf-token": "nope"},
                world.app,
            )
        )
    )
    assert bad.status == 403
    assert world.app.sessions.resolve(session_key, touch=False) is not None
    accepted = _from_handler(
        handle_logout(
            UiRequest(
                "POST",
                "/logout",
                STATIC_DIR,
                b"",
                {**host, "cookie": f"session={session_key}", "x-csrf-token": csrf},
                world.app,
            )
        )
    )
    assert accepted.status == 200
    assert world.app.sessions.resolve(session_key, touch=False) is None


def test_odd_cookies_do_not_authenticate(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    assert _get(world, "theme=dark").status == 401
    assert _get(world, 'session="a\nb"').status == 401
    world.app.sessions.invalidate(None)
    assert len(world.app.sessions) == 0


def test_login_fails_when_the_enabled_role_disappears(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.add("ada", "operator", _SECRET)

    def _missing(self: UserStore, username: str) -> None:
        return None

    monkeypatch.setattr(UserStore, "enabled_role", _missing)
    response = _login(world, "ada", _SECRET)
    assert response.status == 401
    assert _cookie_header(response) is None
    assert _entries(world)[-1].details == "invalid credentials"
    assert len(world.app.sessions) == 0


def test_json_that_is_not_an_object_is_not_an_attempt() -> None:
    from trace_search.ui.pages.login import _parse_json

    assert _parse_json(b"[1]") is None
    assert _parse_json(b"null") is None
    assert _parse_json(b'"ada"') is None


def test_empty_csrf_value_does_not_match() -> None:
    from trace_search.ui.sessions import csrf_matches

    assert csrf_matches("", "") is False
    assert csrf_matches("abc", "") is False
    assert csrf_matches("abc", None) is False


def test_non_canonical_session_key_is_rejected(world: World) -> None:
    from trace_search.ui.sessions import Session

    world.add("ada", "operator", _SECRET)
    planted = (
        "short-key",
        "a" * 42,
        "a" * 44,
        "+" + ("a" * 42),
        "a" * 43 + "=",
    )
    for key in planted:
        world.app.sessions._sessions[key] = Session(
            session_key=key,
            username="ada",
            role="operator",
            csrf_token="b" * 43,
            created_at=0.0,
            last_seen=0.0,
        )
        assert _get(world, f"session={key}").status == 401
        assert world.app.sessions.resolve(key, touch=False) is None
    assert _get(world, f"session={'a' * 43}").status == 401


def test_direct_host_helper_matches_the_allowlist() -> None:
    port = 8765
    assert request_is_allowed(f"127.0.0.1:{port}", None, port) is True
    assert request_is_allowed(f"localhost:{port}", None, port) is True
    assert request_is_allowed(f"[::1]:{port}", None, port) is True
    assert request_is_allowed("evil.example:8765", None, port) is False
    assert request_is_allowed(f"127.0.0.1:{port}", "null", port) is False
    assert request_is_allowed(f"127.0.0.1:{port}", f"http://127.0.0.1:{port}", port) is True
