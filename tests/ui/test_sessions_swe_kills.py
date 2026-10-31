"""SWE kill tests for PR #75. Assertions match the reviewer's file."""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from tests.ui.test_sessions import (
    _OTHER,
    _SECRET,
    EPOCH,
    World,
    _body,
    _cookie_header,
    _cookie_value,
    _csrf,
    _entries,
    _exchange,
    _get,
    _login,
    _post,
    _running,
)
from trace_search.audit import TRACE_AUDIT_HMAC_KEY_ENV, AuditLog, default_log_path, init_hmac_key
from trace_search.ui.cli import build_server
from trace_search.ui.routes import UiRequest, dispatch
from trace_search.ui.server import STATIC_DIR
from trace_search.ui.sessions import (
    AppState,
    LoginLimiter,
    SessionStore,
    session_id_from_cookie,
)
from trace_search.ui.users import UserStore

pytest_plugins = ["tests.ui.test_sessions"]

_EPOCH = 1_700_000_000.0


def test_duplicate_session_cookie_fails_closed(world: World) -> None:
    """Two session cookies must not pick one silently."""
    world.add("ada", "operator", _SECRET)
    first = _cookie_value(_login(world, "ada", _SECRET))
    second = _cookie_value(_login(world, "ada", _SECRET))
    assert _get(world, f"session={first}").status == 501
    assert _get(world, f"session={first}; session={second}").status == 401
    assert _get(world, f"session=zzz; session={first}").status == 401
    assert session_id_from_cookie(f"session={first}; session={second}") is None


def test_csrf_must_match_in_full(world: World) -> None:
    """Kills a prefix compare and a strip of the provided token."""
    world.add("ada", "operator", _SECRET)
    signed_in = _login(world, "ada", _SECRET)
    cookie = f"session={_cookie_value(signed_in)}"
    token = _csrf(signed_in)
    tail = "A" if token[-1] != "A" else "B"
    assert _post(world, "/logout", cookie, token[:-1] + tail).status == 403
    assert _post(world, "/logout", cookie, token[:8] + "x" * (len(token) - 8)).status == 403
    assert _post(world, "/logout", cookie, token + " ").status == 403
    assert _post(world, "/logout", cookie, " " + token).status == 403
    assert _post(world, "/logout", cookie, token).status == 200


def test_cookie_headers_are_exact(world: World) -> None:
    """No Domain, Max-Age, or Expires on the session cookie. Order is pinned."""
    world.add("ada", "operator", _SECRET)
    signed_in = _login(world, "ada", _SECRET)
    key = _cookie_value(signed_in)
    assert _cookie_header(signed_in) == f"session={key}; HttpOnly; SameSite=Strict; Path=/"
    out = _post(world, "/logout", f"session={key}", _csrf(signed_in))
    assert _cookie_header(out) == "session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"


def test_cookie_name_is_case_sensitive(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    key = _cookie_value(_login(world, "ada", _SECRET))
    assert _get(world, f"Session={key}").status == 401
    assert _get(world, f"SESSION={key}").status == 401


def test_refused_requests_do_not_slide_the_idle_window(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    signed_in = _login(world, "ada", _SECRET)
    cookie = f"session={_cookie_value(signed_in)}"
    world.clock.current = EPOCH + 1000
    assert _post(world, "/logout", cookie, "wrong").status == 403
    assert _post(world, "/purge", cookie, _csrf(signed_in)).status == 403
    world.clock.current = EPOCH + 1800
    assert _get(world, cookie).status == 401


def test_expired_sessions_are_deleted() -> None:
    clock = {"t": _EPOCH}
    store = SessionStore(lambda: clock["t"])
    session = store.create("ada", "operator")
    clock["t"] = _EPOCH + 1800
    assert store.resolve(session.session_key, touch=False) is None
    assert len(store) == 0


def test_timeouts_with_a_real_epoch_and_fractions() -> None:
    clock = {"t": _EPOCH + 0.75}
    store = SessionStore(lambda: clock["t"])
    session = store.create("ada", "operator")
    clock["t"] = _EPOCH + 0.75 + 1799.5
    assert store.resolve(session.session_key, touch=False) is not None
    clock["t"] = _EPOCH + 0.75 + 1800
    assert store.resolve(session.session_key, touch=False) is None
    live = store.create("bea", "operator")
    for step in range(1, 17):
        clock["t"] = _EPOCH + 0.75 + 1800 + step * 1700
        assert store.resolve(live.session_key, touch=True) is not None
    clock["t"] = _EPOCH + 0.75 + 1800 + 28800 - 0.5
    assert store.resolve(live.session_key, touch=False) is not None
    clock["t"] = _EPOCH + 0.75 + 1800 + 28800
    assert store.resolve(live.session_key, touch=False) is None


def test_limiter_with_a_real_epoch() -> None:
    clock = {"t": _EPOCH + 0.2}
    limiter = LoginLimiter(lambda: clock["t"])
    for _ in range(5):
        assert limiter.decide("ada", False) == "invalid"
    assert limiter.decide("ada", True) == "blocked"
    clock["t"] = _EPOCH + 0.2 + 899.9
    assert limiter.decide("ada", True) == "blocked"


def test_block_runs_from_the_fifth_failure() -> None:
    clock = {"t": _EPOCH}
    limiter = LoginLimiter(lambda: clock["t"])
    for step in range(5):
        clock["t"] = _EPOCH + step * 200
        assert limiter.decide("ada", False) == "invalid"
    clock["t"] = _EPOCH + 800 + 899
    assert limiter.decide("ada", True) == "blocked"
    clock["t"] = _EPOCH + 800 + 900
    assert limiter.decide("ada", True) == "ok"


def test_limiter_keys_the_exact_username() -> None:
    limiter = LoginLimiter(lambda: _EPOCH)
    for _ in range(5):
        limiter.decide("ADA", False)
    assert limiter.decide("ADA", True) == "blocked"
    assert limiter.decide("ada", True) == "ok"


class _SlowDict(dict[str, list[float]]):
    def get(self, key: str, default: list[float] | None = None) -> list[float] | None:
        value = super().get(key, default)
        threading.Event().wait(0.002)
        return value


def test_limiter_counts_concurrent_failures_exactly() -> None:
    limiter = LoginLimiter(lambda: _EPOCH)
    limiter._failures = _SlowDict()
    gate = threading.Barrier(20)
    results: list[str] = []
    guard = threading.Lock()

    def _guess() -> None:
        gate.wait()
        outcome = limiter.decide("ada", False)
        with guard:
            results.append(outcome)

    threads = [threading.Thread(target=_guess) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count("invalid") == 5
    assert results.count("blocked") == 15


def test_authenticate_runs_on_every_blocked_attempt(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.add("ada", "operator", _SECRET)
    calls: list[str] = []
    real = UserStore.authenticate

    def _wrapped(self: UserStore, username: str, password: str) -> bool:
        calls.append(username)
        return real(self, username, password)

    monkeypatch.setattr(UserStore, "authenticate", _wrapped)
    for _ in range(5):
        assert _login(world, "ada", _OTHER).status == 401
    for _ in range(3):
        assert _login(world, "ada", _SECRET).status == 403
    assert len(calls) == 8


def test_vanished_role_counts_as_a_failure(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    world.add("ada", "operator", _SECRET)
    monkeypatch.setattr(UserStore, "enabled_role", lambda self, username: None)
    for _ in range(5):
        assert _login(world, "ada", _SECRET).status == 401
    assert _login(world, "ada", _SECRET).status == 403


def test_failed_login_with_a_missing_audit_log_does_not_initialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path = tmp_path / "keys" / "audit.key"
    init_hmac_key(key_path)
    monkeypatch.setenv(TRACE_AUDIT_HMAC_KEY_ENV, key_path.read_text(encoding="utf-8").strip())
    log = AuditLog(tmp_path / default_log_path())
    users = UserStore(tmp_path / "users.sqlite")
    users.add_user("ada", "operator", _SECRET)
    app = AppState(
        users=users, audit=log, sessions=SessionStore(), limiter=LoginLimiter(), port=8765
    )
    request = UiRequest(
        "POST", "/login", STATIC_DIR, _body("ada", _OTHER), {"host": "127.0.0.1:8765"}, app
    )
    with pytest.raises(Exception):
        dispatch(request)
    assert not log.log_path.exists()
    assert not log.head_path.exists()


@pytest.mark.parametrize(
    "body",
    [
        b"username=&password=x",
        b'{"username": "", "password": "x"}',
        b"username=ada&username=&password=session-secret-value",
    ],
)
def test_blank_usernames_are_not_attempts(world: World, body: bytes) -> None:
    world.add("ada", "operator", _SECRET)
    response = dispatch(
        UiRequest("POST", "/login", STATIC_DIR, body, {"host": "127.0.0.1:8765"}, world.app)
    )
    assert response.status == 401
    assert len(world.app.sessions) == 0
    assert _entries(world) == []


def test_one_character_username_logs_in(world: World) -> None:
    world.add("a", "operator", _SECRET)
    assert _login(world, "a", _SECRET).status == 200


def test_login_without_app_state_is_unauthorized() -> None:
    request = UiRequest("POST", "/login", STATIC_DIR, _body("ada", _SECRET), {}, None)
    assert dispatch(request).status == 401


def test_unspecified_address_host_is_refused(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    port = world.app.port
    assert _login(world, "ada", _SECRET, host=f"0.0.0.0:{port}").status == 403
    assert _login(world, "ada", _SECRET, origin=f"http://0.0.0.0:{port}").status == 403
    assert len(world.app.sessions) == 0


def test_http_server_reads_origin_and_fails_closed_on_duplicates(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    with _running(world.app) as httpd:
        port = int(httpd.server_address[1])
        foreign, _h, _b = _exchange(
            port,
            "POST",
            "/login",
            _body("ada", _SECRET),
            headers={"Origin": "http://evil.example"},
        )
        assert foreign == 403
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        body = _body("ada", _SECRET)
        conn.putrequest("POST", "/login", skip_host=True)
        conn.putheader("Host", f"127.0.0.1:{port}")
        conn.putheader("Origin", f"http://127.0.0.1:{port}")
        conn.putheader("Origin", "http://evil.example")
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders()
        conn.send(body)
        duplicate = conn.getresponse()
        duplicate.read()
        conn.close()
        assert duplicate.status == 403
    assert len(world.app.sessions) == 0
    assert _entries(world) == []


def test_http_get_with_duplicate_host_still_reads_the_cookie(world: World) -> None:
    world.add("ada", "operator", _SECRET)
    with _running(world.app) as httpd:
        port = int(httpd.server_address[1])
        status, headers, _body_bytes = _exchange(port, "POST", "/login", _body("ada", _SECRET))
        assert status == 200
        key = headers["set-cookie"].split(";", 1)[0].split("=", 1)[1]
        authed, _h, _b = _exchange(
            port, "GET", "/search", headers={"Cookie": f"session={key}"}, extra_host="other:1"
        )
        assert authed == 501


def test_built_server_audits_into_the_verified_log(
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
        UserStore(data / "users.sqlite").add_user("ada", "operator", _SECRET)
        port = app.port
        response = dispatch(
            UiRequest(
                "POST",
                "/login",
                STATIC_DIR,
                _body("ada", _SECRET),
                {"host": f"127.0.0.1:{port}"},
                app,
            )
        )
        assert response.status == 200
        assert json.loads(response.body)["ok"] is True
    finally:
        server.server_close()
    entries = log.read_all()
    assert [entry.operator_username for entry in entries] == ["ada"]
    assert log.verify().valid


def test_space_in_another_cookie_value_does_not_smuggle_a_session(world: World) -> None:
    """A planted ``theme=x session=<other key>`` must not run the request as that other key."""
    world.add("ada", "admin", _SECRET)
    world.add("mal", "operator", _OTHER)
    victim = _login(world, "ada", _SECRET)
    attacker = _login(world, "mal", _OTHER)
    legit = _cookie_value(victim)
    evil = _cookie_value(attacker)
    header = f"session={legit}; theme=x session={evil}"
    assert session_id_from_cookie(header) in (legit, None)
    assert session_id_from_cookie(f"session={legit} session={evil}") is None
    assert session_id_from_cookie(f"session={legit}\tsession={evil}") is None
    assert session_id_from_cookie(f"session={legit}\nsession={evil}") is None
    status = _post(world, "/retention", header, _csrf(victim)).status
    assert status in (501, 401)


def test_empty_cookie_pieces_do_not_hide_a_duplicate(world: World) -> None:
    """An empty piece must not stop the walk, or ``session=A;; session=B`` is accepted."""
    world.add("ada", "operator", _SECRET)
    first = _cookie_value(_login(world, "ada", _SECRET))
    second = _cookie_value(_login(world, "ada", _SECRET))
    assert session_id_from_cookie(f"session={first};; session={second}") is None
    assert _get(world, f"session={first};; session={second}").status == 401


def test_comma_inside_another_cookie_value_is_not_a_separator(world: World) -> None:
    """``theme=a,session`` is one pair. A comma is a legal value character."""
    world.add("ada", "operator", _SECRET)
    key = _cookie_value(_login(world, "ada", _SECRET))
    assert session_id_from_cookie(f"theme=a,session; session={key}") == key


def test_space_before_equals_in_the_session_pair_is_pinned(world: World) -> None:
    """``session =K`` is the session cookie. The space before ``=`` is padding."""
    world.add("ada", "operator", _SECRET)
    key = _cookie_value(_login(world, "ada", _SECRET))
    assert session_id_from_cookie(f"session ={key}") == key


def test_space_around_equals_in_the_session_pair_is_padding(world: World) -> None:
    """``session= K`` and a trailing space are the same cookie. A newline is not."""
    world.add("ada", "operator", _SECRET)
    key = _cookie_value(_login(world, "ada", _SECRET))
    assert session_id_from_cookie(f"session= {key}") == key
    assert session_id_from_cookie(f"session={key} ") == key
    assert session_id_from_cookie(f"session\t={key}") == key
    assert session_id_from_cookie(f"session=\n{key}") is None
    assert session_id_from_cookie(f"session\n={key}") is None


def test_username_cap_counts_characters_not_bytes(world: World) -> None:
    """A 64-character non-ASCII name is still an attempt. The cap is not UTF-8 bytes."""
    name = "\u00e9" * 64
    response = _login(world, name, _OTHER)
    assert response.status == 401
    assert _entries(world)[-1].operator_username == name


def test_other_apps_odd_cookie_does_not_drop_our_session(world: World) -> None:
    """Another localhost app's cookie, placed first, must not hide our session."""
    world.add("ada", "operator", _SECRET)
    key = _cookie_value(_login(world, "ada", _SECRET))
    for other in (
        'prefs={"a":1}',
        'x=a"b',
        "x=a b",
        "x=caf\u00e9",
        "x=a\\b",
        'x="a\x01b"',
    ):
        header = f"{other}; session={key}"
        assert session_id_from_cookie(header) == key, header
        assert _get(world, header).status == 501, header


def test_quoted_session_value_is_not_canonical(world: World) -> None:
    """The cookie we set is not quoted and not backslash-escaped."""
    world.add("ada", "operator", _SECRET)
    key = _cookie_value(_login(world, "ada", _SECRET))
    assert session_id_from_cookie(f'session="{key}"') is None
    assert session_id_from_cookie(f'session="{key[:5]}\\{key[5:]}"') is None


def test_empty_pieces_and_foreign_names_do_not_hide_one_session(world: World) -> None:
    """One real session still counts when other pairs are empty or not named session."""
    world.add("ada", "operator", _SECRET)
    key = _cookie_value(_login(world, "ada", _SECRET))
    assert session_id_from_cookie(f"theme=x;; session={key}") == key
    assert session_id_from_cookie(f"\x01bad; session={key}") == key
    assert session_id_from_cookie(f"Session=x; session={key}") == key
    assert session_id_from_cookie(f"sessionx=1; session={key}") == key
    assert session_id_from_cookie(f"a=b; c=d; session={key}") == key
    assert session_id_from_cookie(f"session={key}; a=b; c=d") == key
    assert _get(world, f"theme=x;; session={key}").status == 501


def _two(world: World) -> tuple[str, str]:
    world.add("ada", "operator", _SECRET)
    world.add("bea", "operator", _OTHER)
    return _cookie_value(_login(world, "ada", _SECRET)), _cookie_value(_login(world, "bea", _OTHER))


def test_identical_duplicate_session_pairs_fail_closed(world: World) -> None:
    """The session name must appear once, even when both values are the same key."""
    a, _b = _two(world)
    assert session_id_from_cookie(f"session={a}; session={a}") is None
    assert _get(world, f"session={a}; theme=dark; session={a}").status == 401


def test_second_session_with_an_equals_in_its_value_still_counts(world: World) -> None:
    """``session=x=y`` is a second session pair. The name is the text before the first ``=``."""
    a, b = _two(world)
    assert session_id_from_cookie(f"session={a}; session=x=y") is None
    assert session_id_from_cookie(f"session=x=y; session={a}") is None
    assert session_id_from_cookie(f"session={a}; session={b}=") is None


def test_a_name_ending_in_session_is_a_different_cookie(world: World) -> None:
    """``xsession`` and ``my_session`` are other cookies."""
    a, b = _two(world)
    assert session_id_from_cookie(f"xsession={b}; session={a}") == a
    assert session_id_from_cookie(f"session={a}; my_session={b}") == a
    assert session_id_from_cookie(f"xsession={a}") is None


def test_repeated_cookie_headers_over_http_fail_closed(world: World) -> None:
    """Two Cookie headers are refused. One Cookie header still authenticates."""
    a, _b = _two(world)
    with _running(world.app) as httpd:
        port = int(httpd.server_address[1])
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("GET", "/search", skip_host=True)
        conn.putheader("Host", f"127.0.0.1:{port}")
        conn.putheader("Cookie", f"session={a}")
        conn.putheader("Cookie", "theme=dark")
        conn.endheaders()
        response = conn.getresponse()
        response.read()
        conn.close()
        assert response.status == 401
        ok, _h, _p = _exchange(port, "GET", "/search", headers={"Cookie": f"session={a}"})
        assert ok == 501


def test_nbsp_is_not_cookie_padding(world: World) -> None:
    """Only space and tab are padding. A leading NBSP is part of the name."""
    world.add("ada", "operator", _SECRET)
    key = _cookie_value(_login(world, "ada", _SECRET))
    assert session_id_from_cookie(f"\xa0session={key}") is None
    assert session_id_from_cookie(f"session={key}\xa0") is None
    assert _get(world, f"\xa0session={key}").status == 401


KEY = "Ab3_-" * 8 + "Ab3"  # 43 chars, canonical alphabet


def test_tab_after_equals_is_padding() -> None:
    assert len(KEY) == 43
    assert session_id_from_cookie(f"session=\t{KEY}") == KEY
    assert session_id_from_cookie(f"theme=dark; session=\t{KEY}; x=1") == KEY


def test_tab_inside_the_value_is_not_padding() -> None:
    assert session_id_from_cookie(f"session={KEY[:20]}\t{KEY[20:]}") is None


def test_only_space_and_tab_are_padding_around_a_pair() -> None:
    # Kills widening the piece strip to CR/LF (q_add_crlf) or to VT/FF
    # (q_all_add_vt): those characters are part of the name, so no pair is
    # named ``session``.
    for junk in ("\r", "\n", "\r\n", "\x0b", "\x0c", "\x1c", "\x85"):
        assert session_id_from_cookie(f"{junk}session={KEY}") is None, repr(junk)
        assert session_id_from_cookie(f"session={KEY}{junk}") is None, repr(junk)
        assert session_id_from_cookie(f"session{junk}={KEY}") is None, repr(junk)
        assert session_id_from_cookie(f"session={junk}{KEY}") is None, repr(junk)
