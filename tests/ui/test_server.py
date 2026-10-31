"""Web server skeleton: route table, headers, static files, and startup guards."""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from trace_search.audit import (
    TRACE_AUDIT_HMAC_KEY_ENV,
    AuditLog,
    default_log_path,
    init_hmac_key,
)
from trace_search.ui.cli import INIT_COMMAND, build_server, main, repo_root_from
from trace_search.ui.routes import ROUTES, UiRequest, dispatch, match_route
from trace_search.ui.server import (
    BIND_HOST,
    SECURITY_HEADERS,
    STATIC_DIR,
    host_is_loopback,
    make_server,
    serve_forever,
)

_REPO = Path(__file__).resolve().parents[2]
_HEADER_VALUES = {
    "content-security-policy": "default-src 'self'",
    "x-frame-options": "DENY",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
}
_TRAVERSAL_PATHS = (
    "/static/../server.py",
    "/static/../../pyproject.toml",
    "/static/%2e%2e/server.py",
    "/static/%2e%2e%2fserver.py",
    "/static/..%2fserver.py",
    "/static/%2e%2e%2f%2e%2e%2fserver.py",
    "/static/%252e%252e/server.py",
    "/static/%252e%252e%252fserver.py",
    "/static/%2fetc%2fpasswd",
    "/static/%2fetc/passwd",
    "/static//etc/passwd",
    "/static/app.js/../../server.py",
    "/static/....//....//server.py",
    "/static/app.js%00.txt",
    "/static/%5c..%5cserver.py",
    "/static/app.js%2f..%2f..%2fserver.py",
    "/static/%2e%2e",
    "/static/..",
    "/static/./app.css",
)
_SOURCE_MARKER = b"ThreadingHTTPServer"
_ROUTE_AUTH = {
    ("GET", "/login"): "public",
    ("POST", "/login"): "public",
    ("GET", "/static/*"): "public",
    ("GET", "/search"): "operator",
    ("GET", "/route"): "operator",
    ("GET", "/clip"): "operator",
    ("POST", "/export"): "operator",
    ("POST", "/retention"): "admin",
    ("POST", "/purge"): "admin",
}


def _snapshot(root: Path) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    if not root.exists():
        return found
    for item in root.rglob("*"):
        rel = item.relative_to(root).as_posix()
        if item.is_symlink():
            found[rel] = b"link:" + os.readlink(item).encode()
        elif item.is_file():
            found[rel] = item.read_bytes()
        else:
            found[rel] = b"dir"
    return found


def _ready_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    key_path = tmp_path / "keys" / "audit.key"
    init_hmac_key(key_path)
    monkeypatch.setenv(TRACE_AUDIT_HMAC_KEY_ENV, key_path.read_text(encoding="utf-8").strip())
    log = AuditLog(tmp_path / default_log_path())
    log.initialize()
    return tmp_path


def _close_if_server(result: object) -> None:
    if isinstance(result, int):
        return
    result.server_close()


def _request(
    port: int, method: str, path: str, body: bytes | None = None
) -> tuple[int, http.client.HTTPMessage, bytes]:
    last_error: Exception | None = None
    headers: dict[str, str] = {}
    if body is not None:
        headers["Content-Length"] = str(len(body))
    for _ in range(50):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            payload = response.read()
            status = response.status
            header_map = response.headers
            conn.close()
            return status, header_map, payload
        except OSError as exc:
            last_error = exc
            time.sleep(0.02)
    raise AssertionError(f"request failed: {last_error}")


@contextmanager
def _running(static_root: Path | None = None) -> Iterator[int]:
    httpd = make_server(0, static_root=static_root)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(httpd.server_address[1])
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _assert_security_headers(headers: http.client.HTTPMessage, *, api: bool) -> None:
    for key, value in _HEADER_VALUES.items():
        assert headers.get(key) == value
    if api:
        assert headers.get("cache-control") == "no-store"
    else:
        assert headers.get("cache-control") is None


def test_security_header_table_matches_every_reply() -> None:
    assert [key for key, _value in SECURITY_HEADERS] == [
        "Content-Security-Policy",
        "X-Frame-Options",
        "X-Content-Type-Options",
        "Referrer-Policy",
    ]
    assert [value for _key, value in SECURITY_HEADERS] == list(_HEADER_VALUES.values())


def test_every_route_has_one_auth_rule() -> None:
    found = {(route.method, route.path_pattern): route.auth for route in ROUTES}
    assert found == _ROUTE_AUTH
    for route in ROUTES:
        assert route.auth in {"public", "operator", "admin"}
        assert callable(route.handler)
    public = [(route.method, route.path_pattern) for route in ROUTES if route.auth == "public"]
    assert public == [("GET", "/login"), ("POST", "/login"), ("GET", "/static/*")]


def test_route_table_has_no_share_token_or_public_view() -> None:
    for route in ROUTES:
        lowered = route.path_pattern.lower()
        assert "share" not in lowered
        assert "token" not in lowered
        assert "public-view" not in lowered
        assert "public_view" not in lowered


@pytest.mark.parametrize(
    ("method", "path", "auth"),
    [(*key, value) for key, value in _ROUTE_AUTH.items() if value != "public"],
)
def test_logged_out_protected_route_is_unauthorized(method: str, path: str, auth: str) -> None:
    route = match_route(method, path)
    assert route is not None
    assert route.auth == auth
    direct = route.handler(UiRequest(method, path, STATIC_DIR))
    gated = dispatch(UiRequest(method, path, STATIC_DIR))
    assert direct.status == 501
    assert gated.status == 401
    assert json.loads(gated.body) == {"error": "unauthorized"}
    assert gated.api is True


def test_token_query_does_not_authorize_or_echo() -> None:
    response = dispatch(UiRequest("GET", "/search?token=secret", STATIC_DIR))
    assert response.status == 401
    assert b"secret" not in response.body


def test_login_page_ignores_query_string() -> None:
    response = dispatch(UiRequest("GET", "/login?next=/admin&token=abc", STATIC_DIR))
    assert response.status == 200
    assert response.api is False
    assert b"token" not in response.body
    assert b"/admin" not in response.body
    assert b"/static/app.css" in response.body
    assert b"/static/app.js" in response.body


def test_post_login_is_public_and_does_not_open_a_session() -> None:
    route = match_route("POST", "/login")
    assert route is not None
    assert route.auth == "public"
    response = dispatch(UiRequest("POST", "/login", STATIC_DIR, b"user=a&secret=b"))
    assert response.status == 401
    assert b"secret" not in response.body


@pytest.mark.parametrize("path", _TRAVERSAL_PATHS)
def test_static_traversal_is_not_found(path: str) -> None:
    response = dispatch(UiRequest("GET", path, STATIC_DIR))
    assert response.status == 404
    assert response.api is True
    assert _SOURCE_MARKER not in response.body
    assert json.loads(response.body) == {"error": "not_found"}


def test_encoded_static_leaf_is_served() -> None:
    plain = dispatch(UiRequest("GET", "/static/app.css", STATIC_DIR))
    encoded = dispatch(UiRequest("GET", "/static/app%2ecss", STATIC_DIR))
    nested = dispatch(UiRequest("GET", "/static/app%252525252ecss", STATIC_DIR))
    assert plain.status == 200
    assert encoded.status == 200
    assert nested.status == 200
    assert encoded.body == plain.body
    assert nested.body == plain.body
    assert encoded.api is False
    assert b"font-family" in encoded.body


def test_invalid_percent_encoding_is_not_found() -> None:
    response = dispatch(UiRequest("GET", "/static/%ff", STATIC_DIR))
    assert response.status == 404
    assert _SOURCE_MARKER not in response.body


def test_static_leaf_rejects_a_non_static_path() -> None:
    from trace_search.ui.routes import static_leaf

    assert static_leaf("/login") is None
    assert static_leaf("/static") is None


def test_missing_login_page_is_not_found(tmp_path: Path) -> None:
    response = dispatch(UiRequest("GET", "/login", tmp_path))
    assert response.status == 404


def test_unreadable_static_file_is_not_found(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    leaf = static_root / "app.css"
    leaf.write_text("hidden", encoding="utf-8")
    leaf.chmod(0)
    try:
        response = dispatch(UiRequest("GET", "/static/app.css", static_root))
    finally:
        leaf.chmod(0o644)
    assert response.status == 404
    assert b"hidden" not in response.body


def test_unknown_and_wrong_method_are_not_found() -> None:
    missing = dispatch(UiRequest("GET", "/missing", STATIC_DIR))
    wrong_method = dispatch(UiRequest("PUT", "/login", STATIC_DIR))
    export_get = dispatch(UiRequest("GET", "/export", STATIC_DIR))
    static_post = dispatch(UiRequest("POST", "/static/app.css", STATIC_DIR))
    assert missing.status == 404
    assert wrong_method.status == 404
    assert export_get.status == 404
    assert static_post.status == 404
    assert json.loads(missing.body) == {"error": "not_found"}


def test_login_case_is_not_rewritten() -> None:
    response = dispatch(UiRequest("GET", "/Login", STATIC_DIR))
    assert response.status == 404


def test_symlink_inside_static_is_refused(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    target = static_root / "login.html"
    target.write_text("inside", encoding="utf-8")
    link = static_root / "app.js"
    link.symlink_to(target)
    response = dispatch(UiRequest("GET", "/static/app.js", static_root))
    assert response.status == 404
    assert b"inside" not in response.body


def test_symlink_outside_static_is_refused(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_bytes(_SOURCE_MARKER)
    link = static_root / "app.css"
    link.symlink_to(secret)
    response = dispatch(UiRequest("GET", "/static/app.css", static_root))
    assert response.status == 404
    assert _SOURCE_MARKER not in response.body


def test_package_static_assets_exist_and_have_no_external_url() -> None:
    for leaf in ("login.html", "app.css", "app.js"):
        text = (STATIC_DIR / leaf).read_text(encoding="utf-8")
        assert "http://" not in text
        assert "https://" not in text
        assert "name=" not in text
    assert "use strict" in (STATIC_DIR / "app.js").read_text(encoding="utf-8")


def test_ui_tree_has_no_external_urls() -> None:
    root = _REPO / "src" / "trace_search" / "ui"
    suffixes = {".py", ".html", ".css", ".js", ".md"}
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if "__pycache__" in path.parts:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "http://" in line or "https://" in line:
                hits.append(f"{path}:{number}:{line}")
    assert hits == []


def test_socket_guard_refuses_public_addresses() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="ui socket guard allows only 127.0.0.1 and ::1"):
            probe.connect(("8.8.8.8", 9))
    finally:
        probe.close()


def test_socket_guard_allows_ipv4_and_ipv6_loopback() -> None:
    ipv4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError) as caught:
            ipv4.connect(("127.0.0.1", 1))
        assert "ui socket guard" not in str(caught.value)
    finally:
        ipv4.close()
    ipv6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError) as caught_v6:
            ipv6.connect(("::1", 1))
        assert "ui socket guard" not in str(caught_v6.value)
    finally:
        ipv6.close()


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.0.2.10", "localhost", "example.com", "", "127.0.0.1.example"],
)
def test_non_loopback_host_exits_1(
    tmp_path: Path, host: str, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _snapshot(tmp_path)
    code = main(["--data", str(tmp_path), "--host", host, "--port", "9"])
    assert code == 1
    assert host_is_loopback(host) is False
    error = capsys.readouterr().err
    assert f"refusing non-loopback host: {host}" in error
    assert _snapshot(tmp_path) == before
    assert not (_REPO / "audit_logs").exists()


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "[::1]", " 127.0.0.1 "])
def test_loopback_host_is_accepted_and_bind_stays_ipv4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
) -> None:
    data = _ready_data(tmp_path, monkeypatch)
    result = build_server(["--data", str(data), "--host", host, "--port", "0"])
    try:
        assert host_is_loopback(host) is True
        assert not isinstance(result, int)
        assert result.server_address[0] == BIND_HOST
        assert result.server_address[0] == "127.0.0.1"
    finally:
        _close_if_server(result)


def test_data_dir_inside_repo_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    before = _snapshot(_REPO / "src")
    code = main(["--data", str(_REPO / "src"), "--host", "127.0.0.1", "--port", "9"])
    error = capsys.readouterr().err
    assert code == 1
    assert "data dir is inside the repository" in error
    assert "AuditLog.initialize()" not in error
    assert _snapshot(_REPO / "src") == before


def test_relative_data_dir_inside_repo_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(_REPO)
    code = main(["--data", "src", "--host", "127.0.0.1", "--port", "9"])
    assert code == 1
    assert "data dir is inside the repository" in capsys.readouterr().err


def test_symlink_outside_repo_into_repo_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    link = tmp_path / "data"
    link.symlink_to(_REPO / "src", target_is_directory=True)
    code = main(["--data", str(link), "--host", "127.0.0.1", "--port", "9"])
    assert code == 1
    assert "data dir is inside the repository" in capsys.readouterr().err


def test_symlink_inside_repo_pointing_outside_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outside = _ready_data(tmp_path / "outside", monkeypatch)
    link = _REPO / ".ui-data-link-test"
    try:
        link.symlink_to(outside, target_is_directory=True)
        result = build_server(["--data", str(link), "--host", "127.0.0.1", "--port", "0"])
        try:
            assert result == 1
        finally:
            _close_if_server(result)
        assert "data dir is inside the repository" in capsys.readouterr().err
    finally:
        link.unlink(missing_ok=True)


def test_missing_audit_log_names_init_command_and_creates_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = {"count": 0}

    def _refuse_initialize(self: AuditLog) -> None:
        called["count"] += 1

    monkeypatch.setattr(AuditLog, "initialize", _refuse_initialize)
    before = _snapshot(tmp_path)
    result = build_server(["--data", str(tmp_path), "--host", "127.0.0.1", "--port", "0"])
    _close_if_server(result)
    error = capsys.readouterr().err
    assert result == 1
    assert called["count"] == 0
    assert INIT_COMMAND in error
    assert "AuditLog.initialize()" in error
    assert "init command" in error
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / default_log_path()).exists()


def test_missing_log_with_audit_dir_present_creates_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit_dir = tmp_path / default_log_path().parent
    audit_dir.mkdir(parents=True)
    before = _snapshot(tmp_path)
    code = main(["--data", str(tmp_path), "--host", "127.0.0.1", "--port", "9"])
    assert code == 1
    assert INIT_COMMAND in capsys.readouterr().err
    assert _snapshot(tmp_path) == before


def test_unverifiable_audit_log_refuses_without_creating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = _ready_data(tmp_path, monkeypatch)
    log_path = data / default_log_path()
    log_path.write_bytes(log_path.read_bytes() + b'\n{"broken":true}\n')
    called = {"count": 0}

    def _refuse_initialize(self: AuditLog) -> None:
        called["count"] += 1

    monkeypatch.setattr(AuditLog, "initialize", _refuse_initialize)
    before = _snapshot(tmp_path)
    result = build_server(["--data", str(data), "--host", "127.0.0.1", "--port", "0"])
    _close_if_server(result)
    error = capsys.readouterr().err
    assert result == 1
    assert called["count"] == 0
    assert "failed verify()" in error
    assert _snapshot(tmp_path) == before


def test_missing_hmac_key_fails_verify_without_creating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(TRACE_AUDIT_HMAC_KEY_ENV, raising=False)
    key_path = tmp_path / "keys" / "audit.key"
    init_hmac_key(key_path)
    log = AuditLog(tmp_path / default_log_path(), key_path=key_path)
    log.initialize()
    before = _snapshot(tmp_path)
    result = build_server(["--data", str(tmp_path), "--host", "127.0.0.1", "--port", "0"])
    _close_if_server(result)
    assert result == 1
    assert "failed verify()" in capsys.readouterr().err
    assert _snapshot(tmp_path) == before


def test_repo_root_lookup(tmp_path: Path) -> None:
    found = repo_root_from(_REPO / "src" / "trace_search" / "ui" / "cli.py")
    assert found == _REPO
    assert repo_root_from(_REPO / "src" / "trace_search" / "ui" / "missing.py") == _REPO
    assert repo_root_from(tmp_path / "nope.py") is None


def test_missing_repo_root_refuses_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = _ready_data(tmp_path, monkeypatch)
    monkeypatch.setattr("trace_search.ui.cli.repo_root_from", lambda _start: None)
    result = build_server(["--data", str(data), "--host", "127.0.0.1", "--port", "0"])
    _close_if_server(result)
    assert result == 1
    assert "repository root" in capsys.readouterr().err


def test_main_serves_on_ipv4_loopback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _ready_data(tmp_path, monkeypatch)
    seen: dict[str, object] = {}

    def _stop(httpd: object) -> None:
        seen["host"] = httpd.server_address[0]
        seen["port"] = httpd.server_address[1]
        httpd.server_close()

    monkeypatch.setattr("trace_search.ui.cli.serve_forever", _stop)
    code = main(["--data", str(data), "--port", "0"])
    assert code == 0
    assert seen["host"] == "127.0.0.1"


def test_script_refuses_non_loopback_host(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "serve_ui.py"),
            "--data",
            str(tmp_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "refusing non-loopback host: 0.0.0.0" in completed.stderr
    assert _snapshot(tmp_path) == {}


def test_codeql_ui_check_includes_package_static() -> None:
    text = (_REPO / ".github" / "workflows" / "codeql.yml").read_text(encoding="utf-8")
    assert "Check for UI code" in text
    assert "src/trace_search/ui/static" in text


def test_http_headers_on_ok_unauthorized_and_not_found() -> None:
    with _running() as port:
        ok, ok_headers, ok_body = _request(port, "GET", "/login")
        static, static_headers, static_body = _request(port, "GET", "/static/app.js")
        denied, denied_headers, denied_body = _request(port, "GET", "/search")
        missing, missing_headers, missing_body = _request(port, "GET", "/no-such")
        escaped, escaped_headers, escaped_body = _request(port, "GET", "/static/%2e%2e/server.py")
        dotted, dotted_headers, dotted_body = _request(port, "GET", "/static/../server.py")
    assert ok == 200
    assert b"<form" in ok_body
    _assert_security_headers(ok_headers, api=False)
    assert static == 200
    assert b"use strict" in static_body
    _assert_security_headers(static_headers, api=False)
    assert denied == 401
    assert json.loads(denied_body) == {"error": "unauthorized"}
    _assert_security_headers(denied_headers, api=True)
    assert missing == 404
    _assert_security_headers(missing_headers, api=True)
    assert escaped == 404
    assert _SOURCE_MARKER not in escaped_body
    _assert_security_headers(escaped_headers, api=True)
    assert dotted == 404
    assert _SOURCE_MARKER not in dotted_body
    _assert_security_headers(dotted_headers, api=True)


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
def test_extra_methods_keep_security_headers(method: str) -> None:
    with _running() as port:
        status, headers, body = _request(port, method, "/login")
    assert status == 404
    _assert_security_headers(headers, api=True)
    if method == "HEAD":
        assert body == b""
    else:
        assert json.loads(body) == {"error": "not_found"}


def test_bad_content_length_does_not_echo() -> None:
    with _running() as port:
        for raw_length in (b"nope", b"0", b"-1", b"2000000"):
            sock = socket.create_connection(("127.0.0.1", port), timeout=2)
            try:
                sock.sendall(
                    b"POST /login HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\n"
                    b"Content-Length: " + raw_length + b"\r\n"
                    b"Connection: close\r\n\r\n"
                )
                payload = b""
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    payload += chunk
            finally:
                sock.close()
            head, _, body = payload.partition(b"\r\n\r\n")
            assert b"401" in head.split(b"\r\n", 1)[0]
            assert b"secret" not in body
            lowered = head.lower()
            assert b"content-security-policy: default-src 'self'" in lowered
            assert b"cache-control: no-store" in lowered


def test_post_body_is_not_echoed() -> None:
    with _running() as port:
        status, headers, body = _request(port, "POST", "/login", b"user=a&secret=zz")
    assert status == 401
    assert b"secret" not in body
    assert b"zz" not in body
    _assert_security_headers(headers, api=True)


def test_unsupported_method_keeps_security_headers() -> None:
    with _running() as port:
        status, headers, body = _request(port, "TRACE", "/login")
    assert status == 501
    assert b"<html" not in body.lower()
    assert json.loads(body) == {"error": "error"}
    _assert_security_headers(headers, api=True)


def test_http_error_keeps_security_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_request: UiRequest) -> None:
        raise RuntimeError("secret failure detail")

    monkeypatch.setattr("trace_search.ui.server.dispatch", _boom)
    with _running() as port:
        status, headers, body = _request(port, "GET", "/login")
    assert status == 500
    assert b"secret failure detail" not in body
    assert json.loads(body) == {"error": "error"}
    _assert_security_headers(headers, api=True)


def test_http_symlink_is_refused(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"OUTSIDE")
    (static_root / "app.js").symlink_to(secret)
    with _running(static_root) as port:
        status, headers, body = _request(port, "GET", "/static/app.js")
    assert status == 404
    assert b"OUTSIDE" not in body
    _assert_security_headers(headers, api=True)


def test_serve_forever_answers_then_stops() -> None:
    httpd = make_server(0)
    thread = threading.Thread(target=serve_forever, args=(httpd,), daemon=True)
    thread.start()
    try:
        status, _headers, body = _request(int(httpd.server_address[1]), "GET", "/static/app.css")
        assert status == 200
        assert b"font-family" in body
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
