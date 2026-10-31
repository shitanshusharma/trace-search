"""Route table for the local UI. There is no share, token, or public-view route."""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from trace_search.ui.sessions import (
    UiRequest as UiRequest,
)
from trace_search.ui.sessions import (
    csrf_matches,
    is_state_changing,
    request_is_allowed,
    session_id_from_cookie,
)

if TYPE_CHECKING:
    from trace_search.ui.pages.login import HandlerResult

AuthRule = Literal["public", "operator", "admin"]

_JSON = "application/json"
_STATIC_PREFIX = "/static/"
_DECODE_LIMIT = 5

STATIC_FILES: dict[str, str] = {
    "login.html": "text/html; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}


@dataclass(frozen=True)
class UiResponse:
    """One HTTP response. ``api`` marks replies that must not be stored."""

    status: int
    body: bytes
    content_type: str
    api: bool
    header_pairs: tuple[tuple[str, str], ...] = ()


Handler = Callable[[UiRequest], UiResponse]


@dataclass(frozen=True)
class Route:
    """One URL, its handler, and who may call it."""

    method: str
    path_pattern: str
    handler: Handler
    auth: AuthRule


def _json_response(error: str, status: int) -> UiResponse:
    payload = json.dumps({"error": error}, separators=(",", ":")).encode("ascii")
    return UiResponse(status, payload, _JSON, True)


def unauthorized() -> UiResponse:
    """Logged-out reply for a route that is not public."""
    return _json_response("unauthorized", 401)


def not_found() -> UiResponse:
    """Reply when no route matches, or a static path is refused."""
    return _json_response("not_found", 404)


def server_error() -> UiResponse:
    """Reply when a handler raises. The client does not see the exception."""
    return _json_response("error", 500)


def unavailable(_request: UiRequest) -> UiResponse:
    """Handler behind the auth gate. Dispatch never calls it while logged out."""
    return _json_response("not_implemented", 501)


def _path_only(url_path: str) -> str:
    return url_path.split("?", 1)[0].split("#", 1)[0]


def _forbidden_leaf(text: str) -> bool:
    if text in {"", ".", ".."}:
        return True
    if "/" in text or "\\" in text or "\x00" in text:
        return True
    return ".." in text


def _decode_leaf(raw: str) -> str | None:
    current = raw
    for _ in range(_DECODE_LIMIT):
        if _forbidden_leaf(current):
            return None
        try:
            decoded = urllib.parse.unquote(current, errors="strict")
        except UnicodeDecodeError:
            return None
        if decoded == current:
            return current
        current = decoded
    if _forbidden_leaf(current):
        return None
    return current


def static_leaf(url_path: str) -> str | None:
    """Return an allowlisted static file leaf, or None when the path is unsafe."""
    path = _path_only(url_path)
    if not path.startswith(_STATIC_PREFIX):
        return None
    decoded = _decode_leaf(path[len(_STATIC_PREFIX) :])
    if decoded is None or decoded not in STATIC_FILES:
        return None
    return decoded


def _contained_file(root: Path, leaf: str) -> Path | None:
    root_resolved = root.resolve()
    candidate = root_resolved / leaf
    if candidate.is_symlink() or not candidate.is_file():
        return None
    if candidate.resolve().parent != root_resolved:
        return None
    return candidate


def open_static(root: Path, leaf: str) -> tuple[bytes, str] | None:
    """Read one allowlisted file. The joined path is the table key, not the URL."""
    for known, content_type in STATIC_FILES.items():
        if leaf != known:
            continue
        candidate = _contained_file(root, known)
        if candidate is None:
            return None
        try:
            return candidate.read_bytes(), content_type
        except OSError:
            return None
    return None


def serve_login_page(request: UiRequest) -> UiResponse:
    """Return the sign-in page. Query strings are ignored."""
    opened = open_static(request.static_root, "login.html")
    if opened is None:
        return not_found()
    body, content_type = opened
    return UiResponse(200, body, content_type, False)


def _from_handler(result: HandlerResult) -> UiResponse:
    pairs: tuple[tuple[str, str], ...] = ()
    if result.set_cookie is not None:
        pairs = (("Set-Cookie", result.set_cookie),)
    if result.csrf_token is not None:
        body = json.dumps(
            {"csrf_token": result.csrf_token, "ok": True},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return UiResponse(result.status, body, _JSON, True, pairs)
    if result.error is not None:
        return _json_response(result.error, result.status)
    body = json.dumps({"ok": True}, separators=(",", ":"), sort_keys=True).encode("ascii")
    return UiResponse(result.status, body, _JSON, True, pairs)


def post_login(request: UiRequest) -> UiResponse:
    """Public login. A bad Host or Origin never reaches the handler."""
    from trace_search.ui.pages.login import handle_login

    return _from_handler(handle_login(request))


def post_logout(request: UiRequest) -> UiResponse:
    """End the session. Dispatch already requires a session and CSRF."""
    from trace_search.ui.pages.login import handle_logout

    return _from_handler(handle_logout(request))


def serve_static(request: UiRequest) -> UiResponse:
    """Serve one package static file. Traversal and symlinks are 404."""
    leaf = static_leaf(request.path)
    if leaf is None:
        return not_found()
    opened = open_static(request.static_root, leaf)
    if opened is None:
        return not_found()
    body, content_type = opened
    return UiResponse(200, body, content_type, False)


def match_route(method: str, url_path: str) -> Route | None:
    """Return the route for this method and path, ignoring the query string."""
    path = _path_only(url_path)
    for route in ROUTES:
        if route.method != method:
            continue
        if route.path_pattern == "/static/*":
            if path.startswith(_STATIC_PREFIX):
                return route
            continue
        if route.path_pattern == path:
            return route
    return None


def _origin_block(request: UiRequest) -> UiResponse | None:
    """403 when a state-changing request fails the loopback Host/Origin check."""
    app = request.app
    if app is None or not is_state_changing(request.method):
        return None
    host_header = request.header_map.get("host")
    origin_header = request.header_map.get("origin")
    if request_is_allowed(host_header, origin_header, app.port):
        return None
    return _json_response("forbidden", 403)


def _authorize(request: UiRequest, route: Route) -> UiResponse | None:
    """401 without a live session. 403 for the wrong role or a bad CSRF token."""
    app = request.app
    if app is None:
        return unauthorized()
    session_key = session_id_from_cookie(request.header_map.get("cookie"))
    session = app.sessions.resolve(session_key, touch=False)
    if session is None or session_key is None:
        return unauthorized()
    live_role = app.users.enabled_role(session.username)
    if live_role != session.role:
        app.sessions.invalidate(session_key)
        return unauthorized()
    if route.auth == "admin" and session.role != "admin":
        return _json_response("forbidden", 403)
    if is_state_changing(request.method):
        provided = request.header_map.get("x-csrf-token")
        if not csrf_matches(session.csrf_token, provided):
            return _json_response("forbidden", 403)
    app.sessions.touch(session_key)
    return None


def dispatch(request: UiRequest) -> UiResponse:
    """Answer one request. Protected routes need a live server-side session."""
    blocked = _origin_block(request)
    if blocked is not None:
        return blocked
    route = match_route(request.method, request.path)
    if route is None:
        return not_found()
    if route.auth != "public":
        denied = _authorize(request, route)
        if denied is not None:
            return denied
    return route.handler(request)


ROUTES: tuple[Route, ...] = (
    Route("GET", "/login", serve_login_page, "public"),
    Route("POST", "/login", post_login, "public"),
    Route("POST", "/logout", post_logout, "operator"),
    Route("GET", "/static/*", serve_static, "public"),
    Route("GET", "/search", unavailable, "operator"),
    Route("GET", "/route", unavailable, "operator"),
    Route("GET", "/clip", unavailable, "operator"),
    Route("POST", "/export", unavailable, "operator"),
    Route("POST", "/retention", unavailable, "admin"),
    Route("POST", "/purge", unavailable, "admin"),
)
