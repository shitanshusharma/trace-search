"""Login and logout handlers.

Login is public: it does not require a CSRF token, because there is no
session yet. It still refuses a bad Host or Origin before this module runs.
A refused request does not create a session and does not write an audit entry.
This module never calls ``AuditLog.initialize``.
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass

from trace_search.audit import AuditAction
from trace_search.ui.sessions import (
    AppState,
    UiRequest,
    clear_session_cookie,
    csrf_matches,
    session_id_from_cookie,
    set_session_cookie,
)

DETAIL_INVALID = "invalid credentials"
DETAIL_BLOCKED = "blocked"
MAX_USERNAME_LENGTH = 64


@dataclass(frozen=True)
class HandlerResult:
    """Status, optional error text, and an optional session cookie."""

    status: int
    error: str | None = None
    csrf_token: str | None = None
    set_cookie: str | None = None


def _usable_pair(username: object, password: object) -> tuple[str, str] | None:
    if not isinstance(username, str) or not isinstance(password, str):
        return None
    if len(username) == 0 or username != username.strip() or len(password) == 0:
        return None
    if len(username) > MAX_USERNAME_LENGTH:
        return None
    return username, password


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse a repeated JSON key instead of keeping the last value."""
    collected: dict[str, object] = {}
    for key, value in pairs:
        if key in collected:
            raise ValueError("duplicate key")
        collected[key] = value
    return collected


def _parse_json(body: bytes) -> tuple[str, str] | None:
    try:
        loaded = json.loads(body, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return None
    if isinstance(loaded, dict):
        return _usable_pair(loaded.get("username"), loaded.get("password"))
    return None


def _parse_form(body: bytes) -> tuple[str, str] | None:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return None
    parsed = urllib.parse.parse_qs(text, keep_blank_values=True)
    usernames = parsed.get("username") or []
    passwords = parsed.get("password") or []
    if len(usernames) != 1 or len(passwords) != 1:
        return None
    return _usable_pair(usernames[0], passwords[0])


def parse_login_body(body: bytes) -> tuple[str, str] | None:
    """Read a JSON object or a form body. Anything else is not an attempt."""
    if body.startswith(b"{"):
        return _parse_json(body)
    return _parse_form(body)


def _decision(app: AppState, username: str, password: str) -> tuple[str, str | None]:
    authenticated = app.users.authenticate(username, password)
    role: str | None = None
    if authenticated:
        role = app.users.enabled_role(username)
        if role is None:
            authenticated = False
    return app.limiter.decide(username, authenticated), role


def _audit(app: AppState, username: str, details: str) -> None:
    app.audit.append(username, AuditAction.LOGIN, details=details)


def _open_session(
    app: AppState,
    username: str,
    role: str,
    presented: str | None,
) -> HandlerResult:
    session = app.sessions.create(username, role)
    try:
        app.audit.append(username, AuditAction.LOGIN)
    except Exception:
        app.sessions.invalidate(session.session_key)
        raise
    if presented is not None and presented != session.session_key:
        app.sessions.invalidate(presented)
    return HandlerResult(
        200,
        csrf_token=session.csrf_token,
        set_cookie=set_session_cookie(session.session_key),
    )


def handle_login(request: UiRequest) -> HandlerResult:
    """Audit one attempt and, on success, issue a new session id."""
    app = request.app
    if app is None:
        return HandlerResult(401, error="unauthorized")
    parsed = parse_login_body(request.body)
    if parsed is None:
        return HandlerResult(401, error="unauthorized")
    username, password = parsed
    decision, role = _decision(app, username, password)
    if decision == "ok" and role is not None:
        presented = session_id_from_cookie(request.header_map.get("cookie"))
        return _open_session(app, username, role, presented)
    if decision == "blocked":
        _audit(app, username, DETAIL_BLOCKED)
        return HandlerResult(403, error="blocked")
    _audit(app, username, DETAIL_INVALID)
    return HandlerResult(401, error="unauthorized")


def handle_logout(request: UiRequest) -> HandlerResult:
    """End the server-side session. The old cookie then fails closed."""
    app = request.app
    if app is None:
        return HandlerResult(401, error="unauthorized")
    session_key = session_id_from_cookie(request.header_map.get("cookie"))
    session = app.sessions.resolve(session_key, touch=False)
    if session is None:
        return HandlerResult(401, error="unauthorized")
    provided = request.header_map.get("x-csrf-token")
    if not csrf_matches(session.csrf_token, provided):
        return HandlerResult(403, error="forbidden")
    app.sessions.invalidate(session.session_key)
    return HandlerResult(200, set_cookie=clear_session_cookie())
