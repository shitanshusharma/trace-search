"""Server-side sessions, login throttling, and the loopback Host/Origin check.

Session ids are ``secrets.token_urlsafe(32)`` and live only in memory. They are
never written into a URL. The audit log is not opened or initialized here.
"""

from __future__ import annotations

import hmac
import http.cookies
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

IDLE_TIMEOUT_SECONDS = 30 * 60
ABSOLUTE_TIMEOUT_SECONDS = 8 * 60 * 60
FAILURE_WINDOW_SECONDS = 15 * 60
BLOCK_SECONDS = 15 * 60
MAX_FAILURES = 5
SESSION_KEY_BYTES = 32

SESSION_COOKIE = "session"
_SESSION_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
_SAFE_METHODS = frozenset({"GET", "HEAD"})
_HTTP_SCHEME = "http" + "://"


@dataclass(frozen=True)
class Session:
    """One authenticated operator session."""

    session_key: str
    username: str
    role: str
    csrf_token: str
    created_at: float
    last_seen: float


def is_session_key(value: str) -> bool:
    """True for a canonical ``token_urlsafe(32)`` value."""
    return _SESSION_KEY_RE.fullmatch(value) is not None


def is_state_changing(method: str) -> bool:
    """True for every method except GET and HEAD."""
    return method not in _SAFE_METHODS


def bound_host_values(port: int) -> frozenset[str]:
    """Host headers that match this process's loopback port."""
    text = str(port)
    return frozenset(
        {
            f"127.0.0.1:{text}",
            f"localhost:{text}",
            f"[::1]:{text}",
        }
    )


def bound_origin_values(port: int) -> frozenset[str]:
    """Origins that match this process's loopback port."""
    text = str(port)
    prefix = _HTTP_SCHEME
    return frozenset(
        {
            f"{prefix}127.0.0.1:{text}",
            f"{prefix}localhost:{text}",
            f"{prefix}[::1]:{text}",
        }
    )


def request_is_allowed(host_header: str | None, origin_header: str | None, port: int) -> bool:
    """Refuse a foreign Host, and a present Origin that is not loopback.

    A missing Origin is allowed. Login and every other state-changing request
    still require this check. CSRF is a separate gate and is still required
    on authenticated state-changing requests when Origin is absent.
    """
    if host_header not in bound_host_values(port):
        return False
    if origin_header is None:
        return True
    return origin_header in bound_origin_values(port)


def csrf_matches(expected: str, provided: str | None) -> bool:
    """Constant-time CSRF compare. A missing or non-text value does not match."""
    if not isinstance(provided, str) or provided == "":
        return False
    try:
        return hmac.compare_digest(expected, provided)
    except (TypeError, ValueError):
        return False


def _cookie_attributes() -> str:
    return "HttpOnly; SameSite=Strict; Path=/"


def set_session_cookie(session_key: str) -> str:
    """Session cookie. No Secure flag: Cut 01 is loopback and has no TLS."""
    return f"{SESSION_COOKIE}={session_key}; {_cookie_attributes()}"


def clear_session_cookie() -> str:
    """Drop the session cookie. The server-side row is deleted separately."""
    return f"{SESSION_COOKIE}=; {_cookie_attributes()}; Max-Age=0"


def session_id_from_cookie(header: str | None) -> str | None:
    """Return the session cookie when it is canonical. Query strings are ignored."""
    if header is None or header == "":
        return None
    jar = http.cookies.SimpleCookie()
    try:
        jar.load(header)
    except http.cookies.CookieError:
        return None
    morsel = jar.get(SESSION_COOKIE)
    if morsel is None:
        return None
    value = morsel.value
    if not is_session_key(value):
        return None
    return value


def _idle_expired(session: Session, now: float) -> bool:
    return now - session.last_seen >= IDLE_TIMEOUT_SECONDS


def _absolute_expired(session: Session, now: float) -> bool:
    return now - session.created_at >= ABSOLUTE_TIMEOUT_SECONDS


def _expired(session: Session, now: float) -> bool:
    return _idle_expired(session, now) or _absolute_expired(session, now)


class SessionStore:
    """In-memory sessions. Nothing is written inside the repository."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = time.time if clock is None else clock
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _now(self) -> float:
        return float(self._clock())

    def create(self, username: str, role: str) -> Session:
        """Open a new session. The id is not derived from any request value."""
        with self._lock:
            now = self._now()
            session_key = secrets.token_urlsafe(SESSION_KEY_BYTES)
            csrf_token = secrets.token_urlsafe(SESSION_KEY_BYTES)
            session = Session(
                session_key=session_key,
                username=username,
                role=role,
                csrf_token=csrf_token,
                created_at=now,
                last_seen=now,
            )
            self._sessions[session_key] = session
            return session

    def resolve(self, session_key: str | None, *, touch: bool) -> Session | None:
        """Return a live session. Expired rows are deleted."""
        if session_key is None or not is_session_key(session_key):
            return None
        with self._lock:
            session = self._sessions.get(session_key)
            if session is None:
                return None
            now = self._now()
            if _expired(session, now):
                del self._sessions[session_key]
                return None
            if not touch:
                return session
            refreshed = replace(session, last_seen=now)
            self._sessions[session_key] = refreshed
            return refreshed

    def touch(self, session_key: str) -> None:
        """Slide the idle window for a session that is still live."""
        self.resolve(session_key, touch=True)

    def invalidate(self, session_key: str | None) -> None:
        """Forget a session. A second call is a no-op."""
        if session_key is None:
            return
        with self._lock:
            self._sessions.pop(session_key, None)


class LoginLimiter:
    """Five failures per username per 15 minutes, then a 15-minute block."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = time.time if clock is None else clock
        self._failures: dict[str, list[float]] = {}
        self._blocked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def _now(self) -> float:
        return float(self._clock())

    def decide(self, username: str, authenticated: bool) -> str:
        """Return ``ok``, ``invalid``, or ``blocked``.

        A successful password does not clear an active block. When there is
        no block, a success drops the remembered failures for that username.
        """
        with self._lock:
            now = self._now()
            if self._is_blocked(username, now):
                return "blocked"
            if not authenticated:
                self._record_failure(username, now)
                return "invalid"
            self._failures.pop(username, None)
            return "ok"

    def _is_blocked(self, username: str, now: float) -> bool:
        until = self._blocked_until.get(username)
        if until is None:
            return False
        if now < until:
            return True
        self._blocked_until.pop(username, None)
        return False

    def _record_failure(self, username: str, now: float) -> None:
        window = FAILURE_WINDOW_SECONDS
        kept = [stamp for stamp in self._failures.get(username, []) if now - stamp < window]
        kept.append(now)
        self._failures[username] = kept
        if len(kept) >= MAX_FAILURES:
            self._blocked_until[username] = now + BLOCK_SECONDS
