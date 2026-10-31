"""Startup checks for ``scripts/serve_ui.py``.

The audit log is never created here. A missing log names ``AuditLog.initialize()``
and returns before any directory or file is written.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trace_search.ui.pages.login import AppState

from trace_search.audit import AuditLog, default_log_path
from trace_search.ui.server import BIND_HOST, UiServer, host_is_loopback, make_server, serve_forever

INIT_COMMAND = "AuditLog.initialize()"


def repo_root_from(start: Path) -> Path | None:
    """Return the checkout that contains ``start``, if one can be proved."""
    try:
        current = start.resolve()
    except (OSError, RuntimeError):
        return None
    for parent in (current, *current.parents):
        marker = parent / "pyproject.toml"
        package = parent / "src" / "trace_search"
        if marker.is_file() and package.is_dir():
            return parent
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the local trace-search UI.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--host", default=BIND_HOST)
    parser.add_argument("--port", type=int, default=8765)
    return parser


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def outside_repo(data_dir: Path, root: Path) -> Path:
    """Resolve ``data_dir`` and refuse a path that sits inside the checkout."""
    expanded = data_dir.expanduser()
    try:
        lexical = Path(os.path.abspath(expanded))
        resolved = expanded.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"data dir is inside the repository: {data_dir}") from exc
    root_resolved = root.resolve()
    lexical_inside = _within(lexical, root_resolved)
    resolved_inside = _within(resolved, root_resolved)
    if lexical_inside or resolved_inside:
        shown = resolved if resolved_inside else lexical
        raise ValueError(f"data dir is inside the repository: {shown}")
    return resolved


def _host_problem(host: str) -> str | None:
    if host_is_loopback(host):
        return None
    return f"refusing non-loopback host: {host}"


def _audit_problem(data_dir: Path) -> str | None:
    log_path = data_dir / default_log_path()
    if not log_path.is_file():
        return (
            f"audit log is missing at {log_path}. "
            f"Run the init command {INIT_COMMAND} before starting the server. "
            "Nothing was created."
        )
    outcome = AuditLog(log_path).verify()
    if outcome.valid:
        return None
    detail = outcome.error_message or "unverifiable"
    return f"audit log at {log_path} failed verify(): {detail}. Refusing to start."


def _checked_data_dir(data_dir: Path, root: Path) -> tuple[Path | None, str | None]:
    try:
        return outside_repo(data_dir, root), None
    except ValueError as exc:
        return None, str(exc)


def build_server(argv: list[str] | None = None) -> UiServer | int:
    """Bind a server, or return 1 when host, data dir, or audit log is refused."""
    parsed = _parser().parse_args(argv)
    host_problem = _host_problem(str(parsed.host))
    if host_problem is not None:
        print(host_problem, file=sys.stderr)
        return 1
    root = repo_root_from(Path(__file__))
    if root is None:
        print("cannot locate the repository root; refusing to start", file=sys.stderr)
        return 1
    data_dir, data_problem = _checked_data_dir(Path(parsed.data), root)
    if data_problem is not None or data_dir is None:
        print(data_problem or "data dir is inside the repository", file=sys.stderr)
        return 1
    audit_problem = _audit_problem(data_dir)
    if audit_problem is not None:
        print(audit_problem, file=sys.stderr)
        return 1
    return make_server(int(parsed.port), app_state=_live_app(data_dir, int(parsed.port)))


def _live_app(data_dir: Path, port: int) -> AppState:
    """Open the user db and attach in-memory sessions. The audit log already exists."""
    from trace_search.ui.pages.login import AppState
    from trace_search.ui.sessions import LoginLimiter, SessionStore
    from trace_search.ui.users import UserStore

    clock = time.time
    return AppState(
        users=UserStore(data_dir / "users.sqlite"),
        audit=AuditLog(data_dir / default_log_path()),
        sessions=SessionStore(clock),
        limiter=LoginLimiter(clock),
        port=port,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the startup checks and serve until the process is stopped."""
    built = build_server(argv)
    if isinstance(built, int):
        return built
    serve_forever(built)
    return 0
