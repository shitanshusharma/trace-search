"""Purge entry point. The audit directory is never a data root.

``default_audit_root`` is ``Path(AUDIT_LOG_DIR)`` imported from
``trace_search.audit.log``. ``validate_roots`` runs before any store is
opened. A data root that equals the audit root, contains it, or resolves
into or onto it through a symlink is refused. ``purge_expired`` then asks
each ``purgeables`` store to delete and returns the sum of those counts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from trace_search.audit.log import AUDIT_LOG_DIR
from trace_search.retention.registry import purgeables

_Clock = Callable[[], float]


def default_audit_root() -> Path:
    """Relative audit directory. ``validate_roots`` resolves it against the cwd."""
    return Path(AUDIT_LOG_DIR)


def _resolved_directory(path: Path, label: str) -> Path:
    """Expand and resolve ``path``. Refuse a non-path, a loop, or a file."""
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a Path, got {path!r}")
    try:
        resolved = path.expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} cannot be resolved: {path}") from exc
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"{label} must be a directory, got {resolved}")
    return resolved


def _strictly_contains(outer: Path, inner: Path) -> bool:
    """True when ``inner`` is strictly inside ``outer``.

    Both arguments are already resolved. Equal paths are not containment,
    so an equality check has to refuse that case on its own.
    """
    if inner == outer:
        return False
    try:
        inner.relative_to(outer)
    except ValueError:
        return False
    return True


def validate_roots(data_root: Path, audit_root: Path) -> None:
    """Refuse a data root the purge could follow into the audit directory.

    Comparison uses ``Path.resolve``, so a symlink onto the audit directory
    compares equal and a symlink into it compares as containment.
    """
    data = _resolved_directory(data_root, "data_root")
    audit = _resolved_directory(audit_root, "audit_root")
    if _strictly_contains(data, audit) or _strictly_contains(audit, data):
        raise ValueError(
            f"data root contains the audit root or resolves into it: data={data} audit={audit}"
        )
    if data == audit:
        raise ValueError(f"data root equals the audit root: {data}")


def purge_expired(
    data_root: Path,
    audit_root: Path | None = None,
    *,
    now: float | None = None,
    clock: _Clock | None = None,
) -> int:
    """Delete expired footage rows under ``data_root`` and return how many.

    ``audit_root`` defaults to ``default_audit_root()``. Roots are validated
    before ``purgeables`` opens a database. The return value is the sum of
    each store's deletion count, including a store that deletes nothing.
    """
    chosen = default_audit_root() if audit_root is None else audit_root
    validate_roots(data_root, chosen)
    deleted = 0
    for store in purgeables(data_root, clock=clock):
        deleted += store.purge_expired(now=now)
    return deleted
