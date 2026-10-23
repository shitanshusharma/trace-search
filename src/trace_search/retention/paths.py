"""Purge entry point. The audit directory is never a data root.

``audit_root`` is required and must be absolute. Callers pass that path;
there is no cwd-relative default. ``validate_roots`` runs before any store
is opened. A relative ``data_root`` or ``audit_root`` is refused before
``Path.resolve``. A data root that equals the audit root, contains it, or
resolves into or onto it through a symlink is refused. ``purge_expired``
then asks each ``purgeables`` store to delete and returns the sum of those
counts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from trace_search.retention.registry import purgeables

_Clock = Callable[[], float]


def _expanded_absolute(path: Path, label: str) -> Path:
    """Return ``path`` after ``expanduser``. Refuse a non-path or a relative path.

    This does not call ``Path.resolve``. A home-relative path can become
    absolute. Anything still relative would be bound to the cwd by resolve.
    """
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a Path, got {path!r}")
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{label} is relative: {path}")
    return expanded


def _resolved_directory(path: Path, label: str) -> Path:
    """Resolve an absolute path. Refuse a loop or a file."""
    try:
        resolved = path.resolve()
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


def validate_roots(data_root: Path, *, audit_root: Path) -> None:
    """Refuse a data root the purge could follow into the audit directory.

    ``audit_root`` is required and must be absolute. A relative ``data_root``
    or ``audit_root`` is refused before ``Path.resolve``. Comparison then uses
    ``Path.resolve``, so a symlink onto the audit directory compares equal and
    a symlink into it compares as containment.
    """
    data_abs = _expanded_absolute(data_root, "data_root")
    audit_abs = _expanded_absolute(audit_root, "audit_root")
    data = _resolved_directory(data_abs, "data_root")
    audit = _resolved_directory(audit_abs, "audit_root")
    if _strictly_contains(data, audit) or _strictly_contains(audit, data):
        raise ValueError(
            f"data root contains the audit root or resolves into it: data={data} audit={audit}"
        )
    if data == audit:
        raise ValueError(f"data root equals the audit root: {data}")


def purge_expired(
    data_root: Path,
    *,
    audit_root: Path,
    now: float | None = None,
    clock: _Clock | None = None,
) -> int:
    """Delete expired footage rows under ``data_root`` and return how many.

    ``audit_root`` is required and must be absolute. Callers pass that path;
    there is no cwd-relative default. Roots are validated before ``purgeables``
    opens a database. The return value is the sum of each store's deletion
    count, including a store that deletes nothing.
    """
    validate_roots(data_root, audit_root=audit_root)
    deleted = 0
    for store in purgeables(data_root, clock=clock):
        deleted += store.purge_expired(now=now)
    return deleted
