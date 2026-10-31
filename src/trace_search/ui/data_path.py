"""Repository-root checks for paths that must stay outside the checkout.

This module imports nothing from the UI package, so account storage can use
it without closing an import cycle through the server.
"""

from __future__ import annotations

import os
from pathlib import Path


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
