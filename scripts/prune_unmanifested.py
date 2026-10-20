"""Drop installed distributions the licence manifest does not list.

The shared lock pins the CI tools as well as the manifest closure, including
protobuf 5.29.6 and mcp 1.30.0. The licence fence scans every installed
distribution and fails when a name is missing from licenses/manifest.json.
The licence job installs the lock, runs this script, then lints and tests.
Other jobs keep the full lock installed.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# pip-licenses does not report pip, setuptools, or wheel. trace-search is the
# project under test. Everything else must be a library row in the manifest
# or a runtime dependency of one of those rows.
SYSTEM_KEEP = frozenset({"pip", "setuptools", "wheel", "trace-search"})
MANIFEST = Path("licenses/manifest.json")


def manifest_names(path: Path = MANIFEST) -> set[str]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for entry in loaded.get("entries", []):
        if isinstance(entry, dict) and entry.get("type") == "library":
            names.add(canonicalize_name(str(entry.get("name", ""))))
    return names


def runtime_keep(manifest: set[str]) -> set[str]:
    """Manifest libraries, system packages, and their non-extra dependencies."""
    keep = set(manifest) | set(SYSTEM_KEEP)
    dists = {canonicalize_name(dist.metadata["Name"]): dist for dist in metadata.distributions()}
    changed = True
    while changed:
        changed = False
        for name in list(keep):
            dist = dists.get(name)
            if dist is None:
                continue
            for raw in dist.requires or []:
                req = Requirement(raw)
                if req.marker is not None and not req.marker.evaluate():
                    continue
                dep = canonicalize_name(req.name)
                if dep not in keep:
                    keep.add(dep)
                    changed = True
    return keep


def unmanifested(keep: set[str]) -> list[str]:
    names = {
        dist.metadata["Name"]
        for dist in metadata.distributions()
        if canonicalize_name(dist.metadata["Name"]) not in keep
    }
    return sorted(names, key=str.lower)


def main() -> int:
    remove = unmanifested(runtime_keep(manifest_names()))
    if not remove:
        print("No CI-only distributions to remove.")
        return 0
    print(f"Removing {len(remove)} CI-only distributions before the licence fence.")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", *remove])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
