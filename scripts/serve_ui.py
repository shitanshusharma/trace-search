#!/usr/bin/env python3
"""Serve the local trace-search UI on 127.0.0.1.

``--data DIR`` must sit outside this repository. The audit log under that
directory must already exist and pass ``verify()``. This command does not
create the log. Initialize it with ``AuditLog.initialize()`` first.
"""

from __future__ import annotations

from trace_search.ui.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
