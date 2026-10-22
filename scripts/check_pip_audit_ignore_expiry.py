#!/usr/bin/env python3
"""Fail the CI pip-audit step once its time-boxed ignore has expired.

The workflow applies GHSA-xcq7-p3pf-p88h only when auditing the installed
requirements-ci.lock environment. This script does not pass the ignore
through. It exits 1 after the expiry date so that ignore cannot outlive
the approval window.

With no arguments the clock is today's UTC date. One YYYY-MM-DD argument
overrides that clock. The workflow and check_all.sh pass no argument.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone

ADVISORY = "GHSA-xcq7-p3pf-p88h"
# Inclusive. 2026-11-21 still passes. 2026-11-22 fails the CI step.
EXPIRY = date(2026, 11, 21)


def ignore_is_current(today: date, expiry: date = EXPIRY) -> bool:
    """Return True when today is on or before the expiry date."""
    return today <= expiry


def argv_date(argv: list[str]) -> date | None:
    """Return the optional YYYY-MM-DD override, or None to use UTC today."""
    if not argv:
        return None
    if len(argv) != 1:
        print("usage: check_pip_audit_ignore_expiry.py [YYYY-MM-DD]", file=sys.stderr)
        raise SystemExit(2)
    try:
        return date.fromisoformat(argv[0])
    except ValueError:
        print("usage: check_pip_audit_ignore_expiry.py [YYYY-MM-DD]", file=sys.stderr)
        raise SystemExit(2) from None


def main(today: date | None = None) -> int:
    current = datetime.now(timezone.utc).date() if today is None else today
    if ignore_is_current(current):
        print(
            f"{ADVISORY} ignore is current through {EXPIRY.isoformat()} "
            f"(today {current.isoformat()})"
        )
        return 0
    print(
        f"{ADVISORY} ignore expired after {EXPIRY.isoformat()} (today {current.isoformat()})",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(argv_date(sys.argv[1:])))
