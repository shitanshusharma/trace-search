#!/usr/bin/env python3
"""Fail the CI pip-audit step once its time-boxed ignore has expired.

The workflow applies GHSA-xcq7-p3pf-p88h only when auditing the installed
requirements-ci.lock environment. This script does not pass the ignore
through. It exits 1 after the expiry date so that ignore cannot outlive
the approval window.
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


def main(today: date | None = None) -> int:
    current = datetime.now(timezone.utc).date() if today is None else today
    if ignore_is_current(current):
        print(
            f"{ADVISORY} ignore is current through {EXPIRY.isoformat()} "
            f"(today {current.isoformat()})"
        )
        return 0
    print(
        f"{ADVISORY} ignore expired on {EXPIRY.isoformat()} (today {current.isoformat()})",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
