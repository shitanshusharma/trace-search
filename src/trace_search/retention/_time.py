"""Convert a retention window from whole days to seconds.

``retention_seconds`` is the one conversion both footage stores use. The
copies that used to live in ``tracking.store`` and ``attributes.store``
are gone. A row whose age is exactly ``max_age_days`` stays, so the result
is ``max_age_days * 86400`` and not one second less.
"""

from __future__ import annotations

# Private _require_day_count is imported here on purpose so both stores share it.
from trace_search.retention.config import _require_day_count

_SECONDS_PER_DAY = 86400


def retention_seconds(max_age_days: int) -> int:
    """Return ``max_age_days * 86400``.

    A bool is a ``TypeError``. A negative count is a ``ValueError``. Zero
    is a valid window.
    """
    days = _require_day_count("max_age_days", max_age_days)
    return days * _SECONDS_PER_DAY


def retention_cutoff(now: float, max_age_days: int) -> float:
    """UTC epoch. A row older than this is expired. A row at this epoch stays."""
    return now - float(retention_seconds(max_age_days))
