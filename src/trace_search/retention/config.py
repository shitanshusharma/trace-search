"""Shipped retention windows.

``FOOTAGE_DERIVED_DAYS`` is 7. ``AUDIT_DAYS`` is 365. ``load`` returns those
two values on a frozen ``RetentionConfig``. It reads no environment variable
and no file. An override file outside the repository is T-P-7.
"""

from __future__ import annotations

FOOTAGE_DERIVED_DAYS = 7
AUDIT_DAYS = 365

__all__ = ["AUDIT_DAYS", "FOOTAGE_DERIVED_DAYS", "RetentionConfig", "load"]


def _require_day_count(label: str, value: object) -> int:
    """Reject a bool, a non-int, or a negative count. Zero is a valid count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int >= 0, got {value!r}")
    if value < 0:
        raise ValueError(f"{label} must be an int >= 0, got {value!r}")
    return value


class RetentionConfig:
    """Read-only footage and audit windows, in whole days.

    ``footage_days`` and ``audit_days`` are properties. The constructor
    stores the counts and then refuses later writes.
    """

    __slots__ = ("_footage_days", "_audit_days")

    _footage_days: int
    _audit_days: int

    def __init__(self, footage_days: int, audit_days: int) -> None:
        footage = _require_day_count("footage_days", footage_days)
        audit = _require_day_count("audit_days", audit_days)
        object.__setattr__(self, "_footage_days", footage)
        object.__setattr__(self, "_audit_days", audit)

    @property
    def footage_days(self) -> int:
        """Footage-derived retention in whole days."""
        return self._footage_days

    @property
    def audit_days(self) -> int:
        """Audit-log retention in whole days."""
        return self._audit_days

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"RetentionConfig is frozen; cannot set {name}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"RetentionConfig is frozen; cannot delete {name}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RetentionConfig):
            return False
        return self.footage_days == other.footage_days and self.audit_days == other.audit_days

    def __hash__(self) -> int:
        return hash((self.footage_days, self.audit_days))


def load() -> RetentionConfig:
    """Return the shipped retention. No environment variable and no file."""
    return RetentionConfig(footage_days=FOOTAGE_DERIVED_DAYS, audit_days=AUDIT_DAYS)
