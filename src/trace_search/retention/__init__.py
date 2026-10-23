"""Shipped retention windows for footage-derived data and the audit log.

``retention_seconds`` is the shared conversion from whole days to seconds.
"""

from ._time import retention_seconds
from .config import AUDIT_DAYS, FOOTAGE_DERIVED_DAYS, RetentionConfig, load

__all__ = [
    "AUDIT_DAYS",
    "FOOTAGE_DERIVED_DAYS",
    "RetentionConfig",
    "load",
    "retention_seconds",
]
