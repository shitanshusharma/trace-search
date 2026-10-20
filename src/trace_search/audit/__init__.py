"""
Audit log module for trace-search.

Provides append-only, HMAC-chained audit logging with tamper detection.
All audit entries are kept for 1 year and survive derived-data retention purges.
"""

from .log import (
    AUDIT_LOG_DIR,
    AuditAction,
    AuditEntry,
    AuditLog,
    VerificationResult,
    is_audit_path,
)

__all__ = [
    "AUDIT_LOG_DIR",
    "AuditAction",
    "AuditEntry",
    "AuditLog",
    "VerificationResult",
    "is_audit_path",
]
