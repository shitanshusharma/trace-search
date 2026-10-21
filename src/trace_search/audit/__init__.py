"""
Audit log module for trace-search.

Append-only, HMAC-chained audit logging with a signed head anchor.
Audit entries are kept for 1 year and survive derived-data retention purges.
"""

from .log import (
    AUDIT_LOG_DIR,
    TRACE_AUDIT_HMAC_KEY_ENV,
    AuditAction,
    AuditEntry,
    AuditLog,
    VerificationResult,
    default_log_path,
    head_anchor_path,
    init_hmac_key,
    is_audit_path,
)

__all__ = [
    "AUDIT_LOG_DIR",
    "TRACE_AUDIT_HMAC_KEY_ENV",
    "AuditAction",
    "AuditEntry",
    "AuditLog",
    "VerificationResult",
    "default_log_path",
    "head_anchor_path",
    "init_hmac_key",
    "is_audit_path",
]
