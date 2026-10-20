"""
Audit log module for trace-search.

Provides append-only, hash-chained audit logging with tamper detection.
All audit entries are kept for 1 year and survive derived-data retention purges.
"""

from .log import AuditEntry, AuditLog, VerificationResult

__all__ = ["AuditLog", "AuditEntry", "VerificationResult"]
