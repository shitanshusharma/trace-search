# Audit Log Module

## Overview

The audit log module provides an **append-only, hash-chained audit log** with tamper detection for trace-search. It records all operator actions with timestamps, operator identity, and action details.

## Design

### Hash Chain Integrity

Each audit log entry contains:
- **timestamp**: IST-aware timestamp (Asia/Kolkata timezone, UTC+05:30)
- **operator_username**: Username of the operator (no other identity fields)
- **action**: Type of action (login, search, view, export, retention_change, purge, etc.)
- **query**: Optional query details (for search actions)
- **case_id**: Optional case identifier
- **purpose**: Optional purpose/reason for the action
- **result_ids**: List of result identifiers returned by the action
- **prev_hash**: SHA-256 hash of the previous entry (empty for first entry)
- **entry_hash**: SHA-256 hash of this entry (excluding this field itself)

The hash chain ensures:
1. **Append-only**: New entries link to the previous entry's hash
2. **Tamper-proof**: Any modification to an entry breaks the hash chain
3. **Order-preserving**: Entries cannot be reordered without detection
4. **Deletion-detection**: Removing an entry breaks the chain at the next entry

### Storage Format

- **Format**: JSON Lines (one JSON object per line)
- **Location**: `audit_logs/audit.jsonl` (default)
- **Encoding**: UTF-8

### Verification

The `verify()` method checks:
1. Each entry's hash matches its computed hash
2. Each entry's `prev_hash` matches the previous entry's `entry_hash`
3. The first entry has an empty `prev_hash`
4. No entries are missing from the chain

When tampering is detected, the verifier returns:
- `valid=False`
- `broken_index`: The index of the first broken entry
- `error_message`: Description of the integrity violation

## Retention and Separation

### 1-Year Retention

Audit log entries are kept for **1 year** (366 days) to comply with DPDP Rule 6 access log requirements.

### Exemption from Derived-Data Purge

The audit log is stored in a **separate location** (`audit_logs/`) from derived data (tracks, embeddings, clips) to ensure it survives the configured retention purge.

**Key separation points:**

1. **Storage location**: `audit_logs/` directory, separate from derived data stores
2. **Retention policy**: Audit log has its own 1-year retention, independent of footage-derived data TTL (default 7 days)
3. **Purge logic**: The retention purge module (T9) must **explicitly exclude** the `audit_logs/` directory

## Usage

```python
from trace_search.audit import AuditLog

# Initialize (default location: audit_logs/audit.jsonl)
audit_log = AuditLog()

# Or specify custom path
audit_log = AuditLog(Path("/data/audit_logs/system.jsonl"))

# Append entries
audit_log.append(
    operator_username="operator_alice",
    action="search",
    query="person with blue backpack",
    case_id="CASE-2024-001",
    purpose="Patient search incident #42",
    result_ids=["track-001", "track-002", "track-003"],
)

# Verify integrity
result = audit_log.verify()
if not result.valid:
    print(f"Tampering detected at index {result.broken_index}")
    print(f"Error: {result.error_message}")

# Query entries
alice_actions = audit_log.query_by_operator("operator_alice")
search_actions = audit_log.query_by_action("search")
case_entries = audit_log.query_by_case_id("CASE-2024-001")

# Count entries
total = audit_log.count()
```

## AC-13 Compliance

This module satisfies AC-13 acceptance criteria:

✅ **50 scripted actions produce 50/50 entries**: The test `test_fifty_actions_produce_fifty_entries` demonstrates 50 actions creating 50 valid entries with all fields populated.

✅ **Tampering detection**: Tests demonstrate that:
- Editing one entry fails verification at the tampered index
- Deleting one entry fails verification at the chain break
- Swapping entries fails verification at the swap point

✅ **All required fields**: Every entry has:
- `timestamp` (IST-aware)
- `operator_username` (who)
- `action` (what)
- `case_id` (why/context)
- `purpose` (reason)
- `result_ids` (what was returned)

## Privacy Compliance

**No identity fields beyond username**: The audit log records only the operator's username, not any PII or biometric data.

**No face data**: The audit log contains no face embeddings, face crops, or facial recognition data.

## Testing

Run audit log tests:
```bash
pytest tests/test_audit_log.py -v
```

Key test cases:
- `test_fifty_actions_produce_fifty_entries`: AC-13 requirement for 50/50 entries
- `test_edit_entry_breaks_verification`: AC-13 tampering detection (edit)
- `test_delete_entry_breaks_verification`: AC-13 tampering detection (delete)
- `test_swap_entries_breaks_verification`: AC-13 tampering detection (reorder)
- `test_audit_log_separate_location`: Storage separation for retention exemption

## Integration with Retention (T9)

When implementing the retention purge module (T9):

1. **Exclude audit logs from derived-data purge**:
   ```python
   # Purge footage-derived data (tracks, embeddings, clips)
   purge_older_than(ttl_days=7, exclude_dirs=["audit_logs"])
   ```

2. **Separate 1-year audit log purge**:
   ```python
   # Only purge audit entries older than 1 year (366 days)
   purge_audit_logs_older_than(days=366)
   ```

3. **Verification after purge**:
   ```python
   # Verify audit log chain is intact after any purge
   result = audit_log.verify()
   assert result.valid, "Audit log compromised after purge"
   ```

## File Manifest

- `src/trace_search/audit/__init__.py`: Module exports
- `src/trace_search/audit/log.py`: AuditLog, AuditEntry, VerificationResult implementations
- `tests/test_audit_log.py`: Comprehensive test suite (14 tests, including AC-13)
- `src/trace_search/audit/README.md`: This file

## License

MIT License. All code uses Python standard library only (no external dependencies).
