# Audit log (T8)

Append-only audit log for Cut 01 AC-13. Each entry is linked with HMAC-SHA256. A signed head anchor records the tip of the chain. The log is kept for 1 year and is stored apart from footage-derived data so a 7-day purge does not delete it.

Python standard library only. No Ultralytics, no AGPL dependencies, no face data, no identity field other than `operator_username`.

## Chain

Every line is one JSON object. The HMAC input is canonical JSON of every field except `entry_hmac`:

```text
json.dumps(payload, sort_keys=True, separators=(",", ":"))
```

`prev_hmac` is the previous line's `entry_hmac`. The first entry uses an empty `prev_hmac`.

Fields on every entry:

| Field | Meaning |
|---|---|
| `timestamp` | ISO-8601 in Asia/Kolkata (`+05:30`). A naive clock is rejected. Aware clocks are converted to IST. |
| `operator_username` | Operator username. Required, non-empty. This is the only identity. |
| `action` | `AuditAction`: `login`, `search`, `view`, `export`, `retention_change`, `purge`. Unknown actions are rejected. |
| `query` | Optional query text. |
| `case_id` | Required and non-empty for `search`, `view`, and `export`. |
| `purpose` | Required and non-empty for `search`, `view`, and `export`. |
| `details` | Optional text for `retention_change` and `purge` (not the search query). |
| `result_ids` | List of result ids. |
| `prev_hmac` | Previous entry HMAC, or `""` for the genesis entry. |
| `entry_hmac` | HMAC-SHA256 of this entry. |

`verify()` never raises. A malformed JSON line, a non-object, a missing key, a wrong type, an extra key, or an unknown action returns `valid=False` and sets `broken_index` to that entry.

## Head anchor

The anchor path is derived from the log path: `<log_path>.head` (for example `audit_logs/audit.jsonl.head`). Two logs do not share an anchor, and the path does not depend on the working directory.

The anchor is JSON with `last_hmac`, `entry_count`, and `anchor_hmac`. `anchor_hmac` is HMAC-SHA256 over the canonical JSON of `last_hmac` and `entry_count`, using the same key as the chain. It is written under the append lock: temp file in the same directory, `fsync`, then `os.replace`.

`verify()` fails when the anchor is missing, unsigned, signed with the wrong key, or does not match the last entry and the count. That includes:

- an empty log with no anchor
- an empty or fully truncated log whose anchor count is greater than 0
- a tail delete, including one that also deletes the anchor or rewrites the anchor without the key
- a tail edit that recomputes the entry HMAC but leaves the old anchor
- a full rehash after an edit, including a rehash with the wrong key

## Key

There is no default key in the code. `verify()` and `append()` never generate a key. A missing key makes `verify()` return `valid=False` with a clear error. `append()` raises.

The key is 32 bytes, hex-encoded. It is loaded from exactly one of:

1. `key_path` passed to `AuditLog` (this wins when both are set)
2. the environment variable `TRACE_AUDIT_HMAC_KEY`

The key file must sit outside `audit_logs/`. `is_audit_path()` is what that check uses. The only way to create a key is `init_hmac_key(path)`:

```python
from pathlib import Path
from trace_search.audit import AuditLog, init_hmac_key

init_hmac_key(Path("/etc/trace-search/audit.key"))
log = AuditLog(key_path=Path("/etc/trace-search/audit.key"))
```

Or, with no key file:

```bash
export TRACE_AUDIT_HMAC_KEY="$(python3 -c 'import os; print(os.urandom(32).hex())')"
```

## Concurrency

`append()` takes one exclusive `fcntl` lock on `<log_path>.auditlock` and holds it while it reads the last entry, writes the new line, `fsync`s it, and writes the anchor. A malformed last line raises `RuntimeError` instead of starting a new chain.

## Where the log lives

`AUDIT_LOG_DIR` (`"audit_logs"`) and `is_audit_path()` are exported from `trace_search.audit`. The default log path is `Path(AUDIT_LOG_DIR) / "audit.jsonl"`. Nothing else builds that path from a string literal.

`is_audit_path()` is true when a path component is `audit_logs`, including absolute paths such as `/data/audit_logs/system.jsonl` from any working directory. T9 should import this helper and skip those paths during the footage-derived purge. A sibling name such as `audit_logs_backup` does not match.

Constructing `AuditLog` does not create the log, the anchor, or a key.

## What this does not detect

An attacker who holds the HMAC key and deletes both the log and the anchor is not detectable on this host. There is no external witness of the previous tip, and the same key can sign a fresh chain that `verify()` accepts. Publishing the anchor outside the host is out of scope for Cut 01.

The 1-year purge checkpoint is also out of scope. The genesis entry must have an empty `prev_hmac`, so dropping a prefix of the chain fails `verify()`. T9 owns the signed checkpoint that a prefix purge needs.

## Usage

```python
from trace_search.audit import AuditAction, AuditLog, init_hmac_key

init_hmac_key("/etc/trace-search/audit.key")
audit_log = AuditLog(key_path="/etc/trace-search/audit.key")

audit_log.append(
    operator_username="operator_alice",
    action=AuditAction.SEARCH,
    query="blue backpack",
    case_id="CASE-2024-001",
    purpose="Incident review",
    result_ids=["track-001", "track-002"],
)

result = audit_log.verify()
if not result.valid:
    print(result.broken_index, result.error_message)
```

## Tests and AC-13 proof

`tests/test_audit_log.py` has 46 tests. They use `tmp_path` only. Covered cases include 50 scripted actions (login, search, view, export, retention change, purge) with every field filled, edit/delete/swap with the broken index, tail delete, tail edit plus rehash, full rehash, forged genesis, rehash from entry 0, tail delete plus a deleted anchor, tail delete plus an anchor rewritten without the key, full rehash with the wrong key, full truncation, two logs that do not share an anchor, `verify()` with no key, and 8 processes × 50 appends (400 entries, no forked `prev_hmac`).

```bash
pytest tests/test_audit_log.py -v
python3 scripts/demo_audit_log.py --json
```

A real `--json` run reports `passed: true`, `fifty_actions.count: 50`, `verification.total_entries: 50`, and tamper indexes edit 5, delete 3, swap 4. CI runs that command and fails if `git status --porcelain` is dirty after pytest.

## Files

- `src/trace_search/audit/__init__.py` exports `AUDIT_LOG_DIR`, `is_audit_path`, `init_hmac_key`, and `AuditLog`
- `src/trace_search/audit/log.py` is the implementation
- `tests/test_audit_log.py` (45 tests)
- `scripts/demo_audit_log.py` prints the AC-13 JSON proof
