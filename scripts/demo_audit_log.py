#!/usr/bin/env python3
"""AC-13 proof for the HMAC audit log.

Run after `pip install -e .`. Prints one JSON document on stdout.

    python scripts/demo_audit_log.py --json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from trace_search.audit import AuditAction, AuditLog, init_hmac_key

_ACTIONS = (
    AuditAction.LOGIN,
    AuditAction.SEARCH,
    AuditAction.VIEW,
    AuditAction.EXPORT,
    AuditAction.RETENTION_CHANGE,
    AuditAction.PURGE,
)


def _head_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _open_log(directory: Path) -> AuditLog:
    key_path = directory / "keys" / "audit.key"
    init_hmac_key(key_path)
    log = AuditLog(directory / "audit.jsonl", key_path=key_path)
    log.initialize()
    return log


def _result_json(result) -> dict:
    return {
        "valid": result.valid,
        "total_entries": result.total_entries,
        "broken_index": result.broken_index,
        "error": result.error_message,
    }


def _fifty(directory: Path) -> dict:
    log = _open_log(directory)
    for index in range(50):
        action = _ACTIONS[index % len(_ACTIONS)]
        details = None
        if action in (AuditAction.RETENTION_CHANGE, AuditAction.PURGE):
            details = f"details-{index}"
        log.append(
            operator_username=f"operator_{(index % 3) + 1}",
            action=action,
            query=f"query-{index}",
            case_id=f"CASE-{index:03d}",
            purpose=f"purpose-{index}",
            details=details,
            result_ids=[f"r{index}"],
        )
    entries = log.read_all()
    verified = log.verify()
    sample_at = (0, 10, 25, 40, 49)
    samples = []
    for index in sample_at:
        entry = entries[index]
        samples.append(
            {
                "index": index,
                "timestamp": entry.timestamp,
                "operator_username": entry.operator_username,
                "action": entry.action,
                "query": entry.query,
                "case_id": entry.case_id,
                "purpose": entry.purpose,
                "details": entry.details,
                "result_ids": entry.result_ids,
                "prev_hmac": entry.prev_hmac,
                "entry_hmac": entry.entry_hmac,
            }
        )
    return {
        "count": log.count(),
        "actions": sorted({entry.action for entry in entries}),
        "verification": _result_json(verified),
        "samples": samples,
    }


def _tamper(directory: Path, kind: str) -> dict:
    log = _open_log(directory)
    for index in range(10):
        log.append(
            f"user_{index}",
            AuditAction.SEARCH,
            case_id=f"CASE-{index:03d}",
            purpose="AC-13 tamper proof",
            result_ids=[f"r{index}"],
        )
    lines = log.log_path.read_text(encoding="utf-8").splitlines()
    if kind == "edit":
        row = json.loads(lines[5])
        row["operator_username"] = "TAMPERED"
        lines[5] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    elif kind == "delete":
        del lines[3]
    elif kind == "swap":
        lines[4], lines[5] = lines[5], lines[4]
    else:
        raise ValueError(kind)
    log.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _result_json(log.verify())


def build_proof() -> dict:
    """Run the AC-13 script in a temporary directory and return the proof."""
    with tempfile.TemporaryDirectory(prefix="trace-audit-") as tmp:
        root = Path(tmp)
        fifty = _fifty(root / "fifty")
        tamper = {
            "edit": _tamper(root / "edit", "edit"),
            "delete": _tamper(root / "delete", "delete"),
            "swap": _tamper(root / "swap", "swap"),
        }
    fifty_ok = (
        fifty["count"] == 50
        and fifty["verification"]["valid"] is True
        and fifty["verification"]["total_entries"] == 50
    )
    tamper_ok = (
        tamper["edit"]["valid"] is False
        and tamper["edit"]["broken_index"] == 5
        and tamper["delete"]["valid"] is False
        and tamper["delete"]["broken_index"] == 3
        and tamper["swap"]["valid"] is False
        and tamper["swap"]["broken_index"] == 4
    )
    return {
        "ac": "AC-13",
        "head": _head_sha(),
        "chain": "HMAC-SHA256",
        "canonical_json": 'json.dumps(..., sort_keys=True, separators=(",", ":"))',
        "anchor": "<log_path>.head",
        "key_env": "TRACE_AUDIT_HMAC_KEY",
        "fifty_actions": fifty,
        "tamper": tamper,
        "passed": fifty_ok and tamper_ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the AC-13 audit-log proof as JSON.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the proof as JSON (this is also the default).",
    )
    parser.parse_args(argv)
    proof = build_proof()
    json.dump(proof, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if proof["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
