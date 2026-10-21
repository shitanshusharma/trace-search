"""Tests for the HMAC-chained audit log (AC-13 and the T8 review)."""

import json
import multiprocessing
import os
import subprocess
import sys
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trace_search.audit import (
    AUDIT_LOG_DIR,
    TRACE_AUDIT_HMAC_KEY_ENV,
    AuditAction,
    AuditEntry,
    AuditLog,
    default_log_path,
    init_hmac_key,
    is_audit_path,
)

_ACTIONS = (
    AuditAction.LOGIN,
    AuditAction.SEARCH,
    AuditAction.VIEW,
    AuditAction.EXPORT,
    AuditAction.RETENTION_CHANGE,
    AuditAction.PURGE,
)


def _create_audit_log(
    tmp_path: Path,
    filename: str = "audit.jsonl",
    *,
    initialized: bool = True,
) -> AuditLog:
    """Build a log whose key lives outside AUDIT_LOG_DIR, under tmp_path."""
    log_path = tmp_path / filename
    key_path = tmp_path / "keys" / f"{filename}.key"
    init_hmac_key(key_path)
    log = AuditLog(log_path, key_path=key_path)
    if initialized:
        log.initialize()
    return log


def _append_login(log: AuditLog, username: str = "user1") -> AuditEntry:
    return log.append(username, AuditAction.LOGIN)


def _read_lines(log: AuditLog) -> list[dict]:
    text = log.log_path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_lines(log: AuditLog, rows: list[dict]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    log.log_path.write_text(payload, encoding="utf-8")


def _rehash(rows: list[dict], key: bytes) -> list[dict]:
    prev = ""
    rebuilt: list[dict] = []
    for row in rows:
        row = dict(row)
        row["prev_hmac"] = prev
        row["entry_hmac"] = ""
        entry = AuditEntry(**row)
        entry.entry_hmac = entry.compute_hmac(key)
        row["entry_hmac"] = entry.entry_hmac
        prev = entry.entry_hmac
        rebuilt.append(row)
    return rebuilt


def _write_anchor(path: Path, key: bytes | None, last_hmac: str, count: int) -> None:
    import hashlib
    import hmac

    body = {"entry_count": count, "last_hmac": last_hmac}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    if key is None:
        signature = "00" * 32
    else:
        signature = hmac.new(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    signed = {
        "anchor_hmac": signature,
        "entry_count": count,
        "last_hmac": last_hmac,
    }
    path.write_text(
        json.dumps(signed, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _concurrent_worker(log_path: str, key_path: str, process_id: int) -> None:
    log = AuditLog(log_path, key_path=key_path)
    for index in range(50):
        log.append(f"proc{process_id}_user{index}", AuditAction.LOGIN)


class TestAuditPathHelper:
    """is_audit_path is the purge exemption check T9 imports."""

    def test_audit_directory_component(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = Path(AUDIT_LOG_DIR) / "nested" / "audit.jsonl"
        assert is_audit_path(target)
        assert is_audit_path(AUDIT_LOG_DIR)
        decoy = tmp_path / "clips" / AUDIT_LOG_DIR / "crop.json"
        assert not is_audit_path(decoy)

    def test_absolute_audit_dir_from_another_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert not is_audit_path("/data/audit_logs/system.jsonl")
        assert not is_audit_path(Path("/var/lib/audit_logs"))
        inside = tmp_path / AUDIT_LOG_DIR / "system.jsonl"
        assert is_audit_path(inside)

    def test_relative_names(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert is_audit_path(AUDIT_LOG_DIR)
        assert is_audit_path(f"{AUDIT_LOG_DIR}/audit.jsonl")
        assert not (tmp_path / AUDIT_LOG_DIR).exists()

    def test_outside_and_sibling_names(self, tmp_path):
        assert not is_audit_path(tmp_path / "data" / "file.txt")
        sibling = tmp_path / "audit_logs_backup" / "file.txt"
        assert not is_audit_path(sibling)

    def test_default_path_uses_constant_and_creates_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert default_log_path() == Path(AUDIT_LOG_DIR) / "audit.jsonl"
        key_path = tmp_path / "operator.key"
        init_hmac_key(key_path)
        log = AuditLog(key_path=key_path)
        assert log.log_path == default_log_path()
        assert log.head_path == Path(f"{log.log_path}.head")
        assert not Path(AUDIT_LOG_DIR).exists()


class TestKeyHandling:
    """The key is explicit. verify() never generates one."""

    def test_verify_without_a_key_fails(self, tmp_path, monkeypatch):
        monkeypatch.delenv(TRACE_AUDIT_HMAC_KEY_ENV, raising=False)
        log = AuditLog(tmp_path / "audit.jsonl")
        result = log.verify()
        assert result.valid is False
        assert result.error_message is not None
        assert "key" in result.error_message.lower()
        assert not (tmp_path / "audit.jsonl").exists()
        assert not list(tmp_path.glob("*.head"))

    def test_append_without_a_key_raises_and_creates_nothing(self, tmp_path, monkeypatch):
        monkeypatch.delenv(TRACE_AUDIT_HMAC_KEY_ENV, raising=False)
        log = AuditLog(tmp_path / "audit.jsonl")
        with pytest.raises(RuntimeError, match="No HMAC key"):
            log.append("user1", AuditAction.LOGIN)
        assert not (tmp_path / "audit.jsonl").exists()

    def test_env_key_is_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setenv(TRACE_AUDIT_HMAC_KEY_ENV, os.urandom(32).hex())
        log = AuditLog(tmp_path / "audit.jsonl")
        log.initialize()
        log.append("user1", AuditAction.LOGIN)
        result = log.verify()
        assert result.valid is True
        assert result.total_entries == 1

    def test_explicit_key_path_wins_over_different_env(self, tmp_path, monkeypatch):
        log = _create_audit_log(tmp_path)
        _append_login(log)
        monkeypatch.setenv(TRACE_AUDIT_HMAC_KEY_ENV, os.urandom(32).hex())
        assert log.verify().valid is True

    def test_key_inside_audit_dir_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        key_path = tmp_path / AUDIT_LOG_DIR / ".audit_key"
        key_path.parent.mkdir(parents=True)
        key_path.write_text(os.urandom(32).hex() + "\n", encoding="utf-8")
        log = AuditLog(tmp_path / AUDIT_LOG_DIR / "audit.jsonl", key_path=key_path)
        with pytest.raises(RuntimeError, match="outside"):
            log.append("user1", AuditAction.LOGIN)
        result = log.verify()
        assert result.valid is False
        assert "outside" in (result.error_message or "")

    def test_init_is_the_only_creator_and_refuses_audit_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        created = init_hmac_key(tmp_path / "keys" / "audit.key")
        assert created.is_file()
        assert created.stat().st_mode & 0o777 == 0o600
        with pytest.raises(FileExistsError):
            init_hmac_key(created)
        with pytest.raises(RuntimeError, match="inside"):
            init_hmac_key(tmp_path / AUDIT_LOG_DIR / "secret.key")


class TestAuditLogBasic:
    def test_empty_log_with_no_anchor_fails(self, tmp_path):
        log = _create_audit_log(tmp_path, initialized=False)
        result = log.verify()
        assert result.valid is False
        assert result.total_entries == 0
        assert result.error_message is not None
        assert "Missing head anchor" in result.error_message

    def test_append_without_initialize_raises(self, tmp_path):
        log = _create_audit_log(tmp_path, initialized=False)
        with pytest.raises(RuntimeError, match="not initialized"):
            _append_login(log)
        assert not log.log_path.exists()
        result = log.verify()
        assert result.valid is False
        assert "Missing head anchor" in (result.error_message or "")

    def test_single_entry_appends_every_field(self, tmp_path):
        log = _create_audit_log(tmp_path)
        entry = log.append(
            operator_username="test_user",
            action=AuditAction.LOGIN,
            query="shift start",
            case_id="CASE-001",
            purpose="Test login",
            result_ids=["result-1"],
        )
        assert entry.operator_username == "test_user"
        assert entry.action == "login"
        assert entry.case_id == "CASE-001"
        assert entry.purpose == "Test login"
        assert entry.query == "shift start"
        assert entry.result_ids == ["result-1"]
        assert entry.prev_hmac == ""
        assert len(entry.entry_hmac) == 64
        assert log.count() == 1
        assert log.head_path == Path(f"{log.log_path}.head")
        assert log.head_path.is_file()
        result = log.verify()
        assert result.valid is True
        assert result.total_entries == 1

    def test_entries_form_a_chain(self, tmp_path):
        log = _create_audit_log(tmp_path)
        first = _append_login(log, "user1")
        second = log.append(
            "user1",
            AuditAction.SEARCH,
            query="red shirt",
            case_id="C1",
            purpose="Test",
        )
        third = log.append(
            "user1",
            AuditAction.VIEW,
            result_ids=["r1", "r2"],
            case_id="C1",
            purpose="Test",
        )
        assert second.prev_hmac == first.entry_hmac
        assert third.prev_hmac == second.entry_hmac
        result = log.verify()
        assert result.valid is True
        assert result.total_entries == 3

    def test_timestamps_are_ist(self, tmp_path):
        log = _create_audit_log(tmp_path)
        entry = _append_login(log)
        assert entry.timestamp.endswith("+05:30")

    def test_clock_converts_to_ist_and_rejects_naive(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        key_path = tmp_path / "keys" / "clock.key"
        init_hmac_key(key_path)
        utc_noon = datetime(2026, 10, 20, 6, 30, tzinfo=timezone.utc)
        log = AuditLog(log_path, key_path=key_path, clock=lambda: utc_noon)
        log.initialize()
        entry = _append_login(log)
        assert entry.timestamp.startswith("2026-10-20T12:00:00+05:30")

        naive = AuditLog(
            tmp_path / "naive.jsonl",
            key_path=key_path,
            clock=lambda: datetime(2026, 10, 20, 12, 0, 0),
        )
        naive.initialize()
        with pytest.raises(ValueError, match="timezone-aware"):
            naive.append("user1", AuditAction.LOGIN)

    def test_malformed_last_line_refuses_append(self, tmp_path):
        log = _create_audit_log(tmp_path)
        _append_login(log)
        with log.log_path.open("a", encoding="utf-8") as handle:
            handle.write("{not-json\n")
        with pytest.raises(RuntimeError, match="malformed"):
            _append_login(log, "user2")
        assert log.count() == 2
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 1


class TestActionValidation:
    def test_unknown_action_rejected(self, tmp_path):
        log = _create_audit_log(tmp_path)
        with pytest.raises(ValueError, match="Invalid action 'serach'"):
            log.append("user1", "serach")
        with pytest.raises(ValueError, match="Invalid action 'SEARCH'"):
            log.append("user1", "SEARCH")

    def test_action_string_and_enum(self, tmp_path):
        log = _create_audit_log(tmp_path)
        enum_entry = log.append("user1", AuditAction.LOGIN)
        assert enum_entry.action == "login"
        string_entry = log.append("user1", "login")
        assert string_entry.action == "login"

    def test_search_view_export_require_case_id_and_purpose(self, tmp_path):
        log = _create_audit_log(tmp_path)
        for action in (AuditAction.SEARCH, AuditAction.VIEW, AuditAction.EXPORT):
            with pytest.raises(ValueError, match="case_id is required"):
                log.append("user1", action, purpose="Test")
            with pytest.raises(ValueError, match="purpose is required"):
                log.append("user1", action, case_id="C1")
            with pytest.raises(ValueError, match="case_id is required"):
                log.append("user1", action, case_id="   ", purpose="Test")
            with pytest.raises(ValueError, match="purpose is required"):
                log.append("user1", action, case_id="C1", purpose="  ")

    def test_non_string_case_id_is_value_error(self, tmp_path):
        log = _create_audit_log(tmp_path)
        with pytest.raises(ValueError, match="case_id is required"):
            log.append("user1", AuditAction.SEARCH, case_id=5, purpose="Test")

    def test_login_allows_omitted_case_fields(self, tmp_path):
        log = _create_audit_log(tmp_path)
        entry = log.append("user1", AuditAction.LOGIN)
        assert entry.case_id is None
        assert entry.purpose is None

    def test_empty_operator_rejected(self, tmp_path):
        log = _create_audit_log(tmp_path)
        with pytest.raises(ValueError, match="operator_username"):
            log.append("   ", AuditAction.LOGIN)

    def test_retention_and_purge_store_details(self, tmp_path):
        log = _create_audit_log(tmp_path)
        change = log.append(
            "admin",
            AuditAction.RETENTION_CHANGE,
            details="footage_ttl=7d audit_ttl=365d",
        )
        purge = log.append(
            "admin",
            AuditAction.PURGE,
            details="purged footage-derived rows older than 7 days",
        )
        assert change.query is None
        assert change.details == "footage_ttl=7d audit_ttl=365d"
        assert purge.details == "purged footage-derived rows older than 7 days"
        assert log.verify().valid is True
        with pytest.raises(ValueError, match="details"):
            log.append("admin", AuditAction.PURGE, details="  ")
        with pytest.raises(ValueError, match="details"):
            log.append("admin", AuditAction.PURGE, details=7)


class TestTamperAndAnchor:
    def _five(self, tmp_path) -> AuditLog:
        log = _create_audit_log(tmp_path)
        for _ in range(5):
            _append_login(log)
        assert log.verify().valid is True
        return log

    def test_edit_names_the_index(self, tmp_path):
        log = self._five(tmp_path)
        rows = _read_lines(log)
        rows[2]["operator_username"] = "TAMPERED"
        _write_lines(log, rows)
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 2
        assert "HMAC mismatch" in (result.error_message or "")

    def test_delete_names_the_index(self, tmp_path):
        log = self._five(tmp_path)
        rows = _read_lines(log)
        del rows[3]
        _write_lines(log, rows)
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 3

    def test_swap_names_the_index(self, tmp_path):
        log = self._five(tmp_path)
        rows = _read_lines(log)
        rows[1], rows[2] = rows[2], rows[1]
        _write_lines(log, rows)
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 1

    def test_tail_delete_fails(self, tmp_path):
        log = self._five(tmp_path)
        rows = _read_lines(log)
        _write_lines(log, rows[:-1])
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 3
        assert "anchor" in (result.error_message or "").lower()

    def test_tail_edit_and_rehash_fails(self, tmp_path):
        log = self._five(tmp_path)
        key = log._get_key()
        rows = _read_lines(log)
        rows[-1]["operator_username"] = "TAMPERED"
        entry = AuditEntry(**rows[-1])
        rows[-1]["entry_hmac"] = entry.compute_hmac(key)
        _write_lines(log, rows)
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 4
        assert "Head anchor HMAC mismatch" in (result.error_message or "")

    def test_full_rehash_after_edit_fails(self, tmp_path):
        log = self._five(tmp_path)
        key = log._get_key()
        rows = _read_lines(log)
        rows[0]["operator_username"] = "TAMPERED"
        _write_lines(log, _rehash(rows, key))
        result = log.verify()
        assert result.valid is False
        assert "anchor" in (result.error_message or "").lower()

    def test_forged_genesis_fails(self, tmp_path):
        log = self._five(tmp_path)
        key = log._get_key()
        forged = AuditEntry(
            timestamp="2026-01-01T00:00:00+05:30",
            operator_username="attacker",
            action="login",
            prev_hmac="",
        )
        forged.entry_hmac = forged.compute_hmac(key)
        rows = [asdict(forged)] + _read_lines(log)
        _write_lines(log, rows)
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 1
        assert "Chain break" in (result.error_message or "")

    def test_chain_rehashed_from_entry_zero_fails(self, tmp_path):
        log = self._five(tmp_path)
        key = log._get_key()
        rows = _read_lines(log)
        rows[0]["query"] = "rewritten-from-genesis"
        _write_lines(log, _rehash(rows, key))
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 4
        assert "Head anchor HMAC mismatch" in (result.error_message or "")

    def test_tail_delete_and_deleted_anchor_fails(self, tmp_path):
        log = self._five(tmp_path)
        rows = _read_lines(log)
        _write_lines(log, rows[:-1])
        log.head_path.unlink()
        result = log.verify()
        assert result.valid is False
        assert "Missing head anchor" in (result.error_message or "")

    def test_tail_delete_and_unsigned_anchor_rewrite_fails(self, tmp_path):
        log = self._five(tmp_path)
        rows = _read_lines(log)
        kept = rows[:-1]
        _write_lines(log, kept)
        _write_anchor(log.head_path, None, kept[-1]["entry_hmac"], len(kept))
        result = log.verify()
        assert result.valid is False
        assert "signature" in (result.error_message or "").lower()

    def test_full_rehash_with_wrong_key_fails(self, tmp_path):
        log = self._five(tmp_path)
        wrong = os.urandom(32)
        rows = _read_lines(log)
        rows[1]["operator_username"] = "TAMPERED"
        rebuilt = _rehash(rows, wrong)
        _write_lines(log, rebuilt)
        _write_anchor(log.head_path, wrong, rebuilt[-1]["entry_hmac"], len(rebuilt))
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 0
        assert "HMAC mismatch" in (result.error_message or "")

    def test_delete_last_two_then_append_raises(self, tmp_path):
        log = self._five(tmp_path)
        _write_lines(log, _read_lines(log)[:-2])
        with pytest.raises(RuntimeError, match="does not match"):
            _append_login(log)
        result = log.verify()
        assert result.valid is False
        assert log.count() == 3

    def test_wipe_then_append_raises(self, tmp_path):
        log = self._five(tmp_path)
        log.log_path.write_text("", encoding="utf-8")
        with pytest.raises(RuntimeError, match="does not match"):
            _append_login(log)
        result = log.verify()
        assert result.valid is False
        assert log.count() == 0

    def test_delete_log_and_anchor_then_append_raises(self, tmp_path):
        log = self._five(tmp_path)
        log.log_path.unlink()
        log.head_path.unlink()
        with pytest.raises(RuntimeError, match="not initialized"):
            _append_login(log)
        assert not log.log_path.exists()
        result = log.verify()
        assert result.valid is False

    def test_missing_anchor_on_nonempty_log_refuses_append(self, tmp_path):
        log = self._five(tmp_path)
        log.head_path.unlink()
        with pytest.raises(RuntimeError, match="Missing head anchor"):
            _append_login(log)
        assert log.count() == 5
        result = log.verify()
        assert result.valid is False
        assert "Missing head anchor" in (result.error_message or "")

    def test_full_truncation_fails(self, tmp_path):
        log = self._five(tmp_path)
        log.log_path.write_text("", encoding="utf-8")
        result = log.verify()
        assert result.valid is False
        assert result.total_entries == 0
        assert result.broken_index == 0
        assert "truncated" in (result.error_message or "")

        log.log_path.unlink()
        deleted = log.verify()
        assert deleted.valid is False
        assert "truncated" in (deleted.error_message or "")

    def test_two_logs_do_not_share_an_anchor(self, tmp_path, monkeypatch):
        key_path = tmp_path / "keys" / "shared.key"
        init_hmac_key(key_path)
        first = AuditLog(tmp_path / "a.jsonl", key_path=key_path)
        second = AuditLog(tmp_path / "b.jsonl", key_path=key_path)
        first.initialize()
        second.initialize()
        _append_login(first, "alice")
        for index in range(3):
            _append_login(second, f"bob{index}")
        assert first.head_path != second.head_path
        assert first.head_path.name == "a.jsonl.head"
        assert second.head_path.name == "b.jsonl.head"
        other = tmp_path / "elsewhere"
        other.mkdir()
        monkeypatch.chdir(other)
        reopened = AuditLog(first.log_path, key_path=key_path)
        assert reopened.head_path == first.head_path
        assert reopened.verify().valid is True
        assert reopened.verify().total_entries == 1
        assert AuditLog(second.log_path, key_path=key_path).verify().total_entries == 3

    def test_wrong_key_file_fails(self, tmp_path):
        log = _create_audit_log(tmp_path)
        _append_login(log)
        other = tmp_path / "keys" / "other.key"
        init_hmac_key(other)
        forged = AuditLog(log.log_path, key_path=other)
        result = forged.verify()
        assert result.valid is False
        assert result.broken_index == 0
        assert "HMAC mismatch" in (result.error_message or "")


class TestVerifyNeverRaises:
    def test_malformed_json(self, tmp_path):
        log = _create_audit_log(tmp_path)
        _append_login(log)
        with log.log_path.open("a", encoding="utf-8") as handle:
            handle.write("{this is not valid json\n")
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 1
        assert "Malformed JSON" in (result.error_message or "")

    def test_null_and_number_lines_name_the_index(self, tmp_path):
        log = _create_audit_log(tmp_path)
        _append_login(log)
        with log.log_path.open("a", encoding="utf-8") as handle:
            handle.write("\nnull\n")
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 1
        assert "wrong type" in (result.error_message or "")

    def test_missing_field_wrong_type_and_extra_key(self, tmp_path):
        log = _create_audit_log(tmp_path)
        _append_login(log)
        missing = {
            "timestamp": "2026-10-20T12:00:00+05:30",
            "operator_username": "user1",
            "result_ids": [],
            "prev_hmac": "",
            "entry_hmac": "abc",
        }
        with log.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(missing) + "\n")
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 1
        assert "Missing required field 'action'" in (result.error_message or "")

        log2 = _create_audit_log(tmp_path, filename="types.jsonl")
        _append_login(log2)
        wrong = {
            "timestamp": "2026-10-20T12:00:00+05:30",
            "operator_username": "user1",
            "action": "login",
            "query": None,
            "case_id": None,
            "purpose": None,
            "result_ids": "not-a-list",
            "prev_hmac": "",
            "entry_hmac": "abc",
        }
        with log2.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(wrong) + "\n")
        typed = log2.verify()
        assert typed.valid is False
        assert typed.broken_index == 1
        assert "result_ids" in (typed.error_message or "")
        assert "wrong type" in (typed.error_message or "")

        log3 = _create_audit_log(tmp_path, filename="extra.jsonl")
        _append_login(log3)
        extra = {
            "timestamp": "2026-10-20T12:00:00+05:30",
            "operator_username": "user1",
            "action": "login",
            "query": None,
            "case_id": None,
            "purpose": None,
            "result_ids": [],
            "prev_hmac": "",
            "entry_hmac": "abc",
            "face_id": "must-not-be-here",
        }
        with log3.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(extra) + "\n")
        planted = log3.verify()
        assert planted.valid is False
        assert planted.broken_index == 1
        assert "Extra fields" in (planted.error_message or "")
        assert "face_id" in (planted.error_message or "")

    def test_non_utf8_log_line_names_the_index(self, tmp_path):
        log = _create_audit_log(tmp_path)
        _append_login(log)
        with log.log_path.open("ab") as handle:
            handle.write(b"\xff\xfe{bad}\n")
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 1
        assert "UTF-8" in (result.error_message or "")

    def test_non_utf8_anchor_returns_invalid(self, tmp_path):
        log = _create_audit_log(tmp_path)
        _append_login(log)
        log.head_path.write_bytes(b"\xff\xfe")
        result = log.verify()
        assert result.valid is False
        assert "UTF-8" in (result.error_message or "")

    def test_blank_details_names_the_index(self, tmp_path):
        log = _create_audit_log(tmp_path)
        for _ in range(3):
            _append_login(log)
        rows = _read_lines(log)
        rows[2]["details"] = "   "
        _write_lines(log, rows)
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 2
        assert "details" in (result.error_message or "")
        assert "entry 2" in (result.error_message or "")

    def test_unknown_action_in_the_log_is_rejected(self, tmp_path):
        log = _create_audit_log(tmp_path)
        _append_login(log)
        row = _read_lines(log)[0]
        row["action"] = "face_scan"
        _write_lines(log, [row])
        result = log.verify()
        assert result.valid is False
        assert result.broken_index == 0
        assert "Unknown action" in (result.error_message or "")


class TestAC13:
    def test_fifty_actions_produce_fifty_entries(self, tmp_path):
        log = _create_audit_log(tmp_path)
        for index in range(50):
            action = _ACTIONS[index % len(_ACTIONS)]
            log.append(
                operator_username=f"operator_{index}",
                action=action,
                query=f"query-{index}",
                case_id=f"CASE-{index:03d}",
                purpose=f"purpose-{index}",
                result_ids=[f"result-{index}"],
            )
        assert log.count() == 50
        entries = log.read_all()
        assert len(entries) == 50
        seen = set()
        for index, entry in enumerate(entries):
            seen.add(entry.action)
            assert entry.operator_username == f"operator_{index}"
            assert entry.timestamp.endswith("+05:30")
            assert entry.query == f"query-{index}"
            assert entry.case_id == f"CASE-{index:03d}"
            assert entry.purpose == f"purpose-{index}"
            assert entry.result_ids == [f"result-{index}"]
            assert len(entry.entry_hmac) == 64
            if index == 0:
                assert entry.prev_hmac == ""
            else:
                assert entry.prev_hmac == entries[index - 1].entry_hmac
        assert seen == {item.value for item in AuditAction}
        result = log.verify()
        assert result.valid is True
        assert result.total_entries == 50

    def test_edit_delete_and_swap_fail(self, tmp_path):
        log = _create_audit_log(tmp_path)
        for index in range(10):
            _append_login(log, f"user_{index}")
        rows = _read_lines(log)

        edited = [dict(row) for row in rows]
        edited[5]["operator_username"] = "TAMPERED"
        _write_lines(log, edited)
        edit_result = log.verify()
        assert edit_result.valid is False
        assert edit_result.broken_index == 5

        deleted = [dict(row) for row in rows]
        del deleted[3]
        _write_lines(log, deleted)
        delete_result = log.verify()
        assert delete_result.valid is False
        assert delete_result.broken_index == 3

        swapped = [dict(row) for row in rows]
        swapped[4], swapped[5] = swapped[5], swapped[4]
        _write_lines(log, swapped)
        swap_result = log.verify()
        assert swap_result.valid is False
        assert swap_result.broken_index == 4


class TestConcurrency:
    def test_eight_processes_fifty_appends_do_not_fork(self, tmp_path):
        log = _create_audit_log(tmp_path)
        context = multiprocessing.get_context("fork")
        processes = []
        for process_id in range(8):
            proc = context.Process(
                target=_concurrent_worker,
                args=(str(log.log_path), str(log.key_path), process_id),
            )
            proc.start()
            processes.append(proc)
        exit_codes = []
        for proc in processes:
            proc.join(timeout=120)
            exit_codes.append(proc.exitcode)
        assert exit_codes == [0, 0, 0, 0, 0, 0, 0, 0]

        result = log.verify()
        assert result.valid is True, result.error_message
        assert result.total_entries == 400
        entries = log.read_all()
        prev_hmacs = [entry.prev_hmac for entry in entries]
        assert prev_hmacs.count("") == 1
        assert len(prev_hmacs) == len(set(prev_hmacs))
        assert log.count() == 400

    def test_verify_during_appends_stays_valid(self, tmp_path):
        log = _create_audit_log(tmp_path)
        _append_login(log, "seed")
        stop = threading.Event()
        seen = threading.Event()
        results: list[bool] = []
        errors: list[str] = []
        guard = threading.Lock()

        def reader() -> None:
            try:
                while not stop.is_set():
                    result = log.verify()
                    with guard:
                        results.append(result.valid)
                        if not result.valid and len(errors) < 5:
                            errors.append(result.error_message or "")
                    seen.set()
            except Exception as exc:
                errors.append(repr(exc))
                stop.set()

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for thread in readers:
            thread.start()
        assert seen.wait(timeout=5)
        for index in range(40):
            log.append(f"user{index}", AuditAction.LOGIN)
        stop.set()
        for thread in readers:
            thread.join(timeout=30)
        assert all(not thread.is_alive() for thread in readers)
        assert results
        assert all(results), errors
        final = log.verify()
        assert final.valid is True
        assert final.total_entries == 41


def test_demo_script_prints_json():
    """The AC-13 script runs outside the repo tree and prints JSON."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "demo_audit_log.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    proof = json.loads(completed.stdout)
    assert proof["passed"] is True
    assert proof["fifty_actions"]["count"] == 50
    assert proof["fifty_actions"]["verification"]["valid"] is True
    assert proof["tamper"]["edit"]["broken_index"] == 5
    assert proof["tamper"]["delete"]["broken_index"] == 3
    assert proof["tamper"]["swap"]["broken_index"] == 4
    assert proof["chain"] == "HMAC-SHA256"
