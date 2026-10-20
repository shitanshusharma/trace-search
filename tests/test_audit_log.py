"""
Tests for the HMAC-chained audit log.

Verifies AC-13 plus Engineering Lead review blockers B1-B3:
- 50 scripted actions produce 50/50 entries with all fields
- Tampering detection (including tail attacks) identifies the broken index
- verify() never raises on malformed data
- Action validation and case_id/purpose requirements
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from trace_search.audit import (
    AUDIT_LOG_DIR,
    AuditAction,
    AuditEntry,
    AuditLog,
    is_audit_path,
)


class TestAuditPathHelper:
    """Tests for is_audit_path() helper function."""

    def test_audit_dir_itself(self):
        """Audit directory itself should be recognized as audit path."""
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                audit_dir = Path(AUDIT_LOG_DIR)
                audit_dir.mkdir()
                assert is_audit_path(audit_dir)
        finally:
            os.chdir(original_cwd)

    def test_file_inside_audit_dir(self):
        """Files inside audit directory should be recognized."""
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                audit_dir = Path(AUDIT_LOG_DIR)
                audit_dir.mkdir()
                audit_file = audit_dir / "audit.jsonl"
                audit_file.touch()
                assert is_audit_path(audit_file)
        finally:
            os.chdir(original_cwd)

    def test_nested_path_inside_audit_dir(self):
        """Nested paths inside audit directory should be recognized."""
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                audit_dir = Path(AUDIT_LOG_DIR)
                nested_dir = audit_dir / "subdir" / "deep"
                nested_dir.mkdir(parents=True)
                nested_file = nested_dir / "file.txt"
                nested_file.touch()
                assert is_audit_path(nested_file)
        finally:
            os.chdir(original_cwd)

    def test_path_outside_audit_dir(self, tmp_path):
        """Paths outside audit directory should not be recognized."""
        other_file = tmp_path / "data" / "file.txt"
        other_file.parent.mkdir()
        other_file.touch()
        assert not is_audit_path(other_file)

    def test_sibling_path(self, tmp_path):
        """Sibling paths should not be recognized."""
        audit_dir = tmp_path / AUDIT_LOG_DIR
        audit_dir.mkdir()
        sibling = tmp_path / "audit_logs_backup" / "file.txt"
        sibling.parent.mkdir()
        sibling.touch()
        assert not is_audit_path(sibling)

    def test_relative_vs_absolute(self):
        """Both relative and absolute paths should work."""
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                audit_dir = Path(AUDIT_LOG_DIR)
                audit_dir.mkdir()

                # Relative path
                assert is_audit_path(AUDIT_LOG_DIR)
                assert is_audit_path(f"{AUDIT_LOG_DIR}/file.txt")

                # Absolute path
                assert is_audit_path(audit_dir)
        finally:
            os.chdir(original_cwd)


class TestAuditLogBasic:
    """Basic audit log functionality tests."""

    def test_empty_log_is_valid(self, tmp_path):
        """Empty log should verify successfully."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        result = audit_log.verify()
        assert result.valid
        assert result.total_entries == 0

    def test_single_entry_append(self, tmp_path):
        """Single entry should be appended with correct fields."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        entry = audit_log.append(
            operator_username="test_user",
            action=AuditAction.LOGIN,
            case_id="CASE-001",
            purpose="Test login",
            result_ids=["result-1"],
        )

        assert entry.operator_username == "test_user"
        assert entry.action == "login"
        assert entry.case_id == "CASE-001"
        assert entry.purpose == "Test login"
        assert entry.result_ids == ["result-1"]
        assert entry.prev_hmac == ""
        assert entry.entry_hmac != ""
        assert audit_log.count() == 1

    def test_multiple_entries_chain(self, tmp_path):
        """Multiple entries should form a valid HMAC chain."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        entry1 = audit_log.append("user1", AuditAction.LOGIN)
        entry2 = audit_log.append(
            "user1", AuditAction.SEARCH, query="red shirt", case_id="C1", purpose="Test"
        )
        entry3 = audit_log.append(
            "user1", AuditAction.VIEW, result_ids=["r1", "r2"], case_id="C1", purpose="Test"
        )

        assert entry2.prev_hmac == entry1.entry_hmac
        assert entry3.prev_hmac == entry2.entry_hmac
        assert audit_log.count() == 3

        result = audit_log.verify()
        assert result.valid
        assert result.total_entries == 3

    def test_ist_timezone(self, tmp_path):
        """Timestamps should be in IST (Asia/Kolkata) timezone."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        entry = audit_log.append("user1", AuditAction.LOGIN)

        # Timestamp should contain +05:30 offset (IST)
        assert "+05:30" in entry.timestamp

    def test_injectable_clock(self, tmp_path):
        """Should accept injectable clock for testing."""
        log_path = tmp_path / "audit.jsonl"
        fixed_time = datetime(2026, 10, 20, 12, 0, 0)

        def mock_clock():
            return fixed_time.replace(tzinfo=datetime.now().astimezone().tzinfo)

        audit_log = AuditLog(log_path, key_path=tmp_path / ".key", clock=mock_clock)

        entry = audit_log.append("user1", AuditAction.LOGIN)
        assert "2026-10-20T12:00:00" in entry.timestamp


class TestActionValidation:
    """Tests for B3: Action validation and required fields."""

    def test_unknown_action_rejected(self, tmp_path):
        """Unknown action string should be rejected."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        with pytest.raises(ValueError, match="Invalid action 'invalid_action'"):
            audit_log.append("user1", "invalid_action")

    def test_valid_action_enum(self, tmp_path):
        """Valid action enum should be accepted."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        entry = audit_log.append("user1", AuditAction.LOGIN)
        assert entry.action == "login"

    def test_valid_action_string(self, tmp_path):
        """Valid action string should be accepted."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        entry = audit_log.append("user1", "login")
        assert entry.action == "login"

    def test_search_requires_case_id(self, tmp_path):
        """Search action requires case_id (PRD §8)."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        with pytest.raises(ValueError, match="case_id is required for action 'search'"):
            audit_log.append("user1", AuditAction.SEARCH, purpose="Test")

    def test_search_requires_purpose(self, tmp_path):
        """Search action requires purpose (PRD §8)."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        with pytest.raises(ValueError, match="purpose is required for action 'search'"):
            audit_log.append("user1", AuditAction.SEARCH, case_id="C1")

    def test_view_requires_case_id_and_purpose(self, tmp_path):
        """View action requires case_id and purpose (PRD §8)."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        with pytest.raises(ValueError, match="case_id is required"):
            audit_log.append("user1", AuditAction.VIEW, purpose="Test")

        with pytest.raises(ValueError, match="purpose is required"):
            audit_log.append("user1", AuditAction.VIEW, case_id="C1")

    def test_export_requires_case_id_and_purpose(self, tmp_path):
        """Export action requires case_id and purpose (PRD §8)."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        with pytest.raises(ValueError, match="case_id is required"):
            audit_log.append("user1", AuditAction.EXPORT, purpose="Test")

        with pytest.raises(ValueError, match="purpose is required"):
            audit_log.append("user1", AuditAction.EXPORT, case_id="C1")

    def test_login_does_not_require_case_id(self, tmp_path):
        """Login action does not require case_id or purpose."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        # Should not raise
        entry = audit_log.append("user1", AuditAction.LOGIN)
        assert entry.case_id is None
        assert entry.purpose is None

    def test_empty_case_id_rejected(self, tmp_path):
        """Empty or whitespace-only case_id should be rejected."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        with pytest.raises(ValueError, match="case_id is required"):
            audit_log.append("user1", AuditAction.SEARCH, case_id="  ", purpose="Test")


class TestTamperingDetection:
    """Tests for B1: Tail tampering and HMAC attacks."""

    def test_delete_last_entry_fails(self, tmp_path):
        """B1: Deleting the last entry should fail verification (head anchor mismatch)."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        for i in range(5):
            audit_log.append("user1", AuditAction.LOGIN)

        # Initial verification
        result = audit_log.verify()
        assert result.valid

        # Delete last line
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(lines[:-1])

        # Verify should fail (head anchor mismatch)
        result = audit_log.verify()
        assert not result.valid
        assert "Head anchor" in result.error_message

    def test_edit_last_entry_and_rehash_fails(self, tmp_path):
        """
        B1: Editing the last entry and recomputing its HMAC should fail
        (wrong key or head anchor).
        """
        log_path = tmp_path / "audit.jsonl"
        key_path = tmp_path / ".key"
        audit_log = AuditLog(log_path, key_path=key_path)

        for i in range(5):
            audit_log.append("user1", AuditAction.LOGIN)

        result = audit_log.verify()
        assert result.valid

        # Edit last entry and rehash it with the SAME key
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()

        entry_dict = json.loads(lines[-1])
        entry_dict["operator_username"] = "TAMPERED"

        # Recompute HMAC with the key
        key = audit_log._get_key()
        entry = AuditEntry(**entry_dict)
        entry.entry_hmac = entry.compute_hmac(key)

        lines[-1] = json.dumps(
            {
                **entry_dict,
                "entry_hmac": entry.entry_hmac,
            },
            separators=(",", ":"),
        ) + "\n"

        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Verify should still fail because head anchor won't match the new HMAC
        result = audit_log.verify()
        assert not result.valid
        assert "Head anchor" in result.error_message

    def test_rehash_entire_chain_fails(self, tmp_path):
        """B1: Rehashing the entire chain should fail (head anchor mismatch)."""
        log_path = tmp_path / "audit.jsonl"
        key_path = tmp_path / ".key"
        audit_log = AuditLog(log_path, key_path=key_path)

        for i in range(5):
            audit_log.append("user1", AuditAction.LOGIN)

        result = audit_log.verify()
        assert result.valid

        # Read all entries
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Modify first entry and rehash the entire chain
        key = audit_log._get_key()
        entries_dict = [json.loads(line) for line in lines]
        entries_dict[0]["operator_username"] = "TAMPERED"

        # Recompute all HMACs
        prev_hmac = ""
        for entry_dict in entries_dict:
            entry_dict["prev_hmac"] = prev_hmac
            entry_dict.pop("entry_hmac", None)
            entry = AuditEntry(**entry_dict)
            entry.entry_hmac = entry.compute_hmac(key)
            entry_dict["entry_hmac"] = entry.entry_hmac
            prev_hmac = entry.entry_hmac

        # Write back
        with open(log_path, "w", encoding="utf-8") as f:
            for entry_dict in entries_dict:
                f.write(json.dumps(entry_dict, separators=(",", ":")) + "\n")

        # Verify should fail because head anchor stores the old last HMAC
        result = audit_log.verify()
        assert not result.valid
        assert "Head anchor" in result.error_message

    def test_wrong_key_fails(self, tmp_path):
        """B1: Verifying with a different key should fail."""
        log_path = tmp_path / "audit.jsonl"
        key_path1 = tmp_path / ".key1"
        key_path2 = tmp_path / ".key2"

        # Create entries with key1
        audit_log1 = AuditLog(log_path, key_path=key_path1)
        for i in range(3):
            audit_log1.append("user1", AuditAction.LOGIN)

        # Verify with key1 should pass
        result = audit_log1.verify()
        assert result.valid

        # Verify with key2 should fail
        audit_log2 = AuditLog(log_path, key_path=key_path2)
        result = audit_log2.verify()
        assert not result.valid
        assert "HMAC mismatch" in result.error_message

    def test_forged_genesis_entry_fails(self, tmp_path):
        """(d) Forged genesis entry should fail verification."""
        log_path = tmp_path / "audit.jsonl"
        key_path = tmp_path / ".key"
        audit_log = AuditLog(log_path, key_path=key_path)

        # Create some entries
        for i in range(3):
            audit_log.append("user1", AuditAction.LOGIN)

        result = audit_log.verify()
        assert result.valid

        # Prepend a forged genesis entry
        key = audit_log._get_key()
        forged_entry = AuditEntry(
            timestamp="2026-01-01T00:00:00+05:30",
            operator_username="attacker",
            action="login",
            prev_hmac="",
        )
        forged_entry.entry_hmac = forged_entry.compute_hmac(key)

        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()

        forged_line = json.dumps(
            {
                "timestamp": forged_entry.timestamp,
                "operator_username": forged_entry.operator_username,
                "action": forged_entry.action,
                "query": None,
                "case_id": None,
                "purpose": None,
                "result_ids": [],
                "prev_hmac": "",
                "entry_hmac": forged_entry.entry_hmac,
            },
            separators=(",", ":"),
        ) + "\n"

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(forged_line)
            f.writelines(lines)

        # Verify should fail (chain break at entry 1)
        result = audit_log.verify()
        assert not result.valid
        assert result.broken_index == 1
        assert "Chain break" in result.error_message


class TestRobustVerification:
    """Tests for B2: verify() never raises."""

    def test_malformed_json(self, tmp_path):
        """B2: Malformed JSON should return valid=False, not raise."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        audit_log.append("user1", AuditAction.LOGIN)

        # Append malformed JSON
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("{this is not valid json\n")

        result = audit_log.verify()
        assert not result.valid
        assert result.broken_index == 1
        assert "Malformed JSON" in result.error_message

    def test_missing_required_field(self, tmp_path):
        """B2: Missing required field should return valid=False, not raise."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        audit_log.append("user1", AuditAction.LOGIN)

        # Append entry missing 'action' field
        with open(log_path, "a", encoding="utf-8") as f:
            bad_entry = {
                "timestamp": "2026-10-20T12:00:00+05:30",
                "operator_username": "user1",
                "result_ids": [],
                "prev_hmac": "",
                "entry_hmac": "abc",
            }
            f.write(json.dumps(bad_entry, separators=(",", ":")) + "\n")

        result = audit_log.verify()
        assert not result.valid
        assert result.broken_index == 1
        assert "Missing required field 'action'" in result.error_message

    def test_wrong_field_type(self, tmp_path):
        """B2: Wrong field type should return valid=False, not raise."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        audit_log.append("user1", AuditAction.LOGIN)

        # Append entry with wrong type for result_ids (should be list)
        with open(log_path, "a", encoding="utf-8") as f:
            bad_entry = {
                "timestamp": "2026-10-20T12:00:00+05:30",
                "operator_username": "user1",
                "action": "login",
                "query": None,
                "case_id": None,
                "purpose": None,
                "result_ids": "not a list",  # Wrong type
                "prev_hmac": "",
                "entry_hmac": "abc",
            }
            f.write(json.dumps(bad_entry, separators=(",", ":")) + "\n")

        result = audit_log.verify()
        assert not result.valid
        assert result.broken_index == 1
        assert "wrong type" in result.error_message
        assert "result_ids" in result.error_message

    def test_extra_field(self, tmp_path):
        """B2: Extra field should return valid=False, not raise."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        audit_log.append("user1", AuditAction.LOGIN)

        # Append entry with extra field
        with open(log_path, "a", encoding="utf-8") as f:
            bad_entry = {
                "timestamp": "2026-10-20T12:00:00+05:30",
                "operator_username": "user1",
                "action": "login",
                "query": None,
                "case_id": None,
                "purpose": None,
                "result_ids": [],
                "prev_hmac": "",
                "entry_hmac": "abc",
                "extra_field": "should not be here",  # Extra field
            }
            f.write(json.dumps(bad_entry, separators=(",", ":")) + "\n")

        result = audit_log.verify()
        assert not result.valid
        assert result.broken_index == 1
        assert "Extra fields" in result.error_message

    def test_key_mismatch_handled_gracefully(self, tmp_path):
        """B2: Key mismatch should return valid=False, not raise."""
        log_path = tmp_path / "audit.jsonl"
        key_path1 = tmp_path / ".key1"
        key_path2 = tmp_path / ".key2"

        # Create audit log with key1
        audit_log1 = AuditLog(log_path, key_path=key_path1)
        audit_log1.append("user1", AuditAction.LOGIN)

        # Verify with key2 (different key)
        audit_log2 = AuditLog(log_path, key_path=key_path2)
        result = audit_log2.verify()

        # Should not raise, should return valid=False
        assert not result.valid
        assert "HMAC" in result.error_message or "key" in result.error_message.lower()


class TestAC13:
    """Tests for AC-13: 50 scripted actions and tampering detection."""

    def test_fifty_actions_produce_fifty_entries(self, tmp_path):
        """AC-13: 50 scripted actions produce 50/50 entries with all fields."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        # Script 50 actions
        actions = []
        for i in range(50):
            if i % 5 == 0:
                actions.append(("login", None, None))
            elif i % 5 == 1:
                actions.append(("search", f"CASE-{i}", "Search query"))
            elif i % 5 == 2:
                actions.append(("view", f"CASE-{i}", "View results"))
            elif i % 5 == 3:
                actions.append(("export", f"CASE-{i}", "Export data"))
            else:
                actions.append(("retention_change", None, None))

        for action_type, case_id, purpose in actions:
            if action_type in ["search", "view", "export"]:
                audit_log.append(
                    f"operator_{len(actions)}",
                    action_type,
                    case_id=case_id,
                    purpose=purpose,
                )
            else:
                audit_log.append(f"operator_{len(actions)}", action_type)

        assert audit_log.count() == 50

        entries = audit_log.read_all()
        assert len(entries) == 50

        # Check all required fields populated
        for i, entry in enumerate(entries):
            assert entry.operator_username, f"Entry {i} missing operator_username"
            assert entry.timestamp, f"Entry {i} missing timestamp"
            assert "+05:30" in entry.timestamp, f"Entry {i} not IST"
            assert entry.action, f"Entry {i} missing action"
            assert entry.entry_hmac, f"Entry {i} missing entry_hmac"

        result = audit_log.verify()
        assert result.valid
        assert result.total_entries == 50

    def test_edit_entry_breaks_verification(self, tmp_path):
        """AC-13: Editing one entry fails verification."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        for i in range(10):
            audit_log.append(f"user_{i}", AuditAction.LOGIN)

        result = audit_log.verify()
        assert result.valid

        # Edit entry 5
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()

        entry_dict = json.loads(lines[5])
        entry_dict["operator_username"] = "TAMPERED"
        lines[5] = json.dumps(entry_dict, separators=(",", ":")) + "\n"

        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        result = audit_log.verify()
        assert not result.valid
        assert result.broken_index == 5

    def test_delete_middle_entry_breaks_verification(self, tmp_path):
        """AC-13: Deleting one entry fails verification."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        for i in range(10):
            audit_log.append(f"user_{i}", AuditAction.LOGIN)

        result = audit_log.verify()
        assert result.valid

        # Delete entry 3
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()

        del lines[3]

        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        result = audit_log.verify()
        assert not result.valid
        assert result.broken_index == 3

    def test_swap_entries_breaks_verification(self, tmp_path):
        """AC-13: Swapping entries fails verification."""
        log_path = tmp_path / "audit.jsonl"
        audit_log = AuditLog(log_path, key_path=tmp_path / ".key")

        for i in range(10):
            audit_log.append(f"user_{i}", AuditAction.LOGIN)

        result = audit_log.verify()
        assert result.valid

        # Swap entries 4 and 5
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()

        lines[4], lines[5] = lines[5], lines[4]

        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        result = audit_log.verify()
        assert not result.valid
        assert result.broken_index == 4
