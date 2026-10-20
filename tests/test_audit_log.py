"""
Tests for the hash-chained audit log.

Verifies AC-13: 50 scripted actions produce 50/50 entries with all fields,
and tampering detection identifies the broken index.
"""

import json
import tempfile
from pathlib import Path

from trace_search.audit import AuditLog


class TestAuditLogBasic:
    """Basic audit log functionality tests."""

    def test_empty_log_is_valid(self):
        """Empty log should verify successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            result = audit_log.verify()
            assert result.valid
            assert result.total_entries == 0

    def test_single_entry_append(self):
        """Single entry should be appended with correct fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            entry = audit_log.append(
                operator_username="test_user",
                action="login",
                case_id="CASE-001",
                purpose="Test login",
                result_ids=["result-1"],
            )

            assert entry.operator_username == "test_user"
            assert entry.action == "login"
            assert entry.case_id == "CASE-001"
            assert entry.purpose == "Test login"
            assert entry.result_ids == ["result-1"]
            assert entry.prev_hash == ""
            assert entry.entry_hash != ""
            assert audit_log.count() == 1

    def test_multiple_entries_chain(self):
        """Multiple entries should form a valid hash chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            entry1 = audit_log.append("user1", "login")
            entry2 = audit_log.append("user1", "search", query="red shirt")
            entry3 = audit_log.append("user1", "view", result_ids=["r1", "r2"])

            assert entry2.prev_hash == entry1.entry_hash
            assert entry3.prev_hash == entry2.entry_hash
            assert audit_log.count() == 3

            result = audit_log.verify()
            assert result.valid
            assert result.total_entries == 3

    def test_ist_timezone(self):
        """Timestamps should be in IST (Asia/Kolkata) timezone."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            entry = audit_log.append("user1", "login")

            # Timestamp should contain +05:30 offset (IST)
            assert "+05:30" in entry.timestamp

    def test_all_fields_populated(self):
        """Entry with all fields should preserve all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            entry = audit_log.append(
                operator_username="operator_alice",
                action="export",
                query="person with blue backpack",
                case_id="CASE-2024-001",
                purpose="Patient search incident #42",
                result_ids=["track-001", "track-002", "track-003"],
            )

            assert entry.operator_username == "operator_alice"
            assert entry.action == "export"
            assert entry.query == "person with blue backpack"
            assert entry.case_id == "CASE-2024-001"
            assert entry.purpose == "Patient search incident #42"
            assert entry.result_ids == ["track-001", "track-002", "track-003"]
            assert entry.timestamp != ""
            assert entry.entry_hash != ""


class TestAuditLogAC13:
    """
    Tests for AC-13: 50 scripted actions produce 50/50 entries,
    and tampering detection identifies the broken index.
    """

    def test_fifty_actions_produce_fifty_entries(self):
        """
        AC-13 requirement: 50 scripted actions produce 50/50 entries
        with who, when, action, case ID and result IDs all populated.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            # Script 50 different actions
            actions = [
                ("operator_1", "login", None, "CASE-001", "Morning shift start", []),
                ("operator_1", "search", "red shirt", "CASE-001", "Search suspect", ["r1", "r2"]),
                ("operator_1", "view", None, "CASE-001", "Review results", ["r1"]),
                ("operator_1", "export", None, "CASE-001", "Export for report", ["r1"]),
                ("operator_2", "login", None, "CASE-002", "Afternoon shift", []),
                ("operator_2", "search", "blue bag", "CASE-002", "Lost item", ["r3"]),
                ("operator_2", "view", None, "CASE-002", "Review", ["r3"]),
                ("operator_1", "retention_change", "7d->14d", "SYS-001", "Policy update", []),
                ("operator_2", "search", "green jacket", "CASE-003", "Incident", ["r4", "r5"]),
                ("operator_2", "view", None, "CASE-003", "Review", ["r4"]),
                ("operator_2", "export", None, "CASE-003", "Export", ["r4", "r5"]),
                ("operator_3", "login", None, "CASE-004", "Night shift", []),
                ("operator_3", "search", "black pants", "CASE-004", "Search", ["r6"]),
                ("operator_3", "view", None, "CASE-004", "Review", ["r6"]),
                ("operator_1", "purge", "2026-10-01", "SYS-002", "Monthly cleanup", []),
                ("operator_3", "search", "yellow shirt", "CASE-005", "Search", ["r7", "r8"]),
                ("operator_3", "view", None, "CASE-005", "Review", ["r7"]),
                ("operator_3", "export", None, "CASE-005", "Export", ["r7"]),
                ("operator_1", "login", None, "CASE-006", "Second login", []),
                ("operator_1", "search", "white shirt", "CASE-006", "Search", ["r9"]),
                ("operator_1", "view", None, "CASE-006", "Review", ["r9"]),
                ("operator_2", "login", None, "CASE-007", "Login", []),
                ("operator_2", "search", "gray bag", "CASE-007", "Search", ["r10"]),
                ("operator_2", "view", None, "CASE-007", "Review", ["r10"]),
                ("operator_2", "export", None, "CASE-007", "Export", ["r10"]),
                ("operator_3", "login", None, "CASE-008", "Login", []),
                ("operator_3", "search", "orange jacket", "CASE-008", "Search", ["r11", "r12"]),
                ("operator_3", "view", None, "CASE-008", "Review", ["r11"]),
                ("operator_1", "retention_change", "14d->7d", "SYS-003", "Revert policy", []),
                ("operator_1", "search", "purple bag", "CASE-009", "Search", ["r13"]),
                ("operator_1", "view", None, "CASE-009", "Review", ["r13"]),
                ("operator_2", "login", None, "CASE-010", "Login", []),
                ("operator_2", "search", "brown shoes", "CASE-010", "Search", ["r14"]),
                ("operator_2", "view", None, "CASE-010", "Review", ["r14"]),
                ("operator_2", "export", None, "CASE-010", "Export", ["r14"]),
                ("operator_3", "search", "pink shirt", "CASE-011", "Search", ["r15"]),
                ("operator_3", "view", None, "CASE-011", "Review", ["r15"]),
                ("operator_1", "purge", "2026-10-15", "SYS-004", "Cleanup", []),
                ("operator_1", "search", "navy pants", "CASE-012", "Search", ["r16", "r17"]),
                ("operator_1", "view", None, "CASE-012", "Review", ["r16"]),
                ("operator_2", "login", None, "CASE-013", "Login", []),
                ("operator_2", "search", "teal shirt", "CASE-013", "Search", ["r18"]),
                ("operator_2", "view", None, "CASE-013", "Review", ["r18"]),
                ("operator_3", "login", None, "CASE-014", "Login", []),
                ("operator_3", "search", "maroon bag", "CASE-014", "Search", ["r19"]),
                ("operator_3", "view", None, "CASE-014", "Review", ["r19"]),
                ("operator_3", "export", None, "CASE-014", "Export", ["r19"]),
                ("operator_1", "search", "beige jacket", "CASE-015", "Search", ["r20"]),
                ("operator_1", "view", None, "CASE-015", "Review", ["r20"]),
                ("operator_2", "retention_change", "7d->30d", "SYS-005", "Extend", []),
            ]

            # Execute all 50 actions
            for username, action, query, case_id, purpose, result_ids in actions:
                audit_log.append(
                    operator_username=username,
                    action=action,
                    query=query,
                    case_id=case_id,
                    purpose=purpose,
                    result_ids=result_ids,
                )

            # Verify count
            assert audit_log.count() == 50, "Should have exactly 50 entries"

            # Verify all entries have all required fields populated
            entries = audit_log.read_all()
            assert len(entries) == 50

            for i, entry in enumerate(entries):
                # Check who (operator_username)
                assert entry.operator_username, f"Entry {i} missing operator_username"
                assert entry.operator_username.startswith("operator_"), \
                    f"Entry {i} has invalid operator: {entry.operator_username}"

                # Check when (timestamp)
                assert entry.timestamp, f"Entry {i} missing timestamp"
                assert "+05:30" in entry.timestamp, \
                    f"Entry {i} timestamp not in IST: {entry.timestamp}"

                # Check action
                assert entry.action, f"Entry {i} missing action"
                assert entry.action in [
                    "login", "search", "view", "export",
                    "retention_change", "purge"
                ], f"Entry {i} has invalid action: {entry.action}"

                # Check case_id (all scripted actions have case_id)
                assert entry.case_id, f"Entry {i} missing case_id"

                # Check purpose (all scripted actions have purpose)
                assert entry.purpose, f"Entry {i} missing purpose"

                # result_ids is a list (may be empty for login actions)
                assert isinstance(entry.result_ids, list), \
                    f"Entry {i} result_ids is not a list"

                # Check hash fields
                assert entry.entry_hash, f"Entry {i} missing entry_hash"

            # Verify chain is valid
            result = audit_log.verify()
            assert result.valid, f"Chain verification failed: {result.error_message}"
            assert result.total_entries == 50

    def test_edit_entry_breaks_verification(self):
        """
        AC-13 requirement: Editing one entry fails verification
        and names the broken index.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            # Create 10 entries
            for i in range(10):
                audit_log.append(
                    operator_username=f"user_{i}",
                    action="search",
                    case_id=f"CASE-{i:03d}",
                    purpose="Test",
                    result_ids=[f"r{i}"],
                )

            # Verify initial chain is valid
            result = audit_log.verify()
            assert result.valid

            # Tamper: edit entry 5 (0-indexed)
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Modify line 5 (entry at index 5)
            entry_dict = json.loads(lines[5])
            entry_dict["operator_username"] = "TAMPERED_USER"
            lines[5] = json.dumps(entry_dict) + "\n"

            with open(log_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            # Verify again - should detect tampering
            result = audit_log.verify()
            assert not result.valid, "Should detect tampering"
            assert result.broken_index == 5, \
                f"Should identify index 5 as broken, got {result.broken_index}"
            assert "hash mismatch" in result.error_message.lower()

    def test_delete_entry_breaks_verification(self):
        """
        AC-13 requirement: Deleting one entry fails verification
        and names the broken index.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            # Create 10 entries
            for i in range(10):
                audit_log.append(
                    operator_username=f"user_{i}",
                    action="view",
                    case_id=f"CASE-{i:03d}",
                    purpose="Test",
                    result_ids=[],
                )

            # Verify initial chain is valid
            result = audit_log.verify()
            assert result.valid

            # Tamper: delete entry 3
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Remove line 3
            del lines[3]

            with open(log_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            # Verify again - should detect deletion (chain break)
            result = audit_log.verify()
            assert not result.valid, "Should detect deletion"
            # After deleting entry 3, entry 4 (now at index 3) will have wrong prev_hash
            assert result.broken_index == 3, \
                f"Should identify index 3 as broken, got {result.broken_index}"
            assert "chain break" in result.error_message.lower()

    def test_swap_entries_breaks_verification(self):
        """
        AC-13 requirement: Swapping entries fails verification
        and names the broken index.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            # Create 10 entries
            for i in range(10):
                audit_log.append(
                    operator_username=f"user_{i}",
                    action="export",
                    case_id=f"CASE-{i:03d}",
                    purpose="Test",
                    result_ids=[f"r{i}"],
                )

            # Verify initial chain is valid
            result = audit_log.verify()
            assert result.valid

            # Tamper: swap entries 4 and 5
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Swap lines 4 and 5
            lines[4], lines[5] = lines[5], lines[4]

            with open(log_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            # Verify again - should detect swap (chain break)
            result = audit_log.verify()
            assert not result.valid, "Should detect swap"
            # Entry at index 4 will have wrong prev_hash
            assert result.broken_index == 4, \
                f"Should identify index 4 as broken, got {result.broken_index}"


class TestAuditLogRetention:
    """Tests for retention and separation from derived data."""

    def test_audit_log_separate_location(self):
        """
        Audit log should be stored in a separate location
        that can be exempted from derived-data purge.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Audit log in dedicated audit_logs/ directory
            log_path = Path(tmpdir) / "audit_logs" / "audit.jsonl"
            audit_log = AuditLog(log_path)

            audit_log.append("user1", "login")

            # Check log is in audit_logs/ subdirectory
            assert log_path.parent.name == "audit_logs"
            assert log_path.exists()

    def test_audit_log_default_location(self):
        """Default audit log location should be in audit_logs/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Use default location
                audit_log = AuditLog()
                audit_log.append("user1", "login")

                # Should be in audit_logs/audit.jsonl relative to cwd
                expected_path = Path(tmpdir) / "audit_logs" / "audit.jsonl"
                assert expected_path.exists()
                assert audit_log.log_path == expected_path
            finally:
                os.chdir(original_cwd)


class TestAuditLogQueries:
    """Tests for audit log query functionality."""

    def test_query_by_operator(self):
        """Should filter entries by operator username."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            audit_log.append("alice", "login")
            audit_log.append("bob", "search", query="test")
            audit_log.append("alice", "view")
            audit_log.append("charlie", "export")
            audit_log.append("alice", "purge")

            alice_entries = audit_log.query_by_operator("alice")
            assert len(alice_entries) == 3
            assert all(e.operator_username == "alice" for e in alice_entries)

    def test_query_by_action(self):
        """Should filter entries by action type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            audit_log.append("user1", "login")
            audit_log.append("user2", "search")
            audit_log.append("user1", "login")
            audit_log.append("user3", "view")
            audit_log.append("user2", "login")

            login_entries = audit_log.query_by_action("login")
            assert len(login_entries) == 3
            assert all(e.action == "login" for e in login_entries)

    def test_query_by_case_id(self):
        """Should filter entries by case ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = AuditLog(log_path)

            audit_log.append("user1", "search", case_id="CASE-001")
            audit_log.append("user1", "view", case_id="CASE-001")
            audit_log.append("user2", "search", case_id="CASE-002")
            audit_log.append("user1", "export", case_id="CASE-001")

            case1_entries = audit_log.query_by_case_id("CASE-001")
            assert len(case1_entries) == 3
            assert all(e.case_id == "CASE-001" for e in case1_entries)
