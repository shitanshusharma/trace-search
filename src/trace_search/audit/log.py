"""
Hash-chained audit log implementation.

Each entry records operator actions with tamper-proof hash chaining.
Entries are append-only and stored in a location exempt from derived-data purge.
"""

import hashlib
import json
import zoneinfo
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# IST timezone for all timestamps
IST = zoneinfo.ZoneInfo("Asia/Kolkata")


@dataclass
class AuditEntry:
    """
    A single audit log entry.

    Fields:
        timestamp: ISO 8601 timestamp in IST timezone (Asia/Kolkata)
        operator_username: Username of the operator performing the action
        action: Type of action (login, search, view, export, retention_change, purge, etc.)
        query: Optional query details (for search actions)
        case_id: Optional case identifier
        purpose: Optional purpose/reason for the action
        result_ids: List of result identifiers returned by the action
        prev_hash: SHA-256 hash of the previous entry (empty for first entry)
        entry_hash: SHA-256 hash of this entry (excluding this field itself)
    """
    timestamp: str
    operator_username: str
    action: str
    query: str | None = None
    case_id: str | None = None
    purpose: str | None = None
    result_ids: list[str] = field(default_factory=list)
    prev_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of this entry.

        The hash includes all fields except entry_hash itself.
        This allows verification of chain integrity.
        """
        # Create a copy of the entry dict without entry_hash
        entry_dict = asdict(self)
        entry_dict.pop("entry_hash", None)

        # Sort keys for deterministic hashing
        canonical = json.dumps(entry_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class VerificationResult:
    """
    Result of audit log verification.

    Fields:
        valid: True if the entire chain is valid, False otherwise
        total_entries: Total number of entries checked
        broken_index: Index of the first broken entry (None if all valid)
        error_message: Description of the verification failure
    """
    valid: bool
    total_entries: int
    broken_index: int | None = None
    error_message: str | None = None


class AuditLog:
    """
    Hash-chained audit log with tamper detection.

    The audit log is stored in a dedicated location separate from derived data
    to ensure it survives retention purges. Entries are kept for 1 year.

    Storage format: JSON Lines (one JSON object per line) for append-only writes.
    Location: audit_logs/ directory (exempt from derived-data purge)
    """

    def __init__(self, log_path: Path | None = None):
        """
        Initialize audit log.

        Args:
            log_path: Path to the audit log file. Defaults to audit_logs/audit.jsonl
                     in the current working directory.
        """
        if log_path is None:
            log_path = Path.cwd() / "audit_logs" / "audit.jsonl"

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Touch the file if it doesn't exist
        if not self.log_path.exists():
            self.log_path.touch()

    def _get_last_entry(self) -> AuditEntry | None:
        """
        Read the last entry from the log file.

        Returns:
            The last AuditEntry, or None if the log is empty.
        """
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return None

        # Read last line
        with open(self.log_path, encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return None

            last_line = lines[-1].strip()
            if not last_line:
                return None

            entry_dict = json.loads(last_line)
            return AuditEntry(**entry_dict)

    def append(
        self,
        operator_username: str,
        action: str,
        query: str | None = None,
        case_id: str | None = None,
        purpose: str | None = None,
        result_ids: list[str] | None = None,
    ) -> AuditEntry:
        """
        Append a new entry to the audit log.

        Args:
            operator_username: Username of the operator
            action: Type of action being logged
            query: Optional query details
            case_id: Optional case identifier
            purpose: Optional purpose/reason
            result_ids: Optional list of result identifiers

        Returns:
            The created AuditEntry with computed hash and chain link.
        """
        # Get timestamp in IST
        now = datetime.now(IST)
        timestamp = now.isoformat()

        # Get previous entry's hash
        last_entry = self._get_last_entry()
        prev_hash = last_entry.entry_hash if last_entry else ""

        # Create entry
        entry = AuditEntry(
            timestamp=timestamp,
            operator_username=operator_username,
            action=action,
            query=query,
            case_id=case_id,
            purpose=purpose,
            result_ids=result_ids or [],
            prev_hash=prev_hash,
        )

        # Compute hash
        entry.entry_hash = entry.compute_hash()

        # Append to file (JSON Lines format)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

        return entry

    def read_all(self) -> list[AuditEntry]:
        """
        Read all entries from the audit log.

        Returns:
            List of all AuditEntry objects in chronological order.
        """
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return []

        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry_dict = json.loads(line)
                    entries.append(AuditEntry(**entry_dict))

        return entries

    def verify(self) -> VerificationResult:
        """
        Verify the integrity of the audit log chain.

        Checks:
        1. Each entry's hash is correctly computed
        2. Each entry's prev_hash matches the previous entry's entry_hash
        3. No entries are missing (chain is unbroken)

        Returns:
            VerificationResult with details of any integrity violations.
        """
        entries = self.read_all()

        if not entries:
            return VerificationResult(
                valid=True,
                total_entries=0,
                error_message="Empty log (valid)",
            )

        # Check first entry has empty prev_hash
        if entries[0].prev_hash != "":
            return VerificationResult(
                valid=False,
                total_entries=len(entries),
                broken_index=0,
                error_message=f"First entry has non-empty prev_hash: {entries[0].prev_hash}",
            )

        # Verify each entry's hash
        for i, entry in enumerate(entries):
            expected_hash = entry.compute_hash()
            if entry.entry_hash != expected_hash:
                return VerificationResult(
                    valid=False,
                    total_entries=len(entries),
                    broken_index=i,
                    error_message=(
                        f"Entry {i} hash mismatch: "
                        f"stored={entry.entry_hash}, computed={expected_hash}"
                    ),
                )

        # Verify chain links
        for i in range(1, len(entries)):
            expected_prev_hash = entries[i - 1].entry_hash
            actual_prev_hash = entries[i].prev_hash

            if actual_prev_hash != expected_prev_hash:
                return VerificationResult(
                    valid=False,
                    total_entries=len(entries),
                    broken_index=i,
                    error_message=(
                        f"Chain break at entry {i}: "
                        f"prev_hash={actual_prev_hash} != "
                        f"expected={expected_prev_hash}"
                    ),
                )

        return VerificationResult(
            valid=True,
            total_entries=len(entries),
            error_message="All entries verified successfully",
        )

    def count(self) -> int:
        """
        Count the number of entries in the audit log.

        Returns:
            Total number of entries.
        """
        return len(self.read_all())

    def query_by_operator(self, operator_username: str) -> list[AuditEntry]:
        """
        Query entries by operator username.

        Args:
            operator_username: Username to filter by

        Returns:
            List of matching AuditEntry objects.
        """
        return [
            entry for entry in self.read_all()
            if entry.operator_username == operator_username
        ]

    def query_by_action(self, action: str) -> list[AuditEntry]:
        """
        Query entries by action type.

        Args:
            action: Action type to filter by

        Returns:
            List of matching AuditEntry objects.
        """
        return [
            entry for entry in self.read_all()
            if entry.action == action
        ]

    def query_by_case_id(self, case_id: str) -> list[AuditEntry]:
        """
        Query entries by case ID.

        Args:
            case_id: Case ID to filter by

        Returns:
            List of matching AuditEntry objects.
        """
        return [
            entry for entry in self.read_all()
            if entry.case_id == case_id
        ]
