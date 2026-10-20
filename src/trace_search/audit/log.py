"""
Hash-chained audit log implementation.

Each entry records operator actions with tamper-proof HMAC-SHA256 chaining.
Entries are append-only and stored in a location exempt from derived-data purge.
"""

import fcntl
import hashlib
import hmac
import json
import os
import zoneinfo
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# IST timezone for all timestamps
IST = zoneinfo.ZoneInfo("Asia/Kolkata")

# Audit log directory - single source of truth for retention purge exemption
AUDIT_LOG_DIR = "audit_logs"

# Key file location (gitignored, never committed)
AUDIT_KEY_FILE = f"{AUDIT_LOG_DIR}/.audit_key"

# Head anchor file (stores last HMAC and entry count for verification)
AUDIT_HEAD_FILE = f"{AUDIT_LOG_DIR}/.audit_head"


class AuditAction(Enum):
    """Valid audit log actions."""

    LOGIN = "login"
    SEARCH = "search"
    VIEW = "view"
    EXPORT = "export"
    RETENTION_CHANGE = "retention_change"
    PURGE = "purge"


def is_audit_path(path: Path | str) -> bool:
    """
    Check if a path is inside the audit log directory.

    This is the single source of truth for the T9 retention purge to decide
    what to exclude from derived-data cleanup.

    Args:
        path: Path to check (relative or absolute)

    Returns:
        True if the path is inside AUDIT_LOG_DIR, False otherwise
    """
    path = Path(path).resolve()
    audit_dir = Path(AUDIT_LOG_DIR).resolve()

    # Check if path is audit_dir itself or a child of it
    try:
        path.relative_to(audit_dir)
        return True
    except ValueError:
        return False


@dataclass
class AuditEntry:
    """
    A single audit log entry.

    Fields:
        timestamp: ISO 8601 timestamp in IST timezone (Asia/Kolkata)
        operator_username: Username of the operator performing the action
        action: Type of action (AuditAction enum value)
        query: Optional query details (for search actions)
        case_id: Case identifier (required for search, view, export)
        purpose: Purpose/reason for the action (required for search, view, export)
        result_ids: List of result identifiers returned by the action
        prev_hmac: HMAC-SHA256 of the previous entry (empty for first entry)
        entry_hmac: HMAC-SHA256 of this entry (excluding this field itself)
    """

    timestamp: str
    operator_username: str
    action: str  # Stored as string, validated against AuditAction
    query: str | None = None
    case_id: str | None = None
    purpose: str | None = None
    result_ids: list[str] = field(default_factory=list)
    prev_hmac: str = ""
    entry_hmac: str = ""

    def compute_hmac(self, key: bytes) -> str:
        """
        Compute HMAC-SHA256 of this entry using the secret key.

        The HMAC includes all fields except entry_hmac itself.
        This prevents tampering even if an attacker can edit and rehash entries.

        Args:
            key: Secret key for HMAC

        Returns:
            Hex-encoded HMAC-SHA256 digest
        """
        # Create a copy of the entry dict without entry_hmac
        entry_dict = asdict(self)
        entry_dict.pop("entry_hmac", None)

        # Canonical JSON: sorted keys, no whitespace
        canonical = json.dumps(entry_dict, sort_keys=True, separators=(",", ":"))
        return hmac.new(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()


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
    HMAC-chained audit log with tamper detection and head anchor.

    The audit log is stored in a dedicated location separate from derived data
    to ensure it survives retention purges. Entries are kept for 1 year.

    Security:
    - HMAC-SHA256 chain with secret key prevents tail tampering and rehashing attacks
    - Head anchor file stores last HMAC and entry count for verification
    - File lock on append prevents race conditions

    Storage format: JSON Lines (one JSON object per line) for append-only writes.
    Location: audit_logs/ directory (exempt from derived-data purge)
    """

    def __init__(
        self,
        log_path: Path | None = None,
        key_path: Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        """
        Initialize audit log.

        Args:
            log_path: Path to the audit log file. Defaults to audit_logs/audit.jsonl
            key_path: Path to the HMAC key file. Defaults to audit_logs/.audit_key
            clock: Callable that returns current datetime (for testing). Defaults to
                  datetime.now(IST)
        """
        if log_path is None:
            log_path = Path(AUDIT_LOG_DIR) / "audit.jsonl"

        if key_path is None:
            key_path = Path(AUDIT_KEY_FILE)

        self.log_path = Path(log_path)
        self.key_path = Path(key_path)
        self.head_path = Path(AUDIT_HEAD_FILE)
        self.clock = clock or (lambda: datetime.now(IST))

        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Touch the file if it doesn't exist
        if not self.log_path.exists():
            self.log_path.touch()

    def _load_or_generate_key(self) -> bytes:
        """
        Load the HMAC key from the key file, or generate a new one.

        The key is a 32-byte random value, stored as hex in the key file.
        The key file is gitignored and never committed.

        Returns:
            32-byte HMAC key

        Raises:
            RuntimeError: If key file doesn't exist and AUDIT_KEY env var is not set
        """
        # Try environment variable first
        env_key = os.environ.get("AUDIT_KEY")
        if env_key:
            try:
                return bytes.fromhex(env_key)
            except ValueError as e:
                raise RuntimeError(
                    "AUDIT_KEY environment variable must be a hex-encoded key"
                ) from e

        # Try key file
        if self.key_path.exists():
            key_hex = self.key_path.read_text().strip()
            return bytes.fromhex(key_hex)

        # Generate new key
        key = os.urandom(32)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.write_text(key.hex())
        self.key_path.chmod(0o600)  # Owner read/write only
        return key

    def _get_key(self) -> bytes:
        """Get the HMAC key, failing loudly if not available."""
        try:
            return self._load_or_generate_key()
        except Exception as e:
            raise RuntimeError(f"Failed to load audit key: {e}") from e

    def _read_head_anchor(self) -> tuple[str, int] | None:
        """
        Read the head anchor file.

        Returns:
            Tuple of (last_hmac, entry_count) or None if file doesn't exist
        """
        if not self.head_path.exists():
            return None

        try:
            data = json.loads(self.head_path.read_text())
            return data["last_hmac"], data["entry_count"]
        except Exception:
            return None

    def _write_head_anchor(self, last_hmac: str, entry_count: int):
        """
        Write the head anchor file.

        Args:
            last_hmac: HMAC of the last entry
            entry_count: Total number of entries
        """
        self.head_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"last_hmac": last_hmac, "entry_count": entry_count}
        self.head_path.write_text(json.dumps(data, separators=(",", ":")))

    def _get_last_entry(self) -> AuditEntry | None:
        """
        Read the last entry from the log file.

        Returns:
            The last AuditEntry, or None if the log is empty
        """
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return None

        # Read last line
        try:
            with open(self.log_path, encoding="utf-8") as f:
                lines = f.readlines()
                if not lines:
                    return None

                last_line = lines[-1].strip()
                if not last_line:
                    return None

                entry_dict = json.loads(last_line)
                return AuditEntry(**entry_dict)
        except Exception:
            return None

    def append(
        self,
        operator_username: str,
        action: str | AuditAction,
        query: str | None = None,
        case_id: str | None = None,
        purpose: str | None = None,
        result_ids: list[str] | None = None,
    ) -> AuditEntry:
        """
        Append a new entry to the audit log.

        Args:
            operator_username: Username of the operator
            action: Type of action (AuditAction enum or string)
            query: Optional query details
            case_id: Case identifier (required for search, view, export)
            purpose: Purpose/reason (required for search, view, export)
            result_ids: Optional list of result identifiers

        Returns:
            The created AuditEntry with computed HMAC and chain link

        Raises:
            ValueError: If action is invalid or case_id/purpose missing for required actions
            RuntimeError: If HMAC key is not available
        """
        # Validate and normalize action
        if isinstance(action, AuditAction):
            action_str = action.value
            action_enum = action
        else:
            try:
                action_enum = AuditAction(action)
                action_str = action
            except ValueError as e:
                valid_actions = ", ".join(a.value for a in AuditAction)
                raise ValueError(
                    f"Invalid action '{action}'. Must be one of: {valid_actions}"
                ) from e

        # Validate case_id and purpose for required actions
        requires_case_id = action_enum in {
            AuditAction.SEARCH,
            AuditAction.VIEW,
            AuditAction.EXPORT,
        }
        if requires_case_id:
            if not case_id or not case_id.strip():
                raise ValueError(
                    f"case_id is required for action '{action_str}' (PRD §8)"
                )
            if not purpose or not purpose.strip():
                raise ValueError(
                    f"purpose is required for action '{action_str}' (PRD §8)"
                )

        # Get HMAC key
        key = self._get_key()

        # Get timestamp
        now = self.clock()
        timestamp = now.isoformat()

        # Get previous entry's HMAC
        last_entry = self._get_last_entry()
        prev_hmac = last_entry.entry_hmac if last_entry else ""

        # Create entry
        entry = AuditEntry(
            timestamp=timestamp,
            operator_username=operator_username,
            action=action_str,
            query=query,
            case_id=case_id,
            purpose=purpose,
            result_ids=result_ids or [],
            prev_hmac=prev_hmac,
        )

        # Compute HMAC
        entry.entry_hmac = entry.compute_hmac(key)

        # Append to file with lock
        with open(self.log_path, "a", encoding="utf-8") as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Update head anchor - count the current entries
        current_count = len([1 for line in self.log_path.read_text().splitlines() if line.strip()])
        self._write_head_anchor(entry.entry_hmac, current_count)

        return entry

    def read_all(self) -> list[AuditEntry]:
        """
        Read all entries from the audit log.

        Returns:
            List of all AuditEntry objects in chronological order.
            Returns empty list if file doesn't exist or is empty.
        """
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return []

        entries = []
        try:
            with open(self.log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry_dict = json.loads(line)
                        entries.append(AuditEntry(**entry_dict))
        except Exception:
            # If read fails, return what we have so far
            pass

        return entries

    def verify(self) -> VerificationResult:
        """
        Verify the integrity of the audit log chain.

        This method NEVER raises exceptions. All error conditions return
        valid=False with a clear error message.

        Checks:
        1. HMAC key is available
        2. Each entry can be parsed as valid JSON
        3. Each entry has all required fields with correct types
        4. Each entry's HMAC is correctly computed with the secret key
        5. Each entry's prev_hmac matches the previous entry's entry_hmac
        6. No entries are missing (chain is unbroken)
        7. Head anchor (if present) matches the last entry

        Returns:
            VerificationResult with details of any integrity violations.
        """
        # Check if key is available
        try:
            key = self._get_key()
        except Exception as e:
            return VerificationResult(
                valid=False,
                total_entries=0,
                error_message=f"HMAC key not available: {e}",
            )

        # Read all entries with robust error handling
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            # Empty log is valid
            return VerificationResult(
                valid=True,
                total_entries=0,
                error_message="Empty log (valid)",
            )

        entries = []
        try:
            with open(self.log_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    # Try to parse JSON
                    try:
                        entry_dict = json.loads(line)
                    except json.JSONDecodeError as e:
                        return VerificationResult(
                            valid=False,
                            total_entries=len(entries),
                            broken_index=i,
                            error_message=f"Malformed JSON at line {i}: {e}",
                        )

                    # Check for required fields and correct types
                    required_fields = {
                        "timestamp": str,
                        "operator_username": str,
                        "action": str,
                        "result_ids": list,
                        "prev_hmac": str,
                        "entry_hmac": str,
                    }
                    optional_fields = {"query": str, "case_id": str, "purpose": str}

                    for field, expected_type in required_fields.items():
                        if field not in entry_dict:
                            return VerificationResult(
                                valid=False,
                                total_entries=len(entries),
                                broken_index=i,
                                error_message=f"Missing required field '{field}' at entry {i}",
                            )
                        if not isinstance(entry_dict[field], expected_type):
                            return VerificationResult(
                                valid=False,
                                total_entries=len(entries),
                                broken_index=i,
                                error_message=(
                                    f"Field '{field}' has wrong type at entry {i}: "
                                    f"expected {expected_type.__name__}, "
                                    f"got {type(entry_dict[field]).__name__}"
                                ),
                            )

                    # Check for extra fields (potential tampering)
                    all_valid_fields = set(required_fields.keys()) | set(
                        optional_fields.keys()
                    )
                    extra_fields = set(entry_dict.keys()) - all_valid_fields
                    if extra_fields:
                        return VerificationResult(
                            valid=False,
                            total_entries=len(entries),
                            broken_index=i,
                            error_message=f"Extra fields at entry {i}: {extra_fields}",
                        )

                    # Try to create AuditEntry
                    try:
                        entry = AuditEntry(**entry_dict)
                    except Exception as e:
                        return VerificationResult(
                            valid=False,
                            total_entries=len(entries),
                            broken_index=i,
                            error_message=f"Invalid entry at {i}: {e}",
                        )

                    entries.append(entry)

        except Exception as e:
            return VerificationResult(
                valid=False,
                total_entries=len(entries),
                error_message=f"Error reading log file: {e}",
            )

        if not entries:
            return VerificationResult(
                valid=True,
                total_entries=0,
                error_message="Empty log (valid)",
            )

        # Check first entry has empty prev_hmac
        if entries[0].prev_hmac != "":
            return VerificationResult(
                valid=False,
                total_entries=len(entries),
                broken_index=0,
                error_message=f"First entry has non-empty prev_hmac: {entries[0].prev_hmac}",
            )

        # Verify each entry's HMAC
        for i, entry in enumerate(entries):
            expected_hmac = entry.compute_hmac(key)
            if entry.entry_hmac != expected_hmac:
                return VerificationResult(
                    valid=False,
                    total_entries=len(entries),
                    broken_index=i,
                    error_message=(
                        f"Entry {i} HMAC mismatch: "
                        f"stored={entry.entry_hmac[:16]}..., "
                        f"computed={expected_hmac[:16]}..."
                    ),
                )

        # Verify chain links
        for i in range(1, len(entries)):
            expected_prev_hmac = entries[i - 1].entry_hmac
            actual_prev_hmac = entries[i].prev_hmac

            if actual_prev_hmac != expected_prev_hmac:
                return VerificationResult(
                    valid=False,
                    total_entries=len(entries),
                    broken_index=i,
                    error_message=(
                        f"Chain break at entry {i}: "
                        f"prev_hmac={actual_prev_hmac[:16]}... != "
                        f"expected={expected_prev_hmac[:16]}..."
                    ),
                )

        # Verify head anchor if present
        head_anchor = self._read_head_anchor()
        if head_anchor:
            last_hmac, entry_count = head_anchor
            if last_hmac != entries[-1].entry_hmac:
                return VerificationResult(
                    valid=False,
                    total_entries=len(entries),
                    broken_index=len(entries) - 1,
                    error_message=(
                        f"Head anchor HMAC mismatch: "
                        f"anchor={last_hmac[:16]}..., "
                        f"last_entry={entries[-1].entry_hmac[:16]}..."
                    ),
                )
            if entry_count != len(entries):
                return VerificationResult(
                    valid=False,
                    total_entries=len(entries),
                    error_message=(
                        f"Head anchor count mismatch: "
                        f"anchor={entry_count}, actual={len(entries)}"
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
            entry
            for entry in self.read_all()
            if entry.operator_username == operator_username
        ]

    def query_by_action(self, action: str | AuditAction) -> list[AuditEntry]:
        """
        Query entries by action type.

        Args:
            action: Action type to filter by (string or AuditAction enum)

        Returns:
            List of matching AuditEntry objects.
        """
        action_str = action.value if isinstance(action, AuditAction) else action
        return [entry for entry in self.read_all() if entry.action == action_str]

    def query_by_case_id(self, case_id: str) -> list[AuditEntry]:
        """
        Query entries by case ID.

        Args:
            case_id: Case ID to filter by

        Returns:
            List of matching AuditEntry objects.
        """
        return [entry for entry in self.read_all() if entry.case_id == case_id]
