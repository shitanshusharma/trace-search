"""
Append-only audit log with an HMAC-SHA256 chain and a signed head anchor.

The key comes from TRACE_AUDIT_HMAC_KEY or from an explicit key file outside
audit_logs/. Nothing in this module generates a key except init_hmac_key().
"""

import fcntl
import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# IST for every stored timestamp (Asia/Kolkata, UTC+05:30).
IST = ZoneInfo("Asia/Kolkata")

# Single source of truth for the directory the retention purge must skip.
AUDIT_LOG_DIR = "audit_logs"

# Env var that holds the hex-encoded HMAC key. There is no default key.
TRACE_AUDIT_HMAC_KEY_ENV = "TRACE_AUDIT_HMAC_KEY"

_KEY_BYTES = 32

_CASE_ACTIONS = frozenset({"search", "view", "export"})

_ENTRY_FIELDS = (
    "timestamp",
    "operator_username",
    "action",
    "query",
    "case_id",
    "purpose",
    "details",
    "result_ids",
    "prev_hmac",
    "entry_hmac",
)

_REQUIRED_ENTRY_TYPES = {
    "timestamp": str,
    "operator_username": str,
    "action": str,
    "result_ids": list,
    "prev_hmac": str,
    "entry_hmac": str,
}

_OPTIONAL_ENTRY_TYPES = {
    "query": str,
    "case_id": str,
    "purpose": str,
    "details": str,
}


class AuditAction(Enum):
    """Operator actions that may be written to the audit log."""

    LOGIN = "login"
    SEARCH = "search"
    VIEW = "view"
    EXPORT = "export"
    RETENTION_CHANGE = "retention_change"
    PURGE = "purge"


def is_audit_path(path: Path | str) -> bool:
    """
    Return True when `path` resolves inside the configured audit directory.

    The configured directory is `Path(AUDIT_LOG_DIR).resolve()` (relative to
    the process working directory). A folder that is merely named
    `audit_logs` somewhere else is not exempt. T9 purge must use this helper
    instead of matching the directory name.
    """
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(Path(AUDIT_LOG_DIR).resolve())
    except (OSError, ValueError):
        return False
    return True


def default_log_path() -> Path:
    """Default audit log path. Built only from AUDIT_LOG_DIR."""
    return Path(AUDIT_LOG_DIR) / "audit.jsonl"


def head_anchor_path(log_path: Path | str) -> Path:
    """Anchor path derived from the log path: `<log_path>.head`."""
    return Path(f"{log_path}.head")


def _lock_path(log_path: Path) -> Path:
    return Path(f"{log_path}.auditlock")


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hmac_hex(key: bytes, payload: dict[str, Any]) -> str:
    digest = hmac.new(key, _canonical(payload).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def _mac_equal(left: object, right: object) -> bool:
    """Constant-time compare that does not raise on length or type mismatch."""
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if len(left) != len(right):
        return False
    return hmac.compare_digest(left, right)


def _decode_key(key_hex: str, source: str) -> bytes:
    try:
        key = bytes.fromhex(key_hex.strip())
    except ValueError as exc:
        raise RuntimeError(f"{source} must be a hex-encoded HMAC key") from exc
    if len(key) != _KEY_BYTES:
        raise RuntimeError(
            f"{source} must be exactly {_KEY_BYTES} bytes ({_KEY_BYTES * 2} hex characters)"
        )
    return key


def init_hmac_key(key_path: Path | str) -> Path:
    """
    Create a new HMAC key file. This is the only key-generation path.

    Refuses to write a key inside AUDIT_LOG_DIR. Does not overwrite an
    existing file. verify() and append() never call this.
    """
    path = Path(key_path)
    if is_audit_path(path):
        raise RuntimeError(f"Refusing to create an HMAC key inside {AUDIT_LOG_DIR}: {path}")
    if path.exists():
        raise FileExistsError(f"HMAC key file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(_KEY_BYTES)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, (key.hex() + "\n").encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


@dataclass
class AuditEntry:
    """One audit log line. `entry_hmac` covers every other field."""

    timestamp: str
    operator_username: str
    action: str
    query: str | None = None
    case_id: str | None = None
    purpose: str | None = None
    details: str | None = None
    result_ids: list[str] = field(default_factory=list)
    prev_hmac: str = ""
    entry_hmac: str = ""

    def hmac_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("entry_hmac", None)
        return payload

    def compute_hmac(self, key: bytes) -> str:
        """HMAC-SHA256 over canonical JSON of every field except entry_hmac."""
        return _hmac_hex(key, self.hmac_payload())


@dataclass
class VerificationResult:
    """Outcome of verify(). verify() returns this and does not raise."""

    valid: bool
    total_entries: int
    broken_index: int | None = None
    error_message: str | None = None


@dataclass
class _Anchor:
    last_hmac: str
    entry_count: int


def _require_nonempty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required and must be a non-empty string")
    return value


def _anchor_error(message: str) -> VerificationResult:
    return VerificationResult(valid=False, total_entries=0, error_message=message)


def _missing_anchor_field(data: dict[str, Any]) -> VerificationResult | None:
    for field_name in ("last_hmac", "entry_count"):
        if field_name not in data:
            return _anchor_error(f"Head anchor is missing {field_name}")
    return None


def _anchor_field_types(data: dict[str, Any]) -> VerificationResult | None:
    if not isinstance(data["last_hmac"], str) or isinstance(data["entry_count"], bool):
        return _anchor_error("Head anchor field has the wrong type")
    if not isinstance(data["entry_count"], int):
        return _anchor_error("Head anchor field has the wrong type")
    return None


def _signed_anchor(data: dict[str, Any], key: bytes) -> _Anchor | VerificationResult:
    expected = _hmac_hex(
        key,
        {"entry_count": data["entry_count"], "last_hmac": data["last_hmac"]},
    )
    if not _mac_equal(data["anchor_hmac"], expected):
        return _anchor_error("Head anchor signature mismatch (unsigned or wrong key)")
    last_hmac = data["last_hmac"]
    entry_count = data["entry_count"]
    if not isinstance(last_hmac, str) or not isinstance(entry_count, int):
        return _anchor_error("Head anchor field has the wrong type")
    return _Anchor(last_hmac=last_hmac, entry_count=entry_count)


def _parse_anchor_object(data: object, key: bytes) -> _Anchor | VerificationResult:
    if not isinstance(data, dict):
        return _anchor_error("Head anchor is malformed: expected a JSON object")
    if "anchor_hmac" not in data or not isinstance(data.get("anchor_hmac"), str):
        return _anchor_error("Head anchor is unsigned (missing anchor_hmac)")
    if data["anchor_hmac"] == "":
        return _anchor_error("Head anchor is unsigned (missing anchor_hmac)")
    missing = _missing_anchor_field(data)
    if missing is not None:
        return missing
    typed = _anchor_field_types(data)
    if typed is not None:
        return typed
    extra = set(data) - {"anchor_hmac", "entry_count", "last_hmac"}
    if extra:
        return _anchor_error(f"Head anchor has extra fields: {sorted(extra)}")
    return _signed_anchor(data, key)


def _invalid_entry(index: int, message: str) -> VerificationResult:
    return VerificationResult(
        valid=False,
        total_entries=index,
        broken_index=index,
        error_message=message,
    )


def _check_entry_types(data: dict[str, Any], index: int) -> VerificationResult | None:
    for field_name, expected in _REQUIRED_ENTRY_TYPES.items():
        if field_name not in data:
            return _invalid_entry(index, f"Missing required field '{field_name}' at entry {index}")
        if not isinstance(data[field_name], expected):
            return _invalid_entry(
                index,
                (
                    f"Field '{field_name}' has the wrong type at entry {index}: "
                    f"expected {expected.__name__}, "
                    f"got {type(data[field_name]).__name__}"
                ),
            )
    for field_name, expected in _OPTIONAL_ENTRY_TYPES.items():
        if field_name not in data or data[field_name] is None:
            continue
        if not isinstance(data[field_name], expected):
            return _invalid_entry(
                index,
                (
                    f"Field '{field_name}' has the wrong type at entry {index}: "
                    f"expected {expected.__name__}, "
                    f"got {type(data[field_name]).__name__}"
                ),
            )
    return None


def _check_entry_shape(data: dict[str, Any], index: int) -> VerificationResult | None:
    if isinstance(data.get("details"), str) and not data["details"].strip():
        return _invalid_entry(index, f"details is blank at entry {index}")
    extra = set(data) - set(_ENTRY_FIELDS)
    if extra:
        return _invalid_entry(index, f"Extra fields at entry {index}: {sorted(extra)}")
    try:
        AuditAction(data["action"])
    except ValueError:
        return _invalid_entry(index, f"Unknown action '{data['action']}' at entry {index}")
    if not isinstance(data["operator_username"], str) or not data["operator_username"].strip():
        return _invalid_entry(index, f"operator_username is empty at entry {index}")
    return None


def _check_case_fields(data: dict[str, Any], index: int) -> VerificationResult | None:
    try:
        action_enum = AuditAction(data["action"])
    except ValueError:
        return None
    if action_enum.value not in _CASE_ACTIONS:
        return None
    case_id = data.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        return _invalid_entry(
            index,
            f"case_id is required for action '{action_enum.value}' at entry {index}",
        )
    purpose = data.get("purpose")
    if not isinstance(purpose, str) or not purpose.strip():
        return _invalid_entry(
            index,
            f"purpose is required for action '{action_enum.value}' at entry {index}",
        )
    return None


def _chain_failure(total: int, index: int, message: str) -> VerificationResult:
    return VerificationResult(
        valid=False,
        total_entries=total,
        broken_index=index,
        error_message=message,
    )


def _check_entry_chain(
    entries: list[AuditEntry],
    key: bytes,
) -> VerificationResult | None:
    if not entries:
        return None
    total = len(entries)
    if entries[0].prev_hmac != "":
        return _chain_failure(total, 0, "First entry has a non-empty prev_hmac")
    for index, entry in enumerate(entries):
        expected = entry.compute_hmac(key)
        if not _mac_equal(entry.entry_hmac, expected):
            return _chain_failure(
                total,
                index,
                (
                    f"Entry {index} HMAC mismatch: "
                    f"stored={entry.entry_hmac[:16]}..., "
                    f"computed={expected[:16]}..."
                ),
            )
    for index in range(1, total):
        expected_prev = entries[index - 1].entry_hmac
        actual_prev = entries[index].prev_hmac
        if not _mac_equal(actual_prev, expected_prev):
            return _chain_failure(
                total,
                index,
                (
                    f"Chain break at entry {index}: "
                    f"prev_hmac={actual_prev[:16]}... != "
                    f"expected={expected_prev[:16]}..."
                ),
            )
    return None


def _check_empty_anchor(anchor: _Anchor) -> VerificationResult:
    if anchor.entry_count > 0:
        return VerificationResult(
            valid=False,
            total_entries=0,
            broken_index=0,
            error_message=(
                f"Audit log is empty or truncated but head anchor count is {anchor.entry_count}"
            ),
        )
    if anchor.entry_count != 0 or anchor.last_hmac != "":
        return VerificationResult(
            valid=False,
            total_entries=0,
            broken_index=0,
            error_message=(
                f"Head anchor does not describe an empty log (count={anchor.entry_count})"
            ),
        )
    return VerificationResult(
        valid=True,
        total_entries=0,
        error_message="Empty log with a signed count of 0",
    )


def _check_anchor_against_log(
    entries: list[AuditEntry],
    anchor: _Anchor | VerificationResult,
) -> VerificationResult:
    if isinstance(anchor, VerificationResult):
        anchor.total_entries = len(entries)
        if entries:
            anchor.broken_index = len(entries) - 1
        return anchor
    if not entries:
        return _check_empty_anchor(anchor)
    total = len(entries)
    if anchor.entry_count != total:
        return _chain_failure(
            total,
            total - 1,
            f"Head anchor count mismatch: anchor={anchor.entry_count}, actual={total}",
        )
    if not _mac_equal(anchor.last_hmac, entries[-1].entry_hmac):
        return _chain_failure(
            total,
            total - 1,
            (
                "Head anchor HMAC mismatch: "
                f"anchor={anchor.last_hmac[:16]}..., "
                f"last_entry={entries[-1].entry_hmac[:16]}..."
            ),
        )
    return VerificationResult(
        valid=True,
        total_entries=total,
        error_message="All entries verified successfully",
    )


def _check_result_ids(data: dict[str, Any], index: int) -> VerificationResult | None:
    result_ids = data.get("result_ids")
    if not isinstance(result_ids, list) or not all(isinstance(item, str) for item in result_ids):
        return _invalid_entry(
            index,
            (f"Field 'result_ids' has the wrong type at entry {index}: expected a list of strings"),
        )
    return None


def _normalize_action(action: str | AuditAction) -> AuditAction:
    if isinstance(action, AuditAction):
        return action
    if not isinstance(action, str):
        raise ValueError(
            "Invalid action type. Must be an AuditAction or one of: "
            + ", ".join(item.value for item in AuditAction)
        )
    try:
        return AuditAction(action)
    except ValueError as exc:
        valid = ", ".join(item.value for item in AuditAction)
        raise ValueError(f"Invalid action '{action}'. Must be one of: {valid}") from exc


class AuditLog:
    """HMAC-chained audit log with a mandatory signed head anchor."""

    def __init__(
        self,
        log_path: Path | str | None = None,
        key_path: Path | str | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        """
        Open an audit log. Does not create a key, an anchor, or the log file.

        Args:
            log_path: Log file. Defaults to `<AUDIT_LOG_DIR>/audit.jsonl`.
            key_path: Explicit key file outside AUDIT_LOG_DIR. When omitted,
                the key is read from TRACE_AUDIT_HMAC_KEY.
            clock: Returns a timezone-aware datetime. Defaults to now in IST.
        """
        self.log_path = Path(log_path) if log_path is not None else default_log_path()
        self.key_path = Path(key_path) if key_path is not None else None
        self.head_path = head_anchor_path(self.log_path)
        self.lock_path = _lock_path(self.log_path)
        self.clock = clock or (lambda: datetime.now(IST))

    def _timestamp(self) -> str:
        now = self.clock()
        if not isinstance(now, datetime):
            raise ValueError("clock must return a datetime")
        if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now.astimezone(IST).isoformat()

    def _get_key(self) -> bytes:
        """
        Load the HMAC key. Never creates one.

        An explicit key_path wins over TRACE_AUDIT_HMAC_KEY. The key file
        must not sit inside AUDIT_LOG_DIR.
        """
        if self.key_path is not None:
            if is_audit_path(self.key_path):
                raise RuntimeError(
                    f"HMAC key file must be outside {AUDIT_LOG_DIR}. Got: {self.key_path}"
                )
            if not self.key_path.is_file():
                raise RuntimeError(
                    f"HMAC key file not found: {self.key_path}. Create one with init_hmac_key()."
                )
            return _decode_key(
                self.key_path.read_text(encoding="utf-8"),
                f"Key file {self.key_path}",
            )

        if TRACE_AUDIT_HMAC_KEY_ENV not in os.environ:
            raise RuntimeError(
                "No HMAC key configured. Set "
                f"{TRACE_AUDIT_HMAC_KEY_ENV} or pass key_path to AuditLog(). "
                "verify() does not create a key."
            )
        raw = os.environ[TRACE_AUDIT_HMAC_KEY_ENV].strip()
        if not raw:
            raise RuntimeError(f"{TRACE_AUDIT_HMAC_KEY_ENV} is empty")
        return _decode_key(raw, TRACE_AUDIT_HMAC_KEY_ENV)

    def _read_anchor(self, key: bytes) -> _Anchor | VerificationResult:
        if not self.head_path.is_file():
            return _anchor_error("Missing head anchor")
        try:
            raw = self.head_path.read_bytes().decode("utf-8")
            data = json.loads(raw)
        except UnicodeDecodeError as exc:
            return _anchor_error(f"Head anchor is not valid UTF-8: {exc}")
        except (OSError, json.JSONDecodeError) as exc:
            return _anchor_error(f"Head anchor is malformed: {exc}")
        return _parse_anchor_object(data, key)

    def _write_anchor(self, key: bytes, last_hmac: str, entry_count: int) -> None:
        """Atomically write `<log>.head` (temp file, fsync, os.replace)."""
        body = {
            "entry_count": entry_count,
            "last_hmac": last_hmac,
        }
        signed = {
            "anchor_hmac": _hmac_hex(key, body),
            "entry_count": entry_count,
            "last_hmac": last_hmac,
        }
        raw = _canonical(signed)
        parent = self.head_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{self.head_path.name}.",
            suffix=".tmp",
            dir=parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.head_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def _log_lines(self) -> list[str]:
        if not self.log_path.exists():
            return []
        if self.log_path.stat().st_size == 0:
            return []
        return self.log_path.read_text(encoding="utf-8").splitlines()

    def _last_entry_and_count(self) -> tuple[AuditEntry | None, int]:
        """
        Return the last entry and the non-empty line count.

        A malformed last line raises RuntimeError so append cannot fork
        the chain by treating the log as empty.
        """
        lines = self._log_lines()
        nonempty = [line.strip() for line in lines if line.strip()]
        if not nonempty:
            return None, 0
        last = nonempty[-1]
        try:
            data = json.loads(last)
            entry = AuditEntry(**data)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Cannot append: last line in the log is malformed: {exc}") from exc
        return entry, len(nonempty)

    @contextmanager
    def _locked(self, exclusive: bool) -> Iterator[None]:
        """Hold the log lock. Exclusive for writers, shared for verify()."""
        if exclusive:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.lock_path.parent.is_dir():
            yield
            return
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(fd, flag)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _require_anchor_matches(
        self,
        key: bytes,
        last_entry: AuditEntry | None,
        count: int,
    ) -> None:
        """Raise when the signed anchor does not describe this tail."""
        anchor = self._read_anchor(key)
        if isinstance(anchor, VerificationResult):
            message = anchor.error_message or "invalid head anchor"
            if count == 0 and message == "Missing head anchor":
                raise RuntimeError(
                    "Audit log is not initialized. Call initialize() to "
                    "write a signed count-0 anchor before appending."
                )
            if count > 0 and message == "Missing head anchor":
                raise RuntimeError(
                    "Missing head anchor for a non-empty audit log. Refusing to append."
                )
            raise RuntimeError(f"Refusing to append: {message}")
        expected = last_entry.entry_hmac if last_entry is not None else ""
        if anchor.entry_count != count or not _mac_equal(anchor.last_hmac, expected):
            raise RuntimeError(
                "Head anchor does not match the log tail "
                f"(anchor count={anchor.entry_count}, log count={count}). "
                "Refusing to append."
            )

    def initialize(self) -> None:
        """
        Write a signed count-0 anchor for a new log.

        append() refuses to create that anchor itself, so deleting the log
        and the anchor cannot be laundered by the next append. A log that
        already has entries, or an anchor that is not an empty tip, raises.
        """
        key = self._get_key()
        with self._locked(exclusive=True):
            _last_entry, count = self._last_entry_and_count()
            if count != 0:
                raise RuntimeError("Cannot initialize a log that already has entries.")
            if self.head_path.is_file():
                anchor = self._read_anchor(key)
                if (
                    isinstance(anchor, _Anchor)
                    and anchor.entry_count == 0
                    and anchor.last_hmac == ""
                ):
                    return
                raise RuntimeError("Refusing to initialize: a head anchor already exists.")
            if not self.log_path.exists():
                self.log_path.touch()
            self._write_anchor(key, "", 0)

    def append(
        self,
        operator_username: str,
        action: str | AuditAction,
        query: str | None = None,
        case_id: str | None = None,
        purpose: str | None = None,
        details: str | None = None,
        result_ids: list[str] | None = None,
    ) -> AuditEntry:
        """
        Append one entry and update the signed head anchor.

        Holds one exclusive fcntl lock across the anchor check, the read of
        the last entry, the append, and the anchor write. The timestamp is
        taken inside that lock. A new log must be initialize()'d first.
        append() raises when the signed anchor does not match the tail.
        """
        operator_username = _require_nonempty_str(operator_username, "operator_username")
        action_enum = _normalize_action(action)
        if query is not None and not isinstance(query, str):
            raise ValueError("query must be a string when provided")
        if details is not None and (not isinstance(details, str) or not details.strip()):
            raise ValueError("details must be a non-empty string when provided")
        if action_enum.value in _CASE_ACTIONS:
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"case_id is required for action '{action_enum.value}'")
            if not isinstance(purpose, str) or not purpose.strip():
                raise ValueError(f"purpose is required for action '{action_enum.value}'")
        else:
            if case_id is not None and (not isinstance(case_id, str) or not case_id.strip()):
                raise ValueError("case_id must be a non-empty string")
            if purpose is not None and (not isinstance(purpose, str) or not purpose.strip()):
                raise ValueError("purpose must be a non-empty string")
        if result_ids is None:
            result_ids = []
        if not isinstance(result_ids, list) or not all(
            isinstance(item, str) for item in result_ids
        ):
            raise ValueError("result_ids must be a list of strings")

        key = self._get_key()

        with self._locked(exclusive=True):
            last_entry, count = self._last_entry_and_count()
            self._require_anchor_matches(key, last_entry, count)
            timestamp = self._timestamp()
            prev_hmac = last_entry.entry_hmac if last_entry else ""
            entry = AuditEntry(
                timestamp=timestamp,
                operator_username=operator_username,
                action=action_enum.value,
                query=query,
                case_id=case_id,
                purpose=purpose,
                details=details,
                result_ids=list(result_ids),
                prev_hmac=prev_hmac,
            )
            entry.entry_hmac = entry.compute_hmac(key)
            line = _canonical(asdict(entry)) + "\n"
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self._write_anchor(key, entry.entry_hmac, count + 1)
        return entry

    def read_all(self) -> list[AuditEntry]:
        """Read every entry. Raises RuntimeError if a line is malformed."""
        entries: list[AuditEntry] = []
        for line in self._log_lines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                entries.append(AuditEntry(**data))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Malformed audit entry at index {len(entries)}: {exc}") from exc
        return entries

    def count(self) -> int:
        """Count non-empty lines. Does not parse them."""
        return sum(1 for line in self._log_lines() if line.strip())

    def _parse_entry(self, raw: str, index: int) -> AuditEntry | VerificationResult:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return _invalid_entry(index, f"Malformed JSON at entry {index}: {exc}")
        if not isinstance(data, dict):
            got = type(data).__name__
            return _invalid_entry(
                index,
                f"Entry {index} has the wrong type: expected object, got {got}",
            )
        for check in (
            _check_entry_types,
            _check_entry_shape,
            _check_case_fields,
            _check_result_ids,
        ):
            failed = check(data, index)
            if failed is not None:
                return failed
        try:
            return AuditEntry(**data)
        except (TypeError, ValueError) as exc:
            return _invalid_entry(index, f"Invalid entry at {index}: {exc}")

    def _load_entries(self) -> list[AuditEntry] | VerificationResult:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return []
        try:
            raw = self.log_path.read_bytes()
        except OSError as exc:
            return VerificationResult(
                valid=False,
                total_entries=0,
                error_message=f"Cannot read audit log: {exc}",
            )
        entries: list[AuditEntry] = []
        for raw_line in raw.splitlines():
            if not raw_line.strip():
                continue
            try:
                text = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                return VerificationResult(
                    valid=False,
                    total_entries=len(entries),
                    broken_index=len(entries),
                    error_message=(f"Log entry {len(entries)} is not valid UTF-8: {exc}"),
                )
            parsed = self._parse_entry(text, len(entries))
            if isinstance(parsed, VerificationResult):
                return parsed
            entries.append(parsed)
        return entries

    def verify(self) -> VerificationResult:
        """
        Check the chain and the signed head anchor.

        Never raises and never creates a key or an anchor. Takes a shared
        flock on the same lock file as append() so an in-flight append cannot
        look like tampering. A missing key, a missing anchor, an unsigned
        anchor, a bad signature, non-UTF-8 bytes, or a count mismatch returns
        valid=False. An empty log with no anchor fails. An empty or truncated
        log whose anchor count is above 0 fails.
        """
        try:
            key = self._get_key()
        except Exception as exc:
            return VerificationResult(
                valid=False,
                total_entries=0,
                error_message=f"HMAC key not available: {exc}",
            )

        try:
            with self._locked(exclusive=False):
                return self._verify_locked(key)
        except Exception as exc:
            return VerificationResult(
                valid=False,
                total_entries=0,
                error_message=f"Cannot verify audit log: {exc}",
            )

    def _verify_locked(self, key: bytes) -> VerificationResult:
        loaded = self._load_entries()
        if isinstance(loaded, VerificationResult):
            return loaded
        chain = _check_entry_chain(loaded, key)
        if chain is not None:
            return chain
        return _check_anchor_against_log(loaded, self._read_anchor(key))

    def query_by_operator(self, operator_username: str) -> list[AuditEntry]:
        return [entry for entry in self.read_all() if entry.operator_username == operator_username]

    def query_by_action(self, action: str | AuditAction) -> list[AuditEntry]:
        action_str = action.value if isinstance(action, AuditAction) else action
        return [entry for entry in self.read_all() if entry.action == action_str]

    def query_by_case_id(self, case_id: str) -> list[AuditEntry]:
        return [entry for entry in self.read_all() if entry.case_id == case_id]
