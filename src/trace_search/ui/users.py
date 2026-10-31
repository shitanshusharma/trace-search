"""Local operator accounts.

The SQLite file must sit outside this repository. Passwords are stored only
as scrypt hashes (D-T7-4: n=2**14, r=8, p=1, 16-byte salt) and compared with
``hmac.compare_digest``. There is no default account.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar, cast

from trace_search.ui.data_path import outside_repo, repo_root_from

_ROLES = frozenset({"operator", "admin"})
# D-T7-4. New hashes use this row.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 64
_SALT_BYTES = 16
_HASH_PARTS = 7
# Verification accepts only the D-T7-4 row stored in the hash.
_ALLOWED_KDF = frozenset({(_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _SCRYPT_DKLEN)})
_KDF_DIGIT_CAP = 8
_DUMMY_SALT = bytes(range(_SALT_BYTES))
_DUMMY_DIGEST = bytes(_SCRYPT_DKLEN)

_CREATE_SQL = (
    "CREATE TABLE IF NOT EXISTS users ("
    "username TEXT PRIMARY KEY NOT NULL, "
    "role TEXT NOT NULL CHECK (role IN ('operator', 'admin')), "
    "password_hash TEXT NOT NULL, "
    "created_at TEXT NOT NULL, "
    "disabled INTEGER NOT NULL)"
)
_INSERT_SQL = (
    "INSERT INTO users (username, role, password_hash, created_at, disabled) VALUES (?, ?, ?, ?, 0)"
)
_SELECT_SQL = "SELECT username, role, password_hash, disabled FROM users WHERE username = ?"
_DISABLE_SQL = "UPDATE users SET disabled = 1 WHERE username = ?"
_ROLE_SQL = "UPDATE users SET role = ? WHERE username = ?"

_T = TypeVar("_T")


def _checked_db_path(db_path: Path) -> Path:
    """Resolve ``db_path`` and refuse one that sits inside the checkout."""
    root = repo_root_from(Path(__file__))
    if root is None:
        raise ValueError("cannot locate the repository root; refusing to open the user db")
    try:
        checked = outside_repo(db_path, root)
    except ValueError as exc:
        message = str(exc).replace(
            "data dir is inside the repository",
            "db path is inside the repository",
            1,
        )
        raise ValueError(message) from exc
    return checked


def _require_username(username: str) -> str:
    if username != username.strip():
        raise ValueError("username must be a non-empty string without surrounding space")
    if username == "":
        raise ValueError("username must be a non-empty string without surrounding space")
    return username


def _require_role(role: str) -> str:
    if role not in _ROLES:
        raise ValueError(f"role must be operator or admin, got {role!r}")
    return role


def _require_password(password: str) -> str:
    if not password:
        raise ValueError("password must be a non-empty string")
    return password


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _password_bytes(password: str) -> bytes:
    return password.encode("utf-8")


def _scrypt(
    password: str,
    salt: bytes,
    n: int = _SCRYPT_N,
    r: int = _SCRYPT_R,
    p: int = _SCRYPT_P,
    dklen: int = _SCRYPT_DKLEN,
) -> bytes:
    return hashlib.scrypt(
        _password_bytes(password),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=dklen,
    )


def _burn_password(password: str) -> None:
    """Run one scrypt so a missing or disabled account is not faster."""
    derived = _scrypt(password, _DUMMY_SALT)
    hmac.compare_digest(derived, _DUMMY_DIGEST)


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = _scrypt(password, salt)
    fields = (
        "scrypt",
        str(_SCRYPT_N),
        str(_SCRYPT_R),
        str(_SCRYPT_P),
        str(_SCRYPT_DKLEN),
        salt.hex(),
        derived.hex(),
    )
    return "$".join(fields)


def _kdf_int(text: str) -> int:
    if len(text) > _KDF_DIGIT_CAP or not text.isascii() or not text.isdigit():
        raise ValueError(text)
    if text != str(int(text)):
        raise ValueError(text)
    return int(text)


def _parsed_hash(stored: str) -> tuple[int, int, int, int, bytes, bytes] | None:
    pieces = stored.split("$")
    if len(pieces) != _HASH_PARTS:
        return None
    if pieces[0] != "scrypt":
        return None
    try:
        n = _kdf_int(pieces[1])
        r = _kdf_int(pieces[2])
        p = _kdf_int(pieces[3])
        dklen = _kdf_int(pieces[4])
        salt = bytes.fromhex(pieces[5])
        expected = bytes.fromhex(pieces[6])
    except ValueError:
        return None
    if (n, r, p, dklen) not in _ALLOWED_KDF:
        return None
    if len(salt) != _SALT_BYTES:
        return None
    if len(expected) != dklen:
        return None
    return n, r, p, dklen, salt, expected


def _password_matches(password: str, stored: str) -> bool:
    parsed = _parsed_hash(stored)
    if parsed is None:
        return False
    n, r, p, dklen, salt, expected = parsed
    derived = _scrypt(password, salt, n, r, p, dklen)
    return hmac.compare_digest(derived, expected)


def _create_private_file(path: Path) -> None:
    """Create a new db file as mode 0600. An existing file is left alone."""
    if path.exists():
        return
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    except FileExistsError:
        return
    os.close(fd)
    os.chmod(path, 0o600)


class UserStore:
    """SQLite accounts with ``operator`` and ``admin`` roles.

    ``username`` is the only name column. Disabled users never authenticate.
    """

    def __init__(self, db_path: Path) -> None:
        checked = _checked_db_path(Path(db_path))
        checked.parent.mkdir(parents=True, exist_ok=True)
        self.db_path: Path = checked
        _create_private_file(checked)
        self._ensure_schema()

    def add_user(self, username: str, role: str, password: str) -> None:
        """Insert one account. The password is hashed and not stored."""
        checked_username = _require_username(username)
        checked_role = _require_role(role)
        checked_password = _require_password(password)
        if self._find(checked_username) is not None:
            raise ValueError(f"username already exists: {checked_username}")
        created_at = _now()
        digest = _hash_password(checked_password)

        def _insert(connection: sqlite3.Connection) -> None:
            connection.execute(
                _INSERT_SQL,
                (checked_username, checked_role, digest, created_at),
            )

        self._using(_insert)

    def disable_user(self, username: str) -> None:
        """Mark an account disabled. A second call leaves it disabled."""
        checked = _require_username(username)
        if self._find(checked) is None:
            raise ValueError(f"unknown user: {checked}")

        def _update(connection: sqlite3.Connection) -> None:
            connection.execute(_DISABLE_SQL, (checked,))

        self._using(_update)

    def set_role(self, username: str, role: str) -> None:
        """Change ``operator`` or ``admin`` for an existing account."""
        checked = _require_username(username)
        checked_role = _require_role(role)
        if self._find(checked) is None:
            raise ValueError(f"unknown user: {checked}")

        def _update(connection: sqlite3.Connection) -> None:
            connection.execute(_ROLE_SQL, (checked_role, checked))

        self._using(_update)

    def authenticate(self, username: str, password: str) -> bool:
        """Return whether this enabled account matches ``password``."""
        row = self._find(username)
        if row is None:
            _burn_password(password)
            return False
        if int(row["disabled"]) != 0:
            _burn_password(password)
            return False
        return _password_matches(password, cast(str, row["password_hash"]))

    def enabled_role(self, username: str) -> str | None:
        """Return ``operator`` or ``admin`` for an enabled account."""
        row = self._find(username)
        if row is None:
            return None
        if int(row["disabled"]) != 0:
            return None
        return cast(str, row["role"])

    def _find(self, username: str) -> sqlite3.Row | None:
        def _query(connection: sqlite3.Connection) -> sqlite3.Row | None:
            cursor = connection.execute(_SELECT_SQL, (username,))
            row = cursor.fetchone()
            if row is None:
                return None
            return cast(sqlite3.Row, row)

        return self._using(_query)

    def _ensure_schema(self) -> None:
        def _create(connection: sqlite3.Connection) -> None:
            connection.execute(_CREATE_SQL)

        self._using(_create)

    def _using(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        connection = sqlite3.connect(self.db_path)
        try:
            connection.row_factory = sqlite3.Row
            with connection:
                return fn(connection)
        finally:
            connection.close()
