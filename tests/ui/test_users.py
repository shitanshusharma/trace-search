"""Operator accounts: hashes, roles, and a database outside the repo."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.manage_users import main
from trace_search.ui.users import UserStore

_REPO = Path(__file__).resolve().parents[2]
_COLUMNS = ("username", "role", "password_hash", "created_at", "disabled")
_PLANTED = ("full_name", "display_name", "identity" + "_id")
_PASSWORD_LITERAL = re.compile(r"password\s*=\s*['\"]", re.IGNORECASE)
_SECRET = "correct-horse"
_OTHER = "other-horse"
_PROMPTED = "prompted-horse"
_UNICODE = "caf\u00e9-horse"


def _store(tmp_path: Path) -> UserStore:
    return UserStore(tmp_path / "users.sqlite")


def _scalar(store: UserStore, sql: str, params: tuple[object, ...] = ()) -> object:
    with sqlite3.connect(store.db_path) as connection:
        row = connection.execute(sql, params).fetchone()
    assert row is not None
    return row[0]


def _count(store: UserStore) -> int:
    return int(_scalar(store, "SELECT COUNT(*) FROM users"))


def _table_info(store: UserStore) -> list[tuple[object, ...]]:
    with sqlite3.connect(store.db_path) as connection:
        rows = connection.execute("PRAGMA table_info(users)").fetchall()
    return [tuple(row) for row in rows]


def _table_sql(store: UserStore) -> str:
    with sqlite3.connect(store.db_path) as connection:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND tbl_name = 'users'"
        ).fetchone()
    assert row is not None
    return str(row[0])


def _snapshot(root: Path) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    if not root.exists():
        return found
    for item in root.rglob("*"):
        rel = item.relative_to(root).as_posix()
        if item.is_symlink():
            found[rel] = b"link:" + os.readlink(item).encode()
        elif item.is_file():
            found[rel] = item.read_bytes()
        else:
            found[rel] = b"dir"
    return found


def _run_cli(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    prompted: str | None,
) -> tuple[int, int]:
    calls = {"count": 0}

    def _getpass(*_args: object, **_kwargs: object) -> str:
        calls["count"] += 1
        if prompted is None:
            raise AssertionError("getpass was called")
        return prompted

    def _input(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("input was called")

    monkeypatch.setattr("scripts.manage_users.getpass.getpass", _getpass)
    monkeypatch.setattr("builtins.input", _input)
    return main(argv), calls["count"]


def test_correct_and_wrong_password(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    assert store.authenticate("ada", _SECRET) is True
    assert store.authenticate("ada", _OTHER) is False
    assert store.authenticate("ada", "") is False


def test_same_password_differs_per_user(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    store.add_user("bea", "admin", _SECRET)
    first = str(_scalar(store, "SELECT password_hash FROM users WHERE username = ?", ("ada",)))
    second = str(_scalar(store, "SELECT password_hash FROM users WHERE username = ?", ("bea",)))
    assert first != second
    assert first.split("$")[5] != second.split("$")[5]
    assert store.authenticate("ada", _SECRET) is True
    assert store.authenticate("bea", _SECRET) is True


def test_stored_hash_uses_scrypt_parameters(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _UNICODE)
    stored = str(_scalar(store, "SELECT password_hash FROM users WHERE username = ?", ("ada",)))
    parts = stored.split("$")
    assert parts[0] == "scrypt"
    assert int(parts[1]) == 2**14
    assert int(parts[2]) == 8
    assert int(parts[3]) == 1
    assert int(parts[4]) == 64
    salt = bytes.fromhex(parts[5])
    digest = bytes.fromhex(parts[6])
    assert len(salt) == 16
    assert len(digest) == 64
    expected = hashlib.scrypt(
        _UNICODE.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=64,
    )
    assert digest == expected
    assert store.authenticate("ada", _UNICODE) is True


def test_db_file_does_not_contain_the_password(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    assert _SECRET.encode("utf-8") not in store.db_path.read_bytes()


def test_authenticate_uses_compare_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    seen: list[tuple[bytes, bytes]] = []
    real = hmac.compare_digest

    def _wrapped(left: object, right: object) -> bool:
        assert isinstance(left, (bytes, bytearray))
        assert isinstance(right, (bytes, bytearray))
        seen.append((bytes(left), bytes(right)))
        return bool(real(left, right))

    monkeypatch.setattr("trace_search.ui.users.hmac.compare_digest", _wrapped)
    assert store.authenticate("ada", _SECRET) is True
    assert store.authenticate("ada", _OTHER) is False
    assert len(seen) == 2
    assert seen[0][0] == seen[0][1]
    assert seen[1][0] != seen[1][1]


def test_pragma_table_info_pins_the_column_set(tmp_path: Path) -> None:
    info = _table_info(_store(tmp_path))
    columns = [row[1] for row in info]
    assert columns == list(_COLUMNS)
    assert [row[2] for row in info] == ["TEXT", "TEXT", "TEXT", "TEXT", "INTEGER"]
    assert [row[3] for row in info] == [1, 1, 1, 1, 1]
    assert [row[4] for row in info] == [None, None, None, None, None]
    assert [row[5] for row in info] == [1, 0, 0, 0, 0]
    for planted in _PLANTED:
        assert planted not in columns


def test_role_check_rejects_other_roles(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert "CHECK (role IN ('operator', 'admin'))" in _table_sql(store)
    with sqlite3.connect(store.db_path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO users (username, role, password_hash, created_at, disabled) "
                "VALUES (?, ?, ?, ?, ?)",
                ("ada", "viewer", "hash", "2026-01-01T00:00:00+00:00", 0),
            )


def test_add_user_rejects_unknown_and_mismatched_role(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="role"):
        store.add_user("ada", "viewer", _SECRET)
    with pytest.raises(ValueError, match="role"):
        store.add_user("ada", "Operator", _SECRET)
    with pytest.raises(ValueError, match="role"):
        store.add_user("ada", "ADMIN", _SECRET)
    assert _count(store) == 0


def test_duplicate_username_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    with pytest.raises(ValueError, match="already exists"):
        store.add_user("ada", "admin", _OTHER)
    assert _count(store) == 1
    assert store.authenticate("ada", _SECRET) is True
    assert store.authenticate("ada", _OTHER) is False


def test_empty_password_and_blank_username_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="password"):
        store.add_user("ada", "operator", "")
    with pytest.raises(ValueError, match="username"):
        store.add_user("", "operator", _SECRET)
    with pytest.raises(ValueError, match="username"):
        store.add_user("  ada", "operator", _SECRET)
    assert _count(store) == 0


def test_disabled_user_cannot_authenticate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    store.disable_user("ada")
    store.disable_user("ada")
    assert _scalar(store, "SELECT disabled FROM users WHERE username = ?", ("ada",)) == 1
    assert _count(store) == 1
    assert store.authenticate("ada", _SECRET) is False


def test_unknown_user_cannot_authenticate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    assert store.authenticate("bea", _SECRET) is False
    with pytest.raises(ValueError, match="unknown user"):
        store.disable_user("bea")
    with pytest.raises(ValueError, match="unknown user"):
        store.set_role("bea", "admin")


def test_set_role_updates_an_existing_account(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    store.set_role("ada", "admin")
    assert _scalar(store, "SELECT role FROM users WHERE username = ?", ("ada",)) == "admin"
    store.set_role("ada", "operator")
    assert _scalar(store, "SELECT role FROM users WHERE username = ?", ("ada",)) == "operator"
    assert store.authenticate("ada", _SECRET) is True
    with pytest.raises(ValueError, match="role"):
        store.set_role("ada", "viewer")
    assert _scalar(store, "SELECT role FROM users WHERE username = ?", ("ada",)) == "operator"


def test_username_is_bound_as_data(tmp_path: Path) -> None:
    store = _store(tmp_path)
    odd = "ada' OR '1'='1"
    store.add_user(odd, "operator", _SECRET)
    assert store.authenticate(odd, _SECRET) is True
    assert store.authenticate("ada", _SECRET) is False


def test_created_at_is_a_utc_timestamp(tmp_path: Path) -> None:
    started = datetime.now(timezone.utc)
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    finished = datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(
        str(_scalar(store, "SELECT created_at FROM users WHERE username = ?", ("ada",)))
    )
    assert parsed.tzinfo is not None
    assert started <= parsed <= finished


def test_reopen_keeps_the_account(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "users.sqlite"
    store = UserStore(path)
    store.add_user("ada", "admin", _SECRET)
    assert path.is_file()
    resolved = store.db_path.resolve()
    assert resolved != _REPO
    assert _REPO not in resolved.parents
    again = UserStore(path)
    assert again.authenticate("ada", _SECRET) is True


def test_relative_path_outside_the_repo_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    store = UserStore(Path("nested") / "users.sqlite")
    store.add_user("ada", "operator", _SECRET)
    assert store.db_path.is_absolute()
    assert store.authenticate("ada", _SECRET) is True


def test_non_spec_hash_framing_does_not_authenticate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    stored = str(_scalar(store, "SELECT password_hash FROM users WHERE username = ?", ("ada",)))
    parts = stored.split("$")
    wide_salt = b"\x11" * 32
    wide_digest = hashlib.scrypt(
        _SECRET.encode("utf-8"),
        salt=wide_salt,
        n=2**14,
        r=8,
        p=1,
        dklen=64,
    )
    variants = (
        "bcrypt$" + "$".join(parts[1:]),
        stored + "$extra",
        "$".join([*parts[:5], wide_salt.hex(), wide_digest.hex()]),
    )
    for framed in variants:
        with sqlite3.connect(store.db_path) as connection:
            connection.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (framed, "ada"),
            )
        assert store.authenticate("ada", _SECRET) is False


@pytest.mark.parametrize(
    "stored",
    [
        "nope",
        "no$1$8$1$64$aa$bb",
        "scrypt$16384$8$1$64$zz$abcd",
        "scrypt$16384$8$1$64$" + ("00" * 8) + "$" + ("00" * 64),
        "scrypt$16384$8$1$64$" + ("00" * 16) + "$" + ("ab" * 8),
    ],
)
def test_tampered_hash_does_not_authenticate(tmp_path: Path, stored: str) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (stored, "ada"),
        )
    assert store.authenticate("ada", _SECRET) is False


def test_repo_root_db_path_is_refused() -> None:
    db = _REPO / "users.sqlite"
    try:
        with pytest.raises(ValueError, match="db path is inside the repository"):
            UserStore(_REPO)
        with pytest.raises(ValueError, match="db path is inside the repository"):
            UserStore(db)
        assert not db.exists()
    finally:
        if db.is_file():
            db.unlink()


@pytest.mark.parametrize(
    "rel",
    [
        "src/users.sqlite",
        "src/trace_search/ui/users.sqlite",
        "tests/ui/users.sqlite",
    ],
)
def test_nested_repo_db_path_is_refused(rel: str) -> None:
    db = _REPO / rel
    before = _snapshot(db.parent)
    try:
        with pytest.raises(ValueError, match="db path is inside the repository"):
            UserStore(db)
        assert _snapshot(db.parent) == before
        assert not db.exists()
    finally:
        if db.is_file():
            db.unlink()


def test_relative_and_dot_paths_inside_the_repo_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_REPO)
    with pytest.raises(ValueError, match="db path is inside the repository"):
        UserStore(Path("src") / "users.sqlite")
    with pytest.raises(ValueError, match="db path is inside the repository"):
        UserStore(Path("."))
    assert not (_REPO / "src" / "users.sqlite").exists()


def test_symlink_outside_into_repo_is_refused(tmp_path: Path) -> None:
    link = tmp_path / "repo-src"
    link.symlink_to(_REPO / "src", target_is_directory=True)
    db = link / "users.sqlite"
    target = _REPO / "src" / "users.sqlite"
    before = _snapshot(_REPO / "src")
    try:
        with pytest.raises(ValueError, match="db path is inside the repository"):
            UserStore(db)
        assert _snapshot(_REPO / "src") == before
        assert not target.exists()
    finally:
        if target.is_file():
            target.unlink()


def test_symlink_file_into_repo_is_refused(tmp_path: Path) -> None:
    target = _REPO / "src" / "trace_search" / "ui" / "users.sqlite"
    link = tmp_path / "db-link"
    link.symlink_to(target)
    before = _snapshot(target.parent)
    try:
        with pytest.raises(ValueError, match="db path is inside the repository"):
            UserStore(link)
        assert _snapshot(target.parent) == before
        assert not target.exists()
    finally:
        if target.is_file():
            target.unlink()


def test_symlink_inside_repo_pointing_outside_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside.sqlite"
    link = _REPO / ".tw2-userdb-link"
    link.symlink_to(outside)
    try:
        with pytest.raises(ValueError, match="db path is inside the repository"):
            UserStore(link)
        assert not outside.exists()
        assert link.is_symlink()
    finally:
        link.unlink(missing_ok=True)


def test_circular_symlink_is_refused(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.symlink_to(right)
    right.symlink_to(left)
    with pytest.raises(ValueError, match="db path is inside the repository"):
        UserStore(left)


def test_missing_repo_root_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trace_search.ui.users.repo_root_from", lambda _start: None)
    with pytest.raises(ValueError, match="repository root"):
        UserStore(tmp_path / "users.sqlite")
    assert not (tmp_path / "users.sqlite").exists()


def test_src_and_scripts_have_no_password_literal() -> None:
    hits: list[str] = []
    for root in (_REPO / "src", _REPO / "scripts"):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for number, line in enumerate(text.splitlines(), start=1):
                if _PASSWORD_LITERAL.search(line):
                    hits.append(f"{path.relative_to(_REPO)}:{number}")
    assert hits == []


def test_cli_add_disable_and_set_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "users.sqlite"
    code, calls = _run_cli(
        ["add", "--db", str(db), "--username", "ada", "--role", "operator"],
        monkeypatch,
        _PROMPTED,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert calls == 1
    assert _PROMPTED not in captured.out
    assert _PROMPTED not in captured.err
    assert UserStore(db).authenticate("ada", _PROMPTED) is True

    code, calls = _run_cli(
        ["set-role", "--db", str(db), "--username", "ada", "--role", "admin"],
        monkeypatch,
        None,
    )
    assert code == 0
    assert calls == 0
    assert _scalar(UserStore(db), "SELECT role FROM users WHERE username = ?", ("ada",)) == "admin"

    code, calls = _run_cli(
        ["disable", "--db", str(db), "--username", "ada"],
        monkeypatch,
        None,
    )
    assert code == 0
    assert calls == 0
    assert UserStore(db).authenticate("ada", _PROMPTED) is False


def test_cli_duplicate_bad_role_and_missing_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "users.sqlite"
    UserStore(db).add_user("ada", "operator", _SECRET)
    code, calls = _run_cli(
        ["add", "--db", str(db), "--username", "ada", "--role", "admin"],
        monkeypatch,
        _PROMPTED,
    )
    duplicate_err = capsys.readouterr().err
    assert code == 1
    assert calls == 1
    assert "already exists" in duplicate_err
    assert _PROMPTED not in duplicate_err

    code, _calls = _run_cli(
        ["add", "--db", str(db), "--username", "bea", "--role", "viewer"],
        monkeypatch,
        _PROMPTED,
    )
    assert code == 1
    assert "role" in capsys.readouterr().err
    assert _count(UserStore(db)) == 1

    code, calls = _run_cli(
        ["disable", "--db", str(db), "--username", "missing"],
        monkeypatch,
        None,
    )
    assert code == 1
    assert calls == 0
    assert "unknown user" in capsys.readouterr().err

    code, calls = _run_cli(
        ["set-role", "--db", str(db), "--username", "missing", "--role", "admin"],
        monkeypatch,
        None,
    )
    assert code == 1
    assert calls == 0

    code, calls = _run_cli(
        ["set-role", "--db", str(db), "--username", "ada", "--role", "viewer"],
        monkeypatch,
        None,
    )
    assert code == 1
    assert calls == 0
    assert _scalar(UserStore(db), "SELECT role FROM users WHERE username = ?", ("ada",)) == (
        "operator"
    )


def test_cli_empty_password_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "users.sqlite"
    code, calls = _run_cli(
        ["add", "--db", str(db), "--username", "ada", "--role", "operator"],
        monkeypatch,
        "",
    )
    assert code == 1
    assert calls == 1
    assert _count(UserStore(db)) == 0


def test_cli_rejects_password_on_argv_without_echoing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "users.sqlite"
    leaked = "from-argv"
    cases = [
        ["add", "--db", str(db), "--username", "ada", "--role", "operator", "--password", leaked],
        ["add", "--db", str(db), "--username", "ada", "--role", "operator", leaked],
        ["add", "--password", leaked],
        ["--password", leaked],
        ["--password=" + leaked],
    ]
    for argv in cases:
        code, calls = _run_cli(argv, monkeypatch, None)
        captured = capsys.readouterr()
        assert code == 2
        assert calls == 0
        assert leaked not in captured.out
        assert leaked not in captured.err
        assert not db.exists() or _count(UserStore(db)) == 0


def test_cli_refuses_db_inside_repo_before_prompt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = _REPO / "users.sqlite"
    try:
        code, calls = _run_cli(
            ["add", "--db", str(db), "--username", "ada", "--role", "operator"],
            monkeypatch,
            _PROMPTED,
        )
        captured = capsys.readouterr()
        assert code == 1
        assert calls == 0
        assert "db path is inside the repository" in captured.err
        assert _PROMPTED not in captured.err
        assert not db.exists()
    finally:
        if db.is_file():
            db.unlink()


def test_cli_missing_or_unknown_command_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    capsys.readouterr()
    assert main(["nope"]) == 2
    captured = capsys.readouterr()
    assert captured.err.strip() == "invalid arguments"
    assert "nope" not in captured.err


def _set_hash(store: UserStore, username: str, stored: str) -> None:
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (stored, username),
        )


def _frame(n: object, r: object, p: object, dklen: object, salt: bytes, digest: bytes) -> str:
    return f"scrypt${n}${r}${p}${dklen}${salt.hex()}${digest.hex()}"


@pytest.mark.parametrize(
    "variant",
    [" " + _SECRET, _SECRET + " ", _SECRET.upper(), _SECRET[:-1], _SECRET + "x"],
)
def test_near_miss_password_is_rejected(tmp_path: Path, variant: str) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    assert store.authenticate("ada", variant) is False
    assert store.authenticate("ada", _SECRET) is True


def test_tampered_digest_tail_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    stored = str(_scalar(store, "SELECT password_hash FROM users WHERE username = ?", ("ada",)))
    head, digest = stored.rsplit("$", 1)
    flipped = digest[:-2] + ("00" if digest[-2:] != "00" else "11")
    _set_hash(store, "ada", head + "$" + flipped)
    assert store.authenticate("ada", _SECRET) is False


@pytest.mark.parametrize("probe", ["ADA", "Ada", "a%", "a_a", "_a_", "%"])
def test_username_lookup_is_exact(tmp_path: Path, probe: str) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    assert store.authenticate(probe, _SECRET) is False
    store.add_user(probe, "admin", _OTHER)
    assert _count(store) == 2
    assert store.authenticate("ada", _SECRET) is True
    assert store.authenticate(probe, _OTHER) is True
    assert store.authenticate(probe, _SECRET) is False


def test_disable_and_set_role_touch_only_one_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    store.add_user("bea", "operator", _OTHER)
    store.disable_user("ada")
    store.set_role("ada", "admin")
    assert store.authenticate("bea", _OTHER) is True
    assert store.authenticate("ada", _SECRET) is False
    assert _scalar(store, "SELECT role FROM users WHERE username = ?", ("bea",)) == "operator"
    assert _scalar(store, "SELECT disabled FROM users WHERE username = ?", ("bea",)) == 0
    assert _scalar(store, "SELECT role FROM users WHERE username = ?", ("ada",)) == "admin"
    assert _scalar(store, "SELECT disabled FROM users WHERE username = ?", ("ada",)) == 1


def test_injection_username_does_not_reach_other_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    odd = "x' OR '1'='1"
    assert store.authenticate(odd, _SECRET) is False
    store.add_user(odd, "operator", _OTHER)
    assert _count(store) == 2
    assert store.authenticate(odd, _SECRET) is False
    assert store.authenticate(odd, _OTHER) is True
    assert store.authenticate("ada", _SECRET) is True


def test_refused_nested_db_path_creates_no_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = _REPO / "src" / "kill-probe-dir"
    assert not parent.exists()
    created: list[Path] = []
    opened: list[object] = []
    real_mkdir = Path.mkdir
    real_open = os.open

    def _mkdir(self: Path, *args: object, **kwargs: object) -> None:
        created.append(self)
        real_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    def _open(*args: object, **kwargs: object) -> int:
        opened.append(args[0] if args else None)
        return real_open(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", _mkdir)
    monkeypatch.setattr(os, "open", _open)
    before = _snapshot(_REPO / "src")
    try:
        with pytest.raises(ValueError, match="db path is inside the repository"):
            UserStore(parent / "deeper" / "users.sqlite")
        assert created == []
        assert opened == []
        assert not parent.exists()
        assert _snapshot(_REPO / "src") == before
    finally:
        if parent.exists():
            for path in sorted(parent.rglob("*"), reverse=True):
                if path.is_dir() and not path.is_symlink():
                    path.rmdir()
                else:
                    path.unlink()
            parent.rmdir()


def test_new_parent_directories_are_created_outside_repo(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "a" / "b" / "users.sqlite")
    store.add_user("ada", "operator", _SECRET)
    assert (tmp_path / "a" / "b").is_dir()
    assert store.authenticate("ada", _SECRET) is True


def test_new_db_file_is_mode_0600(tmp_path: Path) -> None:
    parent_mode = tmp_path.stat().st_mode
    previous = os.umask(0)
    try:
        store = UserStore(tmp_path / "users.sqlite")
        mode = store.db_path.stat().st_mode & 0o777
    finally:
        os.umask(previous)
    assert mode == 0o600
    assert tmp_path.stat().st_mode == parent_mode


def test_lost_create_race_still_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = os.open

    def _open(*args: object, **kwargs: object) -> int:
        fd = real_open(*args, **kwargs)  # type: ignore[arg-type]
        os.close(fd)
        raise FileExistsError

    monkeypatch.setattr("trace_search.ui.users.os.open", _open)
    store = UserStore(tmp_path / "users.sqlite")
    store.add_user("ada", "operator", _SECRET)
    assert store.authenticate("ada", _SECRET) is True


def test_existing_db_mode_is_left_alone(tmp_path: Path) -> None:
    path = tmp_path / "users.sqlite"
    previous = os.umask(0)
    try:
        with sqlite3.connect(path) as connection:
            connection.execute("SELECT 1")
        created = path.stat().st_mode & 0o777
    finally:
        os.umask(previous)
    assert created != 0o600
    again = UserStore(path)
    again.add_user("ada", "operator", _SECRET)
    assert again.authenticate("ada", _SECRET) is True
    assert path.stat().st_mode & 0o777 == created


def test_unknown_and_disabled_pay_for_scrypt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    store.disable_user("ada")
    seen: list[tuple[int, int, int, int, int]] = []
    real = hashlib.scrypt

    def _wrapped(*args: object, **kwargs: object) -> bytes:
        salt = kwargs["salt"]
        assert isinstance(salt, bytes)
        seen.append(
            (
                int(kwargs["n"]),  # type: ignore[call-overload]
                int(kwargs["r"]),  # type: ignore[call-overload]
                int(kwargs["p"]),  # type: ignore[call-overload]
                int(kwargs["dklen"]),  # type: ignore[call-overload]
                len(salt),
            )
        )
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("trace_search.ui.users.hashlib.scrypt", _wrapped)
    monkeypatch.setattr("trace_search.ui.users.hmac.compare_digest", lambda *_a, **_k: True)
    assert store.authenticate("bea", _SECRET) is False
    assert store.authenticate("ada", _SECRET) is False
    assert seen == [(2**14, 8, 1, 64, 16), (2**14, 8, 1, 64, 16)]


def test_verify_uses_stored_scrypt_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    salt = bytes(range(16))
    digest = hashlib.scrypt(
        _SECRET.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=2,
        dklen=64,
    )
    _set_hash(store, "ada", _frame(2**14, 8, 2, 64, salt, digest))
    seen: list[tuple[int, int, int, int]] = []
    real = hashlib.scrypt

    def _wrapped(*args: object, **kwargs: object) -> bytes:
        seen.append(
            (
                int(kwargs["n"]),  # type: ignore[call-overload]
                int(kwargs["r"]),  # type: ignore[call-overload]
                int(kwargs["p"]),  # type: ignore[call-overload]
                int(kwargs["dklen"]),  # type: ignore[call-overload]
            )
        )
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("trace_search.ui.users.hashlib.scrypt", _wrapped)
    assert store.authenticate("ada", _SECRET) is True
    assert seen == [(2**14, 8, 2, 64)]


@pytest.mark.parametrize(
    ("n", "r", "p", "dklen"),
    [
        (2**13, 8, 1, 64),
        (2**14, 4, 1, 64),
        (2**14, 8, 3, 64),
        (2**14, 8, 1, 32),
    ],
)
def test_disallowed_kdf_parameters_do_not_authenticate(
    tmp_path: Path, n: int, r: int, p: int, dklen: int
) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    salt = bytes(range(16))
    digest = hashlib.scrypt(
        _SECRET.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=dklen,
    )
    _set_hash(store, "ada", _frame(n, r, p, dklen, salt, digest))
    assert store.authenticate("ada", _SECRET) is False


@pytest.mark.parametrize("field", [1, 2, 3, 4])
def test_tampered_kdf_field_does_not_authenticate(tmp_path: Path, field: int) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    parts = str(
        _scalar(store, "SELECT password_hash FROM users WHERE username = ?", ("ada",))
    ).split("$")
    parts[field] = "999999999" if field == 1 else "7"
    _set_hash(store, "ada", "$".join(parts))
    assert store.authenticate("ada", _SECRET) is False


def test_non_numeric_kdf_field_does_not_authenticate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_user("ada", "operator", _SECRET)
    parts = str(
        _scalar(store, "SELECT password_hash FROM users WHERE username = ?", ("ada",))
    ).split("$")
    parts[1] = "abc"
    _set_hash(store, "ada", "$".join(parts))
    assert store.authenticate("ada", _SECRET) is False


def test_sqlite_connection_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []
    real_connect = sqlite3.connect

    class _Tracking(sqlite3.Connection):
        def close(self) -> None:
            closed.append(True)
            super().close()

    def _connect(database: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = _Tracking
        return real_connect(database, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("trace_search.ui.users.sqlite3.connect", _connect)
    store = UserStore(tmp_path / "users.sqlite")
    store.add_user("ada", "operator", _SECRET)
    assert store.authenticate("ada", _SECRET) is True
    assert len(closed) >= 3


@pytest.mark.parametrize(
    "argv",
    [
        ["add", "--username", "ada", "--role", "operator"],
        ["add", "--db", "DB", "--role", "operator"],
        ["add", "--db", "DB", "--username", "ada"],
        ["disable", "--username", "ada"],
        ["disable", "--db", "DB"],
        ["set-role", "--username", "ada", "--role", "admin"],
        ["set-role", "--db", "DB", "--role", "admin"],
        ["set-role", "--db", "DB", "--username", "ada"],
    ],
)
def test_cli_required_options(
    argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "users.sqlite"
    UserStore(db).add_user("ada", "operator", _SECRET)
    monkeypatch.chdir(tmp_path)
    argv = [str(db) if part == "DB" else part for part in argv]
    code, calls = _run_cli(argv, monkeypatch, None)
    assert code == 2
    assert calls == 0
    assert _count(UserStore(db)) == 1
    assert _scalar(UserStore(db), "SELECT role FROM users WHERE username = ?", ("ada",)) == (
        "operator"
    )
    assert UserStore(db).authenticate("ada", _SECRET) is True


def test_missing_db_option_creates_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    code, calls = _run_cli(
        ["add", "--username", "ada", "--role", "operator"],
        monkeypatch,
        None,
    )
    assert code == 2
    assert calls == 0
    assert list(empty.iterdir()) == []


@pytest.mark.parametrize("prompt_error", [EOFError, KeyboardInterrupt])
def test_password_prompt_closed_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prompt_error: type[BaseException],
) -> None:
    db = tmp_path / "users.sqlite"

    def _getpass(*_args: object, **_kwargs: object) -> str:
        raise prompt_error()

    monkeypatch.setattr("scripts.manage_users.getpass.getpass", _getpass)
    code = main(["add", "--db", str(db), "--username", "ada", "--role", "operator"])
    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "password prompt closed" in captured.err
    assert _count(UserStore(db)) == 0


def test_padded_role_is_a_value_error(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="role"):
        store.add_user("ada", " admin", _SECRET)
    assert _count(store) == 0


def _script_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(_REPO / "src")
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if previous is None else src + os.pathsep + previous
    env["HOME"] = str(tmp_path)
    return env


def test_script_process_exit_codes(tmp_path: Path) -> None:
    db = tmp_path / "users.sqlite"
    UserStore(db).add_user("ada", "operator", _SECRET)
    script = _REPO / "scripts" / "manage_users.py"
    env = _script_env(tmp_path)
    inside = _REPO / "users.sqlite"

    def run(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, str(script), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )

    try:
        missing = run("disable", "--db", str(db), "--username", "zed")
        assert missing.returncode == 1
        assert b"Traceback" not in missing.stderr
        disabled = run("disable", "--db", str(db), "--username", "ada")
        assert disabled.returncode == 0
        bogus = run("bogus")
        assert bogus.returncode == 2
        assert b"Traceback" not in bogus.stderr
        omitted = run("add")
        assert omitted.returncode == 2
        refused = run(
            "add",
            "--db",
            str(inside),
            "--username",
            "ada",
            "--role",
            "operator",
        )
        assert refused.returncode == 1
        assert b"Traceback" not in refused.stderr
        assert not inside.exists()
        prompted = run(
            "add",
            "--db",
            str(db),
            "--username",
            "bea",
            "--role",
            "operator",
        )
        assert prompted.returncode == 1
        assert b"Traceback" not in prompted.stderr
        assert b"password prompt closed" in prompted.stderr
        assert UserStore(db).authenticate("ada", _SECRET) is False
    finally:
        if inside.is_file():
            inside.unlink()
