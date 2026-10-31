#!/usr/bin/env python3
"""Add, disable, or change the role of a local operator account.

``add`` reads the password with ``getpass``. The password is not accepted
on the command line. The SQLite file must sit outside this repository.
There is no default account.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from trace_search.ui.users import UserStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage operator accounts.",
        exit_on_error=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add = subparsers.add_parser("add", help="Add an account.", exit_on_error=False)
    disable = subparsers.add_parser(
        "disable",
        help="Disable an account.",
        exit_on_error=False,
    )
    set_role = subparsers.add_parser(
        "set-role",
        help="Change an account role.",
        exit_on_error=False,
    )
    for command in (add, disable, set_role):
        command.add_argument("--db", type=Path, required=True)
        command.add_argument("--username", required=True)
    add.add_argument("--role", required=True)
    set_role.add_argument("--role", required=True)
    return parser


def _run(parsed: argparse.Namespace) -> None:
    store = UserStore(Path(parsed.db))
    if parsed.command == "add":
        store.add_user(
            str(parsed.username),
            str(parsed.role),
            getpass.getpass("Password: "),
        )
        return
    if parsed.command == "disable":
        store.disable_user(str(parsed.username))
        return
    if parsed.command == "set-role":
        store.set_role(str(parsed.username), str(parsed.role))
        return
    raise ValueError("unknown command")


def main(argv: list[str] | None = None) -> int:
    """Run one account command. Return a process exit code."""
    parser = _parser()
    try:
        parsed, extra = parser.parse_known_args(argv)
    except argparse.ArgumentError:
        print("invalid arguments", file=sys.stderr)
        return 2
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return int(code)
    if extra:
        print("unrecognized arguments", file=sys.stderr)
        return 2
    try:
        _run(parsed)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
