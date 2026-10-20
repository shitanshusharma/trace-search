#!/usr/bin/env python3
"""Validate GitHub Action pins in this repo's workflows.

Every `uses: owner/repo[/path]@<sha> # vX.Y.Z` pin must be:

1. The commit the named upstream tag points at, after peeling annotated tag
   objects (`refs/tags/vX.Y.Z^{}`).
2. A commit GitHub's commits API returns 200 for (tag objects return 422).
3. Reachable from that tag: GET /repos/{owner}/{repo}/compare/{tag}...{sha}
   returns 200 and status "identical".

A commit that merely exists somewhere in the fork network is not enough.
`.github/workflows/ci.yml` is skipped until PR #3, which SHA-pins that file, merges.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PIN_RE = re.compile(r"uses:\s*(?P<action>\S+?)@(?P<sha>[0-9a-f]{40})\s+#\s*v(?P<version>\d+\S*)")

# PR #3 owns ci.yml. Drop this once that pin change is on main.
SKIP_WORKFLOW_NAMES = {"ci.yml"}

GetFn = Callable[[str], tuple[int, dict]]
LsRemoteFn = Callable[[str, str, str], str]


@dataclass(frozen=True)
class Pin:
    file: str
    line: int
    action: str
    sha: str
    version: str

    @property
    def owner(self) -> str:
        return self.action.split("/")[0]

    @property
    def repo(self) -> str:
        return self.action.split("/")[1]

    @property
    def tag(self) -> str:
        return f"v{self.version}"


def iter_pins(text: str, filename: str) -> list[Pin]:
    pins: list[Pin] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = PIN_RE.search(line)
        if not match:
            continue
        action = match.group("action")
        if len(action.split("/")) < 2:
            continue
        pins.append(
            Pin(
                file=filename,
                line=lineno,
                action=action,
                sha=match.group("sha"),
                version=match.group("version"),
            )
        )
    return pins


def peeled_commit(ls_remote_output: str, tag: str) -> str | None:
    """Return the commit SHA for tag. Prefer the peeled `^{}` line."""
    ref = f"refs/tags/{tag}"
    direct: str | None = None
    peeled: str | None = None
    for raw in ls_remote_output.splitlines():
        parts = raw.split()
        if len(parts) < 2:
            continue
        sha, name = parts[0], parts[1]
        if name == f"{ref}^{{}}":
            peeled = sha
        elif name == ref:
            direct = sha
    return peeled or direct


def check_pin(pin: Pin, get: GetFn, ls_remote: LsRemoteFn) -> list[str]:
    """Return human-readable errors. Empty means the pin is good."""
    errors: list[str] = []
    where = f"{pin.file}:{pin.line} {pin.action}@{pin.sha} # {pin.tag}"
    try:
        remote = ls_remote(pin.owner, pin.repo, pin.tag)
    except OSError as exc:
        return [f"{where}: git ls-remote failed: {exc}"]

    expected = peeled_commit(remote, pin.tag)
    if expected is None:
        return [f"{where}: tag {pin.tag} not found on {pin.owner}/{pin.repo}"]
    if pin.sha != expected:
        return [
            f"{where}: pin is not the peeled commit for {pin.tag} "
            f"(expected {expected}). Tag objects and fork-network commits are rejected."
        ]

    commit_url = f"https://api.github.com/repos/{pin.owner}/{pin.repo}/commits/{pin.sha}"
    status, payload = get(commit_url)
    if status != 200:
        message = payload.get("message", "")
        return [f"{where}: GET commits/{pin.sha} returned {status} {message}".rstrip()]

    compare_url = (
        f"https://api.github.com/repos/{pin.owner}/{pin.repo}/compare/{pin.tag}...{pin.sha}"
    )
    status, payload = get(compare_url)
    if status != 200:
        message = payload.get("message", "")
        return [f"{where}: GET compare/{pin.tag}...{pin.sha} returned {status} {message}".rstrip()]
    compare_status = payload.get("status")
    if compare_status != "identical":
        errors.append(
            f"{where}: commit is not the tagged commit "
            f"(compare status={compare_status!r}, want 'identical')"
        )
        return errors
    print(
        f"OK {pin.owner}/{pin.repo} {pin.tag} pin={pin.sha} "
        f"commits_api=200 compare_api=200 compare_status=identical "
        f"peeled_from_tag={expected}"
    )
    return errors


def workflow_files(root: Path) -> list[Path]:
    workflows = root / ".github" / "workflows"
    if not workflows.is_dir():
        return []
    files = []
    for path in sorted(workflows.glob("*.yml")) + sorted(workflows.glob("*.yaml")):
        if path.name in SKIP_WORKFLOW_NAMES:
            continue
        files.append(path)
    return files


def validate_repo(root: Path, get: GetFn, ls_remote: LsRemoteFn) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for path in workflow_files(root):
        rel = path.relative_to(root).as_posix()
        for pin in iter_pins(path.read_text(), rel):
            key = (pin.owner, pin.repo, pin.sha, pin.tag)
            if key in seen:
                # Still record a same-tag mismatch if the comment differs on a duplicate.
                continue
            seen.add(key)
            errors.extend(check_pin(pin, get, ls_remote))
    return errors


def _read_github(request: urllib.request.Request) -> tuple[int, bytes]:
    # Host is fixed to api.github.com. The path is an action pin from this repo.
    # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, response.read()


def github_get(token: str) -> GetFn:
    def get(url: str) -> tuple[int, dict]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "trace-search-pin-check",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            status, raw_bytes = _read_github(request)
            body = raw_bytes.decode()
            payload = json.loads(body) if body else {}
            return status, payload if isinstance(payload, dict) else {"data": payload}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"message": raw}
            if not isinstance(payload, dict):
                payload = {"message": str(payload)}
            return exc.code, payload

    return get


def git_ls_remote(owner: str, repo: str, tag: str) -> str:
    url = f"https://github.com/{owner}/{repo}.git"
    proc = subprocess.run(
        ["git", "ls-remote", "--tags", url, f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise OSError(proc.stderr.strip() or f"git ls-remote failed for {owner}/{repo}")
    return proc.stdout


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN", "")
    root = Path(".").resolve()
    try:
        errors = validate_repo(root, github_get(token), git_ls_remote)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if errors:
        print(f"Action pin check failed ({len(errors)}):")
        for error in errors:
            print(f"  {error}")
        return 1
    print("All action pins are peeled commits reachable from their upstream tags.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
