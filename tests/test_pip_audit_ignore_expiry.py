"""Expiry and scope of the CI-only pip-audit ignore."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import scripts.check_pip_audit_ignore_expiry as expiry

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "security.yml"
COMMENT = (
    "expires 2026-11-21; affects opentelemetry-api 1.37.0 in semgrep's CI-tools env only; "
    "blocked on semgrep pinning opentelemetry-api~=1.37.0 (latest 1.182.1); "
    "remove when semgrep allows >=1.45.1; approved by Shitanshu 22 Oct 2026; B-79"
)


def test_date_after_expiry_fails() -> None:
    assert expiry.EXPIRY.isoformat() == "2026-11-21"
    assert expiry.ignore_is_current(date(2026, 11, 22)) is False
    assert expiry.main(date(2026, 11, 22)) == 1


def test_date_on_or_before_expiry_passes() -> None:
    assert expiry.ignore_is_current(date(2026, 11, 21)) is True
    assert expiry.main(date(2026, 11, 21)) == 0
    assert expiry.ignore_is_current(date(2026, 10, 22)) is True
    assert expiry.main(date(2026, 10, 22)) == 0


def _pip_audit_script(workflow: str) -> list[str]:
    marker = "name: Run pip-audit and name any vulnerable packages\n"
    rest = workflow[workflow.index(marker) :]
    body = rest[rest.index("run: |\n") + len("run: |\n") :]
    lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("    - "):
            break
        lines.append(line[8:] if line.startswith("        ") else line)
    return lines


def _shell_commands(lines: list[str]) -> list[str]:
    commands: list[str] = []
    buf: list[str] = []
    for line in lines:
        if not buf and (not line.strip() or line.strip().startswith("#")):
            continue
        buf.append(line)
        if line.rstrip().endswith("\\"):
            continue
        commands.append("\n".join(buf))
        buf = []
    if buf:
        commands.append("\n".join(buf))
    return commands


def test_ignore_is_only_on_the_ci_tools_audits() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert workflow.count("--ignore-vuln") == 1
    assert f"--ignore-vuln {expiry.ADVISORY}" in workflow
    lines = workflow.splitlines()
    comment_at = next(i for i, line in enumerate(lines) if COMMENT in line)
    assert lines[comment_at + 1].strip() == f"ci_ignore=(--ignore-vuln {expiry.ADVISORY})"

    script = _pip_audit_script(workflow)
    assert any(line.strip() == "python scripts/check_pip_audit_ignore_expiry.py" for line in script)
    audits = [
        command for command in _shell_commands(script) if command.lstrip().startswith("pip-audit")
    ]
    assert len(audits) == 3
    ci_audits = [command for command in audits if "${ci_ignore[@]}" in command]
    app_audits = [command for command in audits if "-r requirements-dev.lock" in command]
    assert len(ci_audits) == 2
    assert len(app_audits) == 1
    app_audit = app_audits[0]
    assert "--ignore-vuln" not in app_audit
    assert "ci_ignore" not in app_audit
    assert "-r requirements.txt" not in "\n".join(audits)
    assert "-r requirements.lock" not in "\n".join(audits)
    assert "-r requirements-ci.lock" not in "\n".join(audits)

    workflow_hits = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / ".github" / "workflows").glob("*.yml")
        if "--ignore-vuln" in path.read_text(encoding="utf-8")
    ]
    assert workflow_hits == [".github/workflows/security.yml"]


def _step_block(workflow: str, step_name: str) -> str:
    marker = f"- name: {step_name}\n"
    start = workflow.index(marker)
    rest = workflow[start + len(marker) :]
    end = rest.index("\n    - name:")
    return workflow[start : start + len(marker) + end]


def test_cli_after_expiry_exits_1() -> None:
    """A real process must exit 1. Dropping SystemExit around main() fails this."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "scripts/check_pip_audit_ignore_expiry.py", "2026-11-22"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "expired after 2026-11-21" in completed.stderr


def test_expiry_script_runs_before_pip_audit() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    block = _step_block(workflow, "Run pip-audit and name any vulnerable packages")
    assert "continue-on-error" not in block
    assert "set +e" not in block
    script = _pip_audit_script(workflow)
    commands = [command.lstrip() for command in _shell_commands(script)]
    expiry_at = next(
        index
        for index, command in enumerate(commands)
        if command.startswith("python scripts/check_pip_audit_ignore_expiry.py")
    )
    audit_at = next(
        index for index, command in enumerate(commands) if command.startswith("pip-audit")
    )
    assert expiry_at < audit_at
    assert commands[expiry_at] == "python scripts/check_pip_audit_ignore_expiry.py"
