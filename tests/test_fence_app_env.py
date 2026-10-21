"""Plant an unmanifested AGPL distribution in the app environment.

The licence fence must fail. The test installs requirements-dev.lock and
requirements.txt into a fresh virtualenv and does not uninstall anything
before the scan. The runtime pin file is required because the manifest
declares opencv-python-headless, and the fence loads that wheel's libavcodec.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "requirements-dev.lock"


def _venv_python(path: Path) -> Path:
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(result.stderr)
    return path / "bin" / "python"


def _run_fence(python: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONNOUSERSITE"] = "1"
    env["VIRTUAL_ENV"] = str(python.parent.parent)
    return subprocess.run(
        [
            str(python),
            str(REPO / "scripts" / "check_licenses.py"),
            "--requirements=requirements.txt",
            "--requirements=requirements-dev.txt",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_unmanifested_agpl_in_app_env_fails_fence(tmp_path: Path) -> None:
    python = _venv_python(tmp_path / "app")
    upgrade = subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert upgrade.returncode == 0, upgrade.stderr[-2000:]
    install = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            str(LOCK),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert install.returncode == 0, install.stderr[-4000:]
    runtime = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            str(REPO / "requirements.txt"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert runtime.returncode == 0, runtime.stderr[-4000:]

    clean = _run_fence(python)
    assert clean.returncode == 0, clean.stdout + clean.stderr

    site = subprocess.check_output(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        text=True,
    ).strip()
    dist = Path(site) / "ultralytics-8.3.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "\n".join(
            [
                "Metadata-Version: 2.1",
                "Name: ultralytics",
                "Version: 8.3.0",
                "License: AGPL-3.0",
                "",
            ]
        )
    )

    planted = _run_fence(python)
    output = planted.stdout + planted.stderr
    assert planted.returncode != 0, output
    lowered = output.lower()
    assert "ultralytics" in lowered
    assert "not in manifest" in lowered or "agpl" in lowered or "banned" in lowered
