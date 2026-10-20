"""Exactly one subprocess smoke test that the CLI entry point runs."""

import json
import subprocess
import sys
from pathlib import Path


def test_cli_subprocess_smoke(tmp_path):
    """The check_licenses.py CLI runs and exits 0 on a clean MIT manifest."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "good-lib",
                "version": "1.0.0",
                "type": "library",
                "license": "MIT",
                "source_url": "https://example.com/good-lib",
                "sha256": "a" * 64,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_licenses.py",
            "--manifest",
            str(manifest_path),
            "--no-scan-installed",
        ],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout
