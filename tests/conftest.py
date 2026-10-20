"""Pytest configuration and in-process helpers for license fence tests."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

SCHEMA_PATH = Path("licenses/manifest_schema.json")


@pytest.fixture(autouse=True)
def stub_ffmpeg_for_all_tests(monkeypatch, request):
    """
    Autouse fixture that stubs check_ffmpeg_build for all tests.

    FFmpeg-specific tests opt out with @pytest.mark.ffmpeg_real.
    """
    if "ffmpeg_real" in request.keywords:
        return

    def mock_check(*args, **kwargs):
        return [], "NOT_A_DEPENDENCY"

    import scripts.check_licenses as cl

    monkeypatch.setattr(cl, "check_ffmpeg_build", mock_check)


def make_manifest(entries=None, **kwargs):
    """Create a test manifest. Accepts entries= or entries via kwargs."""
    if entries is None:
        entries = kwargs.get("entries", [])
    return {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": entries,
    }


def library_entry(name="good-lib", license_id="MIT", version="1.0.0"):
    return {
        "name": name,
        "version": version,
        "type": "library",
        "license": license_id,
        "source_url": f"https://example.com/{name}",
        "sha256": "a" * 64,
    }


def invoke_main(manifest, extra_args=None, scan_installed=False, tmp_dir=None):
    """
    Call main(argv) in-process and return (exit_code, stdout, stderr).

    Writes the manifest to a temp file. Does not spawn a subprocess.
    """
    import scripts.check_licenses as cl

    if tmp_dir is None:
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
    else:
        tmp_dir = Path(tmp_dir)

    manifest_path = tmp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    argv = ["--manifest", str(manifest_path), "--schema", str(SCHEMA_PATH)]
    if not scan_installed:
        argv.append("--no-scan-installed")
    if extra_args:
        argv.extend(extra_args)

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cl.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()
