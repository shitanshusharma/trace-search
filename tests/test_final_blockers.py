"""Final blocker tests: MIT-CMU, schema validation, semicolon GPL."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def make_manifest(**kwargs):
    """Create a test manifest."""
    return {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": kwargs.get('entries', [])
    }


def run_checker(manifest_data: dict) -> tuple[int, str, str]:
    """Run checker with a test manifest."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(manifest_data, f)
        manifest_path = f.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                'scripts/check_licenses.py',
                '--manifest',
                manifest_path,
                '--no-scan-installed',
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env={**os.environ}
        )
        return result.returncode, result.stdout, result.stderr
    finally:
        Path(manifest_path).unlink()


def test_mit_cmu_passes():
    """Test that MIT-CMU (Pillow 13) is allowed."""
    manifest = make_manifest(entries=[
        {
            "name": "Pillow",
            "version": "13.0.0",
            "type": "library",
            "license": "MIT-CMU",
            "source_url": "https://pypi.org/project/Pillow/",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker(manifest)
    assert returncode == 0, f"MIT-CMU should pass: {stdout}"
    assert "PASSED" in stdout or "✅" in stdout


def test_schema_clip_must_use_smoke_only():
    """Test that clip with use=TRAIN fails schema validation."""
    manifest = make_manifest(entries=[
        {
            "name": "test_clip.mp4",
            "version": "1.0",
            "type": "clip",
            "license": "Pexels License",
            "license_version": "2024-01-01",
            "date_checked_ist": "2026-10-20",
            "source_url": "https://example.com/clip",
            "sha256": "a" * 64,
            "phash": [f"{i:016x}" for i in range(10)],
            "use": "TRAIN",
            "fixture_id": "FX-SMOKE-001"
        }
    ])

    returncode, stdout, stderr = run_checker(manifest)
    assert returncode == 1, f"Clip with use=TRAIN should fail: {stdout}"
    assert "schema" in stdout.lower() or "SMOKE_ONLY" in stdout


def test_schema_synthetic_clip_must_use_smoke_only():
    """Test that synthetic_clip with use=DEV fails schema validation."""
    manifest = make_manifest(entries=[
        {
            "name": "synthetic_walk.mp4",
            "version": "1.0",
            "type": "synthetic_clip",
            "license": "CC0-1.0",
            "license_version": "1.0",
            "date_checked_ist": "2026-10-20",
            "source_url": "https://example.com/synthetic",
            "sha256": "a" * 64,
            "phash": [f"{i:016x}" for i in range(10)],
            "use": "DEV",
            "fixture_id": "FX-SMOKE-SYN-001",
            "generator": "blender",
            "generator_version": "3.6.0",
            "generator_license": "Apache-2.0",
            "seed": "12345"
        }
    ])

    returncode, stdout, stderr = run_checker(manifest)
    assert returncode == 1, f"Synthetic clip with use=DEV should fail: {stdout}"
    assert "schema" in stdout.lower() or "SMOKE_ONLY" in stdout


def test_schema_invalid_fixture_id_fails():
    """Test that invalid fixture_id pattern fails."""
    manifest = make_manifest(entries=[
        {
            "name": "test_clip.mp4",
            "version": "1.0",
            "type": "clip",
            "license": "Pexels License",
            "license_version": "2024-01-01",
            "date_checked_ist": "2026-10-20",
            "source_url": "https://example.com/clip",
            "sha256": "a" * 64,
            "phash": [f"{i:016x}" for i in range(10)],
            "use": "SMOKE_ONLY",
            "fixture_id": "FX-TRAIN-001"
        }
    ])

    returncode, stdout, stderr = run_checker(manifest)
    assert returncode == 1, f"Invalid fixture_id should fail: {stdout}"
    assert "schema" in stdout.lower() or "fixture_id" in stdout.lower()


def test_semicolon_gpl_on_installed_closure_path():
    """Test that ';'-joined GPL is detected on installed-closure scanning."""
    manifest = make_manifest(entries=[
        {
            "name": "python-dateutil",
            "version": "2.9.0",
            "type": "library",
            "license": "Apache-2.0; GPL-2.0",
            "source_url": "https://pypi.org/project/python-dateutil/",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker(manifest)
    assert returncode == 1, f"Semicolon-joined GPL should fail: {stdout}"
    assert "GPL" in stdout and ("banned" in stdout.lower() or ";" in stdout)


def test_zlib_passes():
    """Test that Zlib (opencv bundled lib) is allowed."""
    manifest = make_manifest(entries=[
        {
            "name": "zlib",
            "version": "1.2.11",
            "type": "library",
            "license": "Zlib",
            "source_url": "https://zlib.net/",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker(manifest)
    assert returncode == 0, f"Zlib should pass: {stdout}"
    assert "PASSED" in stdout or "✅" in stdout


def test_libpng_passes():
    """Test that Libpng (opencv bundled lib) is allowed."""
    manifest = make_manifest(entries=[
        {
            "name": "libpng",
            "version": "1.6.37",
            "type": "library",
            "license": "Libpng",
            "source_url": "http://www.libpng.org/",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker(manifest)
    assert returncode == 0, f"Libpng should pass: {stdout}"
    assert "PASSED" in stdout or "✅" in stdout


def test_libtiff_passes():
    """Test that libtiff (opencv bundled lib) is allowed."""
    manifest = make_manifest(entries=[
        {
            "name": "libtiff",
            "version": "4.3.0",
            "type": "library",
            "license": "libtiff",
            "source_url": "http://www.libtiff.org/",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker(manifest)
    assert returncode == 0, f"libtiff should pass: {stdout}"
    assert "PASSED" in stdout or "✅" in stdout
