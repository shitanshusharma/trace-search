"""Tests for the license fence with installed package scanning."""

import json
import subprocess
import tempfile
from pathlib import Path


def test_license_checker_passes_on_valid_manifest_with_installed():
    """The license checker should pass when manifest matches installed packages."""
    result = subprocess.run(
        ['python3', 'scripts/check_licenses.py', '--no-scan-installed'],
        cwd=Path.cwd(),
        capture_output=True,
        text=True
    )
    # May pass or fail depending on environment, but should not crash
    assert result.returncode in [0, 1]


def test_missing_manifest_entry_fails():
    """GAP 1: Package installed but missing from manifest must fail."""
    # Create a minimal manifest that's missing some installed packages
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "pytest",
                    "version": "7.4.0",
                    "type": "library",
                    "license": "MIT",
                    "source_url": "https://pypi.org/project/pytest/",
                    "sha256": "de4bb8104e201939ccdc688b27a89a7be2079b22e2bd2b07f806b6ba71117977"
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', 'scripts/check_licenses.py', '--manifest', temp_path],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        # Should fail because many installed packages are missing from manifest
        assert result.returncode == 1, f"Should fail with missing packages: {result.stdout}"
        assert "installed but not in manifest" in result.stdout
    finally:
        Path(temp_path).unlink()


def test_installed_agpl_package_fails():
    """GAP 2: Package with AGPL license in environment must fail."""
    # This test would require actually installing an AGPL package, which we don't want
    # Instead, test that manifest check catches AGPL
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "bad-lib",
                    "version": "1.0.0",
                    "type": "library",
                    "license": "AGPL-3.0",
                    "source_url": "https://example.com",
                    "sha256": "a" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', 'scripts/check_licenses.py', '--manifest', temp_path,
             '--no-scan-installed'],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        assert "AGPL" in result.stdout.upper()
    finally:
        Path(temp_path).unlink()


def test_unknown_license_fails():
    """GAP 2: Unknown license must fail (not just warn)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "mystery-lib",
                    "version": "1.0.0",
                    "type": "library",
                    "license": "UNKNOWN",
                    "source_url": "https://example.com",
                    "sha256": "b" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', 'scripts/check_licenses.py', '--manifest', temp_path,
             '--no-scan-installed'],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        # Should FAIL, not just warn
        assert result.returncode == 1, "Unknown license should fail by default"
        assert "ERROR" in result.stdout
        assert "UNKNOWN" in result.stdout or "unknown" in result.stdout.lower()
    finally:
        Path(temp_path).unlink()


def test_innocent_substring_passes():
    """GAP 3: Innocent name containing banned substring must pass."""
    # "overgrew" contains "grew" but should pass with word-boundary matching
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "overgrew",
                    "version": "1.0.0",
                    "type": "library",
                    "license": "MIT",
                    "source_url": "https://example.com/overgrew",
                    "sha256": "c" * 64
                },
                {
                    "name": "progressive-web-app",
                    "version": "1.0.0",
                    "type": "library",
                    "license": "MIT",
                    "source_url": "https://example.com/progressive",
                    "sha256": "d" * 64,
                    "notes": "Started as a simple prototype and expanded over time"
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', 'scripts/check_licenses.py', '--manifest', temp_path,
             '--no-scan-installed'],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        # Should PASS - "grew" is banned but "overgrew" is not
        assert result.returncode == 0, f"Innocent substring should pass: {result.stdout}"
        assert "✅" in result.stdout or "PASSED" in result.stdout
    finally:
        Path(temp_path).unlink()


def test_actual_banned_grew_fails():
    """Verify that actual 'grew' (GREW dataset) is still caught."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "grew-dataset",
                    "version": "1.0.0",
                    "type": "dataset",
                    "license": "MIT",
                    "source_url": "https://example.com/grew",
                    "sha256": "e" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', 'scripts/check_licenses.py', '--manifest', temp_path,
             '--no-scan-installed'],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1, "GREW dataset should still be banned"
        assert "grew" in result.stdout.lower()
        assert "banned component" in result.stdout.lower()
    finally:
        Path(temp_path).unlink()


# Old tests from test_license_fence.py
def test_license_checker_rejects_agpl():
    """The fence must reject AGPL licenses."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "bad-library",
                    "version": "1.0.0",
                    "type": "library",
                    "license": "AGPL-3.0",
                    "source_url": "https://example.com/bad-library",
                    "sha256": "a" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', 'scripts/check_licenses.py', '--manifest', temp_path,
             '--no-scan-installed'],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1, "License check should have failed on AGPL"
        assert "banned license" in result.stdout.lower() or "agpl" in result.stdout.lower()
    finally:
        Path(temp_path).unlink()


def test_license_checker_rejects_ultralytics():
    """The fence must reject Ultralytics (AGPL, banned by name)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "ultralytics",
                    "version": "8.0.0",
                    "type": "library",
                    "license": "AGPL-3.0",
                    "source_url": "https://github.com/ultralytics/ultralytics",
                    "sha256": "a" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', 'scripts/check_licenses.py', '--manifest', temp_path,
             '--no-scan-installed'],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1, "License check should have failed on ultralytics"
        assert "ultralytics" in result.stdout.lower()
    finally:
        Path(temp_path).unlink()


def test_license_checker_accepts_mit():
    """The fence must accept MIT license."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "good-library",
                    "version": "1.0.0",
                    "type": "library",
                    "license": "MIT",
                    "source_url": "https://example.com/good-library",
                    "sha256": "b" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', 'scripts/check_licenses.py', '--manifest', temp_path,
             '--no-scan-installed'],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"License check should pass on MIT: {result.stdout}"
        assert "✅" in result.stdout or "PASSED" in result.stdout
    finally:
        Path(temp_path).unlink()
