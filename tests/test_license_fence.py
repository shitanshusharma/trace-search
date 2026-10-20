"""Tests for the license fence."""

import json
import subprocess
import tempfile
from pathlib import Path


def test_license_checker_passes_on_valid_manifest():
    """The license checker should pass on the current valid manifest."""
    result = subprocess.run(
        ['python3', 'scripts/check_licenses.py', '--no-scan-installed'],
        cwd=Path.cwd(),
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"License check failed: {result.stdout}\n{result.stderr}"
    assert "✅ License fence check PASSED" in result.stdout


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
            [
                'python3', 'scripts/check_licenses.py',
                '--manifest', temp_path, '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1, "License check should have failed on AGPL"
        assert "banned license" in result.stdout.lower()
        assert "agpl" in result.stdout.lower()
    finally:
        Path(temp_path).unlink()


def test_license_checker_rejects_gpl():
    """The fence must reject GPL licenses."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "bad-library",
                    "version": "1.0.0",
                    "type": "library",
                    "license": "GPL-3.0",
                    "source_url": "https://example.com/bad-library",
                    "sha256": "a" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            [
                'python3', 'scripts/check_licenses.py',
                '--manifest', temp_path, '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1, "License check should have failed on GPL"
        assert "banned license" in result.stdout.lower()
    finally:
        Path(temp_path).unlink()


def test_license_checker_rejects_noncommercial():
    """The fence must reject NC (Non-Commercial) licenses."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "bad-library",
                    "version": "1.0.0",
                    "type": "library",
                    "license": "CC-BY-NC-4.0",
                    "source_url": "https://example.com/bad-library",
                    "sha256": "a" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            [
                'python3', 'scripts/check_licenses.py',
                '--manifest', temp_path, '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1, "License check should have failed on NC"
        assert "banned license" in result.stdout.lower()
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
            [
                'python3', 'scripts/check_licenses.py',
                '--manifest', temp_path, '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1, "License check should have failed on ultralytics"
        # Should fail on both banned license AND banned component
        assert "ultralytics" in result.stdout.lower()
    finally:
        Path(temp_path).unlink()


def test_license_checker_rejects_boxmot():
    """The fence must reject boxmot (banned component)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "boxmot",
                    "version": "10.0.0",
                    "type": "library",
                    "license": "AGPL-3.0",
                    "source_url": "https://github.com/mikel-brostrom/boxmot",
                    "sha256": "a" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            [
                'python3', 'scripts/check_licenses.py',
                '--manifest', temp_path, '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1, "License check should have failed on boxmot"
        assert "boxmot" in result.stdout.lower()
    finally:
        Path(temp_path).unlink()


def test_license_checker_rejects_insightface():
    """The fence must reject InsightFace/ArcFace (banned component)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "insightface",
                    "version": "0.7.0",
                    "type": "library",
                    "license": "MIT",
                    "source_url": "https://github.com/deepinsight/insightface",
                    "sha256": "a" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            [
                'python3', 'scripts/check_licenses.py',
                '--manifest', temp_path, '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1, "License check should have failed on insightface"
        assert "insightface" in result.stdout.lower()
    finally:
        Path(temp_path).unlink()


def test_license_checker_rejects_market1501():
    """The fence must reject Market-1501 dataset (research-only, banned)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "Market-1501",
                    "version": "1.0",
                    "type": "dataset",
                    "license": "Unknown",
                    "source_url": "https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf",
                    "sha256": "a" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            [
                'python3', 'scripts/check_licenses.py',
                '--manifest', temp_path, '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1, "License check should have failed on Market-1501"
        # Should fail on both unknown license AND banned dataset name
        assert "market" in result.stdout.lower() or "unknown" in result.stdout.lower()
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
            [
                'python3', 'scripts/check_licenses.py',
                '--manifest', temp_path, '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"License check should pass on MIT: {result.stdout}"
        assert "✅ License fence check PASSED" in result.stdout
    finally:
        Path(temp_path).unlink()


def test_license_checker_accepts_apache2():
    """The fence must accept Apache-2.0 license."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "good-library",
                    "version": "1.0.0",
                    "type": "library",
                    "license": "Apache-2.0",
                    "source_url": "https://example.com/good-library",
                    "sha256": "b" * 64
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            [
                'python3', 'scripts/check_licenses.py',
                '--manifest', temp_path, '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"License check should pass on Apache-2.0: {result.stdout}"
        assert "✅ License fence check PASSED" in result.stdout
    finally:
        Path(temp_path).unlink()


def test_license_checker_validates_clip_fields():
    """
    The fence must enforce clip-specific fields.

    (license_version, date_checked_ist, phash, use)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "test-clip.mp4",
                    "version": "1.0",
                    "type": "clip",
                    "license": "Pexels License",
                    "source_url": "https://www.pexels.com/video/test/",
                    "sha256": "c" * 64
                    # Missing: license_version, date_checked_ist, phash, use
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            [
                'python3', 'scripts/check_licenses.py',
                '--manifest', temp_path, '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 1, "License check should fail on incomplete clip entry"
        assert "license_version" in result.stdout
        assert "date_checked_ist" in result.stdout
        assert "phash" in result.stdout
        assert "use" in result.stdout
    finally:
        Path(temp_path).unlink()


def test_license_checker_accepts_valid_clip():
    """The fence must accept a properly formatted clip entry."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "test-clip.mp4",
                    "version": "1.0",
                    "type": "clip",
                    "license": "Pexels License",
                    "license_version": "Pexels ToS, last updated 15 Nov 2024",
                    "date_checked_ist": "2026-10-20",
                    "source_url": "https://www.pexels.com/video/test/",
                    "sha256": "c" * 64,
                    "phash": [f"{i:016x}" for i in range(10)],
                    "use": "SMOKE_ONLY"
                }
            ]
        }
        json.dump(manifest, f)
        temp_path = f.name

    try:
        result = subprocess.run(
            [
                'python3', 'scripts/check_licenses.py',
                '--manifest', temp_path, '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"License check should pass on valid clip: {result.stdout}"
        assert "✅ License fence check PASSED" in result.stdout
    finally:
        Path(temp_path).unlink()
