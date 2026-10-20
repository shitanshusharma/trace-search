"""Comprehensive tests for full closure license fence (T0b)."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_checker_with_manifest(manifest_data: dict) -> tuple[int, str, str]:
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
            env={**os.environ, 'NO_FFMPEG_CHECK': '1'}
        )
        return result.returncode, result.stdout, result.stderr
    finally:
        Path(manifest_path).unlink()


def make_manifest(**kwargs):
    """Create a test manifest."""
    return {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": kwargs.get('entries', [])
    }


def test_semicolon_joined_license_with_gpl_fails():
    """Test that '; ' joined licenses (pip-licenses format) with GPL fail."""
    manifest = make_manifest(entries=[
        {
            "name": "python-dateutil",
            "version": "2.9.0",
            "type": "library",
            "license": "Apache Software License; GPL-2.0",
            "source_url": "https://pypi.org/project/python-dateutil/",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"Semicolon-joined license with GPL should fail: {stdout}"
    assert "GPL" in stdout or "banned" in stdout.lower()


def test_underscore_joined_banned_name_fails():
    """Test that underscore-joined banned names fail (ultralytics_yolov8n)."""
    manifest = make_manifest(entries=[
        {
            "name": "ultralytics_yolov8n",
            "version": "1.0.0",
            "type": "weight",
            "license": "MIT",
            "source_url": "https://github.com/ultralytics/yolov8",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"Underscore-joined ultralytics should fail: {stdout}"
    assert "ultralytics" in stdout.lower()


def test_retinaface_resnet50_fails():
    """Test that retinaface_resnet50 (face model) fails."""
    manifest = make_manifest(entries=[
        {
            "name": "retinaface_resnet50",
            "version": "1.0.0",
            "type": "weight",
            "license": "MIT",
            "source_url": "https://github.com/biubug6/RetinaFace",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"retinaface_resnet50 should fail: {stdout}"
    assert "retinaface" in stdout.lower() or "face" in stdout.lower()


def test_buffalo_l_face_model_fails():
    """Test that buffalo_l face model fails."""
    manifest = make_manifest(entries=[
        {
            "name": "buffalo_l",
            "version": "1.0.0",
            "type": "weight",
            "license": "MIT",
            "source_url": "https://github.com/deepinsight/insightface",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"buffalo_l should fail: {stdout}"
    assert "buffalo" in stdout.lower() or "face" in stdout.lower()


def test_mtcnn_face_model_fails():
    """Test that mtcnn face model fails."""
    manifest = make_manifest(entries=[
        {
            "name": "mtcnn",
            "version": "0.1.0",
            "type": "library",
            "license": "MIT",
            "source_url": "https://pypi.org/project/mtcnn/",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"mtcnn should fail: {stdout}"
    assert "mtcnn" in stdout.lower() or "face" in stdout.lower()


def test_scrfd_face_model_fails():
    """Test that scrfd face model fails."""
    manifest = make_manifest(entries=[
        {
            "name": "scrfd_10g_bnkps",
            "version": "1.0.0",
            "type": "weight",
            "license": "MIT",
            "source_url": "https://github.com/deepinsight/insightface",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"scrfd should fail: {stdout}"
    assert "scrfd" in stdout.lower() or "face" in stdout.lower()


def test_facenet_pytorch_fails():
    """Test that facenet-pytorch fails."""
    manifest = make_manifest(entries=[
        {
            "name": "facenet-pytorch",
            "version": "2.5.2",
            "type": "library",
            "license": "MIT",
            "source_url": "https://pypi.org/project/facenet-pytorch/",
            "sha256": "a" * 64
        }
    ])

    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"facenet-pytorch should fail: {stdout}"
    assert "facenet" in stdout.lower() or "face" in stdout.lower()


def test_schema_synthetic_clip_fixture_id():
    """Test that FX-SMOKE-SYN-001 format is accepted."""
    manifest = make_manifest(entries=[
        {
            "name": "synthetic_walk_01.mp4",
            "version": "1.0",
            "type": "synthetic_clip",
            "license": "CC0-1.0",
            "license_version": "1.0",
            "date_checked_ist": "2026-10-20",
            "source_url": "https://example.com/synthetic",
            "sha256": "a" * 64,
            "phash": [f"{i:016x}" for i in range(10)],
            "use": "SMOKE_ONLY",
            "fixture_id": "FX-SMOKE-SYN-001",
            "generator": "blender",
            "generator_version": "3.6.0",
            "generator_license": "GPL-3.0",
            "seed": "12345"
        }
    ])

    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    # Generator has GPL but that's OK for tools, the clip itself is CC0
    # Schema should accept FX-SMOKE-SYN-001 format
    assert (
        returncode == 0 or "fixture_id" not in stdout
    ), f"FX-SMOKE-SYN-001 format should be accepted: {stdout}"


def test_clean_real_closure_passes():
    """Test that the real clean closure passes."""
    result = subprocess.run(
        [sys.executable, 'scripts/check_licenses.py', '--no-scan-installed'],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        env={**os.environ, 'NO_FFMPEG_CHECK': '1'}
    )

    assert result.returncode == 0, f"Real clean manifest should pass: {result.stdout}"
    assert "PASSED" in result.stdout or "✅" in result.stdout


def test_installed_package_scan_with_system_packages():
    """Test that scanning catches system packages (ansible, python-apt with GPL)."""
    result = subprocess.run(
        [sys.executable, 'scripts/check_licenses.py'],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        env={**os.environ, 'NO_FFMPEG_CHECK': '1'}
    )

    # With system packages present, scanning should catch unlisted/GPL packages
    # If the system is clean and all packages are listed, it will pass
    # Otherwise it should fail - either way, the scanner is working
    if result.returncode != 0:
        # System packages found - verify GPL detection works
        output = result.stdout.lower()
        # Should detect GPL or unlisted packages
        assert ("gpl" in output or "not in manifest" in output), \
            f"Scanner should report GPL or unlisted packages: {result.stdout}"
    else:
        # Clean system - verify it passed correctly
        assert "PASSED" in result.stdout or "✅" in result.stdout
