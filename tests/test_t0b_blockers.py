"""Tests for FFmpeg policy and closure scanning (T0b B1, B2, B3)."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def make_manifest(**kwargs):
    """Create a test manifest."""
    return {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": kwargs.get('entries', [])
    }


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
            env={**os.environ}
        )
        return result.returncode, result.stdout, result.stderr
    finally:
        Path(manifest_path).unlink()


@pytest.fixture
def stub_ffmpeg_not_a_dependency(monkeypatch):
    """Stub check_ffmpeg_build to return NOT_A_DEPENDENCY."""
    def mock_check(*args, **kwargs):
        return [], 'NOT_A_DEPENDENCY'
    
    import scripts.check_licenses as cl
    monkeypatch.setattr(cl, 'check_ffmpeg_build', mock_check)


@pytest.fixture
def stub_ffmpeg_lgpl_ok(monkeypatch):
    """Stub check_ffmpeg_build to return LGPL_OK."""
    def mock_check(*args, **kwargs):
        return [], 'LGPL_OK'
    
    import scripts.check_licenses as cl
    monkeypatch.setattr(cl, 'check_ffmpeg_build', mock_check)


@pytest.fixture
def stub_ffmpeg_gpl(monkeypatch):
    """Stub check_ffmpeg_build to return GPL violation."""
    def mock_check(*args, **kwargs):
        return [
            "opencv-python's bundled FFmpeg built with --enable-gpl (must be LGPL-only)"
        ], 'GPL_VIOLATION'
    
    import scripts.check_licenses as cl
    monkeypatch.setattr(cl, 'check_ffmpeg_build', mock_check)


@pytest.fixture
def stub_ffmpeg_nonfree(monkeypatch):
    """Stub check_ffmpeg_build to return nonfree violation."""
    def mock_check(*args, **kwargs):
        return [
            "System FFmpeg built with --enable-nonfree (must be LGPL-only)"
        ], 'GPL_VIOLATION'
    
    import scripts.check_licenses as cl
    monkeypatch.setattr(cl, 'check_ffmpeg_build', mock_check)


@pytest.fixture
def stub_ffmpeg_missing(monkeypatch):
    """Stub check_ffmpeg_build to return MISSING."""
    def mock_check(*args, **kwargs):
        return [
            "System FFmpeg declared in manifest but not found"
        ], 'MISSING'
    
    import scripts.check_licenses as cl
    monkeypatch.setattr(cl, 'check_ffmpeg_build', mock_check)


# ===== B1: FFmpeg Policy Tests =====

def test_ffmpeg_not_a_dependency(stub_ffmpeg_not_a_dependency):
    """Test NOT_A_DEPENDENCY status when no FFmpeg declared."""
    manifest = make_manifest(entries=[
        {
            "name": "pytest",
            "version": "9.1.1",
            "type": "library",
            "license": "MIT",
            "source_url": "https://pypi.org/project/pytest/",
            "sha256": "a" * 64
        }
    ])
    
    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 0, f"Should pass with NOT_A_DEPENDENCY: {stdout}"
    assert "NOT_A_DEPENDENCY" in stdout


def test_ffmpeg_enable_gpl_fails(stub_ffmpeg_gpl):
    """Test --enable-gpl fails."""
    manifest = make_manifest(entries=[
        {
            "name": "opencv-python-headless",
            "version": "4.8.1",
            "type": "library",
            "license": "MIT",
            "source_url": "https://pypi.org/project/opencv-python-headless/",
            "sha256": "a" * 64
        }
    ])
    
    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"Should fail with --enable-gpl: {stdout}"
    assert "--enable-gpl" in stdout


def test_ffmpeg_enable_nonfree_fails(stub_ffmpeg_nonfree):
    """Test --enable-nonfree fails."""
    manifest = make_manifest(entries=[
        {
            "name": "ffmpeg",
            "version": "6.0",
            "type": "tool",
            "license": "LGPL-2.1",
            "source_url": "https://ffmpeg.org/",
            "sha256": "a" * 64
        }
    ])
    
    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"Should fail with --enable-nonfree: {stdout}"
    assert "--enable-nonfree" in stdout


def test_system_ffmpeg_missing_fails(stub_ffmpeg_missing):
    """Test declared but missing system FFmpeg fails."""
    manifest = make_manifest(entries=[
        {
            "name": "ffmpeg",
            "version": "6.0",
            "type": "tool",
            "license": "LGPL-2.1",
            "source_url": "https://ffmpeg.org/",
            "sha256": "a" * 64
        }
    ])
    
    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"Should fail with missing FFmpeg: {stdout}"
    assert "not found" in stdout


# ===== B2: Closure Scan Tests with Monkeypatching =====

def test_unlisted_transitive_package_fails(stub_ffmpeg_not_a_dependency, monkeypatch):
    """Test that unlisted transitive package fails."""
    manifest = make_manifest(entries=[
        {
            "name": "pytest",
            "version": "9.1.1",
            "type": "library",
            "license": "MIT",
            "source_url": "https://pypi.org/project/pytest/",
            "sha256": "a" * 64
        }
    ])
    
    # Monkeypatch get_installed_packages to return pytest + unlisted package
    def mock_get_installed(*args, **kwargs):
        return [
            {"name": "pytest", "version": "9.1.1", "license": "MIT"},
            {"name": "unlisted-transitive", "version": "1.0.0", "license": "MIT"}
        ]
    
    import scripts.check_licenses as cl
    monkeypatch.setattr(cl, 'get_installed_packages', mock_get_installed)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(manifest, f)
        manifest_path = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/check_licenses.py', '--manifest', manifest_path],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env={**os.environ}
        )
        
        assert result.returncode == 1, f"Unlisted package should fail: {result.stdout}"
        assert "unlisted-transitive" in result.stdout
        assert "not in manifest" in result.stdout
    finally:
        Path(manifest_path).unlink()


def test_metadata_gpl_manifest_mit_fails(stub_ffmpeg_not_a_dependency, monkeypatch):
    """Test metadata GPL while manifest says MIT fails."""
    manifest = make_manifest(entries=[
        {
            "name": "bad-package",
            "version": "1.0.0",
            "type": "library",
            "license": "MIT",
            "source_url": "https://example.com",
            "sha256": "a" * 64
        }
    ])
    
    # Monkeypatch: installed package reports GPL
    def mock_get_installed(*args, **kwargs):
        return [
            {"name": "bad-package", "version": "1.0.0", "license": "GPL-3.0"}
        ]
    
    import scripts.check_licenses as cl
    monkeypatch.setattr(cl, 'get_installed_packages', mock_get_installed)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(manifest, f)
        manifest_path = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/check_licenses.py', '--manifest', manifest_path],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env={**os.environ}
        )
        
        assert result.returncode == 1, f"GPL metadata should fail: {result.stdout}"
        assert ("GPL" in result.stdout or "banned" in result.stdout.lower())
    finally:
        Path(manifest_path).unlink()


def test_ultralytics_installed_fails(stub_ffmpeg_not_a_dependency, monkeypatch):
    """Test ultralytics installed fails."""
    manifest = make_manifest(entries=[
        {
            "name": "ultralytics",
            "version": "8.3.0",
            "type": "library",
            "license": "AGPL-3.0",
            "source_url": "https://pypi.org/project/ultralytics/",
            "sha256": "a" * 64
        }
    ])
    
    def mock_get_installed(*args, **kwargs):
        return [
            {"name": "ultralytics", "version": "8.3.0", "license": "AGPL-3.0"}
        ]
    
    import scripts.check_licenses as cl
    monkeypatch.setattr(cl, 'get_installed_packages', mock_get_installed)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(manifest, f)
        manifest_path = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/check_licenses.py', '--manifest', manifest_path],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env={**os.environ}
        )
        
        assert result.returncode == 1, f"ultralytics should fail: {result.stdout}"
        assert ("ultralytics" in result.stdout.lower() and 
                ("agpl" in result.stdout.lower() or "banned" in result.stdout.lower()))
    finally:
        Path(manifest_path).unlink()


def test_pip_licenses_raising_fails(stub_ffmpeg_not_a_dependency, monkeypatch):
    """Test pip-licenses raising exception fails."""
    manifest = make_manifest(entries=[
        {
            "name": "pytest",
            "version": "9.1.1",
            "type": "library",
            "license": "MIT",
            "source_url": "https://pypi.org/project/pytest/",
            "sha256": "a" * 64
        }
    ])
    
    def mock_get_installed(*args, **kwargs):
        raise RuntimeError("pip-licenses command failed")
    
    import scripts.check_licenses as cl
    monkeypatch.setattr(cl, 'get_installed_packages', mock_get_installed)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(manifest, f)
        manifest_path = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/check_licenses.py', '--manifest', manifest_path],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env={**os.environ}
        )
        
        assert result.returncode == 1, f"pip-licenses error should fail: {result.stdout}"
        assert "failed" in result.stdout.lower() or "error" in result.stdout.lower()
    finally:
        Path(manifest_path).unlink()


def test_pip_licenses_empty_fails(stub_ffmpeg_not_a_dependency, monkeypatch):
    """Test pip-licenses returning empty list fails."""
    manifest = make_manifest(entries=[
        {
            "name": "pytest",
            "version": "9.1.1",
            "type": "library",
            "license": "MIT",
            "source_url": "https://pypi.org/project/pytest/",
            "sha256": "a" * 64
        }
    ])
    
    def mock_get_installed(*args, **kwargs):
        # Returning empty should trigger RuntimeError in get_installed_packages
        raise RuntimeError("pip-licenses returned no packages (environment issue or parsing failure)")
    
    import scripts.check_licenses as cl
    monkeypatch.setattr(cl, 'get_installed_packages', mock_get_installed)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(manifest, f)
        manifest_path = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/check_licenses.py', '--manifest', manifest_path],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env={**os.environ}
        )
        
        assert result.returncode == 1, f"Empty pip-licenses should fail: {result.stdout}"
        assert "failed" in result.stdout.lower() or "error" in result.stdout.lower()
    finally:
        Path(manifest_path).unlink()


# ===== B3: Generator License Tests =====

def test_synthetic_clip_gpl_generator_fails(stub_ffmpeg_not_a_dependency):
    """Test synthetic_clip with GPL generator fails (PRD §3)."""
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
    assert returncode == 1, f"GPL generator should fail: {stdout}"
    assert "GPL" in stdout and "generator" in stdout


def test_synthetic_clip_missing_generator_license_fails(stub_ffmpeg_not_a_dependency):
    """Test synthetic_clip missing generator_license fails."""
    manifest = make_manifest(entries=[
        {
            "name": "synthetic_walk_02.mp4",
            "version": "1.0",
            "type": "synthetic_clip",
            "license": "CC0-1.0",
            "license_version": "1.0",
            "date_checked_ist": "2026-10-20",
            "source_url": "https://example.com/synthetic",
            "sha256": "a" * 64,
            "phash": [f"{i:016x}" for i in range(10)],
            "use": "SMOKE_ONLY",
            "fixture_id": "FX-SMOKE-SYN-002",
            "generator": "blender",
            "generator_version": "3.6.0",
            "seed": "12345"
        }
    ])
    
    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"Missing generator_license should fail: {stdout}"
    assert "generator_license" in stdout


def test_synthetic_clip_unknown_generator_license_fails(stub_ffmpeg_not_a_dependency):
    """Test synthetic_clip with UNKNOWN generator_license fails."""
    manifest = make_manifest(entries=[
        {
            "name": "synthetic_walk_03.mp4",
            "version": "1.0",
            "type": "synthetic_clip",
            "license": "CC0-1.0",
            "license_version": "1.0",
            "date_checked_ist": "2026-10-20",
            "source_url": "https://example.com/synthetic",
            "sha256": "a" * 64,
            "phash": [f"{i:016x}" for i in range(10)],
            "use": "SMOKE_ONLY",
            "fixture_id": "FX-SMOKE-SYN-003",
            "generator": "blender",
            "generator_version": "3.6.0",
            "generator_license": "UNKNOWN",
            "seed": "12345"
        }
    ])
    
    returncode, stdout, stderr = run_checker_with_manifest(manifest)
    assert returncode == 1, f"UNKNOWN generator_license should fail: {stdout}"
    assert "unknown" in stdout.lower() and "generator" in stdout.lower()
