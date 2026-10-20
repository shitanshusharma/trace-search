"""Comprehensive tests for the enhanced license fence."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_checker(manifest_data: dict, extra_args: list = None) -> tuple[int, str]:
    """Helper to run the license checker with a test manifest."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(manifest_data, f)
        temp_path = f.name

    try:
        cmd = [
            sys.executable, 'scripts/check_licenses.py',
            '--manifest', temp_path,
            '--no-scan-installed'
        ]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            env={**os.environ, 'NO_FFMPEG_CHECK': '1'}
        )
        return result.returncode, result.stdout
    finally:
        Path(temp_path).unlink()


def test_lgpl_gpl_dual_license_fails():
    """BLOCKING 2: 'LGPLv2.1+/GPLv2+' dual license must fail."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "bad-dual-lib",
                "version": "1.0.0",
                "type": "library",
                "license": "LGPLv2.1+ OR GPLv2+",
                "source_url": "https://example.com",
                "sha256": "a" * 64
            }
        ]
    }

    returncode, output = run_checker(manifest)
    assert returncode == 1, f"Dual LGPL/GPL should fail: {output}"
    assert "GPL" in output.upper()
    assert "banned" in output.lower()


def test_gplv3_variant_fails():
    """BLOCKING 3: GPLv3/AGPLv3/GPLv2+ must fail."""
    for lic in ["GPLv3", "GPL-3.0", "AGPLv3", "AGPL-3.0-only", "GPLv2+"]:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "gpl-lib",
                    "version": "1.0.0",
                    "type": "library",
                    "license": lic,
                    "source_url": "https://example.com",
                    "sha256": "a" * 64
                }
            ]
        }

        returncode, output = run_checker(manifest)
        assert returncode == 1, f"{lic} should fail: {output}"
        assert "banned" in output.lower() or "not in allowed" in output.lower()


def test_research_use_only_fails():
    """BLOCKING 4: 'MIT (research use only)' must fail."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "research-lib",
                "version": "1.0.0",
                "type": "library",
                "license": "MIT (research use only)",
                "source_url": "https://example.com",
                "sha256": "a" * 64
            }
        ]
    }

    returncode, output = run_checker(manifest)
    assert returncode == 1, f"Research-only should fail: {output}"
    assert "research" in output.lower() or "not in allowed" in output.lower()


def test_sspl_busl_fails():
    """BLOCKING 5: SSPL, BUSL, CC-BY-SA must fail."""
    for lic in ["SSPL-1.0", "BUSL-1.1", "CC-BY-SA-4.0"]:
        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "bad-lib",
                    "version": "1.0.0",
                    "type": "library",
                    "license": lic,
                    "source_url": "https://example.com",
                    "sha256": "a" * 64
                }
            ]
        }

        returncode, output = run_checker(manifest)
        assert returncode == 1, f"{lic} should fail: {output}"


def test_model_zoo_reid_weight_fails():
    """BLOCKING 6: Model-zoo ReID weights must fail."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "osnet_x1_0_duke_256x128.pth",
                "version": "1.0",
                "type": "weight",
                "license": "MIT",
                "source_url": "https://github.com/KaiyangZhou/deep-person-reid",
                "sha256": "a" * 64,
                "notes": "OSNet trained on Duke"
            }
        ]
    }

    returncode, output = run_checker(manifest)
    assert returncode == 1, f"Model-zoo ReID weight should fail: {output}"
    assert "banned component" in output.lower()


def test_laion_clip_fails():
    """BLOCKING 6: LAION/CLIP weights must fail."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "CLIP-ViT-B-32-laion2B",
                "version": "1.0",
                "type": "weight",
                "license": "MIT",
                "source_url": "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                "sha256": "a" * 64
            }
        ]
    }

    returncode, output = run_checker(manifest)
    assert returncode == 1, f"LAION CLIP should fail: {output}"
    assert "laion" in output.lower() or "clip" in output.lower()


def test_insecam_clip_fails():
    """BLOCKING 6: Insecam clip must fail."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "insecam_camera_001.mp4",
                "version": "1.0",
                "type": "clip",
                "license": "Unknown",
                "license_version": "N/A",
                "date_checked_ist": "2026-10-20",
                "source_url": "http://www.insecam.org/en/view/12345/",
                "sha256": "a" * 64,
                "phash": ["0123456789abcdef"] * 10,
                "use": "SMOKE_ONLY"
            }
        ]
    }

    returncode, output = run_checker(manifest)
    assert returncode == 1, f"Insecam clip should fail: {output}"
    assert "insecam" in output.lower() or "unknown" in output.lower()


def test_schema_validation_empty_name_fails():
    """BLOCKING 8: Empty name must fail schema validation."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "",
                "version": "1.0",
                "type": "library",
                "license": "MIT",
                "source_url": "https://example.com",
                "sha256": "a" * 64
            }
        ]
    }

    returncode, output = run_checker(manifest)
    assert returncode == 1, f"Empty name should fail: {output}"
    assert "schema" in output.lower() or "minLength" in output


def test_schema_validation_non_url_fails():
    """BLOCKING 8: Non-URL source_url must fail."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "lib",
                "version": "1.0",
                "type": "library",
                "license": "MIT",
                "source_url": "see slack",
                "sha256": "a" * 64
            }
        ]
    }

    returncode, output = run_checker(manifest)
    assert returncode == 1, f"Non-URL source should fail: {output}"
    assert "schema" in output.lower() or "format" in output


def test_schema_validation_clip_missing_phash_fails():
    """BLOCKING 8: Clip with empty pHash must fail."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "clip.mp4",
                "version": "1.0",
                "type": "clip",
                "license": "MIT",
                "license_version": "1.0",
                "date_checked_ist": "2026-10-20",
                "source_url": "https://example.com",
                "sha256": "a" * 64,
                "phash": [],
                "use": "SMOKE_ONLY"
            }
        ]
    }

    returncode, output = run_checker(manifest)
    assert returncode == 1, f"Empty pHash should fail: {output}"
    assert "schema" in output.lower() or "minItems" in output


def test_schema_validation_empty_manifest_fails():
    """BLOCKING 8: Empty manifest must fail."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": []
    }

    returncode, output = run_checker(manifest)
    assert returncode == 1, f"Empty manifest should fail: {output}"
    assert "schema" in output.lower() or "minItems" in output


def test_innocent_substring_overgrew_passes():
    """GAP 3: 'overgrew' (contains 'grew') must pass."""
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
            }
        ]
    }

    returncode, output = run_checker(manifest)
    assert returncode == 0, f"Innocent substring should pass: {output}"
    assert "✅" in output or "PASSED" in output


def test_unlicensed_fails():
    """BLOCKING 4: UNLICENSED must fail (not prefix-matched)."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "bad-lib",
                "version": "1.0",
                "type": "library",
                "license": "UNLICENSED",
                "source_url": "https://example.com",
                "sha256": "a" * 64
            }
        ]
    }

    returncode, output = run_checker(manifest)
    assert returncode == 1, f"UNLICENSED should fail: {output}"
    assert "banned" in output.lower() or "not in allowed" in output.lower()


def test_commons_clause_fails():
    """BLOCKING 4: Commons Clause must fail."""
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "bad-lib",
                "version": "1.0",
                "type": "library",
                "license": "Apache-2.0 with Commons Clause",
                "source_url": "https://example.com",
                "sha256": "a" * 64
            }
        ]
    }

    returncode, output = run_checker(manifest)
    assert returncode == 1, f"Commons Clause should fail: {output}"
    assert "commons clause" in output.lower() or "banned" in output.lower()
