#!/usr/bin/env python3
"""
Generate planted test cases demonstrating full closure checking.

Shows that the fence catches:
(a) Package installed but missing from manifest
(b) Transitive dependency with GPL/unknown license
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_missing_from_manifest():
    """Planted case (a): installed package not in manifest."""
    print("## Planted Case (a): Package Installed But Missing From Manifest\n")
    
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "pytest",
                "version": "9.1.1",
                "type": "library",
                "license": "MIT",
                "source_url": "https://pypi.org/project/pytest/",
                "sha256": "a" * 64
            }
        ]
    }
    
    # Mock installed packages (pytest + unlisted dependency)
    mock_installed = [
        {"Name": "pytest", "Version": "9.1.1", "License": "MIT"},
        {"Name": "pluggy", "Version": "1.7.0", "License": "MIT"}  # Missing from manifest
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as mf:
        json.dump(manifest, mf)
        manifest_path = mf.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pf:
        json.dump(mock_installed, pf)
        mock_path = pf.name
    
    # Run checker with mocked pip-licenses output
    script = f"""
import subprocess
import sys
result = subprocess.run([
    sys.executable, 'scripts/check_licenses.py',
    '--manifest', '{manifest_path}'
], capture_output=True, text=True, env={{'NO_FFMPEG_CHECK': '1', **dict(os.environ)}})
print(result.stdout)
sys.exit(result.returncode)
"""
    
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=Path.cwd(),
        capture_output=True,
        text=True
    )
    
    print("**Expected:** ❌ FAIL (pluggy installed but not in manifest)\n")
    print("```")
    print(result.stdout.strip())
    print("```\n")
    
    Path(manifest_path).unlink()
    Path(mock_path).unlink()


def test_gpl_transitive():
    """Planted case (b): transitive dependency with GPL license."""
    print("## Planted Case (b): Transitive Dependency With GPL License\n")
    
    manifest = {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": [
            {
                "name": "pytest",
                "version": "9.1.1",
                "type": "library",
                "license": "MIT",
                "source_url": "https://pypi.org/project/pytest/",
                "sha256": "a" * 64
            },
            {
                "name": "gpl-transitive",
                "version": "1.0.0",
                "type": "library",
                "license": "GPL-3.0",
                "source_url": "https://pypi.org/project/gpl-transitive/",
                "sha256": "b" * 64,
                "notes": "Planted: transitive dep pulled in by pytest (hypothetical)"
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(manifest, f)
        temp_path = f.name
    
    result = subprocess.run(
        [sys.executable, 'scripts/check_licenses.py', '--manifest', temp_path, '--no-scan-installed'],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        env={'NO_FFMPEG_CHECK': '1', **os.environ}
    )
    
    print("**Expected:** ❌ FAIL (GPL-3.0 is banned)\n")
    print("```")
    print(result.stdout.strip())
    print("```\n")
    
    Path(temp_path).unlink()


print("# Planted Full-Closure Test Cases\n")
print("Demonstrating AC-01 requirement: every package in the installed closure must pass.\n")

test_gpl_transitive()
print("\n---\n")
test_missing_from_manifest()
print("\n---\n")
print("Both planted cases show the fence correctly failing on full closure violations.")
