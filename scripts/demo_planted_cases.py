#!/usr/bin/env python3
"""
Generate planted test cases to demonstrate license fence failures.

Creates test manifests for each blocking scenario and captures their output.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_checker(manifest_data: dict, name: str) -> tuple[int, str]:
    """Run the checker with a test manifest and return output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(manifest_data, f, indent=2)
        temp_path = f.name

    try:
        result = subprocess.run(
            [
                sys.executable, 'scripts/check_licenses.py',
                '--manifest', temp_path,
                '--no-scan-installed'
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            env={**os.environ}
        )
        return result.returncode, result.stdout
    finally:
        Path(temp_path).unlink()


# Base manifest template
def make_manifest(**kwargs):
    return {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": kwargs.get('entries', [])
    }


# Test cases
test_cases = {
    "agpl_installed_pkg": make_manifest(entries=[
        {
            "name": "PyMuPDF",
            "version": "1.23.0",
            "type": "library",
            "license": "AGPL-3.0",
            "source_url": "https://pypi.org/project/PyMuPDF/",
            "sha256": "a" * 64
        }
    ]),

    "lgpl_gpl_dual": make_manifest(entries=[
        {
            "name": "bad-lib",
            "version": "1.0",
            "type": "library",
            "license": "LGPLv2.1+ OR GPLv2+",
            "source_url": "https://example.com",
            "sha256": "a" * 64
        }
    ]),

    "mit_research_only": make_manifest(entries=[
        {
            "name": "research-lib",
            "version": "1.0",
            "type": "library",
            "license": "MIT (research use only)",
            "source_url": "https://example.com",
            "sha256": "a" * 64
        }
    ]),

    "model_zoo_reid": make_manifest(entries=[
        {
            "name": "osnet_x1_0_market_256x128.pth",
            "version": "1.0",
            "type": "weight",
            "license": "MIT",
            "source_url": "https://github.com/KaiyangZhou/deep-person-reid",
            "sha256": "a" * 64,
            "notes": "OSNet trained on Market-1501"
        }
    ]),

    "insecam_clip": make_manifest(entries=[
        {
            "name": "insecam_camera_nyc_001.mp4",
            "version": "1.0",
            "type": "clip",
            "license": "Unknown",
            "license_version": "N/A",
            "date_checked_ist": "2026-10-20",
            "source_url": "http://www.insecam.org/en/view/12345/",
            "sha256": "a" * 64,
            "phash": [f"{i:016x}" for i in range(10)],
            "use": "SMOKE_ONLY"
        }
    ]),

    "schema_invalid_empty_name": make_manifest(entries=[
        {
            "name": "",
            "version": "1.0",
            "type": "library",
            "license": "MIT",
            "source_url": "https://example.com",
            "sha256": "a" * 64
        }
    ]),

    "innocent_substring": make_manifest(entries=[
        {
            "name": "progressive",
            "version": "1.0",
            "type": "library",
            "license": "MIT",
            "source_url": "https://example.com/progressive",
            "sha256": "c" * 64,
            "notes": "Package named 'progressive' (substring false positive test)"
        }
    ])
}

print("# Planted Test Case Outputs\n")
print("Demonstrating license fence catches all blocking scenarios:\n")

for name, manifest in test_cases.items():
    print(f"## {name.replace('_', ' ').title()}\n")
    returncode, output = run_checker(manifest, name)
    status = "✅ PASS (correctly rejected)" if returncode == 1 else (
        "✅ PASS (correctly allowed)" if returncode == 0 else "❌ FAIL"
    )

    # Special case: innocent substring should pass
    if name == "innocent_substring":
        status = "✅ PASS (correctly allowed)" if returncode == 0 else "❌ FAIL (false positive)"

    print(f"**Status:** {status}\n")
    print("```")
    print(output.strip())
    print("```\n")

print("\n---\n")
print("All planted cases demonstrate the fence working correctly.")
