#!/usr/bin/env python3
"""
License fence checker for trace-search.

Validates the license manifest and checks for:
- Banned licenses (AGPL, GPL, NC, unknown/unrecognized)
- Banned component names (Ultralytics, boxmot, etc.)

Exit codes:
  0 - All checks passed
  1 - Validation errors or banned items found
  2 - Usage error
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Banned license patterns (case-insensitive)
BANNED_LICENSE_PATTERNS = [
    r'\bAGPL\b',
    r'\bGPL\b',
    r'\bGNU\s+General\s+Public\s+License\b',
    r'\bGNU\s+Affero\s+General\s+Public\s+License\b',
    r'\bNC\b',
    r'\bNon-Commercial\b',
    r'\bNonCommercial\b',
    r'\bunknown\b',
    r'\bunrecognized\b',
    r'\bNOASSERTION\b',
]

# Allowed licenses (case-insensitive exact match or SPDX-style)
ALLOWED_LICENSES = {
    'MIT',
    'Apache-2.0',
    'Apache 2.0',
    'BSD',
    'BSD-2-Clause',
    'BSD-3-Clause',
    'ISC',
    'CC0',
    'CC-BY-4.0',
    'CC BY 4.0',
    'LGPL-2.1',
    'LGPL-3.0',
    'PSF',
    'Python Software Foundation',
    'MPL-2.0',
    'Unlicense',
}

# Banned component names (case-insensitive substring match)
BANNED_COMPONENTS = [
    'ultralytics',
    'boxmot',
    'insightface',
    'arcface',
    'opengait',
    'market-1501',
    'market1501',
    'msmt17',
    'dukemtmc',
    'duke-mtmc',
    'cuhk03',
    'casia-b',
    'casia_b',
    'grew',
    'gait3d',
    'objects365',
]

# Dataset/model names that are research-only (case-insensitive)
RESEARCH_ONLY_DATASETS = [
    'market-1501',
    'market1501',
    'msmt17',
    'dukemtmc',
    'duke-mtmc',
    'cuhk03',
    'casia-b',
    'casia_b',
    'grew',
    'gait3d',
]


def load_manifest(manifest_path: Path) -> dict[str, any]:
    """Load and parse the license manifest JSON."""
    try:
        with manifest_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in manifest: {e}", file=sys.stderr)
        sys.exit(1)


def validate_manifest_structure(manifest: dict[str, any]) -> list[str]:
    """Basic validation of manifest structure."""
    errors = []

    if 'manifest_version' not in manifest:
        errors.append("Missing required field: manifest_version")

    if 'last_updated_ist' not in manifest:
        errors.append("Missing required field: last_updated_ist")

    if 'entries' not in manifest:
        errors.append("Missing required field: entries")
        return errors

    if not isinstance(manifest['entries'], list):
        errors.append("Field 'entries' must be an array")
        return errors

    for i, entry in enumerate(manifest['entries']):
        required_fields = ['name', 'version', 'type', 'license', 'source_url', 'sha256']
        for field in required_fields:
            if field not in entry:
                entry_name = entry.get('name', 'unknown')
                errors.append(
                    f"Entry {i} ({entry_name}): missing required field '{field}'"
                )

        # Validate SHA-256 format
        if 'sha256' in entry:
            sha256 = entry['sha256']
            if not re.match(r'^[a-f0-9]{64}$', sha256):
                errors.append(f"Entry {i} ({entry.get('name', 'unknown')}): invalid SHA-256 format")

        # Validate type
        valid_types = ['library', 'model', 'weight', 'dataset', 'clip']
        if 'type' in entry and entry['type'] not in valid_types:
            entry_name = entry.get('name', 'unknown')
            errors.append(
                f"Entry {i} ({entry_name}): invalid type '{entry['type']}'"
            )

        # Clip-specific validation
        if entry.get('type') == 'clip':
            entry_name = entry.get('name', 'unknown')
            if 'license_version' not in entry:
                errors.append(
                    f"Entry {i} ({entry_name}): clips must have 'license_version'"
                )
            if 'date_checked_ist' not in entry:
                errors.append(
                    f"Entry {i} ({entry_name}): clips must have 'date_checked_ist'"
                )
            if 'phash' not in entry:
                errors.append(
                    f"Entry {i} ({entry_name}): clips must have 'phash'"
                )
            if 'use' not in entry:
                errors.append(
                    f"Entry {i} ({entry_name}): clips must have 'use' tag"
                )

    return errors


def check_banned_licenses(manifest: dict[str, any]) -> tuple[list[str], list[str]]:
    """Check for banned license patterns and unknown licenses."""
    violations = []
    warnings = []

    for entry in manifest.get('entries', []):
        license_str = entry.get('license', '').strip()
        name = entry.get('name', 'unknown')

        if not license_str:
            violations.append(f"{name}: empty license field")
            continue

        # Check against banned patterns
        for pattern in BANNED_LICENSE_PATTERNS:
            if re.search(pattern, license_str, re.IGNORECASE):
                violations.append(
                    f"{name}: banned license '{license_str}' "
                    f"(matches pattern: {pattern})"
                )
                break
        else:
            # Check if license is in allowed list
            license_normalized = license_str.upper().strip()
            allowed_normalized = {lic.upper() for lic in ALLOWED_LICENSES}

            # Exact match or starts with allowed prefix
            if not any(license_normalized == allowed or license_normalized.startswith(allowed)
                      for allowed in allowed_normalized):
                # Try to detect if it's LGPL (which is allowed for FFmpeg)
                if 'LGPL' in license_normalized and 'GPL' in license_normalized:
                    # LGPL is okay, it contains GPL but is different
                    continue

                warnings.append(f"{name}: license '{license_str}' not in pre-approved list. "
                              "Verify it's compatible and not GPL/AGPL/NC.")

    return violations, warnings


def check_banned_components(manifest: dict[str, any]) -> list[str]:
    """Check for banned component names."""
    violations = []

    for entry in manifest.get('entries', []):
        name = entry.get('name', '').lower()
        source_url = entry.get('source_url', '').lower()
        notes = entry.get('notes', '').lower()

        for banned in BANNED_COMPONENTS:
            if (banned in name or
                banned in source_url or
                banned in notes):
                violations.append(
                    f"{entry.get('name', 'unknown')}: contains banned component '{banned}'"
                )
                break

    return violations


def main():
    parser = argparse.ArgumentParser(
        description='Validate license manifest and check for banned items.'
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('licenses/manifest.json'),
        help='Path to license manifest (default: licenses/manifest.json)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors'
    )

    args = parser.parse_args()

    print(f"Checking license manifest: {args.manifest}")
    print()

    manifest = load_manifest(args.manifest)

    has_errors = False

    # Structural validation
    print("=== Validating manifest structure ===")
    structure_errors = validate_manifest_structure(manifest)
    if structure_errors:
        has_errors = True
        for error in structure_errors:
            print(f"ERROR: {error}")
    else:
        print("✓ Manifest structure is valid")
    print()

    # License checks
    print("=== Checking for banned licenses ===")
    license_violations, license_warnings = check_banned_licenses(manifest)
    if license_violations:
        has_errors = True
        for violation in license_violations:
            print(f"ERROR: {violation}")
    else:
        print("✓ No banned licenses found")

    if license_warnings:
        print()
        for warning in license_warnings:
            print(f"WARNING: {warning}")
        if args.strict:
            has_errors = True
    print()

    # Component name checks
    print("=== Checking for banned components ===")
    component_violations = check_banned_components(manifest)
    if component_violations:
        has_errors = True
        for violation in component_violations:
            print(f"ERROR: {violation}")
    else:
        print("✓ No banned components found")
    print()

    # Summary
    total_entries = len(manifest.get('entries', []))
    print(f"Summary: Checked {total_entries} entries")

    if has_errors:
        print("\n❌ License fence check FAILED")
        sys.exit(1)
    else:
        print("\n✅ License fence check PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()
