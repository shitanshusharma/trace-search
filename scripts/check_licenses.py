#!/usr/bin/env python3
"""
License fence checker for trace-search.

Validates the license manifest and scans installed packages for:
- Banned licenses (AGPL, GPL, NC, unknown/unrecognized)
- Banned component names (Ultralytics, boxmot, etc.)
- Missing manifest entries for installed packages

Exit codes:
  0 - All checks passed
  1 - Validation errors or banned items found
  2 - Usage error
"""

import argparse
import json
import re
import subprocess
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
    r'^UNKNOWN$',
]

# Allowed licenses (case-insensitive exact match or SPDX-style)
ALLOWED_LICENSES = {
    'MIT',
    'MIT License',
    'Apache-2.0',
    'Apache 2.0',
    'Apache Software License',
    'BSD',
    'BSD License',
    'BSD-2-Clause',
    'BSD-3-Clause',
    'ISC',
    'ISC License (ISCL)',
    'CC0',
    'CC-BY-4.0',
    'CC BY 4.0',
    'LGPL-2.1',
    'LGPL-3.0',
    'PSF',
    'Python Software Foundation',
    'Python Software Foundation License',
    'MPL-2.0',
    'Unlicense',
    'Pexels License',  # For stock footage testing (§2a)
    'Pixabay Content License',  # For stock footage testing (§2a)
}

# Banned component names - use word boundaries to avoid false positives
# Each entry is (pattern, description) tuple
BANNED_COMPONENTS = [
    (r'\bultralytics\b', 'ultralytics'),
    (r'\bboxmot\b', 'boxmot'),
    (r'\binsightface\b', 'insightface'),
    (r'\barcface\b', 'arcface'),
    (r'\bopengait\b', 'opengait'),
    (r'\bmarket-?1501\b', 'market-1501'),
    (r'\bmsmt17\b', 'msmt17'),
    (r'\bduke-?mtmc\b', 'dukemtmc'),
    (r'\bcuhk03\b', 'cuhk03'),
    (r'\bcasia-?b\b', 'casia-b'),
    (r'\bgrew\b', 'grew'),  # Word boundary prevents matching 'overgrew', etc.
    (r'\bgait3d\b', 'gait3d'),
    (r'\bobjects365\b', 'objects365'),
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


def get_installed_packages(include_system: bool = False) -> list[dict[str, str]]:
    """
    Get list of installed packages with their licenses.

    Uses pip-licenses if available, falls back to pip list + metadata.
    Returns list of dicts with 'name', 'version', 'license' keys.

    Args:
        include_system: If False, exclude system-installed packages (default: False)
    """
    try:
        # Try pip-licenses first (faster and more reliable)
        cmd = ['pip-licenses', '--format=json']
        if include_system:
            cmd.append('--with-system')
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        packages = json.loads(result.stdout)
        # Normalize keys
        return [
            {
                'name': pkg.get('Name', ''),
                'version': pkg.get('Version', ''),
                'license': pkg.get('License', 'UNKNOWN')
            }
            for pkg in packages
        ]
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        # Fallback: use pip show on each package
        print("WARNING: pip-licenses not available, using fallback method")
        try:
            cmd = ['pip', 'list', '--format=json']
            if not include_system:
                cmd.append('--user')
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            packages_list = json.loads(result.stdout)
            packages = []
            for pkg in packages_list:
                name = pkg['name']
                version = pkg['version']
                # Get license from pip show
                show_result = subprocess.run(
                    ['pip', 'show', name],
                    capture_output=True,
                    text=True,
                    check=True
                )
                license_str = 'UNKNOWN'
                for line in show_result.stdout.split('\n'):
                    if line.startswith('License:'):
                        license_str = line.split(':', 1)[1].strip()
                        break
                packages.append({
                    'name': name,
                    'version': version,
                    'license': license_str or 'UNKNOWN'
                })
            return packages
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"ERROR: Failed to get installed packages: {e}", file=sys.stderr)
            sys.exit(2)


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
                entry_name = entry.get('name', 'unknown')
                errors.append(f"Entry {i} ({entry_name}): invalid SHA-256 format")

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


def is_license_banned(license_str: str) -> tuple[bool, str]:
    """
    Check if a license string matches banned patterns.

    Returns (is_banned, reason) tuple.
    """
    if not license_str or not license_str.strip():
        return True, "empty license"

    for pattern in BANNED_LICENSE_PATTERNS:
        if re.search(pattern, license_str, re.IGNORECASE):
            return True, f"matches banned pattern: {pattern}"

    return False, ""


def is_license_allowed(license_str: str) -> bool:
    """Check if a license is in the pre-approved list."""
    license_normalized = license_str.upper().strip()
    allowed_normalized = {lic.upper() for lic in ALLOWED_LICENSES}

    # Exact match or starts with allowed prefix
    if any(
        license_normalized == allowed or license_normalized.startswith(allowed)
        for allowed in allowed_normalized
    ):
        return True

    # Special case: LGPL is allowed even though it contains "GPL"
    if 'LGPL' in license_normalized:
        return True

    return False


def check_manifest_licenses(manifest: dict[str, any]) -> list[str]:
    """Check manifest entries for banned licenses."""
    violations = []

    for entry in manifest.get('entries', []):
        license_str = entry.get('license', '').strip()
        name = entry.get('name', 'unknown')

        is_banned, reason = is_license_banned(license_str)
        if is_banned:
            violations.append(f"{name}: banned license '{license_str}' ({reason})")
        elif not is_license_allowed(license_str):
            # Unknown/unrecognized license - fail by default (no --strict needed)
            violations.append(
                f"{name}: license '{license_str}' not in pre-approved list "
                "(verify it's compatible and not GPL/AGPL/NC)"
            )

    return violations


def check_installed_packages(
    installed: list[dict[str, str]],
    manifest: dict[str, any]
) -> tuple[list[str], list[str]]:
    """
    Check installed packages against manifest and for banned licenses.

    Returns (violations, missing_from_manifest) tuple.
    """
    violations = []
    missing = []

    # Build manifest lookup by normalized package name
    manifest_packages = {}
    for entry in manifest.get('entries', []):
        if entry.get('type') == 'library':
            # Normalize: lowercase, replace _ with -
            norm_name = entry.get('name', '').lower().replace('_', '-')
            manifest_packages[norm_name] = entry

    for pkg in installed:
        name = pkg['name']
        version = pkg['version']
        license_str = pkg['license']

        # Normalize package name for comparison
        norm_name = name.lower().replace('_', '-')

        # Check if in manifest
        if norm_name not in manifest_packages:
            missing.append(
                f"{name} ({version}): installed but not in manifest"
            )

        # Check license
        is_banned, reason = is_license_banned(license_str)
        if is_banned:
            violations.append(
                f"{name} ({version}): banned license '{license_str}' ({reason})"
            )
        elif not is_license_allowed(license_str):
            violations.append(
                f"{name} ({version}): license '{license_str}' not in pre-approved list"
            )

    return violations, missing


def check_banned_components(manifest: dict[str, any]) -> list[str]:
    """Check for banned component names using word boundaries."""
    violations = []

    for entry in manifest.get('entries', []):
        name = entry.get('name', '').lower()
        source_url = entry.get('source_url', '').lower()
        notes = entry.get('notes', '').lower()

        for pattern, banned_name in BANNED_COMPONENTS:
            if (
                re.search(pattern, name, re.IGNORECASE) or
                re.search(pattern, source_url, re.IGNORECASE) or
                re.search(pattern, notes, re.IGNORECASE)
            ):
                violations.append(
                    f"{entry.get('name', 'unknown')}: "
                    f"contains banned component '{banned_name}'"
                )
                break

    return violations


def main():
    parser = argparse.ArgumentParser(
        description='Validate license manifest and scan installed packages.'
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('licenses/manifest.json'),
        help='Path to license manifest (default: licenses/manifest.json)'
    )
    parser.add_argument(
        '--scan-installed',
        action='store_true',
        default=True,
        help='Scan installed packages (default: True)'
    )
    parser.add_argument(
        '--no-scan-installed',
        action='store_false',
        dest='scan_installed',
        help='Skip scanning installed packages'
    )
    parser.add_argument(
        '--include-system',
        action='store_true',
        help='Include system-installed packages in scan (default: False)'
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

    # Manifest license checks
    print("=== Checking manifest licenses ===")
    manifest_violations = check_manifest_licenses(manifest)
    if manifest_violations:
        has_errors = True
        for violation in manifest_violations:
            print(f"ERROR: {violation}")
    else:
        print("✓ No banned licenses in manifest")
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

    # Scan installed packages
    if args.scan_installed:
        print("=== Scanning installed packages ===")
        try:
            installed = get_installed_packages(include_system=args.include_system)
            print(f"Found {len(installed)} installed packages")

            install_violations, missing = check_installed_packages(installed, manifest)

            if install_violations:
                has_errors = True
                print()
                for violation in install_violations:
                    print(f"ERROR: {violation}")

            if missing:
                has_errors = True
                print()
                for miss in missing:
                    print(f"ERROR: {miss}")

            if not install_violations and not missing:
                print("✓ All installed packages are clean and in manifest")
        except Exception as e:
            print(f"ERROR: Failed to scan installed packages: {e}")
            has_errors = True
        print()

    # Summary
    total_entries = len(manifest.get('entries', []))
    print(f"Summary: Checked {total_entries} manifest entries")

    if has_errors:
        print("\n❌ License fence check FAILED")
        sys.exit(1)
    else:
        print("\n✅ License fence check PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()
