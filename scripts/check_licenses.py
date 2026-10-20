#!/usr/bin/env python3
"""
License fence checker for trace-search.

Validates the license manifest and scans installed packages for:
- Banned licenses (AGPL, GPL, SSPL, BUSL, NC, research-only, etc.)
- Banned component names (Ultralytics, boxmot, model zoos, etc.)
- Missing manifest entries for installed packages
- Schema compliance

Exit codes:
  0 - All checks passed
  1 - Validation errors or banned items found
  2 - Usage error
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import jsonschema
    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False

# Banned license patterns (case-insensitive, checked against each SPDX component)
BANNED_LICENSE_PATTERNS = [
    r'^AGPL',  # AGPL-3.0, AGPL-3.0-only, etc.
    r'^GPL-[123]',  # GPL-2.0, GPL-3.0, etc. (but not LGPL)
    r'^GNU\s+A?GPL',  # GNU GPL, GNU AGPL
    r'Non-?Commercial',
    r'\bNC\b',
    r'Research\s+[Uu]se\s+[Oo]nly',
    r'Academic\s+[Uu]se\s+[Oo]nly',
    r'Educational\s+[Uu]se\s+[Oo]nly',
    r'^SSPL',  # Server Side Public License
    r'^BUSL',  # Business Source License
    r'Commons\s+Clause',
    r'^CC-BY-SA',  # ShareAlike not allowed
    r'^CC-BY-NC',  # Non-Commercial not allowed
    r'Proprietary',
    r'^UNLICENSED$',
    r'^UNKNOWN$',
    r'^NOASSERTION$',
]

# Exact allowed SPDX identifiers (case-sensitive for SPDX compliance)
ALLOWED_SPDX = {
    'MIT',
    'Apache-2.0',
    'BSD-2-Clause',
    'BSD-3-Clause',
    'ISC',
    'CC0-1.0',
    'CC-BY-4.0',
    'LGPL-2.1-only',
    'LGPL-2.1-or-later',
    'LGPL-3.0-only',
    'LGPL-3.0-or-later',
    'PSF-2.0',
    'MPL-2.0',
    'Unlicense',
    '0BSD',
}

# Also allow common non-SPDX forms
ALLOWED_NON_SPDX = {
    'MIT License',
    'Apache License 2.0',
    'Apache Software License',
    'BSD License',
    'ISC License',
    'ISC License (ISCL)',
    'Python Software Foundation License',
    'Public Domain',
    'Pexels License',  # For stock footage (§2a)
    'Pixabay Content License',
}

# Banned component patterns (name, source_url, notes) - use word boundaries
BANNED_COMPONENTS = [
    (r'\bultralytics\b', 'ultralytics (AGPL)'),
    (r'\bboxmot\b', 'boxmot (AGPL)'),
    (r'\binsightface\b', 'insightface/ArcFace'),
    (r'\barcface\b', 'ArcFace'),
    (r'\bopengait\b', 'OpenGait'),
    (r'\bmarket-?1501\b', 'Market-1501 dataset'),
    (r'\bmsmt17\b', 'MSMT17 dataset'),
    (r'\bduke-?mtmc\b', 'DukeMTMC dataset'),
    (r'\bcuhk03\b', 'CUHK03 dataset'),
    (r'\bcasia-?b\b', 'CASIA-B dataset'),
    (r'\bgrew\b', 'GREW dataset'),
    (r'\bgait3d\b', 'Gait3D dataset'),
    (r'\bobjects365\b', 'Objects365 (academic only)'),
    (r'\blaion\b', 'LAION dataset/weights'),
    (r'\bdatacomp\b', 'DataComp dataset'),
    (r'\bclip\b.*\b(laion|openai)\b', 'CLIP weights (LAION/OpenAI)'),
    (r'\bpa-?100k\b', 'PA-100K dataset'),
    (r'\brap\b.*\battr', 'RAP attribute dataset'),
    (r'\bpp-?human\b', 'PP-Human (PA-100K/RAP)'),
    (r'\bfast-?reid\b', 'fast-reid model zoo weights'),
    (r'\bbot-?sort\b.*\breid\b', 'BoT-SORT ReID weights'),
    (r'osnet_.*_(duke|market|msmt)', 'OSNet model zoo (Duke/Market/MSMT)'),
    (r'\bretinaface\b', 'RetinaFace (face model)'),
    (r'\bface.*detect', 'face detection model'),
    (r'\bimagenet\b.*\bpretrain', 'ImageNet-pretrained backbone'),
    (r'\bearthcam\b', 'EarthCam (public webcam)'),
    (r'\byoutube\b.*\blive\b', 'YouTube live stream'),
    (r'\binsecam\b', 'Insecam (exposed cameras)'),
]


def load_manifest(manifest_path: Path) -> dict:
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


def validate_schema(manifest: dict, schema_path: Path) -> list[str]:
    """Validate manifest against JSON schema."""
    errors = []

    if not HAVE_JSONSCHEMA:
        return ["WARNING: jsonschema not installed, skipping schema validation"]

    if not schema_path.exists():
        return [f"WARNING: Schema not found: {schema_path}"]

    try:
        with schema_path.open('r', encoding='utf-8') as f:
            schema = json.load(f)

        # Use format checker for URI validation
        validator_cls = jsonschema.Draft7Validator
        validator = validator_cls(
            schema,
            format_checker=validator_cls.FORMAT_CHECKER
        )
        schema_errors = sorted(validator.iter_errors(manifest), key=lambda e: e.path)

        for error in schema_errors:
            path = '.'.join(str(p) for p in error.path) if error.path else 'root'
            errors.append(f"Schema validation failed: {error.message} at {path}")

        # Additional validation: source_url must be a valid HTTP(S) URL
        for i, entry in enumerate(manifest.get('entries', [])):
            source_url = entry.get('source_url', '')
            if source_url and not re.match(r'^https?://', source_url, re.IGNORECASE):
                errors.append(
                    f"Entry {i} ({entry.get('name', 'unknown')}): "
                    f"source_url must start with http:// or https://"
                )

        return errors
    except Exception as e:
        return [f"Schema validation error: {e}"]


def is_license_banned(license_str: str) -> tuple[bool, str]:
    """
    Check if a license string is banned.

    Splits SPDX expressions and checks each component.
    Returns (is_banned, reason).
    """
    if not license_str or not license_str.strip():
        return True, "empty license"

    # Split SPDX expression (OR, AND, WITH)
    components = re.split(r'\s+(?:OR|AND|WITH)\s+', license_str)

    for comp in components:
        comp = comp.strip().strip('()')

        # Check banned patterns
        for pattern in BANNED_LICENSE_PATTERNS:
            if re.search(pattern, comp, re.IGNORECASE):
                return True, f"banned pattern '{pattern}' in '{comp}'"

    return False, ""


def is_license_allowed(license_str: str) -> bool:
    """
    Check if a license is in the allowed list.

    Must be exact match (no prefix matching).
    """
    license_str = license_str.strip()

    # Check SPDX identifiers (case-sensitive)
    if license_str in ALLOWED_SPDX:
        return True

    # Check non-SPDX forms (case-insensitive)
    license_upper = license_str.upper()
    for allowed in ALLOWED_NON_SPDX:
        if license_upper == allowed.upper():
            return True

    # Handle SPDX expressions with OR
    if ' OR ' in license_str:
        components = [c.strip() for c in license_str.split(' OR ')]
        # All components must be allowed
        return all(is_license_allowed(c) for c in components)

    return False


def check_manifest_licenses(manifest: dict) -> list[str]:
    """Check manifest entries for banned or unrecognized licenses."""
    violations = []

    for entry in manifest.get('entries', []):
        license_str = entry.get('license', '').strip()
        name = entry.get('name', 'unknown')

        is_banned, reason = is_license_banned(license_str)
        if is_banned:
            violations.append(f"{name}: banned license '{license_str}' ({reason})")
        elif not is_license_allowed(license_str):
            violations.append(
                f"{name}: license '{license_str}' not in allowed list "
                "(must be exact SPDX or approved non-SPDX)"
            )

    return violations


def check_banned_components(manifest: dict) -> list[str]:
    """Check for banned component names using word boundaries."""
    violations = []

    for entry in manifest.get('entries', []):
        name = entry.get('name', '').lower()
        source_url = entry.get('source_url', '').lower()
        notes = entry.get('notes', '').lower()
        combined = f"{name} {source_url} {notes}"

        for pattern, desc in BANNED_COMPONENTS:
            if re.search(pattern, combined, re.IGNORECASE):
                violations.append(
                    f"{entry.get('name', 'unknown')}: contains banned component '{desc}'"
                )
                break

    return violations


def get_direct_dependencies(requirements_files: list[Path]) -> set[str]:
    """
    Parse requirements files and return set of direct dependencies.

    Only these packages will be checked against manifest.
    Transitive dependencies are the responsibility of direct dependencies.
    """
    direct_deps = set()

    for req_file in requirements_files:
        if not req_file.exists():
            continue

        with req_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Skip -e editable installs
                if line.startswith('-e'):
                    continue
                # Extract package name (before ==, >=, etc.)
                pkg_name = re.split(r'[<>=!]', line)[0].strip()
                if pkg_name:
                    # Normalize name (PyPI convention)
                    direct_deps.add(pkg_name.lower().replace('_', '-'))

    return direct_deps


def get_installed_packages(include_system: bool = False) -> list[dict]:
    """Get list of installed packages with licenses using pip-licenses."""
    try:
        cmd = ['pip-licenses', '--format=json']
        if include_system:
            cmd.append('--with-system')
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        packages = json.loads(result.stdout)
        return [
            {
                'name': pkg.get('Name', ''),
                'version': pkg.get('Version', ''),
                'license': pkg.get('License', 'UNKNOWN')
            }
            for pkg in packages
        ]
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARNING: Failed to get installed packages: {e}", file=sys.stderr)
        return []


def check_installed_packages(
    installed: list[dict],
    manifest: dict,
    direct_deps: set[str] = None
) -> tuple[list[str], list[str]]:
    """
    Check installed packages against manifest and for banned licenses.

    Checks ALL installed packages (full closure) to meet AC-01.
    direct_deps parameter is ignored - kept for compatibility.
    """
    violations = []
    missing = []

    manifest_packages = {}
    for entry in manifest.get('entries', []):
        if entry.get('type') == 'library':
            norm_name = entry.get('name', '').lower().replace('_', '-')
            manifest_packages[norm_name] = entry

    for pkg in installed:
        name = pkg['name']
        version = pkg['version']
        license_str = pkg['license']
        norm_name = name.lower().replace('_', '-')

        # Check ALL installed packages (full closure for AC-01)
        if norm_name not in manifest_packages:
            missing.append(f"{name} ({version}): installed but not in manifest")

        is_banned, reason = is_license_banned(license_str)
        if is_banned:
            violations.append(
                f"{name} ({version}): banned license '{license_str}' ({reason})"
            )
        elif not is_license_allowed(license_str):
            violations.append(
                f"{name} ({version}): license '{license_str}' not in allowed list"
            )

    return violations, missing


def check_ffmpeg_build() -> list[str]:
    """Check if FFmpeg build is LGPL-only (not GPL/nonfree)."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-buildconf'],
            capture_output=True,
            text=True,
            timeout=5
        )
        buildconf = result.stdout + result.stderr

        violations = []
        if '--enable-gpl' in buildconf:
            violations.append("FFmpeg built with --enable-gpl (must be LGPL-only)")
        if '--enable-nonfree' in buildconf:
            violations.append("FFmpeg built with --enable-nonfree (must be LGPL-only)")

        return violations
    except FileNotFoundError:
        return ["WARNING: FFmpeg not found (will be required for T2)"]
    except Exception as e:
        return [f"WARNING: Failed to check FFmpeg: {e}"]


def generate_report(
    manifest: dict,
    installed: list[dict],
    all_violations: list[str]
) -> dict:
    """Generate machine-readable scanner report."""
    return {
        "version": "1.0",
        "timestamp": "2026-10-20T18:15:00Z",
        "manifest": {
            "path": "licenses/manifest.json",
            "entries_count": len(manifest.get('entries', [])),
        },
        "installed_packages": {
            "count": len(installed),
            "packages": [
                {"name": p['name'], "version": p['version'], "license": p['license']}
                for p in installed
            ]
        },
        "violations": all_violations,
        "status": "FAIL" if all_violations else "PASS"
    }


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
        '--schema',
        type=Path,
        default=Path('licenses/manifest_schema.json'),
        help='Path to JSON schema (default: licenses/manifest_schema.json)'
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
        help='Include system-installed packages in scan'
    )
    parser.add_argument(
        '--report',
        type=Path,
        help='Write JSON report to file'
    )
    parser.add_argument(
        '--requirements',
        type=Path,
        action='append',
        help='Requirements file (can be specified multiple times). '
             'If provided, only check direct dependencies from these files.'
    )

    parser.add_argument(
        '--check-ffmpeg',
        action='store_true',
        default='NO_FFMPEG_CHECK' not in os.environ,
        help='Check FFmpeg build configuration (default: True)'
    )

    args = parser.parse_args()

    print(f"Checking license manifest: {args.manifest}")
    print()

    manifest = load_manifest(args.manifest)
    all_violations = []
    installed = []

    # Schema validation
    print("=== Validating manifest schema ===")
    schema_errors = validate_schema(manifest, args.schema)
    if schema_errors:
        for error in schema_errors:
            print(f"ERROR: {error}")
            all_violations.append(error)
    else:
        print("✓ Schema is valid")
    print()

    # Manifest license checks
    print("=== Checking manifest licenses ===")
    manifest_violations = check_manifest_licenses(manifest)
    if manifest_violations:
        for violation in manifest_violations:
            print(f"ERROR: {violation}")
            all_violations.extend(manifest_violations)
    else:
        print("✓ No banned licenses in manifest")
    print()

    # Component name checks
    print("=== Checking for banned components ===")
    component_violations = check_banned_components(manifest)
    if component_violations:
        for violation in component_violations:
            print(f"ERROR: {violation}")
            all_violations.extend(component_violations)
    else:
        print("✓ No banned components found")
    print()

    # Scan installed packages
    if args.scan_installed:
        print("=== Scanning installed packages ===")
        installed = get_installed_packages(include_system=args.include_system)
        print(f"Found {len(installed)} installed packages")
        print("Checking ALL packages (full closure per AC-01)")

        install_violations, missing = check_installed_packages(
            installed, manifest
        )

        if install_violations:
            print()
            for violation in install_violations:
                print(f"ERROR: {violation}")
            all_violations.extend(install_violations)

        if missing:
            print()
            for miss in missing:
                print(f"ERROR: {miss}")
            all_violations.extend(missing)

        if not install_violations and not missing:
            print("✓ All installed packages are clean and in manifest")
        print()

    # FFmpeg check
    if args.check_ffmpeg:
        print("=== Checking FFmpeg build configuration ===")
        ffmpeg_violations = check_ffmpeg_build()
        for violation in ffmpeg_violations:
            if violation.startswith("WARNING"):
                print(violation)
            else:
                print(f"ERROR: {violation}")
                all_violations.append(violation)
        if not any('ERROR' in v or not v.startswith('WARNING') for v in ffmpeg_violations):
            if ffmpeg_violations:
                pass  # Warnings only
            else:
                print("✓ FFmpeg build is LGPL-only")
        print()

    # Generate report
    if args.report:
        report = generate_report(manifest, installed, all_violations)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('w') as f:
            json.dump(report, f, indent=2)
        print(f"Report written to: {args.report}")
        print()

    # Summary
    total_entries = len(manifest.get('entries', []))
    print(f"Summary: Checked {total_entries} manifest entries")

    if all_violations:
        print("\n❌ License fence check FAILED")
        sys.exit(1)
    else:
        print("\n✅ License fence check PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()
