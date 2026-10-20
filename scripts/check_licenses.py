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

# Banned component patterns (name, source_url, notes) - normalize before matching
# Normalize: lowercase, replace _ and . with -, then match with (?<![a-z0-9])pattern(?![a-z0-9])
BANNED_COMPONENTS = [
    (r'(?<![a-z0-9])ultralytics(?![a-z0-9])', 'ultralytics (AGPL)'),
    (r'(?<![a-z0-9])boxmot(?![a-z0-9])', 'boxmot (AGPL)'),
    (r'(?<![a-z0-9])insightface(?![a-z0-9])', 'insightface/ArcFace'),
    (r'(?<![a-z0-9])arcface(?![a-z0-9])', 'ArcFace'),
    (r'(?<![a-z0-9])opengait(?![a-z0-9])', 'OpenGait'),
    (r'(?<![a-z0-9])market-?1501(?![a-z0-9])', 'Market-1501 dataset'),
    (r'(?<![a-z0-9])msmt17(?![a-z0-9])', 'MSMT17 dataset'),
    (r'(?<![a-z0-9])duke-?mtmc(?![a-z0-9])', 'DukeMTMC dataset'),
    (r'(?<![a-z0-9])cuhk03(?![a-z0-9])', 'CUHK03 dataset'),
    (r'(?<![a-z0-9])casia-?b(?![a-z0-9])', 'CASIA-B dataset'),
    (r'(?<![a-z0-9])grew(?![a-z0-9])', 'GREW dataset'),
    (r'(?<![a-z0-9])gait3d(?![a-z0-9])', 'Gait3D dataset'),
    (r'(?<![a-z0-9])objects365(?![a-z0-9])', 'Objects365 (academic only)'),
    (r'(?<![a-z0-9])laion(?![a-z0-9])', 'LAION dataset/weights'),
    (r'(?<![a-z0-9])datacomp(?![a-z0-9])', 'DataComp dataset'),
    (
        r'(?<![a-z0-9])clip(?![a-z0-9]).*(?<![a-z0-9])(laion|openai)(?![a-z0-9])',
        'CLIP weights (LAION/OpenAI)',
    ),
    (r'(?<![a-z0-9])pa-?100k(?![a-z0-9])', 'PA-100K dataset'),
    (r'(?<![a-z0-9])rap(?![a-z0-9]).*attr', 'RAP attribute dataset'),
    (r'(?<![a-z0-9])pp-?human(?![a-z0-9])', 'PP-Human (PA-100K/RAP)'),
    (r'(?<![a-z0-9])fast-?reid(?![a-z0-9])', 'fast-reid model zoo weights'),
    (r'(?<![a-z0-9])bot-?sort(?![a-z0-9]).*reid', 'BoT-SORT ReID weights'),
    (r'osnet.*(?<![a-z0-9])(duke|market|msmt)(?![a-z0-9])', 'OSNet model zoo (Duke/Market/MSMT)'),
    (r'(?<![a-z0-9])retinaface(?![a-z0-9])', 'RetinaFace (face model)'),
    (r'(?<![a-z0-9])scrfd(?![a-z0-9])', 'SCRFD (face model)'),
    (r'(?<![a-z0-9])buffalo-?l(?![a-z0-9])', 'Buffalo_L (face model)'),
    (r'(?<![a-z0-9])mtcnn(?![a-z0-9])', 'MTCNN (face model)'),
    (r'(?<![a-z0-9])facenet-pytorch(?![a-z0-9])', 'facenet-pytorch (face model)'),
    (r'(?<![a-z0-9])face.*detect', 'face detection model'),
    (r'(?<![a-z0-9])imagenet(?![a-z0-9]).*pretrain', 'ImageNet-pretrained backbone'),
    (r'(?<![a-z0-9])earthcam(?![a-z0-9])', 'EarthCam (public webcam)'),
    (r'youtube.*live', 'YouTube live stream'),
    (r'(?<![a-z0-9])insecam(?![a-z0-9])', 'Insecam (exposed cameras)'),
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

    Splits on '; ' (pip-licenses format), SPDX OR/AND/WITH, and checks each component.
    Returns (is_banned, reason).
    """
    if not license_str or not license_str.strip():
        return True, "empty license"

    # Split on '; ' (pip-licenses format) and SPDX operators
    # First split on ';', then on SPDX operators within each part
    all_components = []
    for part in license_str.split(';'):
        part = part.strip()
        if part:
            # Split SPDX expression (OR, AND, WITH)
            spdx_components = re.split(r'\s+(?:OR|AND|WITH)\s+', part)
            all_components.extend(spdx_components)

    for comp in all_components:
        comp = comp.strip().strip('()')
        if not comp:
            continue

        # Check banned patterns
        for pattern in BANNED_LICENSE_PATTERNS:
            if re.search(pattern, comp, re.IGNORECASE):
                return True, f"banned pattern '{pattern}' in '{comp}'"

    return False, ""


def is_license_allowed(license_str: str) -> bool:
    """
    Check if a license is in the allowed list.

    Handles '; ' (pip-licenses format) and SPDX expressions.
    For AND/';' expressions: ALL components must be allowed.
    For OR expressions: at least one component must be allowed.
    """
    license_str = license_str.strip()
    if not license_str:
        return False

    # Handle '; ' separated licenses (pip-licenses format)
    # Treat as AND - all must be allowed
    if '; ' in license_str:
        parts = [p.strip() for p in license_str.split('; ')]
        return all(is_license_allowed(p) for p in parts if p)

    # Check SPDX identifiers (case-sensitive)
    if license_str in ALLOWED_SPDX:
        return True

    # Check non-SPDX forms (case-insensitive)
    license_upper = license_str.upper()
    for allowed in ALLOWED_NON_SPDX:
        if license_upper == allowed.upper():
            return True

    # Handle SPDX OR expression - at least one must be allowed
    if ' OR ' in license_str:
        components = [c.strip() for c in license_str.split(' OR ')]
        return any(is_license_allowed(c) for c in components if c)

    # Handle SPDX AND expression - all must be allowed
    if ' AND ' in license_str:
        components = [c.strip() for c in license_str.split(' AND ')]
        return all(is_license_allowed(c) for c in components if c)

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
    """Check for banned component names using normalized matching."""
    violations = []

    for entry in manifest.get('entries', []):
        name = entry.get('name', '')
        source_url = entry.get('source_url', '')
        notes = entry.get('notes', '')

        # Normalize: lowercase, replace _ and . with -
        def normalize(s):
            return s.lower().replace('_', '-').replace('.', '-')

        combined = f"{normalize(name)} {normalize(source_url)} {normalize(notes)}"

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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)

        if not result.stdout or not result.stdout.strip():
            raise RuntimeError("pip-licenses returned empty output")

        packages = json.loads(result.stdout)

        if not packages:
            raise RuntimeError("pip-licenses returned empty package list")

        return [
            {
                'name': pkg.get('Name', ''),
                'version': pkg.get('Version', ''),
                'license': pkg.get('License', 'UNKNOWN')
            }
            for pkg in packages
        ]
    except subprocess.TimeoutExpired:
        print("ERROR: pip-licenses timed out after 30 seconds", file=sys.stderr)
        raise RuntimeError("pip-licenses command timed out")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: pip-licenses failed with exit code {e.returncode}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise RuntimeError(f"pip-licenses command failed: {e}")
    except FileNotFoundError:
        print("ERROR: pip-licenses command not found", file=sys.stderr)
        raise RuntimeError("pip-licenses not installed")
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse pip-licenses JSON output: {e}", file=sys.stderr)
        print(f"stdout: {result.stdout[:500]}", file=sys.stderr)
        raise RuntimeError("pip-licenses returned invalid JSON")
    except Exception as e:
        print(f"ERROR: Unexpected error getting installed packages: {e}", file=sys.stderr)
        raise


def check_installed_packages(
    installed: list[dict],
    manifest: dict,
    direct_deps: set[str] = None
) -> tuple[list[str], list[str]]:
    """
    Check installed packages against manifest and for banned licenses.

    Checks ALL installed packages (full closure) to meet AC-01.
    direct_deps parameter is ignored - kept for compatibility.

    Exempts the project's own package (trace-search) explicitly.
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

        # Explicitly exempt the project's own package
        if norm_name == 'trace-search':
            continue

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
        return ["FFmpeg not found (required for T2+)"]
    except Exception as e:
        return [f"Failed to check FFmpeg: {e}"]


def generate_report(
    manifest: dict,
    installed: list[dict],
    all_violations: list[str],
    manifest_path: Path
) -> dict:
    """Generate machine-readable scanner report."""
    import subprocess
    from datetime import datetime, timezone

    # Get git commit SHA
    try:
        git_sha = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        ).stdout.strip()
    except Exception:
        git_sha = "unknown"

    return {
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "manifest": {
            "path": str(manifest_path),
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
        default=True,
        help='Check FFmpeg build configuration (default: True, set NO_FFMPEG_CHECK=1 to disable)'
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
        try:
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
        except RuntimeError as e:
            print(f"\nERROR: Failed to scan installed packages: {e}")
            all_violations.append(f"Package scan failed: {e}")
        print()

    # FFmpeg check
    if args.check_ffmpeg and 'NO_FFMPEG_CHECK' not in os.environ:
        print("=== Checking FFmpeg build configuration ===")
        ffmpeg_violations = check_ffmpeg_build()
        for violation in ffmpeg_violations:
            print(f"ERROR: {violation}")
            all_violations.append(violation)
        if not ffmpeg_violations:
            print("✓ FFmpeg build is LGPL-only")
        print()

    # Generate report
    if args.report:
        report = generate_report(manifest, installed, all_violations, args.manifest)
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
