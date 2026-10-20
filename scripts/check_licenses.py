#!/usr/bin/env python3
"""
License fence checker for trace-search.

Validates the license manifest and scans installed packages for:
- Banned licenses (AGPL, GPL, SSPL, BUSL, NC, research-only, etc.)
- Banned component names (Ultralytics, boxmot, model zoos, etc.)
- Missing manifest entries for installed packages
- Schema compliance
- OpenCV-bundled FFmpeg (libavcodec) license/configuration

Exit codes:
  0 - All checks passed
  1 - Validation errors or banned items found
  2 - Usage error
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import jsonschema
    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False

from trace_search.ingest.ffmpeg_check import (  # noqa: E402
    inspect_opencv_libavcodec,
    is_opencv_ffmpeg_distribution,
)

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

# Single GPL-family exception (PRD rev 3a / AC-01 / §3).
# Rationale: GCC runtime libraries ship under GPL-3.0 WITH GCC-exception-3.1
# (the GCC Runtime Library Exception). That exception explicitly permits
# redistribution of the runtime with independent programs. This is a named
# allowlist of (package, expression) pairs — GPL matching itself is unchanged.
# Allowed packages only: libgfortran, libquadmath, libgcc_s, libstdc++.
# Allowed expressions only: GPL-3.0 / GPL-3.0-only / GPL-3.0-or-later
# WITH GCC-exception-3.1, plus the "GCC Runtime Library Exception" wording.
# Plain GPL-3.0, GPL-2.0, any other WITH exception, or a non-GCC package
# claiming this exception all fail.
GCC_RUNTIME_EXCEPTION_LIBRARIES = {
    'libgfortran',
    'libquadmath',
    'libgcc-s',  # normalized form of libgcc_s
    'libstdc++',
}

GCC_RUNTIME_EXCEPTION_LICENSES = {
    'GPL-3.0 WITH GCC-exception-3.1',
    'GPL-3.0-only WITH GCC-exception-3.1',
    'GPL-3.0-or-later WITH GCC-exception-3.1',
    'GCC Runtime Library Exception',
}

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
    'HPND',
    'MIT-CMU',
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
    'Historical Permission Notice and Disclaimer (HPND)',
    'MIT CMU License (MIT-CMU)',
    'Pexels License',  # For stock footage (§2a)
    'Pixabay Content License',
    'Zlib',  # opencv-python-headless bundled libs
    'Libpng',  # opencv-python-headless bundled libs
    'libtiff',  # opencv-python-headless bundled libs
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


def normalize_package_name(name: str) -> str:
    """Normalize a package/component name for comparisons."""
    return name.lower().replace('_', '-').replace('.', '-')


def load_manifest(manifest_path: Path) -> dict:
    """Load and parse the license manifest JSON."""
    try:
        with manifest_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in manifest: {e}", file=sys.stderr)
        raise


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


def is_gcc_runtime_exception(license_str: str, component_name: str) -> bool:
    """True only for the named GCC runtime + GCC-exception-3.1 pairing."""
    if not component_name:
        return False
    return (
        normalize_package_name(component_name) in GCC_RUNTIME_EXCEPTION_LIBRARIES
        and license_str.strip() in GCC_RUNTIME_EXCEPTION_LICENSES
    )


def is_license_banned(license_str: str, component_name: str = '') -> tuple[bool, str]:
    """
    Check if a license string is banned.

    Splits on '; ' (pip-licenses format), SPDX OR/AND/WITH, and checks each component.
    Allows GCC Runtime Library Exception for specific GCC runtime libraries only.
    Returns (is_banned, reason).
    """
    if not license_str or not license_str.strip():
        return True, "empty license"

    # Named GCC exception: check the full expression BEFORE splitting on WITH.
    if is_gcc_runtime_exception(license_str, component_name):
        return False, ""

    # Split on '; ' (pip-licenses format) and SPDX operators
    all_components = []
    for part in license_str.split(';'):
        part = part.strip()
        if part:
            spdx_components = re.split(r'\s+(?:OR|AND|WITH)\s+', part)
            all_components.extend(spdx_components)

    for comp in all_components:
        comp = comp.strip().strip('()')
        if not comp:
            continue

        for pattern in BANNED_LICENSE_PATTERNS:
            if re.search(pattern, comp, re.IGNORECASE):
                return True, f"banned pattern '{pattern}' in '{comp}'"

    return False, ""


def is_license_allowed(license_str: str, component_name: str = '') -> bool:
    """
    Check if a license is in the allowed list.

    Handles '; ' (pip-licenses format) and SPDX expressions.
    For AND/';' expressions: ALL components must be allowed.
    For OR expressions: at least one component must be allowed.
    Allows GCC Runtime Library Exception for specific GCC runtime libraries.
    """
    license_str = license_str.strip()
    if not license_str:
        return False

    if is_gcc_runtime_exception(license_str, component_name):
        return True

    # Handle '; ' separated licenses (pip-licenses format)
    # Treat as AND - all must be allowed
    if '; ' in license_str:
        parts = [p.strip() for p in license_str.split('; ')]
        return all(is_license_allowed(p, component_name) for p in parts if p)

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
        return any(is_license_allowed(c, component_name) for c in components if c)

    # Handle SPDX AND expression - all must be allowed
    if ' AND ' in license_str:
        components = [c.strip() for c in license_str.split(' AND ')]
        return all(is_license_allowed(c, component_name) for c in components if c)

    return False


def check_manifest_licenses(manifest: dict) -> list[str]:
    """Check manifest entries for banned or unrecognized licenses and generator licenses."""
    violations = []

    for entry in manifest.get('entries', []):
        license_str = entry.get('license', '').strip()
        name = entry.get('name', 'unknown')
        entry_type = entry.get('type', '')

        is_banned, reason = is_license_banned(license_str, name)
        if is_banned:
            violations.append(f"{name}: banned license '{license_str}' ({reason})")
        elif not is_license_allowed(license_str, name):
            violations.append(
                f"{name}: license '{license_str}' not in allowed list "
                "(must be exact SPDX or approved non-SPDX)"
            )

        # generator_license uses the same allowlist / SPDX parser as packages.
        # Applies to every row that carries a generator: synthetic_clip,
        # code_fixture (FX-CODE), and type generator.
        # Empty, NC, Proprietary, GPL, UNKNOWN, and free-text all fail closed.
        if entry_type in {'synthetic_clip', 'code_fixture', 'generator'}:
            raw_gen = entry.get('generator_license', '')
            generator_license = raw_gen.strip() if isinstance(raw_gen, str) else ''
            if not generator_license:
                violations.append(
                    f"{name}: {entry_type} missing required generator_license field"
                )
            else:
                gen_is_banned, gen_reason = is_license_banned(generator_license)
                if gen_is_banned:
                    violations.append(
                        f"{name}: generator_license '{generator_license}' is banned "
                        f"({gen_reason})"
                    )
                elif not is_license_allowed(generator_license):
                    violations.append(
                        f"{name}: generator_license '{generator_license}' not in allowed list"
                    )

    return violations


def check_banned_components(manifest: dict) -> list[str]:
    """Check for banned component names using normalized matching."""
    violations = []

    for entry in manifest.get('entries', []):
        name = entry.get('name', '')
        source_url = entry.get('source_url', '')
        notes = entry.get('notes', '')
        generator = entry.get('generator', '')

        def normalize(s):
            return s.lower().replace('_', '-').replace('.', '-')

        combined = (
            f"{normalize(name)} {normalize(source_url)} "
            f"{normalize(notes)} {normalize(generator)}"
        )

        for pattern, desc in BANNED_COMPONENTS:
            if re.search(pattern, combined, re.IGNORECASE):
                violations.append(
                    f"{entry.get('name', 'unknown')}: contains banned component '{desc}'"
                )
                break

    return violations


def parse_requirement_name(line: str) -> str | None:
    """Return a normalized package name from one requirements.txt line."""
    line = line.split('#', 1)[0].strip()
    if not line or line.startswith('-'):
        return None
    line = line.split(';', 1)[0].strip()
    line = re.sub(r'\[[^\]]*\]', '', line)
    name = re.split(r'[<>=!~]', line, maxsplit=1)[0].strip().strip(',')
    if not name or not re.match(r'^[A-Za-z0-9]', name):
        return None
    return normalize_package_name(name)


def get_direct_dependencies(requirements_files: list[Path]) -> set[str]:
    """Parse requirements files and return set of direct dependencies."""
    direct_deps = set()
    for req_file in requirements_files:
        if not req_file.exists():
            continue
        for raw in req_file.read_text(encoding='utf-8').splitlines():
            name = parse_requirement_name(raw)
            if name:
                direct_deps.add(name)
    return direct_deps


def check_requirements_in_manifest(
    manifest: dict,
    requirements_files: list[Path],
) -> list[str]:
    """Fail when a requirements.txt package has no library row in the manifest."""
    violations = []
    manifest_libs = {
        normalize_package_name(entry.get('name', ''))
        for entry in manifest.get('entries', [])
        if entry.get('type') == 'library'
    }
    for req_file in requirements_files:
        if not req_file.exists():
            violations.append(f"Requirements file not found: {req_file}")
            continue
        for raw in req_file.read_text(encoding='utf-8').splitlines():
            name = parse_requirement_name(raw)
            if name and name not in manifest_libs:
                violations.append(
                    f"{name}: required by {req_file.name} but not in manifest"
                )
    return violations


def get_installed_packages(include_system: bool = False) -> list[dict]:
    """Get list of installed packages with licenses using pip-licenses."""
    if shutil.which('pip-licenses') is None:
        raise RuntimeError("pip-licenses is not on PATH")
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
        raise RuntimeError("pip-licenses returned invalid JSON")
    except Exception as e:
        print(f"ERROR: Unexpected error getting installed packages: {e}", file=sys.stderr)
        raise


def check_installed_packages(
    installed: list[dict],
    manifest: dict,
    direct_deps: set[str] | None = None
) -> tuple[list[str], list[str]]:
    """
    Check installed packages against manifest and for banned licenses.

    Checks ALL installed packages (full closure) to meet AC-01.
    direct_deps parameter is ignored - kept for compatibility.

    Exempts the project's own package (trace-search) explicitly.
    """
    del direct_deps  # unused; full-closure scan
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

        if norm_name == 'trace-search':
            continue

        if norm_name not in manifest_packages:
            missing.append(f"{name} ({version}): installed but not in manifest")

        is_banned, reason = is_license_banned(license_str, name)
        if is_banned:
            violations.append(
                f"{name} ({version}): banned license '{license_str}' ({reason})"
            )
        elif not is_license_allowed(license_str, name):
            violations.append(
                f"{name} ({version}): license '{license_str}' not in allowed list"
            )

    return violations, missing


def check_ffmpeg_build(manifest: dict) -> tuple[list[str], str]:
    """
    Check FFmpeg via the opencv wheel's bundled libavcodec.

    Distribution names come from ffmpeg_check.OPENCV_FFMPEG_DISTRIBUTIONS.
    System ffmpeg is not a project dependency. When none of those wheels
    is declared, the status is NOT_A_DEPENDENCY.

    Returns:
        (violations, status) where status is one of:
        - 'LGPL_OK': bundled libavcodec is LGPL and has no GPL/nonfree flags
        - 'NOT_A_DEPENDENCY': no opencv FFmpeg dependency declared
        - 'GPL_VIOLATION': bundled FFmpeg is GPL or nonfree
        - 'MISSING': libavcodec could not be loaded (fail-closed)
    """
    opencv_declared = any(
        is_opencv_ffmpeg_distribution(e.get('name', ''))
        for e in manifest.get('entries', [])
    )

    if not opencv_declared:
        return [], 'NOT_A_DEPENDENCY'

    try:
        license_str, config_str = inspect_opencv_libavcodec()
    except Exception as e:
        return [f"Failed to inspect opencv bundled libavcodec: {e}"], 'MISSING'

    if '--enable-gpl' in config_str:
        return (
            ["opencv bundled FFmpeg built with --enable-gpl (must be LGPL-only)"],
            'GPL_VIOLATION',
        )
    if '--enable-nonfree' in config_str:
        return (
            ["opencv bundled FFmpeg built with --enable-nonfree (must be LGPL-only)"],
            'GPL_VIOLATION',
        )
    if 'LGPL' not in license_str.upper():
        return (
            [f"opencv bundled libavcodec license is not LGPL: {license_str}"],
            'GPL_VIOLATION',
        )

    return [], 'LGPL_OK'


def generate_report(
    manifest: dict,
    installed: list[dict],
    all_violations: list[str],
    manifest_path: Path,
    ffmpeg_status: str
) -> dict:
    """Generate machine-readable scanner report."""
    from datetime import datetime, timezone

    try:
        git_sha = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        ).stdout.strip() or "unknown"
    except Exception:
        git_sha = "unknown"

    verdict = "FAIL" if all_violations else "PASS"
    return {
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "ffmpeg_status": ffmpeg_status,
        "verdict": verdict,
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
        "status": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns exit code (0 for pass, 1 for fail)."""
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
        help='Requirements file to require in the manifest '
             '(repeat for more than one). Defaults to requirements.txt '
             'and requirements-dev.txt.'
    )
    parser.add_argument(
        '--check-ffmpeg',
        action='store_true',
        default=True,
        help='Check FFmpeg build configuration (default: True)'
    )

    args = parser.parse_args(argv)

    print(f"Checking license manifest: {args.manifest}")
    print()

    try:
        manifest = load_manifest(args.manifest)
    except (FileNotFoundError, json.JSONDecodeError):
        return 1

    all_violations: list[str] = []
    installed: list[dict] = []

    print("=== Validating manifest schema ===")
    schema_errors = validate_schema(manifest, args.schema)
    if schema_errors:
        for error in schema_errors:
            print(f"ERROR: {error}")
            all_violations.append(error)
    else:
        print("✓ Schema is valid")
    print()

    print("=== Checking manifest licenses ===")
    manifest_violations = check_manifest_licenses(manifest)
    if manifest_violations:
        for violation in manifest_violations:
            print(f"ERROR: {violation}")
        all_violations.extend(manifest_violations)
    else:
        print("✓ No banned licenses in manifest")
    print()

    print("=== Checking for banned components ===")
    component_violations = check_banned_components(manifest)
    if component_violations:
        for violation in component_violations:
            print(f"ERROR: {violation}")
        all_violations.extend(component_violations)
    else:
        print("✓ No banned components found")
    print()

    print("=== Checking declared requirements ===")
    req_files = args.requirements if args.requirements else [
        Path('requirements.txt'),
        Path('requirements-dev.txt'),
    ]
    req_violations = check_requirements_in_manifest(manifest, req_files)
    if req_violations:
        for violation in req_violations:
            print(f"ERROR: {violation}")
        all_violations.extend(req_violations)
    else:
        print("✓ Declared requirements are in the manifest")
    print()

    if args.scan_installed:
        print("=== Scanning installed packages ===")
        try:
            installed = get_installed_packages(include_system=args.include_system)
        except Exception as e:
            print(f"\nERROR: Failed to scan installed packages: {e}")
            all_violations.append(f"Package scan failed: {e}")
            installed = []
        else:
            if not installed:
                message = "Package scan failed: installed package list is empty"
                print(f"ERROR: {message}")
                all_violations.append(message)
            else:
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

    ffmpeg_status = 'NOT_A_DEPENDENCY'
    if args.check_ffmpeg:
        print("=== Checking FFmpeg build configuration ===")
        ffmpeg_violations, ffmpeg_status = check_ffmpeg_build(manifest)
        for violation in ffmpeg_violations:
            print(f"ERROR: {violation}")
            all_violations.append(violation)
        print(f"FFMPEG: {ffmpeg_status}")
        print()

    if args.report:
        report = generate_report(
            manifest, installed, all_violations, args.manifest, ffmpeg_status
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('w') as f:
            json.dump(report, f, indent=2)
        print(f"Report written to: {args.report}")
        print()

    total_entries = len(manifest.get('entries', []))
    print(f"Summary: Checked {total_entries} manifest entries")

    if all_violations:
        print("\n❌ License fence check FAILED")
        return 1

    print("\n✅ License fence check PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
