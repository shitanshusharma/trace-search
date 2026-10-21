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
import hashlib
import json
import re
import shutil
import site
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

# Named exceptions for files inside the pinned opencv-python-headless wheel.
# Shitanshu Sharma, 21 Oct 2026, 7:45 AM IST: "allow OpenSSL and Intel IPP
# as narrow named exceptions for the OpenCV-bundled files only."
# Counsel confirmation required before release.
# These are not global allowlist entries. OpenSSL and
# LicenseRef-Intel-Simplified-Software-License stay out of ALLOWED_SPDX
# and ALLOWED_NON_SPDX. A row matches only when the licence string, the
# library version, the bundling package, that package's version, the file
# path, the manifest row name, and that file's sha256 all match.
# The installed-closure scan hashes the file in site-packages. The
# exception applies only when that on-disk sha256 matches both the
# manifest row and the pinned value. A missing file or a mismatch fails
# closed. Any other package, version, file, name, or hash fails.
OPENCV_BUNDLED_FILE_EXCEPTIONS = (
    {
        'license': 'OpenSSL',
        'version': '1.1.1w',
        'package': 'opencv-python-headless',
        'package_version': '4.10.0.84',
        'name': 'OpenSSL-libcrypto',
        'file': 'opencv_python_headless.libs/libcrypto-337dac8b.so.1.1',
        'sha256': 'd28a738e776f5f5c17b3577b16b6dfc093132060612769d03cf4048eba10b7aa',
    },
    {
        'license': 'OpenSSL',
        'version': '1.1.1w',
        'package': 'opencv-python-headless',
        'package_version': '4.10.0.84',
        'name': 'OpenSSL-libssl',
        'file': 'opencv_python_headless.libs/libssl-28bef1ac.so.1.1',
        'sha256': 'cedc4cdd914b92099831b6684ea346aa3ff2720ae7eb86ba59af4d8bae805a65',
    },
    {
        'license': 'LicenseRef-Intel-Simplified-Software-License',
        'version': '2021.11.0',
        'package': 'opencv-python-headless',
        'package_version': '4.10.0.84',
        'name': 'Intel IPP ICV',
        'file': 'cv2/cv2.abi3.so',
        'sha256': '72c7fe9b389ecd9549b6e9cdca35ee11940de15d13d46722b8412865cafabc78',
    },
    {
        'license': 'LicenseRef-Intel-Simplified-Software-License',
        'version': '2021.11.0',
        'package': 'opencv-python-headless',
        'package_version': '4.10.0.84',
        'name': 'Intel IPP IW',
        'file': 'cv2/cv2.abi3.so',
        'sha256': '72c7fe9b389ecd9549b6e9cdca35ee11940de15d13d46722b8412865cafabc78',
    },
)

# Named exception for Pillow's bundled FreeType only.
# Shitanshu Sharma: "Do all of it, FreeType under FTL only."
# FTL stays out of ALLOWED_SPDX and ALLOWED_NON_SPDX. A row matches only
# when the licence string is the single bare term FTL, and the library
# version, the bundling package, that package's version, the file path,
# the manifest row name, and that file's sha256 all match. The
# installed-closure scan hashes the file in site-packages. The exception
# applies only when that on-disk sha256 matches both the manifest row and
# the pinned value. A missing file or a mismatch fails closed. Any other
# package, version, file, name, or hash fails.
PILLOW_BUNDLED_FILE_EXCEPTIONS = (
    {
        'license': 'FTL',
        'version': '2.14.3',
        'package': 'Pillow',
        'package_version': '13.0.0',
        'name': 'FreeType',
        'file': 'pillow.libs/libfreetype-57f805ba.so.6.20.6',
        'sha256': '5be086dea47fd1c3fadd431ee79fe0b5e5a0c6c38e78f1db55cf2dad17dea430',
    },
)

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
    'IJG',  # Independent JPEG Group; libjpeg-turbo ships IJG AND BSD-3-Clause AND Zlib
    # Pillow / scipy bundled libs. Each id was checked against this set and
    # ALLOWED_NON_SPDX before it was added. FTL is not one of them.
    'MIT-Modern-Variant',  # HarfBuzz Old MIT grant in the Pillow wheel
    'libpng-2.0',  # PNG Reference Library License version 2
    'X11',  # libxcb in the Pillow wheel
    'BSD-3-Clause-Open-MPI',  # LAPACK inside scipy's OpenBLAS shared object
    'MIT-open-group',  # libXau Open Group grant in the Pillow wheel
    'bzip2-1.0.6',  # bzip2 1.0.8 statically contained in Pillow's libfreetype
}

# OpenSSL and LicenseRef-Intel-Simplified-Software-License are not
# allowlisted here. See OPENCV_BUNDLED_FILE_EXCEPTIONS.
# FTL is not allowlisted here. See PILLOW_BUNDLED_FILE_EXCEPTIONS.

# (normalized package name, exact pip-licenses string) -> SPDX id.
# The license text is exact, including case, and only that package is rewritten.
PIP_LICENSES_TO_SPDX = {
    ('imagehash', '2-clause BSD License'): 'BSD-2-Clause',
}

# Parse failures that do not match a banned pattern. Still fail closed.
_UNPARSEABLE = "unparseable license"

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
    'Zlib',  # opencv bundled libs, and zlib-ng inside Pillow libfreetype and libpng
    'Libpng',  # opencv-python-headless bundled libs
    'libtiff',  # opencv-python-headless bundled libs
}

# Banned component patterns. Scanned text is name, source_url, notes, generator,
# and generator_version. Normalize: lowercase, replace _ and . with -, then match
# with (?<![a-z0-9])pattern(?![a-z0-9]).
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


def _matches_bundled_file_exception(
    license_str: str,
    entry: dict | None,
    on_disk: dict[str, list[str]] | None,
    specs: tuple[dict, ...],
) -> bool:
    """True when license_str is one whole bare term and one spec matches.

    A component split out of OR, AND, or WITH does not match, so a banned
    term still fails. When on_disk is set, the installed file's sha256 must
    match the manifest row and the pinned value. None means the disk check
    is not running.
    """
    if not isinstance(entry, dict):
        return False
    raw_license = entry.get('license', '')
    if not isinstance(raw_license, str) or license_str.strip() != raw_license.strip():
        return False
    version = entry.get('version', '')
    package = entry.get('bundled_package', '')
    package_version = entry.get('bundled_package_version', '')
    bundled_file = entry.get('bundled_file', '')
    name = entry.get('name', '')
    sha256 = entry.get('sha256', '')
    fields = (version, package, package_version, bundled_file, name, sha256)
    if not all(isinstance(value, str) and value for value in fields):
        return False
    package_norm = normalize_package_name(package)
    for spec in specs:
        if (
            license_str.strip() == spec['license']
            and version == spec['version']
            and package_norm == normalize_package_name(spec['package'])
            and package_version == spec['package_version']
            and bundled_file == spec['file']
            and name == spec['name']
            and sha256 == spec['sha256']
        ):
            if on_disk is None:
                return True
            digests = on_disk.get(bundled_file, [])
            return digests == [spec['sha256']]
    return False


def is_opencv_bundled_file_exception(
    license_str: str,
    entry: dict | None,
    on_disk: dict[str, list[str]] | None = None,
) -> bool:
    """True only for one named file inside opencv-python-headless 4.10.0.84."""
    return _matches_bundled_file_exception(
        license_str, entry, on_disk, OPENCV_BUNDLED_FILE_EXCEPTIONS
    )


def is_pillow_bundled_file_exception(
    license_str: str,
    entry: dict | None,
    on_disk: dict[str, list[str]] | None = None,
) -> bool:
    """True only for Pillow 13.0.0 libfreetype under the bare licence FTL."""
    return _matches_bundled_file_exception(
        license_str, entry, on_disk, PILLOW_BUNDLED_FILE_EXCEPTIONS
    )


def installed_site_roots() -> list[Path]:
    """Site-packages directories that can hold the OpenCV wheel files."""
    roots: list[Path] = []
    try:
        roots.extend(Path(path) for path in site.getsitepackages())
    except AttributeError:
        print(
            'warning: site.getsitepackages() is unavailable; '
            'named OpenCV bundled-file exceptions fail closed',
            file=sys.stderr,
        )
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None
    if user_site:
        roots.append(Path(user_site))
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hashes_of_installed_file(relative: str, roots: list[Path]) -> list[str]:
    """Distinct sha256 digests of relative under site-packages roots."""
    digests: list[str] = []
    for root in roots:
        candidate = root / relative
        if not candidate.is_file():
            continue
        digest = sha256_file(candidate)
        if digest not in digests:
            digests.append(digest)
    return digests


def _opencv_exception_row(manifest: dict, spec: dict) -> dict | None:
    """Manifest row for one named-exception spec, ignoring sha256."""
    for entry in manifest.get('entries', []):
        package = entry.get('bundled_package', '')
        if (
            entry.get('name') == spec['name']
            and str(entry.get('license', '')).strip() == spec['license']
            and entry.get('version') == spec['version']
            and isinstance(package, str)
            and normalize_package_name(package) == normalize_package_name(spec['package'])
            and entry.get('bundled_package_version') == spec['package_version']
            and entry.get('bundled_file') == spec['file']
        ):
            return entry
    return None


def collect_opencv_bundled_hashes(
    manifest: dict,
    roots: list[Path] | None = None,
) -> dict[str, list[str]]:
    """Hash each named OpenCV file, and Pillow's libfreetype, that the manifest claims."""
    if roots is None:
        roots = installed_site_roots()
    on_disk: dict[str, list[str]] = {}
    for spec in OPENCV_BUNDLED_FILE_EXCEPTIONS + PILLOW_BUNDLED_FILE_EXCEPTIONS:
        relative = spec['file']
        if relative in on_disk:
            continue
        if _opencv_exception_row(manifest, spec) is None:
            continue
        on_disk[relative] = hashes_of_installed_file(relative, roots)
    return on_disk


def opencv_bundled_hash_violations(
    manifest: dict,
    on_disk: dict[str, list[str]] | None,
) -> tuple[list[str], list[str]]:
    """Fail closed when an installed named file is missing or has the wrong hash.

    Returns (violations, ok_lines). Each violation names the file and both hashes.
    """
    if on_disk is None:
        return [], []
    violations: list[str] = []
    ok_lines: list[str] = []
    reported_ok: set[str] = set()
    for spec in OPENCV_BUNDLED_FILE_EXCEPTIONS + PILLOW_BUNDLED_FILE_EXCEPTIONS:
        row = _opencv_exception_row(manifest, spec)
        if row is None:
            continue
        relative = spec['file']
        name = row.get('name', spec['name'])
        manifest_hash = row.get('sha256', '')
        pinned = spec['sha256']
        digests = on_disk.get(relative, [])
        if len(digests) >= 2:
            violations.append(
                f"{name}: bundled file {relative} on-disk sha256 {digests[0]} "
                f"does not match on-disk sha256 {digests[1]}"
            )
        elif not digests:
            violations.append(
                f"{name}: bundled file {relative} is missing "
                f"(manifest sha256 {manifest_hash}, pinned sha256 {pinned})"
            )
        elif digests[0] != manifest_hash or digests[0] != pinned:
            violations.append(
                f"{name}: bundled file {relative} on-disk sha256 {digests[0]} "
                f"does not match manifest sha256 {manifest_hash} "
                f"and pinned sha256 {pinned}"
            )
        elif relative not in reported_ok:
            reported_ok.add(relative)
            ok_lines.append(f"{relative} sha256 {digests[0]}")
    return violations, ok_lines


class SpdxParseError(Exception):
    """An SPDX expression could not be parsed. Callers fail closed."""


_TOKEN_RE = re.compile(r"\(|\)|[A-Za-z0-9][A-Za-z0-9.+-]*")
_OPERATORS = {"AND", "OR", "WITH"}


def _tokenize_spdx(expr: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    length = len(expr)
    while pos < length:
        if expr[pos].isspace():
            pos += 1
            continue
        match = _TOKEN_RE.match(expr, pos)
        if not match:
            raise SpdxParseError(f"unexpected character {expr[pos]!r}")
        tokens.append(match.group(0))
        pos = match.end()
    if not tokens:
        raise SpdxParseError("empty SPDX expression")
    return tokens


class _SpdxParser:
    """SPDX license expression: OR is loosest, then AND, then WITH, plus parentheses."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.index = 0

    def peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def eat(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None:
            raise SpdxParseError("unexpected end of SPDX expression")
        if expected is not None and token != expected:
            raise SpdxParseError(f"expected {expected}")
        self.index += 1
        return token

    def parse(self):
        node = self._parse_or()
        if self.peek() is not None:
            raise SpdxParseError("trailing tokens in SPDX expression")
        return node

    def _parse_or(self):
        node = self._parse_and()
        parts = [node]
        while self.peek() == "OR":
            self.eat("OR")
            parts.append(self._parse_and())
        if len(parts) == 1:
            return node
        return ("OR", parts)

    def _parse_and(self):
        node = self._parse_with()
        parts = [node]
        while self.peek() == "AND":
            self.eat("AND")
            parts.append(self._parse_with())
        if len(parts) == 1:
            return node
        return ("AND", parts)

    def _parse_with(self):
        node = self._parse_primary()
        if self.peek() != "WITH":
            return node
        self.eat("WITH")
        exception = self._parse_primary()
        if not isinstance(node, str) or not isinstance(exception, str):
            raise SpdxParseError("WITH applies only to a license id and an exception id")
        return ("WITH", node, exception)

    def _parse_primary(self):
        token = self.peek()
        if token is None:
            raise SpdxParseError("unexpected end of SPDX expression")
        if token == "(":
            self.eat("(")
            node = self._parse_or()
            if self.peek() != ")":
                raise SpdxParseError("unbalanced parenthesis")
            self.eat(")")
            return node
        if token in _OPERATORS or token == ")":
            raise SpdxParseError(f"unexpected token {token}")
        return self.eat()


def _parse_spdx(expr: str):
    return _SpdxParser(_tokenize_spdx(expr)).parse()


def _id_matches_banned(license_id: str) -> tuple[bool, str]:
    for pattern in BANNED_LICENSE_PATTERNS:
        if re.search(pattern, license_id, re.IGNORECASE):
            return True, f"banned pattern '{pattern}' in '{license_id}'"
    return False, ""


def _id_is_allowed(license_id: str) -> bool:
    if license_id in ALLOWED_SPDX:
        return True
    folded = license_id.upper()
    return any(folded == allowed.upper() for allowed in ALLOWED_NON_SPDX)


def _is_exact_non_spdx(license_str: str) -> bool:
    folded = license_str.upper()
    return any(folded == allowed.upper() for allowed in ALLOWED_NON_SPDX)


def _eval_spdx(node, component_name: str) -> tuple[bool, bool, str]:
    """Return (banned, allowed, reason). A WITH node is allowed only for the GCC exception."""
    if isinstance(node, str):
        banned, reason = _id_matches_banned(node)
        if banned:
            return True, False, reason
        if _id_is_allowed(node):
            return False, True, ""
        return False, False, ""

    kind = node[0]
    if kind == "AND":
        results = [_eval_spdx(child, component_name) for child in node[1]]
        for banned, _allowed, reason in results:
            if banned:
                return True, False, reason
        if all(allowed for _banned, allowed, _reason in results):
            return False, True, ""
        return False, False, ""

    if kind == "OR":
        # PRD §3: a banned term fails the row even inside OR.
        # OR passes only when every term is allowed (GCC exception included).
        results = [_eval_spdx(child, component_name) for child in node[1]]
        for banned, _allowed, reason in results:
            if banned:
                return True, False, reason
        if all(allowed for _banned, allowed, _reason in results):
            return False, True, ""
        return False, False, ""

    if kind == "WITH":
        license_id, exception_id = node[1], node[2]
        expression = f"{license_id} WITH {exception_id}"
        if is_gcc_runtime_exception(expression, component_name):
            return False, True, ""
        banned, reason = _id_matches_banned(license_id)
        if banned:
            return True, False, reason
        return False, False, ""

    raise SpdxParseError("unknown SPDX node")


def _is_single_bare_license_term(license_str: str) -> bool:
    """True for one licence id, with no OR, AND, or WITH around it."""
    try:
        tokens = _tokenize_spdx(license_str)
    except SpdxParseError:
        return False
    return (
        len(tokens) == 1
        and tokens[0] not in _OPERATORS
        and tokens[0] not in {"(", ")"}
    )


def _license_decision(
    license_str: str,
    component_name: str = '',
    entry: dict | None = None,
    on_disk: dict[str, list[str]] | None = None,
) -> tuple[bool, bool, str]:
    """
    Classify one license string.

    Returns (banned, allowed, reason). Unparseable text is not banned; it still
    fails closed. ';' is the pip-licenses conjunction: every part must be allowed.
    AND and OR both require every term to be allowed. A banned term fails either
    operator. WITH is allowed only for the named GCC runtime exception.

    A single bare term is allowed first when it matches
    OPENCV_BUNDLED_FILE_EXCEPTIONS or PILLOW_BUNDLED_FILE_EXCEPTIONS.
    OR, AND, and WITH skip those exceptions and use the parser below unchanged.
    """
    license_str = license_str.strip()
    if not license_str:
        return True, False, "empty license"

    if _is_single_bare_license_term(license_str) and (
        is_opencv_bundled_file_exception(license_str, entry, on_disk)
        or is_pillow_bundled_file_exception(license_str, entry, on_disk)
    ):
        return False, True, ""

    license_str = PIP_LICENSES_TO_SPDX.get(
        (normalize_package_name(component_name), license_str),
        license_str,
    )

    if is_gcc_runtime_exception(license_str, component_name):
        return False, True, ""

    if _is_exact_non_spdx(license_str):
        return False, True, ""

    if ";" in license_str:
        parts = [part.strip() for part in license_str.split(";")]
        decisions = [
            _license_decision(part, component_name, entry, on_disk) for part in parts
        ]
        pending = ""
        for banned, allowed, reason in decisions:
            if banned:
                return True, False, reason
            if not allowed and reason and not pending:
                pending = reason
        if all(allowed for _banned, allowed, _reason in decisions):
            return False, True, ""
        return False, False, pending

    try:
        node = _parse_spdx(license_str)
    except SpdxParseError:
        banned, reason = _id_matches_banned(license_str)
        if banned:
            return True, False, reason
        return False, False, _UNPARSEABLE

    return _eval_spdx(node, component_name)


def _license_failure(
    license_str: str,
    component_name: str = '',
    entry: dict | None = None,
    on_disk: dict[str, list[str]] | None = None,
) -> str | None:
    """Phrase for a failing license, or None when it is allowed."""
    banned, allowed, reason = _license_decision(
        license_str, component_name, entry, on_disk
    )
    if banned:
        return f"banned license '{license_str}' ({reason})"
    if not allowed:
        if reason == _UNPARSEABLE:
            return f"unparseable license '{license_str}'"
        return (
            f"license '{license_str}' not in allowed list "
            "(must be exact SPDX or approved non-SPDX)"
        )
    return None


def is_license_banned(
    license_str: str,
    component_name: str = '',
    entry: dict | None = None,
    on_disk: dict[str, list[str]] | None = None,
) -> tuple[bool, str]:
    """Return whether the license matched a banned pattern, and why."""
    banned, _allowed, reason = _license_decision(
        license_str, component_name, entry, on_disk
    )
    if banned:
        return True, reason
    return False, ""


def is_license_allowed(
    license_str: str,
    component_name: str = '',
    entry: dict | None = None,
    on_disk: dict[str, list[str]] | None = None,
) -> bool:
    """Return whether every required SPDX term is on the allowlist."""
    _banned, allowed, _reason = _license_decision(
        license_str, component_name, entry, on_disk
    )
    return allowed


def check_manifest_licenses(
    manifest: dict,
    on_disk: dict[str, list[str]] | None = None,
) -> list[str]:
    """Check manifest entries for banned or unrecognized licenses and generator licenses."""
    violations = []

    for entry in manifest.get('entries', []):
        license_str = entry.get('license', '').strip()
        name = entry.get('name', 'unknown')
        entry_type = entry.get('type', '')

        failure = _license_failure(license_str, name, entry, on_disk)
        if failure:
            violations.append(f"{name}: {failure}")

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
                gen_banned, _gen_allowed, gen_reason = _license_decision(
                    generator_license
                )
                if gen_banned:
                    violations.append(
                        f"{name}: generator_license '{generator_license}' is banned "
                        f"({gen_reason})"
                    )
                elif gen_reason == _UNPARSEABLE:
                    violations.append(
                        f"{name}: generator_license '{generator_license}' is unparseable"
                    )
                elif not _gen_allowed:
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
        generator_version = entry.get('generator_version', '')

        def normalize(s):
            return s.lower().replace('_', '-').replace('.', '-')

        combined = (
            f"{normalize(name)} {normalize(source_url)} "
            f"{normalize(notes)} {normalize(generator)} "
            f"{normalize(generator_version)}"
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
    A library row matches only when the normalised name and the exact
    installed version both match. A stale version fails closed.
    """
    del direct_deps  # unused; full-closure scan
    violations = []
    missing = []

    manifest_by_name: dict[str, list[dict]] = {}
    for entry in manifest.get('entries', []):
        if entry.get('type') == 'library':
            norm_name = entry.get('name', '').lower().replace('_', '-')
            manifest_by_name.setdefault(norm_name, []).append(entry)

    for pkg in installed:
        name = pkg['name']
        version = str(pkg['version']).strip()
        license_str = pkg['license']
        norm_name = name.lower().replace('_', '-')

        if norm_name == 'trace-search':
            continue

        rows = manifest_by_name.get(norm_name)
        if not rows:
            missing.append(f"{name} ({version}): installed but not in manifest")
        else:
            manifest_versions = [str(row.get('version', '')).strip() for row in rows]
            if version not in manifest_versions:
                listed = ", ".join(manifest_versions)
                violations.append(
                    f"{name}: installed version {version} does not match "
                    f"manifest version {listed}"
                )

        failure = _license_failure(license_str, name)
        if failure:
            violations.append(f"{name} ({version}): {failure}")

    return violations, missing


def check_ffmpeg_build(manifest: dict) -> tuple[list[str], str]:
    """
    Check FFmpeg via the opencv wheel's bundled libavcodec.

    Wheel names are matched by ffmpeg_check.is_opencv_ffmpeg_distribution,
    which is the only normalisation of OPENCV_FFMPEG_DISTRIBUTIONS.
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

    on_disk = None
    if args.scan_installed:
        on_disk = collect_opencv_bundled_hashes(manifest)

    print("=== Checking manifest licenses ===")
    manifest_violations = check_manifest_licenses(manifest, on_disk=on_disk)
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

        if on_disk:
            opencv_on_disk = any(
                spec['file'] in on_disk for spec in OPENCV_BUNDLED_FILE_EXCEPTIONS
            )
            pillow_on_disk = any(
                spec['file'] in on_disk for spec in PILLOW_BUNDLED_FILE_EXCEPTIONS
            )
            if opencv_on_disk:
                print("=== Checking OpenCV bundled file hashes ===")
            if pillow_on_disk:
                print("=== Checking Pillow bundled file hashes ===")
            hash_violations, hash_ok = opencv_bundled_hash_violations(manifest, on_disk)
            for violation in hash_violations:
                print(f"ERROR: {violation}")
                all_violations.append(violation)
            for line in hash_ok:
                print(f"✓ {line}")
            if not hash_violations and opencv_on_disk:
                print("✓ Installed OpenCV bundled files match the manifest and the pin")
            if not hash_violations and pillow_on_disk:
                print("✓ Installed Pillow bundled FreeType file matches the manifest and the pin")
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
