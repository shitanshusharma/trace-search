#!/usr/bin/env python3
"""Banned-licence scan for the CI-tools environment.

``requirements-ci.lock`` is outside AC-01. It stays a separate hashed lock
and is not added to ``licenses/manifest.json``. This check still fail-closes
when an installed CI package carries a banned licence.

Licence text comes from ``License-Expression``, every licence classifier,
and a one-line ``License`` header. The header is still checked when a
classifier is present. Trove tails and shorthands are mapped to SPDX, then
judged by ``scripts/check_licenses.is_license_banned`` (the same parser and
banned list as the app fence). Text that does not map fails closed. There
is no extra allowlist.

Unknown or missing metadata fails closed, so a banned package cannot slip
through by omitting its licence. Two wheels are the exception, and only
when metadata is actually missing. Their upstream LICENSE files were read
at the pinned tag:

- face 26.0.1 is BSD-3-Clause
  https://github.com/mahmoud/face/blob/v26.0.1/LICENSE
- peewee 3.19.0 is MIT
  https://github.com/coleifer/peewee/blob/3.19.0/LICENSE

A banned string on either name still fails. The override does not apply
when metadata is present.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from collections.abc import Iterable
from email.message import Message
from pathlib import Path

from scripts.check_licenses import ALLOWED_SPDX, is_license_banned, normalize_package_name

# Exact name + version. Used only when the wheel has no licence metadata.
UNKNOWN_OVERRIDES: dict[tuple[str, str], tuple[str, str]] = {
    ("face", "26.0.1"): (
        "BSD-3-Clause",
        "https://github.com/mahmoud/face/blob/v26.0.1/LICENSE",
    ),
    ("peewee", "3.19.0"): (
        "MIT",
        "https://github.com/coleifer/peewee/blob/3.19.0/LICENSE",
    ),
}

_UNKNOWN_TOKENS = {"UNKNOWN", "NOASSERTION"}

# SPDX ids this gate understands. MIT-0 is recognised so cffi is not
# fail-closed; it is not added to the app fence allowlist. Banned GPL and
# AGPL ids are recognised by ``_FAMILY_SPDX`` and then rejected.
_KNOWN_BY_FOLD = {spdx.casefold(): spdx for spdx in set(ALLOWED_SPDX) | {"MIT-0"}}

# Trove tails and one-line headers. A bare "BSD" is BSD-3-Clause for this
# banned-licence scan only; the CI wheels that say "BSD" are 3-clause.
# This map is not a manifest and does not change ALLOWED_SPDX.
_EXACT: dict[str, str] = {
    "mit": "MIT",
    "mit license": "MIT",
    "apache software license": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache-2.0 license": "Apache-2.0",
    "bsd": "BSD-3-Clause",
    "bsd license": "BSD-3-Clause",
    "3-clause bsd license": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "2-clause bsd license": "BSD-2-Clause",
    "bsd 2-clause": "BSD-2-Clause",
    "python software foundation license": "PSF-2.0",
    "psfl": "PSF-2.0",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
    "mpl 2.0": "MPL-2.0",
    "mpl2.0": "MPL-2.0",
    "gnu general public license v2 (gplv2)": "GPL-2.0-only",
    "gnu general public license v2 or later (gplv2+)": "GPL-2.0-or-later",
    "gnu general public license v3 (gplv3)": "GPL-3.0-only",
    "gnu general public license v3 or later (gplv3+)": "GPL-3.0-or-later",
    "gnu affero general public license v3 (agplv3)": "AGPL-3.0-only",
    "gnu affero general public license v3 or later (agplv3+)": "AGPL-3.0-or-later",
    "gnu lesser general public license v2 (lgplv2)": "LGPL-2.0-only",
    "gnu lesser general public license v2 or later (lgplv2+)": "LGPL-2.1-or-later",
    "gnu lesser general public license v3 (lgplv3)": "LGPL-3.0-only",
    "gnu lesser general public license v3 or later (lgplv3+)": "LGPL-3.0-or-later",
    "isc license (iscl)": "ISC",
    "isc license": "ISC",
    "iscl": "ISC",
    "the unlicense (unlicense)": "Unlicense",
    "the unlicense": "Unlicense",
    # The Public Domain classifier has no SPDX id of its own. This scan
    # treats it as Unlicense, which is already on the allowed list.
    "public domain": "Unlicense",
}

_FAMILY_SPDX = re.compile(r"(?i)^(AGPL|LGPL|GPL)-\d+(?:\.\d+)?(?:-(?:only|or-later))?$")
_SHORTHAND = re.compile(
    r"(?i)^(?P<family>AGPL|LGPL|GPL)v?(?P<major>\d+)(?:\.(?P<minor>\d+))?(?P<plus>\+)?$"
)
_PAREN = re.compile(r"\(([^)]+)\)\s*$")
_SPLIT = re.compile(r"\s+(?:OR|AND|WITH)\s+|;\s*", re.IGNORECASE)
_OR_LATER = re.compile(r"\bor later\b", re.IGNORECASE)


def licence_values(metadata: Message) -> list[str]:
    """Return expression, classifier tails, and a short License header."""
    found: list[str] = []
    expression = (metadata.get("License-Expression") or "").strip()
    if expression:
        found.append(expression)
    for raw in metadata.get_all("Classifier") or []:
        if str(raw).lower().startswith("license ::"):
            piece = str(raw).split("::")[-1].strip()
            if piece and piece not in found:
                found.append(piece)
    field = metadata.get("License")
    if isinstance(field, str):
        field = field.strip()
        if field and "\n" not in field and len(field) <= 300 and field not in found:
            found.append(field)
    return found


def _unwrap(token: str) -> str:
    """Drop parentheses that wrap the whole token, and nothing else."""
    token = token.strip()
    while len(token) >= 2 and token[0] == "(" and token[-1] == ")":
        inner = token[1:-1].strip()
        if not inner:
            break
        token = inner
    return token


def _components(value: str) -> list[str]:
    """Split SPDX OR/AND/WITH without breaking the English phrase 'or later'."""
    protected = _OR_LATER.sub(" OR_LATER ", _unwrap(value))
    parts: list[str] = []
    for part in _SPLIT.split(protected):
        restored = _unwrap(" ".join(part.replace("OR_LATER", "or later").split()))
        if restored:
            parts.append(restored)
    return parts


def _family_id(token: str) -> str | None:
    match = _FAMILY_SPDX.fullmatch(token.strip())
    if match is None:
        return None
    family = match.group(1).upper()
    return family + token.strip()[match.end(1) :]


def _shorthand_to_spdx(token: str) -> str | None:
    compact = re.sub(r"[\s_]+", "", token)
    match = _SHORTHAND.fullmatch(compact)
    if match is None:
        return None
    family = match.group("family").upper()
    major = match.group("major")
    minor = match.group("minor")
    plus = match.group("plus") is not None
    # PyPI's "LGPLv2+" classifier is the Lesser GPL v2-or-later line.
    if family == "LGPL" and major == "2" and minor is None and plus:
        version = "2.1"
    elif minor is None:
        version = f"{major}.0"
    else:
        version = f"{major}.{minor}"
    suffix = "or-later" if plus else "only"
    return f"{family}-{version}-{suffix}"


def _without_license_word(token: str) -> str | None:
    folded = token.casefold()
    for suffix in (" license", " licence"):
        if not folded.endswith(suffix):
            continue
        stem = token[: -len(suffix)].strip()
        known = _KNOWN_BY_FOLD.get(stem.casefold())
        if known:
            return known
        mapped = _EXACT.get(stem.casefold())
        if mapped:
            return mapped
    return None


def _version_from_prose(folded: str, family: str, plus: bool) -> str | None:
    match = re.search(r"\bv\s*(\d+)(?:\.(\d+))?\b", folded)
    if match is None:
        match = re.search(r"\bversion\s+(\d+)(?:\.(\d+))?\b", folded)
    if match is None:
        return None
    major, minor = match.group(1), match.group(2)
    if family == "LGPL" and major == "2" and minor is None and plus:
        return "2.1"
    if minor is None:
        return f"{major}.0"
    return f"{major}.{minor}"


def _prose_to_spdx(token: str) -> str | None:
    folded = " ".join(token.casefold().split())
    if "lesser general public license" in folded or "library general public license" in folded:
        family = "LGPL"
    elif "affero general public license" in folded:
        family = "AGPL"
    elif "general public license" in folded:
        family = "GPL"
    else:
        return None
    plus = "or later" in folded
    version = _version_from_prose(folded, family, plus)
    if version is None:
        return None
    suffix = "or-later" if plus else "only"
    return f"{family}-{version}-{suffix}"


def canonical_spdx(token: str, depth: int = 0) -> str | None:
    """Map one licence token to an SPDX id, or None when it is not recognised."""
    token = _unwrap(token)
    if not token or depth > 2:
        return None
    known = _KNOWN_BY_FOLD.get(token.casefold())
    if known:
        return known
    family = _family_id(token)
    if family:
        return family
    mapped = _EXACT.get(token.casefold())
    if mapped:
        return mapped
    stripped = _without_license_word(token)
    if stripped:
        return stripped
    short = _shorthand_to_spdx(token)
    if short:
        return short
    if depth == 0:
        paren = _PAREN.search(token)
        if paren is not None:
            inner = canonical_spdx(paren.group(1), depth + 1)
            if inner:
                return inner
    return _prose_to_spdx(token)


def _classify_licence(value: str, dist_name: str) -> tuple[str, str]:
    """Return ok, banned, or unrecognised for one metadata string."""
    unknown = False
    for part in _components(value):
        canonical = canonical_spdx(part)
        if canonical is None:
            banned, reason = is_license_banned(part, dist_name)
            if banned:
                return "banned", reason
            unknown = True
            continue
        banned, reason = is_license_banned(canonical, dist_name)
        if banned:
            return "banned", reason
    if unknown:
        return "unrecognised", ""
    return "ok", ""


def _real_licences(values: list[str]) -> list[str]:
    return [value.strip() for value in values if value.strip().upper() not in _UNKNOWN_TOKENS]


def evaluate_package(
    dist_name: str,
    version: str,
    values: list[str],
) -> tuple[str, str]:
    """Return ``("ok"|"warn"|"fail", message)`` for one installed package."""
    real = _real_licences(values)
    label = f"{dist_name} ({version})"
    if not real:
        override = UNKNOWN_OVERRIDES.get((normalize_package_name(dist_name), version.strip()))
        if override is None:
            return (
                "fail",
                f"{label}: licence metadata is missing or unknown (fail closed)",
            )
        spdx, source_url = override
        return (
            "warn",
            f"{label}: no licence metadata; reviewed upstream {spdx} at {source_url}",
        )

    unrecognised: list[str] = []
    for value in real:
        kind, reason = _classify_licence(value, dist_name)
        if kind == "banned":
            return (
                "fail",
                f"{label}: banned license '{value}' ({reason})",
            )
        if kind == "unrecognised":
            unrecognised.append(value)
    if unrecognised:
        shown = unrecognised[0]
        return (
            "fail",
            f"{label}: unrecognised licence text '{shown}' (fail closed)",
        )
    return "ok", ""


def scan_distributions(
    dists: Iterable[importlib.metadata.Distribution],
) -> tuple[list[str], list[str], int]:
    """Scan installed distributions. Returns warnings, failures, and the count."""
    warnings: list[str] = []
    failures: list[str] = []
    count = 0
    for dist in dists:
        count += 1
        dist_name = (dist.metadata.get("Name") or "").strip()
        version = str(dist.version).strip()
        if not dist_name:
            failures.append("installed distribution has no Name metadata (fail closed)")
            continue
        kind, message = evaluate_package(dist_name, version, licence_values(dist.metadata))
        if kind == "fail":
            failures.append(message)
        elif kind == "warn":
            warnings.append(message)
    return warnings, failures, count


def distributions_at(path: Path | None) -> list[importlib.metadata.Distribution]:
    if path is None:
        return list(importlib.metadata.distributions())
    return list(importlib.metadata.distributions(path=[str(path)]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Banned-licence check for CI tools.")
    parser.add_argument(
        "--path",
        type=Path,
        help="Scan *.dist-info under this directory instead of the current environment",
    )
    parser.add_argument("--report", type=Path, help="Write a JSON report to this path")
    args = parser.parse_args(argv)

    if args.path is not None and not args.path.is_dir():
        print(f"ERROR: scan path is not a directory: {args.path}", file=sys.stderr)
        return 2

    warnings, failures, count = scan_distributions(distributions_at(args.path))
    print(f"CI tools banned-licence check ({count} packages)")
    for message in warnings:
        print(f"WARNING: {message}")
    for message in failures:
        print(f"ERROR: {message}")

    verdict = "FAIL" if failures else "PASS"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "verdict": verdict,
                    "packages": count,
                    "warnings": warnings,
                    "violations": failures,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Report written to: {args.report}")

    if failures:
        print("\nCI tools banned-licence check FAILED")
        return 1
    print("\nCI tools banned-licence check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
