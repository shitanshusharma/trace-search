#!/usr/bin/env python3
"""Banned-licence scan for the CI-tools environment.

``requirements-ci.lock`` is outside AC-01. It stays a separate hashed lock
and is not added to ``licenses/manifest.json``. This check still fail-closes
when an installed CI package carries a banned licence.

Licence text comes from ``License-Expression``, then licence classifiers,
then a one-line ``License`` field. Each string is judged by
``scripts/check_licenses.is_license_banned`` (the same parser and banned
list as the app fence). There is no extra allowlist.

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
import sys
from collections.abc import Iterable
from email.message import Message
from pathlib import Path

from scripts.check_licenses import is_license_banned, normalize_package_name

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


def licence_values(metadata: Message) -> list[str]:
    """Return licence strings from expression, classifiers, or a short header."""
    found: list[str] = []
    expression = (metadata.get("License-Expression") or "").strip()
    if expression:
        found.append(expression)
    for raw in metadata.get_all("Classifier") or []:
        if str(raw).lower().startswith("license ::"):
            piece = str(raw).split("::")[-1].strip()
            if piece:
                found.append(piece)
    if found:
        return found
    field = metadata.get("License")
    if not isinstance(field, str):
        return []
    field = field.strip()
    if field and "\n" not in field and len(field) <= 300:
        return [field]
    return []


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

    for value in real:
        banned, reason = is_license_banned(value, dist_name)
        if banned:
            return (
                "fail",
                f"{label}: banned license '{value}' ({reason})",
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
