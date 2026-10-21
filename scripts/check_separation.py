#!/usr/bin/env python3
"""
Fixture separation checker.

Validates that SMOKE_ONLY and CODE_TEST fixtures do not appear in
training, evaluation, or study datasets.

FAIL-CLOSED design - errors on:
- 0 input files
- Missing directories
- Auto-discovers all media if no explicit paths given
"""

import argparse
import json
import sys
from pathlib import Path

from trace_search.fixtures.separation import (
    SeparationCheckError,
    run_separation_check,
)


def main():
    parser = argparse.ArgumentParser(
        description="Check for forbidden fixtures in training/eval/study datasets. "
        "FAIL-CLOSED: errors on 0 files or missing directories."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("licenses/manifest.json"),
        help="License manifest path (default: licenses/manifest.json)",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        help="Training data directory",
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        help="Development data directory",
    )
    parser.add_argument(
        "--hold-dir",
        type=Path,
        help="Held-out test data directory",
    )
    parser.add_argument(
        "--study-dir",
        type=Path,
        help="Study data directory",
    )
    parser.add_argument(
        "--hamming-threshold",
        type=int,
        default=8,
        help="Hamming distance threshold for pHash matches (default: 8)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Repository root for auto-discovery and the training-job scan (default: cwd)",
    )

    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Error: Manifest file {args.manifest} not found", file=sys.stderr)
        return 1

    try:
        results = run_separation_check(
            manifest_path=args.manifest,
            train_dir=args.train_dir,
            dev_dir=args.dev_dir,
            hold_dir=args.hold_dir,
            study_dir=args.study_dir,
            hamming_threshold=args.hamming_threshold,
            repo_root=args.repo_root,
        )
    except SeparationCheckError as e:
        print(f"Separation check error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error running separation check: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(results, indent=2))
        return 0 if results["passed"] else 1

    print("Fixture Separation Check")
    print("=" * 60)
    print()
    print(f"Manifest: {args.manifest}")
    datasets_str = ", ".join(results["datasets_checked"]) or "none"
    print(f"Datasets checked: {datasets_str}")
    print(f"Files checked: {results['files_checked']}")

    if results.get("message"):
        print(f"Info: {results['message']}")

    print()

    if results["exact_matches"]:
        print("❌ EXACT MATCHES FOUND (SHA-256):")
        for path, fixture_id, reason in results["exact_matches"]:
            print(f"  - {path} matches {fixture_id} ({reason})")
        print()

    if results["near_duplicates"]:
        print("❌ NEAR-DUPLICATES FOUND (pHash):")
        for path, fixture_id, distance in results["near_duplicates"]:
            print(f"  - {path} matches {fixture_id} (Hamming distance: {distance})")
        print()

    if results.get("config_violations"):
        print("❌ TRAINING CONFIG VIOLATIONS:")
        for violation in results["config_violations"]:
            print(f"  - {violation}")
        print()

    if results["passed"]:
        print("✅ PASSED: No forbidden fixtures found")
        return 0
    else:
        print("❌ FAILED: Forbidden fixtures detected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
