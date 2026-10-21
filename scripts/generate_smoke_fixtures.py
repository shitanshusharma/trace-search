#!/usr/bin/env python3
"""
Generate synthetic smoke test fixtures.

This script generates synthetic video clips for smoke testing and
updates the license manifest with their metadata.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from trace_search.fixtures.generator import generate_smoke_fixtures


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic smoke test fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("smoke"),
        help="Output directory for generated clips (default: smoke/)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("licenses/manifest.json"),
        help="License manifest path (default: licenses/manifest.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation (default: 42)",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=5,
        help="Number of clips to generate (default: 5, minimum: 5)",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Update the manifest file with generated fixture entries",
    )

    args = parser.parse_args()

    if args.num_clips < 5:
        print("Error: Must generate at least 5 clips", file=sys.stderr)
        return 1

    print(f"Generating {args.num_clips} synthetic smoke fixtures...")
    print(f"Output directory: {args.output_dir}")
    print(f"Base seed: {args.seed}")
    print()

    try:
        manifest_entries = generate_smoke_fixtures(
            output_dir=args.output_dir,
            base_seed=args.seed,
            num_clips=args.num_clips,
        )
    except Exception as e:
        print(f"Error generating fixtures: {e}", file=sys.stderr)
        return 1

    print()
    print(f"Successfully generated {len(manifest_entries)} clips")

    if args.update_manifest:
        if not args.manifest.exists():
            print(f"Error: Manifest file {args.manifest} not found", file=sys.stderr)
            return 1

        try:
            with open(args.manifest) as f:
                manifest = json.load(f)

            existing_fixture_ids = {
                entry.get("fixture_id")
                for entry in manifest.get("entries", [])
                if "fixture_id" in entry
            }

            new_entries = [
                entry
                for entry in manifest_entries
                if entry["fixture_id"] not in existing_fixture_ids
            ]

            if new_entries:
                manifest["entries"].extend(new_entries)
                manifest["last_updated_ist"] = datetime.now().strftime("%Y-%m-%d")

                with open(args.manifest, "w") as f:
                    json.dump(manifest, f, indent=2)
                    f.write("\n")

                print(f"Added {len(new_entries)} new entries to {args.manifest}")
            else:
                print("All fixtures already in manifest")

        except Exception as e:
            print(f"Error updating manifest: {e}", file=sys.stderr)
            return 1
    else:
        print()
        print("To update the manifest, run with --update-manifest")

    return 0


if __name__ == "__main__":
    sys.exit(main())
