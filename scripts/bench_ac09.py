#!/usr/bin/env python3
"""Print AC-09 latency JSON for the synthetic FX-CODE index.

The numbers are an engineering run. AC-09 is passed only by QA from the
frozen SHA. This script does not claim a pass. It reads no video and it
writes SQLite files only under the work directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from trace_search.api.ac09_bench import run_cases


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the requested cases and print a JSON list."""
    parser = argparse.ArgumentParser(
        description="Measure AC-09 latency on a synthetic FX-CODE index."
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--case", action="append", dest="cases")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="ac09-bench-") as directory:
            _print(run_cases(Path(directory), scale=args.scale, cases=args.cases))
        return 0
    _print(run_cases(args.work_dir, scale=args.scale, cases=args.cases))
    return 0


def _print(payload: object) -> None:
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
