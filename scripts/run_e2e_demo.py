#!/usr/bin/env python3
"""Run the synthetic end-to-end demo.

One command generates T-E-9 clips, runs the pipeline, and scores each
figure's route. The numbers are not evidence for any acceptance criterion.
AC-07 is DEFERRED. A workdir inside the repository is refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from trace_search.pipeline.e2e import LABEL, Schedule, run_demo

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=LABEL)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true", dest="emit_json")
    return parser


def main(argv: list[str] | None = None, *, sink: TextIO | None = None) -> int:
    """Parse the demo flags and return the process exit code."""
    parsed = _parser().parse_args(argv)
    workdir = None if parsed.workdir is None else Path(parsed.workdir)
    return run_demo(
        workdir,
        schedule=Schedule(seed=int(parsed.seed)),
        keep=bool(parsed.keep),
        emit_json=bool(parsed.emit_json),
        sink=sink,
    )


if __name__ == "__main__":
    sys.exit(main())
