#!/usr/bin/env python3
"""Run the M1a smoke pipeline into a TrackStore.

``scripted-figures`` is the CI detector. It runs on the synthetic clips only.
``yolox`` is local-only: it needs ``--weights`` or ``TRACE_SEARCH_YOLOX_WEIGHTS``
and is never run against real weights in CI. This script does not download
weights. Rows are stored with ``derived_from='SMOKE'``. SMOKE output is not
evidence for any acceptance criterion.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from trace_search.detection.scripted_figures import scripted_figures_detector
from trace_search.detection.types import Detector
from trace_search.fixtures.guards import (
    FixtureSeparationError,
    FixtureUseTag,
    LoaderContext,
    check_fixture_allowed,
    identify_fixture_by_content,
    load_manifest_fixtures,
)
from trace_search.ingest.file_ingest import FileVideoSource
from trace_search.pipeline.camera_pipeline import RunStats, run_cameras
from trace_search.tracking.bytetrack import ByteTracker
from trace_search.tracking.store import TrackStore

_DERIVED_FROM = "SMOKE"
_YOLOX_ENV = "TRACE_SEARCH_YOLOX_WEIGHTS"
_BANNER = "SMOKE: not evidence for any AC"
_DESCRIPTION = (
    "Run the M1a smoke pipeline. "
    "scripted-figures is the CI detector for the 5 synthetic clips. "
    "yolox is local-only and needs weights outside the repository. "
    "CI does not download weights. "
    "SMOKE output is not evidence for any acceptance criterion."
)


@dataclass(frozen=True)
class _Args:
    manifest: Path
    clips_dir: Path
    db: Path
    detector: str
    weights: Path | None
    min_clips: int


def _parse(argv: list[str] | None) -> _Args:
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--detector", choices=("scripted-figures", "yolox"), required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--min-clips", type=int, default=5)
    parsed = parser.parse_args(argv)
    weights = parsed.weights
    weight_path = Path(str(weights)) if weights is not None else None
    return _Args(
        manifest=Path(str(parsed.manifest)),
        clips_dir=Path(str(parsed.clips_dir)),
        db=Path(str(parsed.db)),
        detector=str(parsed.detector),
        weights=weight_path,
        min_clips=int(parsed.min_clips),
    )


def _require_yolox_weights(weights: Path | None) -> Path:
    """Return a local weights path. Missing weights never touch a runtime."""
    if weights is not None:
        return weights
    raw = os.environ.get(_YOLOX_ENV, "").strip()
    if raw != "":
        return Path(raw)
    raise FileNotFoundError(
        "YOLOX weights path is not set. "
        f"Pass --weights or set {_YOLOX_ENV}. "
        "The yolox detector is local-only. "
        "CI does not download weights and does not run yolox against real weights."
    )


def _build_detector(choice: str, weights: Path | None) -> Detector:
    if choice == "scripted-figures":
        return scripted_figures_detector()
    if choice == "yolox":
        # Local-only. Import the runtime adapter only after a path is present,
        # so a missing path never imports onnxruntime.
        resolved = _require_yolox_weights(weights)
        from trace_search.detection.yolox_detector import YoloxDetector

        return YoloxDetector(resolved)
    raise ValueError(f"unknown detector {choice!r}")


def _clip_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"clips directory not found: {directory}")
    clips = [
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"
    ]
    clips.sort(key=lambda path: path.name)
    return clips


def _require_fixture(path: Path, manifest: Path) -> str:
    """Refuse the clip before the pipeline opens it. Return the fixture id.

    The guard is the gate for every caller. The tag is read again here so a
    clip that is not SMOKE_ONLY is still refused if the guard is bypassed.
    """
    check_fixture_allowed(path, LoaderContext.SMOKE_TEST, manifest)
    identified = identify_fixture_by_content(path, load_manifest_fixtures(manifest))
    if identified is None:
        raise FixtureSeparationError(
            f"FAIL-CLOSED: no manifest row for {path} (refused before the clip is opened)"
        )
    use_tag, fixture_id = identified
    if use_tag is not FixtureUseTag.SMOKE_ONLY:
        raise FixtureSeparationError(
            f"Fixture {path} identified as {fixture_id} with use tag {use_tag.value} "
            "is FORBIDDEN in smoke_test context (identified by SHA-256 or pHash)."
        )
    return fixture_id


def _new_tracker() -> ByteTracker:
    return ByteTracker()


def _report_clip(fixture_id: str, stats: RunStats) -> None:
    print(f"{fixture_id} frames={stats.frames} detections={stats.detections} tracks={stats.tracks}")


def _run_clip(
    path: Path,
    manifest: Path,
    detector: Detector,
    store: TrackStore,
    run_id: str,
) -> None:
    fixture_id = _require_fixture(path, manifest)
    source = FileVideoSource(camera_id=fixture_id, file_path=path, start_timestamp=0.0)
    stats_by_camera = run_cameras(
        [source],
        detector,
        _new_tracker,
        store,
        run_id,
        fixture_id,
        _DERIVED_FROM,
    )
    stats = stats_by_camera[fixture_id]
    _report_clip(fixture_id, stats)
    recorded = store.count(camera_id=fixture_id, run_id=run_id, derived_from=_DERIVED_FROM)
    if recorded < 1:
        raise RuntimeError(f"{fixture_id} has no persisted SMOKE observations")


def _attempt(
    path: Path,
    manifest: Path,
    detector: Detector,
    store: TrackStore,
    run_id: str,
) -> bool:
    try:
        _run_clip(path, manifest, detector, store, run_id)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return False
    return True


def _finish(ran: int, errors: int, min_clips: int) -> int:
    print(f"{ran} clips, {errors} errors")
    if errors != 0 or ran < min_clips:
        if errors == 0:
            print(f"error: {ran} clips ran, fewer than min-clips {min_clips}")
        return 1
    return 0


def _run_all(args: _Args, detector: Detector) -> int:
    run_id = str(uuid.uuid4())
    print(f"run_id={run_id}")
    store = TrackStore(args.db)
    ran = 0
    errors = 0
    for path in _clip_paths(args.clips_dir):
        if _attempt(path, args.manifest, detector, store, run_id):
            ran += 1
        else:
            errors += 1
    return _finish(ran, errors, args.min_clips)


def main(argv: list[str] | None = None) -> int:
    """Run every clip in ``--clips-dir`` and return the process status."""
    try:
        args = _parse(argv)
        if args.min_clips < 1:
            raise ValueError("--min-clips must be at least 1")
        print(_BANNER)
        detector = _build_detector(args.detector, args.weights)
        return _run_all(args, detector)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
