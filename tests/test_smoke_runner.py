"""M1a smoke runner on synthetic clips. Output is not evidence for any AC."""

from __future__ import annotations

import builtins
import json
import re
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from scripts.run_smoke import main
from trace_search.detection.types import DetClass
from trace_search.fixtures.generator import SyntheticVideoGenerator, generate_smoke_fixtures
from trace_search.fixtures.guards import LoaderContext, check_fixture_allowed
from trace_search.ingest.file_ingest import FileVideoSource
from trace_search.tracking.store import TrackStore
from trace_search.tracking.types import TrackObservation

_REPO = Path(__file__).resolve().parents[1]


def _porcelain() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    return result.stdout


def _smoke_files() -> set[Path]:
    smoke = _REPO / "smoke"
    if not smoke.exists():
        return set()
    return {path for path in smoke.rglob("*") if path.is_file()}


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    document = {"manifest_version": "1.0", "last_updated_ist": "2026-10-21", "entries": entries}
    path.write_text(json.dumps(document), encoding="utf-8")


def _printed_run_id(output: str) -> str:
    match = re.search(r"run_id=([0-9a-fA-F-]{36})", output)
    assert match is not None, output
    return match.group(1)


def _run(argv: list[str]) -> tuple[int, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue() + stderr.getvalue()


@pytest.fixture(scope="module")
def five_synthetic_clips(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, list[str]]:
    """Five generator clips and a manifest that lives outside the repo."""
    root = tmp_path_factory.mktemp("five-smoke")
    clips = root / "clips"
    entries = generate_smoke_fixtures(clips, base_seed=42, num_clips=5)
    manifest = root / "manifest.json"
    _write_manifest(manifest, entries)
    fixture_ids = [str(entry["fixture_id"]) for entry in entries]
    return clips, manifest, fixture_ids


def test_scripted_figures_on_five_synthetic_clips(
    five_synthetic_clips: tuple[Path, Path, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clips, manifest, fixture_ids = five_synthetic_clips
    guarded: list[Path] = []
    real_check = check_fixture_allowed

    def _wrapped(
        file_path: Path,
        context: LoaderContext,
        manifest_path: Path,
        hamming_threshold: int = 8,
    ) -> None:
        assert context is LoaderContext.SMOKE_TEST
        assert manifest_path == manifest
        guarded.append(Path(file_path))
        real_check(file_path, context, manifest_path, hamming_threshold)

    monkeypatch.setattr("scripts.run_smoke.check_fixture_allowed", _wrapped)
    before_status = _porcelain()
    before_smoke = _smoke_files()
    db = tmp_path / "smoke.sqlite"

    code, output = _run(
        [
            "--manifest",
            str(manifest),
            "--clips-dir",
            str(clips),
            "--db",
            str(db),
            "--detector",
            "scripted-figures",
            "--min-clips",
            "5",
        ]
    )

    assert code == 0, output
    assert "5 clips" in output
    assert "0 errors" in output
    assert len(guarded) == 5
    run_id = _printed_run_id(output)
    assert run_id != "smoke"
    store = TrackStore(db)
    assert store.count(run_id=run_id, derived_from="SMOKE") > 0
    for fixture_id in fixture_ids:
        assert fixture_id in output
        assert "frames=" in output
        assert "detections=" in output
        assert "tracks=" in output
        assert store.count(camera_id=fixture_id, run_id=run_id, derived_from="SMOKE") > 0
    assert _porcelain() == before_status
    assert _smoke_files() == before_smoke


@pytest.mark.parametrize("min_clips", ["0", "-1"])
def test_min_clips_below_one_exits_1(tmp_path: Path, min_clips: str) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    code, output = _run(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--clips-dir",
            str(clips),
            "--db",
            str(tmp_path / "smoke.sqlite"),
            "--detector",
            "scripted-figures",
            "--min-clips",
            min_clips,
        ]
    )
    assert code == 1, output
    assert "--min-clips must be at least 1" in output
    assert "0 clips, 0 errors" not in output


def test_min_clips_six_on_five_clips_exits_1(
    five_synthetic_clips: tuple[Path, Path, list[str]],
    tmp_path: Path,
) -> None:
    clips, manifest, _fixture_ids = five_synthetic_clips
    code, output = _run(
        [
            "--manifest",
            str(manifest),
            "--clips-dir",
            str(clips),
            "--db",
            str(tmp_path / "smoke.sqlite"),
            "--detector",
            "scripted-figures",
            "--min-clips",
            "6",
        ]
    )
    assert code == 1, output


def _tagged_plant(
    tmp_path: Path,
    use: str,
    seed: int,
) -> tuple[Path, Path]:
    clips = tmp_path / "clips"
    clips.mkdir()
    planted = clips / "planted.mp4"
    generator = SyntheticVideoGenerator(seed=seed)
    generator.generate_clip(
        planted,
        duration_sec=1.0,
        fps=10,
        width=320,
        height=240,
        num_figures=1,
        num_obstructions=0,
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "use": use,
                "type": "synthetic_clip",
                "sha256": generator.compute_sha256(planted),
                "phash": [],
                "fixture_id": "FX-PLANTED-001",
            }
        ],
    )
    return clips, manifest


@pytest.mark.parametrize(("use", "seed"), [("CODE_TEST", 11), ("HOLD", 13)])
def test_planted_non_smoke_tag_exits_1(tmp_path: Path, use: str, seed: int) -> None:
    clips, manifest = _tagged_plant(tmp_path, use, seed)
    db = tmp_path / "smoke.sqlite"
    code, output = _run(
        [
            "--manifest",
            str(manifest),
            "--clips-dir",
            str(clips),
            "--db",
            str(db),
            "--detector",
            "scripted-figures",
            "--min-clips",
            "1",
        ]
    )
    assert code == 1, output
    assert "FixtureSeparationError" in output
    assert use in output
    assert "FORBIDDEN" in output
    assert TrackStore(db).count(derived_from="SMOKE") == 0


def test_reused_db_ignores_earlier_smoke_rows(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    blank = clips / "blank.mp4"
    generator = SyntheticVideoGenerator(seed=19)
    generator.generate_clip(
        blank,
        duration_sec=0.4,
        fps=10,
        width=320,
        height=240,
        num_figures=0,
        num_obstructions=0,
    )
    fixture_id = "FX-SMOKE-SYN-BLANK"
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "use": "SMOKE_ONLY",
                "type": "synthetic_clip",
                "sha256": generator.compute_sha256(blank),
                "phash": [],
                "fixture_id": fixture_id,
            }
        ],
    )
    db = tmp_path / "smoke.sqlite"
    TrackStore(db).write(
        "earlier-run",
        fixture_id,
        "SMOKE",
        [
            TrackObservation(
                camera_id=fixture_id,
                track_id=1,
                det_class=DetClass.PERSON,
                frame_number=0,
                timestamp_seconds=0.0,
                box=(1.0, 1.0, 20.0, 40.0),
                score=0.9,
            )
        ],
    )

    code, output = _run(
        [
            "--manifest",
            str(manifest),
            "--clips-dir",
            str(clips),
            "--db",
            str(db),
            "--detector",
            "scripted-figures",
            "--min-clips",
            "1",
        ]
    )

    assert code == 1, output
    assert "no persisted SMOKE observations" in output
    assert "FixtureSeparationError" not in output
    run_id = _printed_run_id(output)
    assert run_id != "earlier-run"
    store = TrackStore(db)
    assert store.count(camera_id=fixture_id, run_id="earlier-run", derived_from="SMOKE") == 1
    assert store.count(camera_id=fixture_id, run_id=run_id, derived_from="SMOKE") == 0


def test_planted_clip_without_a_manifest_row_exits_1(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    planted = clips / "planted.mp4"
    SyntheticVideoGenerator(seed=7).generate_clip(
        planted,
        duration_sec=1.0,
        fps=5,
        width=640,
        height=480,
        num_figures=1,
        num_obstructions=0,
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [])
    before_status = _porcelain()

    code, output = _run(
        [
            "--manifest",
            str(manifest),
            "--clips-dir",
            str(clips),
            "--db",
            str(tmp_path / "smoke.sqlite"),
            "--detector",
            "scripted-figures",
            "--min-clips",
            "1",
        ]
    )

    assert code == 1, output
    assert "FixtureSeparationError" in output
    assert "no manifest row" in output
    assert "FAIL-CLOSED" in output
    assert _porcelain() == before_status


def test_guard_is_called_before_the_clip_is_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    clip = clips / "one.mp4"
    generator = SyntheticVideoGenerator(seed=3)
    generator.generate_clip(
        clip,
        duration_sec=1.0,
        fps=10,
        width=640,
        height=480,
        num_figures=2,
        num_obstructions=0,
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "use": "SMOKE_ONLY",
                "type": "synthetic_clip",
                "sha256": generator.compute_sha256(clip),
                "phash": generator.compute_phashes(clip),
                "fixture_id": "FX-SMOKE-SYN-001",
            }
        ],
    )
    events: list[str] = []
    real_check = check_fixture_allowed
    real_open = FileVideoSource.open

    def _wrapped_check(
        file_path: Path,
        context: LoaderContext,
        manifest_path: Path,
        hamming_threshold: int = 8,
    ) -> None:
        events.append("guard")
        real_check(file_path, context, manifest_path, hamming_threshold)

    def _wrapped_open(self: FileVideoSource) -> None:
        events.append("open")
        real_open(self)

    monkeypatch.setattr("scripts.run_smoke.check_fixture_allowed", _wrapped_check)
    monkeypatch.setattr(FileVideoSource, "open", _wrapped_open)

    code, output = _run(
        [
            "--manifest",
            str(manifest),
            "--clips-dir",
            str(clips),
            "--db",
            str(tmp_path / "smoke.sqlite"),
            "--detector",
            "scripted-figures",
            "--min-clips",
            "1",
        ]
    )

    assert code == 0, output
    assert events.index("guard") < events.index("open")


def test_yolox_without_weights_errors_before_onnxruntime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRACE_SEARCH_YOLOX_WEIGHTS", raising=False)
    real_import = builtins.__import__

    def _tracking_import(module: str, *args: object, **kwargs: object) -> object:
        if module == "onnxruntime" or module.startswith("onnxruntime."):
            raise AssertionError(f"imported {module}")
        return real_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _tracking_import)
    code, output = _run(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--clips-dir",
            str(tmp_path),
            "--db",
            str(tmp_path / "smoke.sqlite"),
            "--detector",
            "yolox",
            "--min-clips",
            "5",
        ]
    )
    assert code == 1
    assert "TRACE_SEARCH_YOLOX_WEIGHTS" in output
    assert "--weights" in output
    assert "local-only" in output
    assert "imported onnxruntime" not in output
    assert "InferenceSession" not in output
