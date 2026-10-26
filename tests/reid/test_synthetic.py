"""The in-memory synthetic ReID set and its generator manifest row.

Arrays are built in the test process. Nothing is read from ``smoke/``
and nothing is written into the repo tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import scripts.check_licenses as check_licenses
import trace_search.reid.eval as eval_module
from trace_search.fixtures.guards import FixtureSeparationError, LoaderContext
from trace_search.reid.data import ReidSample, load_reid_samples
from trace_search.reid.synthetic import make_synthetic_reid_set

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "licenses" / "manifest.json"
_SCHEMA = _REPO / "licenses" / "manifest_schema.json"
_SOURCE = _REPO / "src" / "trace_search" / "reid" / "synthetic.py"
_GENERATOR_NAME = "reid/synthetic.py"


def _manifest() -> dict[str, object]:
    loaded = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise AssertionError("manifest must be an object")
    return loaded


def _generator_rows(manifest: dict[str, object]) -> list[dict[str, object]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return []
    rows: list[dict[str, object]] = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("type") == "generator":
            rows.append(entry)
    return rows


def _generator_row() -> dict[str, object]:
    rows = _generator_rows(_manifest())
    assert len(rows) == 1
    return rows[0]


def _with_generator_row(row: dict[str, object]) -> dict[str, object]:
    manifest = _manifest()
    entries = manifest.get("entries")
    assert isinstance(entries, list)
    replaced = False
    updated: list[object] = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("type") == "generator":
            updated.append(row)
            replaced = True
        else:
            updated.append(entry)
    assert replaced
    manifest["entries"] = updated
    return manifest


def _schema_errors(manifest: dict[str, object]) -> list[str]:
    return check_licenses.validate_schema(manifest, _SCHEMA)


def test_same_seed_is_byte_identical_and_a_different_seed_differs() -> None:
    first = make_synthetic_reid_set(seed=0)
    second = make_synthetic_reid_set(seed=0)
    third = make_synthetic_reid_set(seed=1)
    assert len(first) == 32
    for left, right in zip(first, second, strict=True):
        assert left.crop is not None and right.crop is not None
        assert left.crop.tobytes() == right.crop.tobytes()
    assert any(
        left.crop is not None
        and right.crop is not None
        and left.crop.tobytes() != right.crop.tobytes()
        for left, right in zip(first, third, strict=True)
    )


def test_ids_are_fx_synreid_and_the_use_tag_is_dev() -> None:
    samples = make_synthetic_reid_set()
    assert [sample.fixture_id for sample in samples] == [
        f"FX-SYNREID-{number:03d}" for number in range(1, 33)
    ]
    assert [sample.use for sample in samples] == ["DEV"] * len(samples)
    assert {sample.camera_id for sample in samples} == {"cam-0", "cam-1"}
    assert {sample.label for sample in samples} == set(range(8))
    assert all(sample.path is None for sample in samples)
    for sample in samples:
        crop = sample.crop
        assert crop is not None
        assert crop.shape == (64, 32, 3)
        assert crop.dtype == np.uint8


def test_samples_load_in_evaluation_and_are_refused_in_training() -> None:
    samples = make_synthetic_reid_set()
    loaded = load_reid_samples(samples, LoaderContext.EVALUATION, _MANIFEST)
    assert len(loaded) == len(samples)
    with pytest.raises(FixtureSeparationError, match="refused in TRAINING"):
        load_reid_samples(samples, LoaderContext.TRAINING, _MANIFEST)


def test_relabelled_smoke_id_is_still_refused() -> None:
    sample = make_synthetic_reid_set()[0]
    relabelled = ReidSample(
        fixture_id="FX-SMOKE-SYN-001",
        use=sample.use,
        camera_id=sample.camera_id,
        track_id=sample.track_id,
        label=sample.label,
        crop=sample.crop,
        path=None,
    )
    with pytest.raises(FixtureSeparationError, match="FX-SMOKE"):
        load_reid_samples([relabelled], LoaderContext.EVALUATION, _MANIFEST)


class _WriteGuard:
    """Fail the no-write test on a write, including one under the system temp dir."""

    def __init__(self) -> None:
        self.armed = False

    def __call__(self, event: str, args: tuple[object, ...]) -> None:
        if not self.armed or not _is_write_event(event, args):
            return
        raise AssertionError(f"unexpected write event {event} {args!r}")


def _is_write_event(event: str, args: tuple[object, ...]) -> bool:
    if event in {"os.mkdir", "os.rename", "os.remove", "os.rmdir", "sqlite3.connect"}:
        return True
    if event != "open":
        return False
    mode = args[1] if len(args) > 1 else ""
    flags = args[2] if len(args) > 2 else 0
    if isinstance(mode, str) and any(flag in mode for flag in ("w", "a", "x", "+")):
        return True
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND
    return isinstance(flags, int) and bool(flags & write_flags)


_WRITE_GUARD = _WriteGuard()
sys.addaudithook(_WRITE_GUARD)


def test_generator_does_not_write_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The generator, the loader, and the eval command do not write.

    A write under ``tempfile.gettempdir()`` fails this test, as does a
    write in ``tmp_path`` or the work tree. ``np.save`` to the system
    temp directory is the case that used to slip through.
    """
    system_temp = Path(tempfile.gettempdir())
    assert system_temp.is_dir()
    before_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    before_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=_REPO)
    _WRITE_GUARD.armed = True
    try:
        samples = make_synthetic_reid_set()
        load_reid_samples(samples, LoaderContext.EVALUATION, _MANIFEST)
        code = eval_module.main(["--embedder", "placeholder", "--synthetic-seed", "0", "--json"])
    finally:
        _WRITE_GUARD.armed = False
    capsys.readouterr()
    after_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    after_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=_REPO)
    assert code == 0
    assert before_files == after_files
    assert before_status == after_status


def test_camera_shift_moves_the_colour_pattern() -> None:
    """Cameras in one group differ by more than the per-pixel noise.

    Noise is at most 15. The per-camera band move and the darkened band
    push the max channel gap above that bound.
    """
    samples = make_synthetic_reid_set(seed=0)
    by_label: dict[int, dict[str, np.ndarray]] = {}
    for sample in samples:
        assert sample.crop is not None
        by_label.setdefault(sample.label, {})[sample.camera_id] = sample.crop
    for cameras in by_label.values():
        diff = np.abs(cameras["cam-0"].astype(np.int16) - cameras["cam-1"].astype(np.int16))
        assert int(diff.max()) > 15


def test_odd_groups_stay_close_to_their_partner_on_the_same_camera() -> None:
    """Group 1 shares group 0's bands. Group 2 does not.

    Same camera, so the gap is the colours. Noise is at most 15, and the
    shared pair's mean gap stays under that. An unrelated palette does not.
    """
    samples = make_synthetic_reid_set(seed=0)
    partner = _mean_gap(samples, 0, 1)
    unrelated = _mean_gap(samples, 0, 2)
    assert partner < 15
    assert unrelated > 60
    assert partner < unrelated


def test_camera_layouts_do_not_repeat_every_four_cameras() -> None:
    """Camera k and camera k+4 are different layouts.

    The old roll was ``(camera + 1)`` whole bands. On a 4-band crop that
    made camera 4 match camera 0 apart from noise of at most 15. A large
    share of pixels now differs by more than that.
    """
    samples = make_synthetic_reid_set(n_groups=1, per_group=8, n_cameras=8, seed=0)
    crops = _one_crop_per_camera(samples, 8)
    for left, right in _camera_pairs(8):
        assert _max_gap(crops[left], crops[right]) > 15
    height, width = crops[0].shape[:2]
    # At least one full band, not a single marked pixel.
    assert _pixels_over(crops[0], crops[4], 15) >= (height // 4) * width


def test_wrapped_roll_stays_distinct() -> None:
    """A later cycle does not copy an earlier camera.

    At height 4 the band roll repeats every 2 cameras. Camera 4 lands on
    camera 0's rows and darkens the same band. Without the cycle mark
    those crops would differ only by noise of at most 15.
    """
    samples = make_synthetic_reid_set(
        n_groups=1,
        per_group=5,
        n_cameras=5,
        size=(4, 4),
        seed=0,
    )
    crops = _one_crop_per_camera(samples, 5)
    for left, right in _camera_pairs(5):
        assert _max_gap(crops[left], crops[right]) > 15


_NOTE = "Placeholder or synthetic embeddings. Not an AC-06 result."


def test_seed_0_separates_placeholder_from_random_init_osnet(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Seed 0 does not give random-init osnet-lite a perfect score.

    The placeholder and osnet-lite Rank-1 / mAP differ. osnet-lite is
    above the placeholder and is not Rank-1 1.0 and mAP 1.0 together.
    ``evidence`` stays false. The note says this is not an AC-06 result.
    These figures are synthetic diagnostics, not an AC-06 number.
    """
    placeholder = _eval_json("placeholder", capsys)
    osnet = _eval_json("osnet-lite", capsys)
    assert placeholder["evidence"] is False
    assert osnet["evidence"] is False
    assert placeholder["note"] == _NOTE
    assert osnet["note"] == _NOTE
    assert "Not an AC-06 result" in str(osnet["note"])
    assert placeholder["num_valid_query"] == 16
    assert osnet["num_valid_query"] == 16
    assert placeholder["num_gallery"] == 16
    assert osnet["num_gallery"] == 16
    assert placeholder["rank1"] == 0.125
    assert placeholder["mAP"] == 0.2686762889887889
    assert osnet["rank1"] == 0.5
    assert osnet["mAP"] == 0.7083333333333331
    assert osnet["rank1"] > placeholder["rank1"]
    assert osnet["mAP"] > placeholder["mAP"]
    assert not (osnet["rank1"] == 1.0 and osnet["mAP"] == 1.0)


def _eval_json(embedder: str, capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    code = eval_module.main(["--embedder", embedder, "--synthetic-seed", "0", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert isinstance(payload, dict)
    return payload


def _mean_gap(samples: list, left: int, right: int) -> float:
    gap = np.abs(_cam0(samples, left).astype(np.int16) - _cam0(samples, right).astype(np.int16))
    return float(np.mean(gap))


def _cam0(samples: list, label: int) -> np.ndarray:
    for sample in samples:
        if sample.label == label and sample.camera_id == "cam-0" and sample.crop is not None:
            return sample.crop
    raise AssertionError(f"missing cam-0 crop for label {label}")


def _one_crop_per_camera(samples: list, n_cameras: int) -> list[np.ndarray]:
    found: dict[str, np.ndarray] = {}
    for sample in samples:
        if sample.crop is not None and sample.camera_id not in found:
            found[sample.camera_id] = sample.crop
    return [found[f"cam-{camera}"] for camera in range(n_cameras)]


def _camera_pairs(n_cameras: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left in range(n_cameras):
        for right in range(left + 1, n_cameras):
            pairs.append((left, right))
    return pairs


def _max_gap(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.abs(left.astype(np.int16) - right.astype(np.int16)).max())


def _pixels_over(left: np.ndarray, right: np.ndarray, bound: int) -> int:
    diff = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return int(np.any(diff > bound, axis=2).sum())


@pytest.mark.parametrize(
    ("kwargs", "phrase"),
    [
        ({"n_groups": 0}, "n_groups"),
        ({"per_group": 0}, "per_group"),
        ({"n_cameras": 0}, "n_cameras"),
        ({"size": (0, 32)}, "size"),
        ({"size": (64, 0)}, "size"),
        ({"n_groups": 1000}, "999"),
        ({"n_groups": 10, "per_group": 100}, "999"),
    ],
)
def test_generator_rejects_counts_and_size(kwargs: dict[str, object], phrase: str) -> None:
    with pytest.raises(ValueError, match=phrase):
        make_synthetic_reid_set(**kwargs)  # type: ignore[arg-type]


def test_manifest_has_one_generator_row_for_synthetic_py() -> None:
    row = _generator_row()
    assert row["name"] == _GENERATOR_NAME
    assert row["type"] == "generator"
    assert row["license"] == "MIT"
    assert row["generator_license"] == "MIT"
    assert isinstance(row["version"], str) and row["version"] != ""
    assert isinstance(row["seed"], str) and row["seed"] != ""
    assert isinstance(row["source_url"], str) and row["source_url"].startswith("https://")
    notes = row["notes"]
    assert isinstance(notes, str)
    assert "in-memory" in notes.lower() or "In-memory" in notes
    assert "use" not in row
    assert "fixture_id" not in row
    assert _schema_errors(_manifest()) == []


def test_manifest_sha256_matches_synthetic_py() -> None:
    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()
    assert _generator_row()["sha256"] == digest
    assert len(digest) == 64


def test_generator_row_without_seed_fails_schema() -> None:
    row = dict(_generator_row())
    del row["seed"]
    errors = _schema_errors(_with_generator_row(row))
    assert errors
    assert any("seed" in error for error in errors)


def test_generator_row_with_empty_generator_license_fails_schema() -> None:
    row = dict(_generator_row())
    row["generator_license"] = ""
    errors = _schema_errors(_with_generator_row(row))
    assert errors
    assert any("generator_license" in error for error in errors)


def test_generator_row_with_bad_sha256_fails_schema() -> None:
    row = dict(_generator_row())
    row["sha256"] = "ABCDEF"
    errors = _schema_errors(_with_generator_row(row))
    assert errors
    assert any("sha256" in error for error in errors)


def test_generator_row_with_use_dev_fails_schema() -> None:
    row = dict(_generator_row())
    row["use"] = "DEV"
    errors = _schema_errors(_with_generator_row(row))
    assert errors
    assert any("DEV" in error or "use" in error for error in errors)


def test_generator_row_with_fixture_id_fails_schema() -> None:
    row = dict(_generator_row())
    row["fixture_id"] = "FX-SYNREID-001"
    errors = _schema_errors(_with_generator_row(row))
    assert errors
    assert any("fixture_id" in error for error in errors)
