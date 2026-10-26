"""The in-memory synthetic ReID set and its generator manifest row.

Arrays are built in the test process. Nothing is read from ``smoke/``
and nothing is written into the repo tree.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import scripts.check_licenses as check_licenses
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
    assert len(first) == 16
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
        f"FX-SYNREID-{number:03d}" for number in range(1, 17)
    ]
    assert [sample.use for sample in samples] == ["DEV"] * len(samples)
    assert {sample.camera_id for sample in samples} == {"cam-0", "cam-1"}
    assert {sample.label for sample in samples} == {0, 1, 2, 3}
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
    with pytest.raises(FixtureSeparationError, match="not on the ReID allowlist"):
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


def test_generator_does_not_write_files(tmp_path: Path) -> None:
    before_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    before_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=_REPO)
    make_synthetic_reid_set()
    after_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    after_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=_REPO)
    assert before_files == after_files
    assert before_status == after_status


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
    completed = subprocess.run(
        ["sha256sum", str(_SOURCE)],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = completed.stdout.split()[0]
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
