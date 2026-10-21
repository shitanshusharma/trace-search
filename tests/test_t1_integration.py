"""
Integration tests for T1: Smoke fixtures with separation guarantees.

These tests demonstrate that T1 requirements are met:
1. Deterministic generator creates ≥5 clips
2. Loader guards reject SMOKE_ONLY inputs (by content)
3. Separation check finds planted copies
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from trace_search.fixtures.generator import (
    SyntheticVideoGenerator,
    generate_smoke_fixtures,
)
from trace_search.fixtures.guards import (
    FixtureSeparationError,
    LoaderContext,
    check_fixture_allowed,
)
from trace_search.fixtures.separation import check_separation
from trace_search.fixtures.smoke_rows import is_generated_smoke_clip


def test_t1_complete_workflow():
    """
    End-to-end T1 workflow:
    1. Generate fixtures with deterministic seeds
    2. Verify manifest entries
    3. Test loader guards
    4. Test separation check
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        output_dir = tmpdir / "smoke"
        manifest_entries = generate_smoke_fixtures(
            output_dir=output_dir,
            base_seed=42,
            num_clips=5,
        )

        assert len(manifest_entries) == 5, "Should generate 5 clips"

        for entry in manifest_entries:
            assert entry["use"] == "SMOKE_ONLY"
            assert entry["fixture_id"].startswith("FX-SMOKE-SYN-")
            assert len(entry["sha256"]) == 64
            assert len(entry["phash"]) == 10

        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": manifest_entries,
        }
        manifest_path = tmpdir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        train_dir = tmpdir / "train"
        train_dir.mkdir()

        dummy_file = train_dir / "dummy.txt"
        dummy_file.write_text("test")

        result = check_separation(manifest_path, [dummy_file], repo_root=tmpdir)
        assert result["passed"], "Separation should pass on non-video files"


def test_t1_loader_rejects_planted_smoke():
    """
    Demonstrate: loader guard rejects a planted SMOKE_ONLY file by content.

    This proves AC-02b requirement: loader raises hard error on SMOKE_ONLY input
    identified by SHA-256.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        gen = SyntheticVideoGenerator(seed=42)
        smoke_file = tmpdir / "test.mp4"
        gen.generate_clip(
            output_path=smoke_file,
            duration_sec=2.0,
            fps=10,
            width=640,
            height=480,
        )

        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "Test smoke",
                    "version": "1.0.0",
                    "type": "synthetic_clip",
                    "license": "MIT",
                    "source_url": "https://example.com",
                    "sha256": gen.compute_sha256(smoke_file),
                    "license_version": "N/A",
                    "date_checked_ist": "2026-10-20",
                    "phash": gen.compute_phashes(smoke_file, num_samples=10),
                    "use": "SMOKE_ONLY",
                    "fixture_id": "FX-SMOKE-SYN-999",
                    "generator": "Test",
                    "generator_version": "1.0",
                    "generator_license": "MIT",
                    "seed": "test",
                }
            ],
        }
        manifest_path = tmpdir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with pytest.raises(FixtureSeparationError, match="FORBIDDEN"):
            check_fixture_allowed(
                smoke_file,
                LoaderContext.TRAINING,
                manifest_path,
            )

        with pytest.raises(FixtureSeparationError, match="FORBIDDEN"):
            check_fixture_allowed(
                smoke_file,
                LoaderContext.EVALUATION,
                manifest_path,
            )

        with pytest.raises(FixtureSeparationError, match="FORBIDDEN"):
            check_fixture_allowed(
                smoke_file,
                LoaderContext.STUDY,
                manifest_path,
            )


def test_t1_separation_finds_exact_copy():
    """
    Demonstrate: separation check finds a planted exact copy.

    This proves AC-02b requirement: SHA-256 match detection.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        smoke_dir = tmpdir / "smoke"
        smoke_dir.mkdir()

        gen = SyntheticVideoGenerator(seed=99)
        smoke_clip = smoke_dir / "FX-SMOKE-SYN-999.mp4"
        gen.generate_clip(
            output_path=smoke_clip,
            duration_sec=2.0,
            fps=10,
            width=640,
            height=480,
        )

        sha256 = gen.compute_sha256(smoke_clip)

        manifest = {
            "manifest_version": "1.0",
            "last_updated_ist": "2026-10-20",
            "entries": [
                {
                    "name": "Planted smoke",
                    "version": "1.0.0",
                    "type": "synthetic_clip",
                    "license": "MIT",
                    "source_url": "https://example.com",
                    "sha256": sha256,
                    "license_version": "N/A",
                    "date_checked_ist": "2026-10-20",
                    "phash": ["0" * 16] * 10,
                    "use": "SMOKE_ONLY",
                    "fixture_id": "FX-SMOKE-SYN-999",
                    "generator": "Test",
                    "generator_version": "1.0",
                    "generator_license": "MIT",
                    "seed": "test",
                }
            ],
        }
        manifest_path = tmpdir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        train_dir = tmpdir / "train"
        train_dir.mkdir()
        planted_copy = train_dir / "planted.mp4"
        shutil.copy(smoke_clip, planted_copy)

        result = check_separation(manifest_path, [planted_copy])

        assert not result["passed"], "Should FAIL on planted exact copy"
        assert len(result["exact_matches"]) == 1
        assert "FX-SMOKE-SYN-999" in result["exact_matches"][0][1]


def test_t1_determinism_same_seed_same_sha():
    """
    Demonstrate: same seed produces same SHA-256.

    This proves the generator is deterministic.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        gen1 = SyntheticVideoGenerator(seed=42)
        path1 = tmpdir / "test1.mp4"
        gen1.generate_clip(
            output_path=path1,
            duration_sec=2.0,
            fps=10,
            width=640,
            height=480,
        )
        sha1 = gen1.compute_sha256(path1)

        gen2 = SyntheticVideoGenerator(seed=42)
        path2 = tmpdir / "test2.mp4"
        gen2.generate_clip(
            output_path=path2,
            duration_sec=2.0,
            fps=10,
            width=640,
            height=480,
        )
        sha2 = gen2.compute_sha256(path2)

        assert sha1 == sha2, "Same seed MUST produce identical SHA-256"
        assert path1.read_bytes() == path2.read_bytes()


def _assert_regenerated_matches(manifest: dict) -> None:
    """Hash every generated smoke row. The same predicate selects those rows."""
    committed = [entry for entry in manifest["entries"] if is_generated_smoke_clip(entry)]
    assert committed
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first_dir = root / "smoke"
        second_dir = root / "again" / "smoke"
        clip_count = max(len(committed), 5)
        first = generate_smoke_fixtures(output_dir=first_dir, base_seed=42, num_clips=clip_count)
        second = generate_smoke_fixtures(output_dir=second_dir, base_seed=42, num_clips=clip_count)
        assert [entry["fixture_id"] for entry in first] == [
            entry["fixture_id"] for entry in committed
        ]
        for entry, repeat, stored in zip(first, second, committed):
            assert entry == stored
            assert repeat == stored
            assert entry["path"].startswith("smoke/")
            assert "\\" not in entry["path"]
            assert not entry["path"].startswith("/")
            assert ".." not in Path(entry["path"]).parts
            left = first_dir / Path(entry["path"]).name
            right = second_dir / Path(entry["path"]).name
            assert left.read_bytes() == right.read_bytes()
            assert SyntheticVideoGenerator.compute_sha256(left) == stored["sha256"]


def test_regenerate_matches_manifest_entries_byte_for_byte():
    """The generator's rows, including path, match the committed manifest.

    A second run writes the same clip bytes. The manifest SHA-256 is that
    byte string. Every generated smoke row is included, including a later one.
    """
    manifest = json.loads(Path("licenses/manifest.json").read_text(encoding="utf-8"))
    _assert_regenerated_matches(manifest)


def test_planted_tampered_synthetic_sha_fails_regenerate():
    """A generated row with a replaced sha256 fails the regenerate comparison."""
    manifest = json.loads(Path("licenses/manifest.json").read_text(encoding="utf-8"))
    tampered = False
    for entry in manifest["entries"]:
        if is_generated_smoke_clip(entry):
            entry["sha256"] = "0" * 64
            tampered = True
            break
    assert tampered
    with pytest.raises(AssertionError):
        _assert_regenerated_matches(manifest)


def test_regenerate_lowlight_matches_manifest_bytes():
    """Same seed regenerates FX-SMOKE-SYN-002 byte-for-byte against the manifest."""
    manifest = json.loads(Path("licenses/manifest.json").read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["entries"] if item.get("fixture_id") == "FX-SMOKE-SYN-002"
    )
    seed = json.loads(entry["seed"])
    config = seed["config"]
    with tempfile.TemporaryDirectory() as tmp:
        first = Path(tmp) / "a.mp4"
        second = Path(tmp) / "b.mp4"
        kwargs = dict(
            duration_sec=config["duration_sec"],
            fps=30,
            width=1920,
            height=1080,
            num_figures=config["num_figures"],
            background_brightness=config["background_brightness"],
            num_obstructions=config["num_obstructions"],
        )
        gen_a = SyntheticVideoGenerator(seed=seed["base_seed"] + seed["offset"])
        gen_a.generate_clip(output_path=first, **kwargs)
        gen_b = SyntheticVideoGenerator(seed=seed["base_seed"] + seed["offset"])
        gen_b.generate_clip(output_path=second, **kwargs)
        assert first.read_bytes() == second.read_bytes()
        assert gen_a.compute_sha256(first) == entry["sha256"]
        assert gen_a.compute_phashes(first, num_samples=10) == entry["phash"]


def test_smoke_store_is_gitignored_and_untracked():
    """Generated media stays out of git. The test does not skip when smoke/ is absent."""
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "smoke/" in gitignore
    assert "*.mp4" in gitignore
    assert "*.avi" in gitignore
    tracked = subprocess.check_output(["git", "ls-files"], text=True)
    media = [
        line
        for line in tracked.splitlines()
        if line.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    ]
    assert media == []


def test_t1_manifest_has_smoke_entries():
    """Verify manifest contains SMOKE_ONLY entries."""
    manifest_path = Path("licenses/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    smoke_entries = [e for e in manifest["entries"] if e.get("use") == "SMOKE_ONLY"]

    assert len(smoke_entries) >= 5, f"Should have ≥5 SMOKE_ONLY entries, found {len(smoke_entries)}"

    for entry in smoke_entries:
        assert entry["type"] == "synthetic_clip"
        assert entry["generator"] == "SyntheticVideoGenerator"
        assert "fixture_id" in entry
        assert entry["fixture_id"].startswith("FX-SMOKE-SYN-")
        assert "sha256" in entry
        assert "phash" in entry
        assert len(entry["phash"]) == 10
