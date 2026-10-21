"""Separation check: data files only, fail closed, near-duplicates use mp4v."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.media_helpers import write_mp4v_variant
from trace_search.fixtures.content import sha256_file
from trace_search.fixtures.generator import SyntheticVideoGenerator
from trace_search.fixtures.separation import (
    SeparationCheckError,
    auto_discover_data_files,
    check_separation,
    run_separation_check,
)


def _clip(
    path: Path,
    seed: int,
    figures: int = 2,
    brightness: float = 1.0,
) -> SyntheticVideoGenerator:
    gen = SyntheticVideoGenerator(seed=seed)
    gen.generate_clip(
        output_path=path,
        duration_sec=2.0,
        fps=10,
        width=640,
        height=480,
        num_figures=figures,
        background_brightness=brightness,
    )
    return gen


def _manifest(
    path: Path,
    sha: str,
    phashes: list[str],
    fixture_id: str,
    use: str = "SMOKE_ONLY",
) -> None:
    path.write_text(
        json.dumps(
            {
                "manifest_version": "1.0",
                "last_updated_ist": "2026-10-20",
                "entries": [
                    {
                        "name": "fixture",
                        "version": "1.0.0",
                        "type": "synthetic_clip",
                        "license": "MIT",
                        "source_url": "https://example.com",
                        "sha256": sha,
                        "license_version": "N/A",
                        "date_checked_ist": "2026-10-20",
                        "phash": phashes,
                        "use": use,
                        "fixture_id": fixture_id,
                        "generator": "SyntheticVideoGenerator",
                        "generator_version": "1.0.0",
                        "generator_license": "MIT",
                        "seed": "1",
                    }
                ],
            }
        )
    )


def test_near_duplicate_reencoded_mp4v_flagged(tmp_path: Path):
    """A resized mp4v re-encode is a near-duplicate; its SHA-256 differs."""
    original = tmp_path / "original.mp4"
    gen = _clip(original, seed=42)
    sha = gen.compute_sha256(original)
    phashes = gen.compute_phashes(original, num_samples=10)
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, sha, phashes, "FX-SMOKE-SYN-999")

    reencoded = tmp_path / "train" / "reencoded.mp4"
    write_mp4v_variant(original, reencoded, scale=0.5)
    assert sha256_file(reencoded) != sha

    result = check_separation(manifest, [reencoded], repo_root=tmp_path)
    assert result["passed"] is False
    assert result["near_duplicates"]
    assert result["near_duplicates"][0][1] == "FX-SMOKE-SYN-999"
    assert result["near_duplicates"][0][2] <= 8


def test_near_duplicate_unrelated_not_flagged(tmp_path: Path):
    original = tmp_path / "original.mp4"
    gen = _clip(original, seed=42, figures=2, brightness=1.0)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-997",
    )
    unrelated = tmp_path / "train" / "unrelated.mp4"
    other = _clip(unrelated, seed=999, figures=6, brightness=0.25)
    assert other.compute_sha256(unrelated) != gen.compute_sha256(original)

    result = check_separation(manifest, [unrelated], repo_root=tmp_path)
    assert result["passed"] is True
    assert result["exact_matches"] == []
    assert result["near_duplicates"] == []


def test_copy_outside_fixtures_dir_fails(tmp_path: Path):
    """A copy under data/train/smoke/ is a leak. The path part smoke is not a skip."""
    store = tmp_path / "smoke"
    store.mkdir()
    original = store / "FX-SMOKE-SYN-001.mp4"
    gen = _clip(original, seed=7)
    manifest = tmp_path / "licenses" / "manifest.json"
    manifest.parent.mkdir()
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-001",
    )
    leaked = tmp_path / "data" / "train" / "smoke" / "clip.mp4"
    leaked.parent.mkdir(parents=True)
    leaked.write_bytes(original.read_bytes())

    result = run_separation_check(manifest, repo_root=tmp_path)
    assert result["passed"] is False
    assert any(item[1] == "FX-SMOKE-SYN-001" for item in result["exact_matches"])


def test_renamed_dat_copy_fails(tmp_path: Path):
    original = tmp_path / "smoke" / "clip.mp4"
    original.parent.mkdir()
    gen = _clip(original, seed=8)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-002",
    )
    renamed = tmp_path / "notes" / "payload.dat"
    renamed.parent.mkdir()
    renamed.write_bytes(original.read_bytes())

    listed = check_separation(manifest, [renamed], repo_root=tmp_path)
    assert listed["passed"] is False
    assert listed["exact_matches"][0][1] == "FX-SMOKE-SYN-002"

    discovered = run_separation_check(manifest, repo_root=tmp_path)
    assert discovered["passed"] is False
    assert any(item[1] == "FX-SMOKE-SYN-002" for item in discovered["exact_matches"])


def test_renamed_txt_inside_train_fails(tmp_path: Path):
    """A clip copied to data/train/clip.txt is still SHA-checked and fails."""
    original = tmp_path / "smoke" / "clip.mp4"
    original.parent.mkdir()
    gen = _clip(original, seed=21)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-021",
    )
    leaked = tmp_path / "data" / "train" / "clip.txt"
    leaked.parent.mkdir(parents=True)
    leaked.write_bytes(original.read_bytes())

    result = run_separation_check(manifest, repo_root=tmp_path)
    assert result["passed"] is False
    assert result["files_checked"] >= 1
    assert any(item[1] == "FX-SMOKE-SYN-021" for item in result["exact_matches"])


@pytest.mark.parametrize(
    "relative",
    [
        "eval/clip.txt",
        "val/clip.txt",
        "holdout/clip.txt",
        "test/clip.txt",
        "training/clip.txt",
        "Data/Train/clip.txt",
        "TRAIN/clip.txt",
    ],
)
def test_renamed_clip_under_eval_and_cased_roots_fails(tmp_path: Path, relative: str):
    """A smoke clip under any train/eval/study root fails, whatever the case or suffix."""
    original = tmp_path / "smoke" / "clip.mp4"
    original.parent.mkdir()
    gen = _clip(original, seed=34)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-034",
    )
    leaked = tmp_path / relative
    leaked.parent.mkdir(parents=True)
    leaked.write_bytes(original.read_bytes())

    result = run_separation_check(manifest, repo_root=tmp_path)
    assert result["passed"] is False
    assert result["files_checked"] >= 1
    assert any(item[1] == "FX-SMOKE-SYN-034" for item in result["exact_matches"])


def test_clean_tree_passes(tmp_path: Path):
    """A tree with only the smoke store and source files passes."""
    original = tmp_path / "smoke" / "clip.mp4"
    original.parent.mkdir()
    gen = _clip(original, seed=35)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-035",
    )
    source = tmp_path / "src" / "trace_search" / "fixtures"
    source.mkdir(parents=True)
    (source / "helper.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("clean\n")

    result = run_separation_check(manifest, repo_root=tmp_path)
    assert result["passed"] is True
    assert result["exact_matches"] == []
    assert result["near_duplicates"] == []
    assert result["files_checked"] == 0


def test_image_copy_and_reencode_are_flagged(tmp_path: Path):
    """Images get a SHA-256 and a pHash check, including a .dat rename."""
    from PIL import Image

    from trace_search.fixtures.content import perceptual_hashes

    original = tmp_path / "smoke" / "frame.png"
    original.parent.mkdir()
    image = Image.new("RGB", (128, 128))
    pixels = image.load()
    for y in range(128):
        for x in range(128):
            pixels[x, y] = ((x * 2) % 256, (y * 2) % 256, (x + y) % 256)
    image.save(original)
    sha = sha256_file(original)
    phashes = perceptual_hashes(original)
    assert len(phashes) == 1
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, sha, phashes, "FX-SMOKE-SYN-012")

    copied = tmp_path / "data" / "train" / "frame.png"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original.read_bytes())
    exact = check_separation(manifest, [copied], repo_root=tmp_path)
    assert exact["passed"] is False
    assert exact["exact_matches"][0][1] == "FX-SMOKE-SYN-012"

    shifted = image.copy()
    for x in range(2):
        for y in range(2):
            shifted.putpixel((x, y), (255, 0, 0))
    variant = tmp_path / "data" / "train" / "frame.dat"
    shifted.save(variant, format="PNG")
    assert sha256_file(variant) != sha
    near = check_separation(manifest, [variant], repo_root=tmp_path)
    assert near["passed"] is False
    assert near["near_duplicates"]
    assert near["near_duplicates"][0][2] <= 8


def test_reencoded_dat_is_near_duplicate(tmp_path: Path):
    """pHash runs even when the suffix is .dat."""
    original = tmp_path / "original.mp4"
    gen = _clip(original, seed=11)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-003",
    )
    variant_mp4 = tmp_path / "outside" / "clip.mp4"
    variant = tmp_path / "outside" / "clip.dat"
    write_mp4v_variant(original, variant_mp4, scale=0.75)
    variant.write_bytes(variant_mp4.read_bytes())
    variant_mp4.unlink()
    assert sha256_file(variant) != gen.compute_sha256(original)

    result = check_separation(manifest, [variant], repo_root=tmp_path)
    assert result["passed"] is False
    assert result["near_duplicates"]
    assert result["near_duplicates"][0][2] <= 8


def test_separation_fail_closed_zero_files(tmp_path: Path):
    """Explicit empty input fails. The scan stays inside the temp tree."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": "1.0",
                "last_updated_ist": "2026-10-20",
                "entries": [],
            }
        )
    )
    with pytest.raises(SeparationCheckError, match="0 dataset files"):
        check_separation(manifest, [], repo_root=tmp_path)


def test_separation_fail_closed_missing_directory(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"entries": []}')
    missing = tmp_path / "no-such-train"
    with pytest.raises(SeparationCheckError, match="does not exist"):
        run_separation_check(manifest, train_dir=missing, repo_root=tmp_path)


def test_separation_fail_closed_empty_directory(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"entries": []}')
    empty = tmp_path / "train"
    empty.mkdir()
    with pytest.raises(SeparationCheckError, match="0 dataset files"):
        run_separation_check(manifest, train_dir=empty, repo_root=tmp_path)


def test_missing_file_fails(tmp_path: Path):
    original = tmp_path / "original.mp4"
    gen = _clip(original, seed=3)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-004",
    )
    missing = tmp_path / "gone.mp4"
    with pytest.raises(SeparationCheckError, match="missing file"):
        check_separation(manifest, [missing], repo_root=tmp_path)


def test_corrupt_file_fails(tmp_path: Path):
    original = tmp_path / "original.mp4"
    gen = _clip(original, seed=4)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-005",
    )
    corrupt = tmp_path / "data" / "corrupt.mp4"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"this is not a video")
    with pytest.raises(SeparationCheckError, match="FAIL-CLOSED"):
        check_separation(manifest, [corrupt], repo_root=tmp_path)


def test_decode_error_fails(tmp_path: Path):
    original = tmp_path / "original.mp4"
    gen = _clip(original, seed=5)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-006",
    )
    other = tmp_path / "other.mp4"
    _clip(other, seed=50)

    class _Broken:
        def isOpened(self):  # noqa: N802 - OpenCV capture API
            return False

        def release(self):
            return None

    with patch("trace_search.fixtures.content.cv2.VideoCapture", return_value=_Broken()):
        with pytest.raises(SeparationCheckError, match="decode failed"):
            check_separation(manifest, [other], repo_root=tmp_path)


def test_phash_error_fails(tmp_path: Path):
    original = tmp_path / "original.mp4"
    gen = _clip(original, seed=6)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-007",
    )
    other = tmp_path / "other.mp4"
    _clip(other, seed=60)
    with patch(
        "trace_search.fixtures.content.imagehash.phash",
        side_effect=RuntimeError("phash exploded"),
    ):
        with pytest.raises(SeparationCheckError, match="pHash failed"):
            check_separation(manifest, [other], repo_root=tmp_path)


def test_manifest_without_forbidden_rows_fails(tmp_path: Path):
    clip = tmp_path / "data" / "clip.mp4"
    clip.parent.mkdir()
    _clip(clip, seed=12)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": "1.0",
                "last_updated_ist": "2026-10-20",
                "entries": [
                    {
                        "name": "Pillow",
                        "version": "13.0.0",
                        "type": "library",
                        "license": "MIT-CMU",
                        "source_url": "https://pypi.org/project/Pillow/",
                        "sha256": "a" * 64,
                    }
                ],
            }
        )
    )
    with pytest.raises(SeparationCheckError, match="no SMOKE_ONLY or CODE_TEST"):
        check_separation(manifest, [clip], repo_root=tmp_path)


def test_one_step_training_loop_reading_smoke_fails(tmp_path: Path):
    original = tmp_path / "smoke" / "clip.mp4"
    original.parent.mkdir()
    gen = _clip(original, seed=13)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-008",
    )
    (tmp_path / "fit_model.py").write_text(
        "def train():\n"
        "    blob = open('smoke/clip.mp4', 'rb').read()\n"
        "    for _ in range(1):\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
    )
    with pytest.raises(SeparationCheckError, match="training step"):
        run_separation_check(manifest, repo_root=tmp_path)


def test_optimizer_that_does_not_read_smoke_passes(tmp_path: Path):
    original = tmp_path / "smoke" / "clip.mp4"
    original.parent.mkdir()
    gen = _clip(original, seed=14)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-009",
    )
    (tmp_path / "step_once.py").write_text(
        "def train():\n    for _ in range(1):\n        loss.backward()\n        optimizer.step()\n"
    )
    result = run_separation_check(manifest, repo_root=tmp_path)
    assert result["passed"] is True
    assert result["files_checked"] == 0


def test_config_referencing_smoke_fails(tmp_path: Path):
    original = tmp_path / "smoke" / "clip.mp4"
    original.parent.mkdir()
    gen = _clip(original, seed=15)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-010",
    )
    (tmp_path / "train_config.yaml").write_text("dataset: smoke/clip.mp4\n")
    with pytest.raises(SeparationCheckError, match="config or playlist"):
        run_separation_check(manifest, repo_root=tmp_path)


def test_auto_discovery_ignores_py_source(tmp_path: Path):
    fixtures = tmp_path / "src" / "trace_search" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "helper.py").write_text("x = 1\n")
    original = tmp_path / "smoke" / "clip.mp4"
    original.parent.mkdir()
    gen = _clip(original, seed=16)
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        gen.compute_sha256(original),
        gen.compute_phashes(original, num_samples=10),
        "FX-SMOKE-SYN-011",
    )
    discovered = auto_discover_data_files(tmp_path)
    assert all(path.suffix != ".py" for path in discovered)
    assert discovered == []
