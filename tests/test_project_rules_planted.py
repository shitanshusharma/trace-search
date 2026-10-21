"""
Planted failure tests for project rules checker.

These tests verify that the project rules checker catches various violations:
- AGPL dependencies
- Banned imports (insightface, ultralytics, etc.)
- Banned identifiers and function calls (watchlist, add_to_watchlist, face_embeddings, etc.)
- Typed name fields and assignments
- Research dataset references (market1501, msmt17, duke, cuhk03, casia-b, grew, gait3d, objects365)
- Media files that miss entries[].path or sha256, including allow-list bypasses
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def get_checker_path() -> Path:
    """Get absolute path to check_project_rules.py."""
    # Navigate from tests/ to scripts/
    return Path(__file__).parents[1] / "scripts" / "check_project_rules.py"


def test_planted_agpl_dependency():
    """PLANTED FAILURE: AGPL dependency (ultralytics) should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create a requirements file with ultralytics (AGPL)
        req_file = tmppath / "requirements.txt"
        req_file.write_text("ultralytics==8.0.0\n")

        # Create a minimal src directory
        src_dir = tmppath / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on AGPL dependency"
        assert "ultralytics" in result.stdout.lower() or "banned" in result.stdout.lower()


def test_planted_insightface_import():
    """PLANTED FAILURE: insightface import should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory with banned import
        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text("import insightface\n\ndef main():\n    pass\n")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on insightface import"
        assert "insightface" in result.stdout.lower() or "banned" in result.stdout.lower()


def test_planted_watchlist_identifier():
    """PLANTED FAILURE: watchlist identifier should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory with banned identifier
        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text("watchlist = []\n")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on watchlist identifier"
        assert "watchlist" in result.stdout.lower() or "banned" in result.stdout.lower()


def test_planted_watchlist_function_call():
    """PLANTED FAILURE: add_to_watchlist function call should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory with banned function call
        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text("add_to_watchlist(person_id, data)\n")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on add_to_watchlist call"
        assert "watchlist" in result.stdout.lower() or "banned" in result.stdout.lower()


def test_planted_enroll_person_call():
    """PLANTED FAILURE: enroll_person function call should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory with banned function call
        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text("enroll_person(name, features)\n")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on enroll_person call"
        assert "enroll" in result.stdout.lower() or "banned" in result.stdout.lower()


def test_planted_face_embeddings_identifier():
    """PLANTED FAILURE: face_embeddings identifier should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory with banned identifier
        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text("face_embeddings = model.extract(image)\n")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on face_embeddings identifier"
        assert "face" in result.stdout.lower() or "banned" in result.stdout.lower()


def test_planted_name_field_typed():
    """PLANTED FAILURE: typed name: str field should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory with banned name field
        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text("name: str\n")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on name: str field"
        assert "name" in result.stdout.lower() or "banned" in result.stdout.lower()


def test_planted_name_assignment():
    """PLANTED FAILURE: name: str = value assignment should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory with banned assignment
        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text('name: str = "John Doe"\n')

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on name: str = assignment"
        assert "name" in result.stdout.lower() or "banned" in result.stdout.lower()


def test_planted_name_string_assignment():
    """PLANTED FAILURE: name = "..." string assignment should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory with banned assignment
        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text('name = "John Doe"\n')

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on name = string assignment"
        assert "name" in result.stdout.lower() or "banned" in result.stdout.lower()


def test_planted_market1501_reference():
    """PLANTED FAILURE: market1501 dataset reference should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory with dataset reference
        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text('DATASET = "market1501"\n')

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on market1501 reference"
        assert "market" in result.stdout.lower() or "dataset" in result.stdout.lower()


def test_planted_msmt17_reference():
    """PLANTED FAILURE: MSMT17 dataset reference should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text('weights = load_weights("msmt17_reid.pth")\n')

        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on msmt17 reference"
        assert "msmt17" in result.stdout.lower() or "dataset" in result.stdout.lower()


def test_planted_dukemtmc_reference():
    """PLANTED FAILURE: DukeMTMC dataset reference should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text('DATASET_PATH = "/data/DukeMTMC-reID"\n')

        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on DukeMTMC reference"
        assert "duke" in result.stdout.lower() or "dataset" in result.stdout.lower()


def test_planted_cuhk03_reference():
    """PLANTED FAILURE: CUHK03 dataset reference should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text("# Trained on CUHK03\n")

        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on CUHK03 reference"
        assert "cuhk" in result.stdout.lower() or "dataset" in result.stdout.lower()


def test_planted_casiab_reference():
    """PLANTED FAILURE: CASIA-B dataset reference should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text('gait_dataset = "CASIA-B"\n')

        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on CASIA-B reference"
        assert "casia" in result.stdout.lower() or "dataset" in result.stdout.lower()


def test_planted_grew_reference():
    """PLANTED FAILURE: GREW dataset reference should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text("from datasets import GREW\n")

        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on GREW reference"
        assert "grew" in result.stdout.lower() or "dataset" in result.stdout.lower()


def test_planted_gait3d_reference():
    """PLANTED FAILURE: Gait3D dataset reference should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text('model_path = "gait3d_checkpoint.pth"\n')

        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on Gait3D reference"
        assert "gait3d" in result.stdout.lower() or "dataset" in result.stdout.lower()


def test_planted_objects365_reference():
    """PLANTED FAILURE: Objects365 dataset reference should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text("# Pretrained on Objects365\n")

        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on Objects365 reference"
        assert "objects365" in result.stdout.lower() or "dataset" in result.stdout.lower()


def test_planted_mp4_file():
    """PLANTED FAILURE: .mp4 file outside fixtures should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory
        src_dir = tmppath / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")

        # Create a media file outside allowed paths
        media_file = tmppath / "test_video.mp4"
        media_file.write_bytes(b"fake mp4 content")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on .mp4 file outside fixtures"
        assert "mp4" in result.stdout.lower() or "media" in result.stdout.lower()


def test_planted_uppercase_mp4_file():
    """PLANTED FAILURE: .MP4 file (uppercase) outside fixtures should be caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory
        src_dir = tmppath / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")

        # Create a media file with uppercase extension
        media_file = tmppath / "VIDEO.MP4"
        media_file.write_bytes(b"fake mp4 content")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on .MP4 file (case-insensitive)"
        assert "MP4" in result.stdout or "mp4" in result.stdout.lower()


def test_clean_case_passes():
    """Clean case with no violations should pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create src directory with clean code
        src_dir = tmppath / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text("def process_data(data):\n    return data\n")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Clean code should pass. Output: {result.stdout}"


def test_planted_manifested_fixture_passes():
    """PLANTED SUCCESS: A media file in manifest with matching SHA should pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create licenses directory and manifest
        licenses_dir = tmppath / "licenses"
        licenses_dir.mkdir()

        # Create a dummy video file
        fixtures_dir = tmppath / "fixtures"
        fixtures_dir.mkdir()
        video_file = fixtures_dir / "test.mp4"
        video_content = b"fake mp4 content for testing"
        video_file.write_bytes(video_content)

        # Calculate SHA256
        import hashlib

        sha256_hash = hashlib.sha256(video_content).hexdigest()

        # Create manifest with this file. The gate reads entries[].path.
        manifest = {
            "manifest_version": "1.0",
            "entries": [
                {
                    "name": "test clip",
                    "version": "1",
                    "type": "clip",
                    "license": "Pexels License",
                    "source_url": "https://example.com/clip.mp4",
                    "sha256": sha256_hash,
                    "path": "fixtures/test.mp4",
                    "use": "CODE_TEST",
                }
            ],
        }
        manifest_file = licenses_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))

        # Create src directory
        src_dir = tmppath / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Manifested fixture should pass. Output: {result.stdout}"


def test_planted_unmanifested_mp4_fails():
    """PLANTED FAILURE: .mp4 file not in manifest should fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create licenses directory with empty manifest
        licenses_dir = tmppath / "licenses"
        licenses_dir.mkdir()
        manifest = {"manifest_version": "1.0", "entries": []}
        manifest_file = licenses_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))

        # Create src directory
        src_dir = tmppath / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")

        # Create an unmanifested media file
        media_file = tmppath / "test_video.mp4"
        media_file.write_bytes(b"fake mp4 content")

        # Create allow-list file
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        # Run checker
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Should fail on unmanifested .mp4 file"
        assert "mp4" in result.stdout.lower() or "media" in result.stdout.lower()


def test_camera_name_file_name_and_self_name_pass():
    """Ordinary names must pass. The name rule is the bare identifier only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        src_dir = tmppath / "src"
        src_dir.mkdir()
        (src_dir / "ingest.py").write_text(
            "\n".join(
                [
                    "class Ingest:",
                    "    def run(self):",
                    '        self.name = "ingest"',
                    '        camera_name = "cam-01"',
                    '        file_name = "clip.mp4"',
                    '        model_name = "rtdetr"',
                    '        getLogger(name="x")',
                    "        return camera_name, file_name, model_name",
                    "",
                ]
            )
        )
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout


def test_face_crop_national_id_and_identity_id():
    """Non-alphanumeric boundaries catch these identifiers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        src_dir = tmppath / "src"
        src_dir.mkdir()
        (src_dir / "a.py").write_text("face_crop = img[0:10]\nnational_id = x\nidentity_id = 3\n")
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        lowered = result.stdout.lower()
        assert "face_crop" in lowered
        assert "national_id" in lowered
        assert "identity_id" in lowered


def test_exclusion_is_exact_path_not_suffix():
    """The planted fixture path is exempt. The same name under src/ still fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        fixture = tmppath / "tests"
        fixture.mkdir()
        (fixture / "test_project_rules_planted.py").write_text("face_embedding = 1\n")
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")

        only_fixture = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert only_fixture.returncode == 0, only_fixture.stdout

        src = tmppath / "src"
        src.mkdir()
        (src / "real.py").write_text("face_embedding = 1\n")
        with_src = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert with_src.returncode == 1

        nested = tmppath / "src" / "x" / "tests"
        nested.mkdir(parents=True)
        (nested / "test_package.py").write_text("import insightface\n")
        (src / "real.py").unlink()
        suffix_trap = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert suffix_trap.returncode == 1, suffix_trap.stdout


def test_license_checker_path_still_flags_real_imports():
    """scripts/check_licenses.py is not exempt. A real import there fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        scripts = tmppath / "scripts"
        scripts.mkdir()
        (scripts / "check_licenses.py").write_text("import insightface\n")
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "insightface" in result.stdout.lower()


def test_manifest_hash_is_tied_to_path():
    """A manifest hash does not authorize the same bytes at another path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        payload = b"same bytes"
        import hashlib

        digest = hashlib.sha256(payload).hexdigest()
        fixtures = tmppath / "fixtures"
        fixtures.mkdir()
        (fixtures / "test.mp4").write_bytes(payload)
        weights = tmppath / "src"
        weights.mkdir()
        (weights / "weights.pt").write_bytes(payload)
        licenses = tmppath / "licenses"
        licenses.mkdir()
        manifest = {
            "manifest_version": "1.0",
            "entries": [
                {
                    "name": "smoke clip",
                    "version": "1",
                    "type": "clip",
                    "license": "Pexels License",
                    "source_url": "https://example.com/clip.mp4",
                    "sha256": digest,
                    "path": "fixtures/test.mp4",
                    "use": "SMOKE_ONLY",
                }
            ],
        }
        (licenses / "manifest.json").write_text(json.dumps(manifest))
        (tmppath / ".project-rules-allow").write_text("# No exceptions\n")
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "src/weights.pt" in result.stdout
        assert "fixtures/test.mp4" not in result.stdout


def _load_checker():
    path = get_checker_path()
    spec = importlib.util.spec_from_file_location("check_project_rules_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unreadable_media_is_a_violation(monkeypatch, tmp_path):
    module = _load_checker()
    (tmp_path / "clip.mp4").write_bytes(b"abc")
    monkeypatch.chdir(tmp_path)

    def boom(_path):
        raise OSError("permission denied")

    monkeypatch.setattr(module, "hash_file", boom)
    violations = module.check_media_files(set())
    assert violations
    assert any("Unreadable" in item["message"] for item in violations)


def _run(tmppath: Path, files: dict[str, str]) -> subprocess.CompletedProcess[str]:
    for rel, content in files.items():
        path = tmppath / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    allow = tmppath / ".project-rules-allow"
    if not allow.exists():
        allow.write_text("# No exceptions\n")
    return subprocess.run(
        [sys.executable, str(get_checker_path())],
        cwd=tmppath,
        capture_output=True,
        text=True,
    )


def _clip_manifest(path: str, sha: str, use: str = "CODE_TEST") -> str:
    manifest = {
        "manifest_version": "1.0",
        "entries": [
            {
                "name": "clip",
                "version": "1",
                "type": "clip",
                "license": "Pexels License",
                "source_url": "https://example.com/clip.mp4",
                "sha256": sha,
                "path": path,
                "use": use,
            }
        ],
    }
    return json.dumps(manifest)


def test_function_name_add_to_watchlist_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run(Path(tmpdir), {"src/a.py": "def add_to_watchlist(x):\n    return x\n"})
    assert result.returncode == 1
    assert "watchlist" in result.stdout.lower()


def test_class_watchlist_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run(Path(tmpdir), {"src/a.py": "class Watchlist:\n    pass\n"})
    assert result.returncode == 1
    assert "watchlist" in result.stdout.lower()


def test_face_embedding_parameter_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = "def f(face_embedding):\n    return face_embedding\n"
        result = _run(Path(tmpdir), {"src/a.py": source})
    assert result.returncode == 1
    assert "face_embedding" in result.stdout


def test_face_embedding_keyword_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run(Path(tmpdir), {"src/a.py": "f(face_embedding=v)\n"})
    assert result.returncode == 1
    assert "face_embedding" in result.stdout


def test_dict_key_strings_fail():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = 'row = {"face_embedding": v, "national_id": n}\n'
        result = _run(Path(tmpdir), {"src/a.py": source})
    assert result.returncode == 1
    assert "face_embedding" in result.stdout
    assert "national_id" in result.stdout


def test_sql_string_national_id_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = 'SQL = "CREATE TABLE p (national_id TEXT, face_crop BLOB)"\n'
        result = _run(Path(tmpdir), {"src/a.py": source})
    assert result.returncode == 1
    lowered = result.stdout.lower()
    assert "national_id" in lowered
    assert "face_crop" in lowered


def test_import_alias_face_embedding_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run(Path(tmpdir), {"src/a.py": "import os as face_embedding\n"})
    assert result.returncode == 1
    assert "face_embedding" in result.stdout


def test_syntax_error_fails_closed():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run(Path(tmpdir), {"src/a.py": "import insightface\n(\n"})
    assert result.returncode == 1
    assert "failed to parse" in result.stdout.lower()


def test_semgrep_directory_does_not_exempt_imports():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run(Path(tmpdir), {"tests/semgrep/real.py": "import insightface\n"})
    assert result.returncode == 1
    assert "insightface" in result.stdout.lower()


def test_semgrep_rules_fixture_import_is_exact_exemption():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run(Path(tmpdir), {"tests/semgrep/rules.py": "import insightface\n"})
    assert result.returncode == 0, result.stdout


def test_entries_path_and_sha_match_passes():
    """A clip row whose path and sha256 match the file passes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        payload = b"clip-bytes"
        import hashlib

        digest = hashlib.sha256(payload).hexdigest()
        result = _run(
            tmppath,
            {
                "fixtures/test.mp4": "",
                "licenses/manifest.json": _clip_manifest("fixtures/test.mp4", digest),
            },
        )
        (tmppath / "fixtures" / "test.mp4").write_bytes(payload)
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0, result.stdout


def test_entries_sha_mismatch_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        bad = "a" * 64
        result = _run(
            tmppath,
            {
                "fixtures/test.mp4": "",
                "licenses/manifest.json": _clip_manifest("fixtures/test.mp4", bad),
            },
        )
        (tmppath / "fixtures" / "test.mp4").write_bytes(b"real-bytes")
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 1
    assert "fixtures/test.mp4" in result.stdout


def test_listed_path_missing_file_fails():
    """A manifest path whose file is not in the tree fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        digest = "c" * 64
        result = _run(
            tmppath,
            {
                "licenses/manifest.json": _clip_manifest(
                    "fixtures/missing.mp4", digest, "SMOKE_ONLY"
                ),
            },
        )
    assert result.returncode == 1, result.stdout
    assert "fixtures/missing.mp4" in result.stdout
    assert "missing" in result.stdout.lower()


def test_unlisted_mp4_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        result = _run(
            tmppath,
            {
                "clip.mp4": "",
                "licenses/manifest.json": json.dumps({"entries": []}),
            },
        )
        (tmppath / "clip.mp4").write_bytes(b"unlisted")
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 1
    assert "clip.mp4" in result.stdout


def test_files_key_does_not_allow_media():
    """The old files[] shape must not authorize a clip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        payload = b"same"
        import hashlib

        digest = hashlib.sha256(payload).hexdigest()
        manifest = {"files": [{"path": "fixtures/test.mp4", "sha256": digest, "use": "CODE_TEST"}]}
        result = _run(
            tmppath,
            {
                "fixtures/test.mp4": "",
                "licenses/manifest.json": json.dumps(manifest),
            },
        )
        (tmppath / "fixtures" / "test.mp4").write_bytes(payload)
        result = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 1
    assert "fixtures/test.mp4" in result.stdout


def test_parent_and_absolute_paths_do_not_authorize():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        payload = b"clip"
        import hashlib

        digest = hashlib.sha256(payload).hexdigest()
        for bad_path in ("../clip.mp4", "/tmp/clip.mp4"):
            manifest = _clip_manifest(bad_path, digest)
            (tmppath / "licenses").mkdir(exist_ok=True)
            (tmppath / "licenses" / "manifest.json").write_text(manifest)
            (tmppath / "clip.mp4").write_bytes(payload)
            (tmppath / ".project-rules-allow").write_text("# No exceptions\n")
            result = subprocess.run(
                [sys.executable, str(get_checker_path())],
                cwd=tmppath,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 1, bad_path
            assert "clip.mp4" in result.stdout


def test_media_allow_list_still_checks_manifest_sha_and_path():
    """media:<file> does not skip the manifest path and sha256 check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        payload = b"listed"
        import hashlib

        digest = hashlib.sha256(payload).hexdigest()
        (tmppath / "fixtures").mkdir()
        (tmppath / "fixtures" / "test.mp4").write_bytes(payload)
        (tmppath / ".project-rules-allow").write_text("media:fixtures/test.mp4\n")

        (tmppath / "licenses").mkdir()
        (tmppath / "licenses" / "manifest.json").write_text(json.dumps({"entries": []}))
        missing = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert missing.returncode == 1
        assert "fixtures/test.mp4" in missing.stdout

        (tmppath / "licenses" / "manifest.json").write_text(
            _clip_manifest("fixtures/test.mp4", "b" * 64)
        )
        mismatch = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert mismatch.returncode == 1
        assert "fixtures/test.mp4" in mismatch.stdout

        (tmppath / "licenses" / "manifest.json").write_text(
            _clip_manifest("fixtures/test.mp4", digest)
        )
        matched = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert matched.returncode == 0, matched.stdout


def test_media_file_allow_entry_still_checks_manifest():
    """media_file:<path> does not bypass entries[].path or sha256."""
    import hashlib

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        payload = b"listed"
        digest = hashlib.sha256(payload).hexdigest()
        (tmppath / "fixtures").mkdir()
        (tmppath / "licenses").mkdir()
        (tmppath / ".project-rules-allow").write_text("media_file:fixtures/test.mp4\n")

        (tmppath / "fixtures" / "test.mp4").write_bytes(payload)
        (tmppath / "licenses" / "manifest.json").write_text(json.dumps({"entries": []}))
        unlisted = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert unlisted.returncode == 1
        assert "fixtures/test.mp4" in unlisted.stdout

        (tmppath / "fixtures" / "test.mp4").unlink()
        (tmppath / "licenses" / "manifest.json").write_text(
            _clip_manifest("fixtures/test.mp4", digest)
        )
        missing = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert missing.returncode == 1
        assert "missing" in missing.stdout.lower()

        (tmppath / "fixtures" / "test.mp4").write_bytes(payload)
        (tmppath / "licenses" / "manifest.json").write_text(
            _clip_manifest("fixtures/test.mp4", "b" * 64)
        )
        mismatch = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert mismatch.returncode == 1
        assert "sha256" in mismatch.stdout

        (tmppath / "licenses" / "manifest.json").write_text(
            _clip_manifest("fixtures/test.mp4", digest)
        )
        matched = subprocess.run(
            [sys.executable, str(get_checker_path())],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )
        assert matched.returncode == 0, matched.stdout + matched.stderr
