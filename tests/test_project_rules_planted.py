"""
Planted failure tests for project rules checker.

These tests verify that the project rules checker catches various violations:
- AGPL dependencies
- Banned imports (insightface, ultralytics, etc.)
- Banned identifiers and function calls (watchlist, add_to_watchlist, face_embeddings, etc.)
- Typed name fields and assignments
- Research dataset references (market1501, msmt17, duke, cuhk03, casia-b, grew, gait3d, objects365)
- Media files outside allowed paths
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

        # Create manifest with this file
        manifest = {
            "version": "1.0",
            "files": [
                {
                    "path": "fixtures/test.mp4",
                    "sha256": sha256_hash,
                    "use": "CODE_TEST",
                    "licence": "Pexels License",
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
        manifest = {"version": "1.0", "files": []}
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
            "files": [
                {
                    "path": "fixtures/test.mp4",
                    "sha256": digest,
                    "use": "SMOKE_ONLY",
                }
            ]
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
