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
