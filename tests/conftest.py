"""Pytest configuration and in-process helpers for license fence tests."""

from __future__ import annotations

import copy
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

SCHEMA_PATH = Path("licenses/manifest_schema.json")

SmokeSet = tuple[Path, list[dict[str, Any]]]


@pytest.fixture(scope="session")
def canonical_smoke_set(tmp_path_factory: pytest.TempPathFactory) -> SmokeSet:
    """Generate the base-seed-42 five-clip smoke set once per session."""
    from trace_search.fixtures.generator import generate_smoke_fixtures

    clips = tmp_path_factory.mktemp("canonical-smoke")
    entries = generate_smoke_fixtures(clips, base_seed=42, num_clips=5)
    return clips, entries


@pytest.fixture
def canonical_smoke(canonical_smoke_set: SmokeSet) -> SmokeSet:
    """A private copy of the shared smoke entries. The clip files stay shared."""
    clips, entries = canonical_smoke_set
    copied: list[dict[str, Any]] = copy.deepcopy(entries)
    return clips, copied


def canonical_clip(smoke: SmokeSet, fixture_id: str) -> Path:
    """Return one shared clip by its fixture id."""
    clips, entries = smoke
    name = next(
        Path(str(entry["path"])).name for entry in entries if entry["fixture_id"] == fixture_id
    )
    return clips / name


@pytest.fixture(autouse=True)
def stub_ffmpeg_for_all_tests(monkeypatch, request):
    """
    Keep the suite off a real libavcodec.

    check_ffmpeg_build still runs. The inspector is stubbed so a manifest
    that names an OpenCV wheel cannot load a library by accident. Tests that
    mock the inspector override this stub. @pytest.mark.ffmpeg_real opts out
    and is only for a test that loads a real libavcodec.
    """
    if "ffmpeg_real" in request.keywords:
        return

    def mock_inspect():
        raise FileNotFoundError("opencv bundled libavcodec not found")

    import scripts.check_licenses as cl

    monkeypatch.setattr(cl, "inspect_opencv_libavcodec", mock_inspect)


def make_manifest(entries=None, **kwargs):
    """Create a test manifest. Accepts entries= or entries via kwargs."""
    if entries is None:
        entries = kwargs.get("entries", [])
    return {
        "manifest_version": "1.0",
        "last_updated_ist": "2026-10-20",
        "entries": entries,
    }


def library_entry(name="good-lib", license_id="MIT", version="1.0.0"):
    return {
        "name": name,
        "version": version,
        "type": "library",
        "license": license_id,
        "source_url": f"https://example.com/{name}",
        "sha256": "a" * 64,
    }


def invoke_main(manifest, extra_args=None, scan_installed=False, tmp_dir=None):
    """
    Call main(argv) in-process and return (exit_code, stdout, stderr).

    Writes the manifest to a temp file. Does not spawn a subprocess.
    """
    import scripts.check_licenses as cl

    if tmp_dir is None:
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
    else:
        tmp_dir = Path(tmp_dir)

    manifest_path = tmp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    argv = ["--manifest", str(manifest_path), "--schema", str(SCHEMA_PATH)]
    if not scan_installed:
        argv.append("--no-scan-installed")
    if extra_args:
        argv.extend(extra_args)
    # Keep tiny manifests off the repo requirements files unless a test opts in.
    if "--requirements" not in argv:
        empty_req = tmp_dir / "empty-requirements.txt"
        empty_req.write_text("# no third-party dependencies\n", encoding="utf-8")
        argv.extend(["--requirements", str(empty_req)])

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cl.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()
