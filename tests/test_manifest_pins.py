"""Manifest rows for the T1 runtime pins."""

import json
import re
from pathlib import Path

import pytest


def _pins(text: str) -> dict[str, str]:
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--hash"):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)==([^\\\s]+)", line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def _library_by_name(manifest: dict) -> dict[str, dict]:
    return {
        entry["name"].lower().replace("_", "-"): entry
        for entry in manifest["entries"]
        if entry.get("type") == "library"
    }


def _numpy_library_rows(manifest: dict) -> list[dict]:
    return [
        entry
        for entry in manifest["entries"]
        if entry.get("type") == "library" and entry["name"].lower().replace("_", "-") == "numpy"
    ]


def _assert_numpy_row_matches_requirements(requirements: str, manifest: dict) -> None:
    pins = _pins(requirements)
    rows = _numpy_library_rows(manifest)
    assert len(rows) == 1
    assert rows[0]["version"] == pins["numpy"]
    assert not Path("requirements-numpy.txt").exists()


def test_runtime_rows_match_pins_and_numpy_matches_pin():
    manifest = json.loads(Path("licenses/manifest.json").read_text(encoding="utf-8"))
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    pins = _pins(requirements)
    _assert_numpy_row_matches_requirements(requirements, manifest)
    assert pins["numpy"] == "1.26.4"
    by_name = _library_by_name(manifest)
    expected = {
        "pillow": "MIT-CMU",
        "imagehash": "BSD-2-Clause",
        "pywavelets": "MIT AND BSD-3-Clause",
        "scipy": "BSD-3-Clause",
        "opencv-python-headless": "Apache-2.0",
    }
    for name, license_name in expected.items():
        assert by_name[name]["version"] == pins[name]
        assert by_name[name]["license"] == license_name
        assert len(by_name[name]["sha256"]) == 64
    assert by_name["pywavelets"]["version"] == "1.9.0"
    assert by_name["scipy"]["version"] == "1.17.1"
    assert "cp312" not in by_name["pillow"]["notes"]
    assert "cp311" in by_name["pillow"]["notes"]
    assert "cp311" in by_name["scipy"]["notes"]
    assert "cp311" in by_name["pywavelets"]["notes"]


def test_planted_numpy_2_2_0_in_requirements_fails():
    """A requirements.txt pin of numpy==2.2.0 does not match the 1.26.4 row."""
    manifest = json.loads(Path("licenses/manifest.json").read_text(encoding="utf-8"))
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    planted = requirements.replace("numpy==1.26.4", "numpy==2.2.0", 1)
    assert planted != requirements
    with pytest.raises(AssertionError, match=r"1\.26\.4") as caught:
        _assert_numpy_row_matches_requirements(planted, manifest)
    assert "2.2.0" in str(caught.value)
    assert Path("requirements.txt").read_text(encoding="utf-8") == requirements
    assert not Path("requirements-numpy.txt").exists()
