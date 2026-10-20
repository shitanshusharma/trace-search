"""Manifest rows for the T1 runtime pins."""

import json
import re
from pathlib import Path


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


def test_runtime_rows_match_pins_and_numpy_is_absent():
    manifest = json.loads(Path("licenses/manifest.json").read_text(encoding="utf-8"))
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    pins = _pins(requirements)
    numpy_pins = _pins(Path("requirements-numpy.txt").read_text(encoding="utf-8"))
    by_name = {
        entry["name"].lower().replace("_", "-"): entry
        for entry in manifest["entries"]
        if entry.get("type") == "library"
    }
    assert "numpy" not in pins
    assert numpy_pins["numpy"] == "2.2.0"
    assert "-r requirements-numpy.txt" in requirements
    assert "numpy" not in by_name
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
