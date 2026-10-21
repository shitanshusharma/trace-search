"""Tests for the license fence (in-process)."""

import hashlib

from tests.conftest import invoke_main, library_entry, make_manifest


def test_license_checker_passes_on_valid_manifest(monkeypatch):
    """The license checker should pass on the current valid manifest.

    The inspector is patched in this test only. The suite default stays
    fail-closed, so a root conftest is not required.
    """
    import scripts.check_licenses as cl

    monkeypatch.setattr(
        cl,
        "inspect_opencv_libavcodec",
        lambda: ("LGPL version 2.1 or later", "--enable-shared"),
    )
    code = cl.main(["--manifest", "licenses/manifest.json", "--no-scan-installed"])
    assert code == 0


def test_license_checker_rejects_agpl():
    code, stdout, _ = invoke_main(make_manifest([library_entry(license_id="AGPL-3.0")]))
    assert code == 1
    assert "banned license" in stdout.lower()
    assert "agpl" in stdout.lower()


def test_license_checker_rejects_gpl():
    code, stdout, _ = invoke_main(make_manifest([library_entry(license_id="GPL-3.0")]))
    assert code == 1
    assert "banned license" in stdout.lower()


def test_license_checker_rejects_noncommercial():
    code, stdout, _ = invoke_main(make_manifest([library_entry(license_id="CC-BY-NC-4.0")]))
    assert code == 1
    assert "banned license" in stdout.lower()


def test_license_checker_rejects_ultralytics():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="ultralytics", license_id="AGPL-3.0")])
    )
    assert code == 1
    assert "ultralytics" in stdout.lower()


def test_license_checker_rejects_boxmot():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="boxmot", license_id="AGPL-3.0")])
    )
    assert code == 1
    assert "boxmot" in stdout.lower()


def test_license_checker_rejects_insightface():
    code, stdout, _ = invoke_main(make_manifest([library_entry(name="insightface")]))
    assert code == 1
    assert "insightface" in stdout.lower()


def test_license_checker_rejects_market1501():
    manifest = make_manifest([
        {
            "name": "Market-1501",
            "version": "1.0",
            "type": "dataset",
            "license": "Unknown",
            "source_url": "https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng.pdf",
            "sha256": "a" * 64,
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1
    assert "market" in stdout.lower() or "unknown" in stdout.lower()


def test_license_checker_accepts_mit():
    code, stdout, _ = invoke_main(make_manifest([library_entry()]))
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_license_checker_accepts_apache2():
    code, stdout, _ = invoke_main(make_manifest([library_entry(license_id="Apache-2.0")]))
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_license_checker_validates_clip_fields():
    manifest = make_manifest([
        {
            "name": "test-clip.mp4",
            "version": "1.0",
            "type": "clip",
            "license": "Pexels License",
            "source_url": "https://www.pexels.com/video/test/",
            "sha256": "c" * 64,
            "path": "tests/fixtures/media/test-clip.mp4",
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1
    assert "license_version" in stdout
    assert "date_checked_ist" in stdout
    assert "phash" in stdout
    assert "use" in stdout


def test_license_checker_accepts_valid_clip():
    manifest = make_manifest([
        {
            "name": "test-clip.mp4",
            "version": "1.0",
            "type": "clip",
            "license": "Pexels License",
            "license_version": "Pexels ToS, last updated 15 Nov 2024",
            "date_checked_ist": "2026-10-20",
            "source_url": "https://www.pexels.com/video/test/",
            "sha256": "c" * 64,
            "phash": [f"{i:016x}" for i in range(10)],
            "use": "SMOKE_ONLY",
            "fixture_id": "FX-SMOKE-001",
            "path": "tests/fixtures/media/test-clip.mp4",
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 0, stdout
    assert "PASSED" in stdout


# Non-SPDX identifiers already on the fence before the stock-clip approval.
_NON_SPDX_WITHOUT_STOCK = {
    "MIT License",
    "Apache License 2.0",
    "Apache Software License",
    "BSD License",
    "ISC License",
    "ISC License (ISCL)",
    "Python Software Foundation License",
    "Public Domain",
    "Historical Permission Notice and Disclaimer (HPND)",
    "MIT CMU License (MIT-CMU)",
    "Zlib",
    "Libpng",
    "libtiff",
}

_STOCK_NON_SPDX = {
    "Pexels License",
    "Pixabay Content License",
}


def _stock_clip_row(license_id: str, name: str) -> dict:
    """Planted clip row. The path stays inside the temp manifest, not the repo."""
    payload = f"{name}\n{license_id}\n".encode("ascii")
    return {
        "name": name,
        "version": "1.0",
        "type": "clip",
        "license": license_id,
        "license_version": "stock terms checked 2026-10-21",
        "date_checked_ist": "2026-10-21",
        "source_url": "https://example.com/stock-clip",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "phash": [f"{i:016x}" for i in range(10)],
        "use": "SMOKE_ONLY",
        "fixture_id": "FX-SMOKE-013",
        "path": "clips/planted-stock.mp4",
    }


def test_planted_pexels_license_row_passes_the_fence():
    """A temp manifest row licensed 'Pexels License' passes the fence."""
    code, stdout, _ = invoke_main(
        make_manifest([_stock_clip_row("Pexels License", "pexels-planted.mp4")])
    )
    assert code == 0, stdout
    assert "License fence check PASSED" in stdout


def test_planted_pixabay_content_license_row_passes_the_fence():
    """A temp manifest row licensed 'Pixabay Content License' passes the fence."""
    code, stdout, _ = invoke_main(
        make_manifest([_stock_clip_row("Pixabay Content License", "pixabay-planted.mp4")])
    )
    assert code == 0, stdout
    assert "License fence check PASSED" in stdout


def test_planted_pexels_british_spelling_fails_the_fence():
    """British spelling 'Pexels Licence' is not the approved string and fails."""
    code, stdout, _ = invoke_main(
        make_manifest([_stock_clip_row("Pexels Licence", "pexels-near-miss.mp4")])
    )
    assert code == 1, stdout
    assert "unparseable license 'Pexels Licence'" in stdout
    assert "License fence check FAILED" in stdout
    assert "License fence check PASSED" not in stdout


def test_allowed_non_spdx_grew_by_exactly_the_two_stock_licences():
    """ALLOWED_NON_SPDX gained only the two approved stock-clip strings."""
    import scripts.check_licenses as cl

    assert cl.ALLOWED_NON_SPDX == _NON_SPDX_WITHOUT_STOCK | _STOCK_NON_SPDX
    for licence in _STOCK_NON_SPDX:
        licence.encode("ascii")
        assert licence == licence.strip()
        assert "Licence" not in licence
    assert "Pexels Licence" not in cl.ALLOWED_NON_SPDX
