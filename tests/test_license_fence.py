"""Tests for the license fence (in-process)."""

from tests.conftest import invoke_main, library_entry, make_manifest


def test_license_checker_passes_on_valid_manifest():
    """The license checker should pass on the current valid manifest."""
    import scripts.check_licenses as cl

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
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 0, stdout
    assert "PASSED" in stdout
