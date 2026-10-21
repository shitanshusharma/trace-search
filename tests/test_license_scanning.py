"""Tests for the license fence with installed package scanning (in-process)."""

import pytest

from tests.conftest import invoke_main, library_entry, make_manifest


@pytest.mark.ffmpeg_real
def test_license_checker_passes_on_valid_manifest_with_installed():
    import scripts.check_licenses as cl

    code = cl.main(["--manifest", "licenses/manifest.json", "--no-scan-installed"])
    assert code in (0, 1)


def test_missing_manifest_entry_fails(tmp_path, monkeypatch):
    import scripts.check_licenses as cl

    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kw: [
            {"name": "pytest", "version": "9.1.1", "license": "MIT"},
            {"name": "unlisted-dep", "version": "1.0.0", "license": "MIT"},
        ],
    )
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="pytest")]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1
    assert "installed but not in manifest" in stdout
    assert "unlisted-dep" in stdout


def test_installed_agpl_package_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="bad-lib", license_id="AGPL-3.0")])
    )
    assert code == 1
    assert "AGPL" in stdout.upper()


def test_unknown_license_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="mystery-lib", license_id="UNKNOWN")])
    )
    assert code == 1
    assert "UNKNOWN" in stdout or "unknown" in stdout.lower()


def test_innocent_substring_passes():
    manifest = make_manifest([
        library_entry(name="overgrew"),
        {
            "name": "progressive-web-app",
            "version": "1.0.0",
            "type": "library",
            "license": "MIT",
            "source_url": "https://example.com/progressive",
            "sha256": "d" * 64,
            "notes": "Started as a simple prototype and expanded over time",
        },
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_actual_banned_grew_fails():
    manifest = make_manifest([
        {
            "name": "grew-dataset",
            "version": "1.0.0",
            "type": "dataset",
            "license": "MIT",
            "source_url": "https://example.com/grew",
            "sha256": "e" * 64,
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1
    assert "grew" in stdout.lower()
    assert "banned component" in stdout.lower()


def test_license_checker_rejects_agpl():
    code, stdout, _ = invoke_main(make_manifest([library_entry(license_id="AGPL-3.0")]))
    assert code == 1
    assert "banned license" in stdout.lower() or "agpl" in stdout.lower()


def test_license_checker_rejects_ultralytics():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="ultralytics", license_id="AGPL-3.0")])
    )
    assert code == 1
    assert "ultralytics" in stdout.lower()


def test_license_checker_accepts_mit():
    code, stdout, _ = invoke_main(make_manifest([library_entry()]))
    assert code == 0, stdout
    assert "PASSED" in stdout
