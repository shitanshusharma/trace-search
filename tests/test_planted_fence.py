"""In-process planted tests for T0b Engineering Lead blockers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.check_licenses as cl
from tests.conftest import invoke_main, library_entry, make_manifest

SCHEMA = Path("licenses/manifest_schema.json")


def _clip_fields(**overrides):
    base = {
        "name": "stock.mp4",
        "version": "1.0",
        "type": "clip",
        "license": "Pexels License",
        "license_version": "2024-01-01",
        "date_checked_ist": "2026-10-20",
        "source_url": "https://example.com/clip",
        "sha256": "a" * 64,
        "phash": [f"{i:016x}" for i in range(10)],
        "use": "SMOKE_ONLY",
        "fixture_id": "FX-SMOKE-001",
        "path": "tests/fixtures/media/stock.mp4",
    }
    base.update(overrides)
    return base


def _synthetic_fields(**overrides):
    base = {
        "name": "synthetic.mp4",
        "version": "1.0",
        "type": "synthetic_clip",
        "license": "CC0-1.0",
        "license_version": "1.0",
        "date_checked_ist": "2026-10-20",
        "source_url": "https://example.com/synthetic",
        "sha256": "a" * 64,
        "phash": [f"{i:016x}" for i in range(10)],
        "use": "SMOKE_ONLY",
        "fixture_id": "FX-SMOKE-SYN-001",
        "path": "tests/fixtures/media/synthetic.mp4",
        "generator": "trace-procedural",
        "generator_version": "1.0.0",
        "generator_license": "Apache-2.0",
        "seed": "12345",
    }
    base.update(overrides)
    return base


def _code_fixture(**overrides):
    base = {
        "name": "code-sample.py",
        "version": "1.0",
        "type": "code_fixture",
        "license": "MIT",
        "source_url": "https://example.com/code",
        "sha256": "a" * 64,
        "use": "CODE_TEST",
        "fixture_id": "FX-CODE-001",
        "generator": "trace-procedural",
        "generator_version": "1.0.0",
        "generator_license": "Apache-2.0",
        "seed": "1",
    }
    base.update(overrides)
    return base


# ----- 1. generate_report end-to-end -----


def test_report_end_to_end_includes_ffmpeg_commit_verdict(tmp_path):
    """main(['--report', tmp]) writes JSON with ffmpeg_status, git_commit, verdict."""
    report_path = tmp_path / "license-report.json"
    code, stdout, _ = invoke_main(
        make_manifest([library_entry()]),
        extra_args=["--report", str(report_path)],
        tmp_dir=tmp_path,
    )
    assert code == 0, stdout
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["ffmpeg_status"] == "NOT_A_DEPENDENCY"
    assert report["git_commit"]
    assert report["verdict"] == "PASS"
    assert "FFMPEG: NOT_A_DEPENDENCY" in stdout


# ----- 6. GCC runtime exception -----


def test_gcc_runtime_exception_libgfortran_passes():
    manifest = make_manifest([
        library_entry(
            name="libgfortran",
            license_id="GPL-3.0 WITH GCC-exception-3.1",
        )
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_planted_scipy_libgfortran_as_gpl_3_fails():
    """Changing a scipy libgfortran row from the GCC exception to GPL-3.0 fails."""
    manifest = json.loads(Path("licenses/manifest.json").read_text(encoding="utf-8"))
    changed = False
    for entry in manifest["entries"]:
        bundled = entry.get("bundled_in", "")
        if entry.get("name") == "libgfortran" and bundled.startswith("scipy-1.17.1"):
            entry["license"] = "GPL-3.0"
            changed = True
            break
    assert changed
    code, stdout, _stderr = invoke_main(manifest)
    assert code == 1, stdout
    assert "banned license 'GPL-3.0'" in stdout


def test_planted_pillow_libjpeg_turbo_as_gpl_3_fails():
    """Changing the Pillow libjpeg-turbo row to GPL-3.0 fails."""
    manifest = json.loads(Path("licenses/manifest.json").read_text(encoding="utf-8"))
    changed = False
    for entry in manifest["entries"]:
        bundled = entry.get("bundled_in", "")
        if entry.get("name") == "libjpeg-turbo" and bundled.startswith("pillow-13.0.0"):
            entry["license"] = "GPL-3.0"
            changed = True
            break
    assert changed
    code, stdout, _stderr = invoke_main(manifest)
    assert code == 1, stdout
    assert "banned license 'GPL-3.0'" in stdout


def test_plain_gpl3_still_fails():
    code, stdout, _ = invoke_main(make_manifest([library_entry(license_id="GPL-3.0")]))
    assert code == 1, stdout
    assert "banned license 'GPL-3.0'" in stdout
    assert "banned pattern '^GPL-[123]' in 'GPL-3.0'" in stdout


def test_plain_gpl2_still_fails():
    code, stdout, _ = invoke_main(make_manifest([library_entry(license_id="GPL-2.0")]))
    assert code == 1, stdout
    assert "banned license 'GPL-2.0'" in stdout
    assert "banned pattern '^GPL-[123]' in 'GPL-2.0'" in stdout


def test_gpl3_with_classpath_exception_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(license_id="GPL-3.0 WITH Classpath-exception-2.0")])
    )
    assert code == 1, stdout
    assert "GPL-3.0 WITH Classpath-exception-2.0" in stdout
    assert "banned pattern '^GPL-[123]' in 'GPL-3.0'" in stdout


def test_non_gcc_package_claiming_gcc_exception_fails():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(name="fake-lib", license_id="GPL-3.0 WITH GCC-exception-3.1")
        ])
    )
    assert code == 1, stdout
    assert "fake-lib: banned license 'GPL-3.0 WITH GCC-exception-3.1'" in stdout


def test_mit_and_gpl_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(license_id="MIT AND GPL-3.0-only")])
    )
    assert code == 1, stdout
    assert "MIT AND GPL-3.0-only" in stdout
    assert "banned" in stdout.lower()


def test_mit_or_gpl_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(license_id="MIT OR GPL-3.0-only")])
    )
    assert code == 1, stdout
    assert "banned license 'MIT OR GPL-3.0-only'" in stdout


def test_agpl_or_mit_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(license_id="AGPL-3.0-only OR MIT")])
    )
    assert code == 1, stdout
    assert "banned license 'AGPL-3.0-only OR MIT'" in stdout


def test_cc_by_nc_or_mit_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(license_id="CC-BY-NC-4.0 OR MIT")])
    )
    assert code == 1, stdout
    assert "banned license 'CC-BY-NC-4.0 OR MIT'" in stdout


def test_mit_or_apache_passes():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(license_id="MIT OR Apache-2.0")])
    )
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_or_of_only_banned_licenses_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(license_id="GPL-3.0-only OR AGPL-3.0-only")])
    )
    assert code == 1, stdout
    assert "GPL-3.0-only OR AGPL-3.0-only" in stdout


def test_malformed_spdx_expression_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(license_id="(MIT OR GPL-3.0-only")])
    )
    assert code == 1, stdout
    assert "unparseable license '(MIT OR GPL-3.0-only'" in stdout
    assert "banned license '(MIT OR GPL-3.0-only'" not in stdout


def test_spdx_parenthesized_or_with_gpl_fails():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(license_id="(MIT OR GPL-3.0-only) AND Apache-2.0")
        ])
    )
    assert code == 1, stdout
    assert "banned license '(MIT OR GPL-3.0-only) AND Apache-2.0'" in stdout


def test_spdx_parentheses_scope_the_and():
    """Parentheses bind the OR inside AND, so the GPL term is still required."""
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(license_id="GPL-3.0-only AND (MIT OR Apache-2.0)")
        ])
    )
    assert code == 1, stdout
    assert "GPL-3.0-only AND (MIT OR Apache-2.0)" in stdout


def test_with_exception_other_than_gcc_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(license_id="Apache-2.0 WITH LLVM-exception")])
    )
    assert code == 1, stdout
    assert "Apache-2.0 WITH LLVM-exception" in stdout
    assert "PASSED" not in stdout


def test_ijg_bsd_zlib_conjunction_passes():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(
                name="libjpeg-turbo",
                license_id="IJG AND BSD-3-Clause AND Zlib",
            )
        ])
    )
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_openssl_and_intel_licenseref_stay_unlisted():
    for license_id in ("OpenSSL", "LicenseRef-Intel-Simplified-Software-License"):
        code, stdout, _ = invoke_main(
            make_manifest([library_entry(name="pending-lib", license_id=license_id)])
        )
        assert code == 1, stdout
        assert license_id in stdout
        assert "not in allowed list" in stdout


# ----- 4. generator_license allowlist -----


def test_generator_license_gpl_fails():
    code, stdout, _ = invoke_main(
        make_manifest([_synthetic_fields(generator_license="GPL-3.0")])
    )
    assert code == 1, stdout
    assert "generator_license 'GPL-3.0' is banned" in stdout


def test_generator_license_nc_fails():
    code, stdout, _ = invoke_main(
        make_manifest([_synthetic_fields(generator_license="CC-BY-NC-4.0")])
    )
    assert code == 1, stdout
    assert "generator_license 'CC-BY-NC-4.0' is banned" in stdout


def test_generator_license_proprietary_fails():
    code, stdout, _ = invoke_main(
        make_manifest([_synthetic_fields(generator_license="Proprietary")])
    )
    assert code == 1, stdout
    assert "generator_license 'Proprietary' is banned" in stdout


def test_generator_license_free_text_fails():
    code, stdout, _ = invoke_main(
        make_manifest([_synthetic_fields(generator_license="custom blender terms")])
    )
    assert code == 1, stdout
    assert "generator_license 'custom blender terms' is unparseable" in stdout
    assert "is banned" not in stdout


def test_generator_license_empty_fails():
    code, stdout, _ = invoke_main(
        make_manifest([_synthetic_fields(generator_license="")])
    )
    assert code == 1, stdout
    assert "synthetic_clip missing required generator_license field" in stdout


def test_generator_license_apache_passes():
    code, stdout, _ = invoke_main(make_manifest([_synthetic_fields()]))
    assert code == 0, stdout


def test_code_fixture_gpl_generator_license_fails():
    code, stdout, _ = invoke_main(
        make_manifest([_code_fixture(generator_license="GPL-3.0")])
    )
    assert code == 1, stdout
    assert "generator_license 'GPL-3.0' is banned" in stdout


def test_generator_row_nc_license_fails():
    code, stdout, _ = invoke_main(
        make_manifest([_generator_row(generator_license="CC-BY-NC-4.0")])
    )
    assert code == 1, stdout
    assert "generator_license 'CC-BY-NC-4.0' is banned" in stdout


def test_generator_row_free_text_license_fails():
    code, stdout, _ = invoke_main(
        make_manifest([_generator_row(generator_license="see the internal wiki")])
    )
    assert code == 1, stdout
    assert "generator_license 'see the internal wiki' is unparseable" in stdout
    assert "is banned" not in stdout


def test_banned_generator_name_fails():
    code, stdout, _ = invoke_main(
        make_manifest([_code_fixture(generator="ultralytics")])
    )
    assert code == 1, stdout
    assert "banned component" in stdout
    assert "ultralytics" in stdout


def test_banned_generator_name_on_synthetic_clip_fails():
    code, stdout, _ = invoke_main(
        make_manifest([_synthetic_fields(generator="ultralytics")])
    )
    assert code == 1, stdout
    assert "banned component" in stdout
    assert "ultralytics" in stdout


def test_banned_generator_version_fails():
    code, stdout, _ = invoke_main(
        make_manifest([_code_fixture(generator_version="ultralytics-8.1")])
    )
    assert code == 1, stdout
    assert "banned component" in stdout
    assert "ultralytics" in stdout


@pytest.mark.parametrize("field", ["generator", "generator_version", "seed"])
def test_empty_generator_field_fails(field):
    row = _synthetic_fields(**{field: ""})
    errors = _schema_errors(make_manifest([row]))
    assert any(
        f"at entries.0.{field}" in err and "non-empty" in err
        for err in errors
    ), errors


@pytest.mark.parametrize("field", ["generator", "generator_version", "seed"])
def test_whitespace_only_generator_field_fails(field):
    row = _synthetic_fields(**{field: "   "})
    errors = _schema_errors(make_manifest([row]))
    assert any(
        f"at entries.0.{field}" in err and "does not match" in err
        for err in errors
    ), errors


@pytest.mark.parametrize(
    "bad_path",
    ["/clips/stock.mp4", "../stock.mp4", "clips/../stock.mp4", "clips\\stock.mp4"],
)
def test_media_path_must_be_repo_relative(bad_path):
    errors = _schema_errors(make_manifest([_clip_fields(path=bad_path)]))
    assert any(
        "at entries.0.path" in err and "does not match" in err
        for err in errors
    ), errors


def test_empty_license_version_fails():
    errors = _schema_errors(
        make_manifest([_synthetic_fields(license_version="")])
    )
    assert any(
        "at entries.0.license_version" in err and "non-empty" in err
        for err in errors
    ), errors


def test_whitespace_only_license_version_fails():
    errors = _schema_errors(
        make_manifest([_synthetic_fields(license_version="   ")])
    )
    assert any(
        "at entries.0.license_version" in err and "does not match" in err
        for err in errors
    ), errors


# ----- 5. schema fixture_id / type / use -----


def _schema_errors(manifest):
    return cl.validate_schema(manifest, SCHEMA)


def test_schema_fx_smoke_syn_with_code_test_fails_at_use():
    errors = _schema_errors(make_manifest([_synthetic_fields(use="CODE_TEST")]))
    assert any("SMOKE_ONLY" in e and "at entries.0.use" in e for e in errors), errors


def test_schema_fx_code_with_smoke_only_fails_at_use():
    errors = _schema_errors(make_manifest([_code_fixture(use="SMOKE_ONLY")]))
    assert any("CODE_TEST" in e and "at entries.0.use" in e for e in errors), errors


def test_schema_clip_train_fails_at_use():
    errors = _schema_errors(make_manifest([_clip_fields(use="TRAIN")]))
    assert any("at entries.0.use" in e for e in errors), errors


def test_schema_fx_code_as_clip_fails_at_type():
    errors = _schema_errors(
        make_manifest([_clip_fields(fixture_id="FX-CODE-001", use="CODE_TEST")])
    )
    assert any("at entries.0.type" in e or "at entries.0.use" in e for e in errors), errors


def test_schema_fx_smoke_syn_as_clip_fails_at_type():
    errors = _schema_errors(
        make_manifest([_clip_fields(fixture_id="FX-SMOKE-SYN-001")])
    )
    assert any("at entries.0.type" in e or "at entries.0.fixture_id" in e for e in errors), errors


def test_schema_valid_code_fixture_passes():
    code, stdout, _ = invoke_main(make_manifest([_code_fixture()]))
    assert code == 0, stdout


def test_schema_valid_stock_clip_passes():
    code, stdout, _ = invoke_main(make_manifest([_clip_fields()]))
    assert code == 0, stdout


# ----- 3. FFmpeg ctypes / fail-closed -----


def test_ffmpeg_not_a_dependency_when_no_opencv():
    violations, status = cl.check_ffmpeg_build(make_manifest([library_entry()]))
    assert violations == []
    assert status == "NOT_A_DEPENDENCY"


def test_opencv_libavcodec_lgpl_passes(monkeypatch):
    monkeypatch.setattr(
        cl,
        "inspect_opencv_libavcodec",
        lambda: ("LGPL version 2.1 or later", "--enable-shared"),
    )
    violations, status = cl.check_ffmpeg_build(
        make_manifest([library_entry(name="opencv-python-headless")])
    )
    assert violations == []
    assert status == "LGPL_OK"


def test_opencv_libavcodec_gpl_fails(monkeypatch):
    monkeypatch.setattr(
        cl, "inspect_opencv_libavcodec",
        lambda: ("GPL version 2 or later", "--enable-gpl --enable-shared"),
    )
    violations, status = cl.check_ffmpeg_build(
        make_manifest([library_entry(name="opencv-python-headless")])
    )
    assert status == "GPL_VIOLATION"
    assert any("--enable-gpl" in v for v in violations)


def test_opencv_libavcodec_nonfree_fails(monkeypatch):
    monkeypatch.setattr(
        cl, "inspect_opencv_libavcodec",
        lambda: ("LGPL version 2.1 or later", "--enable-nonfree"),
    )
    violations, status = cl.check_ffmpeg_build(
        make_manifest([library_entry(name="opencv-python-headless")])
    )
    assert status == "GPL_VIOLATION"
    assert any("--enable-nonfree" in v for v in violations)


def test_opencv_libavcodec_missing_fails_closed(monkeypatch):
    def boom():
        raise FileNotFoundError("opencv bundled libavcodec not found")

    monkeypatch.setattr(cl, "inspect_opencv_libavcodec", boom)
    violations, status = cl.check_ffmpeg_build(
        make_manifest([library_entry(name="opencv-python-headless")])
    )
    assert status == "MISSING"
    assert any("libavcodec" in v for v in violations)


def test_opencv_contrib_wheel_lgpl_passes(monkeypatch):
    """A contrib-named wheel with an LGPL libavcodec passes, any case."""
    monkeypatch.setattr(
        cl,
        "inspect_opencv_libavcodec",
        lambda: ("LGPL version 2.1 or later", "--enable-shared"),
    )
    violations, status = cl.check_ffmpeg_build(
        make_manifest([library_entry(name="OpenCV_Contrib_Python")])
    )
    assert violations == []
    assert status == "LGPL_OK"


def test_opencv_contrib_wheel_without_clean_libavcodec_fails(tmp_path):
    """A contrib wheel is an FFmpeg distribution, so a missing libavcodec fails."""
    import trace_search.ingest.ffmpeg_check as ffmpeg_check

    assert cl.is_opencv_ffmpeg_distribution is ffmpeg_check.is_opencv_ffmpeg_distribution
    assert "opencv-contrib-python" in ffmpeg_check.OPENCV_FFMPEG_DISTRIBUTIONS
    assert "opencv-contrib-python-headless" in ffmpeg_check.OPENCV_FFMPEG_DISTRIBUTIONS
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="opencv-contrib-python")]),
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "MISSING" in stdout
    assert "libavcodec" in stdout


def test_opencv_contrib_gpl_configuration_fails_fence(tmp_path, monkeypatch):
    """A contrib wheel whose libavcodec reports --enable-gpl fails the fence."""
    monkeypatch.setattr(
        cl,
        "inspect_opencv_libavcodec",
        lambda: ("LGPL version 2.1 or later", "--enable-gpl --enable-shared"),
    )
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="opencv-contrib-python-headless")]),
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "GPL_VIOLATION" in stdout
    assert "--enable-gpl" in stdout


@pytest.mark.parametrize(
    "pkg,lic",
    [
        ("Pillow", "HPND"),
        ("Pillow", "MIT-CMU"),
        ("zlib", "Zlib"),
        ("libpng", "Libpng"),
        ("libtiff", "libtiff"),
    ],
)
def test_allowlisted_opencv_pillow_licenses_pass(pkg, lic):
    code, stdout, _ = invoke_main(make_manifest([library_entry(name=pkg, license_id=lic)]))
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_gpl_still_fails_when_allowlist_includes_hpnd():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(name="Pillow", license_id="HPND"),
            library_entry(name="gpl-package", license_id="GPL-3.0"),
        ])
    )
    assert code == 1, stdout
    assert "gpl-package: banned license 'GPL-3.0'" in stdout


# ----- 8. installed closure planted tests -----


def test_installed_gpl_package_fails_with_specific_message(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kw: [{"name": "gpl-pkg", "version": "1.0.0", "license": "GPL-3.0"}],
    )
    code, stdout, _ = invoke_main(
        make_manifest([library_entry()]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "gpl-pkg (1.0.0): banned license 'GPL-3.0'" in stdout


def test_installed_version_mismatch_fails(tmp_path, monkeypatch):
    """A stale manifest version does not cover a newer installed package."""
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kw: [{"name": "numpy", "version": "2.2.0", "license": "BSD-3-Clause"}],
    )
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(name="numpy", version="1.26.4", license_id="BSD-3-Clause")
        ]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "numpy: installed version 2.2.0 does not match manifest version 1.26.4" in stdout
    assert "PASSED" not in stdout


def test_installed_unlisted_requirements_package_fails(tmp_path, monkeypatch):
    """An installed runtime package with no manifest row fails closed."""
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kw: [{"name": "mystery-runtime", "version": "2.0.0", "license": "MIT"}],
    )
    code, stdout, _ = invoke_main(
        make_manifest([library_entry()]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "mystery-runtime (2.0.0): installed but not in manifest" in stdout


def test_metadata_gpl_manifest_mit_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kw: [{"name": "bad-package", "version": "1.0.0", "license": "GPL-3.0"}],
    )
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="bad-package", license_id="MIT")]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "bad-package (1.0.0): banned license 'GPL-3.0'" in stdout


def test_ultralytics_installed_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kw: [{"name": "ultralytics", "version": "8.3.0", "license": "AGPL-3.0"}],
    )
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="pytest")]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "ultralytics (8.3.0): banned license 'AGPL-3.0'" in stdout


def test_pip_licenses_raising_fails(tmp_path, monkeypatch):
    def boom(**kw):
        raise RuntimeError("pip-licenses command failed")

    monkeypatch.setattr(cl, "get_installed_packages", boom)
    code, stdout, _ = invoke_main(
        make_manifest([library_entry()]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "Failed to scan installed packages: pip-licenses command failed" in stdout


def test_pip_licenses_empty_fails(tmp_path, monkeypatch):
    def boom(**kw):
        raise RuntimeError("pip-licenses returned empty package list")

    monkeypatch.setattr(cl, "get_installed_packages", boom)
    code, stdout, _ = invoke_main(
        make_manifest([library_entry()]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "Failed to scan installed packages: pip-licenses returned empty package list" in stdout


def test_imagehash_two_clause_bsd_license_passes(tmp_path, monkeypatch):
    """pip-licenses 5.5.5 reports ImageHash as this exact free-text string."""
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kw: [{
            "name": "ImageHash",
            "version": "4.3.2",
            "license": "2-clause BSD License",
        }],
    )
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(name="ImageHash", version="4.3.2", license_id="BSD-2-Clause")
        ]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_two_clause_bsd_text_rejected_for_other_packages(tmp_path, monkeypatch):
    """The ImageHash rewrite does not apply to any other package name."""
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kw: [{
            "name": "Pygments",
            "version": "2.21.0",
            "license": "2-clause BSD License",
        }],
    )
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(name="Pygments", version="2.21.0", license_id="BSD-2-Clause")
        ]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "2-clause BSD License" in stdout
    assert "PASSED" not in stdout


def test_unknown_pip_licenses_free_text_stays_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kw: [{
            "name": "ImageHash",
            "version": "4.3.2",
            "license": "2-clause BSD license",
        }],
    )
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(name="ImageHash", version="4.3.2", license_id="BSD-2-Clause")
        ]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "2-clause BSD license" in stdout
    assert "PASSED" not in stdout


def test_semicolon_joined_gpl_on_installed_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kw: [{
            "name": "python-dateutil",
            "version": "2.9.0",
            "license": "Apache-2.0; GPL-2.0",
        }],
    )
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="python-dateutil", license_id="Apache-2.0")]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "banned license 'Apache-2.0; GPL-2.0'" in stdout
    assert "GPL-2.0" in stdout


def _generator_row(**overrides):
    base = {
        "name": "trace-procedural",
        "version": "1.0.0",
        "type": "generator",
        "license": "Apache-2.0",
        "source_url": "https://example.com/trace-procedural",
        "sha256": "b" * 64,
        "generator_license": "Apache-2.0",
        "seed": "1",
    }
    base.update(overrides)
    return base


def _assert_missing_required(entry, field):
    row = dict(entry)
    row.pop(field)
    errors = _schema_errors(make_manifest([row]))
    needle = f"'{field}' is a required property"
    assert any(needle in err for err in errors), errors


CLIP_REQUIRED = [
    "license_version",
    "date_checked_ist",
    "phash",
    "use",
    "fixture_id",
    "path",
]
SYNTHETIC_REQUIRED = [
    "license_version",
    "date_checked_ist",
    "phash",
    "use",
    "fixture_id",
    "path",
    "generator",
    "generator_version",
    "generator_license",
    "seed",
]
CODE_FIXTURE_REQUIRED = [
    "use",
    "fixture_id",
    "generator",
    "generator_version",
    "generator_license",
    "seed",
]
GENERATOR_REQUIRED = ["generator_license", "seed"]


@pytest.mark.parametrize("field", CLIP_REQUIRED)
def test_clip_missing_required_field(field):
    _assert_missing_required(_clip_fields(), field)


@pytest.mark.parametrize("field", SYNTHETIC_REQUIRED)
def test_synthetic_clip_missing_required_field(field):
    _assert_missing_required(_synthetic_fields(), field)


@pytest.mark.parametrize("field", CODE_FIXTURE_REQUIRED)
def test_code_fixture_missing_required_field(field):
    _assert_missing_required(_code_fixture(), field)


@pytest.mark.parametrize("field", GENERATOR_REQUIRED)
def test_generator_missing_required_field(field):
    _assert_missing_required(_generator_row(), field)


def test_pip_licenses_not_on_path_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cl.shutil, "which", lambda _name: None)
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="pytest")]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "Failed to scan installed packages: pip-licenses is not on PATH" in stdout
    assert "PASSED" not in stdout


def test_empty_installed_scan_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "get_installed_packages", lambda **kw: [])
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="pytest")]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "Package scan failed: installed package list is empty" in stdout
    assert "PASSED" not in stdout


def test_requirements_txt_missing_numpy_fails(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.26.4\n", encoding="utf-8")
    dev = tmp_path / "requirements-dev.txt"
    dev.write_text("pytest==9.1.1\n", encoding="utf-8")
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="pytest")]),
        extra_args=[
            "--requirements",
            str(req),
            "--requirements",
            str(dev),
        ],
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "numpy: required by requirements.txt but not in manifest" in stdout


# ----- OpenCV bundled OpenSSL 1.1.1w and Intel IPP (21 Oct 2026) -----

_OPENSSL_LIBCRYPTO = "opencv_python_headless.libs/libcrypto-337dac8b.so.1.1"
_OPENSSL_LIBSSL = "opencv_python_headless.libs/libssl-28bef1ac.so.1.1"
_IPP_FILE = "cv2/cv2.abi3.so"
_OPENCV_PACKAGE = "opencv-python-headless"
_OPENCV_VERSION = "4.10.0.84"


def _scoped_row(name, license_id, version, bundled_file, **overrides):
    row = library_entry(name=name, license_id=license_id, version=version)
    row.update(
        {
            "bundled_package": _OPENCV_PACKAGE,
            "bundled_package_version": _OPENCV_VERSION,
            "bundled_file": bundled_file,
        }
    )
    row.update(overrides)
    return row


def _openssl_row(**overrides):
    bundled_file = overrides.pop("bundled_file", _OPENSSL_LIBCRYPTO)
    version = overrides.pop("version", "1.1.1w")
    return _scoped_row(
        "OpenSSL-libcrypto",
        "OpenSSL",
        version,
        bundled_file,
        **overrides,
    )


def _ipp_row(**overrides):
    bundled_file = overrides.pop("bundled_file", _IPP_FILE)
    version = overrides.pop("version", "2021.11.0")
    return _scoped_row(
        "Intel IPP ICV",
        "LicenseRef-Intel-Simplified-Software-License",
        version,
        bundled_file,
        **overrides,
    )


def test_openssl_real_bundled_files_pass():
    """The two OpenSSL files in the pinned wheel pass. The licence is not global."""
    manifest = json.loads(Path("licenses/manifest.json").read_text(encoding="utf-8"))
    rows = [e for e in manifest["entries"] if e["name"] in {"OpenSSL-libcrypto", "OpenSSL-libssl"}]
    assert len(rows) == 2
    assert {e["bundled_file"] for e in rows} == {_OPENSSL_LIBCRYPTO, _OPENSSL_LIBSSL}
    code, stdout, _ = invoke_main(make_manifest(rows))
    assert code == 0, stdout


def test_ipp_real_bundled_file_passes():
    """Intel IPP ICV and IW pass only on cv2/cv2.abi3.so in the pinned wheel."""
    manifest = json.loads(Path("licenses/manifest.json").read_text(encoding="utf-8"))
    rows = [e for e in manifest["entries"] if e["name"] in {"Intel IPP ICV", "Intel IPP IW"}]
    assert len(rows) == 2
    assert {e["bundled_file"] for e in rows} == {_IPP_FILE}
    code, stdout, _ = invoke_main(make_manifest(rows))
    assert code == 0, stdout


def test_openssl_fails_in_a_different_package():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(name="cryptography", license_id="OpenSSL", version="1.1.1w"),
            _openssl_row(bundled_package="numpy"),
        ])
    )
    assert code == 1, stdout
    assert stdout.count("license 'OpenSSL' not in allowed list") == 2


def test_openssl_fails_for_a_different_version():
    code, stdout, _ = invoke_main(make_manifest([_openssl_row(version="1.1.1k")]))
    assert code == 1, stdout
    assert "OpenSSL-libcrypto" in stdout
    assert "not in allowed list" in stdout


def test_openssl_fails_for_another_opencv_file():
    code, stdout, _ = invoke_main(
        make_manifest([_openssl_row(bundled_file="cv2/cv2.abi3.so")])
    )
    assert code == 1, stdout
    assert "license 'OpenSSL' not in allowed list" in stdout


def test_ipp_fails_in_a_different_package():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(
                name="ipp",
                license_id="LicenseRef-Intel-Simplified-Software-License",
                version="2021.11.0",
            ),
            _ipp_row(bundled_package="intel-ipp"),
        ])
    )
    assert code == 1, stdout
    assert stdout.count("not in allowed list") == 2


def test_ipp_fails_for_another_opencv_file():
    code, stdout, _ = invoke_main(
        make_manifest([
            _ipp_row(bundled_file="opencv_python_headless.libs/libippicv.a")
        ])
    )
    assert code == 1, stdout
    assert "not in allowed list" in stdout


def test_ipp_fails_for_a_different_version():
    code, stdout, _ = invoke_main(make_manifest([_ipp_row(version="2021.10.0")]))
    assert code == 1, stdout
    assert "not in allowed list" in stdout


def test_openssl_or_gpl_still_fails_on_a_named_file():
    """A banned term still fails inside an expression. The exception is not a bypass."""
    code, stdout, _ = invoke_main(
        make_manifest([_openssl_row(license="OpenSSL OR GPL-3.0")])
    )
    assert code == 1, stdout
    assert "banned" in stdout.lower()
    assert "GPL" in stdout


def test_openssl_and_ipp_are_not_globally_allowed():
    assert cl.is_license_allowed("OpenSSL") is False
    assert cl.is_license_allowed("LicenseRef-Intel-Simplified-Software-License") is False
    assert "OpenSSL" not in cl.ALLOWED_SPDX
    assert "LicenseRef-Intel-Simplified-Software-License" not in cl.ALLOWED_SPDX
    assert "OpenSSL" not in cl.ALLOWED_NON_SPDX
    assert "LicenseRef-Intel-Simplified-Software-License" not in cl.ALLOWED_NON_SPDX


def _real_rows(*names):
    manifest = json.loads(Path("licenses/manifest.json").read_text(encoding="utf-8"))
    wanted = set(names)
    return [e for e in manifest["entries"] if e["name"] in wanted]


def test_named_opencv_file_fails_when_sha256_changes():
    """A manifest row whose sha256 is not the pinned file hash fails."""
    swapped = []
    for row in _real_rows(
        "OpenSSL-libcrypto",
        "OpenSSL-libssl",
        "Intel IPP ICV",
        "Intel IPP IW",
    ):
        clone = dict(row)
        assert clone["sha256"] != "0" * 64
        clone["sha256"] = "0" * 64
        swapped.append(clone)
    assert len(swapped) == 4
    code, stdout, _ = invoke_main(make_manifest(swapped))
    assert code == 1, stdout
    assert stdout.count("not in allowed list") == 4
    for row in swapped:
        assert row["name"] in stdout


def test_named_opencv_file_fails_for_a_different_row_name():
    row = dict(_real_rows("OpenSSL-libcrypto")[0])
    row["name"] = "evil-crypto"
    code, stdout, _ = invoke_main(make_manifest([row]))
    assert code == 1, stdout
    assert "evil-crypto" in stdout
    assert "not in allowed list" in stdout


def test_named_row_mit_and_gpl_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="dual", license_id="MIT AND GPL-3.0-only")])
    )
    assert code == 1, stdout
    assert "banned" in stdout.lower()
    assert "GPL-3.0-only" in stdout


def test_ijg_and_gpl_fails():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(name="libjpeg-turbo", license_id="IJG AND GPL-3.0-only")
        ])
    )
    assert code == 1, stdout
    assert "banned" in stdout.lower()
    assert "GPL-3.0-only" in stdout


def test_ijg_and_unknown_term_fails():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(
                name="libjpeg-turbo",
                license_id="IJG AND BSD-3-Clause AND LicenseRef-Unknown",
            )
        ])
    )
    assert code == 1, stdout
    assert "not in allowed list" in stdout


def test_ijg_and_bsd_and_zlib_passes():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(
                name="libjpeg-turbo",
                license_id="IJG AND BSD-3-Clause AND Zlib",
            )
        ])
    )
    assert code == 0, stdout


def test_real_libjpeg_turbo_row_passes():
    rows = _real_rows("libjpeg-turbo")
    opencv = [
        row for row in rows
        if str(row.get("bundled_in", "")).startswith("opencv_python_headless-")
    ]
    pillow = [
        row for row in rows
        if str(row.get("bundled_in", "")).startswith("pillow-")
    ]
    assert len(opencv) == 1
    assert opencv[0]["license"] == "IJG AND BSD-3-Clause AND Zlib"
    assert len(pillow) == 1
    assert pillow[0]["license"] == "IJG AND BSD-3-Clause"
    code, stdout, _ = invoke_main(make_manifest(rows))
    assert code == 0, stdout


def test_swapped_bundled_binary_fails_installed_closure(tmp_path, monkeypatch):
    """A different file at the named path fails while the manifest stays pinned."""
    import hashlib

    rows = _real_rows(
        "OpenSSL-libcrypto",
        "OpenSSL-libssl",
        "Intel IPP ICV",
        "Intel IPP IW",
    )
    libcrypto = next(row for row in rows if row["name"] == "OpenSSL-libcrypto")
    pinned = libcrypto["sha256"]
    relative = libcrypto["bundled_file"]
    assert relative == "opencv_python_headless.libs/libcrypto-337dac8b.so.1.1"
    assert pinned == "d28a738e776f5f5c17b3577b16b6dfc093132060612769d03cf4048eba10b7aa"

    site_root = tmp_path / "site-packages"
    target = site_root / relative
    target.parent.mkdir(parents=True)
    payload = b"not-the-real-libcrypto"
    target.write_bytes(payload)
    actual = hashlib.sha256(payload).hexdigest()
    assert actual != pinned

    monkeypatch.setattr(cl, "installed_site_roots", lambda: [site_root])
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kwargs: [{"name": "trace-search", "version": "0.1.0", "license": "MIT"}],
    )
    code, stdout, _ = invoke_main(
        make_manifest(rows),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert relative in stdout
    assert actual in stdout
    assert pinned in stdout
    assert "does not match" in stdout


def _run_installed_closure(rows, tmp_path, monkeypatch, roots):
    """Run the installed-closure fence against the given site roots."""
    monkeypatch.setattr(cl, "installed_site_roots", lambda: list(roots))
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kwargs: [{"name": "trace-search", "version": "0.1.0", "license": "MIT"}],
    )
    return invoke_main(make_manifest(rows), scan_installed=True, tmp_dir=tmp_path)


def test_missing_bundled_file_fails_installed_closure(tmp_path, monkeypatch):
    """A manifest row whose named file is absent on disk fails closed."""
    rows = _real_rows("OpenSSL-libssl")
    row = rows[0]
    pinned = row["sha256"]
    relative = row["bundled_file"]
    site_root = tmp_path / "site-packages"
    site_root.mkdir()
    code, stdout, _ = _run_installed_closure(rows, tmp_path, monkeypatch, [site_root])
    assert code == 1, stdout
    assert "is missing" in stdout
    assert relative in stdout
    assert f"manifest sha256 {pinned}" in stdout
    assert f"pinned sha256 {pinned}" in stdout


def test_swapped_cv2_abi3_fails_installed_closure(tmp_path, monkeypatch):
    """A different cv2.abi3.so fails both Intel IPP rows."""
    import hashlib

    rows = _real_rows("Intel IPP ICV", "Intel IPP IW")
    relative = "cv2/cv2.abi3.so"
    pinned = rows[0]["sha256"]
    assert rows[1]["sha256"] == pinned
    assert {row["bundled_file"] for row in rows} == {relative}
    site_root = tmp_path / "site-packages"
    target = site_root / relative
    target.parent.mkdir(parents=True)
    payload = b"not-the-real-cv2-abi3"
    target.write_bytes(payload)
    actual = hashlib.sha256(payload).hexdigest()
    assert actual != pinned
    code, stdout, _ = _run_installed_closure(rows, tmp_path, monkeypatch, [site_root])
    assert code == 1, stdout
    assert relative in stdout
    assert actual in stdout
    assert pinned in stdout
    assert "does not match" in stdout
    assert "Intel IPP ICV" in stdout
    assert "Intel IPP IW" in stdout


def test_two_site_roots_different_digests_fail_installed_closure(tmp_path, monkeypatch):
    """Two site roots with different bytes for one named file fail closed."""
    import hashlib

    rows = _real_rows("OpenSSL-libcrypto")
    relative = rows[0]["bundled_file"]
    root_a = tmp_path / "site-a"
    root_b = tmp_path / "site-b"
    file_a = root_a / relative
    file_b = root_b / relative
    file_a.parent.mkdir(parents=True)
    file_b.parent.mkdir(parents=True)
    file_a.write_bytes(b"copy-a")
    file_b.write_bytes(b"copy-b")
    hash_a = hashlib.sha256(b"copy-a").hexdigest()
    hash_b = hashlib.sha256(b"copy-b").hexdigest()
    assert hash_a != hash_b
    code, stdout, _ = _run_installed_closure(
        rows, tmp_path, monkeypatch, [root_a, root_b]
    )
    assert code == 1, stdout
    assert relative in stdout
    assert hash_a in stdout
    assert hash_b in stdout
    assert "does not match on-disk sha256" in stdout


def test_installed_opencv_bundled_files_match_passes(tmp_path, monkeypatch):
    """All three named files hashing to the pin passes the installed closure."""
    rows = _real_rows(
        "OpenSSL-libcrypto",
        "OpenSSL-libssl",
        "Intel IPP ICV",
        "Intel IPP IW",
    )
    pins = {row["bundled_file"]: row["sha256"] for row in rows}
    assert set(pins) == {
        "opencv_python_headless.libs/libcrypto-337dac8b.so.1.1",
        "opencv_python_headless.libs/libssl-28bef1ac.so.1.1",
        "cv2/cv2.abi3.so",
    }
    site_root = tmp_path / "site-packages"
    for relative in pins:
        path = site_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    def fake_sha256(path):
        return pins[str(Path(path).relative_to(site_root)).replace("\\", "/")]

    monkeypatch.setattr(cl, "sha256_file", fake_sha256)
    code, stdout, _ = _run_installed_closure(rows, tmp_path, monkeypatch, [site_root])
    assert code == 0, stdout
    assert "Checking OpenCV bundled file hashes" in stdout
    for relative, digest in pins.items():
        assert f"{relative} sha256 {digest}" in stdout
    assert "License fence check PASSED" in stdout


def test_no_site_roots_rejects_named_exception(tmp_path, monkeypatch):
    """No site-packages roots: the named OpenCV exception is not applied."""
    def unavailable():
        raise AttributeError("getsitepackages")

    monkeypatch.setattr(cl.site, "getsitepackages", unavailable)
    monkeypatch.setattr(cl.site, "getusersitepackages", lambda: None)
    monkeypatch.setattr(
        cl,
        "get_installed_packages",
        lambda **kwargs: [{"name": "trace-search", "version": "0.1.0", "license": "MIT"}],
    )
    assert cl.installed_site_roots() == []
    row = dict(_real_rows("OpenSSL-libcrypto")[0])
    code, stdout, stderr = invoke_main(
        make_manifest([row]),
        scan_installed=True,
        tmp_dir=tmp_path,
    )
    assert code == 1, stdout
    assert "OpenSSL-libcrypto" in stdout
    assert "license 'OpenSSL' not in allowed list" in stdout
    assert "getsitepackages() is unavailable" in stderr


def _on_disk_pin(row):
    """on_disk map whose single digest is this row's pinned sha256."""
    return {row["bundled_file"]: [row["sha256"]]}


def test_real_libcrypto_matching_on_disk_passes(tmp_path, monkeypatch):
    """The real libcrypto row passes when the installed digest matches the pin."""
    row = dict(_real_rows("OpenSSL-libcrypto")[0])
    decision = cl._license_decision(
        row["license"], row["name"], row, _on_disk_pin(row)
    )
    assert decision == (False, True, "")

    site_root = tmp_path / "site-packages"
    target = site_root / row["bundled_file"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"placeholder")

    def fake_sha256(path):
        relative = str(Path(path).relative_to(site_root)).replace("\\", "/")
        assert relative == row["bundled_file"]
        return row["sha256"]

    monkeypatch.setattr(cl, "sha256_file", fake_sha256)
    code, stdout, _ = _run_installed_closure(
        [row], tmp_path, monkeypatch, [site_root]
    )
    assert code == 0, stdout
    assert "License fence check PASSED" in stdout


def test_real_libcrypto_on_disk_mismatch_fails(tmp_path, monkeypatch):
    """A different digest at the named libcrypto path fails the exception."""
    row = dict(_real_rows("OpenSSL-libcrypto")[0])
    site_root = tmp_path / "site-packages"
    target = site_root / row["bundled_file"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not-the-real-libcrypto")
    actual = cl.sha256_file(target)
    assert actual != row["sha256"]
    decision = cl._license_decision(
        row["license"],
        row["name"],
        row,
        {row["bundled_file"]: [actual]},
    )
    assert decision == (False, False, "")
    code, stdout, _ = _run_installed_closure(
        [row], tmp_path, monkeypatch, [site_root]
    )
    assert code == 1, stdout
    assert "license 'OpenSSL' not in allowed list" in stdout
    assert "does not match" in stdout
    assert actual in stdout


def test_openssl_or_mit_on_real_libcrypto_row_fails(tmp_path):
    """OpenSSL OR MIT is compound, so the named exception does not apply."""
    row = dict(_real_rows("OpenSSL-libcrypto")[0])
    row["license"] = "OpenSSL OR MIT"
    decision = cl._license_decision(
        row["license"], row["name"], row, _on_disk_pin(row)
    )
    assert decision == (False, False, "")
    code, stdout, _ = invoke_main(make_manifest([row]), tmp_dir=tmp_path)
    assert code == 1, stdout
    assert "license 'OpenSSL OR MIT' not in allowed list" in stdout


def test_unparseable_license_on_exception_row_fails_closed(tmp_path):
    """An unparseable licence on a named exception row fails closed."""
    row = dict(_real_rows("OpenSSL-libcrypto")[0])
    row["license"] = "(MIT OR GPL-3.0-only"
    decision = cl._license_decision(
        row["license"], row["name"], row, _on_disk_pin(row)
    )
    assert decision == (False, False, cl._UNPARSEABLE)
    code, stdout, _ = invoke_main(make_manifest([row]), tmp_dir=tmp_path)
    assert code == 1, stdout
    assert "unparseable license '(MIT OR GPL-3.0-only'" in stdout
    assert "banned license '(MIT OR GPL-3.0-only'" not in stdout


def test_openssl_semicolon_mit_on_libssl_row_fails(tmp_path):
    """OpenSSL; MIT on the libssl row is not the named exception."""
    row = dict(_real_rows("OpenSSL-libssl")[0])
    row["license"] = "OpenSSL; MIT"
    decision = cl._license_decision(
        row["license"], row["name"], row, _on_disk_pin(row)
    )
    assert decision == (False, False, "")
    code, stdout, _ = invoke_main(make_manifest([row]), tmp_dir=tmp_path)
    assert code == 1, stdout
    assert "license 'OpenSSL; MIT' not in allowed list" in stdout
