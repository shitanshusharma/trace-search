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
