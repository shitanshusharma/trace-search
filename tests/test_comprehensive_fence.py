"""Comprehensive tests for the enhanced license fence (in-process)."""

from tests.conftest import invoke_main, library_entry, make_manifest


def test_lgpl_gpl_dual_license_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="bad-dual-lib", license_id="LGPLv2.1+ OR GPLv2+")])
    )
    assert code == 1, stdout
    assert "GPL" in stdout.upper()
    assert "banned" in stdout.lower()


def test_gplv3_variant_fails():
    for lic in ["GPLv3", "GPL-3.0", "AGPLv3", "AGPL-3.0-only", "GPLv2+"]:
        code, stdout, _ = invoke_main(
            make_manifest([library_entry(name="gpl-lib", license_id=lic)])
        )
        assert code == 1, f"{lic} should fail: {stdout}"
        assert "banned" in stdout.lower() or "not in allowed" in stdout.lower()


def test_research_use_only_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="research-lib", license_id="MIT (research use only)")])
    )
    assert code == 1, stdout
    assert "research" in stdout.lower() or "not in allowed" in stdout.lower()


def test_sspl_busl_fails():
    for lic in ["SSPL-1.0", "BUSL-1.1", "CC-BY-SA-4.0"]:
        code, stdout, _ = invoke_main(
            make_manifest([library_entry(name="bad-lib", license_id=lic)])
        )
        assert code == 1, f"{lic} should fail: {stdout}"


def test_model_zoo_reid_weight_fails():
    manifest = make_manifest([
        {
            "name": "osnet_x1_0_duke_256x128.pth",
            "version": "1.0",
            "type": "weight",
            "license": "MIT",
            "source_url": "https://github.com/KaiyangZhou/deep-person-reid",
            "sha256": "a" * 64,
            "notes": "OSNet trained on Duke",
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1, stdout
    assert "banned component" in stdout.lower()


def test_laion_clip_fails():
    manifest = make_manifest([
        {
            "name": "CLIP-ViT-B-32-laion2B",
            "version": "1.0",
            "type": "weight",
            "license": "MIT",
            "source_url": "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "sha256": "a" * 64,
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1, stdout
    assert "laion" in stdout.lower() or "clip" in stdout.lower()


def test_insecam_clip_fails():
    manifest = make_manifest([
        {
            "name": "insecam_camera_001.mp4",
            "version": "1.0",
            "type": "clip",
            "license": "Unknown",
            "license_version": "N/A",
            "date_checked_ist": "2026-10-20",
            "source_url": "http://www.insecam.org/en/view/12345/",
            "sha256": "a" * 64,
            "phash": ["0123456789abcdef"] * 10,
            "use": "SMOKE_ONLY",
            "path": "tests/fixtures/media/insecam.mp4",
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1, stdout
    assert "insecam" in stdout.lower() or "unknown" in stdout.lower()


def test_schema_validation_empty_name_fails():
    code, stdout, _ = invoke_main(make_manifest([library_entry(name="")]))
    assert code == 1, stdout
    assert "schema" in stdout.lower() or "minLength" in stdout


def test_schema_validation_non_url_fails():
    manifest = make_manifest([
        {
            "name": "lib",
            "version": "1.0",
            "type": "library",
            "license": "MIT",
            "source_url": "see slack",
            "sha256": "a" * 64,
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1, stdout
    assert "schema" in stdout.lower() or "format" in stdout or "http" in stdout.lower()


def test_schema_validation_clip_missing_phash_fails():
    manifest = make_manifest([
        {
            "name": "clip.mp4",
            "version": "1.0",
            "type": "clip",
            "license": "MIT",
            "license_version": "1.0",
            "date_checked_ist": "2026-10-20",
            "source_url": "https://example.com",
            "sha256": "a" * 64,
            "phash": [],
            "use": "SMOKE_ONLY",
            "path": "tests/fixtures/media/clip.mp4",
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1, stdout
    assert "schema" in stdout.lower() or "minItems" in stdout


def test_schema_validation_empty_manifest_fails():
    code, stdout, _ = invoke_main(make_manifest([]))
    assert code == 1, stdout
    assert "schema" in stdout.lower() or "minItems" in stdout


def test_innocent_substring_overgrew_passes():
    code, stdout, _ = invoke_main(make_manifest([library_entry(name="overgrew")]))
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_unlicensed_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="bad-lib", license_id="UNLICENSED")])
    )
    assert code == 1, stdout
    assert "banned" in stdout.lower() or "not in allowed" in stdout.lower()


def test_commons_clause_fails():
    code, stdout, _ = invoke_main(
        make_manifest([library_entry(name="bad-lib", license_id="Apache-2.0 with Commons Clause")])
    )
    assert code == 1, stdout
    assert "commons clause" in stdout.lower() or "banned" in stdout.lower()
