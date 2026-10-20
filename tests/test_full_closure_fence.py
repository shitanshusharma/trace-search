"""Comprehensive tests for full closure license fence (in-process)."""

from tests.conftest import invoke_main, library_entry, make_manifest


def test_semicolon_joined_license_with_gpl_fails():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(name="python-dateutil", license_id="Apache Software License; GPL-2.0")
        ])
    )
    assert code == 1, stdout
    assert "GPL" in stdout or "banned" in stdout.lower()


def test_underscore_joined_banned_name_fails():
    manifest = make_manifest([
        {
            "name": "ultralytics_yolov8n",
            "version": "1.0.0",
            "type": "weight",
            "license": "MIT",
            "source_url": "https://github.com/ultralytics/yolov8",
            "sha256": "a" * 64,
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1, stdout
    assert "ultralytics" in stdout.lower()


def test_retinaface_resnet50_fails():
    manifest = make_manifest([
        {
            "name": "retinaface_resnet50",
            "version": "1.0.0",
            "type": "weight",
            "license": "MIT",
            "source_url": "https://github.com/biubug6/RetinaFace",
            "sha256": "a" * 64,
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1, stdout
    assert "retinaface" in stdout.lower() or "face" in stdout.lower()


def test_buffalo_l_face_model_fails():
    manifest = make_manifest([
        {
            "name": "buffalo_l",
            "version": "1.0.0",
            "type": "weight",
            "license": "MIT",
            "source_url": "https://github.com/deepinsight/insightface",
            "sha256": "a" * 64,
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1, stdout
    assert "buffalo" in stdout.lower() or "face" in stdout.lower()


def test_mtcnn_face_model_fails():
    code, stdout, _ = invoke_main(make_manifest([library_entry(name="mtcnn")]))
    assert code == 1, stdout
    assert "mtcnn" in stdout.lower() or "face" in stdout.lower()


def test_scrfd_face_model_fails():
    manifest = make_manifest([
        {
            "name": "scrfd_10g_bnkps",
            "version": "1.0.0",
            "type": "weight",
            "license": "MIT",
            "source_url": "https://github.com/deepinsight/insightface",
            "sha256": "a" * 64,
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 1, stdout
    assert "scrfd" in stdout.lower() or "face" in stdout.lower()


def test_facenet_pytorch_fails():
    code, stdout, _ = invoke_main(make_manifest([library_entry(name="facenet-pytorch")]))
    assert code == 1, stdout
    assert "facenet" in stdout.lower() or "face" in stdout.lower()


def test_schema_synthetic_clip_fixture_id():
    manifest = make_manifest([
        {
            "name": "synthetic_walk_01.mp4",
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
            "generator": "trace-procedural",
            "generator_version": "3.6.0",
            "generator_license": "Apache-2.0",
            "seed": "12345",
        }
    ])
    code, stdout, _ = invoke_main(manifest)
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_clean_real_closure_passes():
    import scripts.check_licenses as cl

    code = cl.main(["--manifest", "licenses/manifest.json", "--no-scan-installed"])
    assert code == 0


def test_hpnd_license_passes():
    code, stdout, _ = invoke_main(make_manifest([library_entry(name="Pillow", license_id="HPND")]))
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_hpnd_non_spdx_variant_passes():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(
                name="Pillow",
                license_id="Historical Permission Notice and Disclaimer (HPND)",
            )
        ])
    )
    assert code == 0, stdout
    assert "PASSED" in stdout


def test_gpl_still_fails_with_hpnd_allowed():
    code, stdout, _ = invoke_main(
        make_manifest([
            library_entry(name="Pillow", license_id="HPND"),
            library_entry(name="gpl-package", license_id="GPL-3.0"),
        ])
    )
    assert code == 1, stdout
    assert "GPL" in stdout or "banned" in stdout.lower()
