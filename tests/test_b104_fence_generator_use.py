"""B-104: use, phash, and fixture_id are only valid on fixture rows.

A generator (or any other non-fixture type) tagged SMOKE_ONLY or CODE_TEST
must fail both fences, with or without pHashes. Manifests are built in
tmp_path. Nothing here is committed.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.check_licenses as check_licenses
import scripts.check_separation as check_separation
from tests.conftest import invoke_main, make_manifest
from trace_search.fixtures.separation import SeparationCheckError, forbidden_fixtures

SCHEMA = Path("licenses/manifest_schema.json")
MAIN_MANIFEST = Path("licenses/manifest.json")
_SHA = "ab" * 32
_PHASH10 = [f"{index:016x}" for index in range(10)]
_DISALLOWED_TYPES = ("library", "model", "weight", "dataset", "tool", "generator")
_GEN_NAME = "src/trace_search/fixtures/route_generator.py"


def _generator(**overrides: object) -> dict[str, object]:
    """D-T6-1 shape: a generator row with no use, fixture_id, or phash."""
    row: dict[str, object] = {
        "name": "src/trace_search/fixtures/route_generator.py",
        "version": "8af465e2e3ab3e3f9159bfd993cd0082bec62430",
        "type": "generator",
        "license": "MIT",
        "source_url": "https://example.com/route-generator",
        "sha256": _SHA,
        "generator_license": "MIT",
        "seed": "0..2147483647",
    }
    row.update(overrides)
    return row


def _typed(entry_type: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name": f"planted-{entry_type}",
        "version": "1.0.0",
        "type": entry_type,
        "license": "MIT",
        "source_url": "https://example.com/planted",
        "sha256": _SHA,
    }
    if entry_type == "generator":
        row["generator_license"] = "MIT"
        row["seed"] = "1"
    row.update(overrides)
    return row


def _clip(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name": "stock.mp4",
        "version": "1.0",
        "type": "clip",
        "license": "Pexels License",
        "license_version": "2024-01-01",
        "date_checked_ist": "2026-10-20",
        "source_url": "https://example.com/clip",
        "sha256": _SHA,
        "phash": list(_PHASH10),
        "use": "SMOKE_ONLY",
        "fixture_id": "FX-SMOKE-001",
        "path": "tests/fixtures/media/stock.mp4",
    }
    row.update(overrides)
    return row


def _synthetic(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name": "synthetic.mp4",
        "version": "1.0",
        "type": "synthetic_clip",
        "license": "CC0-1.0",
        "license_version": "1.0",
        "date_checked_ist": "2026-10-20",
        "source_url": "https://example.com/synthetic",
        "sha256": _SHA,
        "phash": list(_PHASH10),
        "use": "SMOKE_ONLY",
        "fixture_id": "FX-SMOKE-SYN-001",
        "path": "tests/fixtures/media/synthetic.mp4",
        "generator": "trace-procedural",
        "generator_version": "1.0.0",
        "generator_license": "Apache-2.0",
        "seed": "12345",
    }
    row.update(overrides)
    return row


def _code_fixture(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name": "code-sample.py",
        "version": "1.0",
        "type": "code_fixture",
        "license": "MIT",
        "source_url": "https://example.com/code",
        "sha256": _SHA,
        "phash": list(_PHASH10),
        "use": "CODE_TEST",
        "fixture_id": "FX-CODE-001",
        "generator": "trace-procedural",
        "generator_version": "1.0.0",
        "generator_license": "Apache-2.0",
        "seed": "1",
    }
    row.update(overrides)
    return row


def _write(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(make_manifest(entries)), encoding="utf-8")
    return path


def _run_separation(manifest: Path, repo_root: Path) -> tuple[int, str, str]:
    argv = [
        "check_separation.py",
        "--manifest",
        str(manifest),
        "--repo-root",
        str(repo_root),
    ]
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        with patch.object(sys, "argv", argv):
            code = check_separation.main()
    return code, stdout.getvalue(), stderr.getvalue()


def _assert_license_rejects(tmp_path: Path, row: dict[str, object], reason: str) -> None:
    code, stdout, _stderr = invoke_main(make_manifest([row]), tmp_dir=tmp_path)
    assert code == 1, stdout
    assert reason in stdout
    assert "License fence check PASSED" not in stdout


def _assert_separation_rejects(tmp_path: Path, row: dict[str, object], reason: str) -> None:
    manifest = _write(tmp_path, [row])
    code, stdout, stderr = _run_separation(manifest, tmp_path)
    assert code == 1, stdout + stderr
    assert reason in stderr
    assert "no pHash list" not in stderr
    assert "PASSED" not in stdout


def _license_use(name: str, entry_type: str, use: str) -> str:
    return f"{name}: type '{entry_type}' cannot carry use '{use}'"


def _separation_use(entry_type: str, use: str) -> str:
    return f"FAIL-CLOSED: type '{entry_type}' cannot carry use '{use}'"


def test_check_licenses_rejects_smoke_only_generator_with_10_phashes(tmp_path: Path) -> None:
    row = _generator(use="SMOKE_ONLY", phash=list(_PHASH10))
    _assert_license_rejects(
        tmp_path,
        row,
        _license_use(_GEN_NAME, "generator", "SMOKE_ONLY"),
    )


def test_check_separation_rejects_smoke_only_generator_with_10_phashes(tmp_path: Path) -> None:
    row = _generator(use="SMOKE_ONLY", phash=list(_PHASH10))
    _assert_separation_rejects(
        tmp_path,
        row,
        _separation_use("generator", "SMOKE_ONLY"),
    )


def test_check_licenses_rejects_smoke_only_generator_with_0_phashes(tmp_path: Path) -> None:
    """No phash key. This row passes check_licenses.py on main."""
    _assert_license_rejects(
        tmp_path,
        _generator(use="SMOKE_ONLY"),
        _license_use(_GEN_NAME, "generator", "SMOKE_ONLY"),
    )


def test_check_separation_rejects_smoke_only_generator_with_0_phashes(tmp_path: Path) -> None:
    """No phash key. Separation must name the row type, not a missing pHash list."""
    _assert_separation_rejects(
        tmp_path,
        _generator(use="SMOKE_ONLY"),
        _separation_use("generator", "SMOKE_ONLY"),
    )


def test_check_licenses_rejects_code_test_generator(tmp_path: Path) -> None:
    row = _generator(use="CODE_TEST", phash=list(_PHASH10))
    _assert_license_rejects(
        tmp_path,
        row,
        _license_use(_GEN_NAME, "generator", "CODE_TEST"),
    )


def test_check_separation_rejects_code_test_generator(tmp_path: Path) -> None:
    row = _generator(use="CODE_TEST", phash=list(_PHASH10))
    _assert_separation_rejects(
        tmp_path,
        row,
        _separation_use("generator", "CODE_TEST"),
    )


def test_check_licenses_rejects_phash_on_generator(tmp_path: Path) -> None:
    _assert_license_rejects(
        tmp_path,
        _generator(phash=list(_PHASH10)),
        f"{_GEN_NAME}: type 'generator' cannot carry phash",
    )


def test_check_licenses_rejects_fixture_id_on_generator(tmp_path: Path) -> None:
    _assert_license_rejects(
        tmp_path,
        _generator(fixture_id="FX-SMOKE-001"),
        f"{_GEN_NAME}: type 'generator' cannot carry fixture_id",
    )


def test_schema_rejects_smoke_only_generator_with_10_phashes() -> None:
    errors = check_licenses.validate_schema(
        make_manifest([_generator(use="SMOKE_ONLY", phash=list(_PHASH10))]),
        SCHEMA,
    )
    assert any("SMOKE_ONLY" in error for error in errors), errors


def test_schema_rejects_smoke_only_generator_with_0_phashes() -> None:
    errors = check_licenses.validate_schema(
        make_manifest([_generator(use="SMOKE_ONLY")]),
        SCHEMA,
    )
    assert any("SMOKE_ONLY" in error for error in errors), errors


def test_schema_rejects_phash_on_generator() -> None:
    errors = check_licenses.validate_schema(
        make_manifest([_generator(phash=list(_PHASH10))]),
        SCHEMA,
    )
    assert any(_PHASH10[0] in error for error in errors), errors


def test_schema_rejects_fixture_id_on_generator() -> None:
    errors = check_licenses.validate_schema(
        make_manifest([_generator(fixture_id="FX-SMOKE-001")]),
        SCHEMA,
    )
    assert any("FX-SMOKE-001" in error for error in errors), errors


def test_row_check_rejects_smoke_only_generator_with_10_phashes() -> None:
    violations = check_licenses.check_manifest_licenses(
        make_manifest([_generator(use="SMOKE_ONLY", phash=list(_PHASH10))])
    )
    assert violations == [
        _license_use(_GEN_NAME, "generator", "SMOKE_ONLY"),
        f"{_GEN_NAME}: type 'generator' cannot carry phash",
    ]


def test_row_check_rejects_smoke_only_generator_with_0_phashes() -> None:
    violations = check_licenses.check_manifest_licenses(
        make_manifest([_generator(use="SMOKE_ONLY")])
    )
    assert violations == [_license_use(_GEN_NAME, "generator", "SMOKE_ONLY")]


def test_row_check_rejects_phash_without_use_on_generator() -> None:
    violations = check_licenses.check_manifest_licenses(
        make_manifest([_generator(phash=list(_PHASH10))])
    )
    assert violations == [f"{_GEN_NAME}: type 'generator' cannot carry phash"]


def test_row_check_rejects_fixture_id_without_use_on_generator() -> None:
    violations = check_licenses.check_manifest_licenses(
        make_manifest([_generator(fixture_id="FX-SMOKE-001")])
    )
    assert violations == [f"{_GEN_NAME}: type 'generator' cannot carry fixture_id"]


def test_row_check_reports_phash_and_fixture_id() -> None:
    violations = check_licenses.check_manifest_licenses(
        make_manifest([_generator(phash=list(_PHASH10), fixture_id="FX-SMOKE-001")])
    )
    assert violations == [
        f"{_GEN_NAME}: type 'generator' cannot carry phash",
        f"{_GEN_NAME}: type 'generator' cannot carry fixture_id",
    ]


def test_row_check_names_unknown_when_the_row_has_no_name() -> None:
    row = _generator(use="SMOKE_ONLY")
    del row["name"]
    violations = check_licenses.check_manifest_licenses(make_manifest([row]))
    assert violations == [_license_use("unknown", "generator", "SMOKE_ONLY")]


def test_row_check_names_none_when_the_type_is_missing() -> None:
    row = _generator(use="SMOKE_ONLY")
    del row["type"]
    violations = check_licenses.check_manifest_licenses(make_manifest([row]))
    assert violations == [_license_use(_GEN_NAME, "None", "SMOKE_ONLY")]


def test_row_check_reports_both_violating_rows() -> None:
    rows = [
        _generator(name="gen-a", use="SMOKE_ONLY"),
        _generator(name="gen-b", use="CODE_TEST"),
    ]
    violations = check_licenses.check_manifest_licenses(make_manifest(rows))
    assert violations == [
        _license_use("gen-a", "generator", "SMOKE_ONLY"),
        _license_use("gen-b", "generator", "CODE_TEST"),
    ]


def test_field_message_remains_when_the_license_is_banned() -> None:
    row = _generator(name="gen-gpl", license="GPL-3.0", use="SMOKE_ONLY")
    violations = check_licenses.check_manifest_licenses(make_manifest([row]))
    assert _license_use("gen-gpl", "generator", "SMOKE_ONLY") in violations
    assert any("banned license 'GPL-3.0'" in item for item in violations)


@pytest.mark.parametrize("entry_type", _DISALLOWED_TYPES)
def test_check_licenses_rejects_smoke_only_on_other_types(tmp_path: Path, entry_type: str) -> None:
    row = _typed(entry_type, use="SMOKE_ONLY", phash=list(_PHASH10))
    _assert_license_rejects(
        tmp_path,
        row,
        _license_use(f"planted-{entry_type}", entry_type, "SMOKE_ONLY"),
    )


@pytest.mark.parametrize("entry_type", _DISALLOWED_TYPES)
def test_check_separation_rejects_smoke_only_on_other_types(
    tmp_path: Path, entry_type: str
) -> None:
    row = _typed(entry_type, use="SMOKE_ONLY", phash=list(_PHASH10))
    _assert_separation_rejects(
        tmp_path,
        row,
        _separation_use(entry_type, "SMOKE_ONLY"),
    )


@pytest.mark.parametrize("entry_type", _DISALLOWED_TYPES)
@pytest.mark.parametrize("use", ("SMOKE_ONLY", "CODE_TEST"))
def test_schema_rejects_use_on_other_types(entry_type: str, use: str) -> None:
    errors = check_licenses.validate_schema(
        make_manifest([_typed(entry_type, use=use)]),
        SCHEMA,
    )
    assert any(use in error for error in errors), errors


def test_forbidden_fixtures_names_the_row_type_for_a_smoke_generator() -> None:
    reason = _separation_use("generator", "SMOKE_ONLY")
    with pytest.raises(SeparationCheckError, match=reason):
        forbidden_fixtures([_generator(use="SMOKE_ONLY", phash=list(_PHASH10))])


def test_separation_forbidden_tag_without_type_names_none() -> None:
    row = _generator(use="SMOKE_ONLY", phash=list(_PHASH10))
    del row["type"]
    with pytest.raises(SeparationCheckError, match=_separation_use("None", "SMOKE_ONLY")):
        forbidden_fixtures([row])


def test_row_check_rejects_a_non_string_type() -> None:
    """A list is unhashable. Membership must not raise TypeError."""
    row = _generator(use="SMOKE_ONLY")
    row["type"] = ["generator"]
    violations = check_licenses.check_manifest_licenses(make_manifest([row]))
    assert _license_use(_GEN_NAME, "list", "SMOKE_ONLY") in violations


def test_separation_rejects_a_non_string_type() -> None:
    """A list is unhashable. The fence must name the type, not raise TypeError."""
    row = _generator(use="SMOKE_ONLY", phash=list(_PHASH10))
    row["type"] = ["generator"]
    with pytest.raises(SeparationCheckError, match=_separation_use("list", "SMOKE_ONLY")):
        forbidden_fixtures([row])


def test_check_licenses_accepts_clip_with_use(tmp_path: Path) -> None:
    code, stdout, _stderr = invoke_main(make_manifest([_clip()]), tmp_dir=tmp_path)
    assert code == 0, stdout


def test_check_licenses_accepts_synthetic_clip_with_use(tmp_path: Path) -> None:
    code, stdout, _stderr = invoke_main(make_manifest([_synthetic()]), tmp_dir=tmp_path)
    assert code == 0, stdout


def test_check_licenses_accepts_code_fixture_with_use(tmp_path: Path) -> None:
    code, stdout, _stderr = invoke_main(make_manifest([_code_fixture()]), tmp_dir=tmp_path)
    assert code == 0, stdout


def test_schema_accepts_clip_synthetic_clip_and_code_fixture() -> None:
    for row in (_clip(), _synthetic(), _code_fixture()):
        errors = check_licenses.validate_schema(make_manifest([row]), SCHEMA)
        assert errors == [], errors


def test_check_separation_accepts_clip_with_use(tmp_path: Path) -> None:
    manifest = _write(tmp_path, [_clip()])
    code, stdout, stderr = _run_separation(manifest, tmp_path)
    assert code == 0, stdout + stderr
    assert "PASSED" in stdout


def test_check_separation_accepts_synthetic_clip_with_use(tmp_path: Path) -> None:
    manifest = _write(tmp_path, [_synthetic()])
    code, stdout, stderr = _run_separation(manifest, tmp_path)
    assert code == 0, stdout + stderr
    assert "PASSED" in stdout


def test_check_separation_accepts_code_fixture_with_use(tmp_path: Path) -> None:
    manifest = _write(tmp_path, [_code_fixture()])
    code, stdout, stderr = _run_separation(manifest, tmp_path)
    assert code == 0, stdout + stderr
    assert "PASSED" in stdout


def test_row_check_accepts_clip_synthetic_clip_and_code_fixture() -> None:
    for row in (_clip(), _synthetic(), _code_fixture()):
        assert check_licenses.check_manifest_licenses(make_manifest([row])) == []


def test_dt61_generator_without_use_passes_check_licenses(tmp_path: Path) -> None:
    code, stdout, _stderr = invoke_main(make_manifest([_generator()]), tmp_dir=tmp_path)
    assert code == 0, stdout


def test_dt61_generator_without_use_passes_check_separation(tmp_path: Path) -> None:
    """A clean generator row stays out of the forbidden set. A real clip row still counts."""
    generator_sha = "cd" * 32
    generator = _generator(sha256=generator_sha)
    manifest = _write(tmp_path, [generator, _synthetic()])
    code, stdout, stderr = _run_separation(manifest, tmp_path)
    assert code == 0, stdout + stderr
    sha_to_id, _sha_to_phash = forbidden_fixtures([generator, _synthetic()])
    assert generator_sha not in sha_to_id
    assert sha_to_id[_SHA] == "FX-SMOKE-SYN-001"


def test_main_manifest_passes_check_licenses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        check_licenses,
        "inspect_opencv_libavcodec",
        lambda: ("LGPL version 2.1 or later", "--enable-shared"),
    )
    code = check_licenses.main(["--manifest", str(MAIN_MANIFEST), "--no-scan-installed"])
    assert code == 0


def test_main_manifest_passes_check_separation() -> None:
    code, stdout, stderr = _run_separation(MAIN_MANIFEST, Path("."))
    assert code == 0, stdout + stderr
    assert "PASSED" in stdout
