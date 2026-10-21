"""Planted banned-licence cases for the CI-tools environment."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_ci_licenses as ci
import scripts.check_licenses as fence


def _write_dist(root: Path, dist_name: str, version: str, *headers: str) -> None:
    folder = root / f"{dist_name.replace('-', '_')}-{version}.dist-info"
    folder.mkdir()
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {dist_name}",
        f"Version: {version}",
        *headers,
    ]
    (folder / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("expression", ["AGPL-3.0", "GPL-3.0"])
def test_banned_licence_in_ci_env_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], expression: str
) -> None:
    _write_dist(tmp_path, "planted-banned", "1.0.0", f"License-Expression: {expression}")
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 1, output
    assert expression in output
    assert "banned license" in output


def test_missing_licence_metadata_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dist(tmp_path, "mystery-tool", "0.1.0")
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 1, output
    assert "missing or unknown" in output
    assert "PASSED" not in output


def test_unknown_token_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_dist(tmp_path, "mystery-tool", "0.1.0", "License: UNKNOWN")
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 1, output
    assert "mystery-tool" in output


def test_reviewed_face_and_peewee_warn_instead_of_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dist(tmp_path, "face", "26.0.1")
    _write_dist(tmp_path, "peewee", "3.19.0")
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 0, output
    assert "https://github.com/mahmoud/face/blob/v26.0.1/LICENSE" in output
    assert "BSD-3-Clause" in output
    assert "https://github.com/coleifer/peewee/blob/3.19.0/LICENSE" in output
    assert "MIT" in output
    assert "PASSED" in output


def test_unreviewed_version_of_face_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dist(tmp_path, "face", "26.0.2")
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 1, output
    assert "face (26.0.2)" in output


def test_override_does_not_hide_a_banned_licence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dist(tmp_path, "face", "26.0.1", "License-Expression: AGPL-3.0")
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 1, output
    assert "AGPL-3.0" in output
    assert "PASSED" not in output


def test_semgrep_and_certifi_do_not_trip_the_banned_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """LGPL and MPL pass the CI gate. Do not add an allowlist exception for them."""
    semgrep_banned, semgrep_reason = fence.is_license_banned("LGPL-2.1-or-later", "semgrep")
    assert not semgrep_banned, semgrep_reason
    certifi_spdx, certifi_spdx_reason = fence.is_license_banned("MPL-2.0", "certifi")
    assert not certifi_spdx, certifi_spdx_reason
    _write_dist(tmp_path, "semgrep", "1.182.0", "License-Expression: LGPL-2.1-or-later")
    _write_dist(
        tmp_path,
        "certifi",
        "2026.10.19",
        "Classifier: License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "License: MPL-2.0",
    )
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 0, output
    assert "PASSED" in output


def test_gplv3_classifier_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_dist(
        tmp_path,
        "planted-banned",
        "1.0.0",
        "Classifier: License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    )
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 1, output
    assert "GPLv3" in output
    assert "banned license" in output
    assert "PASSED" not in output


def test_agplv3_plus_classifier_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_dist(
        tmp_path,
        "planted-banned",
        "1.0.0",
        "Classifier: License :: OSI Approved :: "
        "GNU Affero General Public License v3 or later (AGPLv3+)",
    )
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 1, output
    assert "AGPLv3+" in output
    assert "banned license" in output
    assert "PASSED" not in output


def test_bare_gplv3_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_dist(tmp_path, "planted-banned", "1.0.0", "License: GPLv3")
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 1, output
    assert "GPLv3" in output
    assert "banned license" in output
    assert "PASSED" not in output


@pytest.mark.parametrize(
    ("tail", "marker"),
    [
        ("GNU General Public License v2 or later (GPLv2+)", "banned license"),
        ("GNU Affero General Public License v3", "banned license"),
        ("GNU General Public License (GPL)", "fail closed"),
    ],
)
def test_gpl_family_classifiers_fail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tail: str,
    marker: str,
) -> None:
    _write_dist(
        tmp_path,
        "planted-banned",
        "1.0.0",
        f"Classifier: License :: OSI Approved :: {tail}",
    )
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 1, output
    assert tail in output
    assert marker in output
    assert "PASSED" not in output
    if marker == "fail closed":
        assert "banned license" not in output


@pytest.mark.parametrize(
    "tail",
    [
        "ISC License (ISCL)",
        "The Unlicense (Unlicense)",
        "Public Domain",
    ],
)
def test_permissive_classifiers_pass(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tail: str,
) -> None:
    if tail == "Public Domain":
        header = "Classifier: License :: Public Domain"
    else:
        header = f"Classifier: License :: OSI Approved :: {tail}"
    _write_dist(tmp_path, "planted-permissive", "1.0.0", header)
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 0, output
    assert "PASSED" in output
    assert "banned license" not in output


def test_osi_approved_classifier_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_dist(tmp_path, "planted-banned", "1.0.0", "Classifier: License :: OSI Approved")
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 1, output
    assert "OSI Approved" in output
    assert "fail closed" in output
    assert "PASSED" not in output


def test_lgpl_or_later_prose_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_dist(
        tmp_path,
        "planted-lgpl",
        "1.0.0",
        "License: GNU Lesser General Public License v2 or later",
    )
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 0, output
    assert "PASSED" in output


def test_lgplv2_plus_classifier_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_dist(
        tmp_path,
        "planted-lgpl",
        "1.0.0",
        "Classifier: License :: OSI Approved :: "
        "GNU Lesser General Public License v2 or later (LGPLv2+)",
    )
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 0, output
    assert "PASSED" in output
    assert "banned license" not in output


def test_osi_approved_classifier_with_agpl_header_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dist(
        tmp_path,
        "planted-banned",
        "1.0.0",
        "Classifier: License :: OSI Approved",
        "License: AGPL-3.0",
    )
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 1, output
    assert "AGPL-3.0" in output
    assert "banned license" in output
    assert "PASSED" not in output


def test_unknown_header_does_not_hide_a_bsd_classifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dist(
        tmp_path,
        "glom",
        "25.12.0",
        "Classifier: License :: OSI Approved :: BSD License",
        "License: UNKNOWN",
    )
    code = ci.main(["--path", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 0, output
    assert "PASSED" in output
