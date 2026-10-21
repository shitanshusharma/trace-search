"""Planted banned-licence cases for the CI-tools environment."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_ci_licenses as ci
import scripts.check_licenses as fence


def _write_dist(root: Path, dist_name: str, version: str, header: str = "") -> None:
    folder = root / f"{dist_name.replace('-', '_')}-{version}.dist-info"
    folder.mkdir()
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {dist_name}",
        f"Version: {version}",
    ]
    if header:
        lines.append(header)
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


def test_semgrep_and_certifi_do_not_trip_the_banned_list() -> None:
    """LGPL and MPL are not banned. Do not add an allowlist exception for them."""
    semgrep_banned, semgrep_reason = fence.is_license_banned("LGPL-2.1-or-later", "semgrep")
    assert not semgrep_banned, semgrep_reason
    certifi_spdx, certifi_spdx_reason = fence.is_license_banned("MPL-2.0", "certifi")
    assert not certifi_spdx, certifi_spdx_reason
    certifi_classifier, certifi_classifier_reason = fence.is_license_banned(
        "Mozilla Public License 2.0 (MPL 2.0)",
        "certifi",
    )
    assert not certifi_classifier, certifi_classifier_reason
