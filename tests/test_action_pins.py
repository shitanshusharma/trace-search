"""Planted pin checks. Network is mocked; the bad fixture must fail."""

from pathlib import Path

from scripts.validate_action_shas import (
    Pin,
    check_pin,
    iter_pins,
    peeled_commit,
    scan_uses,
    validate_repo,
)

ROOT = Path(__file__).parents[1]
GOOD = "eef61447b9ff4aafe5dcd4e0bbf5d482be7e7871"
PEELED = "662472033e021d55d94146f66f6058822b0b39fd"
TAG_OBJECT = "9887d98ae49f1f598651b556d8c8f02f3ea065cb"
BAD = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_planted_bad_sha_fixture_fails():
    text = (ROOT / "tests" / "fixtures" / "bad-action-pin.yml").read_text()
    pins = iter_pins(text, "tests/fixtures/bad-action-pin.yml")
    assert len(pins) == 1

    def ls_remote(_owner, _repo, _tag):
        return f"{GOOD}\trefs/tags/v4.2.1\n"

    def get(url):
        # Fork-network imposters can still get HTTP 200 from the commits API.
        if "/commits/" in url:
            return 200, {"sha": BAD}
        return 200, {"status": "identical"}

    errors = check_pin(pins[0], get, ls_remote)
    assert errors
    assert "peeled commit" in errors[0]


def test_tag_object_pin_is_rejected():
    remote = f"{TAG_OBJECT}\trefs/tags/v3.27.0\n{PEELED}\trefs/tags/v3.27.0^{{}}\n"
    assert peeled_commit(remote, "v3.27.0") == PEELED
    pin = Pin(
        file=".github/workflows/codeql.yml",
        line=1,
        action="github/codeql-action/init",
        sha=TAG_OBJECT,
        version="3.27.0",
    )

    calls: list[str] = []

    def get(url):
        calls.append(url)
        return 422, {"message": "No commit found for SHA"}

    errors = check_pin(pin, get, lambda *_args: remote)
    assert errors
    assert PEELED in errors[0]
    assert calls == []


def test_api_non_200_fails_even_when_tag_matches():
    pin = Pin("wf.yml", 1, "actions/checkout", GOOD, "4.2.1")
    remote = f"{GOOD}\trefs/tags/v4.2.1\n"

    def get(url):
        if "/compare/" in url:
            return 200, {"status": "identical"}
        return 404, {"message": "Not Found"}

    errors = check_pin(pin, get, lambda *_args: remote)
    assert errors
    assert "404" in errors[0]


def test_good_peeled_pin_passes():
    pin = Pin("wf.yml", 1, "github/codeql-action/analyze", PEELED, "3.27.0")
    remote = f"{TAG_OBJECT}\trefs/tags/v3.27.0\n{PEELED}\trefs/tags/v3.27.0^{{}}\n"
    seen = []

    def get(url):
        seen.append(url)
        if "/compare/" in url:
            return 200, {"status": "identical"}
        return 200, {"sha": PEELED, "commit": {"message": "v3.27.0"}}

    assert check_pin(pin, get, lambda *_args: remote) == []
    assert any(url.endswith(f"/commits/{PEELED}") for url in seen)
    assert any("/compare/" in url for url in seen)
    assert len(seen) == 2


def test_sha_pin_without_version_comment_fails(tmp_path):
    """A 40-hex pin with no comment must fail before any API call can save it."""
    text = (ROOT / "tests" / "fixtures" / "bad-action-pin-no-comment.yml").read_text()
    pins, errors = scan_uses(text, "tests/fixtures/bad-action-pin-no-comment.yml")
    assert pins == []
    assert errors
    assert "vX.Y.Z" in errors[0]

    sha = "a" * 40
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "evil.yml").write_text(
        f"name: evil\njobs:\n  j:\n    steps:\n      - uses: evil/action@{sha}\n"
    )
    calls: list[str] = []

    def get(url):
        calls.append(url)
        return 200, {"sha": sha, "status": "identical"}

    repo_errors = validate_repo(tmp_path, get, lambda *_args: f"{sha}\trefs/tags/v1.0.0\n")
    assert repo_errors
    assert calls == []
    assert "vX.Y.Z" in repo_errors[0]


def test_comment_without_v_and_mutable_tag_fail():
    sha = "b" * 40
    _pins, no_v = scan_uses(f"uses: evil/action@{sha} # 1.0.0\n", "w.yml")
    assert no_v
    _pins, mutable = scan_uses("uses: actions/checkout@v4\n", "w.yml")
    assert mutable
    pins, local_errors = scan_uses("uses: ./.github/actions/local\n", "w.yml")
    assert pins == []
    assert local_errors == []
    pins, docker_errors = scan_uses("uses: docker://alpine:3\n", "w.yml")
    assert pins == []
    assert docker_errors == []
