"""The ReID loader refuses smoke and code-test data and fails closed.

Clips are generated into pytest's temp directory. Nothing is read from
``smoke/`` and nothing is written into the repo tree.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import trace_search.reid as reid
import trace_search.reid.data as data_module
from tests.media_helpers import write_mp4v_variant
from tests.reid.conftest import rgb_frame
from trace_search.fixtures.content import sha256_file
from trace_search.fixtures.guards import FixtureSeparationError, FixtureUseTag, LoaderContext

ReidSample = data_module.ReidSample
load_reid_samples = data_module.load_reid_samples

_REID_CONTEXTS = (LoaderContext.TRAINING, LoaderContext.EVALUATION)
_OTHER_CONTEXTS = (
    LoaderContext.SMOKE_TEST,
    LoaderContext.UNIT_TEST,
    LoaderContext.STUDY,
)

# U+FF26 U+FF38 then ASCII "-SMOKE-SYN-001". NFKC folds the two letters.
_FULLWIDTH_SMOKE_ID = "\uff26\uff38-SMOKE-SYN-001"

_SMOKE_IDS = (
    "FX-SMOKE-SYN-001",
    "fx-smoke-syn-001",
    "Fx-Smoke-Syn-001",
    "  FX-SMOKE-SYN-001 ",
    "FX_SMOKE_SYN_001",
    _FULLWIDTH_SMOKE_ID,
)
_CODE_IDS = (
    "FX-CODE-001",
    "fx_code_001",
)
_FORBIDDEN_TAGS = (
    ("SMOKE_ONLY", "SMOKE_ONLY"),
    ("smoke_only", "SMOKE_ONLY"),
    (" Smoke_Only ", "SMOKE_ONLY"),
    ("CODE_TEST", "CODE_TEST"),
    ("code_test", "CODE_TEST"),
)
_UNKNOWN_TAGS = ("", "TRAINING")
_UNLISTED_IDS = ("FX-HOLD-001", "anything")

# Test-only. Production training patterns stay empty.
_BATCH_PATTERN = re.compile(r"FX-BATCH-\d{3}")


def _sample(
    fixture_id: str,
    use: str = "TRAIN",
    *,
    path: Path | None = None,
    crop: np.ndarray | None = None,
    track_id: int = 1,
    label: int = 0,
) -> ReidSample:
    return ReidSample(
        fixture_id=fixture_id,
        use=use,
        camera_id="cam-1",
        track_id=track_id,
        label=label,
        crop=crop,
        path=path,
    )


def _write_manifest(directory: Path, entries: Sequence[object] | None = None) -> Path:
    path = directory / "manifest.json"
    payload = {
        "manifest_version": "1.0",
        "entries": [] if entries is None else list(entries),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _refused(
    samples: list[ReidSample],
    context: LoaderContext,
    manifest: Path,
    phrase: str,
) -> str:
    with pytest.raises(FixtureSeparationError) as caught:
        load_reid_samples(samples, context, manifest)
    message = str(caught.value)
    assert phrase in message
    return message


def _entry_clip(directory: Path, entry: dict[str, object]) -> Path:
    fixture_id = str(entry["fixture_id"])
    matches = sorted(directory.glob(f"{fixture_id}_*.mp4"))
    assert len(matches) == 1
    clip = matches[0]
    assert sha256_file(clip) == entry["sha256"]
    return clip


def test_reid_sample_fields_are_the_spec_set() -> None:
    assert [field.name for field in fields(ReidSample)] == [
        "fixture_id",
        "use",
        "camera_id",
        "track_id",
        "label",
        "crop",
        "path",
    ]


def test_label_is_an_anonymous_group_index() -> None:
    doc = ReidSample.__doc__ or ""
    assert "anonymous" in doc
    assert "group index" in doc
    assert "not a person id" in doc


def test_package_exports_the_loader() -> None:
    assert reid.ReidSample is ReidSample
    assert reid.load_reid_samples is load_reid_samples


@pytest.mark.parametrize("fixture_id", _SMOKE_IDS)
@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_smoke_id_is_refused(
    fixture_id: str,
    context: LoaderContext,
    tmp_path: Path,
) -> None:
    message = _refused(
        [_sample(fixture_id)],
        context,
        _write_manifest(tmp_path),
        "normalised id starts with FX-SMOKE",
    )
    assert fixture_id in message


@pytest.mark.parametrize("fixture_id", _CODE_IDS)
@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_code_id_is_refused(
    fixture_id: str,
    context: LoaderContext,
    tmp_path: Path,
) -> None:
    message = _refused(
        [_sample(fixture_id)],
        context,
        _write_manifest(tmp_path),
        "normalised id starts with FX-CODE",
    )
    assert fixture_id in message


@pytest.mark.parametrize(("raw_tag", "normalised"), _FORBIDDEN_TAGS)
@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_forbidden_use_tag_is_refused(
    raw_tag: str,
    normalised: str,
    context: LoaderContext,
    tmp_path: Path,
) -> None:
    message = _refused(
        [_sample("FX-HOLD-001", use=raw_tag)],
        context,
        _write_manifest(tmp_path),
        f"forbidden use tag {normalised}",
    )
    assert raw_tag in message


@pytest.mark.parametrize("raw_tag", _UNKNOWN_TAGS)
@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_unknown_use_tag_is_refused(
    raw_tag: str,
    context: LoaderContext,
    tmp_path: Path,
) -> None:
    normalised = raw_tag.strip().upper()
    message = _refused(
        [_sample("FX-HOLD-001", use=raw_tag)],
        context,
        _write_manifest(tmp_path),
        f"unknown use tag {normalised!r}",
    )
    assert raw_tag in message


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_smoke_bytes_are_refused_by_sha256(
    context: LoaderContext,
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    manifest = _write_manifest(tmp_path, entries)
    _refused(
        [_sample("FX-SYNREID-001", path=clip)],
        context,
        manifest,
        "identified by SHA-256 or pHash",
    )


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_renamed_dat_smoke_bytes_are_refused(
    context: LoaderContext,
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    renamed = tmp_path / "renamed.dat"
    renamed.write_bytes(clip.read_bytes())
    assert renamed.suffix == ".dat"
    assert sha256_file(renamed) == entries[0]["sha256"]
    manifest = _write_manifest(tmp_path, entries)
    _refused(
        [_sample("FX-SYNREID-001", path=renamed)],
        context,
        manifest,
        "identified by SHA-256 or pHash",
    )


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_reencoded_smoke_clip_is_refused_by_phash(
    context: LoaderContext,
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    variant = tmp_path / "reencoded.mp4"
    write_mp4v_variant(clip, variant, scale=0.5)
    assert sha256_file(variant) != entries[0]["sha256"]
    manifest = _write_manifest(tmp_path, entries)
    _refused(
        [_sample("FX-SYNREID-001", path=variant)],
        context,
        manifest,
        "identified by SHA-256 or pHash",
    )


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_path_with_no_manifest_row_is_refused(
    context: LoaderContext,
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    manifest = _write_manifest(tmp_path, [])
    _refused(
        [_sample("FX-SYNREID-001", path=clip)],
        context,
        manifest,
        "no manifest row",
    )


@pytest.mark.parametrize("fixture_id", _UNLISTED_IDS)
@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_empty_allowlist_refuses_other_ids(
    fixture_id: str,
    context: LoaderContext,
    tmp_path: Path,
) -> None:
    phrase = "not on the ReID allowlist"
    if context is LoaderContext.TRAINING:
        phrase = "refused in TRAINING"
    message = _refused(
        [_sample(fixture_id)],
        context,
        _write_manifest(tmp_path),
        phrase,
    )
    if context is LoaderContext.EVALUATION:
        assert fixture_id in message


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_one_planted_sample_refuses_the_whole_batch(
    context: LoaderContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nine pattern-matching samples and one planted id: the batch is refused.

    The production allowlist is empty, so nothing matches it. This test
    injects a pattern only here, which makes the nine samples good-looking.
    The planted id still refuses the batch, and the caller gets nothing back.
    """
    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", (_BATCH_PATTERN,))
    monkeypatch.setattr(data_module, "_TRAINING_ID_PATTERNS", (_BATCH_PATTERN,))
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    samples = [
        _sample(f"FX-BATCH-{index:03d}", track_id=index, label=index, crop=crop)
        for index in range(1, 10)
    ]
    planted_id = "fx-smoke-syn-001"
    samples.append(_sample(planted_id, track_id=10, label=10, crop=crop))
    manifest = _write_manifest(tmp_path)
    with pytest.raises(FixtureSeparationError) as caught:
        returned = load_reid_samples(samples, context, manifest)
        pytest.fail(f"loader returned {returned!r}")
    message = str(caught.value)
    assert planted_id in message
    assert "normalised id starts with FX-SMOKE" in message


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_injected_pattern_returns_a_matching_sample(
    context: LoaderContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A test-only pattern can accept one id in evaluation.

    Training still refuses TRAIN, HOLD, and DEV. The production training
    tuple stays empty.
    """
    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", (_BATCH_PATTERN,))
    monkeypatch.setattr(data_module, "_TRAINING_ID_PATTERNS", (_BATCH_PATTERN,))
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    crop[0, 0] = (1, 2, 3)
    sample = _sample("FX-BATCH-004", track_id=4, label=2, crop=crop)
    manifest = _write_manifest(tmp_path)
    if context is LoaderContext.TRAINING:
        _refused([sample], context, manifest, "refused in TRAINING")
        return
    loaded = load_reid_samples([sample], context, manifest)
    assert len(loaded) == 1 and loaded[0] is sample


@pytest.mark.parametrize("context", _OTHER_CONTEXTS)
def test_other_contexts_are_refused(context: LoaderContext, tmp_path: Path) -> None:
    _refused([], context, _write_manifest(tmp_path), "Only TRAINING and EVALUATION")


def test_string_context_is_refused(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    with pytest.raises(FixtureSeparationError, match="Only TRAINING and EVALUATION"):
        load_reid_samples([], "training", manifest)  # type: ignore[arg-type]


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_empty_batch_is_accepted(context: LoaderContext, tmp_path: Path) -> None:
    assert load_reid_samples([], context, _write_manifest(tmp_path)) == []


def _benign_crop(fill: int = 17) -> np.ndarray:
    crop = np.full((8, 8, 3), fill, dtype=np.uint8)
    crop[0, 0] = (fill, fill + 1, fill + 2)
    return crop


def test_training_allowlist_stays_empty() -> None:
    assert data_module._TRAINING_ID_PATTERNS == ()
    assert data_module._EVALUATION_ID_PATTERNS == (data_module._SYNREID_ID,)
    assert data_module._patterns_for(LoaderContext.TRAINING) == ()


def test_recognised_use_tags_match_fixture_use_tag() -> None:
    assert data_module._RECOGNISED_USE_TAGS == frozenset(tag.value for tag in FixtureUseTag)


@pytest.mark.parametrize("use", ("TRAIN", "HOLD", "DEV"))
def test_train_hold_and_dev_are_refused_in_training(
    use: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tag gate refuses these even when the id is injected onto TRAINING."""
    monkeypatch.setattr(data_module, "_TRAINING_ID_PATTERNS", (data_module._SYNREID_ID,))
    sample = _sample("FX-SYNREID-001", use=use, crop=_benign_crop())
    message = _refused(
        [sample],
        LoaderContext.TRAINING,
        _write_manifest(tmp_path),
        "refused in TRAINING",
    )
    assert use in message


@pytest.mark.parametrize("use", ("HOLD", "TRAIN"))
def test_hold_and_train_are_refused_on_synreid_in_evaluation(
    use: str,
    tmp_path: Path,
) -> None:
    sample = _sample("FX-SYNREID-001", use=use, crop=_benign_crop(40))
    message = _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "refused in EVALUATION",
    )
    assert use in message


def test_arabic_indic_digits_are_refused(tmp_path: Path) -> None:
    """Digits that NFKC leaves unchanged are not ASCII ``[0-9]``."""
    fixture_id = "FX-SYNREID-" + "\u0660\u0660\u0661"
    sample = _sample(fixture_id, use="DEV", crop=_benign_crop(21))
    message = _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "not on the ReID allowlist",
    )
    assert fixture_id in message


def test_fullwidth_digits_fold_onto_the_evaluation_allowlist(tmp_path: Path) -> None:
    fixture_id = "FX-SYNREID-\uff10\uff10\uff11"
    sample = _sample(fixture_id, use="DEV", crop=_benign_crop(22))
    loaded = load_reid_samples([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path))
    assert len(loaded) == 1 and loaded[0] is sample


def test_non_string_fixture_id_raises_fixture_separation_error(tmp_path: Path) -> None:
    sample = _sample("FX-SYNREID-001", use="DEV", crop=_benign_crop())
    sample = ReidSample(
        fixture_id=1,  # type: ignore[arg-type]
        use=sample.use,
        camera_id=sample.camera_id,
        track_id=sample.track_id,
        label=sample.label,
        crop=sample.crop,
        path=sample.path,
    )
    with pytest.raises(FixtureSeparationError, match="fixture id must be a string"):
        load_reid_samples([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path))


def test_non_string_use_tag_raises_fixture_separation_error(tmp_path: Path) -> None:
    sample = ReidSample(
        fixture_id="FX-SYNREID-001",
        use=1,  # type: ignore[arg-type]
        camera_id="cam-1",
        track_id=1,
        label=0,
        crop=_benign_crop(),
        path=None,
    )
    with pytest.raises(FixtureSeparationError, match="use tag must be a string"):
        load_reid_samples([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path))


def test_loaded_crop_is_a_read_only_copy(tmp_path: Path) -> None:
    crop = _benign_crop(4)
    original = crop.copy()
    sample = _sample("FX-SYNREID-001", use="DEV", crop=crop)
    loaded = load_reid_samples([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path))
    assert len(loaded) == 1 and loaded[0] is sample
    sealed = loaded[0].crop
    assert sealed is not None
    assert sealed is not crop
    assert sealed.flags.writeable is False
    with pytest.raises(ValueError, match="WRITEABLE"):
        sealed.setflags(write=True)
    crop[0, 0] = (9, 9, 9)
    assert np.array_equal(sealed, original)
    with pytest.raises(ValueError):
        sealed[0, 0] = (1, 1, 1)


def test_smoke_frame_crop_is_refused(
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    sample = _sample("FX-SYNREID-001", use="DEV", crop=rgb_frame(clip))
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "identified by pHash",
    )


def test_known_gap_centre_crop_of_a_smoke_frame_is_accepted(
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    """Known gap: a box cut from a smoke frame is not refused.

    The crop check matches a whole smoke frame by pHash. It does not
    match the centre of that frame. This pins the current behaviour.
    """
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    frame = rgb_frame(clip)
    height, width = frame.shape[:2]
    top = height // 4
    left = width // 4
    centre = np.ascontiguousarray(frame[top : top + height // 2, left : left + width // 2])
    sample = _sample("FX-SYNREID-001", use="DEV", crop=centre)
    loaded = load_reid_samples([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path))
    assert len(loaded) == 1 and loaded[0] is sample


def test_zero_by_zero_crop_is_refused(tmp_path: Path) -> None:
    crop = np.zeros((0, 0, 3), dtype=np.uint8)
    sample = _sample("FX-SYNREID-001", use="DEV", crop=crop)
    _refused([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path), "non-empty HxWx3")


def test_grayscale_crop_is_refused(tmp_path: Path) -> None:
    crop = np.zeros((8, 8), dtype=np.uint8)
    sample = _sample("FX-SYNREID-001", use="DEV", crop=crop)
    _refused([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path), "non-empty HxWx3")


def test_rgba_crop_is_refused(tmp_path: Path) -> None:
    crop = np.zeros((8, 8, 4), dtype=np.uint8)
    sample = _sample("FX-SYNREID-001", use="DEV", crop=crop)
    _refused([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path), "non-empty HxWx3")


def test_float_crop_is_refused(tmp_path: Path) -> None:
    crop = np.zeros((8, 8, 3), dtype=np.float32)
    sample = _sample("FX-SYNREID-001", use="DEV", crop=crop)
    _refused([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path), "non-empty HxWx3")


def test_refused_batch_does_not_change_the_callers_crops(tmp_path: Path) -> None:
    """A later refusal must not swap crops on samples that already passed."""
    first_crop = _benign_crop(4)
    second_crop = _benign_crop(6)
    first_bytes = first_crop.tobytes()
    second_bytes = second_crop.tobytes()
    first = _sample("FX-SYNREID-001", use="DEV", crop=first_crop)
    second = _sample("FX-SYNREID-002", use="HOLD", crop=second_crop)
    _refused(
        [first, second],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "refused in EVALUATION",
    )
    assert first.crop is first_crop
    assert second.crop is second_crop
    assert first_crop.tobytes() == first_bytes
    assert second_crop.tobytes() == second_bytes


def test_training_refuses_every_sample_before_the_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tag outside TRAIN/HOLD/DEV is still refused, before the id check."""
    monkeypatch.setattr(data_module, "_TRAINING_ID_PATTERNS", (data_module._SYNREID_ID,))
    monkeypatch.setattr(
        data_module,
        "_RECOGNISED_USE_TAGS",
        data_module._RECOGNISED_USE_TAGS | {"FUTURE"},
    )
    sample = _sample("FX-SYNREID-001", use="FUTURE", crop=_benign_crop(11))
    message = _refused(
        [sample],
        LoaderContext.TRAINING,
        _write_manifest(tmp_path),
        "refused in TRAINING",
    )
    assert "FUTURE" in message


def test_smoke_bytes_retagged_as_dev_are_refused(
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "fixture_id": "FX-SYNREID-001",
                "use": "DEV",
                "sha256": entries[0]["sha256"],
                "phash": entries[0]["phash"],
            }
        ],
    )
    sample = _sample("FX-SYNREID-001", use="DEV", path=clip)
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        manifest,
        "identified by SHA-256 or pHash",
    )


def test_manifest_row_id_must_match_the_sample(tmp_path: Path) -> None:
    payload = b"plain-bytes-for-row-mismatch"
    source = tmp_path / "row.txt"
    source.write_bytes(payload)
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "fixture_id": "FX-SYNREID-002",
                "use": "DEV",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    )
    sample = _sample("FX-SYNREID-001", use="DEV", path=source)
    _refused([sample], LoaderContext.EVALUATION, manifest, "does not match")


def test_manifest_row_use_must_match_the_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", (_BATCH_PATTERN,))
    payload = b"plain-bytes-for-use-mismatch"
    source = tmp_path / "use.txt"
    source.write_bytes(payload)
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "fixture_id": "FX-BATCH-007",
                "use": "HOLD",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    )
    sample = _sample("FX-BATCH-007", use="TRAIN", path=source)
    _refused([sample], LoaderContext.EVALUATION, manifest, "does not match")


def test_repo_row_must_match_the_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"repo-row-bytes"
    source = tmp_path / "repo-row.txt"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    caller = _write_manifest(
        tmp_path,
        [{"fixture_id": "FX-SYNREID-001", "use": "DEV", "sha256": digest}],
    )
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    repo = _write_manifest(
        pinned,
        [{"fixture_id": "FX-SYNREID-009", "use": "DEV", "sha256": digest}],
    )
    monkeypatch.setattr(data_module, "_repo_manifest_path", lambda: repo)
    sample = _sample("FX-SYNREID-001", use="DEV", path=source)
    _refused([sample], LoaderContext.EVALUATION, caller, "does not match")


def test_missing_path_is_refused(tmp_path: Path) -> None:
    missing = tmp_path / "missing.bin"
    manifest = _write_manifest(
        tmp_path,
        [{"fixture_id": "FX-SYNREID-001", "use": "DEV", "sha256": "a" * 64}],
    )
    sample = _sample("FX-SYNREID-001", use="DEV", path=missing)
    _refused([sample], LoaderContext.EVALUATION, manifest, "FAIL-CLOSED")


def test_manifest_is_loaded_once_and_each_path_is_hashed_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[Path] = []
    hashes: list[Path] = []
    real_load = data_module.load_manifest_fixtures
    real_hash = data_module.perceptual_hashes

    def spy_load(path: Path) -> object:
        loads.append(Path(path))
        return real_load(path)

    def spy_hash(path: Path) -> list[str]:
        hashes.append(Path(path))
        return real_hash(path)

    monkeypatch.setattr(data_module, "load_manifest_fixtures", spy_load)
    monkeypatch.setattr(data_module, "perceptual_hashes", spy_hash)
    entries = []
    samples = []
    for index, name in enumerate(("one.txt", "two.txt"), start=1):
        payload = f"path-sample-{index}".encode()
        source = tmp_path / name
        source.write_bytes(payload)
        entries.append(
            {
                "fixture_id": f"FX-SYNREID-{index:03d}",
                "use": "DEV",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        samples.append(_sample(f"FX-SYNREID-{index:03d}", use="DEV", path=source))
    samples.append(_sample("FX-SYNREID-003", use="DEV", crop=_benign_crop(33)))
    manifest = _write_manifest(tmp_path, entries)
    loaded = load_reid_samples(samples, LoaderContext.EVALUATION, manifest)
    assert len(loaded) == 3
    assert len(loads) == 2
    assert len(hashes) == 2


def test_repo_manifest_path_is_the_checkout_manifest() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert data_module._repo_manifest_path() == repo_root / "licenses" / "manifest.json"
