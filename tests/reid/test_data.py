"""The ReID loader refuses smoke and code-test data and fails closed.

Clips are generated into pytest's temp directory. Nothing is read from
``smoke/`` and nothing is written into the repo tree.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import fields, replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import trace_search.reid as reid
import trace_search.reid.data as data_module
import trace_search.reid.synthetic as synthetic_module
from tests.media_helpers import write_mp4v_variant
from tests.reid.conftest import rgb_frame
from trace_search.fixtures.content import sha256_file
from trace_search.fixtures.guards import FixtureSeparationError, FixtureUseTag, LoaderContext
from trace_search.reid.data import SyntheticProvenance

make_synthetic_reid_set = synthetic_module.make_synthetic_reid_set

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
        "provenance",
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
    """Training still refuses. Evaluation refuses a crop with no provenance.

    The old assertion returned this crop once a test-only id pattern
    matched. A crop is now accepted only when it matches the synthetic
    generator, so this hand-built crop is refused before it is returned.
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
    _refused([sample], context, manifest, "no synthetic provenance")


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
    source = make_synthetic_reid_set(seed=0)[0]
    fixture_id = "FX-SYNREID-\uff10\uff10\uff11"
    sample = replace(source, fixture_id=fixture_id)
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


def test_loaded_crop_base_cannot_be_made_writable(tmp_path: Path) -> None:
    """``crop.base.setflags(write=True)`` fails, and a write through ``base`` fails."""
    sample = make_synthetic_reid_set(seed=0)[0]
    loaded = load_reid_samples([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path))
    sealed = loaded[0].crop
    assert sealed is not None
    base = sealed.base
    assert isinstance(base, np.ndarray)
    with pytest.raises(ValueError, match="WRITEABLE"):
        base.setflags(write=True)
    with pytest.raises(ValueError):
        base[0] = 0


def test_loaded_crop_is_a_read_only_copy(tmp_path: Path) -> None:
    sample = make_synthetic_reid_set(seed=0)[0]
    crop = sample.crop
    assert crop is not None
    original = crop.copy()
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


def test_smoke_centre_50_percent_crop_is_refused(
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    """The old known gap is closed. A 50% centre crop is refused.

    ``load_reid_samples`` raises before it returns the sample.
    """
    _refuse_smoke_crop(tmp_path, smoke_clips, _half_centre)


def test_smoke_centre_80_percent_crop_is_refused(
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    """An 80% centre crop of a smoke frame is refused before return."""
    _refuse_smoke_crop(tmp_path, smoke_clips, _eighty_centre)


def test_smoke_128x64_box_is_refused(
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    """A 128 by 64 centre box is refused in both orientations."""
    _refuse_smoke_crop(tmp_path, smoke_clips, _centre_128x64)
    _refuse_smoke_crop(tmp_path, smoke_clips, _portrait_128x64)


def test_off_centre_128x64_smoke_box_is_refused(
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    """A 128 by 64 box that is not the centre window is still refused."""
    _refuse_smoke_crop(tmp_path, smoke_clips, _off_centre_128x64)


def _refuse_smoke_crop(
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
    cut: Callable[[np.ndarray], np.ndarray],
) -> None:
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    frame = rgb_frame(clip)
    crop = cut(frame)
    sample = _sample("FX-SYNREID-001", use="DEV", crop=crop)
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "no synthetic provenance",
    )


def _half_centre(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    top = height // 4
    left = width // 4
    return np.ascontiguousarray(frame[top : top + height // 2, left : left + width // 2])


def _eighty_centre(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    crop_h = int(height * 0.8)
    crop_w = int(width * 0.8)
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return np.ascontiguousarray(frame[top : top + crop_h, left : left + crop_w])


def _centre_128x64(frame: np.ndarray) -> np.ndarray:
    """Landscape 128 by 64 box (height 64, width 128) from the centre."""
    return _centre_box(frame, 64, 128)


def _portrait_128x64(frame: np.ndarray) -> np.ndarray:
    """Portrait 128 by 64 box (height 128, width 64) from the centre."""
    return _centre_box(frame, 128, 64)


def _centre_box(frame: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    top = (frame.shape[0] - crop_h) // 2
    left = (frame.shape[1] - crop_w) // 2
    return np.ascontiguousarray(frame[top : top + crop_h, left : left + crop_w])


def _off_centre_128x64(frame: np.ndarray) -> np.ndarray:
    """A 64x128 window whose mean is far from the centre window."""
    centre = _centre_box(frame, 64, 128)
    centre_mean = float(centre.mean())
    best: np.ndarray | None = None
    best_gap = 0.0
    for top in range(0, frame.shape[0] - 64, 48):
        for left in range(0, frame.shape[1] - 128, 48):
            box = np.ascontiguousarray(frame[top : top + 64, left : left + 128])
            gap = abs(float(box.mean()) - centre_mean)
            if gap > best_gap:
                best_gap = gap
                best = box
    assert best is not None
    assert best_gap > 20
    return best


def _clip_for(
    directory: Path,
    entries: Sequence[dict[str, object]],
    fixture_id: str,
) -> Path:
    matches = [entry for entry in entries if entry["fixture_id"] == fixture_id]
    assert len(matches) == 1
    return _entry_clip(directory, matches[0])


def _frame_count(clip: Path) -> int:
    cap = cv2.VideoCapture(str(clip))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _sample_frame_indices(total: int) -> set[int]:
    """The 10 frames the smoke pHash list is built from."""
    return {int(index) for index in np.linspace(0, total - 1, 10)}


def _rgb_frames(clip: Path, indices: Sequence[int]) -> dict[int, np.ndarray]:
    """Read RGB frames by walking the clip. The indices are outside the repo."""
    wanted = set(indices)
    last = max(wanted)
    found: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(clip))
    try:
        for index in range(last + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise AssertionError(f"cannot read frame {index} of {clip}")
            if index in wanted:
                found[index] = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return found
    finally:
        cap.release()


def _figure_centred_box(frame: np.ndarray, crop_h: int = 128, crop_w: int = 64) -> np.ndarray:
    """A 128 by 64 box centred on the tallest person-like blob in ``frame``."""
    background = frame[0, 0].astype(np.int16)
    diff = np.abs(frame.astype(np.int16) - background).max(axis=2)
    mask = np.ascontiguousarray((diff > 25).astype(np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    centre: tuple[float, float] | None = None
    best_area = 0
    for label in range(1, int(count)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if height <= width or area < 800 or area > 25000:
            continue
        if area > best_area:
            best_area = area
            centre = (float(centroids[label][1]), float(centroids[label][0]))
    assert centre is not None
    top = int(round(centre[0] - crop_h / 2))
    left = int(round(centre[1] - crop_w / 2))
    top = min(max(top, 0), frame.shape[0] - crop_h)
    left = min(max(left, 0), frame.shape[1] - crop_w)
    box = np.ascontiguousarray(frame[top : top + crop_h, left : left + crop_w])
    assert box.shape == (crop_h, crop_w, 3)
    overlap = np.abs(box.astype(np.int16) - background).max(axis=2)
    assert int((overlap > 25).sum()) > 0
    return box


def _with_provenance(sample: ReidSample, **changes: object) -> ReidSample:
    provenance = sample.provenance
    assert isinstance(provenance, SyntheticProvenance)
    return replace(sample, provenance=replace(provenance, **changes))  # type: ignore[arg-type]


def test_non_sample_figure_boxes_are_refused(
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    """Figure-centred boxes on FX-SMOKE-SYN-003 frames 156-174 are refused.

    Those frames are outside the 10 pHash samples. The boxes have no
    synthetic provenance.
    """
    directory, entries = smoke_clips
    clip = _clip_for(directory, entries, "FX-SMOKE-SYN-003")
    total = _frame_count(clip)
    sampled = _sample_frame_indices(total)
    indices = (156, 162, 168, 174)
    frames = _rgb_frames(clip, indices)
    for index in indices:
        assert index not in sampled
        crop = _figure_centred_box(frames[index])
        sample = _sample("FX-SYNREID-001", use="DEV", crop=crop)
        _refused(
            [sample],
            LoaderContext.EVALUATION,
            _write_manifest(tmp_path),
            "no synthetic provenance",
        )


def test_smoke_frame_213_box_is_refused(
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    """FX-SMOKE-SYN-005 frame 213, rows 100-356 and cols 300-428, is refused.

    The frame is outside the 10 pHash samples. The box is 256 by 128.
    """
    directory, entries = smoke_clips
    clip = _clip_for(directory, entries, "FX-SMOKE-SYN-005")
    index = 213
    assert index not in _sample_frame_indices(_frame_count(clip))
    frame = _rgb_frames(clip, (index,))[index]
    crop = np.ascontiguousarray(frame[100:356, 300:428])
    assert crop.shape == (256, 128, 3)
    sample = _sample("FX-SYNREID-001", use="DEV", crop=crop)
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "no synthetic provenance",
    )


def test_smoke_box_with_copied_provenance_is_refused(
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    """A smoke box that carries a real digest is still a digest mismatch."""
    directory, entries = smoke_clips
    clip = _clip_for(directory, entries, "FX-SMOKE-SYN-005")
    frame = _rgb_frames(clip, (213,))[213]
    crop = np.ascontiguousarray(frame[100:356, 300:428])
    genuine = make_synthetic_reid_set(seed=0)[0]
    sample = replace(genuine, crop=crop)
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "digest mismatch",
    )


def test_genuine_synthetic_crop_is_accepted(tmp_path: Path) -> None:
    """A generator crop with its own provenance is accepted."""
    sample = make_synthetic_reid_set(seed=0)[0]
    crop = sample.crop
    assert crop is not None
    provenance = sample.provenance
    assert provenance is not None
    assert provenance.digest == hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest()
    loaded = load_reid_samples([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path))
    assert loaded == [sample]
    sealed = loaded[0].crop
    assert sealed is not None
    assert np.array_equal(sealed, crop)


def test_large_synthetic_crop_is_accepted(tmp_path: Path) -> None:
    """A generator crop of 128 by 64 is accepted when its provenance matches."""
    sample = make_synthetic_reid_set(
        n_groups=1,
        per_group=1,
        n_cameras=1,
        size=(128, 64),
        seed=1,
    )[0]
    crop = sample.crop
    assert crop is not None
    assert crop.shape == (128, 64, 3)
    loaded = load_reid_samples([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path))
    assert loaded == [sample]
    sealed = loaded[0].crop
    assert sealed is not None
    assert np.array_equal(sealed, crop)


def test_tampered_digest_is_refused(tmp_path: Path) -> None:
    sample = make_synthetic_reid_set(seed=0)[0]
    provenance = sample.provenance
    assert provenance is not None
    assert provenance.digest != "0" * 64
    sample = _with_provenance(sample, digest="0" * 64)
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "tampered digest",
    )


def test_changed_pixel_is_refused(tmp_path: Path) -> None:
    """One changed pixel keeps the stored digest and is a digest mismatch."""
    sample = make_synthetic_reid_set(seed=0)[0]
    crop = sample.crop
    assert crop is not None
    changed = crop.copy()
    changed[0, 0, 0] ^= np.uint8(1)
    sample = replace(sample, crop=changed)
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "digest mismatch",
    )


def test_reshaped_crop_is_refused(tmp_path: Path) -> None:
    sample = make_synthetic_reid_set(seed=0)[0]
    crop = sample.crop
    assert crop is not None
    sample = replace(sample, crop=np.ascontiguousarray(crop[:-1]))
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "digest mismatch",
    )


def test_crop_with_no_provenance_is_refused(tmp_path: Path) -> None:
    sample = replace(make_synthetic_reid_set(seed=0)[0], provenance=None)
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "no synthetic provenance",
    )


@pytest.mark.parametrize("seed", ("nope", True, 1.5))
def test_unknown_seed_is_refused(tmp_path: Path, seed: object) -> None:
    sample = _with_provenance(make_synthetic_reid_set(seed=0)[0], seed=seed)
    _refused([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path), "unknown seed")


def test_invalid_seed_is_refused(tmp_path: Path) -> None:
    sample = _with_provenance(make_synthetic_reid_set(seed=0)[0], seed=-1)
    _refused([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path), "invalid seed")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("height", 0),
        ("width", True),
        ("n_groups", "8"),
        ("per_group", 1.5),
        ("n_cameras", -1),
    ],
)
def test_bad_synthetic_config_is_refused(tmp_path: Path, field: str, value: object) -> None:
    sample = _with_provenance(make_synthetic_reid_set(seed=0)[0], **{field: value})
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        f"bad synthetic config {field}",
    )


def test_wrong_provenance_type_is_refused(tmp_path: Path) -> None:
    sample = replace(
        make_synthetic_reid_set(seed=0)[0],
        provenance="not-a-provenance",  # type: ignore[arg-type]
    )
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "wrong provenance type",
    )


def test_synthetic_set_that_will_not_render_is_refused(tmp_path: Path) -> None:
    """A positive config the generator rejects is a refusal, not an empty set."""
    sample = _with_provenance(make_synthetic_reid_set(seed=0)[0], n_groups=1000, per_group=1)
    _refused(
        [sample],
        LoaderContext.EVALUATION,
        _write_manifest(tmp_path),
        "synthetic set will not render",
    )


def test_unknown_synthetic_id_is_refused(tmp_path: Path) -> None:
    sample = replace(make_synthetic_reid_set(seed=0)[0], fixture_id="FX-SYNREID-999")
    _refused([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path), "unknown id")


def test_regenerated_sample_without_a_crop_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A regenerated id whose crop is missing is an unknown id."""
    sample = make_synthetic_reid_set(seed=0)[0]
    real = synthetic_module.make_synthetic_reid_set

    def spy(*args: object, **kwargs: object) -> list[ReidSample]:
        found = real(*args, **kwargs)  # type: ignore[arg-type]
        return [replace(item, crop=None) for item in found]

    monkeypatch.setattr(synthetic_module, "make_synthetic_reid_set", spy)
    _refused([sample], LoaderContext.EVALUATION, _write_manifest(tmp_path), "unknown id")


def test_one_seed_is_regenerated_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = make_synthetic_reid_set(seed=0)[:2]
    calls: list[int] = []
    real = synthetic_module.make_synthetic_reid_set

    def spy(*args: object, **kwargs: object) -> list[ReidSample]:
        calls.append(int(kwargs["seed"]))  # type: ignore[arg-type]
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(synthetic_module, "make_synthetic_reid_set", spy)
    loaded = load_reid_samples(samples, LoaderContext.EVALUATION, _write_manifest(tmp_path))
    assert loaded == list(samples)
    assert calls == [0]


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
    first = make_synthetic_reid_set(seed=0)[0]
    first_crop = first.crop
    assert first_crop is not None
    second_crop = _benign_crop(6)
    first_bytes = first_crop.tobytes()
    second_bytes = second_crop.tobytes()
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
    samples.append(make_synthetic_reid_set(seed=0)[2])
    manifest = _write_manifest(tmp_path, entries)
    loaded = load_reid_samples(samples, LoaderContext.EVALUATION, manifest)
    assert len(loaded) == 3
    assert len(loads) == 2
    assert len(hashes) == 2


def test_repo_manifest_path_is_the_checkout_manifest() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert data_module._repo_manifest_path() == repo_root / "licenses" / "manifest.json"
