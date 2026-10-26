"""Rank-1 / mAP and the evaluation command.

Vectors and crops are built in the test with numpy. Nothing is read from
``smoke/`` and nothing is written into the repo tree. The JSON evidence
flag is false. These numbers are not an AC-06 result.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import subprocess
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import trace_search.reid.data as data_module
import trace_search.reid.eval as eval_module
from tests.reid.conftest import rgb_frame
from trace_search.fixtures.guards import FixtureSeparationError, LoaderContext
from trace_search.reid.osnet_lite import OsnetLiteEmbedder
from trace_search.reid.placeholder import UntrainedPlaceholderEmbedder

ReidSample = data_module.ReidSample
load_reid_samples = data_module.load_reid_samples
EvalItem = eval_module.EvalItem
rank1_map = eval_module.rank1_map

_NOTE = "Placeholder or synthetic embeddings. Not an AC-06 result."
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / "licenses" / "manifest.json"
_BATCH_PATTERN = re.compile(r"FX-BATCH-\d{3}")
_ANY_FX_PATTERN = re.compile(r"FX-[A-Z0-9-]+")
_JSON_KEYS = (
    "rank1",
    "mAP",
    "cmc_top5",
    "num_query",
    "num_valid_query",
    "num_gallery",
    "evidence",
    "note",
)
_OSNET_LITE_LABEL = "OSNet-style, chained streams, random init"
_PLACEHOLDER_LABEL = "untrained-placeholder-v1"
_CLI_JSON_KEYS = (*_JSON_KEYS, "embedder", "report")
_CLI = [
    "--embedder",
    "placeholder",
    "--synthetic-seed",
    "0",
    "--json",
]


def _axes() -> tuple[tuple[float, ...], ...]:
    """Four orthogonal directions, built in the test."""
    eye = np.eye(4, dtype=np.float64)
    return tuple(tuple(float(value) for value in eye[index]) for index in range(4))


def _item(camera_id: str, track_id: int, label: int, vector: tuple[float, ...]) -> EvalItem:
    return EvalItem(camera_id=camera_id, track_id=track_id, label=label, vector=vector)


def _near(actual: float, expected: Fraction) -> None:
    assert abs(actual - float(expected)) < 1e-12


def _crop(seed: int, shape: tuple[int, int, int] = (12, 8, 3)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.integers(1, 255, size=shape, dtype=np.uint8)
    return np.asarray(values, dtype=np.uint8)


def _reid(
    fixture_id: str,
    *,
    crop: np.ndarray | None,
    path: Path | None = None,
    camera_id: str = "cam-1",
    track_id: int = 1,
    label: int = 1,
    use: str = "TRAIN",
) -> ReidSample:
    return ReidSample(
        fixture_id=fixture_id,
        use=use,
        camera_id=camera_id,
        track_id=track_id,
        label=label,
        crop=crop,
        path=path,
    )


def _hand_split(seed: int) -> tuple[tuple[ReidSample, ...], tuple[ReidSample, ...]]:
    """In-test query and gallery. Ids match the test-only allowlist pattern."""
    query = (
        _reid("FX-BATCH-001", crop=_crop(10 + seed), camera_id="q-a", track_id=1, label=1),
        _reid("FX-BATCH-002", crop=_crop(20 + seed), camera_id="q-b", track_id=2, label=2),
    )
    gallery = (
        _reid("FX-BATCH-003", crop=_crop(30 + seed), camera_id="g-a", track_id=3, label=1),
        _reid("FX-BATCH-004", crop=_crop(40 + seed), camera_id="g-b", track_id=4, label=2),
        _reid("FX-BATCH-005", crop=_crop(50 + seed), camera_id="g-c", track_id=5, label=3),
    )
    return query, gallery


def _write_manifest(directory: Path, entries: Sequence[object] | None = None) -> Path:
    path = directory / "manifest.json"
    payload = {
        "manifest_version": "1.0",
        "entries": [] if entries is None else list(entries),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _count_embed(monkeypatch: pytest.MonkeyPatch, calls: list[int]) -> None:
    real = UntrainedPlaceholderEmbedder.embed

    def wrapped(
        self: UntrainedPlaceholderEmbedder,
        crops: Sequence[np.ndarray],
    ) -> np.ndarray:
        calls.append(len(tuple(crops)))
        return real(self, crops)

    monkeypatch.setattr(UntrainedPlaceholderEmbedder, "embed", wrapped)


def _count_named_embeds(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Record which embedder class would run. A missed class stays at zero."""

    def wrap(cls: type, label: str) -> None:
        real = cls.embed

        def wrapped(self: object, crops: Sequence[np.ndarray]) -> np.ndarray:
            calls.append(label)
            found: np.ndarray = real(self, crops)
            return found

        monkeypatch.setattr(cls, "embed", wrapped)

    wrap(UntrainedPlaceholderEmbedder, "placeholder")
    wrap(OsnetLiteEmbedder, "osnet-lite")


def _hand_built() -> tuple[list[EvalItem], list[EvalItem]]:
    """3 queries and 5 gallery items. Exactly 2 queries are correct at rank 1.

    Directions are rows of ``np.eye(4)``. Cosine distance is 0 on a shared
    direction and 1 on an orthogonal one. Stable ordering then puts:

    - Q0 (label 1, e1): gallery order g-0 (label 1), g-1 (label 1), then
      the orthogonal items. Relevant ranks 1 and 2.
      AP = (1/1 + 2/2) / 2 = 1/1. Top-1 is correct.
    - Q1 (label 2, e2): g-1 (label 1) then g-2 (label 2), both distance 0.
      Relevant rank 2. AP = (1/2) / 1 = 1/2. Top-1 is wrong.
    - Q2 (label 3, e3): g-3 is the only distance-0 item and it matches.
      Relevant rank 1. AP = (1/1) / 1 = 1/1. Top-1 is correct.

    Rank-1 = 2/3. mAP = (1/1 + 1/2 + 1/1) / 3 = 5/6.
    """
    e1, e2, e3, e4 = _axes()
    query = [
        _item("q-0", 10, 1, e1),
        _item("q-1", 11, 2, e2),
        _item("q-2", 12, 3, e3),
    ]
    gallery = [
        _item("g-0", 1, 1, e1),
        _item("g-1", 2, 1, e2),
        _item("g-2", 3, 2, e2),
        _item("g-3", 4, 3, e3),
        _item("g-4", 5, 4, e4),
    ]
    return query, gallery


def test_hand_built_rank1_is_two_thirds_and_aps_are_exact_fractions() -> None:
    query, gallery = _hand_built()
    per_query = (Fraction(1, 1), Fraction(1, 2), Fraction(1, 1))
    top1 = (1.0, 0.0, 1.0)
    for item, fraction, hit in zip(query, per_query, top1, strict=True):
        alone = rank1_map([item], gallery)
        assert alone.num_valid_query == 1
        assert alone.num_invalid_query == 0
        assert alone.rank1 == hit
        _near(alone.mean_ap, fraction)

    combined = rank1_map(query, gallery)
    assert combined.num_query == 3
    assert combined.num_valid_query == 3
    assert combined.num_invalid_query == 0
    assert combined.num_gallery == 5
    _near(combined.rank1, Fraction(2, 3))
    _near(combined.mean_ap, Fraction(5, 6))
    _near(combined.cmc_top5[0], Fraction(2, 3))
    assert combined.cmc_top5[1:] == (1.0, 1.0, 1.0, 1.0)

    payload = combined.as_json()
    assert tuple(payload) == _JSON_KEYS
    assert payload["evidence"] is False
    assert payload["note"] == _NOTE
    assert "num_invalid_query" not in payload
    assert isinstance(payload["cmc_top5"], list)
    assert len(payload["cmc_top5"]) == 5
    _near(float(payload["rank1"]), Fraction(2, 3))  # type: ignore[arg-type]
    _near(float(payload["mAP"]), Fraction(5, 6))  # type: ignore[arg-type]


def test_same_camera_only_match_makes_the_query_invalid() -> None:
    """The true match is only on the query camera, so the query is junk-only."""
    direction = _axes()[0]
    query = [_item("cam-a", 1, 7, direction)]
    gallery = [
        _item("cam-a", 2, 7, direction),
        _item("cam-b", 3, 8, _axes()[1]),
    ]
    result = rank1_map(query, gallery)
    assert result.num_query == 1
    assert result.num_valid_query == 0
    assert result.num_invalid_query == 1
    assert result.num_gallery == 2
    assert result.rank1 == 0.0
    assert result.mean_ap == 0.0
    assert result.cmc_top5 == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_same_camera_junk_is_removed_before_ranking() -> None:
    """A closer same-camera copy does not count. The next item is a distractor.

    After the junk row is removed, the distractor (distance 0) outranks the
    cross-camera match (distance 1). AP = 1/2. Keeping the junk row in the
    list as a non-match would put the true match at rank 3 and AP = 1/3.
    Counting the junk row as a match would make top-1 correct.
    """
    e1, e2 = _axes()[0], _axes()[1]
    query = [_item("cam-q", 1, 1, e1)]
    gallery = [
        _item("cam-q", 2, 1, e1),
        _item("cam-a", 3, 2, e1),
        _item("cam-b", 4, 1, e2),
    ]
    result = rank1_map(query, gallery)
    assert result.num_valid_query == 1
    assert result.num_invalid_query == 0
    assert result.rank1 == 0.0
    _near(result.mean_ap, Fraction(1, 2))


def test_same_camera_different_label_stays_in_the_ranking() -> None:
    e1, e2 = _axes()[0], _axes()[1]
    query = [_item("cam-q", 1, 1, e2)]
    gallery = [
        _item("cam-q", 2, 2, e2),
        _item("cam-g", 3, 1, e1),
    ]
    result = rank1_map(query, gallery)
    assert result.num_valid_query == 1
    assert result.rank1 == 0.0
    _near(result.mean_ap, Fraction(1, 2))


def test_query_with_no_gallery_label_is_left_out_of_cmc_and_map() -> None:
    match = _axes()[0]
    other = _axes()[1]
    query = [
        _item("q-good", 1, 1, match),
        _item("q-missing", 2, 99, match),
    ]
    gallery = [
        _item("g-match", 3, 1, match),
        _item("g-other", 4, 2, other),
    ]
    result = rank1_map(query, gallery)
    assert result.num_query == 2
    assert result.num_valid_query == 1
    assert result.num_invalid_query == 1
    assert result.num_gallery == 2
    assert result.rank1 == 1.0
    assert result.mean_ap == 1.0
    assert result.cmc_top5 == (1.0, 1.0, 1.0, 1.0, 1.0)


def test_perfect_embedding_scores_one() -> None:
    e1, e2, e3 = _axes()[0], _axes()[1], _axes()[2]
    query = [
        _item("q-1", 1, 1, e1),
        _item("q-2", 2, 2, e2),
    ]
    gallery = [
        _item("g-1", 3, 1, e1),
        _item("g-2", 4, 2, e2),
        _item("g-3", 5, 3, e3),
    ]
    result = rank1_map(query, gallery)
    assert result.rank1 == 1.0
    assert result.mean_ap == 1.0
    assert result.cmc_top5 == (1.0, 1.0, 1.0, 1.0, 1.0)
    assert result.num_valid_query == 2
    assert result.num_invalid_query == 0
    payload = result.as_json()
    assert payload["evidence"] is False
    assert payload["note"] == _NOTE


def test_ties_keep_gallery_order() -> None:
    direction = _axes()[0]
    query = [_item("q", 1, 1, direction)]
    gallery = [
        _item("g-wrong", 2, 2, direction),
        _item("g-right", 3, 1, direction),
        _item("g-far", 4, 3, _axes()[1]),
    ]
    result = rank1_map(query, gallery)
    assert result.rank1 == 0.0
    _near(result.mean_ap, Fraction(1, 2))


def test_empty_query_and_empty_gallery_are_defined() -> None:
    empty = rank1_map([], [])
    assert empty.num_query == 0
    assert empty.num_valid_query == 0
    assert empty.num_invalid_query == 0
    assert empty.num_gallery == 0
    assert empty.rank1 == 0.0
    assert empty.mean_ap == 0.0
    assert empty.as_json()["evidence"] is False

    direction = _axes()[0]
    no_gallery = rank1_map([_item("q", 1, 1, direction)], [])
    assert no_gallery.num_invalid_query == 1
    assert no_gallery.num_gallery == 0
    assert no_gallery.rank1 == 0.0
    assert no_gallery.mean_ap == 0.0


def test_unknown_distance_is_refused() -> None:
    with pytest.raises(ValueError, match="cosine"):
        rank1_map([], [], distance="euclidean")


def test_default_distance_is_cosine() -> None:
    query, gallery = _hand_built()
    assert rank1_map(query, gallery).rank1 == rank1_map(query, gallery, distance="cosine").rank1


@pytest.mark.parametrize(
    ("camera_id", "vector", "message"),
    [
        ("", (1.0, 0.0), "camera_id"),
        ("cam", (), "non-empty"),
        ("cam", (0.0, 0.0), "non-zero"),
        ("cam", (math.nan, 1.0), "finite"),
        ("cam", (math.inf, 0.0), "finite"),
        ("cam", "12", "sequence"),
        ("cam", b"12", "sequence"),
    ],
)
def test_eval_item_rejects_bad_fields(camera_id: str, vector: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EvalItem(camera_id, 1, 1, vector)  # type: ignore[arg-type]


def test_vector_length_mismatch_is_refused() -> None:
    query = [_item("q", 1, 1, (1.0, 0.0))]
    gallery = [_item("g", 2, 1, (1.0, 0.0, 0.0))]
    with pytest.raises(ValueError, match="same length"):
        rank1_map(query, gallery)


def test_gallery_vectors_must_share_a_length() -> None:
    query = [_item("q", 1, 1, (1.0, 0.0))]
    gallery = [
        _item("g", 2, 1, (1.0, 0.0)),
        _item("h", 3, 1, (1.0, 0.0, 0.0)),
    ]
    with pytest.raises(ValueError, match="gallery vectors"):
        rank1_map(query, gallery)


def test_production_allowlist_stays_empty() -> None:
    """Training still has no id. Evaluation has only the synthetic prefix."""
    assert data_module._TRAINING_ID_PATTERNS == ()
    assert data_module._EVALUATION_ID_PATTERNS == (data_module._SYNREID_ID,)


@pytest.mark.parametrize("context", (LoaderContext.EVALUATION, LoaderContext.TRAINING))
def test_sample_with_neither_crop_nor_path_is_refused(
    context: LoaderContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", (_BATCH_PATTERN,))
    monkeypatch.setattr(data_module, "_TRAINING_ID_PATTERNS", (_BATCH_PATTERN,))
    sample = _reid("FX-BATCH-001", crop=None, path=None)
    manifest = _write_manifest(tmp_path)
    if context is LoaderContext.TRAINING:
        with pytest.raises(FixtureSeparationError, match="refused in TRAINING") as caught:
            load_reid_samples([sample], context, manifest)
        assert "TRAIN" in str(caught.value)
        return
    with pytest.raises(FixtureSeparationError, match="neither a crop nor a path") as caught:
        load_reid_samples([sample], context, manifest)
    assert "FX-BATCH-001" in str(caught.value)


@pytest.mark.parametrize("context", (LoaderContext.EVALUATION, LoaderContext.TRAINING))
def test_crop_without_path_is_accepted(
    context: LoaderContext,
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    """A crop-only sample cut from a smoke frame is refused by the crop hash."""
    directory, entries = smoke_clips
    fixture_id = str(entries[0]["fixture_id"])
    matches = sorted(directory.glob(f"{fixture_id}_*.mp4"))
    assert len(matches) == 1
    sample = _reid("FX-SYNREID-001", use="DEV", crop=rgb_frame(matches[0]))
    with pytest.raises(FixtureSeparationError, match="identified by pHash"):
        load_reid_samples([sample], context, _write_manifest(tmp_path))


def test_path_without_crop_is_accepted_when_the_manifest_row_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", (_BATCH_PATTERN,))
    payload = b"not-media-eval-sample"
    source = tmp_path / "crop-source.txt"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "fixture_id": "FX-BATCH-007",
                "use": "TRAIN",
                "sha256": digest,
            }
        ],
    )
    sample = _reid("FX-BATCH-007", crop=None, path=source)
    loaded = load_reid_samples([sample], LoaderContext.EVALUATION, manifest)
    assert loaded[0] is sample


def test_cli_prints_json_with_evidence_false(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    query, gallery = _hand_split(1)
    seen: list[tuple[np.ndarray, ...]] = []
    real = UntrainedPlaceholderEmbedder.embed

    def wrapped(
        self: UntrainedPlaceholderEmbedder,
        crops: Sequence[np.ndarray],
    ) -> np.ndarray:
        seen.append(tuple(crops))
        return real(self, crops)

    monkeypatch.setattr(UntrainedPlaceholderEmbedder, "embed", wrapped)
    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", (_BATCH_PATTERN,))
    monkeypatch.setattr(eval_module, "synthetic_evaluation_split", lambda _seed: (query, gallery))
    code = eval_module.main(["--embedder", "placeholder", "--synthetic-seed", "1", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert tuple(payload) == _CLI_JSON_KEYS
    assert payload["evidence"] is False
    assert payload["embedder"] == _PLACEHOLDER_LABEL
    assert str(payload["report"]).splitlines()[0] == f"embedder: {_PLACEHOLDER_LABEL}"
    assert "evidence: false" in str(payload["report"])
    assert payload["note"] == _NOTE
    assert payload["num_query"] == 2
    assert payload["num_gallery"] == 3
    assert payload["num_valid_query"] == 2
    assert isinstance(payload["rank1"], float)
    assert isinstance(payload["mAP"], float)
    assert len(payload["cmc_top5"]) == 5
    assert seen[0][0] is query[0].crop
    assert seen[0][1] is query[1].crop
    assert seen[1][0] is gallery[0].crop
    assert seen[1][1] is gallery[1].crop
    assert seen[1][2] is gallery[2].crop


@pytest.mark.parametrize("embedder_name", ("placeholder", "osnet-lite"))
def test_fx_smoke_is_refused_before_any_embedding(
    embedder_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The allowlist pattern would accept this id. The prefix check refuses it.

    Placeholder and osnet-lite both stop in the evaluation loader. Neither
    embedder is called.
    """
    calls: list[str] = []
    _count_named_embeds(monkeypatch, calls)
    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", (_ANY_FX_PATTERN,))
    smoke = _reid("fx-smoke", crop=_crop(1), camera_id="q", track_id=1, label=1)
    gallery = _reid(
        "FX-BATCH-002",
        crop=_crop(2),
        camera_id="g",
        track_id=2,
        label=1,
    )

    def _split(_seed: int) -> tuple[tuple[ReidSample, ...], tuple[ReidSample, ...]]:
        return (smoke,), (gallery,)

    monkeypatch.setattr(eval_module, "synthetic_evaluation_split", _split)
    try:
        code = eval_module.main(["--embedder", embedder_name, "--synthetic-seed", "0", "--json"])
    except SystemExit as exc:
        code = 0
        pytest.fail(f"{embedder_name} was refused before the loader: {exc.code}")
    captured = capsys.readouterr()
    assert code != 0
    assert calls == []
    assert captured.out == ""
    assert "FX-SMOKE" in captured.err
    assert "rank1" not in captured.out
    assert "mAP" not in captured.out


def test_empty_allowlist_prints_no_metric(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[int] = []
    _count_embed(monkeypatch, calls)
    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", ())
    monkeypatch.setattr(eval_module, "synthetic_evaluation_split", lambda _seed: ((), ()))
    code = eval_module.main(_CLI)
    captured = capsys.readouterr()
    assert code != 0
    assert calls == []
    assert captured.out == ""
    assert "allowlist is empty" in captured.err
    assert "No metric was computed." in captured.err
    assert "rank1" not in captured.out
    assert "mAP" not in captured.err


def test_main_reads_sys_argv_when_argv_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_reid_eval.py", "--embedder", "placeholder", "--synthetic-seed", "0", "--json"],
    )
    code = eval_module.main()
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["evidence"] is False
    assert "AC-06" in payload["note"]
    assert "Placeholder" in payload["note"]


def test_filled_source_is_refused_when_the_allowlist_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[int] = []
    _count_embed(monkeypatch, calls)
    query, gallery = _hand_split(2)
    monkeypatch.setattr(eval_module, "synthetic_evaluation_split", lambda _seed: (query, gallery))
    code = eval_module.main(_CLI)
    captured = capsys.readouterr()
    assert code != 0
    assert calls == []
    assert captured.out == ""
    assert "not on the ReID allowlist" in captured.err


def test_empty_split_with_a_pattern_prints_no_metric(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[int] = []
    _count_embed(monkeypatch, calls)
    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", (_BATCH_PATTERN,))
    monkeypatch.setattr(eval_module, "synthetic_evaluation_split", lambda _seed: ((), ()))
    code = eval_module.main(["--embedder", "placeholder", "--synthetic-seed", "4", "--json"])
    captured = capsys.readouterr()
    assert code == 1
    assert calls == []
    assert captured.out == ""
    assert "non-empty" in captured.err


def test_cli_pins_the_repo_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[Path] = []
    contexts: list[LoaderContext] = []
    query, gallery = _hand_split(3)

    def spy(
        samples: Sequence[ReidSample],
        context: LoaderContext,
        manifest_path: Path,
    ) -> list[ReidSample]:
        seen.append(Path(manifest_path))
        contexts.append(context)
        return list(samples)

    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", (_BATCH_PATTERN,))
    monkeypatch.setattr(data_module, "load_reid_samples", spy)
    monkeypatch.setattr(eval_module, "synthetic_evaluation_split", lambda _seed: (query, gallery))
    code = eval_module.main(["--embedder", "placeholder", "--synthetic-seed", "3", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert seen == [_MANIFEST, _MANIFEST]
    assert contexts == [LoaderContext.EVALUATION, LoaderContext.EVALUATION]
    assert _MANIFEST.is_file()
    payload = json.loads(captured.out)
    assert payload["evidence"] is False


def test_parser_has_no_path_arguments() -> None:
    dests = {action.dest for action in eval_module._parser()._actions}
    assert "manifest" not in dests
    assert "data_dir" not in dests
    assert "weights" not in dests
    assert "embedder" in dests
    assert "synthetic_seed" in dests
    assert "emit_json" in dests


@pytest.mark.parametrize(
    "extra",
    [
        ["--data-dir", "data"],
        ["--manifest", "licenses/manifest.json"],
        ["--weights", "model.pth"],
    ],
)
def test_cli_rejects_path_arguments(
    extra: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        eval_module.main([*_CLI, *extra])
    captured = capsys.readouterr()
    assert caught.value.code != 0
    assert captured.out == ""
    assert "rank1" not in captured.err


def test_osnet_lite_cli_exits_zero_with_the_chained_streams_label(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """osnet-lite runs through the evaluation loader and is labelled as chained.

    The label is not the paper network. ``evidence`` stays false. The note
    still says this is not an AC-06 result.
    """
    contexts: list[LoaderContext] = []
    real_load = data_module.load_reid_samples

    def spy_load(
        samples: Sequence[ReidSample],
        context: LoaderContext,
        manifest_path: Path,
    ) -> list[ReidSample]:
        contexts.append(context)
        return real_load(samples, context, manifest_path)

    constructed: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real_init = OsnetLiteEmbedder.__init__

    def spy_init(self: OsnetLiteEmbedder, *args: object, **kwargs: object) -> None:
        constructed.append((args, dict(kwargs)))
        real_init(self, *args, **kwargs)  # type: ignore[arg-type]

    calls: list[str] = []
    monkeypatch.setattr(data_module, "load_reid_samples", spy_load)
    monkeypatch.setattr(OsnetLiteEmbedder, "__init__", spy_init)
    _count_named_embeds(monkeypatch, calls)
    try:
        code = eval_module.main(["--embedder", "osnet-lite", "--synthetic-seed", "0", "--json"])
    except SystemExit as exc:
        code = 1
        pytest.fail(f"osnet-lite was refused before evaluation: {exc.code}")
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert tuple(payload) == _CLI_JSON_KEYS
    assert payload["evidence"] is False
    assert payload["embedder"] == _OSNET_LITE_LABEL
    assert payload["note"] == _NOTE
    report = str(payload["report"])
    assert report.splitlines()[0] == f"embedder: {_OSNET_LITE_LABEL}"
    assert "evidence: false" in report
    assert _NOTE in report
    assert "evidence: true" not in report
    assert contexts == [LoaderContext.EVALUATION, LoaderContext.EVALUATION]
    assert constructed == [((), {"seed": 0})]
    assert calls == ["osnet-lite", "osnet-lite"]
    assert payload["num_valid_query"] > 0


def test_placeholder_default_matches_the_explicit_placeholder(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitting ``--embedder`` stays on the placeholder. Its label is its name."""
    try:
        code_default = eval_module.main(["--synthetic-seed", "0", "--json"])
    except SystemExit as exc:
        code_default = 1
        pytest.fail(f"--embedder still required, default did not apply: {exc.code}")
    default_out = capsys.readouterr()
    code_explicit = eval_module.main(
        ["--embedder", "placeholder", "--synthetic-seed", "0", "--json"]
    )
    explicit_out = capsys.readouterr()
    assert code_default == 0
    assert code_explicit == 0
    assert default_out.err == ""
    assert explicit_out.err == ""
    default_payload = json.loads(default_out.out)
    explicit_payload = json.loads(explicit_out.out)
    assert default_payload == explicit_payload
    assert tuple(default_payload) == _CLI_JSON_KEYS
    assert default_payload["embedder"] == _PLACEHOLDER_LABEL
    assert default_payload["evidence"] is False
    assert default_payload["note"] == _NOTE
    assert str(default_payload["report"]).splitlines()[0] == f"embedder: {_PLACEHOLDER_LABEL}"
    assert _OSNET_LITE_LABEL not in default_out.out


def test_unknown_embedder_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        eval_module.main(["--embedder", "paper-osnet", "--synthetic-seed", "0", "--json"])
    captured = capsys.readouterr()
    assert caught.value.code != 0
    assert captured.out == ""
    assert "rank1" not in captured.out
    assert "rank1" not in captured.err


def test_json_flag_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    code = eval_module.main(["--embedder", "placeholder", "--synthetic-seed", "0"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "--json is required" in captured.err


def test_negative_seed_prints_no_metric(capsys: pytest.CaptureFixture[str]) -> None:
    code = eval_module.main(["--embedder", "placeholder", "--synthetic-seed", "-1", "--json"])
    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""
    assert "non-negative" in captured.err


def test_seed_must_be_a_non_negative_int() -> None:
    eval_module._require_seed(0)
    with pytest.raises(ValueError, match="non-negative"):
        eval_module._require_seed(-1)
    with pytest.raises(ValueError, match="non-negative"):
        eval_module._require_seed(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        eval_module._require_seed(1.5)  # type: ignore[arg-type]


def test_missing_pinned_manifest_prints_no_metric(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-manifest.json"
    monkeypatch.setattr(eval_module, "pinned_manifest_path", lambda: missing)
    code = eval_module.main(_CLI)
    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""
    assert "pinned manifest is missing" in captured.err


def test_embed_step_refuses_a_loader_sample_without_a_crop(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[int] = []
    _count_embed(monkeypatch, calls)
    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", (_BATCH_PATTERN,))
    bare = _reid("FX-BATCH-001", crop=None, path=Path("not-a-real-crop.txt"))
    good = _reid("FX-BATCH-002", crop=_crop(8), camera_id="g", track_id=2, label=1)

    def spy(
        _samples: Sequence[ReidSample],
        context: LoaderContext,
        _manifest_path: Path,
    ) -> list[ReidSample]:
        assert context is LoaderContext.EVALUATION
        if not spy.used:  # type: ignore[attr-defined]
            spy.used = True  # type: ignore[attr-defined]
            return [bare]
        return [good]

    spy.used = False  # type: ignore[attr-defined]
    monkeypatch.setattr(data_module, "load_reid_samples", spy)
    monkeypatch.setattr(
        eval_module,
        "synthetic_evaluation_split",
        lambda _seed: ((bare,), (good,)),
    )
    code = eval_module.main(_CLI)
    captured = capsys.readouterr()
    assert code != 0
    assert calls == []
    assert captured.out == ""
    assert "only accepts a crop" in captured.err


def test_embedder_row_count_mismatch_prints_no_metric(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    query, gallery = _hand_split(6)
    monkeypatch.setattr(data_module, "_EVALUATION_ID_PATTERNS", (_BATCH_PATTERN,))
    monkeypatch.setattr(eval_module, "synthetic_evaluation_split", lambda _seed: (query, gallery))

    def bad_embed(
        self: UntrainedPlaceholderEmbedder,
        _crops: Sequence[np.ndarray],
    ) -> np.ndarray:
        del self
        return np.zeros((1,), dtype=np.float32)

    monkeypatch.setattr(UntrainedPlaceholderEmbedder, "embed", bad_embed)
    code = eval_module.main(_CLI)
    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""
    assert "embedder returned shape" in captured.err


def test_all_invalid_queries_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every query lacks a cross-camera match, so the command prints no 0.0."""
    calls: list[int] = []
    _count_embed(monkeypatch, calls)
    query = (
        _reid(
            "FX-SYNREID-001",
            use="DEV",
            crop=_crop(1),
            camera_id="cam-a",
            track_id=1,
            label=1,
        ),
    )
    gallery = (
        _reid(
            "FX-SYNREID-002",
            use="DEV",
            crop=_crop(2),
            camera_id="cam-a",
            track_id=2,
            label=1,
        ),
    )
    monkeypatch.setattr(eval_module, "synthetic_evaluation_split", lambda _seed: (query, gallery))
    code = eval_module.main(_CLI)
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "0.0" not in captured.out
    assert "0.0" not in captured.err
    assert "rank1" not in captured.out
    assert "mAP" not in captured.out
    assert "invalid" in captured.err
    assert "No metric was computed." in captured.err
    assert calls == []


def test_synthetic_seed_flag_changes_the_split(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--synthetic-seed`` is forwarded, and the split uses that seed.

    Hard-coding ``make_synthetic_reid_set(seed=0)`` inside
    ``synthetic_evaluation_split`` fails this test.
    """
    seen: list[int] = []
    real = eval_module.synthetic_evaluation_split

    def spy(seed: int) -> tuple[tuple[ReidSample, ...], tuple[ReidSample, ...]]:
        seen.append(seed)
        return real(seed)

    monkeypatch.setattr(eval_module, "synthetic_evaluation_split", spy)
    assert eval_module.main(["--embedder", "placeholder", "--synthetic-seed", "0", "--json"]) == 0
    assert eval_module.main(["--embedder", "placeholder", "--synthetic-seed", "1", "--json"]) == 0
    assert seen == [0, 1]
    capsys.readouterr()
    same_query, same_gallery = real(0)
    other_query, other_gallery = real(1)
    assert _crop_bytes(same_query) == _crop_bytes(real(0)[0])
    assert _crop_bytes(same_gallery) == _crop_bytes(real(0)[1])
    assert _crop_bytes(same_query) != _crop_bytes(other_query)
    assert _crop_bytes(same_gallery) != _crop_bytes(other_gallery)


def _crop_bytes(samples: tuple[ReidSample, ...]) -> bytes:
    parts: list[bytes] = []
    for sample in samples:
        assert sample.crop is not None
        parts.append(sample.crop.tobytes())
    return b"".join(parts)


def test_cli_on_the_synthetic_split_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = eval_module.main(_CLI)
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert tuple(payload) == _CLI_JSON_KEYS
    assert payload["evidence"] is False
    assert payload["embedder"] == _PLACEHOLDER_LABEL
    assert payload["note"] == _NOTE
    assert "AC-06" in str(payload["note"])
    assert "Placeholder" in str(payload["note"])
    assert str(payload["report"]).splitlines()[0] == f"embedder: {_PLACEHOLDER_LABEL}"
    assert payload["num_valid_query"] > 0
    assert payload["num_query"] > 0
    assert payload["num_gallery"] > 0


def test_build_embedder_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="not available"):
        eval_module.build_embedder("paper-osnet", 0)
    embedder = eval_module.build_embedder("placeholder", 0)
    assert isinstance(embedder, UntrainedPlaceholderEmbedder)
    assert embedder.placeholder is True
    assert embedder.name == _PLACEHOLDER_LABEL
    osnet = eval_module.build_embedder("osnet-lite", 7)
    assert isinstance(osnet, OsnetLiteEmbedder)
    assert osnet.placeholder is True
    assert osnet.name == "osnet-lite-random-init"


def test_osnet_lite_eval_embedder_uses_the_given_seed() -> None:
    """Seed 7 from the eval builder matches ``OsnetLiteEmbedder(seed=7)``.

    The comparison is the embedding values. Hard-coding ``seed=0`` in
    ``build_embedder`` fails it. Seed 0 on the same crops differs, so a
    passed-through seed is visible.
    """
    crops = [_crop(11), _crop(12, shape=(16, 10, 3))]
    from_eval = eval_module.build_embedder("osnet-lite", 7).embed(crops)
    direct = OsnetLiteEmbedder(seed=7).embed(crops)
    other = OsnetLiteEmbedder(seed=0).embed(crops)
    np.testing.assert_array_equal(from_eval, direct)
    assert not np.array_equal(from_eval, other)


def test_script_entrypoint_prints_no_metric() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run_reid_eval.py"),
            "--embedder",
            "placeholder",
            "--synthetic-seed",
            "0",
            "--json",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["evidence"] is False
    assert "AC-06" in payload["note"]
    assert "Placeholder" in payload["note"]
    assert payload["num_valid_query"] > 0
    assert completed.stderr == ""


def test_eval_makes_no_network_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def refuse_connect(*_args: object, **_kwargs: object) -> None:
        calls.append("connect")
        raise OSError("refused")

    def refuse_create_connection(*_args: object, **_kwargs: object) -> None:
        calls.append("create_connection")
        raise OSError("refused")

    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    monkeypatch.setattr(socket, "create_connection", refuse_create_connection)
    query, gallery = _hand_built()
    assert rank1_map(query, gallery).num_query == 3
    code = eval_module.main(_CLI)
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert calls == []
