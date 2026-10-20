"""Fixture separation check.

Compares SMOKE_ONLY and CODE_TEST manifest rows to data files by SHA-256 and
by pHash (Hamming distance <= 8). Source files are not data. The top-level
``smoke/`` store is the fixture directory and is not itself a leak; a copy
anywhere else, including ``data/train/smoke/`` or a ``.dat`` rename, is.

Training, eval, val, holdout, test, and study roots are matched
case-insensitively (``eval/``, ``Data/Train/``, ``TRAIN/``). Every file under
those roots is content-hashed, whatever its suffix.

Fail closed:
- explicit empty dataset or empty dataset directory
- missing directory, missing file, unreadable file
- decode or pHash errors
- manifest with no SMOKE_ONLY or CODE_TEST rows
- a code path that computes gradients or runs an optimizer step, including a
  one-step loop, and can read SMOKE_ONLY or CODE_TEST data
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trace_search.fixtures.content import (
    HAMMING_THRESHOLD,
    MEDIA_EXTENSIONS,
    ContentCheckError,
    nearest_hamming,
    perceptual_hashes,
    sha256_file,
)

FORBIDDEN_TAGS = {"SMOKE_ONLY", "CODE_TEST"}
# Casefolded directory names. Any parent with one of these names is a
# training, eval, or study input, including Data/Train and TRAIN.
DATA_DIR_NAMES = {
    "data",
    "datasets",
    "dev",
    "eval",
    "evaluation",
    "fixtures",
    "hold",
    "holdout",
    "study",
    "test",
    "train",
    "training",
    "val",
    "validation",
}
SKIP_ROOT_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "smoke",
    ".pytest_cache",
    "htmlcov",
    "dist",
    "build",
    "__pycache__",
}
# Package and doc trees are not training roots, even when a directory is named fixtures.
CODE_ROOTS = {"src", "tests", "scripts", "docs", "licenses"}
CODE_SCAN_SKIP = {"tests", "licenses", ".git", ".venv", "venv", "node_modules", "__pycache__"}

def _training_pattern(*parts: str) -> re.Pattern[str]:
    """Join a pattern so this file does not contain the token it searches for."""
    return re.compile("".join(parts))


# A training job computes gradients or runs an optimizer step, including one step.
TRAINING_PATTERNS = (
    _training_pattern(r"\.", "backward", r"\s*\("),
    _training_pattern(r"\b", "optimizer", r"\.step\s*\("),
    _training_pattern(r"\b", "opt", r"\.step\s*\("),
    _training_pattern(r"\b", "torch", r"\.optim\b"),
    _training_pattern(r"\b", "Gradient", "Tape", r"\b"),
    _training_pattern(r"\b", "autograd", r"\.grad\s*\("),
    _training_pattern(r"\.", "zero_grad", r"\s*\("),
)
SMOKE_READ_PATTERNS = (
    re.compile(r"SMOKE_ONLY"),
    re.compile(r"\bCODE_TEST\b"),
    re.compile(r"FX-SMOKE"),
    re.compile(r"FX-CODE"),
    re.compile(r"(?:^|[^A-Za-z0-9_])smoke[/\\]"),
    re.compile(r"""['\"]smoke['\"]"""),
)


class SeparationCheckError(Exception):
    """Separation check could not prove the datasets are clean."""


def _relative_parts(path: Path, repo_root: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(repo_root).parts
    except ValueError:
        return path.parts


def _skipped_store_or_tooling(path: Path, repo_root: Path) -> bool:
    parts = _relative_parts(path, repo_root)
    if not parts:
        return True
    if parts[0] in SKIP_ROOT_DIRS:
        return True
    return any(part in {"__pycache__", ".git"} for part in parts)


def _in_training_root(parts: tuple[str, ...]) -> bool:
    """True for files under a training, eval, val, holdout, test, or study root.

    Names are compared case-insensitively, so ``Data/Train`` and ``TRAIN``
    match. ``src/`` and the other code roots are excluded so the package
    directory named ``fixtures`` is not treated as a dataset.
    """
    if len(parts) < 2 or parts[0] in CODE_ROOTS:
        return False
    return any(part.casefold() in DATA_DIR_NAMES for part in parts[:-1])


def auto_discover_data_files(repo_root: Path) -> list[Path]:
    """Media files, plus every file under a training, eval, or study root.

    Suffix does not matter inside those roots: a smoke clip renamed to
    ``eval/clip.txt`` or ``Data/Train/clip.txt`` is still SHA-256 checked.
    pHash runs later, and only when the bytes decode. Source trees are not data.
    """
    if not repo_root.exists():
        raise SeparationCheckError(f"FAIL-CLOSED: repository root does not exist: {repo_root}")

    found: list[Path] = []
    seen: set[Path] = set()
    for path in repo_root.rglob("*"):
        if _skipped_store_or_tooling(path, repo_root):
            continue
        if path.is_symlink() and not path.exists():
            raise SeparationCheckError(f"FAIL-CLOSED: broken symlink {path}")
        if not path.is_file():
            continue
        parts = _relative_parts(path, repo_root)
        in_data_dir = _in_training_root(parts)
        if path.suffix.lower() == ".py" and not in_data_dir:
            continue
        if in_data_dir or path.suffix.lower() in MEDIA_EXTENSIONS:
            key = path.absolute()
            if key not in seen:
                seen.add(key)
                found.append(path)
    return found


def load_manifest_entries(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        raise SeparationCheckError(f"FAIL-CLOSED: manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeparationCheckError(
            f"FAIL-CLOSED: cannot read manifest {manifest_path}: {exc}"
        ) from exc
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise SeparationCheckError("FAIL-CLOSED: manifest has no entries list")
    return entries


def forbidden_fixtures(entries: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Map forbidden SHA-256 to fixture id, and SHA-256 to pHashes."""
    sha_to_id: dict[str, str] = {}
    sha_to_phash: dict[str, list[str]] = {}
    for entry in entries:
        use_tag = entry.get("use")
        if use_tag not in FORBIDDEN_TAGS:
            continue
        fixture_id = entry.get("fixture_id") or "unknown"
        sha = entry.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise SeparationCheckError(
                f"FAIL-CLOSED: {fixture_id} is {use_tag} but has no SHA-256"
            )
        phashes = entry.get("phash")
        if not isinstance(phashes, list) or not phashes:
            raise SeparationCheckError(
                f"FAIL-CLOSED: {fixture_id} is {use_tag} but has no pHash list"
            )
        sha_to_id[sha] = fixture_id
        sha_to_phash[sha] = [str(item) for item in phashes]
    if not sha_to_id:
        raise SeparationCheckError(
            "FAIL-CLOSED: manifest has no SMOKE_ONLY or CODE_TEST rows"
        )
    return sha_to_id, sha_to_phash


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SeparationCheckError(f"FAIL-CLOSED: cannot read {path}: {exc}") from exc


def find_training_job_violations(repo_root: Path, manifest_path: Path | None = None) -> list[str]:
    """Training code that can read SMOKE_ONLY or CODE_TEST data.

    A training job is any file that computes gradients or runs an optimizer
    step, including a one-step loop. It can read forbidden data when the same
    file references the smoke store, a forbidden tag, or an FX-SMOKE / FX-CODE
    id. Config and playlist files outside ``licenses/`` are scanned on their own.
    """
    if not repo_root.exists():
        raise SeparationCheckError(f"FAIL-CLOSED: repository root does not exist: {repo_root}")

    violations: list[str] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        parts = _relative_parts(path, repo_root)
        if not parts or parts[0] in CODE_SCAN_SKIP or parts[0] in SKIP_ROOT_DIRS:
            continue
        if any(part in {"__pycache__", ".git"} for part in parts):
            continue
        if manifest_path is not None and path.resolve() == Path(manifest_path).resolve():
            continue
        suffix = path.suffix.lower()
        if suffix == ".py":
            text = _read_text(path)
            trains = any(pattern.search(text) for pattern in TRAINING_PATTERNS)
            reads_smoke = any(pattern.search(text) for pattern in SMOKE_READ_PATTERNS)
            if trains and reads_smoke:
                violations.append(
                    f"{path}: training step (gradient or optimizer, including a 1-step loop) "
                    "can read SMOKE_ONLY or CODE_TEST data"
                )
        elif suffix in {".yaml", ".yml", ".json"}:
            text = _read_text(path)
            if any(pattern.search(text) for pattern in SMOKE_READ_PATTERNS):
                violations.append(
                    f"{path}: config or playlist references SMOKE_ONLY or CODE_TEST data"
                )
    return violations


def check_separation(
    manifest_path: Path,
    dataset_paths: list[Path] | None = None,
    hamming_threshold: int = HAMMING_THRESHOLD,
    repo_root: Path | None = None,
) -> dict:
    """Check dataset files. An explicit empty list is an error.

    ``dataset_paths is None`` auto-discovers data files under ``repo_root``.
    A code-only repo with zero data files is not an error by itself: the
    manifest must still contain forbidden rows, and the training-job scan
    still runs. Pointing the check at a dataset and handing it nothing is
    an error.
    """
    if repo_root is None:
        repo_root = Path.cwd()
    repo_root = Path(repo_root)

    discovered = dataset_paths is None
    if discovered:
        dataset_paths = auto_discover_data_files(repo_root)
    else:
        dataset_paths = list(dataset_paths)
        if len(dataset_paths) == 0:
            raise SeparationCheckError(
                "FAIL-CLOSED: 0 dataset files. Pass data files or a repo root "
                "that contains them; an empty dataset is not a pass."
            )

    entries = load_manifest_entries(manifest_path)
    sha_to_id, sha_to_phash = forbidden_fixtures(entries)
    training_violations = find_training_job_violations(repo_root, manifest_path)
    if training_violations:
        raise SeparationCheckError(
            "FAIL-CLOSED: training job can read forbidden fixtures:\n"
            + "\n".join(training_violations)
        )

    exact_matches: list[tuple[str, str, str]] = []
    near_duplicates: list[tuple[str, str, int]] = []

    for file_path in dataset_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            raise SeparationCheckError(f"FAIL-CLOSED: missing file {file_path}")
        try:
            file_sha = sha256_file(file_path)
        except ContentCheckError as exc:
            raise SeparationCheckError(f"FAIL-CLOSED: {exc}") from exc

        if file_sha in sha_to_id:
            exact_matches.append((str(file_path), sha_to_id[file_sha], "SHA-256 exact match"))
            continue

        try:
            file_hashes = perceptual_hashes(file_path)
        except ContentCheckError as exc:
            raise SeparationCheckError(f"FAIL-CLOSED: {exc}") from exc

        for forbidden_sha, fixture_id in sha_to_id.items():
            distance = nearest_hamming(file_hashes, sha_to_phash[forbidden_sha])
            if distance is not None and distance <= hamming_threshold:
                near_duplicates.append((str(file_path), fixture_id, distance))
                break

    passed = not exact_matches and not near_duplicates
    return {
        "exact_matches": exact_matches,
        "near_duplicates": near_duplicates,
        "config_violations": [],
        "training_violations": [],
        "files_checked": len(dataset_paths),
        "passed": passed,
        "message": (
            "No data files outside the smoke store; training-job scan is clean."
            if discovered and len(dataset_paths) == 0
            else ""
        ),
    }


def run_separation_check(
    manifest_path: Path,
    train_dir: Path | None = None,
    dev_dir: Path | None = None,
    hold_dir: Path | None = None,
    study_dir: Path | None = None,
    hamming_threshold: int = HAMMING_THRESHOLD,
    repo_root: Path | None = None,
) -> dict:
    """Run the check. A named directory that is missing or empty is an error."""
    if repo_root is None:
        repo_root = Path.cwd()
    repo_root = Path(repo_root)

    named: list[Path] = []
    files: list[Path] | None = None
    for directory in (train_dir, dev_dir, hold_dir, study_dir):
        if directory is None:
            continue
        if not directory.exists() or not directory.is_dir():
            raise SeparationCheckError(
                f"FAIL-CLOSED: specified directory does not exist: {directory}"
            )
        named.append(directory)
        if files is None:
            files = []
        files.extend(path for path in directory.rglob("*") if path.is_file())

    if named and not files:
        raise SeparationCheckError(
            "FAIL-CLOSED: 0 dataset files in " + ", ".join(str(path) for path in named)
        )

    result = check_separation(
        manifest_path,
        files,
        hamming_threshold=hamming_threshold,
        repo_root=repo_root,
    )
    return {
        "datasets_checked": [str(path) for path in named] if named else ["auto-discovered"],
        "files_checked": result["files_checked"],
        "exact_matches": result["exact_matches"],
        "near_duplicates": result["near_duplicates"],
        "config_violations": result["config_violations"],
        "training_violations": result["training_violations"],
        "passed": result["passed"],
        "message": result.get("message", ""),
    }

