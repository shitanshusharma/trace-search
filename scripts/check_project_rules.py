#!/usr/bin/env python3
"""Project rules checker for trace-search.

Enforces PRD hard-stops:
1. No face-recognition, face embedding/crop, gait, watchlist, enrolment or named-identity
   code or dependencies.
2. No research-only datasets or weights (Market-1501, MSMT17, DukeMTMC, Objects365, etc.).
3. No media files committed unless a clip or synthetic_clip manifest entry lists
   that exact repo-relative path and sha256 with an approved use tag.
   Generated SMOKE_ONLY clips at smoke/*.mp4 are not stored in git. Their file
   check is replaced by a structure check, and the regenerate test hashes them.
   The allow-list cannot waive this check.
4. Reviewed allow-list entries in .project-rules-allow for false positives
   other than media.

Exit 0 if all checks pass, exit 1 if violations found.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Import module names. Matched against AST import nodes, not raw text,
# so a pattern table that mentions these names is not itself an import.
BANNED_MODULES = (
    "face_recognition",
    "insightface",
    "arcface",
    "retinaface",
    "deepface",
    "facenet",
    "opengait",
    "ultralytics",
    "boxmot",
)

# Still applied to requirements files and pyproject dependency text.
BANNED_DEP_PATTERNS = (
    r"face_recognition",
    r"insightface",
    r"arcface",
    r"retinaface",
    r"deepface",
    r"facenet",
    r"opengait",
    r"ultralytics",
    r"boxmot",
    r"dlib.*face",
    r"mediapipe.*face",
)

# Non-alphanumeric boundaries, so face_embedding matches and myface_embedding does not.
# Underscore is a boundary, so add_to_watchlist and enroll_person match.
BANNED_IDENT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"watchlist|"
    r"face_embeddings?|"
    r"face_crops?|"
    r"national_id|"
    r"identity_id|"
    r"enrol(?:l(?:ed|ment)?|ment)?|"
    r"gait"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

BANNED_DATASETS = (
    r"market.?1501",
    r"msmt17",
    r"duke.?mtmc",
    r"cuhk03",
    r"casia.?b",
    r"\bgrew\b",
    r"gait3d",
    r"objects365",
    r"model.?zoo.*reid",
    # Person-ReID research sets. Outer boundary is non-alphanumeric.
    # '.?' between parts allows '_' or '-'. Case is ignored by the scanner.
    # MARS is only mars.?(reid|dataset). personx stays one token.
    r"(?<![A-Za-z0-9])viper(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])cuhk.?01(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])cuhk.?02(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])prid(?:2011)?(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])i.?lids.?vid(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])mars.?(reid|dataset)(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])lu.?person(?:.?nl)?(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])sysu.?30k(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])rand.?person(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])personx(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])unreal.?person(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])cloned.?person(?![A-Za-z0-9])",
)

# Bare `name = "..."` / `name: str` only. camera_name and file_name do not match
# because `_` is part of the identifier. Applied to non-Python files; Python uses AST.
NAME_ASSIGN_RE = re.compile(r"""(?<![A-Za-z0-9_])name\s*=\s*['"]""")

MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".h264",
    ".h265",
    ".dav",
    ".webm",
    ".flv",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".pt",
    ".pth",
    ".onnx",
    ".safetensors",
    ".h5",
    ".pb",
    ".tflite",
    ".weights",
    ".ckpt",
    ".bin",
    ".npy",
    ".engine",
}

# Merged manifest schema: clip and synthetic_clip rows use SMOKE_ONLY.
# CODE_TEST is the other schema use. FX-CODE is a fixture_id, not a use.
APPROVED_MEDIA_USES = {"SMOKE_ONLY", "CODE_TEST"}
MEDIA_ENTRY_TYPES = {"clip", "synthetic_clip"}
# Whole-repo tracked clips. Suffixes are compared case-insensitively.
TRACKED_MEDIA_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
SKIP_DIR_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}

# Exact repo-relative paths. A path is exempt only when it equals one of these,
# never because it ends with the same filename.
# scripts/check_licenses.py, tests/test_package.py and pyproject.toml are
# deliberately absent: a real import or watchlist key there must fail.
IDENTIFIER_EXEMPT = {
    # Planted fixture. The same identifier in src/ still fails.
    "tests/test_project_rules_planted.py",
    # Rule text. A watchlist key in any other file still fails.
    ".semgrep.yml",
    # This file's docstring and pattern strings name the banned identifiers.
    "scripts/check_project_rules.py",
}
IMPORT_EXEMPT = {
    "tests/test_project_rules_planted.py",
    # Semgrep --test corpus. Any other file under tests/semgrep/ is scanned.
    "tests/semgrep/rules.py",
}
# Rule tables and planted fixtures name banned datasets so they can reject them.
DATASET_EXEMPT = {
    "scripts/check_project_rules.py",
    "scripts/check_licenses.py",
    "scripts/demo_planted_cases.py",
    "tests/test_project_rules_planted.py",
    "tests/test_license_fence.py",
    "tests/test_license_scanning.py",
    "tests/test_comprehensive_fence.py",
    ".semgrep.yml",
}
# Semgrep --test corpus. Identifier and dataset text only.
# Imports are not covered by this prefix.
EXEMPT_PREFIXES = ("tests/semgrep/",)

ALLOW_LIST_FILE = ".project-rules-allow"
MANIFEST_FILE = "licenses/manifest.json"
REQ_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-dev.lock",
    "requirements-ci.txt",
    "requirements-ci.lock",
    "pyproject.toml",
)


def repo_rel(path: Path) -> str:
    return path.as_posix().removeprefix("./")


def is_prefix_exempt(rel: str) -> bool:
    return any(rel.startswith(prefix) for prefix in EXEMPT_PREFIXES)


def skip_path(path: Path) -> bool:
    return any(part in SKIP_DIR_PARTS for part in path.parts)


def iter_files(pattern: str) -> list[Path]:
    found: list[Path] = []
    for path in Path(".").rglob(pattern):
        if not path.is_file() or skip_path(path):
            continue
        found.append(path)
    return found


def load_allow_list() -> set[str]:
    allow_list: set[str] = set()
    allow_file = Path(ALLOW_LIST_FILE)
    if not allow_file.exists():
        return allow_list
    for line in allow_file.read_text().splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            allow_list.add(entry)
    return allow_list


def load_manifest() -> dict[str, Any]:
    manifest_path = Path(MANIFEST_FILE)
    if not manifest_path.exists():
        return {"entries": []}
    try:
        loaded = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {"entries": []}
    return loaded if isinstance(loaded, dict) else {"entries": []}


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def add_violation(
    violations: list[dict[str, Any]],
    allow_list: set[str],
    *,
    kind: str,
    file: str,
    message: str,
    line: int | None = None,
    match: str = "",
) -> None:
    if line is None:
        violation_id = f"{kind}:{file}"
    else:
        violation_id = f"{file}:{line}:{match}"
    # Every media file goes through entries[].path and sha256.
    # media:<file> and media_file:<file> do not waive that, or any other check.
    if kind != "media_file" and violation_id in allow_list:
        return
    item: dict[str, Any] = {"type": kind, "file": file, "message": message}
    if line is not None:
        item["line"] = line
        item["match"] = match
    violations.append(item)


def module_is_banned(module: str) -> bool:
    lowered = module.lower()
    for name in BANNED_MODULES:
        if lowered == name or lowered.startswith(name + "."):
            return True
    if lowered == "dlib" or lowered.startswith("dlib."):
        return "face" in lowered
    return "mediapipe" in lowered and "face" in lowered


def ident_is_banned(ident: str) -> bool:
    return BANNED_IDENT_RE.search(ident) is not None


def is_str_annotation(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Name) and node.id == "str"


def is_string_constant(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def check_name_targets(node: ast.AST) -> bool:
    """True when this node is a bare identity field `name`, not camera_name or self.name."""
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        if node.target.id != "name":
            return False
        if is_str_annotation(node.annotation):
            return True
        return is_string_constant(node.value)
    if isinstance(node, ast.Assign) and is_string_constant(node.value):
        return any(isinstance(target, ast.Name) and target.id == "name" for target in node.targets)
    return False


def iter_banned_idents(tree: ast.AST):
    """Yield (line, match) for banned names in defs, params, keywords, and strings.

    camera_name, file_name and self.name do not match: the name rule is the bare
    identifier, and the banned-word regex does not treat '_' as part of a word.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ident_is_banned(node.name):
                yield node.lineno, node.name
        elif isinstance(node, ast.arg) and ident_is_banned(node.arg):
            yield node.lineno, node.arg
        elif isinstance(node, ast.keyword) and node.arg and ident_is_banned(node.arg):
            yield node.lineno, node.arg
        elif isinstance(node, ast.alias) and node.asname and ident_is_banned(node.asname):
            yield node.lineno, node.asname
        elif isinstance(node, ast.Name) and ident_is_banned(node.id):
            yield node.lineno, node.id
        elif isinstance(node, ast.Attribute) and ident_is_banned(node.attr):
            yield node.lineno, node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for match in BANNED_IDENT_RE.finditer(node.value):
                yield node.lineno, match.group()
        if check_name_targets(node):
            yield getattr(node, "lineno", 1), "name"


def check_python_file(
    path: Path,
    violations: list[dict[str, Any]],
    allow_list: set[str],
) -> None:
    rel = repo_rel(path)
    try:
        source = path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        add_violation(
            violations,
            allow_list,
            kind="unreadable",
            file=rel,
            message=f"Unreadable Python file: {rel} ({exc})",
        )
        return
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        add_violation(
            violations,
            allow_list,
            kind="syntax_error",
            file=rel,
            line=exc.lineno or 1,
            match="syntax",
            message=f"Python file failed to parse: {exc.msg}",
        )
        return

    if rel not in IMPORT_EXEMPT:
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.Import):
                modules = [(alias.name, node.lineno) for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [(node.module, node.lineno)]
            for module, lineno in modules:
                if module_is_banned(module):
                    add_violation(
                        violations,
                        allow_list,
                        kind="banned_import",
                        file=rel,
                        line=lineno,
                        match=module,
                        message=f"Banned import: {module}",
                    )

    if rel in IDENTIFIER_EXEMPT or is_prefix_exempt(rel):
        return
    for lineno, match in iter_banned_idents(tree):
        if match == "name":
            message = "Bare identity field name is banned"
        else:
            message = f"Banned identifier: {match}"
        add_violation(
            violations,
            allow_list,
            kind="banned_identifier",
            file=rel,
            line=lineno,
            match=match,
            message=message,
        )


def check_dependency_text(
    rel: str,
    content: str,
    violations: list[dict[str, Any]],
    allow_list: set[str],
) -> None:
    for pattern in BANNED_DEP_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            add_violation(
                violations,
                allow_list,
                kind="banned_dependency",
                file=rel,
                line=line_of(content, match.start()),
                match=match.group(),
                message=f"Banned dependency pattern: {pattern}",
            )


def check_datasets(
    rel: str,
    content: str,
    violations: list[dict[str, Any]],
    allow_list: set[str],
) -> None:
    if rel in DATASET_EXEMPT or is_prefix_exempt(rel):
        return
    for pattern in BANNED_DATASETS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            add_violation(
                violations,
                allow_list,
                kind="banned_dataset",
                file=rel,
                line=line_of(content, match.start()),
                match=match.group(),
                message=f"Banned research dataset/weight: {pattern}",
            )


def check_non_python_identifiers(
    path: Path,
    violations: list[dict[str, Any]],
    allow_list: set[str],
) -> None:
    rel = repo_rel(path)
    if rel in IDENTIFIER_EXEMPT or is_prefix_exempt(rel):
        return
    try:
        content = path.read_text()
    except (OSError, UnicodeDecodeError):
        return
    for match in BANNED_IDENT_RE.finditer(content):
        add_violation(
            violations,
            allow_list,
            kind="banned_identifier",
            file=rel,
            line=line_of(content, match.start()),
            match=match.group(),
            message=f"Banned identifier: {match.group()}",
        )
    if path.suffix == ".toml":
        for match in NAME_ASSIGN_RE.finditer(content):
            add_violation(
                violations,
                allow_list,
                kind="banned_identifier",
                file=rel,
                line=line_of(content, match.start()),
                match=match.group(),
                message="Bare identity field name is banned",
            )


def check_imports_and_deps(allow_list: set[str]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for path in iter_files("*.py"):
        rel = repo_rel(path)
        check_python_file(path, violations, allow_list)
        try:
            content = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        check_datasets(rel, content, violations, allow_list)

    for pattern in ("*.yaml", "*.yml", "*.json", "*.toml"):
        for path in iter_files(pattern):
            rel = repo_rel(path)
            check_non_python_identifiers(path, violations, allow_list)
            try:
                content = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            check_datasets(rel, content, violations, allow_list)

    for rel in REQ_FILES:
        path = Path(rel)
        if not path.exists():
            continue
        check_dependency_text(rel, path.read_text(), violations, allow_list)
    return violations


def is_repo_relative_posix(value: str) -> bool:
    """True when path is repo-relative POSIX: no leading '/' and no '..' segment."""
    if not value or value.startswith("/") or "\\" in value:
        return False
    return all(part != ".." for part in value.split("/"))


def manifest_media_by_path(manifest: dict[str, Any]) -> dict[str, str]:
    """Map entries[].path -> sha256 for clip and synthetic_clip rows.

    The hash is only valid for that path. A files[] list is ignored.
    """
    allowed: dict[str, str] = {}
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return allowed
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") not in MEDIA_ENTRY_TYPES:
            continue
        if entry.get("use") not in APPROVED_MEDIA_USES:
            continue
        sha = entry.get("sha256")
        media_path = entry.get("path")
        if not isinstance(sha, str) or not isinstance(media_path, str):
            continue
        if not re.fullmatch(r"[a-fA-F0-9]{64}", sha):
            continue
        if not is_repo_relative_posix(media_path):
            continue
        allowed[media_path] = sha.lower()
    return allowed


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _script_repo() -> Path:
    return Path(__file__).resolve().parent.parent


def _smoke_rows():
    src = _script_repo() / "src"
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    from trace_search.fixtures import smoke_rows

    return smoke_rows


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def recorded_seed_is_int(seed: object) -> bool:
    """The deterministic seed is an int.

    A bare int counts. The manifest stores a JSON object whose base_seed and
    offset are both ints; that pair is the generator seed.
    """
    if _is_int(seed):
        return True
    if not isinstance(seed, str):
        return False
    try:
        parsed = json.loads(seed)
    except json.JSONDecodeError:
        return False
    if _is_int(parsed):
        return True
    if not isinstance(parsed, dict):
        return False
    return _is_int(parsed.get("base_seed")) and _is_int(parsed.get("offset"))


def _phash_is_well_formed(phash: object) -> bool:
    if not isinstance(phash, list) or len(phash) != 10:
        return False
    if len(set(phash)) != 10:
        return False
    return all(isinstance(item, str) and re.fullmatch(r"[a-f0-9]{16}", item) for item in phash)


def _generator_class_exists(name: object) -> bool:
    if not isinstance(name, str) or not name.isidentifier():
        return False
    pattern = re.compile(rf"^class[ \t]+{re.escape(name)}\b", re.MULTILINE)
    root = _script_repo()
    for path in root.rglob("*.py"):
        if not path.is_file() or skip_path(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if pattern.search(text):
            return True
    return False


def _hash_listed_file(
    violations: list[dict[str, Any]],
    allow_list: set[str],
    rel: str,
    expected: str,
) -> None:
    path = Path(rel)
    try:
        digest = hash_file(path)
    except OSError as exc:
        add_violation(
            violations,
            allow_list,
            kind="media_file",
            file=rel,
            message=f"Unreadable media file: {rel} ({exc})",
        )
        return
    if digest.lower() != expected:
        add_violation(
            violations,
            allow_list,
            kind="media_file",
            file=rel,
            message=f"Media file {rel} sha256 does not match entries[].path",
        )


def generated_smoke_clip_violations(
    manifest: dict[str, Any],
    allow_list: set[str],
    checked: set[str],
) -> list[dict[str, Any]]:
    """Structure-check generated smoke clips and do not require the mp4 on disk."""
    violations: list[dict[str, Any]] = []
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return violations
    predicate = _smoke_rows().is_generated_smoke_clip
    for entry in entries:
        if not isinstance(entry, dict) or not predicate(entry):
            continue
        rel = entry.get("path")
        if not isinstance(rel, str):
            continue
        checked.add(rel)
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
            add_violation(
                violations,
                allow_list,
                kind="media_file",
                file=rel,
                message=f"Media file {rel} sha256 must be 64 lowercase hex characters",
            )
        if not recorded_seed_is_int(entry.get("seed")):
            add_violation(
                violations,
                allow_list,
                kind="media_file",
                file=rel,
                message=f"Media file {rel} seed is not an int",
            )
        if not _phash_is_well_formed(entry.get("phash")):
            add_violation(
                violations,
                allow_list,
                kind="media_file",
                file=rel,
                message=f"Media file {rel} pHash is not well-formed",
            )
        generator = entry.get("generator")
        if not _generator_class_exists(generator):
            add_violation(
                violations,
                allow_list,
                kind="media_file",
                file=rel,
                message=f"Media file {rel} generator path named by the row is missing",
            )
        if isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha) and Path(rel).is_file():
            _hash_listed_file(violations, allow_list, rel, sha)
    return violations


def listed_media_violations(
    manifest: dict[str, Any],
    allow_list: set[str],
    checked: set[str],
) -> list[dict[str, Any]]:
    """Hash every other clip row. A missing file still fails."""
    violations: list[dict[str, Any]] = []
    for rel, expected in sorted(manifest_media_by_path(manifest).items()):
        if rel in checked:
            continue
        checked.add(rel)
        if not Path(rel).is_file():
            add_violation(
                violations,
                allow_list,
                kind="media_file",
                file=rel,
                message=f"Manifest lists {rel} but the file is missing",
            )
            continue
        _hash_listed_file(violations, allow_list, rel, expected)
    return violations


def unlisted_media_violations(
    allow_list: set[str],
    checked: set[str],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for path in Path(".").rglob("*"):
        if not path.is_file() or skip_path(path):
            continue
        if path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        rel = repo_rel(path)
        if rel in checked:
            continue
        try:
            hash_file(path)
        except OSError as exc:
            add_violation(
                violations,
                allow_list,
                kind="media_file",
                file=rel,
                message=f"Unreadable media file: {rel} ({exc})",
            )
            continue
        add_violation(
            violations,
            allow_list,
            kind="media_file",
            file=rel,
            message=(
                f"Media file {rel} is not listed at this path in the manifest "
                f"with an approved use tag"
            ),
        )
    return violations


def _git_text(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )


def tracked_media_violations(allow_list: set[str]) -> list[dict[str, Any]]:
    """Fail when git tracks a media file anywhere, or when smoke/ is not gitignored.

    Outside a checkout, nothing is tracked.
    """
    violations: list[dict[str, Any]] = []
    probe = _git_text(["git", "rev-parse", "--is-inside-work-tree"])
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return violations
    listed = _git_text(["git", "ls-files"])
    if listed.returncode != 0:
        add_violation(
            violations,
            allow_list,
            kind="media_file",
            file=".",
            message="git ls-files failed",
        )
    else:
        for rel in listed.stdout.splitlines():
            if Path(rel).suffix.lower() not in TRACKED_MEDIA_SUFFIXES:
                continue
            add_violation(
                violations,
                allow_list,
                kind="media_file",
                file=rel,
                message=f"Tracked media file: {rel}",
            )
    ignored = _git_text(["git", "check-ignore", "-q", "--", "smoke/"])
    if ignored.returncode != 0:
        add_violation(
            violations,
            allow_list,
            kind="media_file",
            file=".gitignore",
            message="smoke/ is not gitignored",
        )
    return violations


def _decorator_parts(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Call):
        node = node.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _is_skip_or_xfail(node: ast.AST) -> bool:
    return any(part in {"skip", "skipif", "xfail"} for part in _decorator_parts(node))


def _marked_skipped(node: ast.AST) -> bool:
    if isinstance(node, (ast.List, ast.Tuple)):
        return any(_is_skip_or_xfail(item) for item in node.elts)
    return _is_skip_or_xfail(node)


def _find_regenerate_test(
    tree: ast.AST, test_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    found: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == test_name:
            found = node
    return found


def _module_has_skip_marker(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "pytestmark"
                and _marked_skipped(node.value)
            ):
                return True
    return False


def regenerate_test_violations(allow_list: set[str]) -> list[dict[str, Any]]:
    """The regenerate test must stay collected so every generated row is hashed."""
    violations: list[dict[str, Any]] = []
    rel = "tests/test_t1_integration.py"
    path = _script_repo() / rel
    test_name = "test_regenerate_matches_manifest_entries_byte_for_byte"
    if not path.is_file():
        add_violation(
            violations,
            allow_list,
            kind="media_file",
            file=rel,
            message=f"{test_name} is missing",
        )
        return violations
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        add_violation(
            violations,
            allow_list,
            kind="media_file",
            file=rel,
            message=f"{test_name} could not be parsed: {exc}",
        )
        return violations
    found = _find_regenerate_test(tree, test_name)
    skipped = found is not None and any(_is_skip_or_xfail(dec) for dec in found.decorator_list)
    if found is None:
        add_violation(
            violations,
            allow_list,
            kind="media_file",
            file=rel,
            message=f"{test_name} is missing",
        )
    elif skipped or _module_has_skip_marker(tree):
        add_violation(
            violations,
            allow_list,
            kind="media_file",
            file=rel,
            message=f"{test_name} has a skip or xfail marker",
        )
    return violations


def check_media_files(allow_list: set[str] | None = None) -> list[dict[str, Any]]:
    """Hash media against entries[].path. The allow-list is not consulted."""
    del allow_list
    allow_list = set()
    manifest = load_manifest()
    checked: set[str] = set()
    violations = generated_smoke_clip_violations(manifest, allow_list, checked)
    violations.extend(listed_media_violations(manifest, allow_list, checked))
    violations.extend(unlisted_media_violations(allow_list, checked))
    violations.extend(tracked_media_violations(allow_list))
    violations.extend(regenerate_test_violations(allow_list))
    return violations


def main() -> int:
    print("Running project rules checks...")
    allow_list = load_allow_list()
    violations = check_imports_and_deps(allow_list)
    violations.extend(check_media_files(allow_list))
    if not violations:
        print("All project rules checks passed.")
        return 0

    print(f"\nFound {len(violations)} project rules violations:\n")
    for violation in violations:
        print(f"  {violation['type']}: {violation['file']}")
        if "line" in violation:
            print(f"    Line {violation['line']}: {violation['match']}")
        print(f"    {violation['message']}")
        print()

    Path("project-rules-violations.json").write_text(json.dumps(violations, indent=2))
    print("Wrote violations to project-rules-violations.json")
    print()
    print("To allow a false positive, add the violation ID to .project-rules-allow:")
    print("  Format: <file>:<line>:<match>")
    print("  media:<file> and media_file:<file> do not waive the manifest path and sha256 check.")
    print("  Each allow-list entry must be reviewed and have a justification comment.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
