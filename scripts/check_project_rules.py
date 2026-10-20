#!/usr/bin/env python3
"""Project rules checker for trace-search.

Enforces PRD hard-stops:
1. No face-recognition, face embedding/crop, gait, watchlist, enrolment or named-identity
   code or dependencies.
2. No research-only datasets or weights (Market-1501, MSMT17, DukeMTMC, Objects365, etc.).
3. No media files committed unless a clip or synthetic_clip manifest entry lists
   that exact repo-relative path and sha256 with an approved use tag.
4. Reviewed allow-list entries in .project-rules-allow for false positives.

Exit 0 if all checks pass, exit 1 if violations found.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
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
REQ_FILES = ("requirements.txt", "requirements-dev.txt", "requirements-ci.txt", "pyproject.toml")


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
    # media:<file> does not waive the manifest path and sha256 check,
    # and it does not silence any other violation in that file.
    if violation_id in allow_list:
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


def check_media_files(allow_list: set[str] | None = None) -> list[dict[str, Any]]:
    if allow_list is None:
        allow_list = load_allow_list()
    violations: list[dict[str, Any]] = []
    allowed = manifest_media_by_path(load_manifest())
    checked: set[str] = set()

    for rel, expected in sorted(allowed.items()):
        path = Path(rel)
        checked.add(rel)
        if not path.is_file():
            add_violation(
                violations,
                allow_list,
                kind="media_file",
                file=rel,
                message=f"Manifest lists {rel} but the file is missing",
            )
            continue
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
            continue
        if digest.lower() != expected:
            add_violation(
                violations,
                allow_list,
                kind="media_file",
                file=rel,
                message=f"Media file {rel} sha256 does not match entries[].path",
            )

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
    print("  A media:<file> line does not replace the manifest path and sha256 check.")
    print("  Each allow-list entry must be reviewed and have a justification comment.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
