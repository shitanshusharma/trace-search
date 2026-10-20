#!/usr/bin/env python3
"""
Project rules checker for trace-search.

Enforces PRD hard-stops:
1. No face-recognition, face embedding/crop, gait, watchlist, enrolment or named-identity
   code/deps (e.g., imports/deps like face_recognition, insightface, arcface, retinaface,
   dlib face models, deepface, facenet, mediapipe face, opengait, ultralytics, boxmot;
   identifiers like watchlist/enrol/face_embedding/gait; API schema fields like
   name/national_id)
2. No research-only datasets or weights (Market-1501, MSMT17, DukeMTMC, Objects365, etc.)
3. No media files (video/image/weights) committed outside manifest-backed fixtures
4. Explicit allow-mechanism for false positives (needs reviewed comment)

Exit 0 if all checks pass, exit 1 if violations found.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# Banned imports/dependencies (case-insensitive patterns)
BANNED_IMPORTS = [
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
]

# Research-only datasets and weights (case-insensitive)
BANNED_DATASETS = [
    r"market.?1501",  # Market-1501, market1501, etc.
    r"msmt17",
    r"duke.?mtmc",  # DukeMTMC, duke-mtmc, etc.
    r"cuhk03",
    r"casia.?b",  # CASIA-B, casia_b, etc.
    r"\bgrew\b",  # GREW dataset (word boundary to avoid matching "grew")
    r"gait3d",
    r"objects365",
    r"model.?zoo.*reid",  # model-zoo ReID weights, model_zoo_reid, etc.
]

# Banned identifiers in code (normalized: _ and . become spaces)
# These patterns are applied after normalizing underscores and dots to spaces
BANNED_IDENTIFIERS = [
    r"\bwatchlist\b",
    r"\benroll?\s?(?:ment|ed|person)?\b",  # enrol, enroll, enrolment, enrolled, enroll person
    r"\bface\s?embeddings?\b",  # face embedding, face embeddings, face_embedding
    r"\bface\s?crops?\b",  # face crop, face crops, face_crop
    r"\bgait(?:\s|[A-Z])|\bgait\b",  # gait , gait., GaitModel, or bare gait
    r"\bname\s*:\s*str\b",  # Typed field: name: str
    r"\bname\s*=\s*['\"]",  # name = "..." or name = '...'
    r"\bnational\s?id\b",
    r"\bidentity\s?id\b",
]

# Media file extensions (case-insensitive matching via Path.suffix.lower())
MEDIA_EXTENSIONS = [
    # Video
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".h264",
    ".h265",
    ".dav",
    ".webm",
    ".flv",
    # Image
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    # Model weights
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
]

# Excluded paths for rule-definition and planted-fixture files
# These files contain banned patterns as part of test cases or rule definitions
# JUSTIFICATION: Each exclusion must have a reviewed reason
EXCLUDED_PATHS = [
    "scripts/check_project_rules.py",  # Contains pattern definitions
    "scripts/check_licenses.py",  # Contains banned component patterns
    "scripts/demo_planted_cases.py",  # Planted test demonstrations
    "tests/test_project_rules_planted.py",  # Planted failure tests
    "tests/test_license_fence.py",  # License fence tests
    "tests/test_license_scanning.py",  # License scanning tests
    "tests/test_comprehensive_fence.py",  # Comprehensive fence tests
    "tests/test_package.py",  # Package tests
    ".semgrep.yml",  # Semgrep rule definitions
    "pyproject.toml",  # Package metadata (name = "..." is package/author name)
]

# Allow-list file (lines starting with # are comments)
ALLOW_LIST_FILE = ".project-rules-allow"

# Manifest file for media SHA verification
MANIFEST_FILE = "licenses/manifest.json"


def load_allow_list() -> set[str]:
    """Load the allow-list from .project-rules-allow file."""
    allow_list: set[str] = set()
    allow_file = Path(ALLOW_LIST_FILE)

    if not allow_file.exists():
        return allow_list

    content = allow_file.read_text()
    for line in content.splitlines():
        line = line.strip()
        # Skip empty lines and comments
        if line and not line.startswith("#"):
            allow_list.add(line)

    return allow_list


def load_manifest() -> dict[str, Any]:
    """Load the license manifest to check media file SHAs."""
    manifest_path = Path(MANIFEST_FILE)
    if not manifest_path.exists():
        return {"files": []}

    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, KeyError):
        return {"files": []}


def normalize_text(text: str) -> str:
    """Normalize text by replacing underscores and dots with spaces for matching."""
    return text.replace("_", " ").replace(".", " ")


def check_imports_and_deps() -> list[dict[str, Any]]:
    """Check for banned imports in Python files and dependencies."""
    violations: list[dict[str, Any]] = []
    allow_list = load_allow_list()

    # Check all Python source files (not just src/)
    for py_file in Path(".").rglob("*.py"):
        # Skip .git directory and excluded paths
        if ".git" in py_file.parts:
            continue
        if any(
            str(py_file) == excl or str(py_file).endswith(f"/{excl}") for excl in EXCLUDED_PATHS
        ):
            continue

        content = py_file.read_text()

        # Check banned imports
        for pattern in BANNED_IMPORTS:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                violation_id = f"{py_file}:{line_num}:{match.group()}"

                if violation_id not in allow_list:
                    violations.append(
                        {
                            "type": "banned_import",
                            "file": str(py_file),
                            "line": line_num,
                            "pattern": pattern,
                            "match": match.group(),
                            "message": f"Banned import/dependency pattern: {pattern}",
                        }
                    )

        # Check banned datasets
        for pattern in BANNED_DATASETS:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                violation_id = f"{py_file}:{line_num}:{match.group()}"

                if violation_id not in allow_list:
                    violations.append(
                        {
                            "type": "banned_dataset",
                            "file": str(py_file),
                            "line": line_num,
                            "pattern": pattern,
                            "match": match.group(),
                            "message": f"Banned research dataset/weight: {pattern}",
                        }
                    )

    # Check requirements files
    req_files = [
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-ci.txt",
    ]
    for req_file in req_files:
        req_path = Path(req_file)
        if not req_path.exists():
            continue

        content = req_path.read_text()
        for pattern in BANNED_IMPORTS:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                violation_id = f"{req_file}:{line_num}:{match.group()}"

                if violation_id not in allow_list:
                    violations.append(
                        {
                            "type": "banned_dependency",
                            "file": str(req_path),
                            "line": line_num,
                            "pattern": pattern,
                            "match": match.group(),
                            "message": f"Banned dependency pattern: {pattern}",
                        }
                    )

    # Check pyproject.toml
    pyproject = Path("pyproject.toml")
    if pyproject.exists():
        content = pyproject.read_text()
        for pattern in BANNED_IMPORTS:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                violation_id = f"pyproject.toml:{line_num}:{match.group()}"

                if violation_id not in allow_list:
                    violations.append(
                        {
                            "type": "banned_dependency",
                            "file": str(pyproject),
                            "line": line_num,
                            "pattern": pattern,
                            "match": match.group(),
                            "message": f"Banned dependency in pyproject.toml: {pattern}",
                        }
                    )

    return violations


def check_identifiers() -> list[dict[str, Any]]:
    """Check for banned identifiers in source files."""
    violations: list[dict[str, Any]] = []
    allow_list = load_allow_list()

    # Check all Python files, YAML, configs (not just src/)
    patterns_to_check = ["*.py", "*.yaml", "*.yml", "*.json", "*.toml"]

    for pattern in patterns_to_check:
        for file in Path(".").rglob(pattern):
            # Skip .git directory, venv, and excluded paths
            if ".git" in file.parts or "venv" in file.parts or ".venv" in file.parts:
                continue
            # Check if file path matches any excluded path
            file_str = str(file)
            if any(
                file_str == excl
                or file_str.endswith(f"/{excl}")
                or file_str.endswith(f"\\{excl}")
                for excl in EXCLUDED_PATHS
            ):
                continue

            try:
                content = file.read_text()
                normalized_content = normalize_text(content)
            except (UnicodeDecodeError, PermissionError):
                continue

            for banned_pattern in BANNED_IDENTIFIERS:
                # Apply pattern to normalized content
                matches = re.finditer(
                    banned_pattern, normalized_content, re.IGNORECASE | re.MULTILINE
                )
                for match in matches:
                    # Find the actual line in original content
                    line_num = normalized_content[: match.start()].count("\n") + 1
                    violation_id = f"{file}:{line_num}:{match.group()}"

                    if violation_id not in allow_list:
                        violations.append(
                            {
                                "type": "banned_identifier",
                                "file": str(file),
                                "line": line_num,
                                "pattern": banned_pattern,
                                "match": match.group(),
                                "message": f"Banned identifier pattern: {banned_pattern}",
                            }
                        )

    return violations


def check_media_files() -> list[dict[str, Any]]:
    """Check for committed media files and verify they're in the manifest."""
    violations: list[dict[str, Any]] = []
    allow_list = load_allow_list()
    manifest = load_manifest()

    # Build a set of allowed media file SHAs from manifest
    allowed_media_shas = {}
    for entry in manifest.get("files", []):
        if entry.get("sha256") and entry.get("use") in [
            "FX-CODE",
            "CODE_TEST",
            "SMOKE_ONLY",
        ]:
            allowed_media_shas[entry["sha256"]] = entry.get("path", "")

    # Check all media files
    for media_file in Path(".").rglob("*"):
        # Skip directories and .git
        if media_file.is_dir() or ".git" in media_file.parts:
            continue

        # Check if file extension matches (case-insensitive)
        if media_file.suffix.lower() not in MEDIA_EXTENSIONS:
            continue

        file_str = str(media_file)
        violation_id = f"media:{file_str}"

        # Skip if in allow-list
        if violation_id in allow_list:
            continue

        # Calculate SHA256 of the file
        import hashlib

        sha256_hash = hashlib.sha256()
        try:
            with open(media_file, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            file_sha = sha256_hash.hexdigest()
        except (OSError, PermissionError):
            continue

        # Check if SHA is in manifest with approved tag
        if file_sha not in allowed_media_shas:
            violations.append(
                {
                    "type": "media_file",
                    "file": file_str,
                    "extension": media_file.suffix,
                    "sha256": file_sha,
                    "message": f"Media file must be in manifest with approved tag: {file_str}",
                }
            )

    return violations


def main() -> int:
    """Run all project rules checks."""
    print("Running project rules checks...")

    all_violations: list[dict[str, Any]] = []

    # Run all checks
    all_violations.extend(check_imports_and_deps())
    all_violations.extend(check_identifiers())
    all_violations.extend(check_media_files())

    if not all_violations:
        print("✅ All project rules checks passed!")
        return 0

    # Report violations
    print(f"\n❌ Found {len(all_violations)} project rules violations:\n")

    for violation in all_violations:
        print(f"  {violation['type']}: {violation['file']}")
        if "line" in violation:
            print(f"    Line {violation['line']}: {violation['match']}")
        print(f"    {violation['message']}")
        print()

    # Write violations to JSON for artifact upload
    with open("project-rules-violations.json", "w") as f:
        json.dump(all_violations, f, indent=2)

    print("Wrote violations to project-rules-violations.json")
    print()
    print("To allow a false positive, add the violation ID to .project-rules-allow:")
    print("  Format: <file>:<line>:<match> or media:<file>")
    print("  Each allow-list entry must be reviewed and have a justification comment.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
