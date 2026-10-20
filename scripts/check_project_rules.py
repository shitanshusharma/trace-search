#!/usr/bin/env python3
"""
Project rules checker for trace-search.

Enforces PRD hard-stops:
1. No face-recognition, face embedding/crop, gait, watchlist, enrolment or named-identity
   code/deps (e.g., imports/deps like face_recognition, insightface, arcface, retinaface,
   dlib face models, deepface, facenet, mediapipe face, opengait, ultralytics, boxmot;
   identifiers like watchlist/enrol/face_embedding/gait; API schema fields like
   name/national_id)
2. No media files (video/image/weights) committed outside approved fixtures paths/allow-list
3. Explicit allow-mechanism for false positives (needs reviewed comment)

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

# Banned identifiers in code (case-insensitive patterns)
BANNED_IDENTIFIERS = [
    r"\bwatchlist\b",
    r"\benrol(?:l)?(?:ment)?\b",
    r"\bface_embedding\b",
    r"\bface_crop\b",
    r"\bgait(?:_|[A-Z])",
    r"\bname\s*[:=]\s*['\"]",  # name field assignments
    r"\bnational_id\b",
    r"\bidentity_id\b",
]

# Media file extensions that should not be committed
MEDIA_EXTENSIONS = [
    # Video (case-insensitive matching below)
    ".mp4", ".mkv", ".avi", ".mov", ".h264", ".h265", ".dav", ".webm", ".flv",
    # Image
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    # Model weights
    ".pt", ".pth", ".onnx", ".safetensors", ".h5", ".pb", ".tflite", ".weights",
]

# Paths where media files are explicitly allowed
MEDIA_ALLOW_PATHS = [
    "fixtures/",
    "tests/fixtures/",
    "docs/diagrams/",  # Allow docs diagrams
    "docs/images/",
]

# Allow-list file (lines starting with # are comments)
ALLOW_LIST_FILE = ".project-rules-allow"


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


def check_imports_and_deps() -> list[dict[str, Any]]:
    """Check for banned imports in Python files and dependencies."""
    violations: list[dict[str, Any]] = []
    allow_list = load_allow_list()
    
    # Check Python source files
    for py_file in Path("src").rglob("*.py"):
        content = py_file.read_text()
        
        for pattern in BANNED_IMPORTS:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count("\n") + 1
                violation_id = f"{py_file}:{line_num}:{match.group()}"
                
                if violation_id not in allow_list:
                    violations.append({
                        "type": "banned_import",
                        "file": str(py_file),
                        "line": line_num,
                        "pattern": pattern,
                        "match": match.group(),
                        "message": f"Banned import/dependency pattern: {pattern}"
                    })
    
    # Check requirements files
    req_files = ["requirements.txt", "requirements-dev.txt", "requirements-ci.txt"]
    for req_file in req_files:
        req_path = Path(req_file)
        if not req_path.exists():
            continue
        
        content = req_path.read_text()
        for pattern in BANNED_IMPORTS:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count("\n") + 1
                violation_id = f"{req_file}:{line_num}:{match.group()}"
                
                if violation_id not in allow_list:
                    violations.append({
                        "type": "banned_dependency",
                        "file": str(req_path),
                        "line": line_num,
                        "pattern": pattern,
                        "match": match.group(),
                        "message": f"Banned dependency pattern: {pattern}"
                    })
    
    # Check pyproject.toml
    pyproject = Path("pyproject.toml")
    if pyproject.exists():
        content = pyproject.read_text()
        for pattern in BANNED_IMPORTS:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count("\n") + 1
                violation_id = f"pyproject.toml:{line_num}:{match.group()}"
                
                if violation_id not in allow_list:
                    violations.append({
                        "type": "banned_dependency",
                        "file": str(pyproject),
                        "line": line_num,
                        "pattern": pattern,
                        "match": match.group(),
                        "message": f"Banned dependency in pyproject.toml: {pattern}"
                    })
    
    return violations


def check_identifiers() -> list[dict[str, Any]]:
    """Check for banned identifiers in Python source files."""
    violations: list[dict[str, Any]] = []
    allow_list = load_allow_list()
    
    for py_file in Path("src").rglob("*.py"):
        content = py_file.read_text()
        
        for pattern in BANNED_IDENTIFIERS:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count("\n") + 1
                violation_id = f"{py_file}:{line_num}:{match.group()}"
                
                if violation_id not in allow_list:
                    violations.append({
                        "type": "banned_identifier",
                        "file": str(py_file),
                        "line": line_num,
                        "pattern": pattern,
                        "match": match.group(),
                        "message": f"Banned identifier pattern: {pattern}"
                    })
    
    return violations


def check_media_files() -> list[dict[str, Any]]:
    """Check for committed media files outside allow-listed paths."""
    violations: list[dict[str, Any]] = []
    allow_list = load_allow_list()
    
    for ext in MEDIA_EXTENSIONS:
        # Search case-insensitively by checking both lower and upper variants
        for pattern in [f"*{ext}", f"*{ext.upper()}"]:
            for media_file in Path(".").rglob(pattern):
                # Skip .git directory
                if ".git" in media_file.parts:
                    continue
                
                # Check if file is in an allowed path
                file_str = str(media_file)
                allowed = any(file_str.startswith(allowed_path) 
                             for allowed_path in MEDIA_ALLOW_PATHS)
                
                violation_id = f"media:{file_str}"
                
                if not allowed and violation_id not in allow_list:
                    violations.append({
                        "type": "media_file",
                        "file": file_str,
                        "extension": media_file.suffix,
                        "message": f"Media file outside allowed paths: {file_str}"
                    })
    
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
    
    print(f"Wrote violations to project-rules-violations.json")
    print()
    print("To allow a false positive, add the violation ID to .project-rules-allow:")
    print("  Format: <file>:<line>:<match> or media:<file>")
    print("  Each allow-list entry must be reviewed and have a justification comment.")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
