"""Content identity for smoke fixtures: SHA-256 and sampled-frame pHash.

Callers must not decide by filename. A renamed ``.dat`` copy is still a video
when its bytes are a video, and a decode or pHash failure is an error.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

PHASH_SAMPLES = 10
HAMMING_THRESHOLD = 8

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
# Extension-agnostic containers. Always attempt a decode; failure is an error.
OPAQUE_EXTENSIONS = {".dat", ".bin"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS | OPAQUE_EXTENSIONS


class ContentCheckError(Exception):
    """A file could not be hashed, decoded, or perceptually hashed."""


def sha256_file(path: Path) -> str:
    """SHA-256 of file bytes. Raises ContentCheckError on I/O failure."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise ContentCheckError(f"SHA-256 failed for {path}: {exc}") from exc


def hamming_distance(hash1: str, hash2: str) -> int:
    """Bit distance between two equal-length hex hashes."""
    if len(hash1) != len(hash2):
        raise ContentCheckError(
            f"pHash length mismatch: {len(hash1)} != {len(hash2)}"
        )
    try:
        return (int(hash1, 16) ^ int(hash2, 16)).bit_count()
    except ValueError as exc:
        raise ContentCheckError(f"pHash is not hex: {exc}") from exc


def nearest_hamming(left: list[str], right: list[str]) -> int | None:
    """Minimum Hamming distance, or None when either list is empty."""
    if not left or not right:
        return None
    best = None
    for a in left:
        for b in right:
            distance = hamming_distance(a, b)
            if best is None or distance < best:
                best = distance
    return best


def _prefix(path: Path, size: int = 64) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(size)
    except OSError as exc:
        raise ContentCheckError(f"cannot read {path}: {exc}") from exc


def looks_like_video(prefix: bytes) -> bool:
    if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
        return True
    if prefix.startswith(b"RIFF") and b"AVI " in prefix[:16]:
        return True
    if prefix.startswith(b"\x1a\x45\xdf\xa3"):
        return True
    return False


def looks_like_image(prefix: bytes) -> bool:
    return (
        prefix.startswith(b"\xff\xd8\xff")
        or prefix.startswith(b"\x89PNG\r\n\x1a\n")
        or prefix.startswith(b"GIF87a")
        or prefix.startswith(b"GIF89a")
        or prefix.startswith(b"BM")
        or prefix.startswith(b"II*\x00")
        or prefix.startswith(b"MM\x00*")
    )


def needs_perceptual_check(path: Path, prefix: bytes) -> bool:
    suffix = path.suffix.lower()
    if suffix in MEDIA_EXTENSIONS:
        return True
    return looks_like_video(prefix) or looks_like_image(prefix)


def _phash_image(image: Image.Image) -> str:
    try:
        digest = str(imagehash.phash(image))
    except Exception as exc:
        raise ContentCheckError(f"pHash failed: {exc}") from exc
    if not digest:
        raise ContentCheckError("pHash failed: empty digest")
    return digest


def _phashes_from_capture(cap: cv2.VideoCapture, label: str) -> list[str]:
    if cap is None or not cap.isOpened():
        raise ContentCheckError(f"decode failed for {label}: cannot open video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ContentCheckError(f"decode failed for {label}: video has no frames")
    samples = min(PHASH_SAMPLES, total)
    indices = np.linspace(0, total - 1, samples, dtype=int)
    hashes: list[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ContentCheckError(f"decode failed for {label}: cannot read frame {idx}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hashes.append(_phash_image(Image.fromarray(rgb)))
    if not hashes:
        raise ContentCheckError(f"decode failed for {label}: no frames hashed")
    return hashes


def _video_phashes(path: Path) -> list[str]:
    cap = cv2.VideoCapture(str(path))
    try:
        return _phashes_from_capture(cap, str(path))
    finally:
        if cap is not None:
            cap.release()


def _video_phashes_via_temp(path: Path) -> list[str]:
    """Decode when the suffix hides an mp4/avi payload (for example ``.dat``)."""
    fd, name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    temp = Path(name)
    try:
        temp.write_bytes(path.read_bytes())
        return _video_phashes(temp)
    except OSError as exc:
        raise ContentCheckError(f"decode failed for {path}: {exc}") from exc
    finally:
        temp.unlink(missing_ok=True)


def _image_phashes(path: Path) -> list[str]:
    try:
        with Image.open(path) as image:
            frame = image.convert("RGB")
            return [_phash_image(frame)]
    except ContentCheckError:
        raise
    except Exception as exc:
        raise ContentCheckError(f"decode failed for {path}: {exc}") from exc


def perceptual_hashes(path: Path) -> list[str]:
    """pHashes for a video (10 sampled frames) or one image.

    Returns an empty list when the file is not media. Raises ContentCheckError
    when the file looks like media but cannot be decoded or hashed.
    """
    prefix = _prefix(path)
    if not needs_perceptual_check(path, prefix):
        return []

    suffix = path.suffix.lower()
    errors: list[Exception] = []
    try_video = (
        suffix in VIDEO_EXTENSIONS
        or suffix in OPAQUE_EXTENSIONS
        or looks_like_video(prefix)
    )
    try_image = (
        suffix in IMAGE_EXTENSIONS
        or suffix in OPAQUE_EXTENSIONS
        or looks_like_image(prefix)
    )

    if try_video:
        try:
            return _video_phashes(path)
        except ContentCheckError as exc:
            errors.append(exc)
        if looks_like_video(prefix) or suffix in OPAQUE_EXTENSIONS or suffix in VIDEO_EXTENSIONS:
            try:
                return _video_phashes_via_temp(path)
            except ContentCheckError as exc:
                errors.append(exc)

    if try_image:
        try:
            return _image_phashes(path)
        except ContentCheckError as exc:
            errors.append(exc)

    detail = errors[-1] if errors else "not a decodable image or video"
    kind = "pHash failed" if "pHash failed" in str(detail) else "decode failed"
    raise ContentCheckError(f"{kind} for {path}: {detail}")
