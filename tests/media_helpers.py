"""Shared helpers for fixture media tests."""

from pathlib import Path

import cv2


def write_mp4v_variant(source: Path, dest: Path, scale: float = 0.5) -> None:
    """Re-encode with mp4v and a resize so the SHA-256 changes."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_w = max(2, int(width * scale) // 2 * 2)
    new_h = max(2, int(height * scale) // 2 * 2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_w, new_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open mp4v writer for {dest}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            resized = cv2.resize(frame, (new_w, new_h))
            bright = cv2.convertScaleAbs(resized, alpha=1.08, beta=2)
            writer.write(bright)
    finally:
        cap.release()
        writer.release()
