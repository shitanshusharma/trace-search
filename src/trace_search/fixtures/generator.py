"""
Deterministic synthetic video generator for smoke testing.

Creates multi-camera-style clips with moving person-like figures, occlusion,
and varied lighting conditions. All outputs are deterministic given a seed.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image


@dataclass
class PersonFigure:
    """A simple person-like rectangular figure."""

    x: float
    y: float
    width: float
    height: float
    color: tuple[int, int, int]
    velocity_x: float
    velocity_y: float

    def update(self, width: int, height: int) -> None:
        """Update position with velocity."""
        self.x += self.velocity_x
        self.y += self.velocity_y

        if self.x < 0 or self.x + self.width > width:
            self.velocity_x *= -1
        if self.y < 0 or self.y + self.height > height:
            self.velocity_y *= -1

        self.x = max(0, min(self.x, width - self.width))
        self.y = max(0, min(self.y, height - self.height))


@dataclass
class Obstruction:
    """Static obstruction for occlusion effects."""

    x: int
    y: int
    width: int
    height: int
    color: tuple[int, int, int]


class SyntheticVideoGenerator:
    """Generates deterministic synthetic video clips for smoke testing."""

    def __init__(self, seed: int = 42):
        """
        Initialize generator with a seed for reproducibility.

        Args:
            seed: Random seed for deterministic generation
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_clip(
        self,
        output_path: Path,
        duration_sec: float = 10.0,
        fps: int = 30,
        width: int = 1920,
        height: int = 1080,
        num_figures: int = 3,
        background_brightness: float = 1.0,
        num_obstructions: int = 2,
    ) -> None:
        """
        Generate a synthetic video clip with moving figures.

        Args:
            output_path: Path to save the video file
            duration_sec: Duration in seconds
            fps: Frames per second
            width: Video width in pixels
            height: Video height in pixels
            num_figures: Number of moving person-like figures
            background_brightness: Brightness multiplier (0.0-1.0, lower = darker)
            num_obstructions: Number of static obstructions for occlusion
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        num_frames = int(duration_sec * fps)
        base_bg = int(40 * background_brightness)

        figures = self._create_figures(num_figures, width, height)
        obstructions = self._create_obstructions(num_obstructions, width, height)

        try:
            for _ in range(num_frames):
                frame = np.full((height, width, 3), base_bg, dtype=np.uint8)

                for figure in figures:
                    figure.update(width, height)
                    x, y = int(figure.x), int(figure.y)
                    w, h = int(figure.width), int(figure.height)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), figure.color, -1)
                    head_radius = w // 4
                    cv2.circle(frame, (x + w // 2, y + head_radius), head_radius, figure.color, -1)

                for obs in obstructions:
                    cv2.rectangle(
                        frame,
                        (obs.x, obs.y),
                        (obs.x + obs.width, obs.y + obs.height),
                        obs.color,
                        -1,
                    )

                writer.write(frame)
        finally:
            writer.release()

    def _create_figures(self, num_figures: int, width: int, height: int) -> list[PersonFigure]:
        """Create moving person-like figures."""
        figures = []
        for _ in range(num_figures):
            fig_width = self.rng.randint(40, 80)
            fig_height = int(fig_width * 2.5)
            x = self.rng.uniform(0, width - fig_width)
            y = self.rng.uniform(0, height - fig_height)

            red, green, blue = (int(channel) for channel in self.rng.randint(80, 200, 3))
            color = (red, green, blue)

            velocity_x = self.rng.uniform(-3, 3)
            velocity_y = self.rng.uniform(-3, 3)

            figures.append(
                PersonFigure(
                    x=x,
                    y=y,
                    width=fig_width,
                    height=fig_height,
                    color=color,
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                )
            )
        return figures

    def _create_obstructions(
        self, num_obstructions: int, width: int, height: int
    ) -> list[Obstruction]:
        """Create static obstructions for occlusion effects."""
        obstructions = []
        for _ in range(num_obstructions):
            obs_width = self.rng.randint(100, 250)
            obs_height = self.rng.randint(200, 400)
            x = self.rng.randint(0, width - obs_width)
            y = self.rng.randint(0, height - obs_height)

            red, green, blue = (int(channel) for channel in self.rng.randint(60, 120, 3))
            color = (red, green, blue)

            obstructions.append(
                Obstruction(x=x, y=y, width=obs_width, height=obs_height, color=color)
            )
        return obstructions

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def compute_phashes(video_path: Path, num_samples: int = 10) -> list[str]:
        """
        Compute perceptual hashes for sampled frames.

        Args:
            video_path: Path to video file
            num_samples: Number of frames to sample (default 10)

        Returns:
            List of 16-character hex pHash strings
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < num_samples:
                raise ValueError(
                    f"Video has {total_frames} frames but {num_samples} samples requested"
                )

            frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
            phashes = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Failed to read frame {idx}")

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                phash = str(imagehash.phash(pil_image))
                phashes.append(phash)

            return phashes
        finally:
            cap.release()


@dataclass(frozen=True)
class SmokeClipConfig:
    """One deterministic clip configuration."""

    clip_label: str
    duration_sec: float
    num_figures: int
    background_brightness: float
    num_obstructions: int


def manifest_row(
    entry: SmokeClipConfig,
    *,
    fixture_id: str,
    sha256: str,
    phashes: list[str],
    base_seed: int,
    offset: int,
) -> dict[str, Any]:
    """Build one manifest object. The schema key is the literal name field."""
    label = entry.clip_label
    stored_config = {
        "name": entry.clip_label,
        "duration_sec": entry.duration_sec,
        "num_figures": entry.num_figures,
        "background_brightness": entry.background_brightness,
        "num_obstructions": entry.num_obstructions,
    }
    return {
        "name": f"Synthetic smoke clip: {label}",
        "version": "1.0.0",
        "type": "synthetic_clip",
        "license": "MIT",
        "source_url": "https://github.com/shitanshusharma/trace-search",
        "sha256": sha256,
        "license_version": "N/A (synthetic)",
        "date_checked_ist": "2026-10-20",
        "phash": phashes,
        "use": "SMOKE_ONLY",
        "fixture_id": fixture_id,
        "path": f"smoke/{fixture_id}_{label}.mp4",
        "generator": "SyntheticVideoGenerator",
        "generator_version": "1.0.0",
        "generator_license": "MIT",
        "seed": json.dumps({"base_seed": base_seed, "offset": offset, "config": stored_config}),
        "notes": f"Deterministic synthetic clip for smoke testing. Config: {label}",
    }


def generate_smoke_fixtures(
    output_dir: Path,
    base_seed: int = 42,
    num_clips: int = 5,
) -> list[dict[str, Any]]:
    """
    Generate a full set of smoke test fixtures.

    Args:
        output_dir: Directory to store generated clips
        base_seed: Base random seed
        num_clips: Number of clips to generate (minimum 5)

    Returns:
        List of manifest entries for each generated clip
    """
    if num_clips < 5:
        raise ValueError("Must generate at least 5 clips")

    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        SmokeClipConfig(
            clip_label="corridor_normal",
            duration_sec=12.0,
            num_figures=3,
            background_brightness=1.0,
            num_obstructions=2,
        ),
        SmokeClipConfig(
            clip_label="entrance_lowlight",
            duration_sec=10.0,
            num_figures=2,
            background_brightness=0.3,
            num_obstructions=1,
        ),
        SmokeClipConfig(
            clip_label="hallway_crowded",
            duration_sec=15.0,
            num_figures=5,
            background_brightness=0.85,
            num_obstructions=3,
        ),
        SmokeClipConfig(
            clip_label="parking_sparse",
            duration_sec=8.0,
            num_figures=2,
            background_brightness=0.95,
            num_obstructions=1,
        ),
        SmokeClipConfig(
            clip_label="lobby_occlusion",
            duration_sec=14.0,
            num_figures=4,
            background_brightness=0.75,
            num_obstructions=4,
        ),
    ]

    manifest_entries: list[dict[str, Any]] = []

    for idx, config in enumerate(configs[:num_clips]):
        fixture_id = f"FX-SMOKE-SYN-{idx + 1:03d}"
        output_path = output_dir / f"{fixture_id}_{config.clip_label}.mp4"

        generator = SyntheticVideoGenerator(seed=base_seed + idx)

        generator.generate_clip(
            output_path=output_path,
            duration_sec=config.duration_sec,
            fps=30,
            width=1920,
            height=1080,
            num_figures=config.num_figures,
            background_brightness=config.background_brightness,
            num_obstructions=config.num_obstructions,
        )

        sha256 = generator.compute_sha256(output_path)
        phashes = generator.compute_phashes(output_path, num_samples=10)
        manifest_entries.append(
            manifest_row(
                config,
                fixture_id=fixture_id,
                sha256=sha256,
                phashes=phashes,
                base_seed=base_seed,
                offset=idx,
            )
        )

        print(f"Generated {fixture_id}: {output_path.name}")
        print(f"  SHA-256: {sha256}")
        print(f"  Duration: {config.duration_sec}s, {config.num_figures} figures")

    return manifest_entries
