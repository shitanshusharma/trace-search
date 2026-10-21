"""Throughput measurement harness for AC-04.

Measures aggregate real-time factor across multiple video streams.
This measures DECODE-ONLY throughput (reading and decoding frames).
It does not include detection, tracking, or other processing.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .file_ingest import FileVideoSource
from .video_source import VideoSource


@dataclass
class StreamStats:
    """Statistics for a single stream."""

    camera_id: str
    total_frames: int = 0
    video_duration_seconds: float = 0.0
    wall_clock_seconds: float = 0.0
    real_time_factor: float = 0.0  # video_duration / wall_clock
    fps: float = 0.0


@dataclass
class ThroughputResult:
    """Result of throughput measurement across multiple streams."""

    streams: list[StreamStats] = field(default_factory=list)
    total_video_duration_seconds: float = 0.0
    total_wall_clock_seconds: float = 0.0
    aggregate_real_time_factor: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "streams": [
                {
                    "camera_id": s.camera_id,
                    "total_frames": s.total_frames,
                    "video_duration_seconds": s.video_duration_seconds,
                    "wall_clock_seconds": s.wall_clock_seconds,
                    "real_time_factor": s.real_time_factor,
                    "fps": s.fps,
                }
                for s in self.streams
            ],
            "total_video_duration_seconds": self.total_video_duration_seconds,
            "total_wall_clock_seconds": self.total_wall_clock_seconds,
            "aggregate_real_time_factor": self.aggregate_real_time_factor,
        }

    def to_json(self, path: Path) -> None:
        """Save result to JSON file.

        Args:
            path: Path to save JSON file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ThroughputHarness:
    """Harness for measuring video ingest throughput.

    Measures how fast the system can ingest video relative to real-time.
    A real-time factor of 1.0 means the system keeps up with live streams.
    AC-04 requires ≥1.0x aggregate real-time factor for 6 streams.
    """

    def __init__(self):
        self._results = ThroughputResult()

    def measure_stream(self, source: VideoSource) -> StreamStats:
        """Measure throughput for a single stream.

        Args:
            source: Video source to measure (must be opened)

        Returns:
            Statistics for the stream

        Raises:
            RuntimeError: If source not opened
        """
        if not source.is_open():
            raise RuntimeError(f"Video source {source.camera_id} not opened")

        stats = StreamStats(camera_id=source.camera_id, fps=source.get_fps())

        start_wall_clock = time.time()
        frame_count = 0

        for frame in source.frames():
            frame_count += 1

        end_wall_clock = time.time()

        stats.total_frames = frame_count
        stats.wall_clock_seconds = end_wall_clock - start_wall_clock
        stats.video_duration_seconds = frame_count / stats.fps if stats.fps > 0 else 0.0

        if stats.wall_clock_seconds > 0:
            stats.real_time_factor = stats.video_duration_seconds / stats.wall_clock_seconds

        return stats

    def measure_multiple_sequential(
        self,
        sources: list[VideoSource],
    ) -> ThroughputResult:
        """Measure throughput for multiple streams processed sequentially.

        This simulates a single-threaded ingest pipeline processing
        multiple camera files one after another.

        Args:
            sources: List of video sources (will be opened/closed automatically)

        Returns:
            Aggregate throughput result
        """
        result = ThroughputResult()
        total_video_time = 0.0
        total_wall_clock = 0.0

        for source in sources:
            with source:
                stats = self.measure_stream(source)
                result.streams.append(stats)
                total_video_time += stats.video_duration_seconds
                total_wall_clock += stats.wall_clock_seconds

        result.total_video_duration_seconds = total_video_time
        result.total_wall_clock_seconds = total_wall_clock

        if total_wall_clock > 0:
            result.aggregate_real_time_factor = total_video_time / total_wall_clock

        return result

    @staticmethod
    def measure_files(
        file_paths: list[Path],
        camera_ids: list[str],
    ) -> ThroughputResult:
        """Convenience method to measure throughput for file paths.

        Args:
            file_paths: List of video file paths
            camera_ids: List of camera IDs (must match file_paths length)

        Returns:
            Aggregate throughput result

        Raises:
            ValueError: If file_paths and camera_ids lengths don't match
        """
        if len(file_paths) != len(camera_ids):
            raise ValueError(
                f"file_paths length ({len(file_paths)}) must match "
                f"camera_ids length ({len(camera_ids)})"
            )

        sources = [
            FileVideoSource(camera_id=cid, file_path=path)
            for cid, path in zip(camera_ids, file_paths)
        ]

        harness = ThroughputHarness()
        return harness.measure_multiple_sequential(sources)
