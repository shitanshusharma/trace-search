"""Tests for video ingest module (T2).

Generates synthetic test videos to avoid dependency on T1 fixtures.
"""

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from trace_search.ingest import (
    CameraConfig,
    CameraRegistry,
    FileVideoSource,
    RTSPVideoSource,
    ThroughputHarness,
    estimate_offset_from_multiple_events,
    estimate_offset_from_sync_event,
)


def generate_test_video(
    path: Path,
    width: int = 640,
    height: int = 480,
    fps: float = 25.0,
    duration_seconds: float = 1.0,
    color_sequence: list = None,
) -> None:
    """Generate a synthetic test video.

    Args:
        path: Output path for video file
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        duration_seconds: Video duration
        color_sequence: List of (B, G, R) tuples for frame colors.
            If None, generates gradient frames.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    frame_count = int(fps * duration_seconds)

    for i in range(frame_count):
        if color_sequence:
            color = color_sequence[i % len(color_sequence)]
            frame = np.full((height, width, 3), color, dtype=np.uint8)
        else:
            # Generate gradient based on frame number
            value = int((i / frame_count) * 255)
            frame = np.full((height, width, 3), value, dtype=np.uint8)

        writer.write(frame)

    writer.release()


class TestCameraRegistry:
    """Tests for camera registry."""

    def test_camera_config_creation(self):
        """Test basic camera config creation."""
        cam = CameraConfig(
            camera_id="cam1",
            brand="hikvision",
            site_map_x=10.0,
            site_map_y=20.0,
            clock_offset_seconds=5.0,
        )

        assert cam.camera_id == "cam1"
        assert cam.brand == "hikvision"
        assert cam.clock_offset_seconds == 5.0

    def test_camera_config_validation(self):
        """Test camera config validation."""
        with pytest.raises(ValueError, match="camera_id cannot be empty"):
            CameraConfig(camera_id="", brand="hikvision", site_map_x=0, site_map_y=0)

        with pytest.raises(ValueError, match="brand cannot be empty"):
            CameraConfig(camera_id="cam1", brand="", site_map_x=0, site_map_y=0)

    def test_apply_offset(self):
        """Test clock offset correction."""
        cam = CameraConfig(
            camera_id="cam1",
            brand="hikvision",
            site_map_x=0,
            site_map_y=0,
            clock_offset_seconds=5.0,
        )

        raw_timestamp = 1000.0
        corrected = cam.apply_offset(raw_timestamp)
        assert corrected == 1005.0

    def test_registry_operations(self):
        """Test camera registry operations."""
        registry = CameraRegistry()

        cam1 = CameraConfig("cam1", "hikvision", 0, 0)
        cam2 = CameraConfig("cam2", "cpplus", 10, 20)

        registry.register(cam1)
        registry.register(cam2)

        assert registry.count() == 2
        assert registry.get("cam1") == cam1
        assert registry.get("cam2") == cam2
        assert registry.get("cam3") is None

    def test_registry_duplicate_error(self):
        """Test that registering duplicate camera fails."""
        registry = CameraRegistry()
        cam = CameraConfig("cam1", "hikvision", 0, 0)

        registry.register(cam)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(cam)

    def test_registry_save_load(self, tmp_path):
        """Test saving and loading registry."""
        registry = CameraRegistry()

        for i in range(6):
            cam = CameraConfig(
                camera_id=f"cam{i + 1}",
                brand="hikvision" if i < 3 else "cpplus",
                site_map_x=i * 10.0,
                site_map_y=i * 5.0,
                clock_offset_seconds=i * 0.5,
            )
            registry.register(cam)

        save_path = tmp_path / "cameras.json"
        registry.save(save_path)

        assert save_path.exists()

        loaded = CameraRegistry.load(save_path)
        assert loaded.count() == 6

        for i in range(6):
            cam_id = f"cam{i + 1}"
            original = registry.get(cam_id)
            loaded_cam = loaded.get(cam_id)

            assert loaded_cam is not None
            assert loaded_cam.camera_id == original.camera_id
            assert loaded_cam.brand == original.brand
            assert loaded_cam.clock_offset_seconds == original.clock_offset_seconds


class TestClockSync:
    """Tests for clock synchronization."""

    def test_single_sync_event(self):
        """Test offset estimation from single sync event."""
        # At wall-clock time T, cameras report:
        timestamps = {
            "cam1": 1000.0,  # Reference
            "cam2": 1005.0,  # 5 seconds fast
            "cam3": 998.0,  # 2 seconds slow
        }

        offsets = estimate_offset_from_sync_event(timestamps, "cam1")

        assert offsets["cam1"] == 0.0
        assert offsets["cam2"] == -5.0  # Need to subtract 5
        assert offsets["cam3"] == 2.0  # Need to add 2

    def test_sync_event_missing_reference(self):
        """Test sync event with missing reference camera."""
        timestamps = {"cam1": 1000.0, "cam2": 1005.0}

        with pytest.raises(ValueError, match="Reference camera cam3 not in timestamps"):
            estimate_offset_from_sync_event(timestamps, "cam3")

    def test_multiple_sync_events(self):
        """Test offset estimation from multiple sync events."""
        # Three sync events with slight variations
        events = [
            {"cam1": 1000.0, "cam2": 1005.0, "cam3": 998.0},
            {"cam1": 2000.0, "cam2": 2005.1, "cam3": 1998.1},
            {"cam1": 3000.0, "cam2": 3004.9, "cam3": 1997.9},
        ]

        offsets = estimate_offset_from_multiple_events(events, "cam1")

        assert offsets["cam1"] == 0.0
        assert abs(offsets["cam2"] - (-5.0)) < 0.1  # Median ~-5.0
        assert abs(offsets["cam3"] - 2.0) < 0.1  # Median ~2.0

    def test_multiple_events_empty(self):
        """Test multiple events with empty list."""
        with pytest.raises(ValueError, match="At least one sync event required"):
            estimate_offset_from_multiple_events([], "cam1")


class TestFileIngest:
    """Tests for file-based video ingestion."""

    def test_basic_file_ingest(self, tmp_path):
        """Test basic file ingestion with synthetic video."""
        video_path = tmp_path / "test.mp4"
        generate_test_video(video_path, duration_seconds=1.0, fps=25.0)

        source = FileVideoSource(
            camera_id="cam1",
            file_path=video_path,
            start_timestamp=1000.0,
        )

        with source:
            assert source.is_open()
            assert source.get_fps() == 25.0

            frames = list(source.frames())
            assert len(frames) >= 20  # Should have ~25 frames

            # Check first frame
            first = frames[0]
            assert first.camera_id == "cam1"
            assert first.frame_number == 0
            assert first.timestamp_seconds == 1000.0
            assert first.image.shape[2] == 3  # BGR

            # Check last frame has later timestamp
            last = frames[-1]
            assert last.timestamp_seconds > first.timestamp_seconds

    def test_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            FileVideoSource("cam1", Path("/nonexistent.mp4"))

    def test_corrupted_file(self, tmp_path):
        """Test error handling for corrupted file."""
        bad_file = tmp_path / "corrupted.mp4"
        bad_file.write_text("not a video file")

        source = FileVideoSource("cam1", bad_file)

        with pytest.raises(IOError, match="Failed to open video file"):
            source.open()

    def test_clock_offset_correction(self, tmp_path):
        """Test that clock offset correction works correctly."""
        video_path = tmp_path / "test.mp4"
        generate_test_video(video_path, duration_seconds=1.0, fps=10.0)

        source = FileVideoSource(
            camera_id="cam1",
            file_path=video_path,
            start_timestamp=1000.0,
        )

        clock_offset = 5.0

        with source:
            frames = list(source.frames())

            # Frame at known position should land at corrected time
            # First frame: 1000.0 + 0.0/10 = 1000.0
            # With offset +5.0 -> 1005.0
            assert abs(frames[0].corrected_timestamp(clock_offset) - 1005.0) < 0.1

            # 5th frame: 1000.0 + 5.0/10 = 1000.5
            # With offset +5.0 -> 1005.5
            if len(frames) > 5:
                assert abs(frames[5].corrected_timestamp(clock_offset) - 1005.5) < 0.1

    def test_six_cameras_with_offsets(self, tmp_path):
        """Test ingesting 6 cameras with different clock offsets (AC-03 proof).

        This test is NOT vacuous. It verifies that:
        1. All 6 cameras are tested (count == 6)
        2. Each camera's corrected timestamp lands within 1s of sync event
        3. Offset estimation from sync events works correctly
        """
        # Known true offsets (what we'll apply to camera clocks)
        true_offsets = [0.0, -2.3, 1.7, -0.8, 3.2, -1.5]

        # Video parameters
        base_start = 1000.0  # All videos start at this timestamp
        video_duration = 10.0  # seconds
        video_fps = 10.0  # fps

        # Wall-clock sync event time (must be within video duration)
        # Place sync event at 5 seconds into the video
        sync_wall_clock = base_start + 5.0  # 1005.0

        # Create registry with 6 cameras
        registry = CameraRegistry()
        camera_ids = []

        for i in range(6):
            cam_id = f"cam{i + 1}"
            camera_ids.append(cam_id)
            cam = CameraConfig(
                camera_id=cam_id,
                brand="hikvision" if i < 3 else "cpplus",
                site_map_x=i * 10.0,
                site_map_y=i * 5.0,
                clock_offset_seconds=true_offsets[i],
            )
            registry.register(cam)

        # Generate 6 test videos
        video_paths = []

        for i in range(6):
            video_path = tmp_path / f"cam{i + 1}.mp4"
            generate_test_video(video_path, duration_seconds=video_duration, fps=video_fps)
            video_paths.append(video_path)

        # Simulate sync event: each camera sees it but reports different time
        # Camera's reported time = wall_clock - true_offset
        # (If camera is 2s fast, true_offset=-2, camera reports wall_clock + 2)
        camera_reported_sync_times = {}
        for i, cam_id in enumerate(camera_ids):
            camera_reported_sync_times[cam_id] = sync_wall_clock - true_offsets[i]

        # Use sync event to estimate offsets
        estimated_offsets = estimate_offset_from_sync_event(camera_reported_sync_times, "cam1")

        # Test: ingest all 6 videos and verify corrected timestamps
        cameras_tested = 0
        errors = []

        for i, video_path in enumerate(video_paths):
            cam_id = camera_ids[i]
            cam = registry.get(cam_id)

            # Each camera's video starts at base_start
            # The sync event occurs at sync_wall_clock
            # So the sync frame is at position: (sync_wall_clock - base_start) seconds
            source = FileVideoSource(
                camera_id=cam_id,
                file_path=video_path,
                start_timestamp=base_start,
            )

            with source:
                frames = list(source.frames())
                expected_frames = int(video_duration * video_fps)
                assert len(frames) >= expected_frames * 0.9, (
                    f"{cam_id} should have ~{expected_frames} frames, got {len(frames)}"
                )

                # Compute which frame should contain the sync event
                # Camera's reported time = sync_wall_clock - true_offset
                # Seconds into video = (camera_reported_time) - base_start
                cam_reported_time = sync_wall_clock - true_offsets[i]
                seconds_into_video = cam_reported_time - base_start
                sync_frame_idx = int(seconds_into_video * video_fps)

                # Guard: verify frame index is in range
                assert 0 <= sync_frame_idx < len(frames), (
                    f"{cam_id} sync frame index {sync_frame_idx} out of range "
                    f"[0, {len(frames)}). Test construction error!"
                )

                # Get the sync frame and apply estimated offset
                sync_frame = frames[sync_frame_idx]
                corrected_time = sync_frame.corrected_timestamp(estimated_offsets[cam_id])

                # AC-03 requirement: ≤1s error after correction
                error = abs(corrected_time - sync_wall_clock)
                errors.append((cam_id, error))

                print(f"{cam_id}: frame {sync_frame_idx}/{len(frames)}, error={error:.3f}s")

                cameras_tested += 1

        # Guard: all 6 cameras must have been tested
        assert cameras_tested == 6, (
            f"Test is vacuous: only {cameras_tested}/6 cameras tested. "
            f"All cameras must be verified."
        )

        # AC-03 assertion: all errors ≤ 1s
        failed = [(cam_id, err) for cam_id, err in errors if err > 1.0]
        assert not failed, (
            f"AC-03 FAILED: {len(failed)}/6 cameras exceeded 1s error after correction: {failed}"
        )

        max_error = max(err for _, err in errors)
        print(f"AC-03 PASSED: All 6 cameras tested, max error = {max_error:.3f}s")


class TestRTSPIngest:
    """Tests for RTSP stream ingestion."""

    def test_rtsp_source_creation(self):
        """Test RTSP source creation (no actual connection)."""
        source = RTSPVideoSource(
            camera_id="cam1",
            rtsp_url="rtsp://example.com/stream",
        )

        assert source.camera_id == "cam1"
        assert source.rtsp_url == "rtsp://example.com/stream"
        assert not source.is_open()

    def test_rtsp_fails_gracefully(self):
        """Test that RTSP connection failure is handled gracefully.

        cv2.VideoCapture is mocked so the test never reaches the network.
        """
        source = RTSPVideoSource(
            camera_id="cam1",
            rtsp_url="rtsp://nonexistent.local/stream",
            reconnect_attempts=1,
            reconnect_delay_seconds=0.1,
        )

        with patch("trace_search.ingest.rtsp_ingest.cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = False
            with pytest.raises(OSError, match="Failed to open RTSP stream"):
                source.open()

    def test_rtsp_credential_redaction(self):
        """Test that RTSP URLs have credentials redacted in errors."""
        from trace_search.ingest import redact_rtsp_credentials

        # Test with credentials
        url_with_creds = "rtsp://user:password@192.168.1.100:554/stream"
        redacted = redact_rtsp_credentials(url_with_creds)
        assert redacted == "rtsp://***@192.168.1.100:554/stream"
        assert "user" not in redacted
        assert "password" not in redacted

        # Test without credentials
        url_without_creds = "rtsp://192.168.1.100:554/stream"
        redacted = redact_rtsp_credentials(url_without_creds)
        assert redacted == url_without_creds

        # Test with only username
        url_user_only = "rtsp://admin@192.168.1.100:554/stream"
        redacted = redact_rtsp_credentials(url_user_only)
        assert redacted == "rtsp://***@192.168.1.100:554/stream"
        assert "admin" not in redacted

        # Test password containing '@'
        url_at_in_pass = "rtsp://user:p@ss@word@192.168.1.100/stream"
        redacted = redact_rtsp_credentials(url_at_in_pass)
        assert redacted == "rtsp://***@192.168.1.100/stream"
        assert "p@ss@word" not in redacted
        assert "user" not in redacted

        # Test uppercase RTSP
        url_upper = "RTSP://admin:secret@camera.local/live"
        redacted = redact_rtsp_credentials(url_upper)
        assert redacted == "RTSP://***@camera.local/live"
        assert "admin" not in redacted
        assert "secret" not in redacted

        # Test rtsps:// (secure)
        url_secure = "rtsps://user:pass123@secure.cam:8554/stream"
        redacted = redact_rtsp_credentials(url_secure)
        assert redacted == "rtsps://***@secure.cam:8554/stream"
        assert "user" not in redacted
        assert "pass123" not in redacted

        # Test mixed case rtsps
        url_mixed = "RtspS://admin:pwd@cam/s"
        redacted = redact_rtsp_credentials(url_mixed)
        assert redacted == "RtspS://***@cam/s"
        assert "admin" not in redacted
        assert "pwd" not in redacted

    def test_rtsp_redacts_hash_in_password(self):
        """'#' inside a password is userinfo, not a URL fragment."""
        from trace_search.ingest import redact_rtsp_credentials

        raw = "rtsp://u:pw#frag@h/s"
        redacted = redact_rtsp_credentials(raw)
        assert redacted == "rtsp://***@h/s"
        assert "pw#frag" not in redacted
        assert "u:" not in redacted

    def test_rtsp_redacts_question_mark_in_password(self):
        """'?' inside a password is userinfo, not a query string."""
        from trace_search.ingest import redact_rtsp_credentials

        raw = "rtsp://u:pw?x@h/s"
        redacted = redact_rtsp_credentials(raw)
        assert redacted == "rtsp://***@h/s"
        assert "pw?x" not in redacted
        assert "u:" not in redacted

    def test_rtsp_redacts_slash_in_password(self):
        """'/' inside a password stays in userinfo up to the last '@'."""
        from trace_search.ingest import redact_rtsp_credentials

        raw = "rtsp://u:p@ss/word@host/s"
        redacted = redact_rtsp_credentials(raw)
        assert redacted == "rtsp://***@host/s"
        assert "ss/word" not in redacted
        assert "p@ss" not in redacted

    def test_rtsp_redacts_space_in_password(self):
        """A space inside a password is redacted with the rest of userinfo."""
        from trace_search.ingest import redact_rtsp_credentials

        raw = "rtsp://u:pa ss@h/s"
        redacted = redact_rtsp_credentials(raw)
        assert redacted == "rtsp://***@h/s"
        assert "pa ss" not in redacted
        assert "u:" not in redacted

    def test_rtsp_redacts_password_with_two_or_more_spaces(self):
        """A password with two or more spaces is redacted, not left in the text."""
        from trace_search.ingest import redact_rtsp_credentials

        raw = "rtsp://u:a b c@h/s"
        redacted = redact_rtsp_credentials(raw)
        assert redacted == "rtsp://***@h/s"
        assert "a b c" not in redacted
        assert "u:" not in redacted

    def test_rtsp_adjacent_urls_do_not_merge(self):
        """Two adjacent RTSP URLs stay two redacted URLs."""
        from trace_search.ingest import redact_rtsp_credentials

        raw = "rtsp://u:p@h/s rtsp://x:y@h2/s"
        redacted = redact_rtsp_credentials(raw)
        assert redacted == "rtsp://***@h/s rtsp://***@h2/s"
        assert "u:p" not in redacted
        assert "x:y" not in redacted

    def test_rtsp_repr_redacts_credentials(self):
        """Test that RTSP source repr redacts credentials."""
        source = RTSPVideoSource(
            camera_id="cam1",
            rtsp_url="rtsp://admin:secret123@camera.local/stream",
        )

        repr_str = repr(source)
        assert "admin" not in repr_str
        assert "secret123" not in repr_str
        assert "***" in repr_str
        assert "camera.local" in repr_str

    def test_rtsp_error_redacts_credentials(self):
        """Test that RTSP connection errors redact credentials."""
        source = RTSPVideoSource(
            camera_id="cam1",
            rtsp_url="rtsp://user:pass@nonexistent.local/stream",
            reconnect_attempts=1,
            reconnect_delay_seconds=0.01,
        )

        with patch("trace_search.ingest.rtsp_ingest.cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = False
            try:
                source.open()
            except OSError as e:
                error_msg = str(e)
                assert "user" not in error_msg
                assert "pass" not in error_msg
                assert "***" in error_msg
                assert "nonexistent.local" in error_msg
                assert "user" not in source._last_error
                assert "pass" not in source._last_error
            else:
                pytest.fail("Should have raised OSError")

    def test_rtsp_exception_message_containing_url_is_redacted(self, caplog):
        """OpenCV/FFmpeg errors that echo the URL must be redacted in last_error and logs."""
        caplog.set_level("WARNING", logger="trace_search.ingest.rtsp_ingest")
        raw_url = "rtsps://admin:p@ss@word@camera.local:8554/live"
        source = RTSPVideoSource(
            camera_id="cam1",
            rtsp_url=raw_url,
            reconnect_attempts=1,
            reconnect_delay_seconds=0.01,
        )

        with patch("trace_search.ingest.rtsp_ingest.cv2.VideoCapture") as mock_cap:
            mock_cap.side_effect = RuntimeError(f"Failed to open {raw_url}")
            with pytest.raises(OSError) as exc_info:
                source.open()

        error_msg = str(exc_info.value)
        assert "admin" not in error_msg
        assert "p@ss@word" not in error_msg
        assert "p@ss" not in error_msg
        assert "***" in error_msg
        assert "camera.local" in error_msg
        assert "admin" not in source._last_error
        assert "p@ss@word" not in source._last_error
        assert "admin" not in caplog.text
        assert "p@ss@word" not in caplog.text


class TestThroughput:
    """Tests for throughput measurement harness."""

    def test_throughput_single_stream(self, tmp_path):
        """Test throughput measurement for single stream."""
        video_path = tmp_path / "test.mp4"
        generate_test_video(video_path, duration_seconds=2.0, fps=25.0)

        harness = ThroughputHarness()

        source = FileVideoSource("cam1", video_path)
        with source:
            stats = harness.measure_stream(source)

        assert stats.camera_id == "cam1"
        assert stats.total_frames >= 40  # ~50 frames for 2s @ 25fps
        assert stats.video_duration_seconds > 1.5
        assert stats.real_time_factor > 0  # Should process faster than real-time

    def test_throughput_multiple_streams(self, tmp_path):
        """Test aggregate throughput for multiple streams."""
        video_paths = []
        camera_ids = []

        for i in range(6):
            video_path = tmp_path / f"cam{i + 1}.mp4"
            generate_test_video(video_path, duration_seconds=1.0, fps=10.0)
            video_paths.append(video_path)
            camera_ids.append(f"cam{i + 1}")

        result = ThroughputHarness.measure_files(video_paths, camera_ids)

        assert len(result.streams) == 6
        assert result.total_video_duration_seconds > 5.0  # ~6 seconds total
        assert result.aggregate_real_time_factor > 0

        # Each stream should be measured
        for i, stats in enumerate(result.streams):
            assert stats.camera_id == f"cam{i + 1}"
            assert stats.total_frames > 0

    def test_throughput_json_export(self, tmp_path):
        """Test JSON export of throughput results."""
        video_path = tmp_path / "test.mp4"
        generate_test_video(video_path, duration_seconds=1.0, fps=25.0)

        result = ThroughputHarness.measure_files([video_path], ["cam1"])

        json_path = tmp_path / "throughput.json"
        result.to_json(json_path)

        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert "streams" in data
        assert "aggregate_real_time_factor" in data
        assert len(data["streams"]) == 1
        assert data["streams"][0]["camera_id"] == "cam1"

    def test_throughput_mismatch_length(self, tmp_path):
        """Test error when file paths and camera IDs lengths don't match."""
        video_path = tmp_path / "test.mp4"
        generate_test_video(video_path, duration_seconds=1.0, fps=25.0)

        with pytest.raises(ValueError, match="length.*must match"):
            ThroughputHarness.measure_files([video_path], ["cam1", "cam2"])


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_six_cameras(self, tmp_path):
        """Integration test: 6 cameras, registry, ingest, offset correction."""
        # Set up registry
        registry = CameraRegistry()
        offsets = [0.0, -2.3, 1.7, -0.8, 3.2, -1.5]

        for i in range(6):
            cam = CameraConfig(
                camera_id=f"cam{i + 1}",
                brand="hikvision" if i < 3 else "cpplus",
                site_map_x=i * 10.0,
                site_map_y=i * 5.0,
                clock_offset_seconds=offsets[i],
            )
            registry.register(cam)

        # Save and reload registry
        registry_path = tmp_path / "cameras.json"
        registry.save(registry_path)
        loaded_registry = CameraRegistry.load(registry_path)

        assert loaded_registry.count() == 6

        # Generate videos
        video_paths = []
        for i in range(6):
            video_path = tmp_path / f"cam{i + 1}.mp4"
            # Use different colors for each camera
            colors = [
                (255, 0, 0),  # Blue
                (0, 255, 0),  # Green
                (0, 0, 255),  # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
            ]
            generate_test_video(
                video_path,
                duration_seconds=2.0,
                fps=10.0,
                color_sequence=[colors[i]],
            )
            video_paths.append(video_path)

        # Measure throughput
        camera_ids = [f"cam{i + 1}" for i in range(6)]
        result = ThroughputHarness.measure_files(video_paths, camera_ids)

        # Verify throughput result
        assert len(result.streams) == 6
        assert result.aggregate_real_time_factor > 0

        # Export result
        throughput_json = tmp_path / "throughput.json"
        result.to_json(throughput_json)
        assert throughput_json.exists()

        # Verify all cameras ingested correctly
        for i in range(6):
            cam_id = f"cam{i + 1}"
            cam = loaded_registry.get(cam_id)
            assert cam is not None

            source = FileVideoSource(cam_id, video_paths[i], start_timestamp=1000.0)
            with source:
                frames = list(source.frames())
                assert len(frames) >= 15  # ~20 frames expected

                # Verify offset correction
                first_corrected = frames[0].corrected_timestamp(cam.clock_offset_seconds)
                expected = 1000.0 + offsets[i]
                assert abs(first_corrected - expected) < 0.1
