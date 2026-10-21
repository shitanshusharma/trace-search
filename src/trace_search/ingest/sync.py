"""Clock synchronization and offset estimation.

Provides utilities to estimate per-camera clock offsets from known sync events.
"""

import statistics


def estimate_offset_from_sync_event(
    camera_timestamps: dict[str, float],
    reference_camera_id: str,
) -> dict[str, float]:
    """Estimate clock offsets from a known sync event.

    A sync event is a moment that appears in multiple cameras simultaneously
    (e.g., a flash, a person passing through a doorway visible to multiple cameras).

    Args:
        camera_timestamps: Dict mapping camera_id -> timestamp when event occurred
        reference_camera_id: Camera to use as time reference (offset = 0)

    Returns:
        Dict mapping camera_id -> offset_seconds to apply

    Raises:
        ValueError: If reference_camera_id not in camera_timestamps

    Example:
        >>> # At wall-clock time T, cameras report:
        >>> timestamps = {"cam1": 100.0, "cam2": 105.0, "cam3": 98.0}
        >>> offsets = estimate_offset_from_sync_event(timestamps, "cam1")
        >>> # cam1: offset=0, cam2: offset=-5 (5s fast), cam3: offset=+2 (2s slow)
        >>> offsets
        {"cam1": 0.0, "cam2": -5.0, "cam3": 2.0}
    """
    if reference_camera_id not in camera_timestamps:
        raise ValueError(f"Reference camera {reference_camera_id} not in timestamps")

    reference_time = camera_timestamps[reference_camera_id]

    offsets = {}
    for camera_id, timestamp in camera_timestamps.items():
        # Offset to add to camera timestamp to match reference
        offsets[camera_id] = reference_time - timestamp

    return offsets


def estimate_offset_from_multiple_events(
    events: list[dict[str, float]],
    reference_camera_id: str,
) -> dict[str, float]:
    """Estimate clock offsets from multiple sync events using median.

    More robust than single-event estimation. Uses median offset across events.

    Args:
        events: List of sync events, each a dict mapping camera_id -> timestamp
        reference_camera_id: Camera to use as time reference

    Returns:
        Dict mapping camera_id -> median offset_seconds to apply

    Raises:
        ValueError: If events is empty or reference camera not in all events
    """
    if not events:
        raise ValueError("At least one sync event required")

    # Validate reference camera appears in all events
    for i, event in enumerate(events):
        if reference_camera_id not in event:
            raise ValueError(
                f"Reference camera {reference_camera_id} not in event {i}"
            )

    # Compute offset for each event
    all_offsets: dict[str, list[float]] = {}
    for event in events:
        event_offsets = estimate_offset_from_sync_event(event, reference_camera_id)
        for camera_id, offset in event_offsets.items():
            if camera_id not in all_offsets:
                all_offsets[camera_id] = []
            all_offsets[camera_id].append(offset)

    # Take median offset for each camera
    median_offsets = {}
    for camera_id, offset_list in all_offsets.items():
        median_offsets[camera_id] = statistics.median(offset_list)

    return median_offsets
