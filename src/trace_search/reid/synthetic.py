"""In-memory flat-colour blocks for the ReID evaluation command.

``make_synthetic_reid_set`` builds the arrays and returns them. It does
not write a file. Each group is solid colour bands. An odd group copies
the previous group's bands and nudges one band, so that pair is a hard
distractor. Seeded noise, a per-camera band shift, and a per-camera
darkening of one band follow. The shift is not a 4-camera cycle.
Once the roll wraps, a cycle mark keeps later cameras apart until
that mark repeats. ``n_cameras`` past the distinct layouts for the
crop height is rejected. There are no human shapes and no text.

Fixture ids are ``FX-SYNREID-nnn``. The use tag on each sample is
``DEV``. That tag lives only on the samples. It is not a manifest
``use`` value.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import numpy.typing as npt

from trace_search.reid.data import ReidSample, SyntheticProvenance

_BANDS = 4
_NOISE_LIMIT = 16
_ID_WIDTH = 3
_DISTRACTOR_LIMIT = 12
_CAMERA_BAND_PERCENT = 70
_CYCLE_BUMP = 80


def make_synthetic_reid_set(
    n_groups: int = 8,
    per_group: int = 4,
    n_cameras: int = 2,
    size: tuple[int, int] = (64, 32),
    seed: int = 0,
) -> list[ReidSample]:
    """Return flat-colour samples. Nothing is written to disk.

    ``size`` is ``(height, width)``. The same ``seed`` returns
    byte-identical crops. ``camera_id`` is ``cam-0``, ``cam-1``, and so
    on. ``label`` is the group index. It is not a person id. An odd
    group copies the previous group's bands and nudges one band, so
    that pair is a hard distractor. ``n_cameras`` above the number of
    distinct layouts for this height is rejected.
    """
    height, width = _require_shape(n_groups, per_group, n_cameras, size)
    rng = np.random.default_rng(seed)
    next_track = {camera: 0 for camera in range(n_cameras)}
    samples: list[ReidSample] = []
    number = 1
    previous: npt.NDArray[np.uint8] | None = None
    for group in range(n_groups):
        colours = _colours_for_group(rng, group, previous)
        if group % 2 == 0:
            previous = colours
        base = _paint(colours, height, width)
        for slot in range(per_group):
            camera = slot % n_cameras
            next_track[camera] += 1
            image = _with_noise(_camera_view(base, camera), rng)
            samples.append(
                _make_sample(
                    number,
                    group,
                    camera,
                    next_track[camera],
                    image,
                    seed=seed,
                    n_groups=n_groups,
                    per_group=per_group,
                    n_cameras=n_cameras,
                    height=height,
                    width=width,
                )
            )
            number += 1
    return samples


def _require_shape(
    n_groups: int,
    per_group: int,
    n_cameras: int,
    size: tuple[int, int],
) -> tuple[int, int]:
    """Reject a set that cannot be named with 3-digit ids or drawn."""
    _require_at_least_one("n_groups", n_groups)
    _require_at_least_one("per_group", per_group)
    _require_at_least_one("n_cameras", n_cameras)
    height, width = _require_size(size)
    total = n_groups * per_group
    if total > 999:
        raise ValueError(
            f"n_groups * per_group is {total}; a synthetic id is 3 digits and stops at 999"
        )
    _require_camera_limit(n_cameras, height)
    return height, width


def _require_camera_limit(n_cameras: int, height: int) -> None:
    """Reject a camera count the crop height cannot keep distinct."""
    limit = _distinct_camera_limit(height)
    if n_cameras > limit:
        raise ValueError(
            f"n_cameras is {n_cameras}; a crop of height {height} supports "
            f"{limit} distinct camera layouts"
        )


def _require_at_least_one(argument: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{argument} must be >= 1, got {value!r}")


def _require_size(size: tuple[int, int]) -> tuple[int, int]:
    height, width = size
    if height < 1 or width < 1:
        raise ValueError(f"size must be > 0 in both dimensions, got {size!r}")
    return height, width


def _colours_for_group(
    rng: np.random.Generator,
    group: int,
    previous: npt.NDArray[np.uint8] | None,
) -> npt.NDArray[np.uint8]:
    """Fresh bands, or a one-band nudge of the previous group."""
    if previous is None or group % 2 == 0:
        return _fresh_colours(rng)
    return _shared_colours(previous, rng, group)


def _fresh_colours(rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    drawn = rng.integers(0, 256, size=(_BANDS, 3), dtype=np.uint8)
    return np.ascontiguousarray(drawn)


def _shared_colours(
    previous: npt.NDArray[np.uint8],
    rng: np.random.Generator,
    group: int,
) -> npt.NDArray[np.uint8]:
    """Copy ``previous`` and nudge one band by at most ``_DISTRACTOR_LIMIT``."""
    shared = np.array(previous, copy=True)
    band = group % _BANDS
    nudge = rng.integers(-_DISTRACTOR_LIMIT, _DISTRACTOR_LIMIT + 1, size=3)
    mixed = shared[band].astype(np.int16) + nudge
    shared[band] = np.clip(mixed, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(shared)


def _paint(
    colours: npt.NDArray[np.uint8],
    height: int,
    width: int,
) -> npt.NDArray[np.uint8]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    edges = np.linspace(0, height, _BANDS + 1, dtype=int)
    for index in range(_BANDS):
        start = int(edges[index])
        stop = int(edges[index + 1])
        image[start:stop, :, :] = colours[index]
    return image


def _with_noise(
    image: npt.NDArray[np.uint8],
    rng: np.random.Generator,
) -> npt.NDArray[np.uint8]:
    noise = rng.integers(0, _NOISE_LIMIT, size=image.shape, dtype=np.uint16)
    mixed = image.astype(np.uint16) + noise
    clipped = np.clip(mixed, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(clipped)


def _camera_view(image: npt.NDArray[np.uint8], camera: int) -> npt.NDArray[np.uint8]:
    """Shift, darken one band, then mark a wrapped camera cycle."""
    shifted = _shift_for_camera(image, camera)
    darkened = _darken_one_band(shifted, camera)
    return _mark_wrapped_cycle(darkened, camera)


def _shift_for_camera(image: npt.NDArray[np.uint8], camera: int) -> npt.NDArray[np.uint8]:
    """Roll the bands by an amount that does not repeat every 4 cameras.

    A roll of ``(camera + 1)`` whole bands has period 4 when the crop is
    4 bands tall, so camera ``k`` matched camera ``k + 4``. Adding
    ``camera`` rows breaks that cycle.
    """
    height = int(image.shape[0])
    band_height = max(height // _BANDS, 1)
    shift = (camera + 1) * band_height + camera
    return np.ascontiguousarray(np.roll(image, shift=shift, axis=0))


def _darken_one_band(image: npt.NDArray[np.uint8], camera: int) -> npt.NDArray[np.uint8]:
    """Keep ``_CAMERA_BAND_PERCENT`` of one band. The band depends on the camera.

    A pure roll keeps each colour's pixel count. Darkening one band
    changes the mix, so two cameras are not translates of each other.
    """
    height = int(image.shape[0])
    edges = np.linspace(0, height, _BANDS + 1, dtype=int)
    band = camera % _BANDS
    start = int(edges[band])
    stop = int(edges[band + 1])
    viewed = np.array(image, copy=True)
    block = viewed[start:stop].astype(np.uint16)
    scaled = block * np.uint16(_CAMERA_BAND_PERCENT) // np.uint16(100)
    viewed[start:stop] = scaled.astype(np.uint8)
    return np.ascontiguousarray(viewed)


def _mark_wrapped_cycle(image: npt.NDArray[np.uint8], camera: int) -> npt.NDArray[np.uint8]:
    """Colour-mark a camera once the roll period has wrapped.

    The mark brightens one channel and writes a corner code. The
    channel repeats every 3 cycles, and the corner code steps by less
    than the noise, so a later camera can match an earlier one.
    ``n_cameras`` is rejected once this height runs out of distinct layouts.
    """
    cycle = camera // _roll_period(int(image.shape[0]))
    if cycle == 0:
        return image
    marked = np.array(image, copy=True)
    channel = cycle % 3
    plane = marked[:, :, channel].astype(np.int16) + _CYCLE_BUMP
    marked[:, :, channel] = np.clip(plane, 0, 255).astype(np.uint8)
    marked[0, 0, 0] = np.uint8(cycle % 256)
    marked[0, 0, 1] = np.uint8((cycle // 256) % 256)
    return np.ascontiguousarray(marked)


def _roll_period(height: int) -> int:
    """How many cameras have a unique band roll at this height."""
    band_height = max(height // _BANDS, 1)
    step = band_height + 1
    return height // math.gcd(step, height)


def _distinct_camera_limit(height: int) -> int:
    """How many leading cameras stay distinct at this height.

    The first repeated camera shares an earlier camera's roll, darkened
    rows, and wrap-mark channel. The corner code has moved by at most
    15, which the noise can hide. Cameras ``0 .. limit-1`` do not repeat.
    A mark step of 12 always shares the darkened rows, so the repeat is
    ``13 * period`` unless a shorter step hits first.
    """
    period = _roll_period(height)
    rows = _darkened_rows(height)
    earliest = 13 * period
    for multiple in (3, 6, 9, 15):
        repeat = _earliest_repeat(period, rows, multiple)
        if repeat is not None and repeat < earliest:
            earliest = repeat
    return earliest


def _earliest_repeat(
    period: int,
    rows: tuple[frozenset[int], ...],
    multiple: int,
) -> int | None:
    """Smallest camera that repeats one ``period`` or more behind it.

    ``multiple`` is a multiple of 3, so both cameras brighten the same
    channel. It stays at most 15, so the corner codes sit inside the noise.
    """
    shift = (multiple * period) % _BANDS
    for offset in range(_BANDS):
        camera = period + offset
        same_rows = rows[camera % _BANDS] == rows[(camera + shift) % _BANDS]
        if same_rows:
            return camera + multiple * period
    return None


def _darkened_rows(height: int) -> tuple[frozenset[int], ...]:
    """Rows each camera band darkens. An empty band darkens nothing."""
    edges = np.linspace(0, height, _BANDS + 1, dtype=int)
    found: list[frozenset[int]] = []
    for band in range(_BANDS):
        start = int(edges[band])
        stop = int(edges[band + 1])
        found.append(frozenset(range(start, stop)))
    return tuple(found)


def _make_sample(
    number: int,
    group: int,
    camera: int,
    track_id: int,
    image: npt.NDArray[np.uint8],
    *,
    seed: int,
    n_groups: int,
    per_group: int,
    n_cameras: int,
    height: int,
    width: int,
) -> ReidSample:
    owned = np.ascontiguousarray(image)
    return ReidSample(
        fixture_id=f"FX-SYNREID-{number:0{_ID_WIDTH}d}",
        use="DEV",
        camera_id=f"cam-{camera}",
        track_id=track_id,
        label=group,
        crop=owned,
        path=None,
        provenance=SyntheticProvenance(
            seed=seed,
            n_groups=n_groups,
            per_group=per_group,
            n_cameras=n_cameras,
            height=height,
            width=width,
            digest=hashlib.sha256(owned.tobytes()).hexdigest(),
        ),
    )
