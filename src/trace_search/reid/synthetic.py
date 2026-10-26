"""In-memory flat-colour blocks for the ReID evaluation command.

``make_synthetic_reid_set`` builds the arrays and returns them. It does
not write a file. Each group is solid colour bands, plus seeded noise
and a shift that moves those bands per camera. There are no human
shapes and no text.

Fixture ids are ``FX-SYNREID-nnn``. The use tag on each sample is
``DEV``. That tag lives only on the samples. It is not a manifest
``use`` value.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from trace_search.reid.data import ReidSample

_BANDS = 4
_NOISE_LIMIT = 16
_ID_WIDTH = 3


def make_synthetic_reid_set(
    n_groups: int = 4,
    per_group: int = 4,
    n_cameras: int = 2,
    size: tuple[int, int] = (64, 32),
    seed: int = 0,
) -> list[ReidSample]:
    """Return flat-colour samples. Nothing is written to disk.

    ``size`` is ``(height, width)``. The same ``seed`` returns
    byte-identical crops. ``camera_id`` is ``cam-0``, ``cam-1``, and so
    on. ``label`` is the group index. It is not a person id.
    """
    height, width = _require_shape(n_groups, per_group, n_cameras, size)
    rng = np.random.default_rng(seed)
    next_track = {camera: 0 for camera in range(n_cameras)}
    samples: list[ReidSample] = []
    number = 1
    for group in range(n_groups):
        base = _paint(_group_colours(rng), height, width)
        for slot in range(per_group):
            camera = slot % n_cameras
            next_track[camera] += 1
            image = _with_noise(_shift_for_camera(base, camera), rng)
            samples.append(_make_sample(number, group, camera, next_track[camera], image))
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
    return height, width


def _require_at_least_one(argument: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{argument} must be >= 1, got {value!r}")


def _require_size(size: tuple[int, int]) -> tuple[int, int]:
    height, width = size
    if height < 1 or width < 1:
        raise ValueError(f"size must be > 0 in both dimensions, got {size!r}")
    return height, width


def _group_colours(rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    drawn = rng.integers(0, 256, size=(_BANDS, 3), dtype=np.uint8)
    return np.ascontiguousarray(drawn)


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


def _shift_for_camera(image: npt.NDArray[np.uint8], camera: int) -> npt.NDArray[np.uint8]:
    """Roll the colour bands down. A sideways roll would move only noise."""
    band_height = max(image.shape[0] // _BANDS, 1)
    shift = (camera + 1) * band_height
    return np.ascontiguousarray(np.roll(image, shift=shift, axis=0))


def _make_sample(
    number: int,
    group: int,
    camera: int,
    track_id: int,
    image: npt.NDArray[np.uint8],
) -> ReidSample:
    return ReidSample(
        fixture_id=f"FX-SYNREID-{number:0{_ID_WIDTH}d}",
        use="DEV",
        camera_id=f"cam-{camera}",
        track_id=track_id,
        label=group,
        crop=image,
        path=None,
    )
