"""Single FFmpeg/libavcodec check for OpenCV wheels.

OPENCV_FFMPEG_DISTRIBUTIONS is the only list of wheel names. The licence
fence imports this module instead of keeping its own copy.
"""

import ctypes
import site
from pathlib import Path

# Wheel distributions that bundle FFmpeg. Match case-insensitively.
OPENCV_FFMPEG_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)

OPENCV_LIBS_DIR_NAMES = (
    "opencv_python.libs",
    "opencv_python_headless.libs",
    "opencv_contrib_python.libs",
    "opencv_contrib_python_headless.libs",
)


def is_opencv_ffmpeg_distribution(name: str) -> bool:
    """True for opencv[-contrib]-python[-headless], any case or underscore."""
    normalized = name.lower().replace("_", "-")
    return normalized in OPENCV_FFMPEG_DISTRIBUTIONS


def find_bundled_libavcodec() -> Path | None:
    """Locate libavcodec shipped beside an installed opencv wheel."""
    roots: list[Path] = []
    try:
        import cv2

        cv2_dir = Path(cv2.__file__).resolve().parent
        roots.extend([cv2_dir, cv2_dir.parent])
    except ImportError:
        pass

    for site_dir in list(site.getsitepackages()) + [site.getusersitepackages()]:
        roots.append(Path(site_dir))

    seen: set[Path] = set()
    for root in roots:
        try:
            key = root.resolve()
        except OSError:
            key = root
        if key in seen:
            continue
        seen.add(key)
        for libs_name in OPENCV_LIBS_DIR_NAMES:
            libs_dir = root / libs_name
            if not libs_dir.is_dir():
                continue
            matches = sorted(libs_dir.glob("libavcodec*"))
            matches += sorted(libs_dir.glob("*avcodec*"))
            if matches:
                return matches[0]
    return None


def inspect_opencv_libavcodec() -> tuple[str, str]:
    """Read avcodec_license() and avcodec_configuration() from the bundled lib.

    Raises if the library is missing or either call fails. Callers fail closed.
    """
    lib_path = find_bundled_libavcodec()
    if lib_path is None:
        raise FileNotFoundError("opencv bundled libavcodec not found")

    lib = ctypes.CDLL(str(lib_path))
    lib.avcodec_license.restype = ctypes.c_char_p
    lib.avcodec_configuration.restype = ctypes.c_char_p
    license_raw = lib.avcodec_license()
    config_raw = lib.avcodec_configuration()
    if license_raw is None or config_raw is None:
        raise RuntimeError(
            "avcodec_license() or avcodec_configuration() returned NULL"
        )
    return (
        license_raw.decode("utf-8", errors="replace"),
        config_raw.decode("utf-8", errors="replace"),
    )
