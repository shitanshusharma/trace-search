"""Single FFmpeg/libavcodec check for OpenCV wheels.

OPENCV_FFMPEG_DISTRIBUTIONS is the only wheel-name list, including the
contrib distributions. The licence fence imports
is_opencv_ffmpeg_distribution and inspect_opencv_libavcodec from this
module. check_ffmpeg_lgpl_compliance() is the fail-closed wrapper CI calls.
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


def _judge_lgpl_evidence(license_str: str, config_str: str) -> tuple[bool, str]:
    """Judge license and configuration strings already read from libavcodec.

    Empty evidence, a non-LGPL license, --enable-gpl, or --enable-nonfree
    fails closed.
    """
    if not license_str.strip():
        return False, "avcodec_license() returned empty evidence"
    if not config_str.strip():
        return False, "avcodec_configuration() returned empty evidence"

    if "LGPL" not in license_str.upper():
        return False, f"License is not LGPL: {license_str!r}"

    config_lower = config_str.lower()
    if "--enable-gpl" in config_lower:
        return False, f"Configuration contains --enable-gpl: {config_str!r}"
    if "--enable-nonfree" in config_lower:
        return False, f"Configuration contains --enable-nonfree: {config_str!r}"

    return True, f"FFmpeg LGPL compliance verified: {license_str}"


def check_ffmpeg_lgpl_compliance() -> tuple[bool, str]:
    """Fail-closed LGPL check for the OpenCV wheel's libavcodec.

    CI and the planted tests call this. It uses find_bundled_libavcodec()
    and inspect_opencv_libavcodec() above, so the wheel-name list stays in
    this module only. Missing evidence, a non-LGPL license, --enable-gpl,
    or --enable-nonfree fails the check.
    """
    lib_path = find_bundled_libavcodec()
    if lib_path is None:
        return False, "Failed to find bundled libavcodec library"
    if not lib_path.exists():
        return False, f"libavcodec not found at {lib_path}"

    try:
        license_str, config_str = inspect_opencv_libavcodec()
    except FileNotFoundError:
        return False, "Failed to find bundled libavcodec library"
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Failed to load libavcodec: {exc}"

    return _judge_lgpl_evidence(license_str, config_str)
