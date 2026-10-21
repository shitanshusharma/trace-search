"""Tests for FFmpeg LGPL compliance check.

Planted cases call check_ffmpeg_lgpl_compliance() itself. They stub
find_bundled_libavcodec / ctypes.CDLL so the production license and
configuration checks run, instead of mocking the function return value.
"""

import ctypes
from pathlib import Path
from unittest.mock import MagicMock, patch

from trace_search.ingest.ffmpeg_check import (
    OPENCV_FFMPEG_DISTRIBUTIONS,
    OPENCV_LIBS_DIR_NAMES,
    check_ffmpeg_lgpl_compliance,
    find_bundled_libavcodec,
    is_opencv_ffmpeg_distribution,
)


class _FakeCFunc:
    def __init__(self, value):
        self.restype = None
        self._value = value

    def __call__(self):
        return self._value


class _FakeLib:
    def __init__(self, license_str, config_str):
        license_bytes = None if license_str is None else license_str.encode()
        config_bytes = None if config_str is None else config_str.encode()
        self.avcodec_license = _FakeCFunc(license_bytes)
        self.avcodec_configuration = _FakeCFunc(config_bytes)


def _run_check(license_str, config_str, *, found=True, exists=True, load_error=None):
    """Run the production check with a planted libavcodec."""
    fake_path = MagicMock(spec=Path)
    fake_path.exists.return_value = exists
    fake_path.__str__.return_value = "/tmp/planted-libavcodec.so"
    fake_path.__fspath__.return_value = "/tmp/planted-libavcodec.so"

    find_ret = fake_path if found else None

    def fake_cdll(_name, *args, **kwargs):
        if load_error is not None:
            raise OSError(load_error)
        return _FakeLib(license_str, config_str)

    with (
        patch(
            "trace_search.ingest.ffmpeg_check.find_bundled_libavcodec",
            return_value=find_ret,
        ),
        patch("trace_search.ingest.ffmpeg_check.ctypes.CDLL", side_effect=fake_cdll),
    ):
        return check_ffmpeg_lgpl_compliance()


def test_ffmpeg_lgpl_compliance_passes():
    """Real bundled FFmpeg must pass LGPL compliance check."""
    passed, message = check_ffmpeg_lgpl_compliance()
    assert passed, f"FFmpeg LGPL compliance check failed: {message}"
    assert "LGPL" in message or "lgpl" in message.lower()


def test_unmocked_ctypes_reads_real_installed_libavcodec():
    """Load libavcodec from the installed wheel via ctypes (no mocks)."""
    lib_path = find_bundled_libavcodec()
    assert lib_path is not None, "bundled libavcodec missing from installed wheel"
    assert lib_path.exists()
    assert "libavcodec" in lib_path.name

    lib = ctypes.CDLL(str(lib_path))
    lib.avcodec_license.restype = ctypes.c_char_p
    lib.avcodec_configuration.restype = ctypes.c_char_p
    license_bytes = lib.avcodec_license()
    config_bytes = lib.avcodec_configuration()
    assert license_bytes is not None
    assert config_bytes is not None
    license_str = license_bytes.decode("utf-8", errors="replace")
    config_str = config_bytes.decode("utf-8", errors="replace")
    assert "LGPL" in license_str.upper()
    assert "--enable-gpl" not in config_str.lower()
    assert "--enable-nonfree" not in config_str.lower()


def test_opencv_contrib_distribution_names_are_recognised():
    """Contrib wheels must be treated as FFmpeg-carrying opencv distributions."""
    assert "opencv-contrib-python" in OPENCV_FFMPEG_DISTRIBUTIONS
    assert "opencv-contrib-python-headless" in OPENCV_FFMPEG_DISTRIBUTIONS
    assert "opencv_contrib_python_headless.libs" in OPENCV_LIBS_DIR_NAMES
    assert "opencv_contrib_python.libs" in OPENCV_LIBS_DIR_NAMES
    assert is_opencv_ffmpeg_distribution("opencv-contrib-python")
    assert is_opencv_ffmpeg_distribution("opencv_contrib_python_headless")
    assert is_opencv_ffmpeg_distribution("opencv-python-headless")
    assert not is_opencv_ffmpeg_distribution("numpy")


def test_planted_gpl_license_fails():
    """Planted test: GPL license string must fail."""
    passed, message = _run_check("GPL version 3 or later", "--disable-autodetect")
    assert not passed
    assert "GPL version 3" in message


def test_planted_nonfree_flag_fails():
    """Planted test: --enable-nonfree in config must fail."""
    passed, message = _run_check(
        "LGPL version 2.1 or later",
        "--enable-shared --enable-nonfree",
    )
    assert not passed
    assert "nonfree" in message.lower()


def test_planted_gpl_flag_fails():
    """Planted test: --enable-gpl in config must fail."""
    passed, message = _run_check(
        "LGPL version 2.1 or later",
        "--enable-shared --enable-gpl",
    )
    assert not passed
    assert "gpl" in message.lower()


def test_planted_gpl_flag_next_to_lgpl_word_fails():
    """--enable-gpl still fails when the config string also says LGPL."""
    passed, message = _run_check(
        "LGPL version 2.1 or later",
        "--enable-gpl --enable-libx264 License: LGPL",
    )
    assert not passed
    assert "--enable-gpl" in message


def test_planted_missing_lib_fails():
    """Planted test: missing libavcodec must fail."""
    passed, message = _run_check("LGPL version 2.1 or later", "--enable-shared", found=False)
    assert not passed
    assert "find" in message.lower() or "missing" in message.lower()


def test_planted_missing_file_fails():
    """Planted test: libavcodec path that does not exist must fail."""
    passed, message = _run_check(
        "LGPL version 2.1 or later",
        "--enable-shared",
        exists=False,
    )
    assert not passed
    assert "not found" in message.lower()


def test_planted_load_error_fails():
    """Planted test: library load error must fail."""
    passed, message = _run_check(
        "LGPL version 2.1 or later",
        "--enable-shared",
        load_error="file not found",
    )
    assert not passed
    assert "load" in message.lower() or "failed" in message.lower()


def test_planted_empty_license_fails():
    """Empty avcodec_license() evidence must fail closed."""
    passed, message = _run_check("", "--enable-shared")
    assert not passed
    assert "empty" in message.lower() or "null" in message.lower()


def test_planted_null_license_fails():
    """NULL avcodec_license() evidence must fail closed."""
    passed, message = _run_check(None, "--enable-shared")
    assert not passed
    assert "null" in message.lower()


def test_planted_empty_configuration_fails():
    """Empty avcodec_configuration() evidence must fail closed."""
    passed, message = _run_check("LGPL version 2.1 or later", "")
    assert not passed
    assert "empty" in message.lower() or "null" in message.lower()
