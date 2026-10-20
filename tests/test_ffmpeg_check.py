"""Coverage for the shared OpenCV libavcodec helper.

These tests exercise src/trace_search/ingest/ffmpeg_check.py. They do not
load a real libavcodec and they do not edit that module.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import trace_search.ingest.ffmpeg_check as fc


@pytest.mark.parametrize(
    ("distribution", "expected"),
    [
        ("opencv-python", True),
        ("OpenCV_Python_Headless", True),
        ("opencv-contrib-python", True),
        ("opencv-contrib-python-headless", True),
        ("opencv-python-gpu", False),
        ("", False),
    ],
)
def test_is_opencv_ffmpeg_distribution(distribution: str, expected: bool) -> None:
    assert fc.is_opencv_ffmpeg_distribution(distribution) is expected


def _block_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)


def _patch_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    real_resolve = Path.resolve

    def flaky_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        if self.name == "unresolvable":
            raise OSError("cannot resolve")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", flaky_resolve)


def test_find_bundled_libavcodec_matches_path_and_skips_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Import failure, a failed resolve, duplicate roots, then a real match."""
    _block_cv2(monkeypatch)
    _patch_resolve(monkeypatch)

    bad = tmp_path / "unresolvable"
    bad.mkdir()
    empty = tmp_path / "empty-site"
    empty.mkdir()
    (empty / "opencv_python.libs").write_text("not a directory", encoding="utf-8")
    (empty / "opencv_python_headless.libs").mkdir()

    user = tmp_path / "user-site"
    libs = user / "opencv_contrib_python.libs"
    libs.mkdir(parents=True)
    (libs / "libavcodec.so.60").write_bytes(b"codec")
    (libs / "zzavcodec.so").write_bytes(b"other")

    monkeypatch.setattr(fc.site, "getsitepackages", lambda: [str(bad), str(bad), str(empty)])
    monkeypatch.setattr(fc.site, "getusersitepackages", lambda: str(user))

    found = fc.find_bundled_libavcodec()
    assert found == libs / "libavcodec.so.60"


def test_find_bundled_libavcodec_uses_cv2_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cv2_dir = tmp_path / "pkg" / "cv2"
    cv2_dir.mkdir(parents=True)
    init = cv2_dir / "__init__.py"
    init.write_text("", encoding="utf-8")
    libs = tmp_path / "pkg" / "opencv_python.libs"
    libs.mkdir()
    (libs / "libavcodec-cv2.so").write_bytes(b"codec")

    module = types.ModuleType("cv2")
    module.__file__ = str(init)
    monkeypatch.setitem(sys.modules, "cv2", module)
    monkeypatch.setattr(fc.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(fc.site, "getusersitepackages", lambda: str(tmp_path / "unused"))

    found = fc.find_bundled_libavcodec()
    assert found == libs / "libavcodec-cv2.so"


def test_find_bundled_libavcodec_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_cv2(monkeypatch)
    empty = tmp_path / "nowhere"
    empty.mkdir()
    monkeypatch.setattr(fc.site, "getsitepackages", lambda: [str(empty)])
    monkeypatch.setattr(fc.site, "getusersitepackages", lambda: str(empty))

    assert fc.find_bundled_libavcodec() is None


class _FakeFn:
    def __init__(self, payload: bytes | None) -> None:
        self.payload = payload
        self.restype = None

    def __call__(self) -> bytes | None:
        return self.payload


class _FakeLib:
    def __init__(self, license_raw: bytes | None, config_raw: bytes | None) -> None:
        self.avcodec_license = _FakeFn(license_raw)
        self.avcodec_configuration = _FakeFn(config_raw)


def test_inspect_raises_when_library_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fc, "find_bundled_libavcodec", lambda: None)
    with pytest.raises(FileNotFoundError, match="libavcodec not found"):
        fc.inspect_opencv_libavcodec()


def test_inspect_decodes_license_and_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    lib_path = Path("/tmp/libavcodec.so")
    monkeypatch.setattr(fc, "find_bundled_libavcodec", lambda: lib_path)
    seen: dict[str, str] = {}

    def fake_cdll(path: str) -> _FakeLib:
        seen["path"] = path
        return _FakeLib(b"LGPLv2.1", b"--disable-gpl")

    monkeypatch.setattr(fc.ctypes, "CDLL", fake_cdll)
    license_str, config_str = fc.inspect_opencv_libavcodec()
    assert license_str == "LGPLv2.1"
    assert config_str == "--disable-gpl"
    assert seen["path"] == str(lib_path)


@pytest.mark.parametrize(
    ("license_raw", "config_raw"),
    [
        (None, b"--disable-gpl"),
        (b"LGPLv2.1", None),
    ],
)
def test_inspect_rejects_null_symbols(
    monkeypatch: pytest.MonkeyPatch,
    license_raw: bytes | None,
    config_raw: bytes | None,
) -> None:
    monkeypatch.setattr(fc, "find_bundled_libavcodec", lambda: Path("/tmp/libavcodec.so"))
    monkeypatch.setattr(
        fc.ctypes,
        "CDLL",
        lambda _path: _FakeLib(license_raw, config_raw),
    )
    with pytest.raises(RuntimeError, match="returned NULL"):
        fc.inspect_opencv_libavcodec()
