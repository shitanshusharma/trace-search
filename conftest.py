"""Keep the licence suite from loading libavcodec when the manifest names OpenCV.

PR #3's tests/conftest.py stubs inspect_opencv_libavcodec with FileNotFoundError
so a test cannot dlopen the bundled library. That stub is correct when the
manifest has no OpenCV row. T1 adds the opencv-python-headless row, and the
real-manifest tests then see a MISSING FFmpeg status. This fixture runs after
that stub and reports an LGPL configuration instead of opening the library.
The CI licence step does not use pytest, so it still calls the real inspector.
"""

import pytest


@pytest.fixture(autouse=True)
def declared_opencv_does_not_dlopen_libavcodec(monkeypatch, request, stub_ffmpeg_for_all_tests):
    """Satisfy the FFmpeg check without loading the wheel's libavcodec."""
    del stub_ffmpeg_for_all_tests
    if "ffmpeg_real" in request.keywords:
        return

    def mock_inspect():
        return ("LGPL version 2.1", "--disable-gpl --disable-nonfree")

    import scripts.check_licenses as cl

    monkeypatch.setattr(cl, "inspect_opencv_libavcodec", mock_inspect)
