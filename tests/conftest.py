"""Pytest configuration for license fence tests."""

import pytest


@pytest.fixture(autouse=True)
def stub_ffmpeg_for_all_tests(monkeypatch, request):
    """
    Autouse fixture that stubs check_ffmpeg_build for all tests.

    FFmpeg-specific tests can opt out with @pytest.mark.ffmpeg_real.
    """
    # Skip stubbing if test is marked as ffmpeg_real
    if 'ffmpeg_real' in request.keywords:
        return

    # Stub check_ffmpeg_build to return NOT_A_DEPENDENCY
    def mock_check(*args, **kwargs):
        return [], 'NOT_A_DEPENDENCY'

    import scripts.check_licenses as cl
    monkeypatch.setattr(cl, 'check_ffmpeg_build', mock_check)
