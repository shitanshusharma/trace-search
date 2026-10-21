"""RTSP stream ingestion (optional, off by default).

Ingests live RTSP streams from cameras. This is a stretch goal and not
required for Cut 01 acceptance criteria.
"""

import logging
import re
import time

try:
    import cv2
except ImportError:
    raise ImportError(
        "opencv-python-headless is required for RTSP ingestion. "
        "Install with: pip install opencv-python-headless"
    )

from .video_source import Frame, VideoSource

logger = logging.getLogger(__name__)


def redact_rtsp_credentials(url: str) -> str:
    """Redact credentials from an RTSP/RTSPS URL.

    Userinfo is everything after the scheme and before the last '@' that
    introduces the host. That split does not use urlsplit, so a password
    may contain '@', '#', '?', '/', or two or more spaces. Scheme case is
    preserved. One URL ends at the next rtsps?:// or newline, so two
    adjacent URLs stay separate. URLs echoed in errors are redacted too.

    Args:
        url: RTSP/RTSPS URL, or text that contains one

    Returns:
        Text with each URL's userinfo replaced by ***
    """
    if not url:
        return url

    pieces: list[str] = []
    cursor = 0
    for match in _RTSP_SCHEME_RE.finditer(url):
        if match.start() < cursor:
            continue
        pieces.append(url[cursor : match.start()])
        tail = url[match.end() :]
        end = _rtsp_span_end(tail)
        pieces.append(_redact_rtsp_span(match.group(0), tail[:end]))
        cursor = match.end() + end
    pieces.append(url[cursor:])
    return "".join(pieces)


_RTSP_SCHEME_RE = re.compile(r"rtsps?://", re.IGNORECASE)


def _rtsp_span_end(tail: str) -> int:
    """Return the end index of one URL in the text after scheme://.

    The span runs to the next rtsps?:// or newline. Spaces stay inside the
    span, including a password that contains two or more spaces. The last
    '@' in the span is the host separator.
    """
    newline = tail.find("\n")
    scheme = _RTSP_SCHEME_RE.search(tail)
    ends = []
    if newline != -1:
        ends.append(newline)
    if scheme is not None:
        ends.append(scheme.start())
    if not ends:
        return len(tail)
    return min(ends)


def _redact_rtsp_span(scheme: str, segment: str) -> str:
    """Replace userinfo in one URL. scheme includes '://'."""
    if "@" not in segment:
        return scheme + segment
    _userinfo, separator, host = segment.rpartition("@")
    if not separator:
        return scheme + segment
    return f"{scheme}***@{host}"


class RTSPVideoSource(VideoSource):
    """Video source that reads from an RTSP stream.

    Note: This is a stretch goal feature, off by default.
    File-based ingest is the primary focus for Cut 01.

    WARNING: Credentials in RTSP URLs are redacted in all errors and logs.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 5.0,
    ):
        """Initialize RTSP video source.

        Args:
            camera_id: Identifier for the camera
            rtsp_url: RTSP stream URL (e.g., rtsp://user:pass@host:port/stream)
            reconnect_attempts: Number of reconnection attempts on failure
            reconnect_delay_seconds: Delay between reconnection attempts
        """
        super().__init__(camera_id)
        self.rtsp_url = rtsp_url
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds

        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 0.0
        self._current_frame_number: int = 0
        self._start_time: float = 0.0
        self._last_error: str = ""  # Store last error with credentials redacted

    def open(self) -> None:
        """Open the RTSP stream.

        Raises:
            IOError: If stream cannot be opened after all reconnection attempts
        """
        last_error = None

        for attempt in range(self.reconnect_attempts):
            try:
                self._cap = cv2.VideoCapture(self.rtsp_url)

                if self._cap.isOpened():
                    self._fps = self._cap.get(cv2.CAP_PROP_FPS)
                    if self._fps <= 0:
                        self._fps = 25.0  # Default for streams

                    self._current_frame_number = 0
                    self._start_time = time.time()
                    self._is_open = True
                    self._last_error = ""
                    return

                last_error = "Stream opened but not ready"

            except Exception as e:
                last_error = redact_rtsp_credentials(str(e))
                logger.warning(
                    "RTSP open attempt %s/%s failed for %s: %s",
                    attempt + 1,
                    self.reconnect_attempts,
                    redact_rtsp_credentials(self.rtsp_url),
                    last_error,
                )

            if attempt < self.reconnect_attempts - 1:
                time.sleep(self.reconnect_delay_seconds)

        self._last_error = redact_rtsp_credentials(str(last_error))
        raise OSError(
            f"Failed to open RTSP stream {redact_rtsp_credentials(self.rtsp_url)} after "
            f"{self.reconnect_attempts} attempts. Last error: {self._last_error}"
        )

    def close(self) -> None:
        """Close the RTSP stream and release resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False

    def read_frame(self) -> Frame | None:
        """Read next frame from stream.

        Returns:
            Frame if available, None if stream ended or error
        """
        if not self._is_open or self._cap is None:
            return None

        ret, image = self._cap.read()
        if not ret or image is None:
            return None

        # For live streams, use wall-clock time as timestamp
        timestamp = time.time()

        frame = Frame(
            image=image,
            timestamp_seconds=timestamp,
            frame_number=self._current_frame_number,
            camera_id=self.camera_id,
        )

        self._current_frame_number += 1
        return frame

    def get_fps(self) -> float:
        """Get frames per second of the stream."""
        return self._fps

    def get_frame_count(self) -> int | None:
        """Get total frame count (always None for streams)."""
        return None  # Streams have no known end

    def __repr__(self) -> str:
        """String representation with redacted credentials."""
        return (
            f"RTSPVideoSource(camera_id={self.camera_id!r}, "
            f"rtsp_url={redact_rtsp_credentials(self.rtsp_url)!r})"
        )
