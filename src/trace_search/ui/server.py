"""Stdlib HTTP server for the local UI. It binds 127.0.0.1 and nothing else."""

from __future__ import annotations

import ipaddress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from trace_search.ui.routes import UiRequest, UiResponse, dispatch, server_error

_ERROR_BODY = b'{"error":"error"}'

BIND_HOST = "127.0.0.1"
_MAX_BODY = 1_048_576

STATIC_DIR = Path(__file__).resolve().parent / "static"

SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("Content-Security-Policy", "default-src 'self'"),
    ("X-Frame-Options", "DENY"),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
)


def host_is_loopback(host: str) -> bool:
    """True only for a literal loopback address. Names are refused."""
    text = host.strip()
    if len(text) >= 2 and text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        parsed = ipaddress.ip_address(text)
    except ValueError:
        return False
    return bool(parsed.is_loopback)


class UiServer(ThreadingHTTPServer):
    """Threaded server bound by ``make_server`` to the IPv4 loopback address."""

    allow_reuse_address = True
    daemon_threads = True
    static_root: Path

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        static_root: Path,
    ) -> None:
        self.static_root = static_root
        super().__init__(server_address, handler)


class UiHandler(BaseHTTPRequestHandler):
    """Dispatch one request and attach the security headers to every reply."""

    server: UiServer

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or raw_length == "":
            return b""
        try:
            length = int(raw_length)
        except ValueError:
            return b""
        if length <= 0 or length > _MAX_BODY:
            return b""
        return self.rfile.read(length)

    def _send(self, response: UiResponse) -> None:
        # The stdlib omits the status line for HTTP/0.9, including a garbled
        # request line and HTTP/9.9, which stay on that default. Answer as 1.0.
        if self.request_version == "HTTP/0.9":
            self.request_version = "HTTP/1.0"
        payload = b"" if self.command == "HEAD" else response.body
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(payload)))
        for key, value in SECURITY_HEADERS:
            self.send_header(key, value)
        if response.api:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _dispatch(self) -> None:
        try:
            request = UiRequest(self.command, self.path, self.server.static_root, self._read_body())
            response = dispatch(request)
        except Exception:
            self.log_error("request failed")
            response = server_error()
        self._send(response)

    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def do_PUT(self) -> None:
        self._dispatch()

    def do_DELETE(self) -> None:
        self._dispatch()

    def do_PATCH(self) -> None:
        self._dispatch()

    def do_HEAD(self) -> None:
        self._dispatch()

    def do_OPTIONS(self) -> None:
        self._dispatch()

    def send_error(
        self,
        code: int,
        _message: str | None = None,
        _explain: str | None = None,
    ) -> None:
        """Replace the stdlib HTML error page so every error keeps the security headers."""
        self._send(UiResponse(int(code), _ERROR_BODY, "application/json", True))

    def log_message(self, _fmt: str, *_args: object) -> None:
        """Keep request lines off stderr. Startup errors are printed by the CLI."""
        return


def make_server(port: int, static_root: Path | None = None) -> UiServer:
    """Bind ``127.0.0.1`` and ``port``. The host argument is not a bind address."""
    root = STATIC_DIR if static_root is None else static_root
    return UiServer((BIND_HOST, port), UiHandler, root)


def serve_forever(httpd: UiServer) -> None:
    """Serve until ``shutdown`` is called."""
    httpd.serve_forever()
