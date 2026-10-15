"""Local web UI backend — pure standard library (http.server), no dependencies.

Serves the static frontend in ``webui/`` and a small JSON API:

    GET  /                 the single-page UI
    GET  /api/health       backend + LLM status
    GET  /api/types        recognized file kinds
    POST /api/summarize    multipart upload -> Summary

The API summarizes uploaded bytes and nothing else. It deliberately has no way
to name a path for the server to open: that turns a local convenience into a
file-read primitive for any web page the user has open, since a browser will
happily post to 127.0.0.1 from any origin. Use the CLI for local files.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import json
import mmap
import time
import re
import secrets
import socket
import tempfile
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from .cache import SummaryCache
from . import __version__
from .config import Config
from .extract import SUPPORTED_KINDS
from .llm import LLMClient
from .summarize import summarize_path

def _find_webui() -> Path:
    """The UI ships inside the package; the older layout is still honoured."""
    here = Path(__file__).resolve().parent
    for cand in (here / "webui", here.parent / "webui"):
        if (cand / "index.html").is_file():
            return cand
    return here / "webui"


WEBUI_DIR = _find_webui()
#: How much of a body we are willing to read only to throw away.
_DRAIN_LIMIT = 4 * 1024 * 1024
#: And how long we will wait for it. Long enough for a body already
#: on the wire, short enough that a false declaration cannot hang us.
def _read1(stream, size: int) -> bytes:
    """One read of at most ``size`` bytes, returning as soon as any arrive.

    ``BufferedReader.read1`` is what makes a deadline in the calling loop mean
    anything: ``read`` blocks until it has the full count, so the loop around
    it never gets to look at the clock. ``rfile`` is a BufferedReader for the
    handler's default buffering, but an unbuffered handler hands back a raw
    SocketIO with no ``read1`` — where ``read`` is already a single recv, so
    the fallback is correct rather than merely tolerable.
    """
    reader = getattr(stream, "read1", None)
    return reader(size) if reader is not None else stream.read(size)


_DRAIN_SECONDS = 2.0
#: And for the whole drain. Per-read alone, a byte every 1.9 seconds
#: holds a thread for as long as the client cares to.
_DRAIN_TOTAL_SECONDS = 5.0
#: How long a whole upload body may take to arrive. 60 MB over a slow
#: link is well inside this; a slowloris is not.
_BODY_SECONDS = 120.0
#: How long a connection may send nothing at all before it is dropped, and
#: the rate it must sustain once past the grace period. Together these are
#: what stops MAX_CONNECTIONS slow readers from being a denial of service:
#: the overall deadline bounds one connection, these bound how cheaply an
#: attacker can hold one.
_IDLE_SECONDS = 15.0
_GRACE_SECONDS = 10.0
_MIN_BYTES_PER_SECOND = 1024

#: How long the request line and headers may take, in total. Generous for a
#: real client — they arrive in one packet — and short enough that holding a
#: thread costs something.
_HEADER_SECONDS = 20.0
#: How many connections may be in flight at once. A local UI needs a handful;
#: the ceiling is here because ThreadingHTTPServer has none, and 400 sockets
#: sending 60 bytes each were 400 threads with an 8 MB stack reservation
#: apiece. Past this the server says 503 rather than accepting work it has no
#: thread for — at 3,000 the accept loop died with "can't start new thread",
#: which takes the whole server down rather than one request.
MAX_CONNECTIONS = 64
AUTH_USER = "sift"

MAX_UPLOAD = 60 * 1024 * 1024  # 60 MB
MAX_MULTIPART_PARTS = 128
_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class _BoundedServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a ceiling on connections in flight.

    The base class has none: one thread per connection, spawned on accept,
    with no accounting. 400 sockets that had sent 60 bytes each held 400
    threads; at 3,000 the accept loop raised "can't start new thread" from
    inside socketserver and the server stopped answering anybody, which is a
    worse failure than refusing the 65th caller.

    Refused with 503 rather than a silent hang-up, so a real client that
    arrives during a burst learns why.
    """

    daemon_threads = True

    def __init__(self, *args, **kwargs) -> None:
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address) -> None:
        if not self._slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            # The thread never started, so nothing will release for us.
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def run_server(host: str, port: int, config: Config,
               client: Optional[LLMClient]) -> int:
    password = secrets.token_urlsafe(18)
    handler = _make_handler(config, client, host, port, password=password)
    httpd = _BoundedServer((host, port), handler)
    httpd.daemon_threads = True
    bound_port = httpd.server_address[1]
    if bound_port != port:
        httpd.RequestHandlerClass = _make_handler(
            config, client, host, bound_port, password=password)
    engine = "no LLM (heuristic)" if client is None else \
        f"{client.provider} @ {client.endpoint}"
    print(f"sift web UI → http://{host}:{bound_port}   [{engine}]")
    print(f"authentication → user={AUTH_USER} password={password}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nsift: shutting down")
    finally:
        httpd.server_close()
    return 0


def _make_handler(config: Config, client: Optional[LLMClient], host: str,
                  port: int, cache: Optional[SummaryCache] = None, *,
                  password: str):
    allowed_hosts = _allowed_hosts(host, port)
    expected_auth = f"{AUTH_USER}:{password}".encode("ascii")
    # An injectable cache, so a test can ask the cache whether it hit rather
    # than reading the answer off the reply — the reply is exactly what must
    # not say so.
    if cache is None:
        cache = SummaryCache(config.cache)

    class Handler(BaseHTTPRequestHandler):
        server_version = "sift/0.1"
        protocol_version = "HTTP/1.1"
        #: A client that opens a connection and then goes quiet must not hold a
        #: thread forever. Without this, 200 half-open POSTs are 200 threads.
        timeout = 30

        def log_message(self, *_args):  # keep the console clean
            if config.verbose:
                super().log_message(*_args)

        def handle_one_request(self):
            """Let a client disconnect without printing a traceback.

            And put a deadline on the request line and headers, which had
            none. `timeout = 30` above is a socket *idle* timeout: any byte
            resets it, and the request line is read with readline(), which
            loops over recv(). A client sending "GET / HTTP/1.1\\r\\nX-Pad: "
            and then one byte every 20 seconds held a thread for as long as
            it cared to — 60 bytes bought a thread, and the body deadline
            added earlier never came into it, because that client never
            finishes its headers and so never reaches the body.

            A watchdog rather than a socket timeout, for the same reason
            read1 was needed below: the timeout is per-recv and the drip
            resets it. It is cancelled the moment the headers are in, so a
            slow *body* is still the body deadline's business.
            """
            watchdog = threading.Timer(_HEADER_SECONDS, self._abandon)
            watchdog.daemon = True
            self._watchdog = watchdog
            watchdog.start()
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except TimeoutError:
                self.close_connection = True
            except OSError:
                # The watchdog shut the socket under us.
                self.close_connection = True
            finally:
                watchdog.cancel()

        def parse_request(self):
            parsed = super().parse_request()
            # Headers are in; whatever takes time from here has its own
            # deadline and its own error message.
            self._watchdog.cancel()
            return parsed

        def _abandon(self):
            """Shut the socket down from the watchdog thread.

            shutdown() rather than close(): the handler thread is sitting in
            recv(), and shutdown makes that return empty — which the base
            class already reads as "client went away" — whereas close()
            would hand it a closed descriptor to trip over.
            """
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        # -- responses ---------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str,
                  close: bool = False,
                  headers: Optional[dict[str, str]] = None) -> None:
            if close:
                # An error answered without reading the body leaves that body
                # in the socket, where keep-alive would parse it as the next
                # request — a request the client smuggled past every check.
                self.close_connection = True
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                if close:
                    self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def _json(self, obj, code: int = 200, close: bool = False,
                  headers: Optional[dict[str, str]] = None) -> None:
            # The encode is inside the try, because it can fail: a summary of
            # a file whose content is a lone surrogate raised here, outside
            # do_POST's handler, and the client received zero bytes — no HTTP
            # response at all. ensure_ascii escapes what utf-8 refuses.
            try:
                body = json.dumps(obj, ensure_ascii=False,
                                  allow_nan=False).encode("utf-8")
            except (ValueError, UnicodeEncodeError):
                body = json.dumps(obj, ensure_ascii=True,
                                  allow_nan=False, default=str).encode("ascii")
            self._send(code, body, "application/json; charset=utf-8",
                       close=close, headers=headers)

        # -- request gate ------------------------------------------------
        def _authorized(self) -> bool:
            values = self.headers.get_all("Authorization") or []
            if len(values) != 1:
                return False
            scheme, separator, token = values[0].partition(" ")
            if not separator or scheme.lower() != "basic" or not token:
                return False
            try:
                supplied = base64.b64decode(token, validate=True)
            except (binascii.Error, ValueError):
                return False
            return hmac.compare_digest(supplied, expected_auth)

        def _require_auth(self):
            return self._json(
                {"error": "authentication required"}, 401, close=True,
                headers={
                    "WWW-Authenticate": 'Basic realm="sift", charset="UTF-8"'
                })

        def _same_origin(self) -> bool:
            """Reject requests a browser made on behalf of another page.

            The UI is same-origin, so a request that carries an Origin at all
            comes from somewhere else. Checking Host too closes DNS rebinding,
            where an attacker's name resolves to 127.0.0.1 and the loopback
            bind stops protecting anything.
            """
            # A missing or empty Host passed the gate entirely: "if host and
            # …" reads as "check it when there is one", and HTTP/1.1 requires
            # one, so its absence is not a reason to skip the check. Not
            # browser-reachable, but it bypassed the stated rebinding defence
            # for every other kind of client.
            host = (self.headers.get("Host") or "").strip().lower()
            if host not in allowed_hosts:
                return False
            origin = self.headers.get("Origin")
            if origin and origin.strip().lower().rstrip("/") not in (
                    f"http://{h}" for h in allowed_hosts):
                return False
            return True

        # -- routes ------------------------------------------------------
        def do_OPTIONS(self):
            if not self._authorized():
                return self._require_auth()
            # No CORS is granted, so there is nothing to preflight.
            self._send(405, b"", "text/plain", close=True)

        def do_GET(self):
            if not self._authorized():
                return self._require_auth()
            # A GET carrying a body is answered and the body left in the
            # socket, where keep-alive parses it as the next request — the
            # very thing _send's `close` flag exists to prevent, honoured on
            # every POST path and on none of these. A browser will not send
            # one; something that is not a browser will, and then it has
            # smuggled a request past the origin check that ran on the first
            # one. Nothing is read: a GET has no body worth having.
            if self.headers.get("Content-Length") or self.headers.get(
                    "Transfer-Encoding"):
                return self._json({"error": "GET must not carry a body"}, 400,
                                  close=True)
            if not self._same_origin():
                return self._json({"error": "cross-origin request refused"}, 403,
                                  close=True)
            path = self.path.split("?", 1)[0]
            if path == "/api/health":
                return self._json(_health(config, client))
            if path == "/api/types":
                return self._json({"kinds": SUPPORTED_KINDS})
            if path in _STATIC:
                return self._static(*_STATIC[path])
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self._authorized():
                return self._require_auth()
            if not self._same_origin():
                return self._json({"error": "cross-origin request refused"}, 403,
                                  close=True)
            if self.path.split("?", 1)[0] != "/api/summarize":
                return self._json({"error": "not found"}, 404, close=True)
            try:
                summ = self._handle_summarize()
            except _BadRequest as exc:
                return self._json({"error": str(exc)}, 400, close=True)
            except Exception:  # noqa: BLE001
                # The message could name a temp path or quote the input back.
                # The console (with -v) gets the detail; the client gets none.
                if config.verbose:
                    import traceback
                    traceback.print_exc()
                return self._json({"error": "could not process this upload"},
                                  400, close=True)
            return self._json(summ.to_dict())

        # -- helpers -----------------------------------------------------
        def _static(self, filename: str, ctype: str) -> None:
            fp = WEBUI_DIR / filename
            try:
                body = fp.read_bytes()
            except OSError:
                # A missing file already had this page; an unreadable one —
                # wrong mode, a bad mount — used to raise past do_GET's
                # handler and send the client zero bytes, no HTTP response at
                # all, with a traceback on the console.
                return self._send(500, _MISSING_UI.encode(),
                                  "text/html; charset=utf-8")
            self._send(200, body, ctype)

        def _read_body_to(self, destination: Path) -> int:
            """Stream the request body to *destination* with bounded memory."""
            # Duplicate Content-Length headers, and Content-Length beside
            # Transfer-Encoding, are how a request is made to mean two things
            # at once: this took the first of the duplicates and never looked
            # at Transfer-Encoding, so a body could be read as one request and
            # the remainder as another, and two responses came back on one
            # connection. RFC 7230 §3.3.3 says reject.
            lengths = self.headers.get_all("Content-Length") or []
            if self.headers.get("Transfer-Encoding"):
                raise _BadRequest("Transfer-Encoding is not supported here; "
                                  "send Content-Length")
            if not lengths:
                raise _BadRequest("Content-Length required")
            if len({v.strip() for v in lengths}) > 1:
                raise _BadRequest("conflicting Content-Length headers")
            raw = lengths[0]
            try:
                length = int(raw)
            except ValueError:
                raise _BadRequest("invalid Content-Length") from None
            if length <= 0:
                raise _BadRequest("empty request")
            if length > MAX_UPLOAD:
                # Read and discard first: replying before the body is on the
                # wire means the client is still sending when the socket
                # closes, so it sees a broken pipe instead of the reason.
                self._drain(length)
                raise _BadRequest(f"upload exceeds {MAX_UPLOAD} bytes")
            # A deadline for the whole body, not per read. `timeout = 30` is
            # a socket idle timeout: any byte resets it, so a client sending
            # one every 29 seconds holds a thread for as long as it likes,
            # and 20 of them are 20 threads. An upload that cannot finish
            # inside this is not an upload sift can serve anyway.
            deadline = time.monotonic() + _BODY_SECONDS
            previous = self.connection.gettimeout()
            remaining = length
            received = 0
            # The sliding window the rate floor is measured over.
            window_start, window_bytes = time.monotonic(), 0
            try:
                with destination.open("wb") as sink:
                    while remaining > 0:
                        left = deadline - time.monotonic()
                        if left <= 0:
                            raise _BadRequest(
                                f"upload did not complete within "
                                f"{_BODY_SECONDS:.0f}s")
                        # The remaining budget goes on the socket, so a client
                        # that stops sending altogether is cut off at the
                        # deadline rather than at whatever the idle timeout is.
                        self.connection.settimeout(min(left, _IDLE_SECONDS))
                        # read1 returns as soon as bytes arrive and every chunk
                        # is written before the next read. Peak request-body
                        # memory is therefore one MiB, independent of upload
                        # size and connection count.
                        chunk = _read1(self.rfile, min(remaining, 1 << 20))
                        if not chunk:
                            raise _BadRequest("upload ended early")
                        sink.write(chunk)
                        received += len(chunk)
                        remaining -= len(chunk)
                        # A sliding rate floor prevents a byte just inside the
                        # idle allowance from holding a connection indefinitely.
                        now = time.monotonic()
                        if now - window_start >= _GRACE_SECONDS:
                            moved = received - window_bytes
                            floor = (_MIN_BYTES_PER_SECOND
                                     * (now - window_start))
                            if moved < floor:
                                raise _BadRequest(
                                    f"upload slower than "
                                    f"{_MIN_BYTES_PER_SECOND} B/s "
                                    f"({moved} bytes in the last "
                                    f"{now - window_start:.0f}s)")
                            window_start, window_bytes = now, received
            except socket.timeout:
                raise _BadRequest(
                    f"upload did not complete within {_BODY_SECONDS:.0f}s"
                ) from None
            finally:
                self.connection.settimeout(previous)
            return received

        def _drain(self, length: int) -> None:
            """Swallow a body we are about to refuse, so the error arrives.

            Bounded twice: by how much we will read, and by how long we will
            wait for it. A client that declares 999 MB and sends one byte is
            not going to send the rest, and waiting for it would turn a
            refusal into a hang.
            """
            remaining = min(length, _DRAIN_LIMIT)
            previous = self.connection.gettimeout()
            self.connection.settimeout(_DRAIN_SECONDS)
            # A deadline for the whole drain, not for each read. Per-read, a
            # client sending one byte every 1.9 seconds resets the clock
            # forever — which made this cheaper to abuse than the timeout it
            # was added beside, not dearer.
            deadline = time.monotonic() + _DRAIN_TOTAL_SECONDS
            try:
                while remaining > 0 and time.monotonic() < deadline:
                    # Same reason as _read_body: read() would swallow the
                    # whole 64 KiB before this condition was looked at again.
                    #
                    # The min() here is tidiness, not correctness, and a
                    # mutation sweep is right that no test can tell it from
                    # max(): every caller of _drain answers with close=True,
                    # so anything read past `remaining` is read from a
                    # connection that is about to end. Said here rather than
                    # pinned with a test that could not honestly fail.
                    chunk = _read1(self.rfile, min(remaining, 1 << 16))
                    if not chunk:
                        return
                    remaining -= len(chunk)
            except OSError:
                return                  # nothing more is coming; refuse anyway
            finally:
                self.connection.settimeout(previous)

        def _handle_summarize(self):
            ctype = self.headers.get("Content-Type", "")
            if not ctype.startswith("multipart/form-data"):
                raise _BadRequest(
                    "send the file as multipart/form-data. This endpoint "
                    "summarizes uploaded bytes only; use the sift CLI for "
                    "files already on this machine.")
            qs = parse_qs(urlparse(self.path).query)
            # disclose_cache=False: the cache key includes the file's
            # digest, so "from cache" answers "has anyone summarized this
            # exact file here before?" for anyone holding the bytes — about
            # a file that is not theirs. The CLI keeps the note; the user
            # there owns the cache.
            eff = replace(config, vision=qs.get("vision", ["1"])[0] != "0",
                          disclose_cache=False)
            with tempfile.TemporaryDirectory(
                    prefix="sift-", dir=config.resolved_spool_dir()) as workdir:
                # The raw multipart body reaches disk chunk by chunk. Parsing
                # it from a memory map finds offsets without materializing the
                # upload; the selected file is then copied in bounded chunks.
                raw = Path(workdir) / ".sift-request-body"
                self._read_body_to(raw)
                filename, start, end = _multipart_file_span_from_path(raw, ctype)
                if start is None:
                    raise _BadRequest("no file part in upload")
                safe = _safe_name(filename)
                target = Path(workdir) / safe
                _copy_range(raw, target, start, end)
                raw.unlink(missing_ok=True)
                summ = summarize_path(target, eff, client, cache,
                                      display_name=filename or safe)
                # Never hand back the temp path: it is neither useful to the
                # client nor ours to disclose.
                # The name goes in before the cache entry is written, not
                # after: renaming here left the entry holding a temporary path
                # that a later CLI run reported as the file's own.
                return summ

    return Handler


def _safe_name(filename: Optional[str]) -> str:
    """A basename we are willing to create, from a name the client chose.

    The client controls this string. Directory components, traversal and the
    special names are stripped rather than trusted, and the length is bounded
    — a 300-character name raised ENAMETOOLONG, whose message named the temp
    directory back at the client.
    """
    raw = (filename or "").replace("\\", "/").split("/")[-1]
    raw = raw.replace("\x00", "").strip().strip(".")
    cleaned = "".join(c for c in raw if c.isprintable() and c not in '<>:"|?*')
    if not cleaned or cleaned in (".", ".."):
        return "upload"
    stem, dot, suffix = cleaned.rpartition(".")
    if dot and len(suffix) <= 16:
        return (stem[:100] or "upload") + "." + suffix
    return cleaned[:120]


def _allowed_hosts(host: str, port: int) -> "set[str]":
    """Host header values that mean 'this server', for rebinding defence.

    Binding to 0.0.0.0 means "every address this machine has", so those are
    exactly the names that mean this server — including the one the banner
    prints. Discarding them left --host 0.0.0.0 refusing its own printed URL,
    every LAN address, and therefore everyone: the flag bound to the world and
    then answered nobody.
    """
    names = {"localhost", "127.0.0.1", "[::1]", "::1"}
    if host in ("0.0.0.0", "::", ""):
        names.add("0.0.0.0")
        names.update(_own_addresses())
    elif host:
        names.add(host)
    return {f"{n}:{port}" for n in names if n} | {n for n in names if n}


def _own_addresses() -> "set[str]":
    """Every address this machine answers on, best effort."""
    import socket
    found = {"0.0.0.0"}
    try:
        name = socket.gethostname()
        found.add(name.lower())
        for info in socket.getaddrinfo(name, None):
            found.add(info[4][0].lower())
    except OSError:
        pass
    try:
        # The address a route to the outside would use, without sending
        # anything: connect() on UDP only fixes the local end.
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 9))     # TEST-NET-1, never routed
            found.add(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass
    return found


class _BadRequest(Exception):
    pass


def _health(config: Config, client: Optional[LLMClient]) -> dict:
    info = {
        "version": __version__,
        "llm_enabled": client is not None,
        "provider": getattr(client, "provider", None),
        "endpoint": getattr(client, "endpoint", None),
        "model": config.model or "(auto)",
        "kinds": len(SUPPORTED_KINDS),
    }
    if client is not None:
        try:
            info["available"] = client.available()
            if info["available"]:
                info["models"] = client.list_models()[:25]
        except Exception:  # noqa: BLE001
            info["available"] = False
    return info


def _boundary(ctype: str) -> bytes:
    """Extract the boundary parameter, whatever else follows it.

    Content-Type parameters may appear in any order (RFC 7231), so reading to
    the end of the header swallows a trailing '; charset=utf-8' into the
    boundary — after which no delimiter ever matches and the raw part markers
    end up inside the file we summarize.
    """
    for param in ctype.split(";")[1:]:
        name, _, value = param.partition("=")
        if name.strip().lower() == "boundary":
            value = value.strip()
            if value.startswith('"'):
                value = value[1:].split('"', 1)[0]
            if value:
                encoded = value.encode("latin-1", "replace")
                if len(encoded) > 200:
                    raise _BadRequest("multipart boundary is too long")
                return encoded
    raise _BadRequest("missing multipart boundary")


def _multipart_file_span(body, ctype: str):
    """Return ``(filename, start, end)`` for one multipart file.

    ``body`` may be bytes or a read-only mmap. Only bounded header slices are
    copied; file content remains where it is. A delimiter counts only at a
    line boundary and only with a legal suffix, so a boundary-shaped byte run
    inside a binary file is not silently treated as the end of the upload.
    """
    delim = b"--" + _boundary(ctype)

    def marker_positions():
        position = 0
        while True:
            position = body.find(delim, position)
            if position < 0:
                return
            at_line_start = (position == 0
                             or body[max(0, position - 2):position] == b"\r\n"
                             or body[max(0, position - 1):position] == b"\n")
            after = position + len(delim)
            suffix = body[after:after + 2]
            legal_suffix = (after == len(body) or suffix.startswith(b"\n")
                            or suffix == b"\r\n" or suffix == b"--")
            if at_line_start and legal_suffix:
                yield position
            position += max(1, len(delim))

    files = []
    markers = iter(marker_positions())
    try:
        marker = next(markers)
    except StopIteration:
        return None, None, None
    parts = 0
    while True:
        try:
            next_marker = next(markers)
            part_end = next_marker
            has_next = True
        except StopIteration:
            next_marker = None
            part_end = len(body)
            has_next = False
        start = marker + len(delim)
        if body[start:start + 2] == b"--":
            break                            # closing delimiter
        parts += 1
        if parts > MAX_MULTIPART_PARTS:
            raise _BadRequest(
                f"too many multipart parts (maximum {MAX_MULTIPART_PARTS})")
        if body[start:start + 2] == b"\r\n":
            start += 2
        elif body[start:start + 1] == b"\n":
            start += 1
        if body[max(start, part_end - 2):part_end] == b"\r\n":
            part_end -= 2
        elif body[max(start, part_end - 1):part_end] == b"\n":
            part_end -= 1

        header_limit = min(part_end, start + 64 * 1024)
        header_end = body.find(b"\r\n\r\n", start, header_limit)
        separator = 4
        if header_end < 0:
            header_end = body.find(b"\n\n", start, header_limit)
            separator = 2
        if header_end >= 0:
            header_text = bytes(body[start:header_end]).decode(
                "utf-8", "replace")
            filename = _header_param(header_text, "filename")
            if filename is not None:
                files.append((filename, header_end + separator, part_end))
                if len(files) > 1:
                    raise _BadRequest(
                        f"send one file per request; got {len(files)}")
        if not has_next:
            break
        marker = next_marker
    if not files:
        return None, None, None
    return files[0]


def _parse_multipart(body: bytes, ctype: str):
    """Compatibility helper for byte-oriented callers and parser tests."""
    filename, start, end = _multipart_file_span(body, ctype)
    if start is None:
        return None, None
    return filename, bytes(body[start:end])


def _multipart_file_span_from_path(path: Path, ctype: str):
    with path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as body:
            return _multipart_file_span(body, ctype)


def _copy_range(source: Path, target: Path, start: int, end: int) -> None:
    """Copy one multipart payload between known offsets in bounded chunks."""
    remaining = max(0, end - start)
    with source.open("rb") as src, target.open("xb") as dst:
        src.seek(start)
        while remaining:
            chunk = src.read(min(remaining, 1 << 20))
            if not chunk:
                raise _BadRequest("upload ended before its multipart boundary")
            dst.write(chunk)
            remaining -= len(chunk)


def _header_param(header: str, key: str) -> Optional[str]:
    """Read a header parameter. Returns None when the parameter is absent.

    Anchored to a parameter boundary, because a plain substring search found
    the key anywhere: "xfilename=" read as "filename=", a form field literally
    named "filename=zzz" turned that field into a file, and an unrelated
    header carrying the text was enough on its own. A parameter starts after
    ";" or at the beginning, and RFC 7230 allows space around the "=".
    """
    for star in (True, False):
        suffix = r"\*" if star else ""
        pattern = re.compile(r"(?:^|;)\s*" + re.escape(key) + suffix + r"\s*=\s*",
                             re.IGNORECASE)
        match = pattern.search(header)
        if not match:
            continue
        value = header[match.end():]
        if star:
            value = value.split(";")[0].strip()
            _, _, encoded = value.partition("''")
            return unquote(encoded or value)
        if value.startswith('"'):
            return value[1:].split('"', 1)[0]
        return value.split(";")[0].strip().strip("\r\n")
    return None


_MISSING_UI = ("<h1>sift</h1><p>Web UI files not found. Run from the project "
               "checkout so <code>webui/</code> is present.</p>")
