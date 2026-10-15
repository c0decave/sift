"""Web server: the security boundary, and the multipart parser behind it.

Every test here corresponds to something that was once exploitable or wrong.
A browser will happily POST to 127.0.0.1 from any page the user has open, so
"it only listens on loopback" is not a boundary — these are.
"""
import base64
import io
import json
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from sift.cache import SummaryCache
from sift.config import Config
from sift.server import (_BadRequest, _parse_multipart, _make_handler,
                         _BoundedServer, run_server)


class LiveServer:
    """The real handler on a real socket, with no LLM behind it."""

    def __init__(self, host="127.0.0.1", cache=None, **settings):
        self.config = Config(no_llm=True, quiet=True, **settings)
        self.host = host
        self.cache = cache
        self.username = "sift"
        self.password = "test-password"

    def __enter__(self):
        handler = _make_handler(self.config, None, self.host, 0, self.cache,
                                password=self.password)
        # The real server class, ceiling and all: a LiveServer that was a
        # plain ThreadingHTTPServer would test a server nobody runs.
        self.httpd = _BoundedServer((self.host, 0), handler)
        self.port = self.httpd.server_address[1]
        # The handler was built with port 0; rebuild it now that we know the
        # real port, so the Host allow-list matches what clients will send.
        self.httpd.RequestHandlerClass = _make_handler(
            self.config, None, self.host, self.port, self.cache,
            password=self.password)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        return False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def address(self) -> tuple:
        """For the tests that speak HTTP by hand rather than through urllib."""
        return self.host, self.port

    @property
    def authorization(self) -> str:
        raw = f"{self.username}:{self.password}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    @property
    def authorization_line(self) -> bytes:
        return f"Authorization: {self.authorization}\r\n".encode("ascii")

    def request(self, path="/api/health", data=None, headers=None, method=None,
                authenticated=True):
        request_headers = dict(headers or {})
        if authenticated:
            request_headers.setdefault("Authorization", self.authorization)
        req = urllib.request.Request(self.url + path, data=data,
                                     headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.read(), dict(exc.headers)
            finally:
                exc.close()

    def upload(self, filename: str, content: bytes, boundary: bytes = b"BB",
               ctype: str = None, headers=None):
        body = (b"--" + boundary + b"\r\n"
                b'Content-Disposition: form-data; name="file"; filename="'
                + filename.encode() + b'"\r\n\r\n' + content
                + b"\r\n--" + boundary + b"--\r\n")
        hdrs = {"Content-Type": ctype or
                f"multipart/form-data; boundary={boundary.decode()}"}
        hdrs.update(headers or {})
        return self.request("/api/summarize", data=body, headers=hdrs)


class SecurityTest(unittest.TestCase):
    def test_every_route_requires_basic_authentication(self):
        with LiveServer() as srv:
            status, body, headers = srv.request(authenticated=False)
        self.assertEqual(status, 401)
        self.assertIn("Basic", headers.get("WWW-Authenticate", ""))
        self.assertEqual(json.loads(body)["error"], "authentication required")

    def test_wrong_and_malformed_credentials_are_refused(self):
        wrong = base64.b64encode(b"sift:not-the-password").decode("ascii")
        with LiveServer() as srv:
            wrong_status, _, _ = srv.request(
                headers={"Authorization": "Basic " + wrong})
            malformed_status, _, _ = srv.request(
                headers={"Authorization": "Basic !!!"})
        self.assertEqual(wrong_status, 401)
        self.assertEqual(malformed_status, 401)

    def test_duplicate_authorization_headers_are_refused(self):
        import http.client

        with LiveServer() as srv:
            host, port = srv.address
            conn = http.client.HTTPConnection(host, port, timeout=10)
            conn.putrequest("GET", "/api/health")
            conn.putheader("Authorization", srv.authorization)
            conn.putheader("Authorization", srv.authorization)
            conn.endheaders()
            self.assertEqual(conn.getresponse().status, 401)
            conn.close()

    def test_valid_credentials_reach_the_requested_route(self):
        with LiveServer() as srv:
            status, body, _ = srv.request()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["llm_enabled"], False)

    def test_no_endpoint_reads_a_server_side_path(self):
        """Naming a path for the server to open is a file-read primitive.

        It was reachable unauthenticated, and — with CORS wide open — from any
        page the user had loaded. There is no allowlist that makes it safe, so
        the capability is gone rather than guarded.
        """
        with LiveServer() as srv:
            status, body, _ = srv.request(
                "/api/summarize", data=b'{"path": "/etc/passwd"}',
                headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertNotIn(b"root:", body)

    def test_cross_origin_post_is_refused(self):
        with LiveServer() as srv:
            status, _, _ = srv.upload("a.txt", b"hello",
                                      headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)

    def test_no_cors_headers_are_granted(self):
        with LiveServer() as srv:
            _, _, headers = srv.request("/api/health")
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_foreign_host_header_is_refused(self):
        """DNS rebinding: an attacker's name resolving to 127.0.0.1."""
        with LiveServer() as srv:
            status, _, _ = srv.request(
                "/api/health", headers={"Host": "evil.attacker.com"})
        self.assertEqual(status, 403)

    def test_own_origin_is_accepted(self):
        with LiveServer() as srv:
            status, _, _ = srv.upload("a.txt", b"x,y\n1,2\n",
                                      headers={"Origin": srv.url})
        self.assertEqual(status, 200)

    def test_oversize_upload_is_refused_by_declared_length(self):
        with LiveServer() as srv:
            status, _, _ = srv.request(
                "/api/summarize", data=b"x",
                headers={"Content-Type": "multipart/form-data; boundary=BB",
                         "Content-Length": str(999 * 1024 * 1024)})
        self.assertIn(status, (400, 413))

    def test_error_response_does_not_leak_internals(self):
        with LiveServer() as srv:
            status, body, _ = srv.request(
                "/api/summarize", data=b"not multipart at all",
                headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(status, 400)
        text = body.decode()
        self.assertNotIn("/tmp/", text)
        self.assertNotIn("Traceback", text)

    def test_temp_path_is_never_returned(self):
        with LiveServer() as srv:
            status, body, _ = srv.upload("report.csv", b"a,b\n1,2\n")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["filename"], "report.csv")
        self.assertNotIn("/tmp/", data["path"])


class UploadTest(unittest.TestCase):
    def test_a_normal_upload_is_summarized(self):
        with LiveServer() as srv:
            status, body, _ = srv.upload("t.csv", b"name,age\nada,36\n")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["kind"], "csv")
        self.assertEqual(data["size_bytes"], 16)

    def test_utf8_filename_survives(self):
        with LiveServer() as srv:
            _, body, _ = srv.upload("naïve 中文.txt", b"hi")
        self.assertEqual(json.loads(body)["filename"], "naïve 中文.txt")


class MultipartTest(unittest.TestCase):
    """The parser is hand-written, so each RFC corner gets a case."""

    def parse(self, body: bytes, ctype: str):
        return _parse_multipart(body, ctype)

    def _part(self, disposition: str, content: bytes, boundary=b"BB") -> bytes:
        return (b"--" + boundary + b"\r\nContent-Disposition: " +
                disposition.encode() + b"\r\n\r\n" + content +
                b"\r\n--" + boundary + b"--\r\n")

    def test_boundary_parameter_may_be_followed_by_others(self):
        """'; charset=utf-8' after the boundary used to become part of it."""
        body = self._part('form-data; name="f"; filename="a.txt"', b"AAAAAA",
                          boundary=b"----X1")
        name, data = self.parse(
            body, "multipart/form-data; boundary=----X1; charset=utf-8")
        self.assertEqual((name, data), ("a.txt", b"AAAAAA"))

    def test_filename_is_decoded_as_utf8(self):
        body = self._part('form-data; name="f"; filename="naïve 中文.txt"', b"hi")
        self.assertEqual(self.parse(body, "multipart/form-data; boundary=BB")[0],
                         "naïve 中文.txt")

    def test_rfc5987_encoded_filename(self):
        body = self._part("form-data; name=\"f\"; filename*=UTF-8''na%C3%AFve.txt",
                          b"hi")
        self.assertEqual(self.parse(body, "multipart/form-data; boundary=BB")[0],
                         "naïve.txt")

    def test_unquoted_filename(self):
        body = self._part("form-data; name=f; filename=plain.txt", b"hi")
        self.assertEqual(self.parse(body, "multipart/form-data; boundary=BB")[0],
                         "plain.txt")

    def test_a_plain_form_field_is_not_mistaken_for_a_file(self):
        body = (b'--BB\r\nContent-Disposition: form-data; name="note"\r\n\r\nxx'
                b'\r\n--BB\r\nContent-Disposition: form-data; name="f"; '
                b'filename="c.txt"\r\n\r\nDATA\r\n--BB--\r\n')
        self.assertEqual(self.parse(body, "multipart/form-data; boundary=BB"),
                         ("c.txt", b"DATA"))

    def test_two_files_are_refused_rather_than_silently_dropped(self):
        body = (b'--BB\r\nContent-Disposition: form-data; name="a"; '
                b'filename="1.txt"\r\n\r\nAA\r\n--BB\r\nContent-Disposition: '
                b'form-data; name="b"; filename="2.txt"\r\n\r\nBB\r\n--BB--\r\n')
        with self.assertRaises(_BadRequest):
            self.parse(body, "multipart/form-data; boundary=BB")

    def test_an_excessive_number_of_multipart_parts_is_refused(self):
        fields = b"".join(
            b'--BB\r\nContent-Disposition: form-data; name="x"\r\n\r\nx\r\n'
            for _ in range(129))
        body = (fields + b'--BB\r\nContent-Disposition: form-data; name="f"; '
                b'filename="a.txt"\r\n\r\nA\r\n--BB--\r\n')
        with self.assertRaises(_BadRequest):
            self.parse(body, "multipart/form-data; boundary=BB")

    def test_preamble_before_the_first_delimiter_is_ignored(self):
        body = b"ignore me\r\n" + self._part(
            'form-data; name="f"; filename="d.txt"', b"DD")
        self.assertEqual(self.parse(body, "multipart/form-data; boundary=BB"),
                         ("d.txt", b"DD"))

    def test_missing_boundary_is_a_bad_request(self):
        with self.assertRaises(_BadRequest):
            self.parse(b"whatever", "multipart/form-data")

    def test_binary_content_is_preserved_byte_for_byte(self):
        blob = bytes(range(256))
        body = self._part('form-data; name="f"; filename="b.bin"', blob)
        self.assertEqual(self.parse(body, "multipart/form-data; boundary=BB")[1],
                         blob)


class EighthAuditServerTest(unittest.TestCase):
    """`sift serve` had not been audited in seven rounds."""

    def test_a_missing_host_header_does_not_pass_the_gate(self):
        """"if host and host not in allowed" read as "check it when there is
        one" — and HTTP/1.1 requires one, so its absence is not a reason to
        skip the rebinding defence."""
        import http.client
        with LiveServer() as srv:
            host, port = srv.address
            conn = http.client.HTTPConnection(host, port, timeout=10)
            conn.putrequest("GET", "/api/health", skip_host=True,
                            skip_accept_encoding=True)
            conn.putheader("Authorization", srv.authorization)
            conn.endheaders()
            self.assertEqual(conn.getresponse().status, 403)
            conn.close()

    def test_binding_to_every_address_does_not_refuse_every_address(self):
        """--host 0.0.0.0 bound to the world and then 403'd the world,
        including the URL its own banner prints."""
        from sift.server import _allowed_hosts
        allowed = _allowed_hosts("0.0.0.0", 8769)
        self.assertIn("0.0.0.0:8769", allowed)
        self.assertIn("127.0.0.1:8769", allowed)
        self.assertIn("localhost:8769", allowed)

    def test_the_default_bind_still_allows_only_this_machine(self):
        from sift.server import _allowed_hosts
        allowed = _allowed_hosts("127.0.0.1", 8080)
        self.assertIn("127.0.0.1:8080", allowed)
        self.assertNotIn("evil.example.com:8080", allowed)

    def test_conflicting_content_lengths_are_refused(self):
        """Two lengths, or a length beside Transfer-Encoding, is how one
        request is made to mean two — and two responses came back on one
        connection."""
        import http.client
        with LiveServer() as srv:
            host, port = srv.address
            for headers in (
                    [("Content-Length", "5"), ("Content-Length", "9")],
                    [("Content-Length", "5"), ("Transfer-Encoding", "chunked")]):
                conn = http.client.HTTPConnection(host, port, timeout=10)
                conn.putrequest("POST", "/api/summarize")
                conn.putheader("Authorization", srv.authorization)
                conn.putheader("Content-Type",
                               "multipart/form-data; boundary=BB")
                for name, value in headers:
                    conn.putheader(name, value)
                conn.endheaders()
                conn.send(b"hello")
                self.assertEqual(conn.getresponse().status, 400, headers)
                conn.close()

    def test_an_unreadable_ui_file_still_produces_a_response(self):
        """An exception outside do_GET's handler sent the client zero bytes —
        no HTTP response at all — and a traceback to the console."""
        import os
        from unittest import mock
        from sift import server as server_module
        with LiveServer() as srv:
            target = server_module.WEBUI_DIR / "app.js"
            if not target.is_file():
                self.skipTest("web UI not present")
            mode = target.stat().st_mode
            os.chmod(target, 0)
            self.addCleanup(os.chmod, target, mode)
            status, body, _ = srv.request("/app.js")
        self.assertEqual(status, 500)
        self.assertTrue(body)

    def test_a_header_parameter_is_read_at_a_parameter_boundary(self):
        """A substring search found "filename=" anywhere: "xfilename=", a
        form field named "filename=zzz", an unrelated header."""
        from sift.server import _header_param
        self.assertIsNone(_header_param(
            'form-data; name="x"; xfilename="hidden.txt"', "filename"))
        self.assertIsNone(_header_param(
            'form-data; name="filename=zzz"', "filename"))
        self.assertEqual(_header_param(
            'form-data; name="f"; filename ="spaced.txt"', "filename"),
            "spaced.txt")
        self.assertEqual(_header_param(
            'form-data; name="f"; filename="real.txt"', "filename"), "real.txt")
        self.assertEqual(_header_param(
            "form-data; filename*=UTF-8''na%C3%AFve.txt", "filename"),
            "naïve.txt")

    def test_an_upload_summary_never_names_the_temporary_path(self):
        """The rename happened after summarize_path returned, so the cache
        entry held the temp path — and a later CLI run reported a file that
        does not exist."""
        with LiveServer() as srv:
            status, body, _ = srv.upload("report.csv", b"a,b\n1,2\n3,4\n")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["filename"], "report.csv")
        self.assertNotIn("sift-", data["path"])
        self.assertNotIn("/tmp", data["path"])


class SlowUploadTest(unittest.TestCase):
    """`timeout = 30` is a socket *idle* timeout: any byte resets it.

    So a client sending one byte every 29 seconds held a thread for as long
    as it liked, and twenty of them were twenty threads — and the drain added
    last round made it cheaper still, because its own deadline was per read.
    Both are whole-operation deadlines now.
    """

    def test_a_body_that_never_finishes_is_refused(self):
        """Refused *at the deadline*, not eventually.

        The earlier version of this test dripped for eight seconds, stopped,
        and asserted that a 400 arrived — which it did, from the socket's own
        idle timeout two seconds after the drip ended. It passed against a
        server whose deadline bounded nothing: `rfile.read(n)` returns only
        once it has all n bytes, looping over recv() internally, and each of
        those recv()s got a fresh idle timeout. A body under a megabyte is one
        read() call, so the deadline was evaluated once and then not again.
        Dripping a byte every 50 ms into a 100 kB body, with the deadline set
        to 3 seconds, was answered after 28.

        So the assertion is the elapsed time, and the drip never stops: if the
        deadline does not do the cutting, nothing else will.
        """
        for budget, tick in ((3.0, 0.05), (3.0, 0.5)):
            elapsed, reply = self._answer_after(budget, tick)
            self.assertTrue(reply.startswith(b"HTTP/1.1 400"), reply[:40])
            self.assertLess(elapsed, budget + 1.5,
                            f"drip every {tick}s ran {elapsed:.1f}s "
                            f"against a {budget}s deadline")

    def _answer_after(self, budget, tick):
        """Open an upload, drip into it forever, and time the reply."""
        import socket
        import threading
        import time
        from unittest import mock

        with mock.patch("sift.server._BODY_SECONDS", budget):
            with LiveServer() as srv:
                host, port = srv.address
                conn = socket.create_connection((host, port), timeout=60)
                self.addCleanup(conn.close)
                request = (b"POST /api/summarize HTTP/1.1\r\n"
                           b"Host: 127.0.0.1:%d\r\n" % port)
                request += srv.authorization_line
                request += (
                    b"Content-Type: multipart/form-data; boundary=BB\r\n"
                    b"Content-Length: 100000\r\n\r\n")
                conn.sendall(request)
                running = [True]

                def drip():
                    while running[0]:
                        time.sleep(tick)
                        try:
                            conn.sendall(b"x")
                        except OSError:
                            return

                threading.Thread(target=drip, daemon=True).start()
                # recv on this thread: timing it from inside a second one
                # measured the test's own scheduling, not the server's, and
                # reported 0.0 seconds for a reply that took 28.
                began = time.monotonic()
                conn.settimeout(60)
                try:
                    reply = conn.recv(40)
                except OSError as exc:
                    reply = repr(exc).encode()
                elapsed = time.monotonic() - began
                running[0] = False
                return elapsed, reply

    def test_the_deadline_is_the_one_doing_the_cutting(self):
        """Doubling the budget doubles the time. Without this, a drip cut off
        by some unrelated idle timeout would look exactly like a working
        deadline — which is how the previous test passed for two rounds."""
        import time
        long_run, _ = self._answer_after(6.0, 0.05)
        short_run, _ = self._answer_after(3.0, 0.05)
        self.assertGreater(long_run - short_run, 1.5,
                           f"6s deadline took {long_run:.1f}s, "
                           f"3s took {short_run:.1f}s")

    def test_an_ordinary_upload_is_unaffected(self):
        with LiveServer() as srv:
            status, body, _ = srv.upload("report.csv", b"a,b\n1,2\n3,4\n" * 500)
        self.assertEqual(status, 200)
        self.assertIn("rows", json.loads(body)["summary"])

    def test_the_drain_has_a_whole_operation_deadline(self):
        from sift.server import _DRAIN_SECONDS, _DRAIN_TOTAL_SECONDS
        self.assertGreater(_DRAIN_TOTAL_SECONDS, _DRAIN_SECONDS)
        self.assertLessEqual(_DRAIN_TOTAL_SECONDS, 30)


class SpoolDirTest(unittest.TestCase):
    """`sift serve` writes each upload to disk while it reads it.

    Where is not a detail on a host whose local disk is the small one and
    whose storage is a mount: the default is the system temporary directory,
    and a run of max_bytes-sized uploads lands squarely on the wrong volume.
    """

    def test_an_upload_is_written_where_it_was_told_and_then_removed(self):
        """Where the file lands, observed from inside the request.

        The first version of this watched the directory from a second thread
        at 1 ms intervals. The workdir exists only while the summary is being
        made, and with no model behind it that is sometimes under a
        millisecond — so the test passed alone and failed in the full suite,
        which is the worst kind of test there is. The observation now happens
        where the file actually is: in the call that receives it.
        """
        import shutil
        import tempfile
        from pathlib import Path
        from unittest import mock

        import sift.server as server

        spool = Path(tempfile.mkdtemp(prefix="spool-"))
        self.addCleanup(shutil.rmtree, spool, ignore_errors=True)
        seen = {}
        real = server.summarize_path

        def watching(path, *args, **kwargs):
            seen["parent"] = Path(path).parent
            seen["exists"] = Path(path).is_file()
            return real(path, *args, **kwargs)

        with mock.patch.object(server, "summarize_path", watching):
            with LiveServer(spool_dir=str(spool)) as srv:
                status, _, _ = srv.upload("report.csv", b"a,b\n1,2\n3,4\n" * 50)
        self.assertEqual(status, 200)
        self.assertTrue(seen.get("exists"), "the upload was not written")
        self.assertEqual(seen["parent"].parent, spool,
                         f"written to {seen['parent']}, not under {spool}")
        self.assertEqual(list(spool.iterdir()), [],
                         "the upload outlived the request")

    def test_upload_is_streamed_instead_of_materialized_with_write_bytes(self):
        """Path.write_bytes requires the complete upload to exist in RAM."""
        with mock.patch.object(
                Path, "write_bytes",
                side_effect=AssertionError("upload was materialized")):
            with LiveServer() as srv:
                status, body, _ = srv.upload(
                    "report.csv", b"a,b\n1,2\n3,4\n" * 500)
        self.assertEqual(status, 200, body)
        self.assertIn("rows", json.loads(body)["summary"])

    def test_a_spool_dir_that_cannot_be_used_is_refused_at_startup(self):
        """Not at the first upload, which could be days later."""
        import tempfile
        from pathlib import Path

        from sift.config import Config, ConfigError, validate

        for bad in ("/does/not/exist", "/etc/hostname"):
            with self.assertRaises(ConfigError):
                validate(Config(spool_dir=bad))
        good = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, good, True)
        validate(Config(spool_dir=good))          # must not raise

    def test_a_found_config_file_may_not_choose_it(self):
        """Same class as endpoint: it decides where other people's uploads
        are written, so a sift.toml in a directory you cd into does not get
        to set it."""
        from sift.config import _TRUSTED_ONLY
        self.assertIn("spool_dir", _TRUSTED_ONLY)


class SlowHeaderTest(unittest.TestCase):
    """The body deadline guarded the one phase an attacker need not enter."""

    def _abandoned_after(self, budget):
        import socket
        import threading
        import time
        from unittest import mock

        with mock.patch("sift.server._HEADER_SECONDS", budget):
            with LiveServer() as srv:
                host, port = srv.address
                conn = socket.create_connection((host, port), timeout=60)
                self.addCleanup(conn.close)
                # A request line and one unterminated header, then a drip.
                conn.sendall(b"POST /api/summarize HTTP/1.1\r\n"
                             b"Host: 127.0.0.1:%d\r\nX-Pad: " % port)
                running = [True]

                def drip():
                    while running[0]:
                        time.sleep(budget / 3)
                        try:
                            conn.sendall(b"x")
                        except OSError:
                            return

                threading.Thread(target=drip, daemon=True).start()
                began = time.monotonic()
                conn.settimeout(60)
                try:
                    conn.recv(40)
                except OSError:
                    pass
                elapsed = time.monotonic() - began
                running[0] = False
                return elapsed

    def test_headers_that_never_finish_are_cut_off(self):
        """`timeout = 30` is a socket idle timeout and the drip resets it.

        The request line is read with readline(), which loops over recv(), so
        60 bytes and a byte every 20 seconds held a thread indefinitely — and
        the body deadline never came into it, because a client that never
        finishes its headers never reaches the body.
        """
        self.assertLess(self._abandoned_after(3.0), 4.5)

    def test_the_header_deadline_is_the_one_doing_the_cutting(self):
        """Doubling the budget doubles the time, or something else did it."""
        long_run = self._abandoned_after(6.0)
        short_run = self._abandoned_after(3.0)
        self.assertGreater(long_run - short_run, 1.5,
                           f"6s took {long_run:.1f}s, 3s took {short_run:.1f}s")

    def test_an_ordinary_request_is_not_affected(self):
        with LiveServer() as srv:
            status, _, _ = srv.upload("t.csv", b"a,b\n1,2\n")
        self.assertEqual(status, 200)


class ConnectionCeilingTest(unittest.TestCase):
    """One thread per connection, with no ceiling, is the whole attack.

    400 sockets that had each sent 60 bytes held 400 threads with an 8 MB
    stack reservation apiece; at 3,000 the accept loop raised "can't start
    new thread" from inside socketserver and the server stopped answering
    anyone. Refusing the 65th caller is the smaller failure.
    """

    def test_past_the_ceiling_the_answer_is_503_not_silence(self):
        import socket
        from unittest import mock

        from sift.server import MAX_CONNECTIONS

        with mock.patch("sift.server.MAX_CONNECTIONS", 8):
            with LiveServer() as srv:
                host, port = srv.address
                held = []
                for _ in range(12):
                    conn = socket.create_connection((host, port), timeout=5)
                    self.addCleanup(conn.close)
                    # Opened and left mid-header, which is what holds a slot.
                    conn.sendall(b"GET /api/health HTTP/1.1\r\nX-Pad: ")
                    held.append(conn)
                last = held[-1]
                last.settimeout(5)
                try:
                    reply = last.recv(40)
                except OSError as exc:
                    reply = repr(exc).encode()
        self.assertTrue(reply.startswith(b"HTTP/1.1 503"), reply[:40])
        self.assertGreater(MAX_CONNECTIONS, 8, "the shipped ceiling is higher")

    def test_a_slot_is_returned_when_the_connection_ends(self):
        """Otherwise the ceiling is a countdown to a dead server."""
        with LiveServer() as srv:
            for _ in range(6):
                status, _, _ = srv.request("/api/health")
                self.assertEqual(status, 200)


class GetWithBodyTest(unittest.TestCase):
    def test_a_get_carrying_a_body_is_refused(self):
        """Answered and not drained, the body is parsed as the next request.

        Every POST path sets `close`; do_GET did not, so a second request
        smuggled inside the first one's body was served — past the origin
        check that had already run on the outer request.
        """
        import socket

        with LiveServer() as srv:
            host, port = srv.address
            conn = socket.create_connection((host, port), timeout=10)
            self.addCleanup(conn.close)
            smuggled = (b"GET /api/types HTTP/1.1\r\n"
                        b"Host: 127.0.0.1:%d\r\n" % port
                        + srv.authorization_line + b"\r\n")
            request = (b"GET /api/health HTTP/1.1\r\n"
                       b"Host: 127.0.0.1:%d\r\n" % port)
            request += srv.authorization_line
            request += (b"Content-Length: %d\r\n\r\n" % len(smuggled))
            conn.sendall(request + smuggled)
            conn.settimeout(5)
            seen = b""
            try:
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    seen += chunk
            except OSError:
                pass
        self.assertEqual(seen.count(b"HTTP/1.1"), 1, seen[:200])
        self.assertIn(b"400", seen[:20])


class BodyFramingTest(unittest.TestCase):
    def test_the_body_read_stops_at_content_length(self):
        """`min(remaining, 1 << 20)` as `max` swallows the next request.

        read1 returns whatever the buffer holds, so asking for a megabyte
        when three bytes remain hands back the pipelined request that
        followed the body — which is then never answered, and whose bytes end
        up inside the file being summarized. Measured: with `max`, one
        response instead of two and the second request gone.
        """
        import socket

        with LiveServer() as srv:
            host, port = srv.address
            body = (b"--BB\r\nContent-Disposition: form-data; name=\"file\"; "
                    b"filename=\"a.csv\"\r\n\r\nx,y\r\n1,2\r\n--BB--\r\n")
            second = (b"GET /api/types HTTP/1.1\r\n"
                      b"Host: 127.0.0.1:%d\r\n" % port
                      + srv.authorization_line + b"\r\n")
            conn = socket.create_connection((host, port), timeout=10)
            self.addCleanup(conn.close)
            request = (b"POST /api/summarize HTTP/1.1\r\n"
                       b"Host: 127.0.0.1:%d\r\n" % port)
            request += srv.authorization_line
            request += (
                b"Content-Type: multipart/form-data; boundary=BB\r\n"
                b"Content-Length: %d\r\n\r\n" % len(body))
            conn.sendall(request + body + second)
            conn.settimeout(8)
            seen = b""
            try:
                while True:
                    chunk = conn.recv(8192)
                    if not chunk:
                        break
                    seen += chunk
            except OSError:
                pass
        self.assertEqual(seen.count(b"HTTP/1.1"), 2, seen[:300])
        self.assertIn(b"kinds", seen)


class StartupCredentialsTest(unittest.TestCase):
    def test_startup_prints_the_bound_address_and_generated_credentials(self):
        out = io.StringIO()
        with mock.patch("sift.server.secrets.token_urlsafe",
                        return_value="generated-secret") as generate, \
             mock.patch.object(_BoundedServer, "serve_forever"), \
             redirect_stdout(out):
            run_server("127.0.0.1", 0, Config(no_llm=True), None)

        banner = out.getvalue()
        generate.assert_called_once_with(18)
        self.assertRegex(banner, r"http://127\.0\.0\.1:\d+")
        self.assertNotIn("http://127.0.0.1:0", banner)
        self.assertIn("user=sift", banner)
        self.assertIn("password=generated-secret", banner)


class ServerConstantsTest(unittest.TestCase):
    """The values themselves, which every other test patches out of the way."""

    def test_the_shipped_deadlines_are_the_ones_intended(self):
        from sift.server import (_BODY_SECONDS, _DRAIN_SECONDS,
                                 _DRAIN_TOTAL_SECONDS, _HEADER_SECONDS,
                                 MAX_CONNECTIONS)
        # A header phase that outlives the body deadline would make the
        # tighter of the two pointless: real headers arrive in one packet.
        self.assertLessEqual(_HEADER_SECONDS, 30)
        self.assertGreaterEqual(_HEADER_SECONDS, 5)
        self.assertLess(_HEADER_SECONDS, _BODY_SECONDS)
        self.assertGreater(_DRAIN_TOTAL_SECONDS, _DRAIN_SECONDS)
        self.assertGreaterEqual(MAX_CONNECTIONS, 16)
        self.assertLessEqual(MAX_CONNECTIONS, 512)

    def test_nothing_the_server_starts_can_outlive_it(self):
        """A non-daemon watchdog keeps the process alive for its full
        interval after Ctrl+C, and non-daemon handler threads keep it alive
        for as long as a client holds a connection."""
        import socket
        import threading

        from sift.server import _BoundedServer

        self.assertTrue(_BoundedServer.daemon_threads)
        with LiveServer() as srv:
            host, port = srv.address
            conn = socket.create_connection((host, port), timeout=5)
            self.addCleanup(conn.close)
            conn.sendall(b"GET /api/health HTTP/1.1\r\nX-Pad: ")
            deadline = time.monotonic() + 2
            timers = []
            while time.monotonic() < deadline and not timers:
                timers = [t for t in threading.enumerate()
                          if isinstance(t, threading.Timer)]
                time.sleep(0.01)
            self.assertTrue(timers, "the header watchdog never started")
            for timer in timers:
                self.assertTrue(timer.daemon, "watchdog is not a daemon")


if __name__ == "__main__":
    unittest.main()


class UploadNamingTest(unittest.TestCase):
    """The client chooses the filename; the server decides what to create."""

    def test_traversal_and_separators_are_stripped(self):
        from sift.server import _safe_name
        for given, expected in (
                ("../../etc/passwd", "passwd"),
                ("..\\..\\windows\\system32\\cmd.exe", "cmd.exe"),
                ("/etc/shadow", "shadow"),
                ("..", "upload"),
                ("", "upload"),
                (None, "upload"),
                ("normal.csv", "normal.csv")):
            self.assertEqual(_safe_name(given), expected, repr(given))

    def test_a_very_long_name_is_bounded(self):
        from sift.server import _safe_name
        name = _safe_name("x" * 300 + ".csv")
        self.assertLessEqual(len(name), 110)
        self.assertTrue(name.endswith(".csv"))

    def test_a_null_byte_does_not_survive(self):
        from sift.server import _safe_name
        self.assertNotIn("\x00", _safe_name("evil\x00.txt"))

    def test_the_real_extension_reaches_detection(self):
        """Written under its own name, so a .csv is detected as a table."""
        with LiveServer() as srv:
            _, body, _ = srv.upload("data.csv", b"id,name\n1,Ada\n2,Bob\n")
        self.assertEqual(json.loads(body)["kind"], "csv")

    def test_the_same_upload_twice_is_served_from_cache(self):
        """And the response does not say so.

        The cache key includes the file's digest, so "from cache" in a reply
        answers "has anyone summarized this exact file here before?" for
        anyone holding the bytes — about a file that is not theirs. The CLI
        keeps the note; the user there owns the cache.

        Checked on the cache itself rather than on the reply, because the
        reply is the thing that must not say it.
        """
        cache_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, cache_dir, ignore_errors=True)
        cache = SummaryCache(enabled=True, directory=cache_dir)
        with LiveServer(cache=cache) as srv:
            _, first, _ = srv.upload("again.csv", b"id,name\n1,Ada\n")
            _, body, _ = srv.upload("again.csv", b"id,name\n1,Ada\n")
        self.assertEqual(1, cache.hits, "the second upload did not hit")
        self.assertNotIn("from cache", json.loads(body)["notes"])
        # A hit and a miss are indistinguishable from outside, which is the
        # point: same summary, same fields, nothing naming the difference.
        self.assertEqual(json.loads(first)["summary"],
                         json.loads(body)["summary"])


class ConcurrentUploadTest(unittest.TestCase):
    """Twenty uploads at once, each with its own secret.

    The server answers each request on its own thread and the redaction
    remembers what it hid per file. Two things could go wrong and only one of
    them is visible without looking: an upload could carry another's
    password, or an upload could lose its own answer to another's thread.
    """

    COUNT = 20

    def _body(self, index):
        return (f"Uebergabe {index}\n"
                f"Der Dienstnutzer heisst svc_{index}.\n"
                f"Das Kennwort lautet Geheim{index:03d}Wort{index}!\n"
                f"Rechnung {4000 + index} ist offen.\n").encode()

    def test_no_upload_carries_another_upload_s_secret(self):
        import threading
        results, lock = {}, threading.Lock()
        with LiveServer() as srv:
            def work(index):
                status, body, _ = srv.upload(f"file_{index}.txt",
                                             self._body(index),
                                             boundary=f"B{index}".encode())
                with lock:
                    results[index] = (status, body.decode("utf-8", "replace"))

            threads = [threading.Thread(target=work, args=(i,))
                       for i in range(self.COUNT)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(self.COUNT, len(results))
        for index, (status, body) in sorted(results.items()):
            self.assertEqual(200, status, index)
            payload = json.loads(body)
            # The answer is about the file that was sent, not another one.
            self.assertEqual(f"file_{index}.txt", payload.get("filename"))
            self.assertNotIn(f"Geheim{index:03d}Wort{index}", body,
                             f"upload {index} published its own secret")
            for other in range(self.COUNT):
                if other != index:
                    self.assertNotIn(f"Geheim{other:03d}Wort{other}", body,
                                     f"upload {index} carries {other}'s")
            self.assertIn(f"svc_{index}", body)
