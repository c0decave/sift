"""Files a viewer showed nothing of (a review's viewer, 2,007 real
files): a mail whose one part is not text, an RTF or HTML file holding a NUL
byte, a single compressed stream, an archive format sift did not know -- and
the e-mail subject the heuristic summary wrote with repr().

Each class names the route the file took to "nothing" and what it gives now.
"""
import base64
import shutil
import tempfile
import unittest
from pathlib import Path

from sift.detect import detect
from sift.extract import extract

BIG = 25_000_000


class _Files(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return path

    def ex(self, name, data):
        path = self.write(name, data)
        return extract(path, detect(path), BIG)


_HEAD = ("From: Ops <ops@example.org>\r\nTo: dev@example.org\r\n"
         "Subject: {subject}\r\nDate: Mon, 1 Sep 2026 10:00:00 +0200\r\n"
         "Message-ID: <1@example.org>\r\nMIME-Version: 1.0\r\n")


def mail(content_type, body, extra="", subject="Quarterly key"):
    """One message, its Content-Type and body given; ``extra`` more headers."""
    return (_HEAD.format(subject=subject) + f"Content-Type: {content_type}\r\n"
            + extra + "\r\n" + body + "\r\n").encode()


#: What an enveloped S/MIME message carries: DER bytes, base64 on the wire.
_DER = b"\x30\x82\x01\x00" + bytes(range(256)) * 2
_DER_B64 = base64.encodebytes(_DER).decode()
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 50


class SingleNonTextPartTest(_Files):
    """A message that is one part, and that part is not text.

    email_msg._content returned what EmailMessage.get_content returns, and
    for application/* or image/* that is bytes: _ALTERNATIVE_MISMATCH.search
    then raised TypeError, extract()'s boundary swallowed it, and the viewer
    was handed no headers, no body and no attachment list -- for every
    S/MIME-encrypted mail. A message/rfc822 part is "multipart" to Python and
    neither get_body nor iter_attachments looks into it, so a mail that *is*
    a forwarded message came out with an empty body and no attachment.
    """

    def _headers_are_there(self, ex):
        self.assertNotIn("extractor_error", ex.metadata)
        headers = ex.metadata["headers"]
        self.assertIn("ops@example.org", headers["from"])
        self.assertEqual(headers["subject"], "Quarterly key")
        self.assertIn("Date", ex.text.split("## Body")[0])

    def test_an_smime_enveloped_mail_lists_its_one_part(self):
        ex = self.ex("smime.eml", mail(
            'application/pkcs7-mime; smime-type=enveloped-data; '
            'name="smime.p7m"', _DER_B64,
            "Content-Transfer-Encoding: base64\r\n"
            'Content-Disposition: attachment; filename="smime.p7m"\r\n'))
        self._headers_are_there(ex)
        self.assertEqual(ex.metadata["attachments"], [
            {"filename": "smime.p7m", "content_type": "application/pkcs7-mime",
             "size": len(_DER), "size_human": ex.metadata["attachments"][0]
             ["size_human"]}])
        self.assertEqual(ex.metadata["attachment_count"], 1)
        self.assertIn(str(len(_DER)), ex.metadata["attachments"][0]["size_human"])
        self.assertIn("Attachments: smime.p7m (application/pkcs7-mime",
                      ex.text)
        self.assertEqual(ex.text.split("## Body", 1)[1].strip(), "")

    def test_an_image_only_mail_lists_the_image(self):
        ex = self.ex("image.eml", mail(
            "image/png", base64.encodebytes(_PNG).decode(),
            "Content-Transfer-Encoding: base64\r\n"))
        self._headers_are_there(ex)
        [entry] = ex.metadata["attachments"]
        self.assertEqual((entry["filename"], entry["content_type"],
                          entry["size"]), ("(unnamed)", "image/png", len(_PNG)))
        self.assertNotIn("contents_are", entry)       # it is what it says
        self.assertNotIn("PNG", ex.text.split("## Body", 1)[1])

    def test_a_mail_that_is_a_forwarded_message_shows_the_message(self):
        inner = ("From: orig@example.net\r\nTo: ops@example.org\r\n"
                 "Subject: The original\r\n\r\nThe forwarded text.\r\n")
        ex = self.ex("fwd.eml", mail("message/rfc822", inner))
        self._headers_are_there(ex)
        [entry] = ex.metadata["attachments"]
        self.assertEqual(entry["content_type"], "message/rfc822")
        self.assertEqual(entry["filename"], "The original.eml")
        self.assertGreater(entry["size"], 0)
        self.assertIn("## Attached message", ex.text)
        self.assertIn("The original", ex.text)
        self.assertIn("    The forwarded text.", ex.text)

    def test_a_signed_mail_still_reads_its_text(self):
        """multipart/signed: the control -- it worked, and still does."""
        body = ("--b1\r\nContent-Type: text/plain\r\n\r\nSigned body text.\r\n"
                "--b1\r\nContent-Type: application/pkcs7-signature; "
                'name="smime.p7s"\r\nContent-Transfer-Encoding: base64\r\n'
                'Content-Disposition: attachment; filename="smime.p7s"\r\n\r\n'
                + _DER_B64 + "\r\n--b1--")
        ex = self.ex("signed.eml", mail(
            'multipart/signed; protocol="application/pkcs7-signature"; '
            'micalg=sha-256; boundary="b1"', body))
        self._headers_are_there(ex)
        self.assertIn("Signed body text.", ex.text)
        self.assertEqual([a["filename"] for a in ex.metadata["attachments"]],
                         ["smime.p7s"])

    def test_a_plain_mail_is_unchanged(self):
        ex = self.ex("plain.eml", mail("text/plain; charset=utf-8",
                                       "Hello, the plain body."))
        self._headers_are_there(ex)
        self.assertEqual(ex.metadata["attachments"], [])
        self.assertEqual(ex.metadata["body_format"], "text/plain")
        self.assertIn("Hello, the plain body.", ex.text)

    def test_a_text_part_sent_as_an_attachment_is_still_the_body(self):
        """The line drawn is the type, not the disposition: text is read."""
        ex = self.ex("att.eml", mail(
            "text/plain", "Notes in a file.",
            'Content-Disposition: attachment; filename="notes.txt"\r\n'))
        self.assertIn("Notes in a file.", ex.text)
        self.assertEqual(ex.metadata["attachments"], [])

    def test_a_type_python_has_no_handler_for_is_an_attachment(self):
        """get_content raised KeyError for "model/*", and the fallback decoded
        the bytes as the body -- a glTF file shown as mojibake text."""
        ex = self.ex("model.eml", mail("model/gltf-binary", _DER_B64,
                                       "Content-Transfer-Encoding: base64\r\n"))
        self._headers_are_there(ex)
        self.assertEqual([a["content_type"] for a in
                          ex.metadata["attachments"]], ["model/gltf-binary"])
        self.assertEqual(ex.text.split("## Body", 1)[1].strip(), "")

    def test_the_lone_part_is_checked_against_its_name(self):
        """A lone part is an attachment like any other: a "PDF" that is a
        Windows program is said to be one."""
        pe = bytearray(b"MZ" + b"\x00" * 0x3a + b"\x40\x00\x00\x00")
        pe += b"PE\x00\x00" + b"\x00" * 64
        ex = self.ex("pe.eml", mail(
            'application/pdf; name="Rechnung.pdf"',
            base64.encodebytes(bytes(pe)).decode(),
            "Content-Transfer-Encoding: base64\r\n"))
        [entry] = ex.metadata["attachments"]
        self.assertEqual(entry["contents_are"], "pe-executable")
        self.assertTrue(any("Rechnung.pdf" in w for w in ex.warnings))

    def test_an_empty_lone_part_is_listed_as_empty(self):
        ex = self.ex("empty.eml", mail("application/octet-stream", "",
                                       "Content-Transfer-Encoding: base64\r\n"))
        self._headers_are_there(ex)
        [entry] = ex.metadata["attachments"]
        self.assertEqual(entry["size"], 0)

    def test_damaged_base64_still_gives_the_headers(self):
        ex = self.ex("bad64.eml", mail(
            "application/pkcs7-mime", "@@@ not base64 !!!\r\nQUJD=\r\n=",
            "Content-Transfer-Encoding: base64\r\n"))
        self._headers_are_there(ex)
        self.assertEqual(len(ex.metadata["attachments"]), 1)

    def test_a_charset_python_does_not_know_still_gives_the_mail(self):
        """_content's fallback decoded with the same unknown charset that
        get_content had just failed on, and the LookupError ended the mail
        as the TypeError did."""
        ex = self.ex("charset.eml", mail(
            "text/plain; charset=x-no-such-charset", "Hello café"))
        self._headers_are_there(ex)
        self.assertIn("Hello caf", ex.text)

    def test_an_mbox_of_such_messages_is_read_as_text(self):
        """The mailbox reader took the bytes as a body and wrote b'\\x89PNG…'
        into the rendering."""
        one = mail("image/png", base64.encodebytes(_PNG).decode(),
                   "Content-Transfer-Encoding: base64\r\n")
        data = (b"From ops@example.org Mon Sep  1 10:00:00 2026\n"
                + one.replace(b"\r\n", b"\n")
                + b"\nFrom ops@example.org Mon Sep  1 11:00:00 2026\n"
                + mail("text/plain", "Second one.").replace(b"\r\n", b"\n"))
        ex = self.ex("box.mbox", data)
        self.assertNotIn("extractor_error", ex.metadata)
        self.assertEqual(ex.metadata["message_count"], 2)
        first = ex.metadata["messages"][0]
        self.assertEqual((first["attachments"], first["body_chars"]), (1, 0))
        self.assertNotIn("b'", ex.text)
        self.assertNotIn("PNG", ex.text)
        self.assertIn("Second one.", ex.text)

    def test_a_forward_whose_original_is_one_image_lists_the_image(self):
        inner = ("From: orig@example.net\r\nSubject: Scan\r\n"
                 "MIME-Version: 1.0\r\nContent-Type: image/png\r\n"
                 "Content-Transfer-Encoding: base64\r\n\r\n"
                 + base64.encodebytes(_PNG).decode())
        body = ("--B\r\nContent-Type: text/plain\r\n\r\nfyi\r\n"
                "--B\r\nContent-Type: message/rfc822\r\n\r\n" + inner
                + "\r\n--B--")
        ex = self.ex("fwdimg.eml", mail('multipart/mixed; boundary="B"', body))
        self.assertNotIn("extractor_error", ex.metadata)
        types = [a["content_type"] for a in ex.metadata["attachments"]]
        self.assertEqual(types, ["message/rfc822", "image/png"])
        self.assertIn("Scan", ex.text)
        self.assertNotIn("b'", ex.text)


_RTF = (rb"{\rtf1\ansi\ansicpg1252\deff0 {\fonttbl{\f0 Arial;}}\f0\fs24 "
        rb"Sehr geehrte Damen und Herren,\par anbei die Unterlagen.\par}")
_HTML = ('<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
         "<title>Bericht</title></head>\n<body><h1>Umsatz</h1><p>"
         + "Grüße aus München. " * 40 + "</p></body></html>\n")


class NulInTextTest(_Files):
    """An RTF or HTML file holding a NUL byte was binary.

    detect._looks_binary called any NUL in the first 64 KiB binary, before
    the text sniffers ever saw the file: WordPad ends an RTF with one, a
    test page pads its end with 8,192, and the rtf and html kinds were
    lost (3 of 5 real .rtf files, 1 of 25 .htm). Now "{\\rtf" is a magic
    number, NULs that pad the end of a file are padding, and NULs inside
    it are allowed where they are rare and the rest is clean text -- no
    other control character, and UTF-8, or at most a quarter high bytes.
    Measured on the 50,007 files under /usr/lib and /usr/share of an Arch
    host that hold a NUL and have no magic: 2 became text, both text with
    one stray NUL.
    """

    def kind(self, name, data):
        return detect(self.write(name, data)).kind

    def test_an_rtf_ending_in_a_nul_is_rtf_and_reads_as_text(self):
        ex = self.ex("brief.rtf", _RTF + b"\x00")
        self.assertEqual(ex.kind, "rtf")
        self.assertIn("Sehr geehrte Damen und Herren", ex.text)
        self.assertNotIn("\x00", ex.text)

    def test_an_rtf_with_a_nul_is_rtf_without_its_extension(self):
        self.assertEqual(self.kind("brief", _RTF + b"\x00"), "rtf")
        self.assertEqual(self.kind("brief.dat", _RTF + b"\x00" * 3), "rtf")

    def test_html_padded_with_nuls_is_html(self):
        ex = self.ex("seite.htm", _HTML.encode() + b"\x00" * 8192)
        self.assertEqual(ex.kind, "html")
        self.assertEqual(ex.metadata.get("title"), "Bericht")
        self.assertNotIn("\x00", ex.text)

    def test_a_rare_nul_inside_html_keeps_it_html(self):
        data = _HTML.encode()
        middle = len(data) // 2
        for name, body in (
                ("one.html", data[:middle] + b"\x00" + data[middle:]),
                ("three.html", data[:40] + b"\x00" + data[40:middle] + b"\x00"
                 + data[middle:-20] + b"\x00" + data[-20:]),
                ("latin1.htm", _HTML.encode("latin-1") + b"\x00")):
            with self.subTest(name):
                ex = self.ex(name, body)
                self.assertEqual(ex.kind, "html")
                self.assertIn("Umsatz", ex.text)
                self.assertNotIn("\x00", ex.text)

    def test_a_log_with_a_stray_nul_is_a_log(self):
        log = b"".join(b"2026-09-01 10:00:%02d INFO request %d served\n"
                       % (i % 60, i) for i in range(400))
        self.assertEqual(self.kind("app.log", log + b"\x00"), "log")
        self.assertEqual(self.kind("mid.log", log[:5000] + b"\x00"
                                   + log[5000:]), "log")

    def test_binaries_with_nuls_stay_binary(self):
        import random
        rnd = random.Random(7)
        table = b"".join(b"name%d\x00\x00\x00\x00value%d\x00" % (i, i)
                         for i in range(300))
        for name, data in (
                # grub's pxeboot.img: four bytes of code, then zeros. Three
                # of the four are high bytes and one is DEL.
                ("boot.img", b"\xb2\x7f\xea\x00\x82" + b"\x00" * 1019),
                ("table.dat", table),
                ("noise.bin", bytes(rnd.getrandbits(8) for _ in range(4096))),
                ("wide.txt", "plain text\n".encode("utf-32-le") * 20),
                ("half.txt", b"a\x00" * 10 + b"bcdefgh" * 50),
                ("zeros.bin", b"\x00" * 600),
                ("tiny.bin", b"\x01\x00")):
            with self.subTest(name):
                self.assertEqual(self.kind(name, data), "binary")

    def test_formats_with_magic_keep_their_kind(self):
        elf = b"\x7fELF\x02\x01\x01" + b"\x00" * 57 + b"code" * 10
        pe = b"MZ" + b"\x00" * 0x3a + b"\x40\x00\x00\x00" + b"PE\x00\x00" \
            + b"\x00" * 64
        png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 30)
        for name, data, kind in (("prog", elf, "elf-binary"),
                                 ("prog.exe", pe, "pe-executable"),
                                 ("bild.png", png, "image")):
            with self.subTest(name):
                self.assertEqual(self.kind(name, data), kind)

    def test_many_nuls_inside_text_stay_binary(self):
        """Rare means rare: a 4 KiB block of NULs in a log -- the shape a
        crash leaves -- is more than one per hundred bytes of the head."""
        log = b"2026-09-01 10:00:00 INFO served\n" * 200
        self.assertEqual(self.kind("crash.log", log[:3000] + b"\x00" * 4096
                                   + log[3000:]), "binary")

    def test_padding_is_padding_only_where_it_ends_the_file(self):
        """The head is 64 KiB. Past it the file goes on, and NULs that end
        the head are not known to end the file."""
        from sift.detect import HEAD_BYTES
        text = b"line of text\n" * ((HEAD_BYTES - 2000) // 13)
        body = text + b"\x00" * (HEAD_BYTES - len(text)) + b"\x01\x02" * 5000
        self.assertEqual(self.kind("long.txt", body), "binary")
        self.assertEqual(self.kind("short.txt", text + b"\x00" * 1500),
                         "text")


_LOG = b"".join(b"2026-09-01 10:%02d:00 INFO job %d finished\n" % (i % 60, i)
                for i in range(300))

#: hello.tar.Z as bsdtar 3.8 wrote it (bsdtar --format ustar -cZf): a tar of
#: hello.txt, 2,048 bytes when gzip -dc reads it.
_TAR_Z = bytes.fromhex(
    "1f9d9068cab061f3c6051d3c7400285cc8b0a1c38710234a9c48b1228c8b3668d00001"
    "e0a247181c3f5e0c29d2a34610316ac8a861a3c68d19326ed400012346cc183700d0ac"
    "c8b3a7cf9f401fd69943278c9c8e30e4bc7993302844a54c9d422c49f22347a958b36a"
    "ddca956bc0816f40dc7923870d19055f098a256b16ad40b563cb9eed4ab7aeddbb78f3"
    "eaddcbb7afdfbf80030b1e4cb8b0e1c388132b5eccb8b1e3c790234b9e4c592a")


def _tar(members, compression=""):
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:" + compression) as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class CompressedStreamTest(_Files):
    """A single compressed file -- app.log.gz, disk.img.bz2 -- is one
    member, named after the file without its compression suffix, with its
    decompressed size.

    fallback._archive_members listed zip and tar and nothing else: a .gz of
    a log went to tarfile, which could not read it as a tar, and came back
    with no members -- the same as an empty archive. A bz2 whose content
    begins with 512 zero bytes is even a tar to tarfile: an empty one. xz
    and zstd files were not archives at all without a tar inside, and .zst,
    .txz, .tzst and .Z had no kind.
    """

    def listing(self, name, data):
        ex = self.ex(name, data)
        self.assertNotIn("extractor_error", ex.metadata)
        self.assertEqual(ex.kind, "archive")
        return ex.metadata, ex

    def test_each_compressor_gives_one_member_and_its_size(self):
        import bz2
        import gzip
        import lzma
        streams = [("app.log.gz", gzip.compress(_LOG), "gzip"),
                   ("app.log.bz2", bz2.compress(_LOG), "bzip2"),
                   ("app.log.xz", lzma.compress(_LOG), "xz"),
                   ("app.log.Z", fx_compress(_LOG), "compress")]
        try:                            # Python 3.14 and later
            from compression import zstd
        except ImportError:
            zstd = None
        if zstd is not None:
            streams.append(("app.log.zst", zstd.compress(_LOG), "zstd"))
        for name, data, form in streams:
            with self.subTest(name):
                meta, ex = self.listing(name, data)
                self.assertEqual(meta["archive_members"], ["app.log"])
                self.assertEqual(meta["archive_member_count"], 1)
                self.assertEqual(meta["archive_member_sizes"], [len(_LOG)])
                self.assertEqual(meta["archive_format"], form)
                self.assertNotIn("archive_limit", meta)
                self.assertIn("app.log (", ex.text)

    def test_the_kind_comes_from_the_bytes_too(self):
        import bz2
        import lzma
        for name, data in (("blob", bz2.compress(_LOG)),
                           ("blob2", lzma.compress(_LOG)),
                           ("blob3", fx_compress(_LOG))):
            with self.subTest(name):
                meta, _ = self.listing(name, data)
                self.assertEqual(meta["archive_members"], [name])

    def test_a_stream_beginning_with_zeros_is_not_an_empty_tar(self):
        import bz2
        meta, _ = self.listing("disk.img.bz2",
                               bz2.compress(b"\x00" * 4096 + _LOG))
        self.assertEqual(meta["archive_members"], ["disk.img"])
        self.assertEqual(meta["archive_member_sizes"], [4096 + len(_LOG)])

    def test_a_compressed_tar_still_lists_the_tar(self):
        for name, compression in (("src.tgz", "gz"), ("src.tar.bz2", "bz2"),
                                  ("src.txz", "xz")):
            with self.subTest(name):
                meta, _ = self.listing(name, _tar(
                    [("a.txt", b"alpha"), ("b/c.txt", b"gamma!")],
                    compression))
                self.assertEqual(meta["archive_members"],
                                 ["a.txt", "b/c.txt"])
                self.assertEqual(meta["archive_member_sizes"], [5, 6])

    def test_an_empty_compressed_tar_is_an_empty_tar(self):
        meta, _ = self.listing("nothing.tar.gz", _tar([], "gz"))
        self.assertEqual(meta["archive_members"], [])
        self.assertEqual(meta["archive_member_count"], 0)

    def test_a_tar_z_is_one_tar_of_its_size(self):
        """tarfile does not read compress (.Z); the stream is listed as the
        tar it holds. The fixture is bsdtar's, so the reader is checked
        against the real format, not only against compress_lzw."""
        meta, _ = self.listing("hello.tar.Z", _TAR_Z)
        self.assertEqual(meta["archive_members"], ["hello.tar"])
        self.assertEqual(meta["archive_member_sizes"], [2048])

    def test_compress_at_every_code_width_and_after_a_clear(self):
        import random
        rnd = random.Random(5)
        noise = bytes(rnd.getrandbits(8) for _ in range(150_000))
        for maxbits in (9, 12, 16):
            with self.subTest(maxbits=maxbits):
                meta, _ = self.listing("noise.Z", fx_compress(noise, maxbits))
                self.assertEqual(meta["archive_member_sizes"], [len(noise)])

    def test_several_members_of_one_stream_add_up(self):
        import gzip
        meta, _ = self.listing("two.log.gz",
                               gzip.compress(b"first\n") + gzip.compress(_LOG))
        self.assertEqual(meta["archive_member_sizes"], [6 + len(_LOG)])

    def test_the_name_loses_only_a_compression_suffix(self):
        import gzip
        data = gzip.compress(_LOG)
        for name, member in (("x.tgz", "x.tar"), ("backup.2026.gz", "backup.2026"),
                             ("report.pdf", "report.pdf"), (".gz", ".gz")):
            with self.subTest(name):
                meta, _ = self.listing(name, data)
                self.assertEqual(meta["archive_members"], [member])

    def test_a_bomb_stops_at_the_ratio_and_says_so(self):
        """40 MiB of zeros in a few hundred bytes. Documents and source
        tarballs expand up to 176:1 in the census of real files; every
        stream past 1,000:1 there was a zero-filled image."""
        import bz2
        import time
        bomb = bz2.compress(b"\x00" * (40 << 20))
        self.assertLess(len(bomb), 2_000)
        start = time.process_time()
        meta, ex = self.listing("huge.img.bz2", bomb)
        self.assertLess(time.process_time() - start, 5.0)
        self.assertEqual(meta["archive_members"], ["huge.img"])
        self.assertEqual(meta["archive_member_sizes"], [None])
        self.assertIn("1,000", meta["archive_limit"])
        self.assertIn("at least", meta["archive_note"])

    def test_the_byte_budget_bounds_what_is_read(self):
        import gzip
        from unittest import mock
        from sift.extract import fallback
        with mock.patch.object(fallback, "MAX_STREAM_BYTES", 5_000):
            meta, _ = self.listing("app.log.gz", gzip.compress(_LOG))
        self.assertEqual(meta["archive_member_sizes"], [None])
        self.assertIn("5,000", meta["archive_limit"])

    def test_compress_past_its_input_budget_says_at_least(self):
        from unittest import mock
        from sift.extract import fallback
        data = fx_compress(_LOG * 3)
        self.assertGreater(len(data), 1_000)
        with mock.patch.object(fallback, "MAX_LZW_INPUT", 1_000):
            meta, _ = self.listing("app.log.Z", data)
        self.assertEqual(meta["archive_member_sizes"], [None])
        self.assertIn("only the first", meta["archive_note"])
        self.assertNotIn("archive_limit", meta)

    def test_a_stream_cut_short_says_what_it_read(self):
        import gzip
        import lzma
        for name, data in (("cut.log.gz", gzip.compress(_LOG)[:-40]),
                           ("cut.log.xz", lzma.compress(_LOG)[:-40])):
            with self.subTest(name):
                meta, _ = self.listing(name, data)
                self.assertEqual(meta["archive_members"], ["cut.log"])
                self.assertEqual(meta["archive_member_sizes"], [None])
                self.assertIn("at least", meta["archive_note"])

    def test_compress_has_no_end_marker_but_knows_an_impossible_code(self):
        """A .Z cut short reads as a shorter .Z -- the format has no end
        marker to miss. A first code past the 256 literals no compressor
        writes, and that is said."""
        meta, _ = self.listing("cut.log.Z", fx_compress(_LOG)[:-40])
        self.assertLess(meta["archive_member_sizes"][0], len(_LOG))
        meta, _ = self.listing("bad.log.Z", b"\x1f\x9d\x90\x2c\x01" * 3)
        self.assertEqual(meta["archive_member_sizes"], [None])
        self.assertIn("damaged", meta["archive_note"])

    def test_garbage_after_a_stream_is_said_not_counted(self):
        import gzip
        meta, _ = self.listing("tail.log.gz", gzip.compress(_LOG)
                               + b"not a gzip member at all")
        self.assertEqual(meta["archive_member_sizes"], [len(_LOG)])
        self.assertIn("after the stream", meta["archive_note"])

    def test_zstd_without_a_decoder_is_listed_without_a_size(self):
        from unittest import mock
        from sift.extract import fallback
        data = b"\x28\xb5\x2f\xfd" + b"\x00" * 20
        with mock.patch.object(fallback, "_zstd", None):
            meta, _ = self.listing("app.log.zst", data)
        self.assertEqual(meta["archive_members"], ["app.log"])
        self.assertEqual(meta["archive_member_sizes"], [None])
        self.assertIn("zstd", meta["archive_note"])

    def test_the_stream_is_read_under_the_file_deadline(self):
        import gzip
        from unittest import mock
        from sift import deadline
        from sift.extract import fallback
        path = self.write("app.log.gz", gzip.compress(_LOG))
        with mock.patch("sift.deadline.time.monotonic",
                        side_effect=[0.0] + [2.0] * 50):
            with deadline.limit(1.0):
                with self.assertRaises(deadline.DeadlineExceeded):
                    fallback._stream_listing(path, "gzip")


def fx_compress(data, maxbits=16):
    from tests._fixtures import compress_lzw
    return compress_lzw(data, maxbits)


def _ar(members, gnu_long=False):
    out = bytearray(b"!<arch>\n")
    table = b""
    if gnu_long:
        table = b"".join(name.encode() + b"/\n" for name, _ in members)
        out += (b"//" + b" " * 46 + b"%-10d`\n" % len(table)) + table
        if len(table) % 2:
            out += b"\n"
    offset = 0
    for name, data in members:
        if gnu_long:
            field = b"/%d" % offset
            offset += len(name) + 2
        else:
            field = name.encode()
        out += (field.ljust(16) + b"0".ljust(12) + b"0".ljust(6) * 2
                + b"100644".ljust(8) + b"%-10d" % len(data) + b"`\n") + data
        if len(data) % 2:
            out += b"\n"
    return bytes(out)


def _cpio(members):
    out = bytearray()
    for name, data in members + [("TRAILER!!!", b"")]:
        raw = name.encode() + b"\x00"
        head = (b"070701" + b"%08X" % 1 + b"%08X" % 0o100644 + b"0" * 8 * 4
                + b"%08X" % len(data) + b"0" * 8 * 4 + b"%08X" % len(raw)
                + b"0" * 8)
        out += head + raw
        out += b"\x00" * (-len(out) % 4)
        out += data
        out += b"\x00" * (-len(out) % 4)
    return bytes(out)


def _cab(names_sizes):
    import struct
    files = b""
    for name, size in names_sizes:
        files += struct.pack("<IIHHHH", size, 0, 0, 0, 0, 0x20) \
            + name.encode() + b"\x00"
    header = struct.pack("<4sIIIIIBBHHHHH", b"MSCF", 0, 0, 0, 44, 0, 3, 1, 1,
                         len(names_sizes), 0, 0, 0)
    folder = struct.pack("<IHH", 0, 0, 0)
    return header + folder + files


class ArchiveFormatTest(_Files):
    """.deb, .a, .cpio, .cab and .iso were binary: no magic sift knew, no
    extension it mapped. Each is an archive now; ar, cpio and cab are
    listed from their headers (the declared sizes), an ISO image by sift's
    reader of its directories (test_iso9660; a user request: it was
    named and not listed before)."""

    def listing(self, name, data):
        ex = self.ex(name, data)
        self.assertNotIn("extractor_error", ex.metadata)
        self.assertEqual(ex.kind, "archive")
        return ex.metadata

    def test_a_deb_lists_its_three_members(self):
        deb = _ar([("debian-binary", b"2.0\n"), ("control.tar.xz", b"x" * 7),
                   ("data.tar.xz", b"y" * 10)])
        for name in ("pkg_1.0_amd64.deb", "pkg"):
            with self.subTest(name):
                meta = self.listing(name, deb)
                self.assertEqual(meta["archive_members"],
                                 ["debian-binary", "control.tar.xz",
                                  "data.tar.xz"])
                self.assertEqual(meta["archive_member_sizes"], [4, 7, 10])
                self.assertEqual(meta["archive_format"], "ar")

    def test_a_static_library_with_long_names(self):
        meta = self.listing("libx.a", _ar(
            [("a_rather_long_object_name.o", b"\x7fELF" + b"\x00" * 8),
             ("b.o", b"\x7fELF")], gnu_long=True))
        self.assertEqual(meta["archive_members"],
                         ["a_rather_long_object_name.o", "b.o"])

    def test_an_empty_ar_is_an_empty_archive(self):
        meta = self.listing("empty.deb", b"!<arch>\n")
        self.assertEqual(meta["archive_member_count"], 0)

    def test_a_cut_ar_lists_what_is_there(self):
        deb = _ar([("debian-binary", b"2.0\n"), ("data.tar.xz", b"y" * 100)])
        meta = self.listing("cut.deb", deb[:-50])
        self.assertEqual(meta["archive_members"],
                         ["debian-binary", "data.tar.xz"])

    def test_a_cpio_lists_its_files_but_not_its_trailer(self):
        data = _cpio([("etc/hosts", b"127.0.0.1 localhost\n"),
                      ("init", b"#!/bin/sh\n")])
        for name in ("initrd.cpio", "initrd"):
            with self.subTest(name):
                meta = self.listing(name, data)
                self.assertEqual(meta["archive_members"], ["etc/hosts", "init"])
                self.assertEqual(meta["archive_member_sizes"], [20, 10])

    def test_an_odc_cpio_lists_its_files_too(self):
        def odc(name, data):
            raw = name.encode() + b"\x00"
            return (b"070707" + b"0" * 6 * 7 + b"0" * 11 + b"%06o" % len(raw)
                    + b"%011o" % len(data) + raw + data)
        meta = self.listing("old.cpio", odc("a.txt", b"alpha")
                            + odc("b.txt", b"beta!!") + odc("TRAILER!!!", b""))
        self.assertEqual(meta["archive_members"], ["a.txt", "b.txt"])
        self.assertEqual(meta["archive_member_sizes"], [5, 6])

    def test_a_cpio_header_without_a_name_is_damage(self):
        data = b"070701" + b"0" * 104 + b"TRAILER!!!\x00" + b"\x00" * 390
        meta = self.listing("junk.cpio", data)
        self.assertEqual(meta["archive_members"], [])
        self.assertIn("damaged", meta["archive_note"])

    def test_a_cab_lists_its_files(self):
        meta = self.listing("drivers.cab", _cab([("a.inf", 120),
                                                 ("b.sys", 4096)]))
        self.assertEqual(meta["archive_members"], ["a.inf", "b.sys"])
        self.assertEqual(meta["archive_member_sizes"], [120, 4096])

    def test_a_cab_that_claims_more_than_it_holds(self):
        import struct
        data = bytearray(_cab([("a.inf", 120)]))
        data[28:30] = struct.pack("<H", 500)          # cFiles: 500
        meta = self.listing("bad.cab", bytes(data))
        self.assertEqual(meta["archive_members"], ["a.inf"])

    def test_an_iso_image_whose_root_is_unreadable_says_so(self):
        """A primary volume descriptor whose root record is zeros: an
        archive, nothing listed, and a note that names the format."""
        iso = bytearray(40_000)
        iso[0x8000:0x8007] = b"\x01CD001\x01"
        for name in ("disc.iso", "disc.img"):
            with self.subTest(name):
                meta = self.listing(name, bytes(iso))
                self.assertEqual(meta["archive_members"], [])
                self.assertIn("ISO 9660", meta["archive_note"])

    def test_the_new_extensions_claim_an_archive(self):
        from sift.detect import EXT_MAP
        for ext in (".zst", ".txz", ".tzst", ".z", ".cab", ".cpio", ".iso",
                    ".deb"):
            with self.subTest(ext):
                self.assertEqual(EXT_MAP[ext][0], "archive")

    def test_text_named_like_an_archive_is_text(self):
        """The extension is a claim; the bytes decide, and say so."""
        for name in ("notes.deb", "notes.cpio", "notes.zst"):
            with self.subTest(name):
                path = self.write(name, "Plain notes, nothing packed.\n")
                self.assertEqual(detect(path).kind, "text")

    def test_text_that_starts_like_a_signature_is_text(self):
        for name, data in (("a.txt", "070701 is a date, not a cpio header\n"),
                           ("b.txt", "!<arch> is how an ar file starts\n"),
                           ("c.txt", "MSCF stands for nothing here\n")):
            with self.subTest(name):
                self.assertEqual(detect(self.write(name, data)).kind, "text")


def _icon(kind, images, reserved=0):
    """An ICO (``kind`` 1) or a CUR (2) of ``images``, each (width, data):
    the directory, then each image's data where its entry says."""
    import struct
    head = struct.pack("<HHH", reserved, kind, len(images))
    at, entries, body = 6 + 16 * len(images), b"", b""
    for width, data in images:
        entries += struct.pack("<BBBBHHII", width, width, 0, 0, 1, 32, len(data), at + len(body))
        body += data
    return head + entries + body


#: A 16x16 DIB as an icon holds it: every other byte of the header and of
#: the gray pixels is a NUL, as UTF-16BE text has them.
_DIB = (b"\x00\x28" + b"\x00\x10" * 2 + b"\x00\x20" * 6) + b"\x00\x7f" * 600


class IconTest(_Files):
    """.cur was binary: no extension sift mapped, no magic it knew; the
    viewer showed 24 cursors as a hex dump that both browsers draw. An .ico
    whose NULs alternate was UTF-16 text to the sniffer, and the viewer
    showed it as text. An icon and a cursor are known by their directory
    now: reserved 0, type 1 or 2, at least one image, each entry naming data
    after the directory and inside the file."""

    def test_an_icon_and_a_cursor_are_images_by_their_bytes(self):
        for kind, suffix in ((1, "ico"), (2, "cur")):
            data = _icon(kind, [(32, _PNG), (16, _DIB)])
            for name in (f"a.{suffix}", "a.bin", "noext"):
                with self.subTest(kind=kind, name=name):
                    det = detect(self.write(name, data))
                    self.assertEqual((det.kind, det.extra.get("detected_by")), ("image", "magic"))

    def test_an_icon_that_reads_like_utf16_is_an_image(self):
        from sift.detect import utf16_encoding
        data = _icon(1, [(16, _DIB)])
        self.assertEqual(utf16_encoding(data), "utf-16-be")   # what made it text
        self.assertEqual(detect(self.write("x.ico", data)).kind, "image")

    def test_the_cur_extension_claims_an_image(self):
        from sift.detect import EXT_MAP
        self.assertEqual(EXT_MAP[".cur"], ("image", None))

    def test_a_cursor_is_extracted_as_one(self):
        ex = self.ex("pointer.cur", _icon(2, [(32, _DIB)]))
        self.assertEqual((ex.kind, ex.metadata["format"], ex.mime), ("image", "CUR", "image/x-icon"))
        ex = self.ex("app.ico", _icon(1, [(32, _DIB)]))
        self.assertEqual((ex.kind, ex.metadata["format"], ex.mime), ("image", "ICO", "image/x-icon"))

    def test_bytes_that_only_begin_like_an_icon_are_not_one(self):
        import struct
        good = _icon(2, [(32, _DIB)])
        entry = 6
        broken = {
            "reserved": _icon(2, [(32, _DIB)], reserved=1),
            "type 3": good[:2] + b"\x03\x00" + good[4:],
            "no image": good[:4] + b"\x00\x00" + good[6:],
            "empty image": good[:entry + 8] + struct.pack("<I", 0) + good[entry + 12:],
            "inside the directory": good[:entry + 12] + struct.pack("<I", 10) + good[entry + 16:],
            "past the end": good[:entry + 12] + struct.pack("<I", len(good)) + good[entry + 16:],
            "cut in the directory": good[:14],
            "an MPEG pack header": b"\x00\x00\x01\xba" + good[4:],
        }
        for why, data in broken.items():
            with self.subTest(why):
                self.assertNotEqual(detect(self.write("x.bin", data)).kind, "image")

    def test_an_icon_named_as_one_but_damaged_is_still_claimed_by_its_name(self):
        """No magic, bytes that are binary: the name decides, as it did."""
        data = _icon(1, [(32, _PNG)], reserved=7)
        self.assertEqual(detect(self.write("x.ico", data)).kind, "image")
        self.assertEqual(detect(self.write("x.cur", data)).kind, "image")


class SubjectQuotingTest(_Files):
    """The heuristic summary of a mail wrote its subject with ``!r``
    (summarize._heur_email). repr() doubles a backslash and escapes a quote,
    so a password the body states and the subject repeats was no longer
    the value redaction had learned from the body: it reached the summary
    in its repr() spelling, which nothing hid."""

    def _summary(self, name, secret, subject=None):
        from sift.config import Config
        from sift.summarize import summarize_path
        path = self.write(name, mail(
            "text/plain; charset=utf-8",
            f"Hi team,\r\nthe new vault password: {secret}\r\nregards",
            subject=secret if subject is None else subject))
        return summarize_path(path, Config(no_llm=True, quiet=True,
                                           cache=False), None, None)

    def test_a_secret_subject_is_hidden_however_it_is_spelled(self):
        import dataclasses
        import json
        from corpus_secrets import SUBJECT_SECRET_CORE, SUBJECT_SECRETS
        for index, secret in enumerate(SUBJECT_SECRETS):
            with self.subTest(secret=secret):
                summ = self._summary(f"s{index}.eml", secret)
                said = json.dumps(dataclasses.asdict(summ), ensure_ascii=False)
                self.assertNotIn(SUBJECT_SECRET_CORE, said)
                self.assertIn("Email from", summ.summary)

    def test_an_ordinary_subject_is_quoted_as_it_is_written(self):
        """No repr(): the subject reads as the mail writes it, quotes and
        backslashes included, between typographic quotes."""
        subject = 'Re: "Q3" figures \\ Anna\'s draft'
        summ = self._summary("plain.eml", "nothing secret here",
                             subject=subject)
        self.assertIn(f"with subject “{subject}”", summ.summary)

    def test_a_document_title_is_hidden_the_same_way(self):
        """The twin: _heur_office wrote a document's title with ``!r``, and
        a docx titled with the password its text states leaked it."""
        import dataclasses
        import json
        import zipfile
        from corpus_secrets import SUBJECT_SECRET_CORE, SUBJECT_SECRETS
        from sift.config import Config
        from sift.summarize import summarize_path
        w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        for index, secret in enumerate(SUBJECT_SECRETS[:2]):
            with self.subTest(secret=secret):
                xml = (secret.replace("&", "&amp;").replace("<", "&lt;")
                       .replace('"', "&quot;"))
                path = self.dir / f"t{index}.docx"
                with zipfile.ZipFile(path, "w") as zf:
                    zf.writestr("[Content_Types].xml", "<Types/>")
                    zf.writestr("word/document.xml",
                                f'<?xml version="1.0"?><w:document xmlns:w='
                                f'"{w}"><w:body><w:p><w:r><w:t>Vault '
                                f"password: {xml}</w:t></w:r></w:p></w:body>"
                                "</w:document>")
                    zf.writestr("docProps/core.xml",
                                '<?xml version="1.0"?><cp:coreProperties '
                                'xmlns:cp="http://schemas.openxmlformats.org/'
                                'package/2006/metadata/core-properties" '
                                'xmlns:dc="http://purl.org/dc/elements/1.1/">'
                                f"<dc:title>{xml}</dc:title>"
                                "</cp:coreProperties>")
                summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                                   cache=False), None, None)
                said = json.dumps(dataclasses.asdict(summ), ensure_ascii=False)
                self.assertNotIn(SUBJECT_SECRET_CORE, said)
                self.assertIn("titled “", summ.summary)

    def test_a_mail_without_a_subject_says_nothing_about_one(self):
        from sift.config import Config
        from sift.summarize import summarize_path
        path = self.write("nosubj.eml", "From: a@example.org\r\n"
                          "To: b@example.org\r\n\r\nbody\r\n")
        summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                           cache=False), None, None)
        self.assertNotIn("subject", summ.summary)


if __name__ == "__main__":
    unittest.main()
