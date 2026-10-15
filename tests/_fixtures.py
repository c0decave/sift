"""Self-contained fixture builders so tests don't depend on examples/."""
from __future__ import annotations

import struct
import zipfile
import zlib
from pathlib import Path

CSV_TEXT = ("id,name,score,active\n"
            "1,Alice,9.5,true\n2,Bob,7.0,false\n3,Cleo,8.25,true\n")
TSV_TEXT = "a\tb\tc\n1\t2\t3\n4\t5\t6\n"
EML_TEXT = ("From: Sender <a@example.com>\r\nTo: Rcv <b@example.com>\r\n"
            "Subject: Quarterly numbers\r\nDate: Mon, 03 Aug 2026 09:00:00 +0000\r\n"
            "MIME-Version: 1.0\r\nContent-Type: text/plain\r\n\r\n"
            "Hello team, the revenue is up 12 percent this quarter.\r\n")
BAT_TEXT = ("@echo off\r\nREM cleanup\r\nset X=1\r\n"
            "del /q C:\\temp\\*\r\nxcopy A B /E /Y\r\n")
JSON_TEXT = '{"name": "svc", "port": 8080, "tags": ["a", "b"], "nested": {"x": 1}}'
HTML_TEXT = ("<!doctype html><html><head><title>Report</title>"
             "<meta name=description content='A test page'></head>"
             "<body><h1>Heading</h1><p>Some body text here.</p>"
             "<a href=#>link</a></body></html>")
LOG_TEXT = ("2026-01-01 00:00:00 INFO start\n"
            "2026-01-01 00:00:01 ERROR boom failed\n"
            "2026-01-01 00:00:02 WARNING careful now\n")


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def make_png(path: Path, w: int = 8, h: int = 5, rgb=(10, 20, 30)) -> Path:
    def chunk(typ: bytes, data: bytes) -> bytes:
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = (b"\x00" + bytes(rgb) * w) * h
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


_CTYPES = ('<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
           'package/2006/content-types"><Default Extension="xml" '
           'ContentType="application/xml"/></Types>')
_RELS = ('<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.'
         'org/package/2006/relationships"/>')


def make_docx(path: Path, paragraphs=("Title Para", "Second paragraph body.")) -> Path:
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ps = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    doc = f'<?xml version="1.0"?><w:document xmlns:w="{W}"><w:body>{ps}</w:body></w:document>'
    core = ('<?xml version="1.0"?><cp:coreProperties xmlns:cp="http://schemas.'
            'openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title>'
            '<dc:creator>Tester</dc:creator></cp:coreProperties>')
    _zip(path, {"[Content_Types].xml": _CTYPES, "_rels/.rels": _RELS,
                "word/document.xml": doc, "docProps/core.xml": core})
    return path


def make_xlsx(path: Path) -> Path:
    S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    wb = f'<?xml version="1.0"?><workbook xmlns="{S}"><sheets>' \
         f'<sheet name="Data" sheetId="1"/></sheets></workbook>'
    sst = f'<?xml version="1.0"?><sst xmlns="{S}" count="3" uniqueCount="3">' \
          f'<si><t>Item</t></si><si><t>Qty</t></si><si><t>Apples</t></si></sst>'
    sheet = f'<?xml version="1.0"?><worksheet xmlns="{S}"><sheetData>' \
            f'<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>' \
            f'<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>42</v></c></row>' \
            f'</sheetData></worksheet>'
    _zip(path, {"[Content_Types].xml": _CTYPES, "_rels/.rels": _RELS,
                "xl/workbook.xml": wb, "xl/sharedStrings.xml": sst,
                "xl/worksheets/sheet1.xml": sheet})
    return path


def _zip(path: Path, files: dict) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)


def compress_lzw(data: bytes, maxbits: int = 16) -> bytes:
    """``data`` as Unix compress writes it (a .Z file, block mode): LZW codes
    of 9 bits up to ``maxbits``, packed least significant bit first, the
    codes of one width padded to a whole group of that many bytes before the
    width grows or the table is cleared, as ncompress and gzip's unlzw read
    them. A full table is cleared at once, which ncompress does only when
    the ratio drops; both are valid streams."""
    out = bytearray(b"\x1f\x9d" + bytes([0x80 | maxbits]))
    acc = nacc = 0
    written = 0                          # bits of codes written so far
    origin = 0                           # where the current width began
    n_bits, maxmax = 9, 1 << maxbits
    maxcode = (1 << n_bits) - 1          # 511 to start, whatever maxbits is
    decoder_free, first = 257, True      # the reader's next entry number

    def emit(code):
        nonlocal acc, nacc, written
        acc |= code << nacc
        nacc += n_bits
        written += n_bits
        while nacc >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nacc -= 8

    def pad():
        nonlocal acc, nacc, written, origin
        gap = -(written - origin) % (n_bits * 8)
        written += gap
        nacc += gap
        while nacc >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nacc -= 8
        origin = written

    def reset():
        return {bytes([i]): i for i in range(256)}, 257

    table, free = reset()
    word = b""
    for value in data:
        grown = word + bytes([value])
        if grown in table:
            word = grown
            continue
        emit(table[word])
        if not first and decoder_free < maxmax:
            decoder_free += 1
        first = False
        if decoder_free > maxcode:
            # The readers' rule, to the letter, before whatever code comes
            # next: a width past the last one grows too (-b9 writes 10-bit
            # codes once the table is full), and only the width equal to
            # maxbits holds maxmax.
            pad()
            n_bits += 1
            maxcode = maxmax if n_bits == maxbits else (1 << n_bits) - 1
        if free < maxmax:
            table[grown] = free
            free += 1
        else:
            emit(256)                    # CLEAR: the reader starts over
            pad()
            n_bits, decoder_free = 9, 256
            maxcode = (1 << n_bits) - 1
            table, free = reset()
        word = bytes([value])
    if word:
        emit(table[word])
    if nacc:
        out.append(acc & 0xFF)
    return bytes(out)
