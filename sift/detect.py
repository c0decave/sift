"""File-type detection by extension *and* content sniffing.

Extension gives a hint; content decides. Binary magic numbers (PDF, ZIP-based
office formats, images, ELF/PE) override the extension, because a ``.txt`` that
is really a PDF should be treated as a PDF.
"""
from __future__ import annotations

import csv
import io
import json
import mimetypes
import re
import stat
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .zipguard import ZipLimitError, check as check_zip

HEAD_BYTES = 65536

# canonical-kind -> default mime
MIME = {
    "text": "text/plain",
    "markdown": "text/markdown",
    "log": "text/plain",
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "email": "message/rfc822",
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "yaml": "application/yaml",
    "toml": "application/toml",
    "ini": "text/plain",
    "xml": "application/xml",
    "html": "text/html",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "rtf": "application/rtf",
    "sqlite": "application/vnd.sqlite3",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "odt": "application/vnd.oasis.opendocument.text",
    "ods": "application/vnd.oasis.opendocument.spreadsheet",
    "odp": "application/vnd.oasis.opendocument.presentation",
    "epub": "application/epub+zip",
    "image": "image/*",
    "batch": "application/x-bat",
    "shell": "application/x-sh",
    "powershell": "application/x-powershell",
    "source-code": "text/x-source",
    "archive": "application/octet-stream",
    "pe-executable": "application/vnd.microsoft.portable-executable",
    "elf-binary": "application/x-elf",
    "office-legacy": "application/x-ole-storage",
    "empty": "inode/x-empty",
    "binary": "application/octet-stream",
    "unknown": "application/octet-stream",
}

# extension -> (kind, language-or-None)
EXT_MAP: dict[str, tuple[str, Optional[str]]] = {
    ".txt": ("text", None), ".text": ("text", None), ".rst": ("text", None),
    ".md": ("markdown", None), ".markdown": ("markdown", None),
    ".log": ("log", None), ".out": ("log", None),
    ".csv": ("csv", None), ".tsv": ("tsv", None), ".tab": ("tsv", None),
    ".eml": ("email", None), ".mbox": ("email", None),
    ".json": ("json", None), ".jsonl": ("jsonl", None), ".ndjson": ("jsonl", None),
    ".yaml": ("yaml", None), ".yml": ("yaml", None),
    ".toml": ("toml", None),
    ".ini": ("ini", None), ".cfg": ("ini", None), ".conf": ("ini", None),
    ".env": ("ini", None), ".properties": ("ini", None),
    ".xml": ("xml", None), ".plist": ("xml", None), ".svg": ("xml", None),
    ".html": ("html", None), ".htm": ("html", None), ".xhtml": ("html", None),
    ".pdf": ("pdf", None),
    ".docx": ("docx", None), ".odt": ("odt", None), ".rtf": ("rtf", None),
    ".sqlite": ("sqlite", None), ".sqlite3": ("sqlite", None),
    ".db": ("sqlite", None),
    ".ods": ("ods", None), ".odp": ("odp", None), ".epub": ("epub", None),
    ".xlsx": ("xlsx", None), ".xlsm": ("xlsx", None),
    ".pptx": ("pptx", None),
    ".doc": ("office-legacy", None), ".xls": ("office-legacy", None),
    ".ppt": ("office-legacy", None),
    ".png": ("image", None), ".jpg": ("image", None), ".jpeg": ("image", None),
    ".gif": ("image", None), ".bmp": ("image", None), ".webp": ("image", None),
    ".tif": ("image", None), ".tiff": ("image", None), ".ico": ("image", None),
    ".heic": ("image", None), ".heif": ("image", None), ".avif": ("image", None),
    ".cur": ("image", None),
    ".bat": ("batch", "batch"), ".cmd": ("batch", "batch"),
    ".sh": ("shell", "shell"), ".bash": ("shell", "shell"), ".zsh": ("shell", "shell"),
    ".ps1": ("powershell", "powershell"), ".psm1": ("powershell", "powershell"),
    ".py": ("source-code", "python"), ".pyw": ("source-code", "python"),
    ".js": ("source-code", "javascript"), ".mjs": ("source-code", "javascript"),
    ".ts": ("source-code", "typescript"), ".jsx": ("source-code", "javascript"),
    ".tsx": ("source-code", "typescript"),
    ".java": ("source-code", "java"), ".kt": ("source-code", "kotlin"),
    ".c": ("source-code", "c"), ".h": ("source-code", "c"),
    ".cpp": ("source-code", "cpp"), ".cc": ("source-code", "cpp"),
    ".hpp": ("source-code", "cpp"), ".cs": ("source-code", "csharp"),
    ".go": ("source-code", "go"), ".rs": ("source-code", "rust"),
    ".rb": ("source-code", "ruby"), ".php": ("source-code", "php"),
    ".swift": ("source-code", "swift"), ".scala": ("source-code", "scala"),
    ".sql": ("source-code", "sql"), ".r": ("source-code", "r"),
    ".pl": ("source-code", "perl"), ".pm": ("source-code", "perl"),
    ".lua": ("source-code", "lua"),
    ".zip": ("archive", None), ".tar": ("archive", None), ".gz": ("archive", None),
    ".tgz": ("archive", None), ".bz2": ("archive", None), ".xz": ("archive", None),
    ".7z": ("archive", None), ".rar": ("archive", None), ".jar": ("archive", None),
    ".zst": ("archive", None), ".zstd": ("archive", None),
    ".txz": ("archive", None), ".tzst": ("archive", None),
    ".tbz2": ("archive", None), ".tbz": ("archive", None),
    ".z": ("archive", None), ".taz": ("archive", None),
    ".cab": ("archive", None), ".cpio": ("archive", None),
    ".iso": ("archive", None), ".deb": ("archive", None),
    ".exe": ("pe-executable", None), ".dll": ("pe-executable", None),
    ".so": ("elf-binary", None), ".elf": ("elf-binary", None), ".o": ("elf-binary", None),
}


@dataclass
class Detection:
    path: Path
    size: int
    ext: str
    kind: str
    mime: str
    confidence: float
    is_binary: bool
    language: Optional[str] = None
    extra: dict = field(default_factory=dict)


def _kind_of(mode: int) -> str:
    """Name a non-regular file type, for an error message worth reading."""
    for test, name in ((stat.S_ISFIFO, "named pipe"), (stat.S_ISSOCK, "socket"),
                       (stat.S_ISCHR, "character device"),
                       (stat.S_ISBLK, "block device"), (stat.S_ISDIR, "directory")):
        if test(mode):
            return name
    return "special file"


#: Names that identify a file type on their own, no extension involved.
_BY_NAME = {
    "makefile": ("source-code", "make"),
    "gnumakefile": ("source-code", "make"),
    "dockerfile": ("source-code", "dockerfile"),
    "containerfile": ("source-code", "dockerfile"),
    "modelfile": ("source-code", "modelfile"),
    "rakefile": ("source-code", "ruby"),
    "gemfile": ("source-code", "ruby"),
    "vagrantfile": ("source-code", "ruby"),
    "cmakelists.txt": ("source-code", "cmake"),
    "justfile": ("source-code", "just"),
    "procfile": ("source-code", "procfile"),
}


def detect(path: Path) -> Detection:
    path = Path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        # Opening a FIFO blocks until a writer appears — forever, in practice,
        # and with -r that hangs the whole run with no output at all. Character
        # and block devices are equally not files to summarize.
        raise OSError(f"not a regular file ({_kind_of(info.st_mode)})")
    size = info.st_size
    ext = path.suffix.lower()
    with path.open("rb") as fh:
        head = fh.read(HEAD_BYTES)

    if size == 0:
        return Detection(path, 0, ext, "empty", MIME["empty"], 1.0, False)

    # What the name claims, worked out before anything else, so every route
    # out of this function can record it. Detection was right about a Windows
    # executable called Invoice_Q4_2024.pdf every time and never said the two
    # disagreed — which is the single most useful sentence a file recognizer
    # can produce, and it was the one thing missing.
    ext_kind, ext_lang = EXT_MAP.get(ext, (None, None))

    # 1) binary magic wins outright
    magic_kind = _magic(head, path, size)
    if magic_kind:
        lang = None
        return Detection(path, size, ext, magic_kind, MIME.get(magic_kind, MIME["binary"]),
                         0.98, magic_kind not in {"xml", "html", "rtf"}, lang,
                         extra={"detected_by": "magic", "claimed_kind": ext_kind})

    is_binary = _looks_binary(head, whole=size <= len(head))
    if path.name.lower() in _BY_NAME and ext_kind in (None, "text"):
        # The name is more specific than the extension. CMakeLists.txt has
        # one — ".txt" — and it says nothing, so the entry for it was dead.
        ext_kind, ext_lang = _BY_NAME[path.name.lower()]
    elif ext_kind is None:
        # Files whose name is their type. A Makefile has no extension and was
        # classified as plain text, which meant its "#" lines were not
        # comments to anything that asks — and a Makefile recipe saying
        # "# ignore all previous instructions and rebuild" was reported as
        # text addressed to an assistant.
        ext_kind, ext_lang = _BY_NAME.get(path.name.lower(), (None, None))

    if is_binary:
        kind = ext_kind if ext_kind in {"office-legacy", "archive", "pe-executable",
                                        "elf-binary", "image"} else "binary"
        return Detection(path, size, ext, kind, MIME.get(kind, MIME["binary"]),
                         0.6 if ext_kind else 0.4, True, None,
                         extra={"detected_by": "binary-heuristic",
                                "claimed_kind": ext_kind})

    # An extension that names a binary container, on content that is text,
    # is a claim the file itself contradicts: a .docx is a zip, a .pdf begins
    # "%PDF-", and neither survives being readable. Reaching here means the
    # magic did not match and the bytes are not binary, so the extension is
    # simply wrong — believing it filed "{\\rtf1 …}" as a Word document at 75%
    # confidence, handed it to the docx extractor, and left _masquerade
    # unable to speak because the claim and the verdict agreed.
    #
    # claimed_kind keeps the original either way: what the name promised is
    # the whole point of the warning downstream.
    claimed_kind = ext_kind
    if ext_kind in _MUST_BE_BINARY:
        ext_kind, ext_lang = None, None

    # 2) text: combine extension hint with content sniffing
    wide = utf16_encoding(head)
    text_sample = (head.decode(wide, errors="replace") if wide
                   else head.decode("utf-8", errors="replace"))
    sniffed, conf, meta = _sniff_text(text_sample, ext_kind)
    kind = sniffed or ext_kind or "text"
    # The sniffed language outranks the extension's: an extensionless script
    # has no extension to speak for it, which is what the shebang rule is
    # for, and a .bat file beginning "#!/bin/sh" is a shell script whatever
    # its name says. Without this every such file reported language=None and
    # was classified "Source code" rather than "Python source code".
    language = meta.get("language") or ext_lang
    if sniffed and sniffed != ext_kind and meta.get("language") is None:
        language = None if ext_kind and ext_kind != sniffed else ext_lang
    confidence = conf if sniffed else (0.75 if ext_kind else 0.5)
    return Detection(path, size, ext, kind, MIME.get(kind, MIME["text"]),
                     round(confidence, 2), False, language,
                     extra={"detected_by": "sniff" if sniffed else "extension",
                            "claimed_kind": claimed_kind, **meta})


# ----------------------------------------------------------------------
#: Kinds whose file format is a binary container or a binary stream. None of
#: them can be readable text, so an extension claiming one over text content
#: has been contradicted by the file.
_MUST_BE_BINARY = frozenset({
    "docx", "xlsx", "pptx", "odt", "ods", "odp", "epub", "pdf", "archive",
    "image", "pe-executable", "elf-binary", "office-legacy"})


def _magic(head: bytes, path: Path, size: Optional[int] = None) -> Optional[str]:
    """The kind the first bytes ``head`` say, or None; ``size`` the file's
    size where it is known (an icon's images lie inside the file)."""
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"{\\rtf"):
        # The first five characters of every RTF document, by definition,
        # and a document holding a NUL -- WordPad ends one with it -- was
        # binary before the text sniffer that knew this could look.
        return "rtf"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return _zip_kind(path)
    if head.startswith(b"\x89PNG\r\n\x1a\n") or head[:3] == b"\xff\xd8\xff" \
            or head[:6] in (b"GIF87a", b"GIF89a"):
        return "image"
    if _looks_like_bmp(head):
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image"
    if head[4:12] in (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypmif1", b"ftypavif"):
        return "image"
    if _looks_like_icon(head, len(head) if size is None else size):
        return "image"
    if head.startswith(b"\x7fELF"):
        return "elf-binary"
    if _looks_like_pe(head):
        return "pe-executable"
    if head[:16] == b"SQLite format 3\x00":
        # The format writes its own name into the first sixteen bytes, which
        # is as unambiguous a signature as any in this table.
        return "sqlite"
    if head[:4] == b"\xd0\xcf\x11\xe0":  # OLE2 (legacy doc/xls/ppt/msg)
        return "office-legacy"
    if head[:2] == b"\x1f\x8b":
        return "archive"
    if head[:6] == b"7z\xbc\xaf\x27\x1c" or head[:4] == b"Rar!":
        return "archive"
    if stream_format(head) or _archive_format(head):
        return "archive"
    return None


_ICON_DIRECTORY = struct.Struct("<HHH")        # reserved, type, count of images
_ICON_ENTRY = struct.Struct("<BBBBHHII")       # ..., the data's length and offset


def _looks_like_icon(head: bytes, size: int) -> bool:
    """An ICO (type 1) or a Windows cursor (type 2): reserved 0, at least
    one image, and each directory entry ``head`` holds naming data of some
    length that begins after the whole directory and inside the file of
    ``size`` bytes. Four bytes are too few a signature ("\0\0\1\0" also
    begins other data), and an icon's alternating NULs read as UTF-16 text
    to the sniffer below; its directory is what tells it. The entries
    ``head`` holds are asked, at least the first. As a host application's
    rasters._is_icon, which decides what the viewer draws."""
    if len(head) < _ICON_DIRECTORY.size + _ICON_ENTRY.size:
        return False
    reserved, kind, count = _ICON_DIRECTORY.unpack_from(head)
    if reserved or kind not in (1, 2) or not count:
        return False
    directory = _ICON_DIRECTORY.size + count * _ICON_ENTRY.size
    held = min(count, (len(head) - _ICON_DIRECTORY.size) // _ICON_ENTRY.size)
    for n in range(held):
        length, offset = _ICON_ENTRY.unpack_from(
            head, _ICON_DIRECTORY.size + n * _ICON_ENTRY.size)[6:]
        if not length or not directory <= offset < size:
            return False
    return True


def stream_format(head: bytes) -> Optional[str]:
    """The compressor of a single compressed stream, by its magic number:
    "gzip", "bzip2", "xz", "zstd" or "compress" (.Z) -- or None.

    More than the first bytes where they are few: bzip2's "BZh" and the
    level digit are three letters of text, so the block (or end-of-stream)
    magic after them is asked too; compress's flags byte has two bits that
    are always zero and a code width of 9 to 16.
    """
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:3] == b"BZh" and head[3:4] in _BZIP2_LEVELS \
            and head[4:10] in (b"1AY&SY", b"\x17rE8P\x90"):
        return "bzip2"
    if head[:6] == b"\xfd7zXZ\x00":
        return "xz"
    if head[:4] == b"\x28\xb5\x2f\xfd":
        return "zstd"
    if head[:2] == b"\x1f\x9d" and len(head) > 2 and not head[2] & 0x60 \
            and 9 <= head[2] & 0x1F <= 16:
        return "compress"
    return None


_BZIP2_LEVELS = tuple(bytes([digit]) for digit in b"123456789")

#: A cpio header in its two portable forms: "newc" (and "crc") of thirteen
#: eight-digit hex fields, and "odc" of octal ones.
_CPIO_HEADER = re.compile(rb"07070[12][0-9A-Fa-f]{104}|070707[0-7]{70}")


def _archive_format(head: bytes) -> Optional[str]:
    """An archive format sift lists or names without a zip or a stream:
    "tar", "ar" (a .deb, a static library), "cpio", "cab", "7z" or
    "iso9660" -- or None. Each asks more than the magic where the magic is
    short or could open a text; 7z's six bytes are no text's."""
    if head[:6] == b"7z\xbc\xaf\x27\x1c":
        return "7z"
    if _looks_like_tar(head):
        return "tar"
    if head[:8] == b"!<arch>\n" and (len(head) == 8
                                     or head[66:68] == b"`\n"):
        return "ar"
    if _CPIO_HEADER.match(head):
        return "cpio"
    if head[:8] == b"MSCF\x00\x00\x00\x00":
        return "cab"
    # The first volume descriptor, at sector 16: a type, "CD001" and
    # version 1 (UDF's "BEA01" begins the same place).
    if len(head) > 0x8006 and head[0x8001:0x8006] in (b"CD001", b"BEA01") \
            and head[0x8000] in (0, 1, 2, 3, 0xFF) and head[0x8006] == 1:
        return "iso9660"
    return None


def _looks_like_tar(head: bytes) -> bool:
    """"ustar" at 257 and a header checksum that adds up, as tarfile checks
    it: the bytes summed with the checksum field read as blanks, unsigned
    or signed."""
    if len(head) < 512 or head[257:262] != b"ustar":
        return False
    field = head[148:156].split(b"\x00", 1)[0].strip()
    try:
        stored = int(field, 8)
    except ValueError:
        return False
    block = head[:148] + b" " * 8 + head[156:512]
    unsigned = sum(block)
    signed = sum(b - 256 if b > 127 else b for b in block)
    return stored in (unsigned, signed)


#: The first bytes each format is required to start with. Used to ask
#: whether a file matches what its name and Content-Type claim — a different
#: and easier question than what it really is, and the more useful one for an
#: attachment: a .pdf that does not begin "%PDF-" is wrong whatever else it
#: turns out to be, and a crafted or damaged Windows executable that
#: _looks_like_pe declines to confirm still does not begin "%PDF-".
_SIGNATURES = {
    "pdf": (b"%PDF-",),
    "archive": (b"PK\x03\x04", b"PK\x05\x06", b"\x1f\x8b", b"7z\xbc\xaf\x27\x1c",
                b"Rar!", b"ustar"),
    "docx": (b"PK\x03\x04", b"PK\x05\x06"),
    "xlsx": (b"PK\x03\x04", b"PK\x05\x06"),
    "pptx": (b"PK\x03\x04", b"PK\x05\x06"),
    "odt": (b"PK\x03\x04",), "ods": (b"PK\x03\x04",), "odp": (b"PK\x03\x04",),
    "epub": (b"PK\x03\x04",),
    "image": (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a",
              b"BM", b"RIFF", b"II*\x00", b"MM\x00*"),
    "pe-executable": (b"MZ",),
    "elf-binary": (b"\x7fELF",),
    "office-legacy": (b"\xd0\xcf\x11\xe0",),
    "sqlite": (b"SQLite format 3\x00",),
    "rtf": (b"{\\rtf",),
}


def contradicts_its_name(promised: str, head: bytes) -> bool:
    """Whether the bytes cannot be what the name and type claim.

    Only formats with a required opening signature can be asked, and only in
    this direction: a file that does not begin the way its kind must begin is
    wrong, whatever it is. Asking the other question — what *is* it —
    needs every format to be recognised, and a hand-built or damaged
    executable is recognised by nothing.
    """
    starts = _SIGNATURES.get(promised)
    if not starts or not head:
        return False
    if promised == "image" and _looks_like_bmp(head):
        return False
    if promised == "image" and head[4:12].startswith(b"ftyp"):
        return False                     # HEIC and friends: signature at 4
    return not any(head.startswith(start) for start in starts)


def _looks_like_bmp(head: bytes) -> bool:
    """Two ASCII letters are not a signature.

    'BM' alone made "BMW and Mercedes are German car makers." a bitmap — and
    since magic beats the extension, nothing downstream could recover. A real
    BMP follows it with a little-endian file size, and the four reserved bytes
    at offset 6 are zero.
    """
    if head[:2] != b"BM" or len(head) < 18:
        return False
    if head[6:10] != b"\x00\x00\x00\x00":
        return False
    header_size = int.from_bytes(head[14:18], "little")
    return header_size in (12, 40, 52, 56, 64, 108, 124)


def _looks_like_pe(head: bytes) -> bool:
    """'MZ' plus a PE header where the DOS stub says it is.

    Without the second check, any text beginning "MZ" — a name, a note — was
    reported as a Windows executable with 98% confidence.
    """
    if head[:2] != b"MZ" or len(head) < 0x40:
        return False
    offset = int.from_bytes(head[0x3C:0x40], "little")
    if 0 < offset <= len(head) - 4:
        return head[offset:offset + 4] == b"PE\x00\x00"
    # The PE header sits beyond our sample; fall back to the DOS stub, which
    # real executables carry and prose does not.
    return b"This program cannot be run in DOS mode" in head


#: The OpenDocument member that names the format, rather than the mere
#: presence of a 'mimetype' entry — which made .ods, .odp and .epub all "odt",
#: and an epub then failed on a content.xml it does not have, yielding no text
#: and no metadata where the archive listing would at least have been true.
_ODF_MIME = {
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.presentation": "odp",
    "application/epub+zip": "epub",
}


def _looks_like_epub(names: "set[str]") -> bool:
    """An OPF package beside the container the spec requires."""
    return ("META-INF/container.xml" in names
            and any(n.endswith(".opf") for n in names))


def _zip_kind(path: Path) -> str:
    try:
        check_zip(path)
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if "mimetype" in names:
                try:
                    declared = zf.read("mimetype").decode("ascii", "replace").strip()
                except (KeyError, OSError):
                    declared = ""
                if declared in _ODF_MIME:
                    return _ODF_MIME[declared]
                if declared and not _looks_like_epub(names):
                    return "archive"
    except ZipLimitError:
        # Classification must not pay the cost the extractor refuses. The
        # fallback reports the declared count and the precise limit.
        return "archive"
    except Exception:  # noqa: BLE001
        # Not just BadZipFile and OSError. A corrupted central directory made
        # zipfile raise NotImplementedError("zip file version 9.1"), which
        # went straight past this and out of sift — one damaged .docx in a
        # directory aborted the whole run, and every file after it was never
        # looked at. Whatever the archive turns out to be unreadable *as*,
        # the honest answer here is that it is an archive.
        return "archive"
    if _looks_like_epub(names):
        # META-INF/container.xml is the marker every epub has, and the OPF is
        # what the extractor reads. Routing on the "mimetype" member alone
        # meant an epub without one — or with one that lies — went to the
        # binary fallback and came back as a hex dump of a format sift knows.
        return "epub"
    if "word/document.xml" in names:
        return "docx"
    # A ZIP merely containing an 'xl/' path is not a workbook — an archive with
    # xl/notes.txt was reported as "Excel spreadsheet with 0 sheet(s)". The
    # workbook part is the thing that makes it one.
    if "xl/workbook.xml" in names:
        return "xlsx"
    if "ppt/presentation.xml" in names:
        return "pptx"
    if "content.xml" in names and "styles.xml" in names:
        return "odt"
    return "archive"


_TEXT_CTRL = {0x08, 0x09, 0x0a, 0x0c, 0x0d, 0x1b}  # bs, tab, lf, ff, cr, esc


def utf16_encoding(head: bytes) -> Optional[str]:
    """Name the UTF-16 flavour of this sample, or None.

    Every other byte of UTF-16LE ASCII is a NUL, so the "contains NUL means
    binary" rule condemned the default output of PowerShell's Out-File and
    Notepad's "Unicode" — reported as binary, entropy measured, text never
    extracted. A BOM settles it; without one, the NULs land in a strict
    alternating pattern that arbitrary binary does not produce.
    """
    if head.startswith(b"\xff\xfe") and not head.startswith(b"\xff\xfe\x00\x00"):
        return "utf-16"
    if head.startswith(b"\xfe\xff"):
        return "utf-16"
    sample = head[:4096]
    if len(sample) < 16 or b"\x00" not in sample:
        return None
    pairs = len(sample) // 2
    even_nul = sum(1 for i in range(pairs) if sample[2 * i] == 0)
    odd_nul = sum(1 for i in range(pairs) if sample[2 * i + 1] == 0)
    # ASCII text in UTF-16 puts a NUL in the same half of nearly every pair.
    if odd_nul > pairs * 0.85 and even_nul < pairs * 0.05:
        return "utf-16-le"
    if even_nul > pairs * 0.85 and odd_nul < pairs * 0.05:
        return "utf-16-be"
    return None


#: Where a text holds NULs, at most one per this many of its bytes -- or one
#: in a text shorter than that. The measurement that bounds it is the other
#: condition: what is left must be clean text (_stray_nuls_in_text). Over
#: every file of /usr/lib and /usr/share of an Arch host (215,775), the
#: 50,007 that hold a NUL and have no magic, the rule turns 2 into text --
#: both text, each with one stray NUL; any bound from one in 50 to one in
#: 1,000 turned the same 2. One in 100 keeps an HTML page with three NULs
#: in 2 KB.
_RARE_NUL = 100


def _stray_nuls_in_text(head: bytes, whole: bool) -> bool:
    """Whether a head holding NULs is text that holds a few.

    WordPad ends an RTF with a NUL, a test page pads its end with 8,192 of
    them, and a crashed writer leaves one in a log; the rule that any NUL
    means binary lost all of them (3 of 5 real .rtf files, 1 of 25 .htm).
    NULs that end the file are padding (``whole``: the head is the whole
    file; the NULs that end a longer file's head need not end the file).
    The rest may hold NULs where they are rare (_RARE_NUL), and otherwise
    has to be cleaner text than a file without NULs needs to be: no control
    character but the ones text uses (DEL counts as one), and UTF-8 -- or,
    for a latin-1 or cp1252 text, at most a quarter of high bytes. That
    last bound is grub's pxeboot.img: four bytes of code, three of them
    high and one DEL, then 1,019 NULs.
    """
    rest = head.rstrip(b"\x00") if whole else head
    if not rest:
        return False
    if rest.count(b"\x00") > max(1, len(rest) // _RARE_NUL):
        return False
    text = rest.replace(b"\x00", b"")
    if _NOT_TEXT_CONTROL.search(text):
        return False
    for trim in (0, 1, 2, 3):
        try:
            text[:len(text) - trim].decode("utf-8")
            return True
        except UnicodeDecodeError:
            continue
    return (len(text) - len(text.translate(None, _HIGH_BYTES))) * 4 \
        <= len(text)


#: A control character text does not use (_TEXT_CTRL are the ones it does),
#: DEL included; and the bytes above ASCII.
_NOT_TEXT_CONTROL = re.compile(
    b"[" + re.escape(bytes(b for b in range(0x20) if b not in _TEXT_CTRL))
    + b"\x7f]")
_HIGH_BYTES = bytes(range(0x80, 0x100))


def _looks_binary(head: bytes, whole: bool = False) -> bool:
    """Whether the first HEAD_BYTES of a file are binary; ``whole`` when they
    are the whole file."""
    if utf16_encoding(head):
        return False
    if b"\x00" in head:
        return not _stray_nuls_in_text(head, whole)
    if not head:
        return False
    # Many control characters (other than common whitespace) => binary.
    ctrl = sum(1 for b in head if b < 0x20 and b not in _TEXT_CTRL)
    if ctrl / len(head) > 0.10:
        return True
    # High bytes are fine for text: valid UTF-8 (incl. CJK/emoji) or latin-1.
    # Trim up to 3 trailing bytes in case a multibyte char was cut at the buffer.
    for trim in (0, 1, 2, 3):
        try:
            head[:len(head) - trim].decode("utf-8")
            return False
        except UnicodeDecodeError:
            continue
    return False  # not UTF-8, but no NUL/control run -> treat as latin-1 text


#: A timestamp at the head of the line, in the shapes loggers actually emit.
_LOG_STAMP = re.compile(
    r"^[\[\(<]?\s*(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}"
    r"|\d{2}/\d{2}/\d{4}[ T]\d{2}:\d{2}"
    r"|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
    r"|\d{2}:\d{2}:\d{2}[.,]?\d*)")
_LOG_LEVEL = re.compile(
    r"\b(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|SEVERE|FATAL|CRITICAL)\b")
#: Below this, "most lines start with a time" is a coincidence.
_MIN_LOG_LINES = 5


def _looks_like_log(sample: str) -> bool:
    """Timestamps at the head of most lines, with levels somewhere in them.

    Both halves are required. A diary or a changelog has dates and no levels;
    a source file can mention INFO and has no timestamps. A log has both, on
    line after line, which is what makes it worth digesting rather than
    previewing.
    """
    lines = [ln for ln in sample.splitlines() if ln.strip()][:200]
    if len(lines) < _MIN_LOG_LINES:
        return False
    stamped = sum(1 for ln in lines if _LOG_STAMP.match(ln))
    levelled = sum(1 for ln in lines if _LOG_LEVEL.search(ln))
    if stamped / len(lines) >= 0.7 and levelled:
        return True
    return levelled / len(lines) >= 0.7 and stamped


def _headers_start_the_file(sample: str) -> bool:
    """True when the file opens with a header block, as a message does."""
    for line in sample.splitlines():
        if not line.strip():
            return False                 # a blank line before any header
        if line.startswith(("From ", ">From ")):
            continue                     # the mbox separator
        return bool(_EMAIL_HDR.match(line))
    return False


#: Headers that mean something delivered this, rather than someone typed it.
#: A note can carry Date and Subject; it does not carry a Received line or a
#: Message-ID, and it rarely carries From on the first line of a .txt.
_ROUTING_HEADERS = {"from", "received", "return-path", "message-id",
                    "delivered-to", "mime-version", "bcc"}

_EMAIL_HDR = re.compile(
    r"^(from|to|cc|bcc|subject|date|received|return-path|message-id|"
    r"mime-version|content-type|delivered-to)\s*:", re.IGNORECASE | re.MULTILINE)



#: What a shebang names, by interpreter. Matched on the program name, not as a
#: substring of the whole line: "sh" in "#!/usr/share/bin/prog" is not a shell.
_SHEBANG_KIND = {
    "python": ("source-code", "python"), "python2": ("source-code", "python"),
    "python3": ("source-code", "python"),
    "perl": ("source-code", "perl"), "ruby": ("source-code", "ruby"),
    "node": ("source-code", "javascript"), "php": ("source-code", "php"),
    "lua": ("source-code", "lua"), "Rscript": ("source-code", "r"),
    "sh": ("shell", "shell"), "bash": ("shell", "shell"),
    "zsh": ("shell", "shell"), "ksh": ("shell", "shell"),
    "dash": ("shell", "shell"), "ash": ("shell", "shell"),
    "csh": ("shell", "shell"), "tcsh": ("shell", "shell"),
    "fish": ("shell", "shell"),
    "pwsh": ("powershell", None), "powershell": ("powershell", None),
}


def _shebang_interpreter(line: str) -> str:
    """The program a "#!" line names, through /usr/bin/env and its options."""
    words = line[2:].strip().split()
    if not words:
        return ""
    name = words[0].rsplit("/", 1)[-1]
    if name == "env":
        for word in words[1:]:
            if not word.startswith("-") and "=" not in word:
                return word.rsplit("/", 1)[-1]
        return ""
    return name


#: Tokens only a batch file has. Any one of them decides.
_BATCH_STRONG = re.compile(
    r"^[ \t]*(?:@?echo\s+(?:off|on)\b|setlocal\b|endlocal\b|goto\s*:eof\b)",
    re.IGNORECASE | re.MULTILINE)
#: Tokens batch shares with other languages. Two different ones are needed,
#: because Perl's "goto &$AUTOLOAD" and a stray "call" are not evidence — and
#: with "\s*" after the anchor, which matches newlines, the anchor did not
#: even hold them to a line start.
_BATCH_WEAK = (
    re.compile(r"^[ \t]*goto\s+:?\w", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[ \t]*call\s+:?[\w.\\]+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"%[A-Za-z_]\w*%"),
    re.compile(r"^[ \t]*rem\s", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[ \t]*set\s+\"?[A-Za-z_]\w*=", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[ \t]*if\s+(?:not\s+)?exist\s", re.IGNORECASE | re.MULTILINE),
)


def _looks_like_batch(sample: str) -> bool:
    """Only for files whose extension does not already say batch.

    Two weak signals was not evidence: "Add the tool to %PATH%." beside "set
    SIFT_HOME=C:\\tools" is a README, and four of five hand-written Windows
    notes were classified as batch scripts. That is not cosmetic — "batch" is
    in _CODE_KINDS, so every line of the misclassified file beginning "#",
    "//" or ";" became an exempt comment and the injection check went quiet on
    it. A real script has a strong marker and typically four weak ones; the
    prose samples had none and exactly two.
    """
    if _BATCH_STRONG.search(sample):
        return True
    return sum(1 for pattern in _BATCH_WEAK if pattern.search(sample)) >= 3


#: A run of two or more spaces inside a line — how a fixed-width report
#: separates its columns, and something a one-column list never has.
_ALIGNED = re.compile(r"\S {2,}\S")


def _looks_fixed_width(sample: str) -> bool:
    """Whether the lines are aligned into columns with spaces.

    Asked only when no delimiter was found, to tell a fixed-width report
    from a one-column CSV. Both have no delimiter; only one of them has
    columns, and the columns are visible as the runs of spaces holding them
    apart. Most of the lines have to show it, so one wrapped sentence in a
    list of names does not turn the list into a report.
    """
    lines = [ln for ln in sample.splitlines()[:40] if ln.strip()]
    if len(lines) < 3:
        return False
    aligned = sum(1 for ln in lines if _ALIGNED.search(ln))
    return aligned >= max(2, len(lines) * 0.6)


def _sniff_text(sample: str, ext_kind: Optional[str]) -> tuple[Optional[str], float, dict]:
    stripped = sample.lstrip()
    low = stripped[:4096].lower()

    if stripped.startswith("{\\rtf"):
        # A real signature, even though the file is text: "{\rtf" is the
        # first four characters of every RTF document by definition. Read as
        # plain text the preview was "{\rtf1\ansi\deff0 {\fonttbl{\f0
        # Arial;}}", which says nothing about the letter inside it.
        return "rtf", 0.95, {}

    if low.startswith("<?xml"):
        if "<html" in low or "xhtml" in low:
            return "html", 0.9, {}
        return "xml", 0.9, {}
    if low.startswith("<!doctype html") or low.startswith("<html") or "<head" in low[:200]:
        return "html", 0.9, {}

    if stripped[:1] in "{[":
        try:
            # RecursionError, not just ValueError: json's decoder recurses per
            # nesting level, and 1,000 open brackets in a 2 kB file exceed the
            # interpreter's limit. It is not a ValueError, so it escaped
            # detect(), escaped summarize_path — whose docstring promises
            # never to raise for a bad file — and ended the whole run, losing
            # every file already summarized. The cheapest denial of service
            # in the tool, at two kilobytes.
            json.loads(stripped)
            return "json", 0.95, {}
        except RecursionError:
            return "json", 0.6, {"nesting": "deeper than the parser allows"}
        except ValueError:
            # maybe JSON lines
            first = stripped.splitlines()[0] if stripped.splitlines() else ""
            try:
                json.loads(first)
                return "jsonl", 0.75, {}
            except ValueError:
                pass

    header_block = sample[:2048]
    if _EMAIL_HDR.search(header_block):
        names = set(m.group(1).lower() for m in _EMAIL_HDR.finditer(header_block))
        # Two header-shaped lines was enough, whatever the extension. A
        # meeting note that opens "Date: / Subject: / Attendees:" is not an
        # email, and calling it one at 85% confidence handed it to a parser
        # that swallows every header-shaped line — so the attendees vanished
        # from the summary, the metadata and the prompt, and the reader was
        # told the note was "an email from an unnamed sender to an unnamed
        # recipient".
        #
        # What a message has and a note does not is routing: something put it
        # in the mailbox. Where the extension already says email, sniffing is
        # confirmation and this bar does not apply.
        routed = bool(names & _ROUTING_HEADERS)
        # And the block has to be at the top. _EMAIL_HDR searches the first
        # 2 KB; a parser reads headers from byte zero only. When the two
        # disagree sift asserted "email" and then parsed nothing: a README
        # with a "From:" line under a "# Contact" heading became an "Email
        # message (85% confidence)" with no headers at all, its own markdown
        # structure lost, summarized as "Email from an unnamed sender to an
        # unnamed recipient".
        if ext_kind != "email" and not _headers_start_the_file(sample):
            names = set()
        if ext_kind == "email" or (len(names) >= 2 and routed):
            return "email", 0.85, {"header_hits": len(names)}

    # The shebang answers the question outright, so it is asked first. It used
    # to be asked after the batch rule, and every Perl module on the machine
    # came back as a Windows batch script.
    if sample.startswith("#!"):
        interpreter = _shebang_interpreter(sample.splitlines()[0])
        if interpreter in _SHEBANG_KIND:
            kind, language = _SHEBANG_KIND[interpreter]
            return kind, 0.85, ({"language": language} if language else {})

    if _looks_like_batch(sample) and ext_kind in (None, "text", "batch"):
        return "batch", 0.8, {}

    if ext_kind in (None, "text", "csv", "tsv"):
        delim = _csv_delimiter(sample, ext_kind)
        if delim:
            return ("tsv" if delim == "\t" else "csv"), 0.8, {"delimiter": delim}
        if ext_kind in ("csv", "tsv") and _looks_fixed_width(sample):
            # A table whose columns are aligned with spaces is a fixed-width
            # report, and reading it as a delimited one loses it: a bank
            # statement came back as "3 rows and 1 column. Columns:
            # Kontoauszug März 2026 (code)", the statement itself nowhere.
            #
            # Only when the alignment is there. A one-column CSV — a list of
            # names exported from a spreadsheet — has no delimiter either and
            # is a perfectly ordinary table.
            return "text", 0.5, {
                "extension_contradicted":
                    f"the name says {ext_kind}, but the columns are aligned "
                    f"with spaces rather than separated by a delimiter; read "
                    f"as text, since a fixed-width report has no delimiter to "
                    f"split on"}

    # Asked after the structured formats, so a CSV of timestamps stays a CSV.
    # Log analysis was reached only through the .log and .out extensions, and
    # application.txt and output.txt are ordinary names for a log: a file
    # whose last two lines were ERROR and SEVERE was summarized as "Plain
    # text with 32 lines and ~222 words", with a 45-word preview that stopped
    # long before them. The machinery that would have surfaced those two
    # lines already existed and was simply never reached.
    if ext_kind in (None, "text") and _looks_like_log(sample):
        return "log", 0.7, {}

    if re.match(r"^\s*\[[^\]\n]+\]\s*$", sample, re.MULTILINE) and "=" in sample:
        if ext_kind in (None, "text", "ini"):
            return "ini", 0.7, {}

    return None, 0.0, {}


#: Below this, "consistent delimiter counts" is not evidence of a table.
#: "Hello, world\nGoodbye, world" met the old bar and was reported as a CSV
#: with two rows, two columns and invented col1/col2 headers, overriding an
#: explicit .txt extension.
_MIN_CSV_LINES = 4


#: Excel and several exporters write this as the first line so the
#: spreadsheet opens with the right separator. Nothing read it, so the line
#: became the header row and the real delimiter was never found: a German
#: price list came out as two columns called "sep=;" and "col2", with every
#: price split down the decimal comma.
_SEP_DIRECTIVE = re.compile(r"^\s*sep\s*=\s*(.)\s*$", re.IGNORECASE)


def sep_directive(sample: str) -> Optional[str]:
    """The delimiter a leading ``sep=`` line declares, if there is one."""
    for line in sample.splitlines():
        if not line.strip():
            continue
        match = _SEP_DIRECTIVE.match(line)
        return match.group(1) if match else None
    return None


def _splits_numbers(lines: "list[str]", delim: str) -> bool:
    """True when this delimiter appears to be cutting numbers in half.

    Only the comma doubles as a decimal mark, so only the comma is asked.
    What it looks for is a pair of fields that together spell one amount:
    the left one ends in a bare number, the right one begins with one to
    three digits, and what follows those digits is a separator or the end of
    the field — not a letter, which is what "50mg" is, and not a dot, which
    is what "19.99" is. Written without those two exclusions it dismantled a
    comma-separated clinical export into two columns because every dosage
    field began with digits.

    Each line is parsed on its own. Zipping a *filtered* row list against the
    unfiltered lines paired every row with the wrong line as soon as one line
    had no comma in it — so a title above a table changed the verdict, and
    the last rows fell off the end of the zip.
    """
    if delim != ",":
        return False
    cuts = looked = 0
    for line in lines:
        if not any(other in line for other in ";\t|"):
            continue                     # nothing else it could have been
        try:
            row = next(iter(csv.reader([line], delimiter=delim)), [])
        except csv.Error:
            continue
        if len(row) < 2:
            continue
        looked += 1
        for left, right in zip(row, row[1:]):
            left, right = left.strip(), right.strip()
            if not _ENDS_IN_NUMBER.search(left):
                continue
            digits = 0
            while digits < len(right) and right[digits].isdigit():
                digits += 1
            after = right[digits:digits + 1]
            if 1 <= digits <= 3 and not (after.isalpha() or after == "."):
                cuts += 1
                break
    return looked >= 3 and cuts >= looked * 0.8


#: The left half of a cut decimal: a bare number, at the end of the field,
#: with a separator or the start of the field in front of it.
_ENDS_IN_NUMBER = re.compile(r"(?:^|[;\t| ])[-+]?\d+$")


def _comma_cuts_amounts(lines: "list[str]") -> bool:
    """The same question as _splits_numbers, asked without its precondition.

    _splits_numbers skips any line with no ";", tab or "|" in it, because it
    exists to choose *between* candidates and a line with only commas offers
    no choice. That makes it blind to the case where the comma is the only
    candidate and is not a delimiter at all: a fixed-width bank statement has
    no other separator anywhere, so every line was skipped, the function
    returned False, and "-1200,13" became a column called "-1200" and one
    called "13" whose "range 13–55" is a range of cents.

    Kept separate rather than folded in, because dropping the precondition
    there would make an ordinary "1,2,3" integer table look like cut amounts
    and change a ranking that a good deal of this file depends on. Here the
    answer is only ever used together with _looks_fixed_width, and the two
    together are unambiguous.
    """
    cuts = looked = 0
    for line in lines:
        try:
            row = next(iter(csv.reader([line], delimiter=",")), [])
        except csv.Error:
            continue
        if len(row) < 2:
            continue
        looked += 1
        for left, right in zip(row, row[1:]):
            left, right = left.strip(), right.strip()
            if not _ENDS_IN_NUMBER.search(left):
                continue
            digits = 0
            while digits < len(right) and right[digits].isdigit():
                digits += 1
            after = right[digits:digits + 1]
            if 1 <= digits <= 3 and not (after.isalpha() or after == "."):
                cuts += 1
                break
    return looked >= 3 and cuts >= looked * 0.8


def _rank_delimiter(lines: "list[str]", delim: str):
    """(agreement, fields) for a candidate delimiter, or None.

    csv.reader rather than split(), so a quoted "Smith, John" is one field.
    """
    try:
        rows = [r for r in csv.reader(lines, delimiter=delim) if r]
    except csv.Error:
        return None
    counts = [len(r) for r in rows]
    if not counts:
        return None
    common = max(set(counts), key=counts.count)
    if common < 2:
        return None
    return counts.count(common) / len(counts), common


def _csv_delimiter(sample: str, ext_kind: Optional[str] = None) -> Optional[str]:
    lines = [ln for ln in sample.splitlines() if ln.strip()][:20]
    declared = sep_directive(sample)
    if declared:
        lines = lines[1:]
    floor = 2 if ext_kind in ("csv", "tsv") else _MIN_CSV_LINES
    if len(lines) < floor:
        return None
    if declared:
        # The directive says which character separates the fields; it does
        # not say the file is a table. Returning here jumped every guard
        # below — the column-count floor, the markdown check, the
        # is-this-prose checks — so a note beginning "sep=|" was reported as
        # a one-column CSV whose column was named after its first sentence.
        #
        # But it is not overruled on a consistency score either. Requiring
        # 0.9 threw away the declared separator of a bank statement with one
        # ragged line and of every legitimate one-column export — in the same
        # commit that argued against thresholds twenty lines below. The file
        # is telling us the answer; the guards below only decide whether it
        # is a table at all.
        delim = declared
    else:
        delim = None
    block = "\n".join(lines)
    # Scored here rather than sniffed. csv.Sniffer prefers the comma whenever
    # comma counts are consistent, and in a European export they are — the
    # decimal comma appears once per amount. A header row normally rescues
    # that, because its cells hold no commas; a headerless bank statement has
    # nothing to rescue it, and "2026-01-03;-42,50;Rewe Markt" came out as
    # two columns with every amount cut in half.
    #
    # The rule is the one a person uses: the delimiter that splits every line
    # into the same number of fields, and into more of them.
    # Ranked on agreement first, then field count — but a candidate is not
    # discarded for being below a threshold. A decimal comma achieves perfect
    # agreement by construction, one per amount, so on a file with any
    # irregularity at all the artefact scored 1.00 and the real delimiter
    # 0.75 and was thrown away before the comparison: a bank statement with
    # one four-field line came back as two columns with every amount cut in
    # half. The true delimiter is the one that carries the file's own
    # irregularity, so agreement alone is the wrong sole test.
    #
    # Field count breaks the tie, and a candidate that splits into more
    # fields is preferred once it is within reach of the best agreement.
    if delim is None:
        ranked = []
        for candidate in (",", ";", "\t", "|"):
            rank = _rank_delimiter(lines, candidate)
            if rank is None or rank[0] < 0.6:
                continue
            # A comma that splits amounts at their decimal mark reaches
            # perfect agreement by construction, one per amount — so raw
            # agreement systematically prefers the artefact over the real
            # separator on any file with an irregular line. Ranking by field
            # count instead swapped the error round: a text column with
            # commas in it splits into *more* fields than the semicolon that
            # actually separates the columns, so a parts list came back with
            # "Werkstoff;Preis;Menge" inside one field. Neither number is the
            # test; whether the split cuts numbers in half is.
            ranked.append(((not _splits_numbers(lines, candidate),
                            round(rank[0], 1), rank[1]), candidate))
        if ranked:
            delim = max(ranked)[1]
            if (delim == "," and _comma_cuts_amounts(lines)
                    and _looks_fixed_width(sample)):
                # A comma that cuts amounts at their decimal mark, in a file
                # whose columns are held apart by runs of spaces. Either
                # signal alone is not enough — "1,2,3" trips the first on a
                # perfectly ordinary integer table, and a one-column list
                # trips neither — but together they are a German fixed-width
                # statement, and reading it as a table split "-1200,13" into
                # a column called "-1200" and one called "13" whose "range
                # 13–55" is a range of cents.
                return None
    if delim is None:
        try:
            delim = csv.Sniffer().sniff(block, delimiters=",;\t|").delimiter
        except csv.Error:
            return None
    counts = [ln.count(delim) for ln in lines]
    # The count most lines share, not the first line's. Requiring line one to
    # carry the delimiter rejected every table with a title above it — a
    # measurement series under "Messreihe Februar 2026" came back as three
    # comma columns holding 21, 4;44 and 2, numbers that appear nowhere in
    # the file. The guard is here to keep prose out, and prose is answered by
    # the checks below, which read what the fields actually contain.
    common = max(set(counts), key=counts.count)
    if common < 1 or counts.count(common) < len(counts) * 0.6:
        return None
    if ext_kind in ("csv", "tsv"):
        return delim
    # Overriding a non-tabular extension needs more than one separator per
    # line: prose has commas too. Real tables have several columns, and their
    # fields are short.
    if min(counts) < 2:
        return None
    try:
        # Split the way the extractor will. Splitting on the raw delimiter made
        # a correctly-quoted "Smith, John" look like an extra column, so a real
        # contacts export was judged ragged and rejected.
        rows = [r for r in csv.reader(io.StringIO("\n".join(lines)),
                                      delimiter=delim) if r]
    except csv.Error:
        rows = [ln.split(delim) for ln in lines]
    if not rows:
        return None
    if _looks_like_markdown_table(lines):
        return None
    # Prose has no columns: its "fields" are sentence fragments of wildly
    # differing length, and the split is an artefact. A table's columns are
    # consistent down the file. Testing per column rather than over all fields
    # is what lets "id,severity,comment" keep a sentence in its last column —
    # an earlier, blunter rule rejected any table with one wordy column.
    if not _columns_are_consistent(rows):
        return None
    if _every_column_is_wordy(rows) or _reads_as_sentences(rows):
        return None
    return delim


def _looks_like_markdown_table(lines: list[str]) -> bool:
    """A pipe table is a table, but not one our CSV reader parses correctly.

    Splitting it on the pipe counts the |---|---| rule as a data row and, when
    the outer pipes are present, invents an empty column at each end. GitHub
    Markdown allows omitting the outer pipes, so requiring a leading one let
    the commonest written form straight through.
    """
    piped = [ln.strip() for ln in lines if "|" in ln]
    if len(piped) < len(lines) * 0.5:
        return False
    # A markdown rule sits directly under the header and has one cell per
    # column. Accepting a rule-shaped line anywhere in the file — and counting
    # "+" and "=" as rule characters — meant an ASCII ledger's
    # "=====|=====" separator, a trailing "---|---|---", or a record of null
    # placeholders ("-|-|-") turned a real pipe-delimited export into prose,
    # losing its rows, its columns and their names.
    rule = (len(piped) > 1 and set(piped[1]) <= set("|-: ") and "-" in piped[1]
            and _pipe_cells(piped[1]) == _pipe_cells(piped[0]))
    outer = sum(1 for ln in piped if ln.startswith("|") and ln.endswith("|"))
    return rule or outer >= len(piped) * 0.8


def _pipe_cells(line: str) -> int:
    """Cells in a pipe-delimited row, with the optional outer pipes removed."""
    return len(line.strip("|").split("|"))


def _reads_as_sentences(rows: list[list[str]]) -> bool:
    """Running text punctuated with the delimiter, rather than a table.

    In prose the delimiter is followed by a space and the fields are phrases:
    ", we ship", ", then we iterate". A CSV written with ", " separators has
    the space but short values, and a CSV with a sentence in its last column
    has neither — so both signals are needed, and neither is the full stop.
    Requiring a trailing full stop let a page of notes without one become
    "A CSV table with 3 rows and 3 columns: First, we ship, then we iterate".
    """
    spaced = sum(1 for r in rows for f in r[1:] if f[:1] in (" ", "\t"))
    tail_fields = sum(max(0, len(r) - 1) for r in rows) or 1
    if spaced < tail_fields * 0.8:
        return False
    phrases = sum(1 for r in rows for f in r if len(f.split()) >= 2)
    cells = sum(len(r) for r in rows) or 1
    return phrases >= cells * 0.5


def _columns_are_consistent(rows: list[list[str]]) -> bool:
    """Same number of fields on nearly every line, as a table has."""
    widths = [len(r) for r in rows]
    common = max(set(widths), key=widths.count)
    return widths.count(common) >= len(widths) * 0.8


def _every_column_is_wordy(rows: list[list[str]]) -> bool:
    """True when no column looks like a value — the signature of prose.

    A real table has at least one short, regular column: an id, a code, a
    price, a date. Four lines of "First, we ship, then we iterate." have none.
    """
    width = max(len(r) for r in rows)
    for col in range(width):
        cells = [r[col].strip() for r in rows if col < len(r) and r[col].strip()]
        if not cells:
            continue
        if max(len(c) for c in cells) <= 40 and \
                sum(c.count(" ") for c in cells) <= len(cells):
            return False                # this column holds values, not prose
    return True


def guess_mimetype(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"
