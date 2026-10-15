"""Stand-ins for the optional packages, so their code paths actually run.

sift ships three branches that only execute when ``pypdf``, ``PyYAML`` or
``Pillow`` is installed, and the README promises they "raise fidelity when
present". None of them had ever executed: the machine this was written on
had none of the three and PyPI was unreachable, so every test took the
ImportError path. The opposite happens too — a machine with PyYAML ran the
other branch under two tests about sift without it — so missing() takes a
package away as surely as installed() puts a stand-in in its place.

What is under test is sift's own handling — the types these libraries really
return, the shapes they really produce — not the libraries. So the stubs
reproduce the awkward parts on purpose: pypdf hands back ``TextStringObject``
and ``IndirectObject`` rather than ``str``, ``metadata`` may be ``None``, a
page's ``extract_text()`` may return ``None``, and a camera writes
``UserComment`` as bytes behind an eight-byte character-code prefix.
"""
from __future__ import annotations

import io
import json
import sys
import types
from contextlib import contextmanager


# ---- pypdf ------------------------------------------------------------
class TextStringObject(str):
    """What pypdf returns for a string entry — a str subclass, but not a str
    literal, and code that type-checks rather than coerces trips over it."""


class IndirectObject:
    """A reference pypdf resolves lazily; only str() gets you the value."""

    def __init__(self, value) -> None:
        self._value = value

    def __str__(self) -> str:
        return str(self._value)


class _Page:
    def __init__(self, text) -> None:
        self._text = text

    def extract_text(self, visitor_operand_before=None):
        return self._text          # may be None, as pypdf's can be

    def extract_xform_text(self, xform, *args, **kwargs):
        return ""                  # the stand-in's pages draw no forms


class _PdfReader:
    """Reads a JSON description of a document, so a test can state the shape
    it wants without needing a real PDF writer."""

    def __init__(self, stream) -> None:
        spec = json.loads(stream.read().decode("utf-8", "replace"))
        self.pages = [_Page(t) for t in spec["pages"]]
        info = spec.get("metadata")
        if info is None:
            self.metadata = None
            return
        self.metadata = {}
        if "title" in info:
            self.metadata["/Title"] = TextStringObject(info["title"])
        if "author" in info:
            self.metadata["/Author"] = IndirectObject(info["author"])
        if "producer" in info:
            self.metadata["/Producer"] = info["producer"]


def _not_for_json(*args, **kwargs):
    raise AssertionError("the stand-in reads JSON: pypdf's parsing of a PDF is never reached")


# What sift reads with of pypdf's PdfReader (sift.extract.pdf._READER_NEEDED): the stand-in has it, as a pypdf sift
# reads with does, and never reaches it.
for _name in ("_read_pdf15_xref_stream", "_read_xref", "_read_xref_subsections", "_get_object_from_stream",
              "_rebuild_xref_table", "_find_pdf_objects", "_find_pdf_trailers", "get_object",
              "cache_get_indirect_object", "cache_indirect_object", "read_object_header"):
    setattr(_PdfReader, _name, _not_for_json)
_PdfReader._XREF_STREAM_TRAILER_KEYS = ()


def pdf_bytes(pages, metadata=None) -> bytes:
    """The blob _PdfReader understands. Not a PDF; the reader is the stub."""
    return json.dumps({"pages": pages, "metadata": metadata}).encode()


# ---- PyYAML -----------------------------------------------------------
class YAMLError(Exception):
    pass


def _scalar(text: str):
    text = text.strip()
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text in ("null", "~", ""):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text.strip("\"'")


def _block(lines, start: int, indent: int):
    """Enough of YAML's block syntax for the shapes a config file has.

    Not a YAML implementation — a stand-in for one, so the tests can drive
    sift's handling of parsed output. Anything it cannot read raises
    YAMLError, which is what PyYAML does and what sift catches.
    """
    if start < len(lines) and lines[start][1].startswith("- "):
        out, i = [], start
        while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
            out.append(_scalar(lines[i][1][2:]))
            i += 1
        return out, i
    out, i = {}, start
    while i < len(lines) and lines[i][0] == indent:
        depth, line = lines[i]
        if ":" not in line:
            raise YAMLError(f"could not parse: {line[:40]}")
        key, _, rest = line.partition(":")
        if rest.strip():
            out[key.strip()] = _scalar(rest)
            i += 1
        else:
            if i + 1 >= len(lines) or lines[i + 1][0] <= depth:
                out[key.strip()] = None
                i += 1
            else:
                out[key.strip()], i = _block(lines, i + 1, lines[i + 1][0])
    return out, i


def _safe_load_all(text):
    """Documents split on ``---``, each read as a block mapping."""
    for doc in text.split("\n---\n"):
        if not doc.strip():
            continue
        lines = [(len(ln) - len(ln.lstrip(" ")), ln.strip())
                 for ln in doc.splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
        if not lines:
            continue
        parsed, consumed = _block(lines, 0, lines[0][0])
        if consumed != len(lines):
            raise YAMLError(f"could not parse from: {lines[consumed][1][:40]}")
        yield parsed


# ---- Pillow -----------------------------------------------------------
_EXIF: dict = {}


def set_exif(tags: dict) -> None:
    """What the next Image.open() will report. Keys are EXIF tag numbers."""
    global _EXIF
    _EXIF = dict(tags)


class _Image:
    def __init__(self, path, size=(16, 12), mode="RGB") -> None:
        self.size = size
        self.mode = mode

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def getexif(self):
        return dict(_EXIF)


_EXIF_TAGS = {271: "Make", 272: "Model", 274: "Orientation", 305: "Software",
              306: "DateTime", 270: "ImageDescription", 37510: "UserComment",
              315: "Artist", 33432: "Copyright"}


@contextmanager
def installed(pypdf=False, yaml=False, pillow=False):
    """Put the requested stubs on sys.modules for the duration.

    pypdf and Pillow are imported inside the functions that use them, so
    putting them on sys.modules is enough. PyYAML is imported at module level
    in sift.extract.structured, so that module is reloaded — which is itself
    worth knowing: a stub installed without the reload silently changes
    nothing, and the test would have passed the ImportError path while
    claiming to exercise the other one.
    """
    added = []
    reload_after = []
    saved = {name: sys.modules.get(name, _ABSENT)
             for name in ("pypdf", "yaml", "PIL", "PIL.Image", "PIL.ExifTags")}
    if pypdf:
        mod = types.ModuleType("pypdf")
        mod.PdfReader = _PdfReader
        # pypdf's configuration API, which sift bounds what it decodes with (6.18.0 and later), never reached.
        mod.apply_configuration = mod.get_configuration = _not_for_json
        sys.modules["pypdf"] = mod
        added.append("pypdf")
    if yaml:
        mod = types.ModuleType("yaml")
        mod.safe_load_all = _safe_load_all
        mod.safe_load = lambda text: next(iter(_safe_load_all(text)), None)
        mod.YAMLError = YAMLError
        sys.modules["yaml"] = mod
        added.append("yaml")
        reload_after.append("sift.extract.structured")
    if pillow:
        package = types.ModuleType("PIL")
        image = types.ModuleType("PIL.Image")
        image.open = _Image
        tags = types.ModuleType("PIL.ExifTags")
        tags.TAGS = dict(_EXIF_TAGS)
        package.Image = image
        package.ExifTags = tags
        sys.modules["PIL"] = package
        sys.modules["PIL.Image"] = image
        sys.modules["PIL.ExifTags"] = tags
        added += ["PIL", "PIL.Image", "PIL.ExifTags"]
    import importlib
    for name in reload_after:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
    try:
        yield
    finally:
        # What was there before, not nothing: inside missing() that is the
        # None that keeps the package away, and on a machine that has the
        # package it is the real module other code already holds.
        for name in added:
            _restore(name, saved[name])
        for name in reload_after:
            if name in sys.modules:
                importlib.reload(sys.modules[name])
        set_exif({})


_ABSENT = object()

#: The packages missing() can take away, and the sift modules that import
#: them when they load — which have to be read again, both when the package
#: goes and when it comes back, or they keep the one they found first.
_IMPORTED_AT_LOAD = {"pypdf": (), "yaml": ("sift.extract.structured",),
                     "PIL": ()}


@contextmanager
def missing(*packages):
    """Make ``import package`` fail for the duration, as on a machine that
    does not have it — whether or not this one does.

    None on sys.modules is how Python says so: an import of that name, or
    of anything under it, raises ModuleNotFoundError. The entries that were
    there come back afterwards.
    """
    unknown = [name for name in packages if name not in _IMPORTED_AT_LOAD]
    if unknown:
        raise ValueError(f"missing() knows {sorted(_IMPORTED_AT_LOAD)}, "
                         f"not {unknown}")
    hidden = [name for package in packages
              for name in [package] + sorted(
                  n for n in sys.modules if n.startswith(package + "."))]
    saved = {name: sys.modules.get(name, _ABSENT) for name in hidden}
    for name in hidden:
        sys.modules[name] = None
    _reload(packages)
    try:
        yield
    finally:
        for name, before in saved.items():
            _restore(name, before)
        _reload(packages)


def _restore(name: str, before) -> None:
    if before is _ABSENT:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = before


def _reload(packages) -> None:
    import importlib
    for package in packages:
        for module in _IMPORTED_AT_LOAD[package]:
            if module in sys.modules:
                importlib.reload(sys.modules[module])


__all__ = ["installed", "missing", "pdf_bytes", "set_exif", "io"]
