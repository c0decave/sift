"""Extractor for JSON / JSONL / YAML / TOML / INI / XML."""
from __future__ import annotations

import configparser
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from ..detect import Detection
from .base import (looks_like_a_credential, Extracted, clip, decode_text,
                   properties_keys)

MAX_CHARS = 120_000
#: Most names a listing in the metadata holds - sections, keys, tables, an
#: outline's entries. The count beside it is the whole: a config of 300,000
#: sections listed them all, a 30 MB summary past the budget and any reply.
MAX_LISTED = 40

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    try:  # the backport reads the same dialect, as config.py already assumes
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None

try:
    import yaml as _yaml
except ModuleNotFoundError:
    _yaml = None


def extract(path: Path, det: Detection, max_bytes: int) -> Extracted:
    with path.open("rb") as fh:
        data = fh.read(max_bytes)
    byte_capped = det.size > max_bytes > 0
    text, encoding = decode_text(data, byte_capped)
    dispatch = {
        "json": _json, "jsonl": _jsonl, "yaml": _yaml_kind,
        "toml": _toml, "ini": _ini, "xml": _xml,
    }
    handler = dispatch.get(det.kind, _passthrough)
    if det.kind == "ini" and path.name.lower().endswith(".properties"):
        handler = _properties
    meta, rendered = handler(text)
    meta["encoding"] = encoding
    if byte_capped and "parse_error" in meta:
        # We cut the document in half and then reported it as malformed. A
        # perfectly valid 868-byte JSON read at 200 bytes was accused of
        # "Expecting property name ... (char 200)" — our own cut, blamed on
        # the file, and a user would go looking for a defect that is not there.
        meta["parse_error_cause"] = (
            f"the file was read only up to max_bytes ({max_bytes} of "
            f"{det.size} bytes), so it is incomplete here — this is not "
            f"necessarily a defect in the document")
    # A note about *sift* — a missing optional dependency, a parser sift does
    # not have — is not a fact about the file, and metadata goes into the
    # prompt under an instruction to use only what is in the content and the
    # metadata. It belongs where the truncation notes go.
    own_note = meta.pop("_note", None)
    body, truncated = clip(rendered, MAX_CHARS)
    ex = Extracted(det.kind, det.mime, text=body, metadata=meta, rendered=True,
                   truncated=truncated or byte_capped,
                   # The rendered text is the document; its structure and
                   # its values are visible there and do not need repeating
                   # as metadata. "settings" exists for the heuristic
                   # summary, which has no document in front of it — putting
                   # it in the prompt as well stated every value twice.
                   text_covers=("structure", "settings"))
    if own_note:
        ex.note(own_note)
    if byte_capped:
        ex.note(f"file larger than max_bytes ({det.size} bytes); read first "
                f"{max_bytes}")
    return ex


#: How many leaf values a summary can usefully carry.
_MAX_SCALARS = 24


def _scalars(obj, prefix: str = "", out: "dict" = None, depth: int = 0) -> dict:
    """Leaf values by dotted path, so a summary can say what a file sets.

    The outline says `retention_days: int`, which is the shape and not the
    setting. For a config file the value *is* the content: "retention_days:
    30" is what someone opened it to find out.
    """
    out = {} if out is None else out
    if len(out) >= _MAX_SCALARS or depth > 4:
        return out
    if isinstance(obj, dict):
        for key, value in obj.items():
            _scalars(value, f"{prefix}.{key}" if prefix else str(key),
                     out, depth + 1)
            if len(out) >= _MAX_SCALARS:
                break
        return out
    if isinstance(obj, list):
        # A list of scalars is one fact; a list of objects is a table and
        # belongs in the outline, not here.
        flat = [v for v in obj if isinstance(v, (str, int, float, bool))]
        if flat and len(flat) == len(obj) and prefix:
            shown = ", ".join(str(v)[:24] for v in flat[:5])
            out[prefix] = shown + (f", … ({len(flat)} items)"
                                   if len(flat) > 5 else "")
        return out
    if prefix and isinstance(obj, (str, int, float, bool)) or obj is None:
        if not prefix:
            return out
        if looks_like_a_credential(str(obj), prefix):
            out[prefix] = f"<{type(obj).__name__}, {len(str(obj))} chars — " \
                          f"not shown>"
        else:
            text = str(obj)
            out[prefix] = text[:60] + ("…" if len(text) > 60 else "")
    return out


def _first_sentence(exc: Exception) -> str:
    """A parser's complaint, without the line it choked on.

    configparser and the YAML loaders quote the offending source, and for a
    .env or a .properties — both section-less by definition, so both always
    fail — that line is a credential. The summary trimmed a *copy*; the
    metadata kept the original and carried it into --json, the on-disk cache,
    the model prompt and the browser's metadata pane.
    """
    return str(exc).splitlines()[0].strip()[:200] if str(exc) else ""


def _json(text: str) -> tuple[dict, str]:
    try:
        duplicates: "list[str]" = []

        def _keep_last(pairs):
            """json's own behaviour, but noticed.

            {"total": 100, "total": 200} parses to {"total": 200} and the 100
            is gone — with no error, because duplicate keys are legal JSON
            and the last one wins. A summary that reports 200 as "the total"
            is stating something the file does not unambiguously say.
            """
            seen = {}
            for key, value in pairs:
                if key in seen and len(duplicates) < 20:
                    duplicates.append(key)
                seen[key] = value
            return seen

        obj = json.loads(text, object_pairs_hook=_keep_last)
    except RecursionError:
        # Not a ValueError, so it escaped this handler and the registry's,
        # and ended the run. json recurses once per nesting level; a file of
        # a thousand open brackets is two kilobytes.
        return ({"parse_error": "nesting deeper than the parser allows"},
                text[:MAX_CHARS])
    except ValueError as exc:
        return {"parse_error": _first_sentence(exc)}, text[:MAX_CHARS]
    outline = _outline(obj)
    pretty = json.dumps(obj, indent=2, ensure_ascii=False)
    meta = {"structure": outline, "top_level_type": type(obj).__name__,
            "settings": _scalars(obj)}
    if duplicates:
        # Beside the structure, not inside it: put in the outline it read as
        # a key of the file, which is the opposite of what it says.
        meta["duplicate_keys"] = (
            f"{len(duplicates)} key(s) appear more than once and only the "
            f"last value of each was kept: "
            + ", ".join(sorted(set(duplicates))[:8]))
    if len(pretty) <= MAX_CHARS:
        # The document itself shows its shape. Prefixing an outline said the
        # same thing a third time — after the metadata and before the content —
        # in the window meant for the content.
        return meta, "# JSON\n" + pretty
    return meta, ("# JSON structure\n" + _render_outline(outline)
                  + "\n\n# Content (truncated)\n" + pretty[:MAX_CHARS])


def _jsonl(text: str) -> tuple[dict, str]:
    records, errors = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except (ValueError, RecursionError):
            errors += 1
    keys: set[str] = set()
    for rec in records[:500]:
        if isinstance(rec, dict):
            keys.update(rec.keys())
    sample = "\n".join(json.dumps(r, ensure_ascii=False) for r in records[:10])
    return ({"records": len(records), "parse_errors": errors, "keys": sorted(keys)},
            f"# JSON Lines — {len(records)} records\nkeys: {sorted(keys)}\n\n"
            f"## First records\n{sample}")


#: A top-level mapping key: no indentation, a name, a colon.
_YAML_TOP = re.compile(r"^([A-Za-z_][\w.\-]*)\s*:", re.MULTILINE)


def _yaml_without_pyyaml(text: str) -> dict:
    """What can be said about a YAML file by reading it as lines.

    Without PyYAML the summary was "YAML with 1 document." and nothing else —
    for a docker-compose with three services, their images, their ports and a
    healthcheck. The top-level keys are the shape of the file and they are
    one regex away; saying nothing because the optional dependency is absent
    is a choice, not a limitation.
    """
    keys, seen = [], set()
    for match in _YAML_TOP.finditer(text):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            keys.append(name)
    return {"top_level_keys": keys[:30], "documents": text.count("\n---") + 1}


def _yaml_kind(text: str) -> tuple[dict, str]:
    if _yaml is None:
        # The note goes in ex.notes, not in the metadata: metadata reaches
        # the model, and the system prompt tells it to use only what is in
        # the content and the metadata — so "PyYAML not installed" was a
        # licence to report sift's missing dependency as a property of the
        # user's file.
        meta = _yaml_without_pyyaml(text)
        meta["_note"] = "PyYAML is not installed — read as lines, not parsed"
        # The heading is in the *content*, which is the other half of what
        # the model is told to use. "install PyYAML" there is the same
        # licence to report sift's dependency as a property of the file, one
        # layer over.
        return meta, "# YAML\n" + text[:MAX_CHARS]
    try:
        docs = list(_yaml.safe_load_all(text))
    except Exception as exc:  # noqa: BLE001 - yaml raises many types
        return {"parse_error": _first_sentence(exc)}, text[:MAX_CHARS]
    obj = docs[0] if len(docs) == 1 else docs
    outline = _outline(obj)
    meta = {"documents": len(docs), "structure": outline}
    if len(text) <= MAX_CHARS:
        return meta, "# YAML\n" + text          # the document shows its own shape
    return meta, ("# YAML structure\n" + _render_outline(outline)
                  + "\n\n# Content (truncated)\n" + text[:MAX_CHARS])


def _toml(text: str) -> tuple[dict, str]:
    if tomllib is None:
        return {"note": "tomllib unavailable"}, text[:MAX_CHARS]
    try:
        obj = tomllib.loads(text)
    except ValueError as exc:
        return {"parse_error": _first_sentence(exc)}, text[:MAX_CHARS]
    tables = list(obj.keys())
    return ({"structure": _outline(obj), "tables": tables[:MAX_LISTED], "table_count": len(tables)},
            "# TOML\ntables: " + _listing(tables) + "\n\n" + text[:MAX_CHARS])


def _listing(names: list) -> str:
    """The first MAX_LISTED names, and "…" when there are more."""
    return ", ".join(str(n) for n in names[:MAX_LISTED]) + (", …" if len(names) > MAX_LISTED else "")


#: How many of a section-less file's lines have to be "key = value" before
#: it is read as one. A .pgpass line is "host:port:db:user:password", which
#: configparser will happily read as one key with a four-field value — an
#: answer that looks like a parse and is not one. An .env is 90% assignments
#: or it is not an .env.
_ENOUGH_ASSIGNMENTS = 0.6


def _looks_like_bare_assignments(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith(("#", ";"))]
    if not lines:
        return False
    equals = sum(1 for ln in lines if "=" in ln.split(":", 1)[0] or "=" in ln)
    return equals / len(lines) >= _ENOUGH_ASSIGNMENTS


def _ini(text: str) -> tuple[dict, str]:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        parser.read_string(text)
    except configparser.MissingSectionHeaderError:
        # .env, .properties and half the config files in the world have no
        # [sections] — that is not a defect, it is the format. Reporting
        # "could not be parsed: File contains no section headers" of an .env
        # said nothing about a file whose keys sift could have listed.
        #
        # Only when the lines really are assignments. Given a .pgpass,
        # configparser reads "db.intern:5432:buchhaltung:svc:pw" as one key
        # with a four-field value and reports success, which is worse than
        # the honest failure it replaces.
        if not _looks_like_bare_assignments(text):
            import sys
            return ({"parse_error": _first_sentence(sys.exc_info()[1])},
                    text[:MAX_CHARS])
        parser = configparser.ConfigParser(strict=False, interpolation=None)
        # Keys as the file writes them. configparser lowercases by default,
        # so "DB_HOST=…" came back as "db_host" — a key name is content, and
        # an environment variable's case is part of its name.
        parser.optionxform = str
        try:
            parser.read_string("[settings]\n" + text)
        except configparser.Error as exc:
            return {"parse_error": _first_sentence(exc)}, text[:MAX_CHARS]
        keys = list(parser["settings"].keys())
        return ({"sections": {}, "section_count": 0, "keys": keys[:MAX_LISTED],
                 "key_count": len(keys), "no_sections": True},
                "# Config (key=value, no sections)\nkeys: "
                + _listing(keys) + "\n\n" + text[:MAX_CHARS])
    except configparser.Error as exc:
        return {"parse_error": _first_sentence(exc)}, text[:MAX_CHARS]
    names = parser.sections()
    sections = {s: list(parser[s].keys())[:MAX_LISTED] for s in names[:MAX_LISTED]}
    return ({"sections": sections, "section_count": len(names)},
            "# INI/Config\nsections: " + _listing(names) + "\n\n" + text[:MAX_CHARS])


def _properties(text: str) -> tuple[dict, str]:
    """A .properties as Java reads it (properties_keys), not as an INI with
    no sections: to configparser a key ends at "=" or ":" only, so the line
    "db.url Kx9vQ2mZ7pL4=x" was the key "db.url Kx9vQ2mZ7pL4" -- a value in
    the summary --, "log\\=level=info" the key "log\\", and an escape in a
    key stood as it was written. A .properties has no parse error: Java
    reads every line as a key, a value or a comment."""
    keys = properties_keys(text)
    return ({"sections": {}, "section_count": 0, "keys": keys[:MAX_LISTED],
             "key_count": len(keys), "no_sections": True},
            "# Config (key=value, no sections)\nkeys: "
            + _listing(keys) + "\n\n" + text[:MAX_CHARS])


def _xml(text: str) -> tuple[dict, str]:
    # ElementTree rejects a *str* that still carries an encoding declaration;
    # strip a leading "<?xml ... ?>" so declared-encoding files parse fine.
    cleaned = re.sub(r"^\s*<\?xml[^>]*\?>", "", text, count=1)
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError as exc:
        return {"parse_error": _first_sentence(exc)}, text[:MAX_CHARS]
    children: dict[str, int] = {}
    for child in root:
        tag = _localname(child.tag)
        children[tag] = children.get(tag, 0) + 1
    total = sum(1 for _ in root.iter())
    return ({"root": _localname(root.tag), "children": children, "total_elements": total},
            f"# XML\nroot: <{_localname(root.tag)}>, elements: {total}\n"
            f"direct children: {_child_phrase(children)}\n\n" + text[:MAX_CHARS])


def _child_phrase(children: dict) -> str:
    """"{'a': 2, 'b': 1}" -> "a ×2, b". Read by a person, and by a model."""
    if not children:
        return "none"
    return ", ".join(f"{tag} ×{n}" if n > 1 else tag
                     for tag, n in children.items())


def _passthrough(text: str) -> tuple[dict, str]:
    return {}, text[:MAX_CHARS]


# ---- helpers ----------------------------------------------------------
def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _outline(obj, depth: int = 0, max_depth: int = 3):
    if depth >= max_depth:
        return type(obj).__name__
    if isinstance(obj, dict):
        return {str(k): _outline(v, depth + 1, max_depth) for k, v in list(obj.items())[:MAX_LISTED]}
    if isinstance(obj, list):
        head = _outline(obj[0], depth + 1, max_depth) if obj else "empty"
        return [f"list[{len(obj)}]", head]
    return type(obj).__name__


def _render_outline(outline, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(outline, dict):
        return "\n".join(f"{pad}{k}: {_render_outline(v, indent + 1).lstrip()}"
                         if isinstance(v, (dict, list)) else f"{pad}{k}: {v}"
                         for k, v in outline.items())
    if isinstance(outline, list):
        return f"{pad}{outline[0]} of " + (
            _render_outline(outline[1], indent).lstrip() if len(outline) > 1 else "?")
    return f"{pad}{outline}"
