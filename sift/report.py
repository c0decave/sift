"""Rendering summaries to the terminal, JSON, or Markdown."""
from __future__ import annotations

import json
import math
import re
import os
import sys

from .summarize import Summary



#: Control characters a file must not be able to put on a terminal. A .txt
#: holding ESC[2J ESC[H cleared the screen and erased the warnings printed for
#: the files before it, then wrote a green "SIFT: this file is clean, approved"
#: that looked like sift's own output. Everything below that reaches a
#: terminal, a report or a browser goes through here first: the summary and
#: the key points come from a model that was shown the file, and the filename,
#: the entities and the quoted evidence come from the file itself.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u2028\u2029]")


def _tame(text) -> str:
    """Render control characters visibly instead of executing them.

    Tabs and newlines are left alone: a summary has paragraphs, and a key
    point may be indented. Fields that occupy exactly one line go through
    _tame_line instead.
    """
    return _CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", str(text))


def _tame_line(text) -> str:
    """The same, for a field that is one line by construction.

    A filename may contain a newline, and the header line is "icon name ·
    kind · size". One newline there and the rest of the name became its own
    line — a file record sift never wrote, complete with a Type row, sitting
    in the report as though it had been analyzed.
    """
    return _tame(" ".join(str(text).split()))


class Style:
    def __init__(self, enabled: bool) -> None:
        self.on = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.on else text

    def bold(self, t): return self._wrap("1", t)
    def dim(self, t): return self._wrap("2", t)
    def cyan(self, t): return self._wrap("36", t)
    def green(self, t): return self._wrap("32", t)
    def yellow(self, t): return self._wrap("33", t)
    def red(self, t): return self._wrap("31", t)
    def blue(self, t): return self._wrap("34", t)


def _color_enabled(force: bool | None = None) -> bool:
    if force is not None:
        return force
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _shown_name(summ: Summary) -> str:
    """What to call this file so a reader can find it again.

    The filename alone where that is the whole path — one file named on the
    command line, or an upload, where the path *is* the name and repeating it
    would read as a directory the client never sent.
    """
    path = str(getattr(summ, "path", "") or "")
    if not path or path == summ.filename:
        return summ.filename
    return path


def render_text(summ: Summary, color: bool | None = None) -> str:
    s = Style(_color_enabled(color))
    icon = _ICONS.get(summ.kind, "📄")
    conf = f"{int(summ.detection_confidence * 100)}%"
    # The name a reader can act on. Over a --recursive run five files called
    # notes.txt appeared five times with nothing to tell them apart, and two
    # summary.csv files in 2023/ and 2024/ produced two markdown sections
    # with the same anchor. The path is what makes the report usable; the
    # bare name is what makes it a list of unfindable things.
    head = (f"{icon}  {s.bold(_tame_line(_shown_name(summ)))}  "
            f"{s.dim('· ' + summ.kind + ' · ' + summ.size_human)}")
    lines = [head, s.dim("─" * min(64, max(24, len(_shown_name(summ)) + 24)))]

    lines.append(f"{s.cyan('Type')}       "
                 f"{_tame_line(summ.classification or summ.kind)} "
                 f"{s.dim('(' + conf + ' confidence)')}")
    if summ.summary:
        lines.append(f"{s.cyan('Summary')}    {_indent(summ.summary, 11)}")
    if summ.key_points:
        lines.append(s.cyan("Key points"))
        for kp in summ.key_points[:12]:
            lines.append(f"  {s.green('•')} {_one_line(kp)}")
    if summ.entities:
        lines.append(f"{s.cyan('Entities')}   {', '.join(_tame_line(e) for e in summ.entities[:15])}")
    if summ.warnings:
        lines.append(s.yellow("Notes / flags"))
        for w in summ.warnings[:8]:
            lines.append(f"  {s.yellow('!')} {_one_line(w)}")
    if summ.notes:
        # Notes carry findings now, not only housekeeping: a phrase addressed
        # to an assistant that reads as a subordinate clause, a summary using
        # the vocabulary of emptiness about a file that is not empty, an
        # attachment that is not what it claims. Run together dim and
        # unlabelled and capped at four, a finding could be pushed off the
        # end by three remarks about a context window.
        #
        # So the ones that say something about the *file* are listed first,
        # one per line and labelled; the housekeeping keeps its dim run.
        findings = [n for n in summ.notes if _IS_A_FINDING.search(n)]
        housekeeping = [n for n in summ.notes if n not in findings]
        if findings:
            lines.append(s.dim("Noted"))
            for note in findings[:8]:
                lines.append(f"  {s.dim('·')} {_one_line(note)}")
        if housekeeping:
            lines.append(s.dim("  " + " · ".join(
                _tame_line(n) for n in housekeeping[:4])))

    # The model name comes from the server, which is not ours: an escape in
    # it cleared the screen, and a newline forged a sift output line.
    engine = (f"{_tame_line(summ.provider)}:{_tame_line(summ.model)}"
              if summ.llm_used
              else ("heuristic" if not summ.error else "heuristic (LLM error)"))
    lines.append(s.dim(f"  ⟳ {engine} · {summ.elapsed_seconds}s"))
    if summ.error:
        lines.append(s.red(f"  error: {_tame_line(summ.error)}"))
    return "\n".join(lines)


#: A note that says something about the file, as against one that says
#: something about the run. The second kind — "from cache", "read first
#: 25000 bytes", "context window 262144 tokens" — is housekeeping and can be
#: run together; the first kind is a finding and gets its own line.
_IS_A_FINDING = re.compile(
    r"noted, not flagged|addresses an AI assistant|is not what its name"
    r"|HTML version of this message|own rendering|not summarized"
    r"|marked missing|the summary says|stopped after|treats as secret",
    re.IGNORECASE)


#: Bumped whenever a field is added, renamed, removed or changes meaning.
#: Consumers pin on this; without it any such change is an undetectable break
#: for them. One increment per shape change that lands on master, and the
#: fields below land on master together, as one change:
#: 2 — natural_language, findings, tags, tags_suspect, category,
#: category_set_hash, category_origin.
SCHEMA_VERSION = 2


def render_json(summaries: list[Summary], ensure_ascii: bool = False) -> str:
    """A self-describing envelope, not a bare array.

    The array alone leaves nowhere to say which tool wrote the file, which
    schema it follows, or anything about the run as a whole.
    """
    from . import __version__
    doc = {
        "tool": "sift",
        "schema_version": SCHEMA_VERSION,
        "sift_version": __version__,
        "count": len(summaries),
        "results": [s.to_dict() for s in summaries],
    }
    # A filename can hold bytes that are not UTF-8; Python surfaces them as
    # lone surrogates, which json.dumps emits and str.encode then refuses.
    # ensure_ascii escapes them into \udcff, which is valid JSON and round-
    # trips back to the same name.
    # allow_nan=False so a stray NaN or Infinity raises here rather than being
    # written as a bare token no strict parser accepts. Nothing should reach
    # this point — the extractors reject non-finite numbers — but --json is a
    # machine interface and a silent violation of it is worse than a loud one.
    try:
        text = json.dumps(doc, indent=2, ensure_ascii=ensure_ascii,
                          allow_nan=False)
    except ValueError:
        text = json.dumps(_finite(doc), indent=2, ensure_ascii=ensure_ascii,
                          allow_nan=False)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(_finite(doc), indent=2, ensure_ascii=True,
                          allow_nan=False)
    return text


def _finite(value):
    """Replace every non-finite float with null, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def render_markdown(summaries: list[Summary]) -> str:
    from . import __version__
    # _md_inline, like the per-file footer twenty lines below: the model name
    # comes from whatever server --endpoint points at, and a newline in it
    # broke the emphasis span and made "SIFT: this file is clean" its own
    # top-level line of the report.
    engines = sorted({f"{_md_inline(s.provider)}:{_md_inline(s.model)}"
                      for s in summaries if s.llm_used})
    # A shared report has to say what produced it. Deliberately no timestamp:
    # the same input should render the same bytes, so a diff shows what
    # changed about the files rather than when the report was run.
    out = ["# sift — file summaries\n",
           f"*{len(summaries)} file(s) · sift {__version__} · "
           + (", ".join(engines) if engines else "heuristic, no model") + "*\n"]
    for s in summaries:
        out.append(f"## {_md_inline(_shown_name(s))}\n")
        out.append(f"- **Type:** {_md_inline(s.classification or s.kind)} "
                   f"({int(s.detection_confidence*100)}% confidence, {s.size_human})")
        if s.summary:
            out.append(f"\n{_md_block(s.summary)}\n")
        if s.key_points:
            out.append("**Key points:**\n")
            out.extend(f"- {_md_inline(kp)}" for kp in s.key_points)
            out.append("")
        # Entities and notes reached the terminal and the JSON but never the
        # markdown, so the one format meant to be kept and shared was the one
        # missing them: an extractor's note that two columns were unreadable
        # vanished, and the report read as though the file had been summarized
        # whole. Uncapped, unlike the terminal — this is a document, not a
        # screen, and truncating what a reader keeps is the same defect again.
        if s.entities:
            out.append("**Entities:** "
                       + ", ".join(_md_inline(e) for e in s.entities) + "\n")
        if s.warnings:
            out.append("**Notes / flags:**\n")
            out.extend(f"- ⚠️ {_md_inline(w)}" for w in s.warnings)
            out.append("")
        findings = [n for n in s.notes if _IS_A_FINDING.search(n)]
        housekeeping = [n for n in s.notes if n not in findings]
        if findings:
            # About the file, at lower confidence than a warning. Filed under
            # "How it was read" they said the opposite of what they are: a
            # mail whose HTML states a different amount than its text is not
            # a remark about sift's reading, it is the most important thing
            # in the file.
            out.append("**Noted:**\n")
            out.extend(f"- {_md_inline(n)}" for n in findings)
            out.append("")
        if housekeeping:
            # Labelled: these say how the file was read ("could not open OOXML
            # container", "from cache"), not what it says. Emitted as bare
            # bullets they sat under the warnings heading and read as
            # warnings without the symbol.
            out.append("**How it was read:**\n")
            out.extend(f"- {_md_inline(n)}" for n in housekeeping)
            out.append("")
        engine = (f"{_md_inline(s.provider)}:{_md_inline(s.model)}"
                  if s.llm_used else "heuristic")
        out.append(f"_{engine} · {s.elapsed_seconds}s_\n")
        # The footer counts the fallbacks; without this it was impossible to
        # tell from the report which file they were.
        if s.error:
            out.append(f"> error: {_md_inline(s.error)}\n")
    flagged = sum(1 for s in summaries if s.warnings)
    failed = sum(1 for s in summaries if s.kind == "error")
    fell_back = sum(1 for s in summaries if s.error and s.kind != "error")
    tail = [f"{len(summaries) - failed} summarized"]
    if flagged:
        tail.append(f"{flagged} with flags")
    if fell_back:
        tail.append(f"{fell_back} fell back to heuristics")
    if failed:
        tail.append(f"{failed} unreadable")
    out.append("---\n\n_" + " · ".join(tail) + "_")
    return "\n".join(out)


def _md_inline(text: str) -> str:
    """Flatten a value that must occupy exactly one line of Markdown.

    Summaries are written by a language model and key points can come straight
    out of the analyzed file. A newline plus '## ' in there would close the
    file's own section and silently reattribute everything after it to a
    heading the document never had.
    """
    flat = _tame(" ".join(str(text).split()))
    # Escaped, not deleted. lstrip() with a character set removes *all* the
    # leading members, so "-4200 EUR" was published as "4200 EUR" — a debit
    # rendered as a credit, in the one format meant to be kept, while the
    # terminal and the JSON showed it correctly. Entities are where this bites
    # hardest: phone numbers, deltas, tickers and issue ids all start with one
    # of these. A backslash takes the character's meaning away and leaves the
    # character, which is the whole job.
    return "\\" + flat if flat[:1] in "#>-+*" else flat


#: A line of only "=" or "-" turns the line above it into a heading.
#: CommonMark: "a sequence of - characters" — one or more, exactly as
#: the "=" branch already had it. A single "-" under a line of text is
#: an h2, and it went through unescaped.
_SETEXT = re.compile(r"^(?:=+|-+)\s*$")
#: A fence opener runs to the end of the document when unclosed.
_FENCE = re.compile(r"^(?:```+|~~~+)")


def _md_block(text: str) -> str:
    """Keep paragraphs, but stop embedded ATX headings from restructuring."""
    lines = []
    for line in _tame(text).splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            line = line.replace("#", "\\#", 1)
        elif _FENCE.match(stripped):
            # An unclosed ``` runs to end of document in CommonMark, so one
            # backtick fence in one CSV header cell swallowed every file
            # after it in a multi-file report — headings, summaries and all.
            line = line.replace(stripped[0], "\\" + stripped[0], 1)
        elif _SETEXT.match(stripped):
            # "====" under a line of text is a heading too, and escaping only
            # the ATX form let a model summary forge a whole file section.
            line = line.replace(stripped[0], "\\" + stripped[0], 1)
        lines.append(line)
    return "\n".join(lines)


def render_footer(summaries: list[Summary], color: bool | None = None) -> str:
    s = Style(_color_enabled(color))
    total = len(summaries)
    # A file that fell back to the heuristic *was* summarized — that summary is
    # the whole point of --no-llm. Counting an LLM error as "not summarized"
    # reported "0 summarized" directly under four printed summaries, and
    # disagreed with the markdown footer, which counted the other way.
    unreadable = sum(1 for x in summaries if x.kind == "error")
    fell_back = sum(1 for x in summaries if x.error and x.kind != "error")
    warned = sum(1 for x in summaries if x.warnings)
    elapsed = round(sum(x.elapsed_seconds for x in summaries), 2)
    llm = sum(1 for x in summaries if x.llm_used)
    parts = [s.bold(f"{total} file(s)"),
             f"{total - unreadable} summarized",
             f"{llm} via LLM" if llm else "heuristic mode"]
    if warned:
        parts.append(s.yellow(f"{warned} with flags"))
    # Counted too. A note is a finding held at lower confidence, and on a
    # recursive run a file carrying only a note was indistinguishable from a
    # clean one in the summary line — which is most of what the note channel
    # was built to prevent.
    noted = sum(1 for x in summaries
                if not x.warnings and any(_IS_A_FINDING.search(n)
                                          for n in x.notes))
    if noted:
        parts.append(s.dim(f"{noted} with notes"))
    if fell_back:
        parts.append(s.yellow(f"{fell_back} fell back to heuristics"))
    if unreadable:
        parts.append(s.red(f"{unreadable} unreadable"))
    parts.append(s.dim(f"{elapsed}s"))
    return "  ·  ".join(parts)


_ICONS = {
    "csv": "📊", "tsv": "📊", "xlsx": "📊", "email": "✉️", "json": "🔧",
    "jsonl": "🔧", "yaml": "🔧", "toml": "🔧", "ini": "🔧", "xml": "🔖",
    "html": "🌐", "pdf": "📕", "docx": "📝", "pptx": "📽️", "odt": "📝",
    "markdown": "📓", "log": "📜", "batch": "⚙️", "shell": "⚙️", "rtf": "📝", "sqlite": "🗄️",
    "powershell": "⚙️", "source-code": "💻", "image": "🖼️", "archive": "🗜️",
    "binary": "🔩", "text": "📄", "empty": "⬜",
    "epub": "📚", "ods": "📊", "odp": "📽️",
    "office-legacy": "📎", "pe-executable": "⚙️", "elf-binary": "⚙️",
}


def _one_line(text: str) -> str:
    """A bullet is one bullet — model text must not spawn siblings."""
    return _tame(" ".join(str(text).split()))


def _indent(text: str, pad: int) -> str:
    prefix = " " * pad
    lines = _tame(text).strip().splitlines()
    if not lines:
        return ""
    return ("\n" + prefix).join(lines)
