"""Command-line interface for sift."""
from __future__ import annotations

import argparse
import fnmatch
import os
import stat
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Optional

from . import __version__
from .cache import SummaryCache
from . import config as config_module
from .config import Config, ConfigError
from .extract import SUPPORTED_KINDS
from .llm import build_client, LLMError
from . import report
from .summarize import ModelHealth, summarize_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sift",
        description="Recognize & summarize files with a local LLM "
                    "(Ollama / LM Studio / MLX-LM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    p.add_argument("paths", nargs="*", help="files or directories to analyze")
    p.add_argument("--version", action="version", version=f"sift {__version__}")

    llm = p.add_argument_group("LLM")
    llm.add_argument("--provider",
                     choices=["auto", "ollama", "lmstudio", "mlx-lm",
                              "openai", "none"],
                     help="LLM backend (default: auto-detect Ollama, LM "
                          "Studio, then MLX-LM)")
    llm.add_argument("--model", help="model name (default: a model already "
                                     "loaded on the server)")
    llm.add_argument("--endpoint", help="override backend URL")
    llm.add_argument("--api-key", help="API key/token if the server requires one")
    llm.add_argument("--allow-model-load", action="store_true",
                     help="permit loading a cold model (this evicts the "
                          "model currently held in memory; off by default)")
    llm.add_argument("--no-stream", action="store_true",
                     help="wait for the whole answer instead of streaming it "
                          "(a proxy may close the silent connection)")
    llm.add_argument("--no-llm", action="store_true",
                     help="skip the LLM; extraction + heuristic summary only")
    llm.add_argument("--no-vision", action="store_true",
                     help="do not send images to vision models")
    llm.add_argument("--models", action="store_true",
                     help="list the server's models with load state and exit")

    gen = p.add_argument_group("generation")
    gen.add_argument("--context", type=int, dest="context_tokens",
                     help="model context window in tokens (default: ask the server, else 8192)")
    gen.add_argument("--max-output", type=int, dest="max_output_tokens",
                     help="max tokens for the summary (default 2000)")
    gen.add_argument("--temperature", type=float, help="sampling temperature (default 0.2)")
    gen.add_argument("--timeout", type=float, help="per-request timeout seconds")

    trav = p.add_argument_group("traversal & output")
    trav.add_argument("-r", "--recursive", action="store_true",
                      help="descend into directories")
    trav.add_argument("--include", action="append", metavar="GLOB",
                      help="only files matching this glob (repeatable)")
    trav.add_argument("--exclude", action="append", metavar="GLOB",
                      help="skip files matching this glob (repeatable)")
    trav.add_argument("--max-bytes", type=int, help="max bytes read per file")
    trav.add_argument("--model-time-budget", type=float, metavar="SECONDS",
                      help="seconds of model time one file may use across all "
                           "its parts (default 900; 0 removes the limit)")
    trav.add_argument("--time-budget", type=float, metavar="SECONDS",
                      help="seconds one file may take before it is given up "
                           "on (default 60; 0 removes the limit)")
    out = trav.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="output JSON")
    out.add_argument("--markdown", action="store_true", help="output Markdown")
    trav.add_argument("-o", "--output", help="write output to a file instead of stdout")
    trav.add_argument("--config", help="path to a TOML config file")
    trav.add_argument("--no-cache", action="store_true",
                      help="re-summarize even files whose content is unchanged")
    trav.add_argument("--clear-cache", action="store_true",
                      help="delete every cached summary and exit")
    trav.add_argument("-q", "--quiet", action="store_true", help="suppress progress")
    trav.add_argument("-v", "--verbose", action="store_true")
    trav.add_argument("--list-types", action="store_true",
                      help="list recognized file kinds and exit")
    return p


def build_serve_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sift serve", description="Run the local web UI.")
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (default 127.0.0.1, local only)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--provider",
                   choices=["auto", "ollama", "lmstudio", "mlx-lm",
                            "openai", "none"])
    p.add_argument("--model")
    p.add_argument("--endpoint")
    p.add_argument("--api-key")
    p.add_argument("--allow-model-load", action="store_true",
                   help="permit loading a cold model (evicts the resident one)")
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--timeout", type=float)
    p.add_argument("--context", type=int, dest="context_tokens")
    # The model settings the traversal parser has. Missing here, and it is
    # the web UI that needs them most: a reasoning model spends its output
    # budget thinking before it writes, so the default was too small for the
    # one interface where a user cannot pass a flag per file.
    p.add_argument("--max-output", type=int, dest="max_output_tokens")
    p.add_argument("--temperature", type=float)
    p.add_argument("--no-vision", action="store_true", dest="no_vision")
    # Same shape as the traversal parser's, so one derivation covers both:
    # store_true on "no_cache", inverted where the config is built. Two
    # spellings of one setting is how a flag comes to work in one command
    # and not the other.
    p.add_argument("--no-cache", action="store_true",
                   help="do not read or write the summary cache")
    p.add_argument("--max-bytes", type=int)
    p.add_argument("--time-budget", type=float)
    p.add_argument("--model-time-budget", type=float)
    p.add_argument("--spool-dir", metavar="DIR",
                   help="where an upload is written while it is read "
                        "(default: the system temporary directory)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--config")
    return p


EPILOG = """\
examples:
  sift report.pdf                       summarize one file
  sift ./inbox -r                       recurse a directory
  sift examples/*.eml --json -o out.json   machine-readable output
  sift data.csv --no-llm                offline: extraction + heuristic summary
  sift --models                         what the server has, and what is loaded
  sift --provider lmstudio photo.jpg    force LM Studio (vision model)
  sift serve                            open the web UI at http://127.0.0.1:8765
  sift -- serve                         analyze a file that is named "serve"

exit codes:
  0  at least one file was analyzed
  1  nothing could be analyzed, or a named path does not exist
  2  bad invocation or unreadable configuration
  3  --models: no LLM backend reachable
  130  interrupted
"""


def _config_from_args(args) -> Config:
    # Derived from the dataclass, not hand-listed. The list version had to be
    # kept in step with two parsers by hand, and it was not: --spool-dir was
    # added to the serve parser, documented in the README as the way to keep
    # uploads off a full disk, and dropped silently here. Nothing failed —
    # `sift serve --spool-dir /mnt/bulk` exited 0 and spooled to /tmp — which
    # is the worst shape a bug can take, because a flag that does nothing
    # looks exactly like a flag that worked.
    overrides = {}
    defaults = Config()
    for spec in fields(Config):
        value = getattr(args, spec.name, None)
        if value is None:
            continue
        # A store_true that was not passed is False, and that is the user
        # saying nothing — not the user asking for False, which would quietly
        # overrule the config file. Turning something off is spelled with an
        # explicit --no-* flag, handled just below.
        if value is False and isinstance(getattr(defaults, spec.name), bool):
            continue
        overrides[spec.name] = value
    for flag, setting in (("no_stream", "stream"), ("no_vision", "vision"),
                          ("no_cache", "cache")):
        if getattr(args, flag, False):
            overrides[setting] = False
    if getattr(args, "json", False):
        overrides["output_format"] = "json"
    elif getattr(args, "markdown", False):
        overrides["output_format"] = "markdown"
    cfg_path = Path(args.config) if getattr(args, "config", None) else None
    return Config.load(cfg_path, overrides)


def _kind_name(mode: int) -> str:
    """What a non-regular file is, for a message worth reading."""
    import stat as _stat
    for test, name in ((_stat.S_ISFIFO, "a named pipe"),
                       (_stat.S_ISSOCK, "a socket"),
                       (_stat.S_ISDIR, "a symlink to a directory"),
                       (_stat.S_ISBLK, "a block device"),
                       (_stat.S_ISCHR, "a character device")):
        if test(mode):
            return name
    return "not a regular file"


def collect_files(paths: list[str], config: Config,
                  problems: Optional[list[str]] = None) -> list[Path]:
    """Gather the files to analyze, reporting what could not be reached.

    ``problems`` collects anything skipped. A tool that says "9 files
    summarized" after silently walking past a directory it could not open has
    told the user something false, so nothing is skipped quietly.
    """
    report_problem = problems.append if problems is not None else (lambda m: None)
    found: list[Path] = []
    filtered: list[tuple] = []          # what the include/exclude globs removed
    seen: dict[Path, int] = {}          # resolved target -> index in `found`

    def matches(fp: Path, base: Optional[Path],
                globs: list[str]) -> Optional[str]:
        """The glob that matched, or None. Globs match the name, and — under
        a directory — the relative path.

        Matching only the basename makes --exclude 'node_modules/*' a silent
        no-op, which reads as "nothing matched" rather than "unsupported".
        """
        names = [fp.name]
        if base is not None:
            try:
                names.append(fp.relative_to(base).as_posix())
            except ValueError:
                pass
        # fnmatchcase, not fnmatch: fnmatch normalises through os.path.normcase,
        # which on Windows also folds separators. But an extension is not
        # case-sensitive to anyone: detection reads REPORT.CSV as a CSV, and
        # --include '*.csv' silently returned six files of seven with no
        # message and no count discrepancy. Both spellings are tried.
        return next((g for n in names for g in globs
                     if fnmatch.fnmatchcase(n, g)
                     or fnmatch.fnmatchcase(n.lower(), g.lower())), None)

    def add(fp: Path, base: Optional[Path] = None) -> None:
        try:
            if not fp.is_file():
                # collect_files' own docstring says nothing is skipped
                # quietly, and this returned in silence for every named pipe,
                # dangling symlink and symlinked directory found during a
                # walk — the branch that reports them covers only paths named
                # on the command line. "2 file(s) summarized" over a
                # directory holding six entries told the user something
                # false.
                if fp.exists() or fp.is_symlink():
                    try:
                        mode = fp.lstat().st_mode
                        what = ("a broken symlink" if fp.is_symlink()
                                and not fp.exists() else _kind_name(mode))
                    except OSError:
                        what = "not a regular file"
                    report_problem(f"skipped {fp}: {what}")
                return
            rp = fp.resolve()
        except OSError as exc:
            report_problem(f"cannot stat {fp}: {exc.strerror or exc}")
            return
        # Both filters say what they removed. A file dropped by a filter is
        # a file the run did not look at, and "2 file(s) · 2 summarized" over
        # a directory of three is a false statement however the third was
        # dropped — by a --exclude on the command line, or by an `exclude`
        # in a sift.toml sitting beside the files, which is how an attacker
        # who can write one file makes sift not read another.
        if config.include and not matches(fp, base, config.include):
            filtered.append((fp, "did not match --include", ""))
            return
        hit = matches(fp, base, config.exclude) if config.exclude else None
        if hit:
            filtered.append((fp, "excluded by", hit))
            return
        if rp in seen:
            # Same file reached twice (a symlink beside its target). Keep the
            # real name: reporting only 'alias.txt' looks like 'real.txt' was
            # never scanned, and which one wins would depend on sort order.
            kept = found[seen[rp]]
            if kept.is_symlink() and not fp.is_symlink():
                found[seen[rp]] = fp
            return
        seen[rp] = len(found)
        found.append(fp)

    def walk(root: Path) -> list[Path]:
        """Like rglob, but says so when a subtree cannot be read."""
        out: list[Path] = []
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(current.iterdir())
            except OSError as exc:
                report_problem(f"cannot read {current}: {exc.strerror or exc}")
                continue
            for entry in entries:
                try:
                    is_dir = entry.is_dir()
                except OSError as exc:
                    report_problem(f"cannot stat {entry}: {exc.strerror or exc}")
                    continue
                if entry.is_symlink() and not config.follow_symlinks:
                    # For a *file*, a link that stays inside the tree is not
                    # an escape: it names something we were going to read
                    # anyway, and calling that "skipped" was simply untrue.
                    # For a *directory* there is no such exception — pointing
                    # back inside the tree is precisely how a link loops, and
                    # `loopdir -> .` would walk until the stack gave out.
                    #
                    # A link that leaves it is an arbitrary read, and the
                    # guard used to cover only directories — so a single
                    # `ln -s ~/.ssh/id_rsa quarterly-figures.txt` in a shared
                    # folder put a private key in the summary, the terminal,
                    # the report, the on-disk cache and the model prompt,
                    # with nothing saying it had been a link. That is the
                    # whole shared-folder threat this tool is pointed at.
                    try:
                        target = entry.resolve()
                        inside = (not is_dir
                                  and target.is_relative_to(root.resolve()))
                    except (OSError, ValueError):
                        target, inside = None, False
                    if not inside:
                        where = f" -> {target}" if target else ""
                        what = ("a symlink to a directory" if is_dir else
                                f"a symlink out of {root}")
                        report_problem(
                            f"skipped {entry}{where}: {what} "
                            f"(follow_symlinks is off)")
                        continue
                if is_dir:
                    stack.append(entry)
                else:
                    out.append(entry)
        return out

    for raw in paths:
        path = Path(raw)
        try:
            is_dir, is_file = path.is_dir(), path.is_file()
        except OSError as exc:
            report_problem(f"cannot stat {raw}: {exc.strerror or exc}")
            continue
        if is_dir:
            if config.recursive:
                children = walk(path)
            else:
                try:
                    children = sorted(p for p in path.iterdir() if not p.is_dir())
                except OSError as exc:
                    report_problem(f"cannot read {path}: {exc.strerror or exc}")
                    continue
            for child in sorted(children):
                add(child, base=path)
        elif is_file:
            add(path)
        elif path.exists():
            # A named pipe or device is not missing — saying so would send the
            # user looking for a typo that is not there.
            report_problem(f"not a regular file, skipped: {raw}")
        else:
            report_problem(f"not found: {raw}")
    # A symlink and its target are one file; report it under its real name so
    # One line per filter, not one per file: excluding a node_modules tree is
    # the ordinary case and a thousand lines of it would bury the message
    # that matters. Up to three names, then a count.
    for reason, glob in sorted({(r, g) for _, r, g in filtered}):
        hits = [fp for fp, r, g in filtered if (r, g) == (reason, glob)]
        shown = ", ".join(fp.name for fp in sorted(hits)[:3])
        more = f" and {len(hits) - 3} more" if len(hits) > 3 else ""
        report_problem(f"{len(hits)} file(s) {reason}"
                       + (f" {glob!r}" if glob else "") + f": {shown}{more}")
    # A symlink and its target are one file; report it under its real name so
    # the canonical path never disappears because an alias sorted first.
    return sorted(found, key=lambda p: (str(p.parent), p.name))


#: Exit codes. Documented in README so scripts can branch on them.
EXIT_OK = 0
EXIT_NOTHING = 1        # nothing could be analyzed
EXIT_USAGE = 2          # bad invocation or unreadable configuration
EXIT_NO_BACKEND = 3     # --models only: no LLM to ask
EXIT_INTERRUPTED = 130  # Ctrl-C, by shell convention


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point. Turns every expected failure into a message, not a trace."""
    try:
        code = _main(argv)
        # Flush here, inside the guard. Left to the interpreter's shutdown, a
        # closed pipe surfaces as "Exception ignored" noise and exit code 120.
        sys.stdout.flush()
        return code
    except KeyboardInterrupt:
        print("\nsift: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except ConfigError as exc:
        print(f"sift: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except LLMError as exc:
        print(f"sift: {exc}", file=sys.stderr)
        return EXIT_NOTHING
    except BrokenPipeError:
        # `sift … | head` closes the pipe early. Silence the interpreter's
        # second attempt to flush stdout at exit, which prints its own noise.
        _devnull_stdout()
        return EXIT_OK
    except OSError as exc:
        print(f"sift: {exc.strerror or exc}: {exc.filename or ''}".rstrip(": "),
              file=sys.stderr)
        return EXIT_NOTHING


def _devnull_stdout() -> None:
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    except (OSError, ValueError):  # pragma: no cover - stdout already gone
        pass


def _main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        return _serve(argv[1:])
    if argv and argv[0] == "--":
        argv = argv[1:]          # escape hatch for a file actually named "serve"

    args = build_parser().parse_args(argv)

    if getattr(args, "clear_cache", False):
        removed = SummaryCache().clear()
        print(f"sift: removed {removed} cached summaries", file=sys.stderr)
        return EXIT_OK

    if getattr(args, "list_types", False):
        print("Recognized kinds:")
        for k in SUPPORTED_KINDS:
            print(f"  {k}")
        return EXIT_OK

    if getattr(args, "models", False):
        return _list_models(_config_from_args(args))

    if not args.paths:
        build_parser().print_help()
        return EXIT_USAGE

    config = _config_from_args(args)
    config_module.validate(config)
    problems: list[str] = []
    files = collect_files(args.paths, config, problems)
    for problem in problems:
        print(f"sift: {problem}", file=sys.stderr)
    if not files:
        print("sift: no matching files", file=sys.stderr)
        return EXIT_NOTHING

    output_path = getattr(args, "output", None)
    if output_path:
        _check_writable(output_path)     # fail before doing the work, not after
    client = build_client(config)
    cache = SummaryCache(config.cache)
    _print_banner(config, client, len(files))

    summaries = []
    stream = config.output_format == "text"
    out_parts = []
    live = _Progress(client, len(files), config)
    # The same protection summarize_many describes, in the loop that actually
    # runs: three identical failures in a row mean a broken server, and asking
    # it once per file behind a --timeout is minutes of waiting for an answer
    # that cannot come.
    health = ModelHealth()
    for i, fp in enumerate(files, 1):
        live.start(i, fp.name)
        paused = client is not None and health.paused()
        summ = summarize_path(fp, config, None if paused else client, cache)
        if paused:
            summ.notes.append(health.note_for())
        health.record(summ, paused)
        live.finish()
        summaries.append(summ)
        if config.verbose:
            _print_details(summ)
        if stream:
            # color=False for the file: the terminal's escape codes are not
            # something anyone wants written into report.txt.
            out_parts.append(report.render_text(summ, color=False if output_path
                                                else None))
            if not output_path:
                _write(out_parts[-1] + "\n")

    text = _finalize_output(config, summaries, out_parts, to_file=bool(output_path))
    if output_path:
        # errors="replace", and only as a net. One unencodable character —
        # a lone surrogate out of one file — used to raise here after every
        # summary in the run had been produced, leaving a zero-byte report
        # and losing all of them. The extractors are where such a character
        # gets neutralized; this is so that the one they miss costs a
        # question mark rather than the report.
        Path(output_path).write_text(text, encoding="utf-8",
                                     errors="replace")
        if not config.quiet:
            print(f"sift: wrote {len(summaries)} summaries to {output_path}",
                  file=sys.stderr)
    elif not stream:
        _write(text)

    if stream and not output_path and not config.quiet:
        print(report.render_footer(summaries), file=sys.stderr)

    unreadable = sum(1 for s in summaries if s.kind == "error")
    if unreadable == len(summaries) or problems and not summaries:
        return EXIT_NOTHING
    # A path the user named but we could not find is a failure of the run even
    # when the other files worked — otherwise a typo'd filename exits 0.
    return EXIT_NOTHING if any(p.startswith("not found:") for p in problems) \
        else EXIT_OK


def _finalize_output(config, summaries, out_parts, to_file: bool = False) -> str:
    if config.output_format == "json":
        # A file is written as UTF-8, so it can hold the real characters. On a
        # terminal that cannot encode them, escaping keeps the JSON both valid
        # and lossless instead of mangling names and content.
        return report.render_json(summaries,
                                  ensure_ascii=not (to_file or _stdout_is_utf8()))
    if config.output_format == "markdown":
        return report.render_markdown(summaries)
    body = "\n\n".join(out_parts)
    return body + "\n\n" + report.render_footer(summaries, color=False)


class _Progress:
    """The per-file line, updated as the answer streams in.

    A local model takes a minute or more per file. Without this the run shows
    one still line for that whole time, which looks exactly like a hang — and
    the first thing anyone does about a hang is kill it.
    """

    def __init__(self, client, total: int, config: Config) -> None:
        self.total = total
        self.live = (client is not None and not config.quiet
                     and config.output_format == "text" and sys.stderr.isatty())
        self.style = report.Style(report._color_enabled())
        self.enabled = not config.quiet and config.output_format == "text"
        self.prefix = ""
        self.chars = self.thinking = 0
        self.started = time.monotonic()
        if client is not None and self.live:
            client.on_progress = self._tick

    def start(self, index: int, name: str) -> None:
        self.prefix = f"[{index}/{self.total}] {name}"
        self.chars = self.thinking = 0
        self.started = time.monotonic()
        if self.enabled:
            end = "" if self.live else "\n"
            print(self.style.dim(f"{self.prefix} …"), file=sys.stderr, end=end,
                  flush=True)

    def _tick(self, content: int, reasoning: int) -> None:
        self.chars += content
        self.thinking += reasoning
        # The elapsed time, because the token count alone does not say whether
        # six minutes on one file is this model's normal pace or a stall.
        elapsed = int(time.monotonic() - self.started)
        clock = f"{elapsed}s" if elapsed < 90 else f"{elapsed // 60}m{elapsed % 60:02d}s"
        state = (f"thinking, {self.thinking // 4} tokens, {clock}"
                 if not self.chars else f"{self.chars // 4} tokens, {clock}")
        print("\r\x1b[2K" + self.style.dim(f"{self.prefix} … {state}"),
              file=sys.stderr, end="", flush=True)

    def finish(self) -> None:
        if self.live:
            print("\r\x1b[2K", file=sys.stderr, end="", flush=True)


def _print_details(summ) -> None:
    """What -v is for: how the decision was reached, on stderr.

    Kept off stdout so it never contaminates --json, and shown per file so a
    wrong classification can be traced to the detection that caused it.
    """
    s = report.Style(report._color_enabled())
    engine = (f"{summ.provider}:{summ.model}" if summ.llm_used
              else "heuristic")
    facts = [f"kind={summ.kind}", f"mime={summ.mime or '-'}",
             f"confidence={summ.detection_confidence:.2f}",
             f"bytes={summ.size_bytes}", f"engine={engine}",
             f"{summ.elapsed_seconds}s"]
    if summ.language:
        facts.insert(3, f"language={summ.language}")
    if summ.truncated:
        facts.append("TRUNCATED")
    print(s.dim(f"      {summ.filename}: " + "  ".join(facts)), file=sys.stderr)
    for key, value in list(summ.metadata.items())[:8]:
        rendered = str(value)
        if len(rendered) > 90:
            rendered = rendered[:87] + "…"
        print(s.dim(f"        {key}: {rendered}"), file=sys.stderr)
    if summ.error:
        print(s.red(f"        error: {summ.error}"), file=sys.stderr)


def _stdout_is_utf8() -> bool:
    return (sys.stdout.encoding or "").lower().replace("-", "") in (
        "utf8", "utf8mb4", "cp65001")


def _write(text: str) -> None:
    """print() that survives a stdout which cannot encode the text.

    A POSIX-locale CI runner or a Windows console on cp1252 cannot print the
    kind icons, let alone a Japanese filename. Losing the whole report to a
    UnicodeEncodeError is a worse answer than a few escaped characters.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        sys.stdout.flush()
        sys.stdout.buffer.write(text.encode(encoding, "backslashreplace") + b"\n")
        sys.stdout.buffer.flush()


def _check_writable(output_path: str) -> None:
    """Refuse an impossible -o before spending minutes on the analysis.

    Asked, never attempted. Probing by opening the target created a file
    where a dangling symlink pointed and replaced the symlink itself with a
    regular file, blocked forever on a named pipe, and — on a path that did
    not exist — created and then deleted it, taking with it whatever another
    writer had put there in between. A pre-flight check must not change
    anything, so this one only asks the questions the filesystem can answer
    without being touched.
    """
    target = Path(output_path)
    if target.is_dir():
        raise ConfigError(f"-o {output_path}: is a directory")
    try:
        info = target.lstat()
    except OSError:
        info = None
    if info is not None and stat.S_ISFIFO(info.st_mode):
        # Writing to a pipe with no reader blocks forever. That is a choice
        # the user may have made deliberately, so it is not refused — but it
        # must not happen here, before the analysis has even started.
        return
    if target.exists():
        if not os.access(target, os.W_OK):
            raise ConfigError(f"-o {output_path}: Permission denied")
        return
    parent = target.parent if str(target.parent) else Path(".")
    if not parent.is_dir():
        raise ConfigError(f"-o {output_path}: {parent} does not exist")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise ConfigError(f"-o {output_path}: Permission denied")


def _print_banner(config: Config, client, n: int) -> None:
    if config.quiet or config.output_format != "text":
        return
    s = report.Style(report._color_enabled())
    if config.no_llm:
        engine = s.yellow("heuristic mode (--no-llm)")
    elif client is None:
        engine = s.yellow("no local LLM reachable — falling back to heuristics "
                          "(start Ollama, LM Studio, or MLX-LM)")
    else:
        engine = s.green(f"{client.provider} @ {client.endpoint}"
                         + (f" · {config.model}" if config.model else " · auto model"))
    print(s.bold(f"sift {__version__}") + s.dim(f" · {n} file(s) · ") + engine + "\n",
          file=sys.stderr)


def _list_models(config: Config) -> int:
    """Show the backend's models and which of them are resident.

    Printed before any work so the user can see what sift would pick — and
    confirm for themselves that it is a model already in memory.
    """
    client = build_client(config)
    if client is None:
        print("sift: no local LLM reachable (tried Ollama, LM Studio, and "
              "MLX-LM). Start one, or pass --endpoint.", file=sys.stderr)
        return 3
    try:
        models = client.inventory()
    except LLMError as exc:
        print(f"sift: {exc}", file=sys.stderr)
        return 3

    s = report.Style(report._color_enabled())
    print(s.bold(f"{client.provider} @ {client.endpoint}"))
    if not models:
        print("  (server reports no models)")
        return 3
    for m in sorted(models, key=lambda m: (m.loaded is not True, m.id)):
        state = {True: s.green("loaded    "), False: s.dim("not loaded"),
                 None: s.dim("unknown   ")}[m.loaded]
        ctx = f"  ctx {m.context}" if m.context else ""
        print(f"  {state}  {m.id}  [{m.kind}]{ctx}")

    chat = _describe_pick(client, models, need_vision=False)
    vision = _describe_pick(client, models, need_vision=True)
    print(f"\nsift would use: {chat}")
    print(f"for images:     {vision}")
    if not config.allow_model_load and any(m.loaded is None for m in models):
        print(s.dim("(this server does not report residency; models marked "
                    "unknown are used as advertised)"))
    elif not config.allow_model_load:
        print(s.dim("(cold models are never loaded; --allow-model-load "
                    "permits it)"))
    return 0


def _describe_pick(client, models, *, need_vision: bool) -> str:
    """What ensure_model() would choose, without contacting the server again."""
    from .llm.base import _pick
    choice = _pick(models, need_vision=need_vision, loaded_only=True)
    if choice is not None:
        state = ("already loaded" if choice.loaded is True
                 else "load state not reported")
        return f"{choice.id} ({state})"
    cold = _pick(models, need_vision=need_vision, loaded_only=False)
    if cold is None:
        return "nothing suitable — heuristic summary instead"
    return f"nothing loaded; {cold.id} only with --allow-model-load"


def _serve(argv: list[str]) -> int:
    from .server import run_server
    args = build_serve_parser().parse_args(argv)
    try:
        config = _config_from_args(args)
        # The summarize path validates and serve did not, so `sift serve
        # --timeout 0` started happily and `--spool-dir /typo` started and
        # then answered 400 to every upload, forever, with the reason
        # nowhere. A server is the one mode where the person who made the
        # mistake is not the person who meets it.
        config_module.validate(config)
    except ConfigError as exc:
        print(f"sift: {exc}", file=sys.stderr)
        return 2
    client = build_client(config)
    return run_server(args.host, args.port, config, client)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
