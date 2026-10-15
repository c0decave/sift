"""Summary cache, keyed by what actually determines the answer.

Summarizing a directory with a local model costs minutes: the operator's 40B
took 663 seconds for nine small files. Re-running after changing one file
should cost one file, not nine. That only holds if the key is exact — a cache
that returns a summary produced by a different model, or from a file that has
since changed, is worse than no cache, because the answer looks fresh.

So the key covers everything that can change the answer: the file's content
(hashed, not its mtime — a restored backup has a new mtime and identical
bytes), the provider and model, the generation settings, and a version stamp
that is bumped whenever prompts or extraction change. Nothing else is stored,
and an unreadable or corrupt entry is simply a miss.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

#: Bumped by hand when the storage format changes. It is not the whole story:
#: what actually decides whether an old entry is still right is the code that
#: produced it, and that is fingerprinted below.
CACHE_FORMAT = 5

#: A shape such as {"cache_version": ..., "summary": ...} can occur in a
#: project JSON document by accident. Ownership must be explicit before sift
#: deletes anything below an operator-selected SIFT_CACHE_DIR.
CACHE_OWNER = "sift-summary-cache"

_CHUNK = 1 << 20
#: Cache entries are bounded by the extracted text and metadata. Refuse to
#: parse anything larger while deciding whether clear() owns it: a user may
#: point SIFT_CACHE_DIR at a directory containing arbitrary files.
_MAX_ENTRY_FILE = 8 * 1024 * 1024

#: Modules whose behaviour determines a summary. Anything else — the CLI, the
#: report renderer, the server — changes how a summary is displayed, not what
#: it says.
_ANSWER_MODULES = ("summarize.py", "detect.py", "chunk.py",
                   "llm", "extract")


def _code_fingerprint() -> str:
    """A digest of the code that decides what a summary says.

    Relying on someone remembering to bump a constant is how a cache goes
    stale: the heuristics were rewritten twice in one session and the old
    answers kept being served, which looked exactly like the rewrite not
    working. Deriving the version from the source makes forgetting impossible.
    Unreadable sources (a zipped install) fall back to the manual number.
    """
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    try:
        paths: list[Path] = []
        for entry in _ANSWER_MODULES:
            target = here / entry
            paths.extend(sorted(target.rglob("*.py")) if target.is_dir()
                         else [target])
        for path in paths:
            digest.update(path.read_bytes())
    except OSError:  # pragma: no cover - installed without sources
        return "nosrc"
    return digest.hexdigest()[:12]


#: What the cache may occupy before the oldest entries are evicted. A
#: summary is a few kilobytes, so this is room for tens of thousands of
#: them — and a hard stop on an unauthenticated disk-fill.
MAX_CACHE_BYTES = 256 * 1024 * 1024

def _occupied(info) -> int:
    """What an entry costs the volume, not what it says it contains.

    A 900-byte summary occupies a 4 KB block. Counting st_size understated a
    60,000-entry cache by 2.6x — 94.7 MB claimed against 245.8 MB allocated.
    st_blocks is in 512-byte units by POSIX definition and absent on
    Windows, where st_size is the best available answer.
    """
    blocks = getattr(info, "st_blocks", None)
    return blocks * 512 if blocks else info.st_size


CACHE_VERSION = f"{CACHE_FORMAT}-{_code_fingerprint()}"


def cache_dir() -> Path:
    base = os.environ.get("SIFT_CACHE_DIR")
    if base:
        return Path(base)
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "sift"


def file_digest(path: Path) -> str:
    """SHA-256 of the file's bytes, read in chunks so size does not matter."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def cache_key(content_hash: str, config, provider: str, model: str,
              endpoint: str = "", filename: str = "") -> str:
    """One key per (content, engine, settings) triple.

    The settings are in the key rather than the value because a summary made
    with a 2000-token ceiling is a different summary from one made with 800,
    and silently serving the wrong one is the failure this whole module exists
    to avoid.

    The endpoint is in the key too, and that is not belt-and-braces. When the
    model is auto-picked it is still unresolved at lookup time — empty for
    every server — so without the address, two different machines answering on
    the same provider shared one entry. The suite caught it: a summary produced
    by a stub server was served for a run against a dead one.

    So is the filename, for two reasons: the extension steers detection, and
    the name is in the prompt. Identical bytes named t.txt and t.csv are a
    text file and a table, and the cache served the first answer for both —
    "Plain text with 3 lines" for a CSV.
    """
    parts = [
        str(CACHE_VERSION), content_hash, provider or "-", model or "-",
        endpoint or "-", filename or "-",
        str(config.context_tokens), str(config.max_output_tokens),
        f"{config.temperature:.3f}", str(config.max_bytes),
        "vision" if config.vision else "novision",
        "llm" if not config.no_llm else "heuristic",
    ]
    # What changes the answer only when it is asked for, and only then: a
    # default run's key is the one it had before these settings existed.
    # The active-content scan reads further than max_bytes only where
    # active_max_bytes asks it to (extract/active.py reach).
    from .extract.active import reach
    scanned = reach(config.max_bytes, getattr(config, "active_max_bytes", 0))
    if scanned != config.max_bytes:
        parts.append(f"active={scanned}")
    tags = getattr(config, "tags", 0)
    if tags:
        parts.append(f"tags={tags}")
    categories = getattr(config, "categories", None) or []
    rules = getattr(config, "category_rules", None) or []
    if categories or rules:
        from .categories import category_set_hash
        parts.append(f"categories={category_set_hash(categories, rules)}")
    material = "\x1f".join(parts)
    # surrogateescape, not plain utf-8: a filename holding an invalid byte
    # arrives as a lone surrogate, which utf-8 refuses to encode. That raised
    # a UnicodeEncodeError — a ValueError, not an OSError — straight past
    # summarize_path's promise never to raise for a bad file, and the whole
    # run died on a directory listing.
    return hashlib.sha256(
        material.encode("utf-8", "surrogateescape")).hexdigest()


class SummaryCache:
    """A directory of JSON files. Absent, unreadable and stale all mean miss."""

    def __init__(self, enabled: bool = True, directory: Optional[Path] = None):
        self.enabled = enabled
        self.dir = Path(directory) if directory else cache_dir()
        self.hits = 0
        self.evicted = 0
        #: Running total of what the cache occupies, so the ordinary write
        #: does not walk the directory. None until the first write, which
        #: measures what is already there — a fresh SummaryCache per CLI run
        #: otherwise starts at zero and never notices the ceiling.
        self._estimate = None
        self.misses = 0

    def _path(self, key: str) -> Path:
        # Two levels of fan-out: one directory with 100k entries is slow to
        # list and unpleasant to inspect by hand.
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            # surrogateescape on both sides. Writing with it and reading
            # without meant an entry naming an undecodable file was written
            # once and never readable again: every run a miss, every run
            # another write. Half a fix is worse than none — it looks like a
            # working cache and is a growing pile of unreadable files.
            with self._path(key).open("r", encoding="utf-8",
                                      errors="surrogateescape") as fh:
                entry = json.load(fh)
        except (OSError, ValueError):
            self.misses += 1
            return None
        if (not isinstance(entry, dict)
                or entry.get("cache_owner") != CACHE_OWNER
                or entry.get("cache_version") != CACHE_VERSION):
            self.misses += 1
            return None
        self.hits += 1
        # Mark it used, so eviction runs least-recently-*used* rather than
        # oldest-written. Without this an attacker's fresh uploads evicted
        # the entries the defender actually re-reads — the wrong end of the
        # cache. Explicit rather than relying on st_atime, which most
        # filesystems are mounted not to keep.
        try:
            os.utime(self._path(key), None)
        except OSError:
            pass                        # a read-only cache is still a cache
        return entry.get("summary")

    def put(self, key: str, summary: dict) -> None:
        if not self.enabled:
            return
        target = self._path(key)
        entry = {"cache_owner": CACHE_OWNER,
                 "cache_version": CACHE_VERSION, "stored_at": int(time.time()),
                 "summary": summary}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a crash mid-write must not leave a truncated
            # entry that later reads as a valid summary.
            temp = None
            try:
                with tempfile.NamedTemporaryFile(
                        # errors="surrogateescape": a summary can carry an
                        # undecodable filename, and a cache write must never
                        # be the thing that ends a run.
                        "w", encoding="utf-8", errors="surrogateescape",
                        dir=target.parent,
                        prefix=".tmp-", delete=False) as fh:
                    temp = Path(fh.name)
                    json.dump(entry, fh, ensure_ascii=False)
                temp.replace(target)
            finally:
                # Named before the dump, not after: when the dump raised — the
                # lone-surrogate case — the file was never named and never
                # removed, so every run over the same input left another one
                # behind, and clear() counts *.json and could not see them.
                if temp is not None and temp.exists():
                    temp.unlink(missing_ok=True)
            # Cheap in the ordinary case: the running estimate is updated
            # per write and the directory is walked only when it says the
            # ceiling might have been crossed.
            if self._estimate is None:
                # Once per process, before the first write is counted: the
                # estimate started at zero, so it only ever counted *this*
                # run's writes. A CLI run over 600 files writes about 1% of
                # the ceiling, so _make_room was never called and the cache
                # grew without bound across runs — measured at nine times the
                # ceiling after twenty short runs, with zero evictions. The
                # walk is right; doing it per write was what was wrong.
                self._make_room()
            else:
                self._estimate += _occupied(target.stat())
                if self._estimate > MAX_CACHE_BYTES:
                    self._make_room()
        except (OSError, ValueError):
            # ValueError covers UnicodeEncodeError: a file whose *content* is
            # a lone surrogate — 13 bytes of {"\ud800": 1} — escaped this and
            # killed the whole run. surrogateescape handles a name that is not
            # UTF-8; nothing handles a surrogate the JSON parser handed back
            # as a genuine character. A cache that cannot write is still a
            # cache.
            pass

    def _make_room(self) -> None:
        """Keep the cache under its ceiling, least recently *used* out first.

        There was no ceiling and no eviction. The key includes the filename,
        so thirty uploads of identical content under thirty names wrote
        thirty entries, and on a host whose local disk is the small volume
        that is an unauthenticated way to fill it.

        Four things this has to get right, each of which it got wrong first:

        Cost. Walking the directory on every write is O(entries) forever —
        measured at 613 ms per summary at 60,000 entries, which the ceiling
        alone guarantees the cache will reach. The walk now happens only
        when the running estimate says it might be needed, so the ordinary
        write costs nothing.

        Size. st_size is the logical length; a 900-byte entry occupies a
        4 KB block. Summed over 60,000 small files that is 94.7 MB counted
        against 245.8 MB allocated — a "256 MB" cache was 660 MB of the
        small volume. st_blocks is what the volume actually loses.

        Concurrency. One `stat` of a file another process has just unlinked
        raised FileNotFoundError out of the listing comprehension, and the
        except-clause skipped the whole trim: a cache four times over its
        ceiling stayed there. Each entry is now stat'ed on its own.

        Use, not age. get() never touched the file, so eviction ran on write
        time — an attacker's fresh uploads evicted the defender's most
        reused entries first, which is the wrong end of the cache.
        """
        entries = []
        total = 0
        for path in self._owned_paths():
            try:
                info = path.stat()
            except OSError:
                continue                 # vanished under us; not our business
            size = _occupied(info)
            entries.append((info.st_mtime, size, path))
            total += size
        self._estimate = total
        if total <= MAX_CACHE_BYTES:
            return
        # Down to three quarters, not to exactly the ceiling: evicting one
        # entry per write once full would make every subsequent write pay for
        # a full directory scan and a delete.
        goal = MAX_CACHE_BYTES * 3 // 4
        for _, size, path in sorted(entries):
            if total <= goal:
                break
            try:
                path.unlink()
                total -= size
                self.evicted += 1
            except OSError:
                continue
        self._estimate = total

    def _owned_paths(self):
        """Yield only files sift can prove it created.

        Deliberately one directory deep. ``SIFT_CACHE_DIR`` may name a project
        tree, so a recursive walk would turn unrelated JSON into cache data
        before its contents were even considered.
        """
        if not self.dir.is_dir():
            return
        try:
            buckets = list(self.dir.iterdir())
        except OSError:
            return
        for bucket in buckets:
            if (bucket.is_symlink() or not bucket.is_dir()
                    or len(bucket.name) != 2
                    or any(c not in "0123456789abcdef" for c in bucket.name)):
                continue
            try:
                candidates = list(bucket.iterdir())
            except OSError:
                continue
            for path in candidates:
                key = path.name[:-5] if path.name.endswith(".json") else ""
                if (path.is_symlink() or len(key) != 64
                        or any(c not in "0123456789abcdef" for c in key)
                        or bucket.name != key[:2]
                        or not self._owns_entry(path)):
                    continue
                yield path

    def clear(self) -> int:
        """Remove sift entries, never the directory named by the operator.

        ``SIFT_CACHE_DIR`` is an arbitrary path. Recursively deleting that
        path made a typo such as a project directory destructive. An entry is
        ours only when it sits at the exact two-level path produced by
        :meth:`_path` and carries sift's cache envelope. Everything else is
        preserved, including symlinks and nested directories.
        """
        if not self.dir.is_dir():
            return 0
        count = 0
        touched = set()
        for path in list(self._owned_paths()):
            try:
                path.unlink()
            except OSError:
                continue
            count += 1
            touched.add(path.parent)
        for bucket in touched:
            try:
                bucket.rmdir()           # only when nothing else is there
            except OSError:
                pass
        return count

    @staticmethod
    def _owns_entry(path: Path) -> bool:
        """Whether *path* has the bounded on-disk envelope written by put()."""
        try:
            if not path.is_file() or path.stat().st_size > _MAX_ENTRY_FILE:
                return False
            with path.open("r", encoding="utf-8",
                           errors="surrogateescape") as fh:
                entry = json.load(fh)
        except (OSError, ValueError):
            return False
        return (isinstance(entry, dict)
                and entry.get("cache_owner") == CACHE_OWNER
                and isinstance(entry.get("cache_version"), str)
                and isinstance(entry.get("stored_at"), (int, float))
                and not isinstance(entry.get("stored_at"), bool)
                and isinstance(entry.get("summary"), dict))

    def stats(self) -> str:
        total = self.hits + self.misses
        if not total:
            return ""
        return f"{self.hits}/{total} from cache"
