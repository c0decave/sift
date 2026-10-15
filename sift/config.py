"""Configuration: layered defaults ← config file ← environment ← CLI overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - older interpreters
    try:  # the backport reads the same dialect
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


class ConfigError(Exception):
    """A setting could not be read. Always reported, never silently dropped."""


DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_LMSTUDIO_ENDPOINT = "http://localhost:1234/v1"
DEFAULT_MLX_LM_ENDPOINT = "http://localhost:8080/v1"

# Providers that speak the OpenAI /chat/completions dialect.
OPENAI_DIALECT = {"lmstudio", "mlx-lm", "openai"}


@dataclass
class Config:
    """Runtime configuration for a sift run.

    Every field has a sensible default so ``Config()`` alone yields a working
    setup that auto-detects a running Ollama, LM Studio, or MLX-LM instance.
    """

    # --- LLM connection -------------------------------------------------
    provider: str = "auto"          # auto | ollama | lmstudio | mlx-lm |
                                    # openai | none
    model: str = ""                  # empty => auto-pick, preferring resident
    endpoint: str = ""               # empty => provider default
    api_key: str = ""                # LM Studio ignores it; some servers need it
    stream: bool = True              # stream tokens: a silent connection is
                                     # closed by proxies mid-generation
    allow_model_load: bool = False   # may we make the server load a cold model?
                                     # False keeps the user's resident model in
                                     # place — loading evicts it.

    # --- generation -----------------------------------------------------
    # 0 means "ask the server": LM Studio and Ollama report the window the
    # model was actually loaded with, and guessing 8192 wastes most of a large
    # one while guessing large overflows a small one. Any explicit value wins.
    context_tokens: int = 0
    # A ceiling, not a target: unused budget costs nothing. 800 was too tight
    # for reasoning models, whose thinking counts against the same limit — they
    # ran out mid-sentence and reported a cut-off summary as a finished one.
    max_output_tokens: int = 2000
    temperature: float = 0.2         # low => faithful, deterministic summaries
    timeout: float = 180.0           # seconds per HTTP call

    # --- extraction -----------------------------------------------------
    max_bytes: int = 25_000_000      # read at most this many bytes per file
    #: How far the active-content scan reads (extract/active.py), where that
    #: is further than max_bytes; 0 = as far as max_bytes. A viewer that hands
    #: a renderer more of a file than sift reads -- a host application gives pdf.js a
    #: PDF of up to 64 MiB -- sets it to at least that much, or the scan of a
    #: clean file past max_bytes says "not checked" and the file is refused.
    #: The extractors still read max_bytes.
    active_max_bytes: int = 0
    time_budget: float = 60.0        # seconds one file may take; 0 = no limit
    #: Seconds of *model* time one file may consume, across every call made
    #: for it. Separate from time_budget because the two bound different
    #: things: time_budget bounds sift's own loops, which are the ones nobody
    #: has measured yet, and this bounds a long file's map-reduce, which is
    #: one call per chunk and can be four hundred of them. When it runs out
    #: the parts already summarized are reduced anyway, so the answer is a
    #: partial summary that says so rather than a failure. 0 = no limit.
    model_time_budget: float = 900.0
    #: Whether a summary may say it came from the cache. True for the CLI,
    #: where the user owns the cache and the note explains the speed. False
    #: over HTTP: the key includes the file's digest, so "from cache" tells
    #: an uploader that this exact file has been summarized here before —
    #: about someone else's file, to someone who only has the bytes.
    disclose_cache: bool = True
    vision: bool = True              # send images to vision-capable models
    # Where `sift serve` writes an upload while it reads it. Empty means the
    # system temporary directory, which is local disk on most machines — and
    # on a host whose local disk is the small one, a run of max_bytes-sized
    # uploads lands in exactly the wrong place. Point this at the large mount
    # and nothing else changes: the file is deleted as soon as the summary is
    # made, and the CLI never spools at all.
    spool_dir: str = ""

    # --- traversal ------------------------------------------------------
    recursive: bool = False
    include: list[str] = field(default_factory=list)   # glob allow-list
    exclude: list[str] = field(default_factory=list)   # glob deny-list
    follow_symlinks: bool = False

    cache: bool = True               # reuse summaries for unchanged content;
                                     # the key covers model and settings too

    # --- output ---------------------------------------------------------
    output_format: str = "text"      # text | json | markdown
    quiet: bool = False
    verbose: bool = False

    # --- behaviour ------------------------------------------------------
    no_llm: bool = False             # extraction + heuristic summary only

    # --- classification -------------------------------------------------
    #: How many tags to ask for per file. 0 — the CLI's default — asks for
    #: none, and then the prompt, the parser and the cache key are the ones
    #: sift had before tags existed.
    tags: int = 0
    #: The categories a file may be put in — Summary.category is one of
    #: these or None. Empty (the default) asks for none. See categories.py.
    categories: list[str] = field(default_factory=list)
    #: Rules that assign a category without a model, first match wins: in a
    #: config file, [[sift.category_rules]] tables. See categories.py.
    category_rules: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    def resolved_endpoint(self, provider: Optional[str] = None) -> str:
        provider = provider or self.provider
        if self.endpoint:
            given = self.endpoint.rstrip("/")
            # An endpoint is where the *API* lives, and people write down
            # where the *server* lives — the address LM Studio prints, the
            # one in the browser. Taken verbatim, "http://host:1234" made
            # every request go to /chat/completions instead of
            # /v1/chat/completions, and LM Studio answers that with HTTP 200
            # and {"error": "Unexpected endpoint or method"} — so no status
            # check fired and every summary failed with "sent no stream
            # events", while --models kept working, because the inventory
            # call uses a different base. Working discovery and broken
            # summaries is the worst pair of symptoms to hand someone.
            if provider in OPENAI_DIALECT and not _has_api_path(given):
                return given + "/v1"
            return given
        if provider == "mlx-lm":
            return DEFAULT_MLX_LM_ENDPOINT
        if provider in OPENAI_DIALECT:
            return DEFAULT_LMSTUDIO_ENDPOINT
        return DEFAULT_OLLAMA_ENDPOINT

    def resolved_spool_dir(self) -> Optional[str]:
        """Where an upload is written, or None for the system default.

        One place, because the check and the use disagreeing is worse than
        either being wrong: validate() expanded `~` and the server passed the
        raw string, so `spool_dir = "~/spool"` passed validation and then
        made every upload fail with the reason nowhere in sight.
        """
        if not self.spool_dir:
            return None
        return str(Path(self.spool_dir).expanduser())

    def with_overrides(self, **overrides: Any) -> "Config":
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        config_path: Optional[Path] = None,
        overrides: Optional[dict[str, Any]] = None,
    ) -> "Config":
        """Build a Config from defaults, an optional TOML file, env, overrides."""
        cfg = cls()
        cfg = _apply_file(cfg, config_path)
        cfg = _apply_env(cfg)
        if overrides:
            cfg = cfg.with_overrides(**overrides)
        return cfg


# ----------------------------------------------------------------------

#: What each numeric setting can mean at all, and the flag that sets it. Kept
#: here rather than in the CLI because the CLI is only one of three ways in:
#: checking the argparse namespace left SIFT_TEMPERATURE=99 and a config file
#: saying max_bytes = -5 to sail through, which is precisely the "a setting
#: that quietly did nothing looks exactly like a setting that worked" the
#: README promises against. --temperature 99 was refused; SIFT_TEMPERATURE=99
#: reached the wire.
#: (lowest meaningful, highest meaningful, the flag, value meaning "unset").
#: context_tokens defaults to 0 for "ask the server what it loaded"; checking
#: the argparse namespace never saw that because an unset flag is None there.
_RANGES: dict = {
    "context_tokens": (256, None, "--context", 0),
    "max_output_tokens": (16, None, "--max-output", None),
    "temperature": (0.0, 2.0, "--temperature", None),
    "timeout": (1.0, None, "--timeout", None),
    "max_bytes": (1, None, "--max-bytes", None),
    "active_max_bytes": (1, None, "active_max_bytes", 0),
}

#: Settings whose value must come from a fixed set. An unknown *value* has to
#: fail the way an unknown *key* does: output_format = "yaml" printed nothing
#: at all and reported "1 summarized" with exit code 0.
_CHOICES: dict = {
    "output_format": ("text", "json", "markdown"),
    "provider": ("auto", "ollama", "lmstudio", "mlx-lm", "openai", "none"),
}


def check_value(key: str, value, where: str = "") -> None:
    """Raise if a setting cannot mean anything. *where* names the source."""
    prefix = f"{where}: " if where else ""
    if key in _RANGES and value is not None:
        low, high, flag, unset = _RANGES[key]
        if value == unset:
            return
        if value < low or (high is not None and value > high):
            bound = f"{low}–{high}" if high is not None else f"at least {low}"
            raise ConfigError(f"{prefix}{key} must be {bound}, got {value} "
                              f"(set with {flag})")
    if key in _CHOICES and value is not None:
        allowed = _CHOICES[key]
        if value not in allowed:
            raise ConfigError(f"{prefix}{key} must be one of "
                              f"{', '.join(allowed)}, got {value!r}")


#: The most tags a file may be asked for. A tag is a filter, and a list of
#: fifty is a summary.
MAX_TAGS = 32


def validate(cfg: "Config") -> None:
    """Every setting, however it arrived — flag, env var or config file."""
    for key in list(_RANGES) + list(_CHOICES):
        check_value(key, getattr(cfg, key, None))
    tags = getattr(cfg, "tags", 0)
    if isinstance(tags, bool) or not isinstance(tags, int) \
            or not 0 <= tags <= MAX_TAGS:
        raise ConfigError(f"tags must be a whole number from 0 to "
                          f"{MAX_TAGS}, got {tags!r}")
    from .categories import check_rules
    try:
        check_rules(cfg.categories, cfg.category_rules)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    # Checked when the server starts rather than when the first upload
    # arrives: a typo in a mount path would otherwise turn into a 500 for
    # whoever uploaded first, minutes or days later.
    if cfg.spool_dir:
        spool = Path(cfg.resolved_spool_dir())
        if not spool.is_dir():
            raise ConfigError(f"spool_dir is not a directory: {cfg.spool_dir}")
        if not os.access(spool, os.W_OK | os.X_OK):
            raise ConfigError(f"spool_dir is not writable: {cfg.spool_dir}")



#: Settings a config file found by *searching* may not set. A sift.toml sitting
#: in a directory you happened to cd into is data, not instruction: one holding
#: endpoint = "http://elsewhere/v1" silently sent every analyzed file there,
#: with an api_key of its choosing and permission to evict the operator's
#: loaded models. A project-local file has every business setting exclude or
#: max_bytes and none setting where the contents go. Naming the file with
#: --config or SIFT_CONFIG is the user saying they meant it.
#: spool_dir is here for the same reason as endpoint: it decides where an
#: uploaded file is written, and a directory you cd into does not get to
#: choose that. The first version of this note said a readable spool_dir
#: "hands every upload to whoever can read it", which is not true —
#: TemporaryDirectory creates the workdir 0700 whatever the parent's mode is,
#: so the file inside is unreachable. The real objection is simpler and does
#: not need the exaggeration: a found config file should not be able to send
#: other people's uploads to a volume of its choosing, whether to fill it, to
#: park them somewhere a later process will pick them up, or to make every
#: upload fail by naming a path that does not exist.
_TRUSTED_ONLY = ("endpoint", "api_key", "provider", "model",
                 "allow_model_load", "follow_symlinks", "spool_dir")


def _config_search_paths() -> "list[tuple[Path, bool]]":
    """Where to look, and whether the user pointed at it.

    SIFT_CONFIG is the user naming a file as surely as --config is, so it is
    trusted the same way — and a SIFT_CONFIG that does not exist is an error
    rather than a quiet fall-through to ./sift.toml, which would hand the run
    to whatever happened to be in the working directory.
    """
    # (path, trusted, required). "required" rather than comparing the string
    # back later: Path() collapses "./" and duplicate slashes, so
    # SIFT_CONFIG=./sift.toml — the most natural way to write it — did not
    # compare equal to its own Path, and a missing file was silently ignored,
    # falling through to whatever ./sift.toml happened to be.
    paths: "list[tuple[Path, bool, bool]]" = []
    env = os.environ.get("SIFT_CONFIG")
    if env:
        paths.append((Path(env).expanduser(), True, True))
    paths.append((Path.cwd() / "sift.toml", False, False))
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    # The user's own config directory is theirs; only the working directory is
    # somewhere a file can arrive without them putting it there.
    paths.append((base / "sift" / "config.toml", True, False))
    return paths


def cls_defaults() -> dict:
    """Every setting's default value, for comparing a config file against."""
    return {f.name: getattr(Config(), f.name) for f in fields(Config)}


def _apply_file(cfg: Config, config_path: Optional[Path]) -> Config:
    """Read the first config file that exists.

    A file the user named explicitly is held to a stricter standard than one we
    merely went looking for: if they passed --config, a missing or broken file
    is an error. Ignoring it would run with defaults while the user believes
    their settings applied — the failure mode where sift looks like it worked.
    """
    explicit = config_path is not None
    if tomllib is None:
        if explicit:
            raise ConfigError(
                f"cannot read {config_path}: this interpreter has no tomllib "
                f"(Python < 3.11) and the 'tomli' backport is not installed.")
        return cfg
    candidates = ([(config_path, True, True)] if explicit
                  else _config_search_paths())
    for path, trusted, required in candidates:
        if not path:
            continue
        if not path.is_file():
            if required:
                raise ConfigError(f"config file not found: {path}")
            continue
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except OSError as exc:
            if explicit:
                raise ConfigError(f"cannot read {path}: {exc}") from exc
            continue
        except ValueError as exc:
            raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
        section = data.get("sift", data)
        if not isinstance(section, dict):
            raise ConfigError(f"{path}: the [sift] section must be a table")
        if "sift" in data:
            # TOML forces bare keys above the first table, so pasting a
            # snippet at the top of a file that also has [sift] is the natural
            # mistake — and those keys were silently discarded, which is the
            # one thing _validated exists to prevent.
            stray = [k for k, v in data.items()
                     if k != "sift" and not isinstance(v, dict)]
            if stray:
                raise ConfigError(
                    f"{path}: {', '.join(sorted(stray))} "
                    f"{'is' if len(stray) == 1 else 'are'} above the [sift] "
                    f"table and would be ignored — move "
                    f"{'it' if len(stray) == 1 else 'them'} below it")
        settings = _validated(section, path)
        if not trusted:
            # Only where the value would actually change something. The
            # project's own README says to copy sift.example.toml to
            # ./sift.toml, and that example sets all six of these to their
            # defaults — provider = "auto", endpoint = "", api_key = "",
            # allow_model_load = false — so the refusal protected against
            # nothing and broke every run in that directory with exit 2. A
            # setting that says what sift would have done anyway cannot
            # redirect a file anywhere.
            defaults = cls_defaults()
            reached = [k for k in _TRUSTED_ONLY
                       if k in settings and settings[k] != defaults.get(k)]
            if reached:
                # Named individually, because "where the contents are sent"
                # is true of endpoint and merely confusing for follow_symlinks
                # — and a warning a reader cannot map onto the setting in
                # front of them teaches them to click through it.
                why = {"endpoint": "where the contents of every file you "
                                   "analyze are sent",
                       "api_key": "what credential is sent with them",
                       "provider": "which backend receives them",
                       "model": "which model receives them",
                       "allow_model_load": "whether the server may evict the "
                                           "model someone else is using",
                       "follow_symlinks": "whether a link out of the tree is "
                                          "followed",
                       "spool_dir": "where an uploaded file is written on "
                                    "disk"}
                raise ConfigError(
                    f"{path} was found by searching, not named by you, and it "
                    f"sets {', '.join(reached)} — which decides "
                    + "; ".join(why[k] for k in reached)
                    + f". Pass --config {path} if you meant it.")
        cfg = cfg.with_overrides(**settings)
        break
    return cfg


def _validated(section: dict, path: Path) -> dict[str, Any]:
    """Reject unknown keys and wrong types instead of dropping them quietly."""
    known = Config.__dataclass_fields__
    clean: dict[str, Any] = {}
    for raw_key, value in section.items():
        key = raw_key.replace("-", "_")
        if key not in known:
            close = _closest(key, known)
            hint = f" Did you mean {close!r}?" if close else ""
            raise ConfigError(f"{path}: unknown setting {raw_key!r}.{hint}")
        clean[key] = _coerce(key, value, known[key].type, path)
        check_value(key, clean[key], str(path))
    return clean


def _coerce(key: str, value: Any, declared: Any, path: Path) -> Any:
    """Check a config value against the field's declared type."""
    name = declared if isinstance(declared, str) else getattr(declared, "__name__", "")
    if name.startswith("list[dict"):
        # Tables — [[sift.category_rules]] — kept as tables. Turning every
        # item into a string, as the branch below does for globs, made the
        # setting impossible to write in a config file at all.
        if not isinstance(value, list) or not all(isinstance(item, dict)
                                                  for item in value):
            raise ConfigError(
                f"{path}: {key} must be a list of tables — write each as "
                f"[[sift.{key}]]")
        return [dict(item) for item in value]
    if name.startswith("list"):
        if isinstance(value, str):
            if key not in ("include", "exclude"):
                raise ConfigError(
                    f"{path}: {key} must be a list, not a string — write "
                    f'{key} = ["{value}"]')
            raise ConfigError(
                f"{path}: {key} must be a list of globs, not a string — write "
                f'{key} = ["{value}"]. As a string it is read character by '
                f"character and matches everything.")
        if not isinstance(value, list):
            raise ConfigError(f"{path}: {key} must be a list")
        return [str(v) for v in value]
    if name == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: {key} must be true or false")
        return value
    if name == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: {key} must be a whole number")
        return value
    if name == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: {key} must be a number")
        return float(value)
    if name == "str":
        if not isinstance(value, str):
            raise ConfigError(f"{path}: {key} must be a string")
        return value
    return value


def _closest(key: str, known) -> Optional[str]:
    import difflib
    match = difflib.get_close_matches(key, list(known), n=1, cutoff=0.7)
    return match[0] if match else None


def _as_bool(raw: str) -> bool:
    """Parse an env-var boolean, rejecting anything ambiguous.

    Raising on an unknown value matters: silently reading SIFT_ALLOW_MODEL_LOAD
    ="no" as true would evict the user's model behind their back.
    """
    low = raw.strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a boolean: {raw!r}")


_ENV_MAP = {
    "SIFT_PROVIDER": ("provider", str),
    "SIFT_MODEL": ("model", str),
    "SIFT_ENDPOINT": ("endpoint", str),
    "SIFT_API_KEY": ("api_key", str),
    "SIFT_CONTEXT": ("context_tokens", int),
    "SIFT_MAX_OUTPUT": ("max_output_tokens", int),
    "SIFT_TEMPERATURE": ("temperature", float),
    "SIFT_TIMEOUT": ("timeout", float),
    "SIFT_MAX_BYTES": ("max_bytes", int),
    "SIFT_ALLOW_MODEL_LOAD": ("allow_model_load", _as_bool),
    "SIFT_STREAM": ("stream", _as_bool),
    "SIFT_CACHE": ("cache", _as_bool),
    "SIFT_TIME_BUDGET": ("time_budget", float),
    "SIFT_SPOOL_DIR": ("spool_dir", str),
}


def _apply_env(cfg: Config) -> Config:
    """Apply SIFT_* variables, refusing values that cannot be read.

    A discarded env var is invisible: SIFT_TIMEOUT=30s silently keeps the
    default, and SIFT_ALLOW_MODEL_LOAD=maybe would decide on its own whether
    the user's loaded model may be evicted.
    """
    updates: dict[str, Any] = {}
    for env_key, (field_name, caster) in _ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        try:
            updates[field_name] = caster(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{env_key}={raw!r} is not valid: {exc}") from exc
        check_value(field_name, updates[field_name], env_key)
    return cfg.with_overrides(**updates) if updates else cfg


def _has_api_path(endpoint: str) -> bool:
    """Whether an endpoint already names the API and not just the server.

    Anything with a path component is left alone: a reverse proxy may serve
    the API under /llm or /openai, and second-guessing that would break the
    setups that are already right.
    """
    from urllib.parse import urlparse
    try:
        path = urlparse(endpoint).path.strip("/")
    except ValueError:
        return True
    return bool(path)
