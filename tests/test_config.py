"""Configuration: the layers, their order, and what happens to a bad value.

A setting that silently does nothing looks exactly like a setting that worked,
which is why every rejection path here is asserted rather than assumed. The
precedence claim in the README is load-bearing — scripts depend on a flag
beating an exported variable — so each layer is checked against the one below.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sift.config import Config, ConfigError, _as_bool


class EnvGuard(unittest.TestCase):
    """Every test runs with a clean SIFT_* environment."""

    def setUp(self):
        keep = {k: v for k, v in os.environ.items() if not k.startswith("SIFT_")}
        patcher = mock.patch.dict(os.environ, keep, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, body: str, name="sift.toml") -> Path:
        path = self.dir / name
        path.write_text(body, encoding="utf-8")
        return path


class PrecedenceTest(EnvGuard):
    """CLI flag > environment > config file > default, for every type."""

    def test_default_when_nothing_is_set(self):
        cfg = Config.load()
        self.assertEqual(cfg.provider, "auto")
        self.assertEqual(cfg.timeout, 180.0)
        self.assertTrue(cfg.stream)

    def test_file_beats_default(self):
        path = self.write('[sift]\nprovider = "ollama"\ntimeout = 45.0\n'
                          'context_tokens = 16384\nstream = false\n')
        cfg = Config.load(path)
        self.assertEqual(cfg.provider, "ollama")
        self.assertEqual(cfg.timeout, 45.0)
        self.assertEqual(cfg.context_tokens, 16384)
        self.assertFalse(cfg.stream)

    def test_env_beats_file(self):
        path = self.write('[sift]\nprovider = "ollama"\ntimeout = 45.0\n')
        os.environ["SIFT_PROVIDER"] = "lmstudio"
        os.environ["SIFT_TIMEOUT"] = "12.5"
        cfg = Config.load(path)
        self.assertEqual(cfg.provider, "lmstudio")
        self.assertEqual(cfg.timeout, 12.5)

    def test_flag_beats_env(self):
        os.environ["SIFT_PROVIDER"] = "lmstudio"
        os.environ["SIFT_MODEL"] = "from-env"
        cfg = Config.load(None, {"provider": "ollama", "model": "from-flag"})
        self.assertEqual(cfg.provider, "ollama")
        self.assertEqual(cfg.model, "from-flag")

    def test_an_unset_flag_does_not_erase_the_layer_below(self):
        """argparse hands us None for every flag the user did not type."""
        os.environ["SIFT_MODEL"] = "from-env"
        cfg = Config.load(None, {"provider": None, "model": None})
        self.assertEqual(cfg.model, "from-env")

    def test_all_four_layers_at_once(self):
        path = self.write('[sift]\nprovider = "ollama"\nmodel = "from-file"\n'
                          'timeout = 45.0\n')
        os.environ["SIFT_MODEL"] = "from-env"
        cfg = Config.load(path, {"provider": "lmstudio"})
        self.assertEqual(cfg.provider, "lmstudio")   # flag
        self.assertEqual(cfg.model, "from-env")      # env over file
        self.assertEqual(cfg.timeout, 45.0)          # file over default
        self.assertEqual(cfg.temperature, 0.2)       # default


class RejectionTest(EnvGuard):
    """A value we cannot read is an error that names it, never a shrug."""

    def test_a_named_config_that_does_not_exist(self):
        with self.assertRaises(ConfigError) as ctx:
            Config.load(self.dir / "nope.toml")
        self.assertIn("not found", str(ctx.exception))

    def test_malformed_toml(self):
        path = self.write("not = = toml [[[\n")
        with self.assertRaises(ConfigError) as ctx:
            Config.load(path)
        self.assertIn("not valid TOML", str(ctx.exception))

    def test_an_unknown_key_is_named_and_a_near_miss_suggested(self):
        path = self.write('[sift]\nprovder = "ollama"\n')
        with self.assertRaises(ConfigError) as ctx:
            Config.load(path)
        self.assertIn("provder", str(ctx.exception))
        self.assertIn("provider", str(ctx.exception))

    def test_a_string_where_a_number_belongs(self):
        path = self.write('[sift]\ncontext_tokens = "lots"\n')
        with self.assertRaises(ConfigError):
            Config.load(path)

    def test_include_written_as_a_string_is_refused_with_the_fix(self):
        """As a string it iterates character by character and matches all."""
        path = self.write('[sift]\ninclude = "*.csv"\n')
        with self.assertRaises(ConfigError) as ctx:
            Config.load(path)
        self.assertIn('["*.csv"]', str(ctx.exception))

    def test_a_list_of_globs_is_accepted(self):
        path = self.write('[sift]\ninclude = ["*.csv", "*.eml"]\n')
        self.assertEqual(Config.load(path).include, ["*.csv", "*.eml"])

    def test_a_number_where_a_boolean_belongs(self):
        path = self.write("[sift]\nstream = 1\n")
        with self.assertRaises(ConfigError):
            Config.load(path)

    def test_an_env_value_that_cannot_be_parsed(self):
        os.environ["SIFT_TIMEOUT"] = "30s"
        with self.assertRaises(ConfigError) as ctx:
            Config.load()
        self.assertIn("SIFT_TIMEOUT", str(ctx.exception))

    def test_an_ambiguous_boolean_never_reads_as_true(self):
        """SIFT_ALLOW_MODEL_LOAD decides whether a loaded model may be evicted.

        Guessing here would swap the user's model on the strength of the word
        "maybe".
        """
        os.environ["SIFT_ALLOW_MODEL_LOAD"] = "maybe"
        with self.assertRaises(ConfigError):
            Config.load()

    def test_boolean_spellings_that_are_unambiguous(self):
        for raw, expected in (("1", True), ("true", True), ("YES", True),
                              ("on", True), ("0", False), ("false", False),
                              ("no", False), ("Off", False)):
            self.assertEqual(_as_bool(raw), expected, raw)
        for raw in ("maybe", "", "2", "y", "n"):
            with self.assertRaises(ValueError, msg=raw):
                _as_bool(raw)

    def test_an_empty_env_var_is_ignored_rather_than_rejected(self):
        """Unsetting by export SIFT_MODEL= is a normal shell idiom."""
        os.environ["SIFT_MODEL"] = ""
        self.assertEqual(Config.load().model, "")


class ValidationLayerTest(EnvGuard):
    """Range and choice checks belong on the merged config, not on argparse.

    Checking the namespace meant --temperature 99 was refused with exit 2
    while SIFT_TEMPERATURE=99 reached the wire, and a config file saying
    max_bytes = -5 produced "read length must be non-negative" buried in a
    note under a successful-looking run.
    """

    def test_an_out_of_range_env_var_is_refused(self):
        for var, value in (("SIFT_TEMPERATURE", "99.0"),
                           ("SIFT_MAX_BYTES", "-5"),
                           ("SIFT_MAX_OUTPUT", "0"),
                           ("SIFT_CONTEXT", "10")):
            os.environ[var] = value
            with self.assertRaises(ConfigError, msg=var) as ctx:
                Config.load()
            self.assertIn(var, str(ctx.exception))
            del os.environ[var]

    def test_an_out_of_range_config_value_is_refused(self):
        path = self.write("max_bytes = -5\n")
        with self.assertRaises(ConfigError):
            Config.load(path)

    def test_the_unset_sentinel_is_not_out_of_range(self):
        """context_tokens defaults to 0, meaning "ask the server"."""
        self.assertEqual(Config.load().context_tokens, 0)

    def test_an_unknown_output_format_is_refused_like_an_unknown_key(self):
        """It printed nothing at all and reported "1 summarized", exit 0."""
        path = self.write('output_format = "yaml"\n')
        with self.assertRaises(ConfigError) as ctx:
            Config.load(path)
        self.assertIn("text, json, markdown", str(ctx.exception))

    def test_an_unknown_provider_is_refused(self):
        path = self.write('provider = "gpt5"\n')
        with self.assertRaises(ConfigError):
            Config.load(path)

    def test_keys_above_the_sift_table_are_refused_not_dropped(self):
        """TOML forces bare keys above the first table, so this is the
        natural mistake — and they were discarded in silence."""
        path = self.write('max_bytes = 123\n\n[sift]\nquiet = true\n')
        with self.assertRaises(ConfigError) as ctx:
            Config.load(path)
        self.assertIn("max_bytes", str(ctx.exception))


class DiscoveryTest(EnvGuard):
    def test_a_config_found_by_search_is_not_required_to_exist(self):
        """Without --config, an absent file is normal, not an error."""
        with mock.patch("sift.config._config_search_paths",
                        return_value=[(self.dir / "absent.toml", False, False)]):
            self.assertEqual(Config.load().provider, "auto")

    def test_a_broken_config_found_by_search_is_still_reported(self):
        """Silently running on defaults would hide the user's own typo."""
        path = self.write("nonsense = = =\n", name="found.toml")
        with mock.patch("sift.config._config_search_paths",
                        return_value=[(path, False, False)]):
            with self.assertRaises(ConfigError):
                Config.load()

    def test_a_found_config_may_not_redirect_where_the_data_goes(self):
        """A sift.toml in a directory you cd into is data, not instruction.

        One holding an endpoint sent the contents of every analyzed file
        there, with an api_key of its choosing and permission to evict the
        operator's loaded models.
        """
        path = self.write('endpoint = "http://elsewhere.example/v1"\n',
                          name="sift.toml")
        with mock.patch("sift.config._config_search_paths",
                        return_value=[(path, False, False)]):
            with self.assertRaises(ConfigError) as ctx:
                Config.load()
        self.assertIn("--config", str(ctx.exception))
        # Named by the user, the same file is honoured.
        self.assertEqual(Config.load(path).endpoint,
                         "http://elsewhere.example/v1")

    def test_a_found_config_may_still_set_the_harmless_settings(self):
        path = self.write('max_bytes = 5000\nexclude = ["*.tmp"]\n',
                          name="sift.toml")
        with mock.patch("sift.config._config_search_paths",
                        return_value=[(path, False, False)]):
            cfg = Config.load()
        self.assertEqual(cfg.max_bytes, 5000)
        self.assertEqual(cfg.exclude, ["*.tmp"])

    def test_a_sift_config_that_does_not_exist_is_an_error(self):
        """Falling through to ./sift.toml would hand the run to whatever
        happened to be in the working directory."""
        os.environ["SIFT_CONFIG"] = str(self.dir / "absent.toml")
        with self.assertRaises(ConfigError):
            Config.load()

    def test_the_working_directory_is_not_a_trusted_source(self):
        """The trust boundary itself, which no test could tell from its
        opposite: every mutation of it survived the suite."""
        from sift.config import _config_search_paths
        paths = dict((p.name, trusted) for p, trusted, _ in _config_search_paths())
        self.assertIs(paths["sift.toml"], False)
        self.assertIs(paths["config.toml"], True)

    def test_every_reach_out_setting_is_refused_from_an_untrusted_file(self):
        """One entry dropped from _TRUSTED_ONLY was invisible to the suite."""
        from sift.config import _TRUSTED_ONLY
        for key, value in (("endpoint", '"http://elsewhere/v1"'),
                           ("api_key", '"stolen"'),
                           ("provider", '"lmstudio"'),
                           ("model", '"some-model"'),
                           ("allow_model_load", "true"),
                           ("follow_symlinks", "true")):
            self.assertIn(key, _TRUSTED_ONLY)
            path = self.write(f"{key} = {value}\n", name="sift.toml")
            with mock.patch("sift.config._config_search_paths",
                            return_value=[(path, False, False)]):
                with self.assertRaises(ConfigError, msg=key) as ctx:
                    Config.load()
            self.assertIn(key, str(ctx.exception))

    def test_the_same_settings_are_honoured_from_a_trusted_file(self):
        path = self.write('endpoint = "http://elsewhere/v1"\n'
                          "allow_model_load = true\n", name="config.toml")
        with mock.patch("sift.config._config_search_paths",
                        return_value=[(path, True, False)]):
            cfg = Config.load()
        self.assertEqual(cfg.endpoint, "http://elsewhere/v1")
        self.assertTrue(cfg.allow_model_load)

    def test_a_timeout_of_zero_is_refused(self):
        """Below one second nothing can complete; the floor was untested."""
        path = self.write("timeout = 0.0\n")
        with self.assertRaises(ConfigError):
            Config.load(path)

    def test_a_relative_sift_config_that_is_missing_is_still_an_error(self):
        """Path() collapses "./" and "//", so comparing the string back
        never matched — and SIFT_CONFIG=./sift.toml is how anyone writes it."""
        for spelling in ("./missing.toml", str(self.dir) + "//missing.toml",
                         str(self.dir) + "/./missing.toml"):
            os.environ["SIFT_CONFIG"] = spelling
            with self.assertRaises(ConfigError, msg=spelling):
                Config.load()

    def test_a_relative_sift_config_that_exists_is_loaded(self):
        import os as _os
        path = self.write('provider = "ollama"\n', name="real.toml")
        cwd = _os.getcwd()
        _os.chdir(self.dir)
        self.addCleanup(_os.chdir, cwd)
        os.environ["SIFT_CONFIG"] = "./real.toml"
        self.assertEqual(Config.load().provider, "ollama")

    def test_sift_config_env_var_points_at_the_file(self):
        path = self.write('[sift]\nprovider = "ollama"\n', name="elsewhere.toml")
        os.environ["SIFT_CONFIG"] = str(path)
        self.assertEqual(Config.load().provider, "ollama")

    def test_a_bare_table_without_the_sift_header_also_works(self):
        path = self.write('provider = "ollama"\n')
        self.assertEqual(Config.load(path).provider, "ollama")


class EndpointTest(EnvGuard):
    def test_provider_defaults(self):
        self.assertEqual(Config().resolved_endpoint("ollama"),
                         "http://localhost:11434")
        self.assertEqual(Config().resolved_endpoint("lmstudio"),
                         "http://localhost:1234/v1")
        self.assertEqual(Config().resolved_endpoint("mlx-lm"),
                         "http://localhost:8080/v1")

    def test_an_explicit_endpoint_wins_for_every_provider(self):
        cfg = Config(endpoint="http://203.0.113.20:1234/v1/")
        self.assertEqual(cfg.resolved_endpoint("ollama"),
                         "http://203.0.113.20:1234/v1")
        self.assertEqual(cfg.resolved_endpoint("lmstudio"),
                         "http://203.0.113.20:1234/v1")

    def test_mlx_lm_server_address_gets_the_openai_api_path(self):
        cfg = Config(endpoint="http://203.0.113.10:8081")
        self.assertEqual(cfg.resolved_endpoint("mlx-lm"),
                         "http://203.0.113.10:8081/v1")

    def test_mlx_lm_reverse_proxy_path_is_preserved(self):
        cfg = Config(endpoint="http://model.example/mlx/openai")
        self.assertEqual(cfg.resolved_endpoint("mlx-lm"),
                         "http://model.example/mlx/openai")


class FlagsReachTheConfigTest(unittest.TestCase):
    """A flag that does nothing looks exactly like a flag that worked."""

    def test_every_flag_named_after_a_setting_arrives(self):
        """The override dict used to be hand-written beside two parsers.

        --spool-dir was added to the serve parser, documented in the README as
        the way to keep uploads off a full disk, and left out of the dict.
        `sift serve --spool-dir /mnt/bulk` exited 0 and spooled to /tmp. The
        dict is derived from the dataclass now; this test is what keeps the
        derivation honest, because the failure is silent by construction.
        """
        from dataclasses import fields
        from sift import cli
        from sift.config import Config

        settings = {f.name for f in fields(Config)}
        for build in (cli.build_parser, cli.build_serve_parser):
            parser = build()
            named = {a.dest for a in parser._actions} & settings
            self.assertTrue(named)
            for dest in sorted(named):
                action = next(a for a in parser._actions if a.dest == dest)
                if action.const is True or action.nargs == 0:
                    continue                   # store_true, checked below
                sample = {"provider": "ollama", "output_format": "json"}.get(
                    dest, "7" if dest in {"context_tokens", "max_output_tokens",
                                          "max_bytes"} else
                    "1.5" if dest in {"temperature", "timeout",
                                      "time_budget",
                                      "model_time_budget"} else "x")
                args = parser.parse_args(
                    [action.option_strings[-1], sample]
                    + ([] if build is cli.build_serve_parser else ["f.txt"]))
                # Through the real function, with only the file layer
                # neutralised: reading the override dict directly would test
                # the implementation rather than what reaches Config.
                from unittest import mock
                with mock.patch.object(Config, "load",
                                       staticmethod(lambda p=None, o=None: Config(
                                           **{k: v for k, v in (o or {}).items()}))):
                    cfg = cli._config_from_args(args)
                self.assertNotEqual(
                    getattr(cfg, dest), getattr(Config(), dest),
                    f"{action.option_strings[-1]} never reaches Config.{dest}")

    def test_a_store_true_that_was_not_passed_does_not_overrule_the_file(self):
        """False from argparse is "not mentioned", not "turn it off"."""
        from unittest import mock

        from sift import cli
        from sift.config import Config

        args = cli.build_serve_parser().parse_args([])
        with mock.patch.object(Config, "load",
                               staticmethod(lambda p=None, o=None: Config(**(o or {})))):
            overrides_applied = cli._config_from_args(args)
        self.assertFalse(overrides_applied.no_llm)
        self.assertTrue(overrides_applied.stream)

    def test_the_negations_turn_exactly_their_own_setting_off(self):
        """--no-stream, --no-vision and --no-cache are the only way to say
        False. A mutation that made them set True, or that fired without the
        flag, survived — nothing checked which setting each one moves.
        """
        from unittest import mock

        from sift import cli
        from sift.config import Config

        def resolved(argv, build=cli.build_parser):
            args = build().parse_args(argv)
            with mock.patch.object(
                    Config, "load",
                    staticmethod(lambda p=None, o=None: Config(**(o or {})))):
                return cli._config_from_args(args)

        # The serve parser has --no-stream and neither of the others, so a
        # wrong default in the lookup turns vision and the cache off for
        # every `sift serve` and says nothing.
        serve = resolved([], build=cli.build_serve_parser)
        self.assertTrue(serve.vision)
        self.assertTrue(serve.cache)
        self.assertTrue(serve.stream)
        self.assertFalse(resolved(["--no-stream"],
                                  build=cli.build_serve_parser).stream)
        for flag, setting in (("--no-stream", "stream"),
                              ("--no-vision", "vision"),
                              ("--no-cache", "cache")):
            with_flag = resolved([flag, "f.txt"])
            without = resolved(["f.txt"])
            self.assertFalse(getattr(with_flag, setting), flag)
            self.assertTrue(getattr(without, setting),
                            f"{setting} is off without {flag}")
            # And it moves nothing else.
            for other in ("stream", "vision", "cache"):
                if other != setting:
                    self.assertTrue(getattr(with_flag, other),
                                    f"{flag} also turned off {other}")

    def test_a_flag_that_was_passed_wins_over_the_file(self):
        """The boolean guard skips False, which must not mean skipping True.

        `if value is False and isinstance(default, bool)` mutated to `is
        True` drops every store_true flag instead of every absent one, and
        --recursive stops working while everything stays green.
        """
        from unittest import mock

        from sift import cli
        from sift.config import Config

        args = cli.build_parser().parse_args(["--recursive", "--no-llm", "f.txt"])
        with mock.patch.object(
                Config, "load",
                staticmethod(lambda p=None, o=None: Config(**(o or {})))):
            cfg = cli._config_from_args(args)
        self.assertTrue(cfg.recursive)
        self.assertTrue(cfg.no_llm)

    def test_serve_refuses_a_setting_it_cannot_use(self):
        """At startup, not at the first upload — which could be days later.

        `sift serve --spool-dir /typo` started happily and then answered 400
        to every upload with the reason nowhere in sight. A server is the one
        mode where the person who made the mistake is not the person who
        meets it.
        """
        from unittest import mock

        from sift import cli

        with mock.patch("sift.cli.build_client", lambda cfg: None), \
             mock.patch("sift.server.run_server",
                        lambda h, p, c, cl: 0):
            self.assertEqual(cli._serve(["--spool-dir", "/does/not/exist"]), 2)
            self.assertEqual(cli._serve(["--timeout", "0"]), 2)
            self.assertEqual(cli._serve(["--no-llm"]), 0)

    def test_a_tilde_means_the_same_to_the_check_and_to_the_server(self):
        import shutil
        import tempfile
        from pathlib import Path

        from sift.config import Config, validate

        home = Path.home() / "sift-spool-check"
        home.mkdir(exist_ok=True)
        self.addCleanup(shutil.rmtree, home, True)
        cfg = Config(spool_dir="~/sift-spool-check")
        validate(cfg)                      # must not raise
        self.assertEqual(cfg.resolved_spool_dir(), str(home))
        self.assertIsNone(Config().resolved_spool_dir())


if __name__ == "__main__":
    unittest.main()


class BothParsersOfferTheSameSettingsTest(unittest.TestCase):
    """`sift serve` was missing --max-output, --temperature, --no-vision and
    --no-cache, which the traversal parser has.

    The web UI is the interface that needs them most: it is the one where a
    user cannot pass a flag per file, and a reasoning model spends its output
    budget thinking before it writes anything. The existing test checked that
    every flag *named after a setting* reaches the config; it did not check
    that both parsers offer the same settings, so a flag simply absent from
    one of them was invisible.
    """

    #: Settings that belong to walking a directory, not to serving one file.
    WALKING_ONLY = {"recursive", "include", "exclude", "output_format",
                    "output_path"}

    def test_serve_offers_every_model_setting_the_cli_does(self):
        from dataclasses import fields
        from sift import cli
        from sift.config import Config

        settings = {f.name for f in fields(Config)}
        traversal = {a.dest for a in cli.build_parser()._actions} & settings
        serving = {a.dest for a in cli.build_serve_parser()._actions} & settings
        missing = sorted(traversal - serving - self.WALKING_ONLY)
        self.assertEqual([], missing, f"sift serve cannot set: {missing}")

    def test_the_settings_serve_has_alone_are_about_serving(self):
        from dataclasses import fields
        from sift import cli
        from sift.config import Config

        settings = {f.name for f in fields(Config)}
        traversal = {a.dest for a in cli.build_parser()._actions} & settings
        serving = {a.dest for a in cli.build_serve_parser()._actions} & settings
        self.assertEqual({"spool_dir"}, serving - traversal)
