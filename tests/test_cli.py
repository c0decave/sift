import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from sift.cli import build_parser, build_serve_parser, collect_files, main
from sift.config import Config
from tests import _fixtures as fx


def run(args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        # Removed when the test ends. Without this the suite left one
        # directory per test method behind — 43 a run, and a machine
        # that had run it often enough held 45,000 of them.
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.csv = fx.write(self.dir / "d.csv", fx.CSV_TEXT)
        fx.write(self.dir / "note.txt", "plain prose here")

    def test_json_output(self):
        code, out, _ = run(["--no-llm", "--json", str(self.csv)])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(doc["tool"], "sift")
        self.assertEqual(doc["schema_version"], 2)
        data = doc["results"]
        self.assertEqual(doc["count"], 1)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["kind"], "csv")

    def test_text_output(self):
        code, out, _ = run(["--no-llm", str(self.csv)])
        self.assertEqual(code, 0)
        self.assertIn("csv", out)

    def test_markdown_output(self):
        code, out, _ = run(["--no-llm", "--markdown", str(self.csv)])
        self.assertEqual(code, 0)
        self.assertIn("# sift", out)

    def test_directory_recursive(self):
        code, out, _ = run(["--no-llm", "--json", "-r", str(self.dir)])
        self.assertEqual(code, 0)
        data = json.loads(out)["results"]
        self.assertGreaterEqual(len(data), 2)

    def test_include_glob(self):
        files = collect_files([str(self.dir)], Config(include=["*.csv"]))
        self.assertEqual([f.name for f in files], ["d.csv"])

    def test_list_types(self):
        code, out, _ = run(["--list-types"])
        self.assertEqual(code, 0)
        self.assertIn("csv", out)

    def test_no_args_returns_help(self):
        code, _, _ = run([])
        self.assertEqual(code, 2)

    def test_mlx_lm_is_accepted_by_both_parsers(self):
        args = build_parser().parse_args(
            ["--provider", "mlx-lm", "--endpoint",
             "http://203.0.113.10:8081", "report.pdf"])
        serve = build_serve_parser().parse_args(
            ["--provider", "mlx-lm", "--endpoint",
             "http://203.0.113.10:8081"])
        self.assertEqual(args.provider, "mlx-lm")
        self.assertEqual(serve.provider, "mlx-lm")

    def test_unknown_openai_model_state_is_not_claimed_as_loaded(self):
        from types import SimpleNamespace

        from sift.cli import _describe_pick
        from sift.llm.base import ModelInfo

        model = ModelInfo(id="mlx-model", loaded=None, kind="llm")
        described = _describe_pick(SimpleNamespace(), [model], need_vision=False)
        self.assertIn("load state not reported", described)
        self.assertNotIn("already loaded", described)

    def test_missing_file(self):
        code, _, _ = run(["--no-llm", str(self.dir / "nope.csv")])
        self.assertEqual(code, 1)

    def test_output_to_file(self):
        out_path = self.dir / "out.json"
        code, _, _ = run(["--no-llm", "--json", "-o", str(out_path), str(self.csv)])
        self.assertEqual(code, 0)
        data = json.loads(out_path.read_text())["results"]
        self.assertEqual(data[0]["kind"], "csv")


class SilentSkipTest(unittest.TestCase):
    """collect_files' own docstring: nothing is skipped quietly."""

    def test_a_walk_reports_what_it_stepped_over(self):
        """It returned in silence for every named pipe, dangling symlink and
        symlinked directory found during a walk — the branch that reports
        them covers only paths named on the command line. "2 files
        summarized" over a directory holding six entries is a false
        statement."""
        import os
        import shutil
        import tempfile
        from pathlib import Path

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "real.txt").write_text("hello there\n")
        os.mkfifo(root / "pipe.txt")
        (root / "broken.csv").symlink_to(root / "nope.csv")
        (root / "sub").mkdir()
        (root / "sub" / "subfile.txt").write_text("more\n")
        (root / "loopdir").symlink_to(root / "sub")

        code, out, err = run(["--no-llm", "-r", str(root), "-q"])
        self.assertEqual(code, 0)
        for expected in ("pipe.txt", "broken.csv", "loopdir"):
            self.assertIn(expected, err, err)
        self.assertIn("a named pipe", err)
        self.assertIn("a broken symlink", err)
        self.assertIn("follow_symlinks is off", err)


if __name__ == "__main__":
    unittest.main()
