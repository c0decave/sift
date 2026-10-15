"""The package has to build with the tools people actually have, and load
on the Python they actually run.

`license = { text = "MIT" }` is the table form from before PEP 639. Beside
`license-files` — which is PEP 639 — current setuptools refuses it outright:
building a wheel with setuptools 84.0.0 failed in metadata generation with
"project.license must be string", so `pip install .` did not work at all.
The license expression that replaces it has its own two conditions: a
setuptools that reads it (77 is the first), and no `License ::` classifier
beside it, which setuptools rejects as superseded.

Checked by reading pyproject.toml rather than by building, because the
suite runs on the standard library and offline.
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

#: The first setuptools that reads a PEP 639 license expression.
PEP639_SETUPTOOLS = (77,)

_REQUIREMENT = re.compile(r"^\s*setuptools\s*(?P<spec>[^;]*)")
_LOWER = re.compile(r"(?:>=|~=|==)\s*(?P<version>\d+(?:\.\d+)*)")


def setuptools_floor(requirement: str):
    """The lowest setuptools a requirement string admits, or None.

    None both for a string about another package and for one with no lower
    bound at all — either way nothing guarantees a new enough setuptools.
    """
    match = _REQUIREMENT.match(requirement)
    if not match:
        return None
    floors = [tuple(int(part) for part in found.group("version").split("."))
              for found in _LOWER.finditer(match.group("spec"))]
    return max(floors) if floors else None


def packaging_problems(data: dict) -> list:
    """What stops this pyproject from building with current setuptools."""
    problems = []
    project = data.get("project", {})
    license_value = project.get("license")
    if not isinstance(license_value, str) or not license_value.strip():
        problems.append(f"project.license is {license_value!r}, not an SPDX "
                        f"expression string")
    requires = data.get("build-system", {}).get("requires", [])
    floors = [f for f in map(setuptools_floor, requires) if f is not None]
    if not floors or max(floors) < PEP639_SETUPTOOLS:
        problems.append(f"build-system.requires {requires!r} admits a "
                        f"setuptools older than "
                        f"{'.'.join(map(str, PEP639_SETUPTOOLS))}")
    if isinstance(license_value, str):
        classifiers = [c for c in project.get("classifiers", [])
                       if c.startswith("License ::")]
        if classifiers:
            problems.append(f"license classifiers beside a license "
                            f"expression: {classifiers}")
    return problems


class RequirementFloorTest(unittest.TestCase):

    def test_a_plain_lower_bound(self):
        self.assertEqual(setuptools_floor("setuptools>=77"), (77,))
        self.assertEqual(setuptools_floor("setuptools >= 77.0.3"), (77, 0, 3))

    def test_no_bound_or_another_package_admits_anything(self):
        self.assertIsNone(setuptools_floor("setuptools"))
        self.assertIsNone(setuptools_floor("wheel>=0.40"))
        self.assertIsNone(setuptools_floor("setuptools<90"))

    def test_the_highest_lower_bound_counts_and_markers_do_not(self):
        self.assertEqual(setuptools_floor("setuptools>=61,<90"), (61,))
        self.assertEqual(setuptools_floor("setuptools~=77.0"), (77, 0))
        self.assertEqual(
            setuptools_floor('setuptools>=77; python_version >= "3.99"'),
            (77,))


@unittest.skipIf(sys.version_info < (3, 11), "tomllib arrived in Python 3.11")
class PyprojectBuildsTest(unittest.TestCase):

    def test_the_real_pyproject_has_no_packaging_problem(self):
        import tomllib
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        self.assertEqual([], packaging_problems(data))
        self.assertEqual("MIT", data["project"]["license"])
        self.assertEqual(["LICENSE"], data["project"]["license-files"])

    def test_the_old_form_is_caught_on_every_count(self):
        """The shape master an earlier version had, which setuptools 84 refused."""
        import tomllib
        old = tomllib.loads(
            '[build-system]\nrequires = ["setuptools>=61"]\n'
            '[project]\nlicense = { text = "MIT" }\n'
            'license-files = ["LICENSE"]\n'
            'classifiers = ["License :: OSI Approved :: MIT License"]\n')
        problems = packaging_problems(old)
        self.assertEqual(2, len(problems), problems)
        self.assertIn("not an SPDX expression", problems[0])
        self.assertIn("older than 77", problems[1])

    def test_an_expression_with_a_classifier_is_still_refused(self):
        import tomllib
        mixed = tomllib.loads(
            '[build-system]\nrequires = ["setuptools>=77"]\n'
            '[project]\nlicense = "MIT"\n'
            'classifiers = ["License :: OSI Approved :: MIT License",\n'
            '               "Topic :: Utilities"]\n')
        problems = packaging_problems(mixed)
        self.assertEqual(1, len(problems), problems)
        self.assertIn("License :: OSI Approved :: MIT License", problems[0])

    def test_an_empty_expression_or_no_setuptools_bound_is_refused(self):
        self.assertEqual(2, len(packaging_problems(
            {"build-system": {"requires": ["setuptools"]},
             "project": {"license": " "}})))
        self.assertEqual(2, len(packaging_problems({})))


#: What the compiler warns with about "\\i": a SyntaxWarning from Python
#: 3.12, a DeprecationWarning from 3.6 to 3.11 (the language reference,
#: "Changed in version 3.6" and "Changed in version 3.12"). Looking for the
#: 3.12 category alone found nothing on 3.9 to 3.11, and the negative control
#: failed there with 1 != 0.
ESCAPE_WARNING = (SyntaxWarning if sys.version_info >= (3, 12)
                  else DeprecationWarning)
#: Part of the message on every version: 3.11 to 3.13 open with it, 3.14
#: writes '"\\i" is an invalid escape sequence. Such sequences …'.
_ESCAPE_MESSAGE = "invalid escape sequence"
#: The -W filter that turns it into an error. By category alone where the
#: category is SyntaxWarning; by message too where it is DeprecationWarning,
#: so an unrelated deprecation in the import chain cannot fail the check.
ESCAPE_AS_ERROR = ("error::SyntaxWarning" if ESCAPE_WARNING is SyntaxWarning
                   else f"error:{_ESCAPE_MESSAGE}:DeprecationWarning")


def invalid_escapes(source: str, name: str) -> list:
    """Every invalid-escape warning compiling ``source`` raises, as
    "name:line: msg", under either category a supported Python uses.

    compile() on the text, never an import: an import of a module whose .pyc
    is already cached compiles nothing and warns about nothing. Recorded, not
    raised, because raising stops at the first one in a file and hid a second
    one in test_regressions.py.
    """
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile(source, name, "exec")
    return [f"{name}:{w.lineno}: {w.message}" for w in caught
            if issubclass(w.category, (SyntaxWarning, DeprecationWarning))
            and _ESCAPE_MESSAGE in str(w.message)]


class SourceCompilesCleanlyTest(unittest.TestCase):
    """"\\i" in a docstring is an invalid escape. Python 3.12 made that a
    SyntaxWarning, printed on every fresh compile, and the documentation
    says a future version makes it a SyntaxError. `python3 -m sift --list-types` printed one for rtf.py."""

    def test_the_rtf_module_imports_with_syntax_warnings_as_errors(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as fresh:
            # An empty bytecode cache, so the import has to compile.
            run = subprocess.run(
                [sys.executable, "-X", f"pycache_prefix={fresh}",
                 "-W", ESCAPE_AS_ERROR, "-c", "import sift.extract.rtf"],
                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(0, run.returncode, run.stderr[-600:])

    def test_no_source_file_has_an_invalid_escape(self):
        files = sorted((ROOT / "sift").rglob("*.py")) + sorted(
            (ROOT / "tests").glob("*.py"))
        # A scan that opened nothing would pass; the package alone has 30+.
        self.assertGreater(len(files), 40, [str(f) for f in files])
        found = [problem for path in files
                 for problem in invalid_escapes(
                     path.read_text(encoding="utf-8"),
                     str(path.relative_to(ROOT)))]
        self.assertEqual([], found)

    def test_the_check_sees_an_invalid_escape(self):
        """The negative control: the same docstring, not raw."""
        found = invalid_escapes('def f():\n    """The \\info group."""\n',
                                "planted.py")
        self.assertEqual(1, len(found), found)
        self.assertIn("planted.py:2", found[0])
        self.assertEqual([], invalid_escapes(
            'def f():\n    r"""The \\info group."""\n', "raw.py"))

    def test_the_check_uses_the_category_this_python_warns_with(self):
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile('x = "\\i"\n', "planted.py", "exec")
        categories = {w.category for w in caught
                      if _ESCAPE_MESSAGE in str(w.message)}
        self.assertEqual({ESCAPE_WARNING}, categories)

    def test_the_import_check_fails_on_a_planted_escape(self):
        """The subprocess check's own negative control: the same flags on a
        module that has the escape must fail."""
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as fresh:
            Path(fresh, "planted_escape.py").write_text(
                'X = "\\info"\n', encoding="utf-8")
            run = subprocess.run(
                [sys.executable, "-X", f"pycache_prefix={fresh}",
                 "-W", ESCAPE_AS_ERROR, "-c", "import planted_escape"],
                cwd=fresh, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(0, run.returncode, run.stderr[-600:])

    def test_the_docstring_still_says_what_it_said(self):
        """A raw string keeps the backslash the invalid escape kept."""
        from sift.extract import rtf
        self.assertEqual("The \\info group's fields, read on their own.",
                         rtf._document_properties.__doc__)


if __name__ == "__main__":
    unittest.main()
