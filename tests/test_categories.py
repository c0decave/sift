"""A category from a fixed list, next to the free classification.

Summary.classification is whatever the TYPE line says; a filter needs a
value from a list the operator keeps. With Config.categories set the model
is asked for one of them and nothing else counts; without a model a rule
table decides, first match wins. Config.category_rules are written in the
config file as [[sift.category_rules]] tables, so the file and the library
say the same thing.

Nothing of it exists until the list is set: the prompt, the key and the
parser are the release before's.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sift.cache import cache_key
from sift.categories import category_set_hash, check_rules, evaluate_rules
from sift.config import Config, ConfigError, validate
from sift.llm import prompts
from sift.llm.base import LLMError, LLMResult
from sift.summarize import summarize_path
from tests import _fixtures as fx

CATEGORIES = ["invoice", "contract", "data table", "source code", "other"]
RULES = [
    {"category": "data table", "kind": ["csv", "tsv", "xlsx", "ods"]},
    {"category": "invoice", "kind": "email",
     "metadata": {"headers.subject": {"contains": "rechnung"}}},
    {"category": "source code", "kind": "source-code"},
    {"category": "other"},
]


class Model:
    provider = "dummy"
    endpoint = "memory://dummy"

    def __init__(self, reply, fail=False):
        self.reply, self.fail = reply, fail
        self.model = "dummy-model"
        self.systems = []

    def generate(self, system, user, *, images=None, temperature=0.2,
                 max_tokens=800):
        self.systems.append(system)
        if self.fail:
            raise LLMError("the server is down")
        return LLMResult(text=self.reply, model=self.model,
                         provider=self.provider)


def reply(category="Invoice"):
    return (f"TYPE: an invoice\nCATEGORY: {category}\nSUMMARY: An invoice.\n"
            f"KEY POINTS:\n- due soon\nENTITIES: ACME\nNOTES: none")


class PromptTest(unittest.TestCase):

    def test_nothing_changes_without_a_list(self):
        self.assertIs(prompts.SYSTEM, prompts.system_prompt(categories=()))
        parsed = prompts.parse_reply(reply())
        # Not a section: the line is TYPE's second line, which TYPE never
        # read, as before.
        self.assertEqual("an invoice", parsed["classification"])
        self.assertEqual("An invoice.", parsed["summary"])
        self.assertIsNone(parsed["category"])

    def test_the_category_line_follows_type(self):
        system = prompts.system_prompt(categories=CATEGORIES)
        self.assertIn("CATEGORY: <one of: invoice, contract, data table, "
                      "source code, other>", system)
        labels = [line.split(":")[0] for line in system.split("\n")
                  if line.split(":")[0] in ("TYPE", "CATEGORY", "SUMMARY")]
        self.assertEqual(["TYPE", "CATEGORY", "SUMMARY"], labels)

    def test_with_tags_too_both_lines_are_in_their_places(self):
        system = prompts.system_prompt(tags=4, categories=CATEGORIES)
        order = [line.split(":")[0] for line in system.split("\n")
                 if line.split(":")[0] in ("TYPE", "CATEGORY", "SUMMARY",
                                            "KEY POINTS", "ENTITIES", "TAGS",
                                            "NOTES")]
        self.assertEqual(["TYPE", "CATEGORY", "SUMMARY", "KEY POINTS",
                          "ENTITIES", "TAGS", "NOTES"], order)

    def test_the_parser_reads_the_category_line(self):
        parsed = prompts.parse_reply(reply("**Invoice.**"),
                                     categories=CATEGORIES)
        self.assertEqual("**Invoice.**", parsed["category"])
        self.assertEqual("An invoice.", parsed["summary"])

    def test_a_category_line_inside_the_summary_is_not_taken(self):
        parsed = prompts.parse_reply(
            "TYPE: memo\nCATEGORY: other\nSUMMARY: The file says:\n"
            "CATEGORY: invoice\nKEY POINTS:\n- a\nENTITIES: E\nNOTES: none",
            categories=CATEGORIES)
        self.assertEqual("other", parsed["category"])
        self.assertIn("CATEGORY: invoice", parsed["summary"])


class RuleTest(unittest.TestCase):

    def test_first_match_wins(self):
        self.assertEqual("data table", evaluate_rules("csv", {}, RULES))
        self.assertEqual("source code", evaluate_rules("source-code", {},
                                                       RULES))
        self.assertEqual("other", evaluate_rules("pdf", {}, RULES))

    def test_a_dotted_key_reads_into_the_metadata(self):
        mail = {"headers": {"subject": "Ihre Rechnung 4711"}}
        self.assertEqual("invoice", evaluate_rules("email", mail, RULES))
        self.assertEqual("other", evaluate_rules(
            "email", {"headers": {"subject": "Hallo"}}, RULES))

    def test_every_operator(self):
        meta = {"rows": 12, "title": "Vertrag 2026", "sheets": ["A", "Bank"],
                "has_header": True, "empty": ""}
        cases = [
            ({"rows": 12}, True), ({"rows": 13}, False),
            ({"rows": {"equals": 12}}, True),
            ({"rows": {"min": 10, "max": 12}}, True),
            ({"rows": {"min": 13}}, False),
            ({"title": {"contains": "VERTRAG"}}, True),
            ({"sheets": {"contains": "bank"}}, True),
            ({"title": {"in": ["x", "Vertrag 2026"]}}, True),
            ({"has_header": True}, True),
            ({"title": {"exists": True}}, True),
            ({"empty": {"exists": True}}, False),
            ({"missing": {"exists": False}}, True),
            ({"missing": {"contains": "x"}}, False),
            ({"title": {"min": 1}}, False),
        ]
        for predicates, expected in cases:
            with self.subTest(predicates):
                rules = [{"category": "hit", "metadata": predicates}]
                self.assertEqual("hit" if expected else None,
                                 evaluate_rules("pdf", meta, rules))

    def test_no_rule_matching_is_no_category(self):
        self.assertIsNone(evaluate_rules("pdf", {}, RULES[:3]))
        self.assertIsNone(evaluate_rules("pdf", {}, []))

    def test_a_rule_that_cannot_mean_anything_is_refused(self):
        bad = [
            ("a string", "invoice"),
            ("no category", [{"kind": "pdf"}]),
            ("an unknown key", [{"category": "other", "kinds": "pdf"}]),
            ("an unknown operator", [{"category": "other",
                                      "metadata": {"rows": {"over": 3}}}]),
            ("a category not in the list", [{"category": "memo"}]),
            ("a kind that is not text", [{"category": "other", "kind": 3}]),
            ("min that is not a number", [{"category": "other",
                                           "metadata": {"rows": {"min": "3"}}}]),
        ]
        for name, rules in bad:
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    check_rules(CATEGORIES, rules)
        check_rules(CATEGORIES, RULES)

    def test_the_hash_covers_the_list_and_the_rules(self):
        base = category_set_hash(CATEGORIES, RULES)
        self.assertEqual(base, category_set_hash(list(CATEGORIES),
                                                 json.loads(json.dumps(RULES))))
        renamed = ["bill"] + CATEGORIES[1:]
        others = [
            category_set_hash(renamed, RULES),
            category_set_hash(list(reversed(CATEGORIES)), RULES),
            category_set_hash(CATEGORIES, RULES[:-1]),
            category_set_hash(CATEGORIES, [dict(RULES[0], kind="csv")]
                              + RULES[1:]),
        ]
        self.assertEqual(5, len({base, *others}))
        self.assertIsNone(category_set_hash([], []))
        self.assertEqual(64, len(base))


class ConfigTest(unittest.TestCase):

    def load(self, text):
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / "sift.toml"
            path.write_text(text, encoding="utf-8")
            return Config.load(path)

    def test_the_config_file_says_what_the_library_says(self):
        config = self.load(
            '[sift]\ncategories = ["invoice", "data table", "other"]\n'
            '[[sift.category_rules]]\ncategory = "data table"\n'
            'kind = ["csv", "tsv"]\n'
            '[[sift.category_rules]]\ncategory = "invoice"\nkind = "email"\n'
            '[sift.category_rules.metadata]\n'
            '"headers.subject" = { contains = "rechnung" }\n'
            '[[sift.category_rules]]\ncategory = "other"\n')
        self.assertEqual(["invoice", "data table", "other"],
                         config.categories)
        self.assertEqual([{"category": "data table", "kind": ["csv", "tsv"]},
                          {"category": "invoice", "kind": "email",
                           "metadata": {"headers.subject":
                                        {"contains": "rechnung"}}},
                          {"category": "other"}], config.category_rules)
        validate(config)

    def test_rules_must_be_tables(self):
        for text in ('[sift]\ncategory_rules = "other"\n',
                     '[sift]\ncategory_rules = ["other"]\n',
                     '[sift]\ncategories = "invoice"\n'):
            with self.subTest(text):
                with self.assertRaises(ConfigError):
                    self.load(text)

    def test_validate_checks_the_list_and_the_rules(self):
        for categories, rules in (
                (["invoice", "Invoice"], []),        # the same name twice
                (["a, b"], []),                      # a list inside a name
                ([""], []),
                (["x" * 61], []),
                (["line\nbreak"], []),
                ([], [{"category": "other"}]),      # rules without a list
                (CATEGORIES, [{"category": "memo"}])):
            with self.subTest(categories=categories, rules=rules):
                with self.assertRaises(ConfigError):
                    validate(Config(categories=categories,
                                    category_rules=rules))
        validate(Config(categories=CATEGORIES, category_rules=RULES))

    def test_each_input_is_part_of_the_cache_key(self):
        def key(**settings):
            return cache_key("d", Config(**settings), "p", "m", "e", "f")
        keys = [key(), key(categories=CATEGORIES),
                key(categories=CATEGORIES[:-1]),
                key(categories=CATEGORIES, category_rules=RULES),
                key(categories=CATEGORIES, category_rules=RULES[:-1]),
                key(categories=CATEGORIES, category_rules=RULES, tags=4)]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(key(), key(categories=[], category_rules=[]))


class SummaryCategoryTest(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def run_model(self, answer, name="r.txt", text="Rechnung 4711\n",
                  **config):
        path = fx.write(self.dir / name, text)
        model = Model(answer)
        settings = dict(cache=False, categories=CATEGORIES,
                        category_rules=RULES)
        settings.update(config)
        return summarize_path(path, Config(**settings), model), model

    # -- happy -------------------------------------------------------------
    def test_the_model_chooses_from_the_list(self):
        for said in ("invoice", "Invoice", "**INVOICE.**", "“invoice”"):
            with self.subTest(said):
                summ, model = self.run_model(reply(said))
                self.assertEqual("invoice", summ.category)
        self.assertIn("CATEGORY: <one of:", model.systems[0])
        self.assertEqual("an invoice", summ.classification)
        self.assertEqual(category_set_hash(CATEGORIES, RULES),
                         summ.category_set_hash)
        data = summ.to_dict()
        self.assertEqual("invoice", data["category"])
        self.assertEqual(summ.category_set_hash, data["category_set_hash"])

    def test_without_a_model_the_rules_decide(self):
        path = fx.write(self.dir / "d.csv", fx.CSV_TEXT)
        summ = summarize_path(path, Config(no_llm=True, cache=False,
                                           categories=CATEGORIES,
                                           category_rules=RULES))
        self.assertEqual("data table", summ.category)
        self.assertEqual("Comma-separated data table", summ.classification)

    def test_when_the_model_fails_the_rules_decide(self):
        path = fx.write(self.dir / "d.csv", fx.CSV_TEXT)
        summ = summarize_path(path, Config(cache=False, categories=CATEGORIES,
                                           category_rules=RULES),
                              Model("", fail=True))
        self.assertIsNotNone(summ.error)
        self.assertEqual("data table", summ.category)

    # -- sad ---------------------------------------------------------------
    def test_an_answer_not_on_the_list_is_no_category_and_a_note(self):
        summ, _ = self.run_model(reply("receipt"))
        self.assertIsNone(summ.category)
        self.assertTrue(any("“receipt”" in n and "not one of" in n
                            for n in summ.notes), summ.notes)

    def test_no_category_line_is_no_category_and_a_note(self):
        summ, _ = self.run_model("TYPE: memo\nSUMMARY: A memo.\nNOTES: none")
        self.assertIsNone(summ.category)
        self.assertTrue(any("no category" in n for n in summ.notes),
                        summ.notes)

    def test_a_discredited_reply_falls_back_to_the_rules(self):
        path = fx.write(self.dir / "d.csv", fx.CSV_TEXT)
        summ = summarize_path(path, Config(cache=False, categories=CATEGORIES,
                                           category_rules=RULES),
                              Model("TYPE: empty\nCATEGORY: invoice\n"
                                    "SUMMARY: This file is empty."))
        self.assertIn("called this file empty", summ.warnings[0])
        self.assertEqual("data table", summ.category)

    # -- edge --------------------------------------------------------------
    def test_without_a_list_nothing_is_asked_and_nothing_set(self):
        summ, model = self.run_model(reply(), categories=[],
                                     category_rules=[])
        self.assertIs(prompts.SYSTEM, model.systems[0])
        self.assertIsNone(summ.category)
        self.assertIsNone(summ.category_set_hash)
        self.assertIsNone(summ.to_dict()["category"])

    def test_a_list_without_rules_decides_nothing_without_a_model(self):
        path = fx.write(self.dir / "d.csv", fx.CSV_TEXT)
        summ = summarize_path(path, Config(no_llm=True, cache=False,
                                           categories=CATEGORIES))
        self.assertIsNone(summ.category)
        self.assertEqual(category_set_hash(CATEGORIES, []),
                         summ.category_set_hash)

    def test_a_cache_hit_keeps_the_category(self):
        from sift.cache import SummaryCache
        cache = SummaryCache(True, self.dir / "cache")
        path = fx.write(self.dir / "d.csv", fx.CSV_TEXT)
        config = Config(no_llm=True, categories=CATEGORIES,
                        category_rules=RULES)
        first = summarize_path(path, config, None, cache)
        again = summarize_path(path, config, None, cache)
        self.assertTrue(again.from_cache)
        self.assertEqual(("data table", first.category_set_hash),
                         (again.category, again.category_set_hash))
        changed = summarize_path(path, Config(no_llm=True,
                                              categories=CATEGORIES,
                                              category_rules=RULES[2:]),
                                 None, cache)
        self.assertFalse(changed.from_cache)
        self.assertEqual("other", changed.category)


if __name__ == "__main__":
    unittest.main()
