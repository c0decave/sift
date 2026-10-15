"""Categories from a fixed list, and the rules that assign them without a
model.

Summary.classification is the model's own words for what a file is — free
text, as useful to read as it is useless to filter on. A category is one
value from a list the operator keeps (Config.categories). With a model, the
model is asked for one of them and nothing else counts; without one, the
rule table (Config.category_rules) decides, first match wins.

A rule is a table:

    category  the category it assigns — one of Config.categories
    kind      optional: a detected kind, or a list of them
    metadata  optional: {key: predicate}, all of which must hold

A metadata key may be dotted to reach into a nested value — "headers.subject"
for an email's subject. A predicate is a value, which must equal the
metadata's, or a table of operators:

    equals    the value equals this
    in        the value is one of this list
    contains  this text is in the value, ignoring case — in any item, for a
              list
    exists    the value is there and not empty (true), or not (false)
    min, max  the value is a number no smaller / no larger than this

A rule with neither kind nor metadata matches everything: put it last, as
the catch-all.

After the list changes, classify() puts a file that was summarized under
the old list into the new one — from its stored summary, not the file.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

from .extract.base import scrub
from .llm import prompts
from .llm.base import ERROR_CHARS, LLMError, call_kind, recording_calls, repeatable

#: The keys a rule may have, and the operators a predicate may use.
RULE_KEYS = frozenset({"category", "kind", "metadata"})
OPERATORS = frozenset({"equals", "in", "contains", "exists", "min", "max"})
#: The longest category name. It goes into the prompt, once per call.
MAX_NAME = 60


def check_categories(categories) -> None:
    """Raise ValueError unless ``categories`` is a list of distinct names
    that can stand in the prompt's "one of: a, b, c"."""
    if not isinstance(categories, (list, tuple)):
        raise ValueError(f"categories must be a list of names, not "
                         f"{type(categories).__name__}")
    seen = set()
    for name in categories:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"a category must be a name, not {name!r}")
        if len(name) > MAX_NAME:
            raise ValueError(f"the category {name[:20]!r}… is longer than "
                             f"{MAX_NAME} characters")
        if any(ch in name for ch in ",;\n\r\t") or name != name.strip():
            raise ValueError(f"the category {name!r} may not contain a comma, "
                             f"a semicolon or a line break, or start or end "
                             f"with a blank: the prompt lists them")
        if name.casefold() in seen:
            raise ValueError(f"the category {name!r} is in the list twice")
        seen.add(name.casefold())


def check_rules(categories, rules) -> None:
    """Raise ValueError unless every rule can mean something."""
    check_categories(categories)
    if not isinstance(rules, (list, tuple)):
        raise ValueError("category_rules must be a list of tables")
    if rules and not categories:
        raise ValueError("category_rules assign categories, and categories "
                         "is empty")
    for number, rule in enumerate(rules, 1):
        where = f"category rule {number}"
        if not isinstance(rule, dict):
            raise ValueError(f"{where} must be a table, not "
                             f"{type(rule).__name__}")
        unknown = set(rule) - RULE_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown key(s) "
                             f"{', '.join(sorted(map(str, unknown)))} — a rule "
                             f"has {', '.join(sorted(RULE_KEYS))}")
        if rule.get("category") not in categories:
            raise ValueError(f"{where}: its category {rule.get('category')!r} "
                             f"is not one of the categories")
        kind = rule.get("kind")
        kinds = kind if isinstance(kind, list) else [kind]
        if kind is not None and not all(isinstance(k, str) for k in kinds):
            raise ValueError(f"{where}: kind must be a kind or a list of them")
        predicates = rule.get("metadata", {})
        if not isinstance(predicates, dict):
            raise ValueError(f"{where}: metadata must be a table of "
                             f"predicates")
        for key, predicate in predicates.items():
            _check_predicate(f"{where}, metadata {key!r}", predicate)


def _check_predicate(where: str, predicate) -> None:
    if not isinstance(predicate, dict):
        return                          # a plain value: equality
    unknown = set(predicate) - OPERATORS
    if unknown or not predicate:
        raise ValueError(f"{where}: unknown operator(s) "
                         f"{', '.join(sorted(map(str, unknown))) or 'none'} — "
                         f"the operators are {', '.join(sorted(OPERATORS))}")
    for bound in ("min", "max"):
        if bound in predicate and (isinstance(predicate[bound], bool)
                                   or not isinstance(predicate[bound],
                                                     (int, float))):
            raise ValueError(f"{where}: {bound} must be a number")
    if "in" in predicate and not isinstance(predicate["in"], list):
        raise ValueError(f"{where}: in must be a list")
    if "contains" in predicate and not isinstance(predicate["contains"], str):
        raise ValueError(f"{where}: contains must be text")
    if "exists" in predicate and not isinstance(predicate["exists"], bool):
        raise ValueError(f"{where}: exists must be true or false")


_MISSING = object()


def _lookup(metadata: dict, key: str):
    """metadata[key], or a dotted path into nested tables."""
    if key in metadata:
        return metadata[key]
    value = metadata
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _empty(value) -> bool:
    return value is _MISSING or value in (None, "", [], {})


def _holds(value, predicate) -> bool:
    if not isinstance(predicate, dict):
        return value is not _MISSING and value == predicate
    for operator, wanted in predicate.items():
        if operator == "exists":
            if (not _empty(value)) != wanted:
                return False
            continue
        if value is _MISSING:
            return False
        if operator == "equals" and value != wanted:
            return False
        if operator == "in" and value not in wanted:
            return False
        if operator == "contains":
            items = value if isinstance(value, list) else [value]
            if not any(wanted.casefold() in str(item).casefold()
                       for item in items):
                return False
        if operator in ("min", "max"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if operator == "min" and value < wanted:
                return False
            if operator == "max" and value > wanted:
                return False
    return True


def evaluate_rules(kind: str, metadata: Optional[dict], rules) -> Optional[str]:
    """The category of the first rule that matches, or None."""
    metadata = metadata if isinstance(metadata, dict) else {}
    for rule in rules or ():
        wanted = rule.get("kind")
        if wanted is not None:
            kinds = wanted if isinstance(wanted, list) else [wanted]
            if kind not in kinds:
                continue
        predicates = rule.get("metadata") or {}
        if all(_holds(_lookup(metadata, key), predicate)
               for key, predicate in predicates.items()):
            return rule.get("category")
    return None


def category_set_hash(categories, rules) -> Optional[str]:
    """A fingerprint of the list and the rules, or None when there is no
    list. A stored category is current while its hash is: renaming a
    category, reordering the list — which the prompt shows in order — or
    changing a rule makes every earlier assignment outdated."""
    if not categories and not rules:
        return None
    material = json.dumps({"categories": list(categories),
                           "rules": list(rules)},
                          sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def match_category(answer, categories) -> Optional[str]:
    """The listed category a model's answer names, or None.

    Case is ignored and so is the decoration a model adds — "**Invoice.**",
    "“invoice”" — but nothing else: "invoices" and "an invoice" are not on
    the list, and a strict category is only worth having if it is strict.
    """
    if not isinstance(answer, str):
        return None
    said = answer.strip().strip(" .!*_`\"'“”‘’()[]").casefold()
    for name in categories:
        if name.casefold() == said:
            return name
    return None


@dataclass
class CategoryResult:
    """What classify() decided, and on what authority."""
    #: One of the categories, or None.
    category: Optional[str]
    #: "reclassify" when the model's answer counted — including an answer
    #: that named no listed category, which leaves category None — and
    #: "rule" when the rule table decided: no model, a model that failed,
    #: or a summary with nothing in it to judge.
    origin: str
    #: category_set_hash() of the list and rules it was decided under.
    category_set_hash: Optional[str]
    #: Why category is None when that is not the plain answer: what the
    #: model said instead, or that there was nothing to ask about. "" else.
    note: str = ""
    #: The model's failure when the rules had to stand in for it, so a
    #: caller can try again later rather than keep the rules' answer.
    error: Optional[str] = None
    #: Every call to the model this took (llm.CallUsage), kind "reclassify".
    calls: list = field(default_factory=list)


#: Room for one line — and for the thinking a reasoning model does first;
#: a model that runs out of room thinking is asked once more with enough.
RECLASSIFY_MAX_TOKENS = 256

#: The longest answer a note repeats. The model's words, not sift's.
_SHOWN = 60


def classify(summary, categories, client=None, *, rules=()) -> CategoryResult:
    """The category for a file that was summarized before the list changed.

    ``summary`` is what Summary.to_dict() stored (or the Summary itself).
    With a ``client`` the model is asked one short question about the
    stored TYPE, SUMMARY, KEY POINTS and TAGS — never the file — and its
    answer counts only if it names a listed category. Without one, or when
    the call fails, the rules decide from the stored kind and metadata, as
    they do when a file is summarized without a model.

    Raises ValueError for arguments that cannot mean anything: a summary
    that is not a table, a list or rules check_rules() refuses.
    """
    if hasattr(summary, "to_dict"):
        summary = summary.to_dict()
    if not isinstance(summary, dict):
        raise ValueError(f"summary must be a stored summary (a dict), not "
                         f"{type(summary).__name__}")
    rules = list(rules or ())
    check_rules(categories, rules)
    digest = category_set_hash(categories, rules)
    if not categories:
        return CategoryResult(None, "rule", digest,
                              note="no categories are configured")

    def by_rules(note: str = "", error: Optional[str] = None,
                 calls: Optional[list] = None) -> CategoryResult:
        chosen = evaluate_rules(summary.get("kind") or "",
                                summary.get("metadata"), rules)
        return CategoryResult(chosen, "rule", digest, note, error,
                              list(calls or []))

    if client is None:
        return by_rules()
    # Scrubbed once more on the way out. What to_dict() stored is redacted
    # already; a summary stored by another version, or edited by hand, is
    # not something this function wrote. tags_suspect stays out: the file
    # may have asked for those. A tag the redaction would change is dropped,
    # not masked, as at summary time: a mask says nothing about a file.
    fields = scrub({"classification": _text(summary.get("classification")),
                    "summary": _text(summary.get("summary")),
                    "key_points": _texts(summary.get("key_points"))})
    fields["tags"] = [tag for tag in _texts(summary.get("tags"))
                      if scrub(tag, "tags") == tag]
    if not any(fields.values()):
        return by_rules(note="the summary holds nothing to judge by — the "
                             "rules decided")
    system, user = prompts.build_reclassify_prompt(categories=categories,
                                                   **fields)
    with recording_calls() as calls, call_kind("reclassify"):
        try:
            result = client.generate(system, user, temperature=0.0,
                                     max_tokens=RECLASSIFY_MAX_TOKENS)
        except LLMError as exc:
            return by_rules(error=repeatable(str(exc), ERROR_CHARS),
                            calls=calls)
        except Exception as exc:  # noqa: BLE001
            # The net summarize_path has under the same clients: a body of
            # a shape nobody has seen raised AttributeError in the stream
            # parsers. A reclassify job over thousands of files must not
            # end on one of them.
            return by_rules(error=repeatable(f"{type(exc).__name__} from the "
                                             f"model server: {exc}",
                                             ERROR_CHARS), calls=calls)
    return _judged(prompts.category_answers(result.text), categories,
                   digest, list(calls))


def _judged(answers: list, categories, digest, calls) -> CategoryResult:
    named = []
    for answer in answers:
        chosen = match_category(answer, categories)
        if chosen is not None and chosen not in named:
            named.append(chosen)
    if len(named) == 1:
        return CategoryResult(named[0], "reclassify", digest, calls=calls)
    if named:
        note = (f"the model gave more than one category "
                f"({', '.join(named)}) — none was taken")
    elif not any(answer.strip() for answer in answers):
        note = "the model gave no category from the list"
    else:
        said = " ".join(answers[-1].split())
        shown = said if len(said) <= _SHOWN else said[:_SHOWN - 1] + "…"
        note = (f"the model answered the category “{shown}”, which is not "
                f"one of the configured categories")
    return CategoryResult(None, "reclassify", digest, note, calls=calls)


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _texts(value) -> list:
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value
            if isinstance(item, str) and item.strip()]
