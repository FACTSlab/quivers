"""Programmatic derivation of keyword / builtin sets from the QVR grammar.

The QVR grammar at ``grammars/qvr/src/grammar.json`` is the single source
of truth for surface syntax. Highlighters, REPL completers, and the LSP
all consume the sets derived here so a grammar edit propagates to every
downstream surface without hand-maintained mirrors.

The walker traverses each rule's ``CHOICE`` / ``SEQ`` / ``FIELD`` /
``REPEAT`` / ``PREC`` / ``ALIAS`` / ``IMMEDIATE_TOKEN`` / ``TOKEN``
structure, collects every ``STRING`` terminal, and partitions the
collected literals by name shape and by the rule that contains them:

* literals appearing as the ``constructor`` field value of
  :grammar:`discrete_constructor` or :grammar:`continuous_constructor`
  are object/type builtins.
* literals appearing as a member of :grammar:`composition_level`,
  :grammar:`_param_kind`, :grammar:`object_kind`, :grammar:`scalar_kind`,
  or :grammar:`sort_kind` are keyword modifiers (kept under
  ``KEYWORDS`` for the unified colouring).
* literals of pure ASCII word shape (``[A-Za-z_][A-Za-z0-9_]*``) that
  are not numeric / operator are keywords.
* literals containing any non-word character are operators or
  punctuation.

The walker is pure: it depends only on the JSON file that accompanies
the grammar, so it works at import time even when the shared library
has not been compiled yet.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path


_WORD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PUNCTUATION_CHARS = frozenset("(){}[],.;")
_OPERATOR_CHARS = frozenset(r"+-*/\=<>~|@:!.&%?")
_OPERATOR_EXTRAS = frozenset(
    {
        "->",
        "<-",
        "=>",
        ">>",
        "<<",
        ">>>",
        ">=>",
        "|-",
        "|->",
        "⊢",
        "--",
        "*>",
        "+>",
        "$>",
        "%>",
        "&&>",
        "||>",
        "~>",
        "?>",
    }
)


def _grammar_json_path() -> Path:
    # Prefer the package-local copy under
    # ``quivers/dsl/_grammar_data/grammar.json`` (bundled with the
    # wheel). Fall back to the canonical in-tree path
    # ``grammars/qvr/src/grammar.json`` when running from a source
    # checkout that hasn't synced the package copy yet.
    here = Path(__file__).resolve()
    packaged = here.parent / "_grammar_data" / "grammar.json"
    if packaged.is_file():
        return packaged
    for parent in here.parents:
        candidate = parent / "grammars" / "qvr" / "src" / "grammar.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "grammar.json not found at "
        f"{packaged} or under any parent of {here}; "
        "programmatic keyword derivation requires the grammar JSON "
        "bundled with the quivers package."
    )


def _walk_strings(
    node: object, *, in_field: str | None, in_rule: str
) -> Iterator[tuple[str, str | None, str]]:
    """Yield every ``(literal, field_name, owning_rule)`` triple in ``node``.

    ``in_field`` carries the nearest enclosing ``FIELD`` name so callers
    can tell which production a literal belongs to (e.g. the ``FinSet``
    string in :grammar:`discrete_constructor` sits under
    ``field('constructor', ...)``). ``in_rule`` is the top-level rule
    name the walker entered.
    """
    if not isinstance(node, dict):
        return
    kind = node.get("type")
    if kind == "STRING":
        value = node.get("value")
        if isinstance(value, str):
            yield (value, in_field, in_rule)
        return
    if kind == "FIELD":
        name = node.get("name") if isinstance(node.get("name"), str) else None
        content = node.get("content")
        yield from _walk_strings(content, in_field=name, in_rule=in_rule)
        return
    if kind in {"SEQ", "CHOICE"}:
        for m in node.get("members", []):
            yield from _walk_strings(m, in_field=in_field, in_rule=in_rule)
        return
    if kind in {
        "REPEAT",
        "REPEAT1",
        "PREC",
        "PREC_LEFT",
        "PREC_RIGHT",
        "PREC_DYNAMIC",
        "TOKEN",
        "IMMEDIATE_TOKEN",
        "ALIAS",
    }:
        yield from _walk_strings(
            node.get("content"), in_field=in_field, in_rule=in_rule
        )
        return


def _load() -> dict[str, object]:
    raw = _grammar_json_path().read_text(encoding="utf-8")
    return json.loads(raw)


def _collect() -> dict[str, frozenset[str]]:
    """Return the partitioned literal sets derived from the grammar."""
    grammar = _load()
    rules = grammar.get("rules", {})
    if not isinstance(rules, dict):
        raise ValueError("grammar.json has no `rules` table")

    keywords: set[str] = set()
    builtin_types: set[str] = set()
    operators: set[str] = set()
    composition_levels: set[str] = set()
    param_kinds: set[str] = set()
    sort_kinds: set[str] = set()

    constructor_owners = {"discrete_constructor", "continuous_constructor"}

    for rule_name, body in rules.items():
        for literal, field_name, _owner in _walk_strings(
            body, in_field=None, in_rule=rule_name
        ):
            if (
                rule_name in constructor_owners
                and field_name == "constructor"
                and _WORD_RE.match(literal)
            ):
                builtin_types.add(literal)
                continue
            if rule_name == "composition_level" and _WORD_RE.match(literal):
                composition_levels.add(literal)
                continue
            if rule_name in {"object_kind", "scalar_kind"} and _WORD_RE.match(literal):
                param_kinds.add(literal)
                continue
            if rule_name == "sort_kind" and _WORD_RE.match(literal):
                sort_kinds.add(literal)
                continue
            if rule_name == "morphism_kind" and _WORD_RE.match(literal):
                param_kinds.add(literal)
                continue
            if not _WORD_RE.match(literal):
                if literal in _OPERATOR_EXTRAS:
                    operators.add(literal)
                elif len(literal) >= 1 and all(c in _OPERATOR_CHARS for c in literal):
                    operators.add(literal)
                continue
            keywords.add(literal)

    keywords |= composition_levels
    keywords |= sort_kinds

    builtin_functions = _functional_builtins_from(rules)
    # Combinator / intrinsic names parse as keyword strings at the head of
    # their expression rule, but they are not surface keywords: colour them
    # as builtin functions, not keywords.
    keywords -= builtin_functions

    return {
        "keywords": frozenset(keywords),
        "builtin_types": frozenset(builtin_types | param_kinds),
        "operators": frozenset(operators),
        "composition_levels": frozenset(composition_levels),
        "sort_kinds": frozenset(sort_kinds),
        "builtin_functions": frozenset(builtin_functions),
    }


def _functional_builtins_from(rules: dict[str, object]) -> set[str]:
    seen: set[str] = set()
    for rule_name, body in rules.items():
        if not rule_name.endswith("_expr"):
            continue
        for literal, _field, _owner in _walk_strings(
            body, in_field=None, in_rule=rule_name
        ):
            if _WORD_RE.match(literal):
                seen.add(literal)
                break
    return seen


_SETS: dict[str, frozenset[str]] = _collect()

KEYWORDS: frozenset[str] = _SETS["keywords"]
BUILTIN_TYPES: frozenset[str] = _SETS["builtin_types"]
OPERATORS: frozenset[str] = _SETS["operators"]
COMPOSITION_LEVELS: frozenset[str] = _SETS["composition_levels"]
SORT_KINDS: frozenset[str] = _SETS["sort_kinds"]


BUILTIN_FUNCTIONS: frozenset[str] = _SETS["builtin_functions"]
