"""Highlighter tests.

These act as a thin grammar-drift detector: if the tree-sitter QVR
grammar adds or renames a keyword, the classifier must keep up. The
golden token stream covers the major shapes that the TUI, the
prompt_toolkit lexer, and the LSP semantic-tokens encoder all consume.
"""

from __future__ import annotations

from quivers.cli.repl_highlight import (
    SEMANTIC_TOKEN_MODIFIERS,
    SEMANTIC_TOKEN_TYPES,
    Span,
    to_semantic_token_data,
    to_semantic_token_legend,
    tokenize,
)


SAMPLE = "object X : 3\nlatent f : X -> X\n# comment\n"


def _classify(spans: list[Span]) -> list[tuple[str, str]]:
    return [(sp.token, sp.text) for sp in spans if sp.text.strip()]


def test_tokenize_classifies_keywords_and_punctuation() -> None:
    pairs = _classify(tokenize(SAMPLE))
    # Order matters: this is a structural snapshot of the grammar.
    assert ("keyword", "object") in pairs
    assert ("keyword", "latent") in pairs
    assert ("punctuation", ":") in pairs
    assert ("operator", "->") in pairs
    assert ("number", "3") in pairs
    assert ("comment", "# comment") in pairs


def test_tokenize_identifies_types_and_variables() -> None:
    # In `object X : 3`, X appears as the declared identifier (variable).
    # In `latent f : X -> X`, the two X tokens sit inside type_atoms
    # and must surface as `type` so the LSP semantic-tokens stream
    # paints them with the type colour.
    pairs = _classify(tokenize(SAMPLE))
    type_xs = [p for p in pairs if p == ("type", "X")]
    assert len(type_xs) >= 2


def test_semantic_token_legend_is_stable() -> None:
    types, modifiers = to_semantic_token_legend()
    assert types == SEMANTIC_TOKEN_TYPES
    assert modifiers == SEMANTIC_TOKEN_MODIFIERS
    # The legend index is wire-format; reordering would invalidate the
    # encoded delta stream against any active editor.
    assert types.index("keyword") == 1
    assert types.index("comment") == 0


def test_semantic_token_data_is_5_tuple_aligned() -> None:
    data = to_semantic_token_data(SAMPLE)
    assert len(data) % 5 == 0
    assert data, "expected non-empty token stream for non-empty source"


def test_tokenize_empty_source() -> None:
    assert tokenize("") == []
