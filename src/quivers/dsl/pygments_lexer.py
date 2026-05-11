"""Pygments lexer for the QVR domain-specific language.

The lexer wraps the in-tree tree-sitter parser (loaded via the
:mod:`quivers.dsl._dev_grammar` shim) so it always reflects the
authoritative grammar. A regex-based fallback is retained for the
case where the tree-sitter parser is unavailable (e.g. when the
shared library cannot be built) — the fallback recognises the most
common keywords and operators but does not aim for full grammar
parity.

Registers the ``qvr`` language alias so that code blocks tagged
``qvr`` in MkDocs, Sphinx, or any Pygments-based renderer get
proper syntax highlighting.

Registration via the ``[project.entry-points]`` table in
``pyproject.toml``::

    [project.entry-points."pygments.lexers"]
    qvr = "quivers.dsl.pygments_lexer:QvrLexer"
"""

from __future__ import annotations

from typing import Iterator

from pygments.lexer import RegexLexer, words
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
    _TokenType,
)


# ---------------------------------------------------------------------------
# tree-sitter node-kind → Pygments token mapping
# ---------------------------------------------------------------------------


_KEYWORD_TOKENS = {
    "quantale",
    "category",
    "rule",
    "schema",
    "object",
    "alias",
    "bundle",
    "let",
    "output",
    "where",
    "type",
    "space",
    "continuous",
    "stochastic",
    "discretize",
    "embed",
    "program",
    "draw",
    "observe",
    "return",
    "latent",
    "observed",
}

_BUILTIN_FUNCTION_TOKENS = {
    "identity",
    "fan",
    "repeat",
    "stack",
    "scan",
    "parser",
    "ccg",
    "lambek",
    "chart_fold",
    "marginalize",
    "curry_right",
    "curry_left",
    "FreeResiduated",
    "FreeMonoid",
}

_BUILTIN_TYPE_TOKENS = {
    "Euclidean",
    "Simplex",
    "PositiveReals",
    "UnitInterval",
    "ProductSpace",
}

_QUANTALE_NAMES = {
    "product_fuzzy",
    "boolean",
    "lukasiewicz",
    "godel",
    "tropical",
}

_OPERATOR_TOKENS = {
    "->",
    "=>",
    "<-",
    ">>",
    "<<",
    ">=>",
    "~",
    "@",
    "*",
    "+",
    "/",
    "\\",
    "=",
}


def _node_kind_to_pygments_token(
    kind: str, text: str, parent_kind: str | None
) -> _TokenType:
    """Map a tree-sitter node kind to a Pygments token type."""
    if kind == "doc_comment":
        return Comment.Doc
    if kind == "line_comment":
        return Comment.Single
    if kind == "integer":
        return Number.Integer
    if kind == "float":
        return Number.Float
    if kind == "signed_number":
        return Number
    if kind == "identifier":
        # Context-sensitive tagging: identifiers inside type-like
        # productions colour as types; inside constructors as
        # builtin types/functions; the default is a variable.
        if text in _BUILTIN_FUNCTION_TOKENS:
            return Name.Builtin
        if text in _BUILTIN_TYPE_TOKENS:
            return Name.Class
        if text in _QUANTALE_NAMES:
            return String.Symbol
        if parent_kind in {
            "type_atom",
            "type_effect_apply",
            "space_atom",
            "space_constructor",
            "space_constructor_bare",
        }:
            return Name.Class
        if parent_kind == "schema_parameter":
            return Name.Variable
        if parent_kind in {"enum_set_literal"}:
            return Name.Constant
        return Name.Variable
    if kind in _OPERATOR_TOKENS:
        return Operator
    if kind == "(":
        return Punctuation
    if kind in {")", "[", "]", "{", "}", ",", ":", "."}:
        return Punctuation
    if kind in _KEYWORD_TOKENS:
        return Keyword
    return Text


# ---------------------------------------------------------------------------
# tree-sitter parser singleton
# ---------------------------------------------------------------------------


_TS_PARSER = None
_TS_AVAILABLE: bool | None = None


def _try_load_parser():
    """Best-effort load of the local tree-sitter parser.

    Returns ``(parser, language)`` on success, or ``None`` when the
    shared library cannot be built / loaded; the regex fallback
    activates in that case.
    """
    global _TS_PARSER, _TS_AVAILABLE
    if _TS_AVAILABLE is False:
        return None
    if _TS_PARSER is not None:
        return _TS_PARSER
    try:
        import ctypes
        import tree_sitter
        from quivers.dsl._dev_grammar import _build_shared_lib, _grammar_dir

        gd = _grammar_dir()
        lib_path = _build_shared_lib(gd)
        lib = ctypes.CDLL(str(lib_path))
        lib.tree_sitter_qvr.restype = ctypes.c_void_p
        language = tree_sitter.Language(lib.tree_sitter_qvr())
        parser = tree_sitter.Parser(language)
        # Hold the dlopen handle so the language pointer stays valid.
        _TS_PARSER = (parser, language, lib)
        _TS_AVAILABLE = True
        return _TS_PARSER
    except Exception:  # noqa: BLE001
        _TS_AVAILABLE = False
        return None


# ---------------------------------------------------------------------------
# tree-sitter-driven token stream
# ---------------------------------------------------------------------------


def _tree_sitter_tokens(
    text: str,
) -> Iterator[tuple[int, _TokenType, str]]:
    """Yield ``(index, token_type, text)`` tuples from the tree-sitter parse.

    Walks every leaf node in source order and emits Pygments tokens
    according to :func:`_node_kind_to_pygments_token`. Whitespace
    between leaf nodes is reproduced verbatim as ``Text``.
    """
    handle = _try_load_parser()
    if handle is None:
        return
    parser, _, _ = handle
    src_bytes = text.encode("utf-8")
    tree = parser.parse(src_bytes)

    # Tree-sitter produces a tree of nodes with byte ranges. Walk
    # leaf-first in source order, emitting tokens for non-whitespace
    # leaves and reproducing the inter-leaf whitespace as Text.
    leaves: list = []

    def walk(node, parent_kind: str | None) -> None:
        if not node.children:
            leaves.append((node, parent_kind))
            return
        for c in node.children:
            walk(c, node.type)

    walk(tree.root_node, None)

    cursor = 0
    for node, parent_kind in leaves:
        start = node.start_byte
        end = node.end_byte
        if start > cursor:
            gap = src_bytes[cursor:start].decode("utf-8")
            if gap:
                yield (cursor, Text, gap)
        node_text = src_bytes[start:end].decode("utf-8")
        token = _node_kind_to_pygments_token(node.type, node_text, parent_kind)
        yield (start, token, node_text)
        cursor = end
    if cursor < len(src_bytes):
        tail = src_bytes[cursor:].decode("utf-8")
        if tail:
            yield (cursor, Text, tail)


# ---------------------------------------------------------------------------
# regex-based fallback lexer
# ---------------------------------------------------------------------------


class _QvrFallbackLexer(RegexLexer):
    """Regex-based fallback Pygments lexer.

    Activates when the tree-sitter parser is unavailable. Recognises
    the most common keywords / operators / built-ins; does not aim
    for full grammar parity — when high fidelity matters, ensure the
    tree-sitter shared library can be built.
    """

    name = "QVR (fallback)"
    aliases = ["qvr-fallback"]
    filenames: list[str] = []
    mimetypes: list[str] = []

    tokens = {
        "root": [
            (r"##.*$", Comment.Doc),
            (r"#.*$", Comment.Single),
            (
                words(tuple(sorted(_KEYWORD_TOKENS)), suffix=r"\b"),
                Keyword,
            ),
            (
                words(tuple(sorted(_QUANTALE_NAMES)), suffix=r"\b"),
                String.Symbol,
            ),
            (
                words(tuple(sorted(_BUILTIN_TYPE_TOKENS)), suffix=r"\b"),
                Name.Class,
            ),
            (
                words(tuple(sorted(_BUILTIN_FUNCTION_TOKENS)), suffix=r"\b"),
                Name.Builtin,
            ),
            (r"->|=>|<-|>=>|>>|<<|~|@|\\|/|\*|\+|=", Operator),
            (r"-?\d+\.\d+", Number.Float),
            (r"-?\d+", Number.Integer),
            (r"[a-z_]+(?==)", Name.Attribute),
            (r"[(),:.\[\]{}]", Punctuation),
            (r"[A-Z]\w*", Name.Class),
            (r"[a-z_]\w*", Name.Variable),
            (r"\s+", Text),
        ],
    }


# ---------------------------------------------------------------------------
# public Pygments lexer
# ---------------------------------------------------------------------------


class QvrLexer(RegexLexer):
    """Pygments lexer for ``.qvr`` (quivers DSL) files.

    Drives on the in-tree tree-sitter parser when the shared library
    is buildable; falls through to the regex-based
    :class:`_QvrFallbackLexer` otherwise.

    The class inherits from :class:`pygments.lexer.RegexLexer` for
    Pygments registration plumbing only; :meth:`get_tokens_unprocessed`
    is overridden to bypass the regex engine when the tree-sitter
    parser is available.
    """

    name = "QVR"
    aliases = ["qvr"]
    filenames = ["*.qvr"]
    mimetypes = ["text/x-qvr"]

    tokens = _QvrFallbackLexer.tokens

    def get_tokens_unprocessed(
        self, text: str
    ) -> Iterator[tuple[int, _TokenType, str]]:
        """Yield Pygments tokens for ``text``.

        Tries the tree-sitter–driven path first; on any failure
        (parser not loadable, parse error, etc.) falls through to
        the regex-based parent implementation.
        """
        if _try_load_parser() is not None:
            try:
                produced_any = False
                for tok in _tree_sitter_tokens(text):
                    produced_any = True
                    yield tok
                if produced_any:
                    return
            except Exception:  # noqa: BLE001
                # Any tree-sitter failure → regex fallback.
                pass
        yield from super().get_tokens_unprocessed(text)


__all__ = ["QvrLexer"]
