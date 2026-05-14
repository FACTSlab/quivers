"""Pygments lexer for the QVR domain-specific language.

The lexer drives on the in-tree tree-sitter parser (loaded via the
:mod:`quivers.dsl._dev_grammar` shim) so it always reflects the
authoritative grammar — there is no regex approximation. When the
shared library cannot be built, lexer construction raises with a
typed diagnostic so the failure is visible at the rendering site
rather than silently producing a degraded highlight.

Registers the ``qvr`` language alias so that code blocks tagged
``qvr`` in MkDocs, Sphinx, or any Pygments-based renderer get
proper syntax highlighting.

Registration via the ``[project.entry-points]`` table in
``pyproject.toml``::

    [project.entry-points."pygments.lexers"]
    qvr = "quivers.dsl.pygments_lexer:QvrLexer"
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator

import tree_sitter
from pygments.lexer import Lexer
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

from quivers.dsl._dev_grammar import _build_shared_lib, _grammar_dir


# ---------------------------------------------------------------------------
# tree-sitter node-kind → Pygments token mapping
# ---------------------------------------------------------------------------


_KEYWORD_TOKENS = {
    # module-level declaration keywords
    "quantale",
    "semigroupoid",
    "bilinear_form",
    "composition_rule",
    "contraction",
    "category",
    "rule",
    "schema",
    "object",
    "alias",
    "bundle",
    "let",
    "output",
    "export",
    "where",
    "type",
    "space",
    "kernel",
    "discretize",
    "embed",
    "program",
    "latent",
    "observed",
    # program-block step keywords
    "observe",
    "return",
    "marginalize",
    "in",
    "for",
    # effect signature keywords
    "Pure",
    "Sample",
    "Score",
    "Marginal",
    "over",
    "via",
    # contraction declaration body keywords
    "wiring",
    # deduction blocks
    "deduction",
    "atoms",
    "semiring",
    "start",
    "depth",
    "lexicon",
    "from",
    "with",
    "axioms",
    "learnable",
    "signature",
    "compressor",
    # structural-compression blocks
    "sorts",
    "constructors",
    "binders",
    "vertex_kinds",
    "edge_kinds",
    "binds",
    "dim",
    "vocab",
    "encoder",
    "decoder",
    "loss",
    "weight",
    "on",
    "of",
    "chart",
    # encoder body shapes / slots
    "iterations",
    "readout",
    "init",
    "message",
    "update",
    "var_init",
    "as",
    "recurrent",
    "attention",
    # decoder body slots
    "structure",
    "primitive",
    "factor",
    "binder_select",
    "body",
    "recursive",
    # sort-kind tokens
    "data",
    "index",
}

_BUILTIN_FUNCTION_TOKENS = {
    # morphism combinators
    "identity",
    "fan",
    "repeat",
    "stack",
    "scan",
    "parser",
    "ccg",
    "lambek",
    "chart_fold",
    "parse",
    "curry_right",
    "curry_left",
    # object constructors
    "FreeResiduated",
    "FreeMonoid",
    # let-expression builtins
    "sigmoid",
    "exp",
    "log",
    "abs",
    "softplus",
    "cumsum",
    "softmax",
    "log1p",
    "sqrt",
    "neg",
    "length",
    "map",
    "filter",
    "fold",
    "logsumexp",
    "logsumexp_over",
    "cholesky_quad_form",
}

_BUILTIN_TYPE_TOKENS = {
    "Euclidean",
    "Simplex",
    "PositiveReals",
    "UnitInterval",
    "ProductSpace",
    "CholeskyFactor",
    # program-parameter type tags
    "FinSet",
    "Real",
    "Mor",
}

_QUANTALE_NAMES = {
    "product_fuzzy",
    "boolean",
    "lukasiewicz",
    "godel",
    "tropical",
    # semiring names used by `deduction { semiring … }`
    "LogProb",
    "Boolean",
    "Viterbi",
    "Counting",
    "ProductFuzzy",
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
    # deduction-rule sequent arrow
    "|-",
    "⊢",
    # encoder / decoder body arrow
    "|->",
    # effect signature marker
    "!",
    # graph undirected-edge arrow
    "--",
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
    if kind == "string":
        return String
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
            "sort_decl",
            "constructor_decl",
            "binder_decl",
            "binder_var_decl",
            "binder_arg_decl",
            "vertex_kind_decl",
            "edge_kind_decl",
        }:
            return Name.Class
        if parent_kind == "schema_parameter":
            return Name.Variable
        if parent_kind in {"enum_set_literal"}:
            return Name.Constant
        return Name.Variable
    if kind in _OPERATOR_TOKENS:
        return Operator
    if kind in {"(", ")", "[", "]", "{", "}", ",", ":", "."}:
        return Punctuation
    if kind in _KEYWORD_TOKENS:
        return Keyword
    return Text


# ---------------------------------------------------------------------------
# tree-sitter parser singleton
# ---------------------------------------------------------------------------


_TS_PARSER: tuple[tree_sitter.Parser, tree_sitter.Language, ctypes.CDLL] | None = None


def _load_parser() -> tuple[tree_sitter.Parser, tree_sitter.Language, ctypes.CDLL]:
    """Load the in-tree tree-sitter parser. Raises on failure.

    The returned tuple keeps the ``CDLL`` handle alive alongside the
    parser so the language pointer stays valid for the parser's
    lifetime.
    """
    global _TS_PARSER
    if _TS_PARSER is not None:
        return _TS_PARSER

    gd = _grammar_dir()
    lib_path = _build_shared_lib(gd)
    lib = ctypes.CDLL(str(lib_path))
    lib.tree_sitter_qvr.restype = ctypes.c_void_p
    language_ptr = lib.tree_sitter_qvr()
    # tree-sitter 0.24+ prefers a PyCapsule over a raw integer.
    # Wrap the void* in a capsule so the language handle uses the
    # modern API and we don't emit a DeprecationWarning.
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
    PyCapsule_New.restype = ctypes.py_object
    capsule = PyCapsule_New(language_ptr, b"tree_sitter.Language", None)
    language = tree_sitter.Language(capsule)
    parser = tree_sitter.Parser(language)
    _TS_PARSER = (parser, language, lib)
    return _TS_PARSER


# ---------------------------------------------------------------------------
# public Pygments lexer
# ---------------------------------------------------------------------------


class QvrLexer(Lexer):
    """Pygments lexer for ``.qvr`` (quivers DSL) files.

    The lexer is a thin walker over the in-tree tree-sitter parse;
    the grammar is the single source of truth. There is no regex
    approximation — when the shared library can't be built, the
    lexer raises a typed exception so the failure is visible at
    the rendering site rather than silently emitting a degraded
    highlight.
    """

    name = "QVR"
    aliases = ["qvr"]
    filenames = ["*.qvr"]
    mimetypes = ["text/x-qvr"]

    def get_tokens_unprocessed(
        self, text: str
    ) -> Iterator[tuple[int, _TokenType, str]]:
        """Yield ``(index, token_type, text)`` tuples for ``text``."""
        parser, _, _ = _load_parser()
        src_bytes = text.encode("utf-8")
        tree = parser.parse(src_bytes)

        # Walk leaf-first in source order, emitting tokens for
        # each leaf and reproducing inter-leaf whitespace as Text.
        leaves: list[tuple[tree_sitter.Node, str | None]] = []

        def walk(node: tree_sitter.Node, parent_kind: str | None) -> None:
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
            token = _node_kind_to_pygments_token(
                node.type,
                node_text,
                parent_kind,
            )
            yield (start, token, node_text)
            cursor = end
        if cursor < len(src_bytes):
            tail = src_bytes[cursor:].decode("utf-8")
            if tail:
                yield (cursor, Text, tail)


__all__ = ["QvrLexer"]
