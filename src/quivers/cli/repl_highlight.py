"""Live syntax highlighting for QVR source, shared across all frontends.

The pipeline:

1. Re-parse the buffer via panproto's cached :class:`ParseEmitLens`
   (sub-millisecond on edits after warm-up).
2. Walk the resulting schema's leaf vertices in source order.
3. Map each ``(vertex.kind, parent.kind, literal_text)`` triple to a
   semantic token type, via a single :data:`STYLE_TABLE`.
4. Emit a list of :class:`Span` records carrying ``(byte_start,
   byte_end, token, text)``.

Frontends consume :class:`Span` lists in two flavours:

- Textual / Rich pull a ``rich.text.Text`` via :func:`to_rich_text`.
- prompt_toolkit consumes ``(Pygments token, text)`` pairs via
  :func:`to_pygments_pairs`.
- The LSP encodes spans into the SemanticTokens delta stream via
  :func:`to_semantic_token_legend` and :func:`to_semantic_token_data`.

The same :data:`STYLE_TABLE` drives Rich styling and the LSP legend so
themes never drift across surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quivers.dsl.pygments_lexer import (
    _ALGEBRA_NAMES,
    _BUILTIN_FUNCTION_TOKENS,
    _BUILTIN_TYPE_TOKENS,
    _KEYWORD_TOKENS,
    _OPERATOR_TOKENS,
    _load_parser,
)


# ---------------------------------------------------------------------------
# semantic token vocabulary
# ---------------------------------------------------------------------------


# Every QVR token reduces to one of these; the order is significant for
# the LSP semantic-token legend index.
SEMANTIC_TOKEN_TYPES: tuple[str, ...] = (
    "comment",
    "keyword",
    "operator",
    "punctuation",
    "string",
    "number",
    "variable",
    "type",
    "function",
    "namespace",
    "decorator",
    "error",
)


SEMANTIC_TOKEN_MODIFIERS: tuple[str, ...] = (
    "declaration",
    "definition",
    "deprecated",
)


# Rich style strings keyed by semantic token. Centralised so the TUI,
# the prompt_toolkit lexer, and the Jupyter kernel pick up the same
# colours.
STYLE_TABLE: dict[str, str] = {
    "comment": "italic dim",
    "keyword": "bold magenta",
    "operator": "cyan",
    "punctuation": "dim",
    "string": "green",
    "number": "yellow",
    "variable": "default",
    "type": "bold blue",
    "function": "bold cyan",
    "namespace": "bold yellow",
    "decorator": "magenta",
    "error": "bold red underline",
}


_PUNCT = {"(", ")", "[", "]", "{", "}", ",", ":", "."}


@dataclass(frozen=True)
class Span:
    """One styled run of source."""

    start: int
    end: int
    token: str
    text: str
    line: int = 0
    col: int = 0
    end_line: int = 0
    end_col: int = 0


# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------


def tokenize(
    source: str | bytes,
    *,
    env_kinds: dict[str, str] | None = None,
) -> list[Span]:
    """Return the styled span list for ``source``.

    Drives directly on the tree-sitter parse so that anonymous tokens
    (keywords, punctuation) are preserved — panproto's Schema view
    strips those, so we go to tree-sitter for highlighting and keep
    panproto for parsing into the didactic AST.

    ``env_kinds`` enables semantic highlighting: a mapping from name
    to one of ``"type"``, ``"function"``, ``"namespace"`` (or any
    SEMANTIC_TOKEN_TYPES value). Identifiers the grammar leaves as
    ``"variable"`` are upgraded to their env-known kind, so a name
    looks the same across the input pane, ``:type``, ``:info``, and
    any other surface — regardless of whether the surrounding context
    parses as a valid declaration.

    Failure path: if the parser raises, the entire source is returned
    as a single ``error`` span. Callers can downgrade to a no-highlight
    render but the buffer still displays.
    """
    if isinstance(source, str):
        src_bytes = source.encode("utf-8")
    else:
        src_bytes = source
    if not src_bytes:
        return []
    try:
        parser, _, _ = _load_parser()
        tree = parser.parse(src_bytes)
    except Exception:
        return [
            Span(
                start=0,
                end=len(src_bytes),
                token="error",
                text=src_bytes.decode("utf-8", errors="replace"),
                line=0,
                col=0,
                end_line=src_bytes.count(b"\n"),
                end_col=0,
            )
        ]

    leaves: list[tuple[Any, str | None]] = []

    def walk(node: Any, parent_kind: str | None) -> None:
        if not node.children:
            leaves.append((node, parent_kind))
            return
        for c in node.children:
            walk(c, node.type)

    walk(tree.root_node, None)

    spans: list[Span] = []
    cursor = 0
    for leaf, parent_kind in leaves:
        sb = leaf.start_byte
        eb = leaf.end_byte
        if sb > cursor:
            gap = src_bytes[cursor:sb].decode("utf-8", errors="replace")
            spans.append(
                _position(
                    Span(start=cursor, end=sb, token="variable", text=gap),
                    src_bytes,
                )
            )
        text = src_bytes[sb:eb].decode("utf-8", errors="replace")
        token = _classify(leaf.type, text, parent_kind)
        # Semantic upgrade: if the grammar produced a generic
        # "variable" classification but the env knows this name as a
        # type/function/namespace, paint it the env colour. This is
        # the seam that gives consistent highlighting across all REPL
        # output surfaces.
        if env_kinds and token == "variable":
            env_kind = env_kinds.get(text)
            if env_kind:
                token = env_kind
        spans.append(
            _position(Span(start=sb, end=eb, token=token, text=text), src_bytes)
        )
        cursor = eb
    if cursor < len(src_bytes):
        tail = src_bytes[cursor:].decode("utf-8", errors="replace")
        spans.append(
            _position(
                Span(
                    start=cursor,
                    end=len(src_bytes),
                    token="variable",
                    text=tail,
                ),
                src_bytes,
            )
        )
    return spans


def _classify(kind: str, text: str, parent_kind: str | None) -> str:
    if kind == "doc_comment":
        return "comment"
    if kind == "line_comment":
        return "comment"
    if kind == "integer" or kind == "float" or kind == "signed_number":
        return "number"
    if kind == "string":
        return "string"
    if kind == "identifier":
        # When a tree-sitter parse error puts a known keyword in the
        # 'identifier' bucket (because the surrounding production
        # didn't match), the text still tells us what the user wrote.
        # Treat that as a keyword so output stays self-consistent.
        if text in _KEYWORD_TOKENS:
            return "keyword"
        if text in _BUILTIN_FUNCTION_TOKENS:
            return "function"
        if text in _BUILTIN_TYPE_TOKENS:
            return "type"
        if text in _ALGEBRA_NAMES:
            return "namespace"
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
            return "type"
        return "variable"
    if kind in _OPERATOR_TOKENS:
        return "operator"
    if kind in _PUNCT:
        return "punctuation"
    if kind in _KEYWORD_TOKENS:
        return "keyword"
    return "variable"


def _position(span: Span, source: bytes) -> Span:
    """Attach line/col coordinates to a span."""
    sl, sc = _byte_to_line_col(source, span.start)
    el, ec = _byte_to_line_col(source, span.end)
    return Span(
        start=span.start,
        end=span.end,
        token=span.token,
        text=span.text,
        line=sl,
        col=sc,
        end_line=el,
        end_col=ec,
    )


def _byte_to_line_col(source: bytes, byte_offset: int) -> tuple[int, int]:
    if byte_offset > len(source):
        byte_offset = len(source)
    prefix = source[:byte_offset]
    line = prefix.count(b"\n")
    last_nl = prefix.rfind(b"\n")
    if last_nl < 0:
        col = byte_offset
    else:
        col = byte_offset - last_nl - 1
    return line, col


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------


def to_rich_text(
    source: str, *, env_kinds: dict[str, str] | None = None
) -> Any:
    """Build a :class:`rich.text.Text` from the highlighted source.

    Imported lazily so importing :mod:`quivers.cli.repl_highlight` does
    not pull rich in for callers that just want raw spans.
    """
    from rich.text import Text

    out = Text()
    for span in tokenize(source, env_kinds=env_kinds):
        style = STYLE_TABLE.get(span.token, "")
        out.append(span.text, style=style)
    return out


def to_pygments_pairs(source: str) -> list[tuple[Any, str]]:
    """Build ``(Pygments token, text)`` pairs suitable for prompt_toolkit.

    prompt_toolkit's PygmentsLexer wraps a Pygments Lexer that yields
    ``(index, token, text)``; here we deliver the simpler ``(token,
    text)`` form directly used by ``prompt_toolkit.formatted_text``.
    """
    from pygments.token import (
        Comment,
        Keyword,
        Name,
        Number,
        Operator,
        Punctuation,
        String,
        Text,
    )

    mapping = {
        "comment": Comment,
        "keyword": Keyword,
        "operator": Operator,
        "punctuation": Punctuation,
        "string": String,
        "number": Number,
        "variable": Name.Variable,
        "type": Name.Class,
        "function": Name.Builtin,
        "namespace": String.Symbol,
        "decorator": Name.Decorator,
        "error": Text,
    }
    out: list[tuple[Any, str]] = []
    for span in tokenize(source):
        out.append((mapping.get(span.token, Text), span.text))
    return out


# ---------------------------------------------------------------------------
# LSP semantic-tokens helpers
# ---------------------------------------------------------------------------


def to_semantic_token_legend() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the ``(types, modifiers)`` legend exported at server init."""
    return SEMANTIC_TOKEN_TYPES, SEMANTIC_TOKEN_MODIFIERS


def to_semantic_token_data(
    source: str, *, env_kinds: dict[str, str] | None = None
) -> list[int]:
    """Encode ``source`` as the LSP SemanticTokens 5-tuple stream.

    The stream is delta-encoded as ``[deltaLine, deltaStart, length,
    tokenType, tokenModifiers]`` per LSP 3.17. See
    https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_semanticTokens
    for the encoding rules.
    """
    type_index = {name: i for i, name in enumerate(SEMANTIC_TOKEN_TYPES)}
    out: list[int] = []
    prev_line = 0
    prev_col = 0
    for span in tokenize(source, env_kinds=env_kinds):
        if span.token in ("variable",) and not span.text.strip():
            # Pure whitespace at the start of the buffer; skip.
            continue
        if span.line != prev_line:
            prev_col = 0
        delta_line = span.line - prev_line
        delta_col = span.col - prev_col
        # Tokens that cross a newline aren't supported by the protocol;
        # split conservatively at the next newline.
        text = span.text
        length = len(text.encode("utf-8"))
        if "\n" in text:
            first_segment = text.split("\n", 1)[0]
            length = len(first_segment.encode("utf-8"))
        out.extend(
            [
                delta_line,
                delta_col,
                length,
                type_index.get(span.token, type_index["variable"]),
                0,
            ]
        )
        prev_line = span.line
        prev_col = span.col
    return out


__all__ = [
    "STYLE_TABLE",
    "SEMANTIC_TOKEN_MODIFIERS",
    "SEMANTIC_TOKEN_TYPES",
    "Span",
    "to_pygments_pairs",
    "to_rich_text",
    "to_semantic_token_data",
    "to_semantic_token_legend",
    "tokenize",
]
