"""Shared target-grammar services for the transpiler pipeline.

Every backend reduces to the same flow:

1. The source is the QVR [`Module`][quivers.dsl.ast_nodes.Module] AST
   (a didactic tagged-union tree). The
   [`QVR_PROGRAM_PROTOCOL`][quivers.dsl.program_theory.QVR_PROGRAM_PROTOCOL]
   panproto schema captures the *declaration shell* (object_decl,
   kernel_decl, domain/codomain) but not the program-step bodies
   (sample / observe / let / score with their family applications and
   axis specs); transpilation needs the bodies, so the source is the
   Module AST rather than the extracted schema.
2. [`Lower`][quivers.transpile.lower.Lower] constructs the shared structural
   IR, and a registered renderer constructs a fresh `panproto.Schema` in the
   target tree-sitter grammar's auto-derived theory, using
   `panproto.SchemaBuilder`. Vertex kinds
   match the grammar's `node-types.json`; identifier text is set via
   ``literal-value`` constraints; field-labelled edges use the field
   name as the edge kind.
3. `panproto.AstParserRegistry.emit_pretty`
   walks the target grammar's `grammar.json` productions to render the
   schema back to source bytes. No string templating in quivers.

The renderer registry in [`quivers.transpile`][quivers.transpile] owns the
complete source-to-target path. This module supplies the cached grammar
registry, target protocols, and the schema-to-bytes mapping used by renderers
and law tests.
"""

from __future__ import annotations

import didactic.api as dx
import panproto
from panproto._native import AstParserRegistry as _NativeAstParserRegistry

_REGISTRY: _NativeAstParserRegistry | None = None


def parser_registry() -> _NativeAstParserRegistry:
    """Cached process-wide `AstParserRegistry`.

    Construction walks every installed ``panproto.grammars`` entry-point
    pack; doing it once amortises that work across every transpile call.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = panproto.AstParserRegistry()
    return _REGISTRY


def target_protocol(grammar: str) -> panproto.Protocol:
    """Synthesise a `panproto.Protocol` handle for
    a tree-sitter grammar.

    Tree-sitter grammars are not registered as builtin protocols
    (``panproto.get_builtin_protocol(grammar)`` raises ``KeyError`` for
    every grammar in `AstParserRegistry().protocol_names()`). The
    panproto API does, however, accept a string theory name in
    `Protocol.from_theories`; the
    resulting Protocol is suitable for fresh
    `schema()` builders that emit through
    the grammar's auto-derived theory.
    """
    # `schema_theory` is documented to accept either a `Theory` instance
    # or a string theory name; the published stub types it as `Theory`
    # only, so we cast at the boundary.
    return panproto.Protocol.from_theories(
        name=grammar,
        schema_theory=grammar,  # type: ignore[arg-type]
        obj_kinds=[],
    )


class EmitPretty(dx.Mapping[panproto.Schema, bytes]):
    """`Mapping[panproto.Schema, bytes]` over
    `emit_pretty`."""

    def __init__(self, grammar: str) -> None:
        self._grammar = grammar

    def forward(self, schema: panproto.Schema) -> bytes:
        return bytes(parser_registry().emit_pretty(self._grammar, schema))
