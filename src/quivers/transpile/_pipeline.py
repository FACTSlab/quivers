"""The realize-from-AST pipeline shared by every backend.

Every backend reduces to the same flow:

1. The source is the QVR [`Module`][quivers.dsl.ast_nodes.Module] AST
   (a didactic tagged-union tree). The
   [`QVR_PROGRAM_PROTOCOL`][quivers.dsl.program_theory.QVR_PROGRAM_PROTOCOL]
   panproto schema captures the *declaration shell* (object_decl,
   kernel_decl, domain/codomain) but not the program-step bodies
   (sample / observe / let / score with their family applications and
   axis specs); transpilation needs the bodies, so the source is the
   Module AST rather than the extracted schema.
2. A backend-supplied [`SchemaTransform`][quivers.transpile.SchemaTransform]
   walks the Module and constructs a fresh `panproto.Schema` in the
   target tree-sitter grammar's auto-derived theory, using
   [`panproto.SchemaBuilder`][panproto.SchemaBuilder]. Vertex kinds
   match the grammar's `node-types.json`; identifier text is set via
   ``literal-value`` constraints; field-labelled edges use the field
   name as the edge kind.
3. [`panproto.AstParserRegistry.emit_pretty`][panproto.AstParserRegistry.emit_pretty]
   walks the target grammar's `grammar.json` productions to render the
   schema back to source bytes. No string templating in quivers.

The [`SchemaTransform`][quivers.transpile.SchemaTransform] is a
[`didactic.api.Mapping`][didactic.api.Mapping] whose ``forward`` takes
the Module and returns the target schema, so backends compose with
``>>`` and share base walkers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import didactic.api as dx
import panproto
from panproto._native import AstParserRegistry as _NativeAstParserRegistry

if TYPE_CHECKING:
    from quivers.dsl.ast_nodes import Module


_REGISTRY: _NativeAstParserRegistry | None = None


def parser_registry() -> _NativeAstParserRegistry:
    """Cached process-wide [`AstParserRegistry`][panproto.AstParserRegistry].

    Construction walks every installed ``panproto.grammars`` entry-point
    pack; doing it once amortises that work across every transpile call.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = panproto.AstParserRegistry()
    return _REGISTRY


def target_protocol(grammar: str) -> panproto.Protocol:
    """Synthesise a [`panproto.Protocol`][panproto.Protocol] handle for
    a tree-sitter grammar.

    Tree-sitter grammars are not registered as builtin protocols
    (``panproto.get_builtin_protocol(grammar)`` raises ``KeyError`` for
    every grammar in `AstParserRegistry().protocol_names()`). The
    panproto API does, however, accept a string theory name in
    [`Protocol.from_theories`][panproto.Protocol.from_theories]; the
    resulting Protocol is suitable for fresh
    [`schema()`][panproto.Protocol.schema] builders that emit through
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


class SchemaTransform(dx.Mapping):
    """[`Mapping[Module, panproto.Schema]`][didactic.api.Mapping].

    Subclasses override [`forward`][didactic.api.Mapping.forward] to
    walk the [`Module`][quivers.dsl.ast_nodes.Module] AST and build the
    target schema via [`panproto.SchemaBuilder`][panproto.SchemaBuilder].
    The Mapping superclass supplies ``>>`` composition with
    [`EmitPretty`][quivers.transpile._pipeline.EmitPretty] and any
    other downstream Mapping.
    """

    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(module): received {module!r}"
        )


class EmitPretty(dx.Mapping):
    """[`Mapping[panproto.Schema, bytes]`][didactic.api.Mapping] over
    [`emit_pretty`][panproto.AstParserRegistry.emit_pretty]."""

    def __init__(self, grammar: str) -> None:
        self._grammar = grammar

    def forward(self, schema: panproto.Schema) -> bytes:  # type: ignore[override]
        return bytes(parser_registry().emit_pretty(self._grammar, schema))


def realize(module: Module, *, grammar: str, transform: SchemaTransform) -> bytes:
    """Run the full pipeline for one backend.

    Parameters
    ----------
    module
        The parsed [`Module`][quivers.dsl.ast_nodes.Module] AST.
    grammar
        The tree-sitter grammar name.
    transform
        The QVR-to-target-schema mapping for this backend.

    Returns
    -------
    bytes
        The transpiled source. Always passes through
        [`emit_pretty`][panproto.AstParserRegistry.emit_pretty]; never
        constructed by string interpolation.
    """
    pipeline = cast(
        "dx.Mapping[Module, bytes]",
        transform >> EmitPretty(grammar),
    )
    return cast("bytes", pipeline(module))
