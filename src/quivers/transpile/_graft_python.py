"""Graft a parsed Python expression into a per-render SchemaBuilder.

Several renderers need to emit a complex Python expression whose
shape is more natural to author as a literal source string than as
hand-built schema-builder vertices. This module parses the literal
through panproto's Python tree-sitter grammar at call time and
remaps every vertex id into the caller's `SchemaBuilder` so the
sub-graph plugs into the per-render schema cleanly.

The primary use site is the GP-kernel covariance emission for
numpyro / pyro / pymc / edward2: the kernel expression is parsed
once per emit (a few microseconds in tree-sitter) and the result
becomes the right-hand side of an `IRDeterministic` assignment.

Note: the parsed source must be a *single expression* (Python's
`expression_statement` wrapping is unwrapped by this module). The
caller is responsible for substituting any data-input names into
the literal before parsing (e.g. `length_scale=1.0` becomes the
literal `1.0` in the source string).
"""

from __future__ import annotations

import panproto

from quivers.transpile._pipeline import parser_registry


def graft_python_expression(
    pctx,
    source: str,
    *,
    fresh_prefix: str,
) -> str:
    """Parse ``source`` as a Python expression and graft the parsed
    subtree into the per-render `SchemaBuilder` carried by ``pctx``.

    Returns the vertex id of the rooted expression in the per-render
    schema. The caller wires that id under whatever edge is
    appropriate (e.g. ``assignment.right``, ``call.argument``).

    Vertex ids are rewritten through ``pctx.fresh(fresh_prefix)`` so
    repeated grafts in the same render do not collide.
    """
    parsed: panproto.Schema = parser_registry().parse_with_protocol(
        "python",
        source.encode("utf-8"),
        f"<graft:{fresh_prefix}>",
    )
    # The parsed schema is a `module` with the parsed expression as
    # its direct child (or under an `expression_statement` wrapper).
    root_id, expr_id = _find_module_and_expression(parsed)
    subtree = _subtree_ids(parsed, expr_id)
    id_map: dict[str, str] = {}
    for old in subtree:
        new = pctx.fresh(fresh_prefix)
        id_map[old] = new
        kind = next(v.kind for v in parsed.vertices if v.id == old)
        pctx.v(new, kind)
        for cstr in parsed.constraints_for(old):
            if cstr.sort in _SOURCE_POSITION_SORTS:
                # Drop byte-offset / interstitial layout constraints
                # since the grafted subtree lives at a different
                # position in the per-render source than it did in
                # the parse-time literal source.
                continue
            pctx.constraint(new, cstr.sort, cstr.value)
    for edge in parsed.edges:
        if edge.src in id_map and edge.tgt in id_map:
            pctx.e(id_map[edge.src], id_map[edge.tgt], edge.kind)
    del root_id
    return id_map[expr_id]


#: Sorts of constraints that should be elided from a grafted subtree.
#: `start-byte` / `end-byte` / `interstitial-N-start-byte` carry the
#: parse-time source position; the per-render printer would otherwise
#: emit a newline before the grafted subtree's first byte (the
#: subtree's `start-byte=0` doesn't match the position the new
#: parent's `=` sign sits at). The traversal-order constraints
#: (`ptrace-*`, `chose-alt-fingerprint`, `chose-alt-child-kinds`,
#: `literal-value`, `interstitial-N`) are preserved.
_SOURCE_POSITION_SORTS: frozenset[str] = frozenset({
    "start-byte",
    "end-byte",
    "interstitial-0-start-byte",
    "interstitial-1-start-byte",
    "interstitial-2-start-byte",
    "interstitial-3-start-byte",
    "interstitial-4-start-byte",
    "interstitial-5-start-byte",
    "interstitial-6-start-byte",
    "interstitial-7-start-byte",
    "interstitial-8-start-byte",
    "interstitial-9-start-byte",
})


def _find_module_and_expression(
    schema: panproto.Schema,
) -> tuple[str, str]:
    """Locate the module's first expression-shaped child and return
    ``(module_id, expression_id)``.

    Tree-sitter Python parses a bare expression source as a module
    whose direct child is either the expression itself (when the
    grammar accepts the form as a `simple_statement`) or an
    `expression_statement` wrapper. Either shape is unwrapped to
    the inner expression.
    """
    module_id: str | None = None
    for v in schema.vertices:
        if v.kind == "module":
            module_id = v.id
            break
    if module_id is None:
        raise ValueError(
            "graft_python_expression: parsed schema has no `module` "
            "root; cannot locate the expression to graft"
        )
    kind_by_id = {v.id: v.kind for v in schema.vertices}
    for edge in schema.edges:
        if edge.src != module_id:
            continue
        tgt_kind = kind_by_id.get(edge.tgt)
        if tgt_kind == "expression_statement":
            stmt_id = edge.tgt
            for e2 in schema.edges:
                if e2.src == stmt_id:
                    return module_id, e2.tgt
        if tgt_kind in (
            "call", "binary_operator", "unary_operator",
            "identifier", "attribute", "integer", "float",
            "subscript", "parenthesized_expression",
        ):
            return module_id, edge.tgt
    raise ValueError(
        "graft_python_expression: parsed schema's module has no "
        "expression-shaped child; the source must be a single "
        "Python expression"
    )


def _subtree_ids(schema: panproto.Schema, root: str) -> set[str]:
    """Return every vertex id reachable from ``root`` via outgoing
    edges of ``schema``."""
    seen: set[str] = {root}
    frontier: list[str] = [root]
    while frontier:
        src = frontier.pop()
        for edge in schema.edges:
            if edge.src == src and edge.tgt not in seen:
                seen.add(edge.tgt)
                frontier.append(edge.tgt)
    return seen


__all__ = ["graft_python_expression"]
