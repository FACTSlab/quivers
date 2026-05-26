"""Per-backend structural-assertion helpers.

A transpile output's panproto schema must satisfy a backend-specific
shape contract that goes beyond "parses with the tree-sitter grammar."
The helpers here walk an emitted
[`panproto.Schema`][panproto.Schema] and assert presence, counts,
nesting, and field-edge wiring of specific vertices. They catch
silent regressions like empty blocks, missing tilde statements, or
duplicated `with pymc.Model()` instantiations.

The assertion vocabulary is grammar-agnostic; each backend supplies
the vertex kinds and field labels it cares about. Common idioms:

- [`vertices_of_kind`][tests.transpile._structural.vertices_of_kind]
  returns every vertex whose `kind` matches.
- [`children_of`][tests.transpile._structural.children_of] returns
  the targets of outgoing `child_of` edges from a vertex.
- [`field_target`][tests.transpile._structural.field_target] returns
  the single target of a named field edge (the tree-sitter
  field-label edge kind).
- [`literal_value`][tests.transpile._structural.literal_value]
  returns the `literal-value` constraint text on a vertex, if any.
- [`assert_unique_kind`][tests.transpile._structural.assert_unique_kind]
  asserts exactly N vertices of a given kind exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import panproto


def vertices_of_kind(schema: panproto.Schema, kind: str) -> list[panproto.Vertex]:
    """Every vertex in ``schema`` whose `kind` matches ``kind``."""
    return [v for v in schema.vertices if v.kind == kind]


def vertex_ids_of_kind(schema: panproto.Schema, kind: str) -> list[str]:
    """The vertex ids of every vertex with the given kind."""
    return [v.id for v in schema.vertices if v.kind == kind]


def children_of(schema: panproto.Schema, vertex_id: str) -> list[str]:
    """Target vertex ids of every outgoing `child_of` edge."""
    return [
        e.tgt for e in schema.edges
        if e.src == vertex_id and e.kind == "child_of"
    ]


def outgoing_edges_named(
    schema: panproto.Schema, vertex_id: str, kind: str
) -> list[str]:
    """Target vertex ids of every outgoing edge with the given `kind`.

    Used for tree-sitter field-label edges where the edge `kind` is
    the field name (e.g., `name`, `function`, `arguments`, `body`).
    """
    return [e.tgt for e in schema.edges if e.src == vertex_id and e.kind == kind]


def field_target(
    schema: panproto.Schema, vertex_id: str, field: str
) -> str:
    """The single target of the named field edge on ``vertex_id``.

    Raises ``AssertionError`` if the field is missing or has more
    than one target (every tree-sitter field edge is singular).
    """
    matches = outgoing_edges_named(schema, vertex_id, field)
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly 1 `{field}` edge from {vertex_id!r}; "
            f"got {len(matches)}: {matches!r}"
        )
    return matches[0]


def literal_value(
    schema: panproto.Schema, vertex_id: str
) -> str | None:
    """The `literal-value` constraint text on ``vertex_id``, or None."""
    for c in schema.constraints_for(vertex_id):
        if c.sort == "literal-value":
            return c.value
    return None


def assert_unique_kind(
    schema: panproto.Schema,
    kind: str,
    expected: int,
    *,
    context: str = "",
) -> list[str]:
    """Assert exactly ``expected`` vertices of ``kind`` exist.

    Returns the matching vertex ids on success (callers chain into
    `outgoing_edges_named` etc.).
    """
    ids = vertex_ids_of_kind(schema, kind)
    if len(ids) != expected:
        raise AssertionError(
            f"{context + ': ' if context else ''}"
            f"expected exactly {expected} vertex(es) of kind {kind!r}; "
            f"got {len(ids)}: {ids!r}"
        )
    return ids


def descend(schema: panproto.Schema, vertex_id: str) -> set[str]:
    """Every vertex id reachable from ``vertex_id`` via outgoing edges
    (transitive closure). Used to scope assertions to a sub-tree."""
    seen: set[str] = {vertex_id}
    frontier = [vertex_id]
    while frontier:
        v = frontier.pop()
        for e in schema.edges:
            if e.src == v and e.tgt not in seen:
                seen.add(e.tgt)
                frontier.append(e.tgt)
    return seen


# ---------------------------------------------------------------------------
# Backend-specific shape assertions for the canonical
# beta-Bernoulli fixture, parametrised over backend name.
# ---------------------------------------------------------------------------


def assert_stan_beta_bernoulli(schema: panproto.Schema) -> None:
    """The Stan output for ``flip`` must have: a `program` root with
    `data`, `parameters`, and `model` children in that block order;
    one `top_var_decl_no_assign` for `y` (real) under `data`; one
    `top_var_decl_no_assign` for `theta` (real) under `parameters`;
    two `sampling_statement`s under `model` whose `name`-field
    targets carry `literal-value` `beta` and `bernoulli`."""
    program = assert_unique_kind(schema, "program", 1)[0]
    blocks = children_of(schema, program)
    block_kinds = [
        next(v for v in schema.vertices if v.id == b).kind for b in blocks
    ]
    assert "data" in block_kinds, f"missing `data` block; got {block_kinds}"
    assert "parameters" in block_kinds, (
        f"missing `parameters` block; got {block_kinds}"
    )
    assert "model" in block_kinds, f"missing `model` block; got {block_kinds}"

    [model_id] = [b for b in blocks if vertex_kind(schema, b) == "model"]
    stmts = [
        c for c in children_of(schema, model_id)
        if vertex_kind(schema, c) == "sampling_statement"
    ]
    dist_names = sorted(
        literal_value(schema, field_target(schema, s, "name")) or ""
        for s in stmts
    )
    assert dist_names == ["bernoulli", "beta"], (
        f"expected dist names [bernoulli, beta]; got {dist_names}"
    )


def vertex_kind(schema: panproto.Schema, vertex_id: str) -> str:
    """The `kind` of the vertex with the given id."""
    for v in schema.vertices:
        if v.id == vertex_id:
            return v.kind
    raise AssertionError(f"no vertex with id {vertex_id!r}")


def assert_numpyro_beta_bernoulli(schema: panproto.Schema) -> None:
    """The NumPyro output must have exactly one `function_definition`
    named `model` whose body contains: one `assignment` whose right
    side is a `numpyro.sample("theta", numpyro.distributions.Beta(...))`
    call; one `numpyro.sample("y", ..., obs=y)` call (as a bare
    call inside the block, no assignment)."""
    [fn] = assert_unique_kind(schema, "function_definition", 1)
    name_target = field_target(schema, fn, "name")
    assert literal_value(schema, name_target) == "model", (
        f"function name = {literal_value(schema, name_target)!r}, expected 'model'"
    )

    sample_calls = [
        v.id for v in schema.vertices if v.kind == "call"
        and _call_is_attribute(schema, v.id, ("numpyro", "sample"))
    ]
    assert len(sample_calls) == 2, (
        f"expected 2 numpyro.sample calls; got {len(sample_calls)}"
    )

    sample_names = sorted(
        _string_literal_arg(schema, call_id) for call_id in sample_calls
    )
    assert sample_names == ["theta", "y"], (
        f"expected sample names [theta, y]; got {sample_names}"
    )


def _call_is_attribute(
    schema: panproto.Schema,
    call_id: str,
    chain: tuple[str, ...],
) -> bool:
    """True iff ``call_id`` is a Python `call` whose callee is the
    chained attribute access spelt by ``chain``."""
    fn_targets = outgoing_edges_named(schema, call_id, "function")
    if len(fn_targets) != 1:
        return False
    return _attribute_matches(schema, fn_targets[0], chain)


def _attribute_matches(
    schema: panproto.Schema,
    vertex_id: str,
    chain: tuple[str, ...],
) -> bool:
    """True iff ``vertex_id`` is an attribute-access chain spelling
    ``chain`` (left-recursive in tree-sitter Python: the deepest
    object is the first element of ``chain``)."""
    if len(chain) == 0:
        return False
    if len(chain) == 1:
        return (
            vertex_kind(schema, vertex_id) == "identifier"
            and literal_value(schema, vertex_id) == chain[0]
        )
    if vertex_kind(schema, vertex_id) != "attribute":
        return False
    object_targets = outgoing_edges_named(schema, vertex_id, "object")
    attribute_targets = outgoing_edges_named(schema, vertex_id, "attribute")
    if len(object_targets) != 1 or len(attribute_targets) != 1:
        return False
    return (
        _attribute_matches(schema, object_targets[0], chain[:-1])
        and literal_value(schema, attribute_targets[0]) == chain[-1]
    )


def _string_literal_arg(
    schema: panproto.Schema, call_id: str
) -> str | None:
    """First positional `string` argument's literal content, or None."""
    arg_lists = outgoing_edges_named(schema, call_id, "arguments")
    if len(arg_lists) != 1:
        return None
    arg_list = arg_lists[0]
    for child in children_of(schema, arg_list):
        if vertex_kind(schema, child) == "string":
            for sc in children_of(schema, child):
                if vertex_kind(schema, sc) == "string_content":
                    return literal_value(schema, sc)
    return None


__all__ = [
    "assert_numpyro_beta_bernoulli",
    "assert_stan_beta_bernoulli",
    "assert_unique_kind",
    "children_of",
    "descend",
    "field_target",
    "literal_value",
    "outgoing_edges_named",
    "vertex_ids_of_kind",
    "vertex_kind",
    "vertices_of_kind",
]
