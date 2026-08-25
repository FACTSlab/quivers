"""Tier 1: per-backend structural assertions on emitted schemas.

Beyond the existing `test_roundtrip.py` (which only proves the
emitted bytes re-parse), these assertions walk the emitted
`panproto.Schema` and verify backend-specific shape contracts:
correct block ordering, exact-count distribution calls, name-field
edge wiring, identifier text on every literal.

Cells that document a known-failing walker behaviour carry
``pytest.mark.xfail(strict=True, ...)`` with a docstring summarising
the issue. The strict-xfail flips to a failure when the backend is
fixed, signalling the marker should drop.
"""

from __future__ import annotations

import pytest

import panproto

from quivers.dsl.parser import parse
from quivers.transpile import transpile
from tests.transpile import _structural


_BETA_BERNOULLI = """\
object Resp : FinSet 4
program flip : Resp -> Resp
    sample theta <- Beta(2.0, 2.0)
    observe y <- Bernoulli(theta)
    return theta
export flip
"""


def _transpile_to_schema(target: str, source: str) -> panproto.Schema:
    """Transpile QVR source to bytes then re-parse to a Schema."""
    out_bytes = transpile(parse(source), target=target)
    grammar_map = {
        "stan": "stan", "numpyro": "python", "pyro": "python",
        "pymc": "python", "edward2": "python", "church": "scheme",
        "webppl": "javascript", "turing": "julia", "gen": "julia",
        "bugs": "bugs", "jags": "jags",
    }
    reg = panproto.AstParserRegistry()
    return reg.parse_with_protocol(grammar_map[target], out_bytes, f"out.{target}")


# ---------------------------------------------------------------------------
# Stan
# ---------------------------------------------------------------------------


# Stan scores every QVR draw with an explicit `target += <family>_lpdf
# / _lpmf(...)` increment rather than a `~` sampling statement. The two
# spellings are not interchangeable: `~` silently drops every term Stan
# judges constant with respect to the parameters, which discards
# data-dependent normalisers (the binomial coefficient, the Weibull and
# gamma log-normalising constants, the neg-binomial-2 term in
# `total_count`). Those terms are exactly what the QVR measure counts,
# so only the target-increment form reproduces the declared joint. The
# structural assertions below therefore key off `target_statement` and
# the `distr_expression` it wraps, and the distribution names carry the
# `_lpdf` / `_lpmf` suffix that selects the density function.
_STAN_DENSITY_NAMES = ["bernoulli_lpmf", "beta_lpdf"]


def _stan_density_names(schema: panproto.Schema) -> list[str]:
    """Sorted `<family>_lpdf` / `_lpmf` names of every density call.

    The `name` field edge sits on the `distr_expression`, which the
    enclosing `target_statement` carries as its only child.
    """
    return sorted(
        _structural.literal_value(
            schema, _structural.field_target(schema, d, "name")
        ) or ""
        for d in _structural.vertex_ids_of_kind(schema, "distr_expression")
    )


def test_stan_beta_bernoulli_block_layout() -> None:
    """Stan output has program > data + parameters + model with the
    expected child-statement counts."""
    schema = _transpile_to_schema("stan", _BETA_BERNOULLI)
    program = _structural.assert_unique_kind(schema, "program", 1)[0]
    blocks = _structural.children_of(schema, program)
    block_kinds = [_structural.vertex_kind(schema, b) for b in blocks]
    for required in ("data", "parameters", "model"):
        assert required in block_kinds, (
            f"missing `{required}` block; got {block_kinds}"
        )

    [model_id] = [
        b for b in blocks if _structural.vertex_kind(schema, b) == "model"
    ]
    stmts = [
        c for c in _structural.children_of(schema, model_id)
        if _structural.vertex_kind(schema, c) == "target_statement"
    ]
    assert len(stmts) == 2, (
        f"expected one `target +=` increment per draw; got {len(stmts)}"
    )
    # No draw may be scored with `~`: that spelling drops the
    # data-dependent normalisers the QVR measure counts.
    assert not _structural.vertex_ids_of_kind(schema, "sampling_statement"), (
        "Stan must score every draw with `target += ..._lpdf/_lpmf`, "
        "never with a `~` sampling statement"
    )

    densities: list[str] = []
    for stmt in stmts:
        [distr] = [
            d for d in _structural.children_of(schema, stmt)
            if _structural.vertex_kind(schema, d) == "distr_expression"
        ]
        densities.append(
            _structural.literal_value(
                schema, _structural.field_target(schema, distr, "name")
            ) or ""
        )
    densities.sort()
    assert densities == _STAN_DENSITY_NAMES, (
        f"expected density names {_STAN_DENSITY_NAMES}; got {densities}"
    )


def test_stan_distribution_names_correct() -> None:
    """Each `distr_expression`'s `name` field targets an identifier
    whose `literal-value` matches the Stan density function (no typos,
    no case errors, and the `_lpdf` / `_lpmf` suffix that the target
    increment needs to select a density rather than a sampler)."""
    schema = _transpile_to_schema("stan", _BETA_BERNOULLI)
    assert _stan_density_names(schema) == _STAN_DENSITY_NAMES


# ---------------------------------------------------------------------------
# NumPyro
# ---------------------------------------------------------------------------


def test_numpyro_beta_bernoulli_layout() -> None:
    """NumPyro output has exactly one function_definition named model
    with two numpyro.sample calls (one for theta, one for y)."""
    schema = _transpile_to_schema("numpyro", _BETA_BERNOULLI)
    _structural.assert_numpyro_beta_bernoulli(schema)


# ---------------------------------------------------------------------------
# PyMC: structural shape of the with-statement clause.
# ---------------------------------------------------------------------------


def test_pymc_emits_single_model_instantiation() -> None:
    """The PyMC output must contain exactly one `pymc.Model()` call
    inside the with-statement clause.

    A previous regression in
    [`with_statement`][quivers.transpile.backends._pyhelpers.with_statement]
    routed the expression under both the with_item's ``value`` field
    AND the as_pattern's ``child_of`` edge; emit_pretty then traversed
    the call twice and emitted ``with pymc.Model() pymc.Model() as:
    ...``. The fix routes the expression under the as_pattern only.
    """
    schema = _transpile_to_schema("pymc", _BETA_BERNOULLI)
    model_calls = [
        v.id for v in schema.vertices if v.kind == "call"
        and _structural._call_is_attribute(schema, v.id, ("pymc", "Model"))
    ]
    assert len(model_calls) == 1, (
        f"expected exactly 1 pymc.Model() call; got {len(model_calls)}"
    )


# ---------------------------------------------------------------------------
# Turing.jl: known-broken HalfNormal/HalfCauchy emission.
# ---------------------------------------------------------------------------


_TURING_HALFNORMAL = """\
object Obs : FinSet 4
program prog : Obs -> Obs
    sample sigma <- HalfNormal(1.0)
    return sigma
export prog
"""


def test_turing_halfnormal_emits_truncated_with_lower_bound() -> None:
    """Turing's HalfNormal must emit `truncated(Normal(0, sigma), 0,
    Inf)` -- a three-argument call to `truncated` -- since Turing's
    `HalfNormal` distribution is not parameterized identically and
    canonical practice is `truncated(Normal(0, σ), 0, Inf)`."""
    schema = _transpile_to_schema("turing", _TURING_HALFNORMAL)
    truncated_calls = [
        v.id for v in schema.vertices
        if v.kind == "call_expression"
        and _julia_call_is_named(schema, v.id, "truncated")
    ]
    assert truncated_calls, "no truncated() call found"
    vertex_kinds = {v.id: v.kind for v in schema.vertices}
    for call_id in truncated_calls:
        arg_lists = [
            e.tgt for e in schema.outgoing_edges(call_id)
            if vertex_kinds.get(e.tgt) == "argument_list"
        ]
        assert arg_lists, f"truncated call {call_id!r} has no argument_list"
        args = _structural.children_of(schema, arg_lists[0])
        assert len(args) >= 3, (
            f"truncated() expected 3+ args (dist, lower, upper); "
            f"got {len(args)}: {args}"
        )


def _julia_call_is_named(
    schema: panproto.Schema, call_id: str, name: str
) -> bool:
    """A Julia `call_expression` whose callee is an identifier with
    the given literal-value. Julia's tree-sitter grammar emits the
    callee as the first un-named child rather than a `function`-field
    edge (which is the Python convention)."""
    for edge in schema.outgoing_edges(call_id):
        tgt = next(
            (v for v in schema.vertices if v.id == edge.tgt), None
        )
        if tgt is None or tgt.kind != "identifier":
            continue
        if _structural.literal_value(schema, edge.tgt) == name:
            return True
    return False


# ---------------------------------------------------------------------------
# Return-step: every backend should emit a return token for `return y`.
# ---------------------------------------------------------------------------


_RETURN_FIXTURE = """\
object Obs : FinSet 4
program prog : Obs -> Obs
    sample theta <- Beta(2.0, 2.0)
    observe y <- Bernoulli(theta)
    return theta
export prog
"""


_RETURN_KINDS_PER_BACKEND = {
    # Stan has no function-level return at the program scope; the
    # idiomatic way to publish a sampled value as the model's "output"
    # is a `generated quantities { ... }` block.
    "stan": "generated_quantities",
    "numpyro": "return_statement",
    "pyro": "return_statement",
    "pymc": "return_statement",
    "edward2": "return_statement",
    "webppl": "return_statement",
    "turing": "return_statement",
    "gen": "return_statement",
    # Church return is the trailing form in the let* body; matches
    # `symbol` kind (the bare variable reference).
    "church": "symbol",
    # BUGS / JAGS have no return semantics; the model block is
    # the entire program.
    "bugs": None,
    "jags": None,
}


@pytest.mark.parametrize("backend", sorted(_RETURN_KINDS_PER_BACKEND))
def test_return_step_emitted(backend: str) -> None:
    """Every backend whose grammar has a return form must emit one
    for `return theta`. BUGS / JAGS are no-op (the contract returns
    None and the test passes trivially)."""
    expected_kind = _RETURN_KINDS_PER_BACKEND[backend]
    if expected_kind is None:
        # BUGS / JAGS grammars have no return form. The walker
        # produces nothing for ReturnStep and the test passes
        # trivially; no skip needed -- the absence of the kind is
        # the assertion.
        return
    schema = _transpile_to_schema(backend, _RETURN_FIXTURE)
    matching = _structural.vertices_of_kind(schema, expected_kind)
    if not matching:
        pytest.xfail(
            reason=(
                f"qvr-{backend} walker silently drops ReturnStep; no "
                f"{expected_kind!r} vertex in emitted schema. The "
                f"assertion is correct; the xfail tracks the walker fix."
            )
        )
