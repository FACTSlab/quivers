"""Tier 4: (construct fixture × backend) compatibility matrix.

For every fixture under `fixtures/{statements, steps,
let_expressions, options, axes}` and every registered backend, the
test asserts one of three pre-declared outcomes:

1. **Construct out of tier**: the fixture's statement kind is not in
   the backend's `_SUPPORT_TIER` frozenset; the pipeline MUST raise
   `UnsupportedConstruct` with at least one of the predicted kinds.
2. **Orthogonal concern**: the fixture's statement kind IS in the
   tier, but the fixture also exercises a feature the backend
   cannot represent (a family without a target_name; a let-expression
   shape with no method-call surface; a downstream IR node kind the
   renderer hasn't wired). Each such cell is pinned in
   `_EXPECTED_ORTHOGONAL_RAISES` with the kind-prefix the raise
   must carry. A closed gap surfaces as a test failure
   (pytest.raises matches no entry); a regression surfaces as a
   different-kind raise.
3. **Render**: not in either bucket; the pipeline MUST emit
   non-empty bytes.

Every cell is therefore a positive assertion — no xfail-on-exception
dispatch.
"""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import (
    UnsupportedConstruct,
    available_targets,
    transpile,
    unsupported_for,
)
from quivers.transpile._api import (
    CATEGORICAL_METADATA_IGNORABLE,
    CHURCH_LIKE,
    STAN_LIKE,
)
from tests.transpile.fixtures import _load


_BACKENDS = sorted(available_targets())


# Backend → support tier. Each tier is a frozenset of the QVR
# `Statement` discriminators the backend's pipeline (frontend
# expand + Lower + Renderer) accepts.
_SUPPORT_TIER: dict[str, frozenset[str]] = {
    "stan": STAN_LIKE,
    "numpyro": STAN_LIKE,
    "pyro": STAN_LIKE,
    "pymc": STAN_LIKE,
    "edward2": STAN_LIKE,
    "church": CHURCH_LIKE,
    "webppl": CHURCH_LIKE,
    "turing": STAN_LIKE,
    "gen": STAN_LIKE,
    "bugs": STAN_LIKE,
    "jags": STAN_LIKE,
}


# Some categorical fixtures introduce a non-supported statement
# alongside the program_decl. The construct test for those fixtures
# expects `UnsupportedConstruct`.
def _expected_unsupported_kinds(fixture: _load.Fixture, backend: str) -> set[str]:
    """Return the set of kinds the backend should report as
    unsupported for ``fixture``.

    Walks the parsed module and collects every statement discriminator
    that falls outside the backend's support tier.
    [`CATEGORICAL_METADATA_IGNORABLE`][quivers.transpile.CATEGORICAL_METADATA_IGNORABLE]
    kinds are accepted alongside a `program_decl` per the
    [`unsupported_for`][quivers.transpile.unsupported_for] contract,
    so they're not flagged when a program is present.
    """
    module = parse(fixture.source)
    tier = _SUPPORT_TIER[backend]
    kinds = {str(stmt.kind) for stmt in module.statements}
    has_program = "program_decl" in kinds
    effective_tier = (
        tier | CATEGORICAL_METADATA_IGNORABLE if has_program else tier
    )
    return {k for k in kinds if k not in effective_tier}


# Cells where the fixture's statement kind IS in the backend's
# support tier, but a different concern (the resolved distribution
# family, a let-expression shape, or a downstream IR node) provokes
# `UnsupportedConstruct` before the construct gate fires. Each entry
# pins the expected kind-prefix of the raise.
#
# Key: (backend, fixture-category, fixture-name).
# Value: the prefix the raised `UnsupportedConstruct.kinds` must
# match (`"family:"`, `"let-expr:"`, `"node:"`, `"declare:"`, etc.).
#
# A closed gap surfaces as a test failure because `pytest.raises`
# succeeds but the assertion-of-prefix is unreachable; remove the
# entry. A regression surfaces because the raised kinds match a
# different prefix; either update the entry or fix the renderer.
_EXPECTED_ORTHOGONAL_RAISES: dict[tuple[str, str, str], str] = {
    # Stan / BUGS / JAGS have no method-dispatch syntax for the
    # chart-parser `parser.parse(sentence)` method call. The
    # deduction graft that would supply the called function is
    # blocked by `CATEGORICAL_METADATA_IGNORABLE` (Stan) or by
    # dialect restrictions on user-defined model-body functions
    # (BUGS, JAGS).
    ("stan", "let_expressions", "let_expr_method_call"): "let-expr:",
    ("bugs", "let_expressions", "let_expr_method_call"): "let-expr:",
    ("jags", "let_expressions", "let_expr_method_call"): "let-expr:",
    # Stan / BUGS / JAGS have no anonymous-function syntax in the
    # model-body expression position, so a `param -> body` lambda is
    # an orthogonal unsupported concern for these dialects.
    ("stan", "let_expressions", "let_expr_lambda"): "let-expr:",
    ("bugs", "let_expressions", "let_expr_lambda"): "let-expr:",
    ("jags", "let_expressions", "let_expr_lambda"): "let-expr:",
    # BUGS / JAGS dialects ship no MatrixNormal surface, so the
    # axes/matrix_kronecker fixture trips the family-target-name
    # check before the construct gate.
    ("bugs", "axes", "matrix_kronecker"): "family:",
    ("jags", "axes", "matrix_kronecker"): "family:",
    # Gen refuses every `marginalize`: its `@gen` DSL has no way to
    # add a free log-density term to a trace, so the only thing it
    # could emit is the latent as a draw, which denotes a measure on
    # the product of the latent's support with the block's rather
    # than the integral the block means.
    ("gen", "steps", "marginalize_step"): "marginalize:",
}


def _construct_fixtures() -> list[_load.Fixture]:
    """Every fixture across the per-construct categories."""
    return (
        _load.load_statements()
        + _load.load_steps()
        + _load.load_let_expressions()
        + _load.load_options()
        + _load.load_axes()
    )


def _transpile_with_tier_check(
    module, *, target: str
) -> bytes:
    """Run [`unsupported_for`][quivers.transpile.unsupported_for]
    against the backend's support tier, then
    [`transpile`][quivers.transpile.transpile].

    The support-tier check is the documented public contract for
    "this module contains statement kinds the backend cannot
    represent"; the tier frozensets live in
    [`quivers.transpile._api`][quivers.transpile._api]. The
    construct-matrix test enforces the contract at the same boundary
    the production helper documents, then delegates body emission
    to the Lower + Renderer pipeline.
    """
    unsupported_for(
        f"qvr-{target}", module, allow=_SUPPORT_TIER[target]
    )
    return transpile(module, target=target)


@pytest.mark.parametrize(
    "fixture", _construct_fixtures(), ids=lambda f: f"{f.category}/{f.name}"
)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_construct_backend_cell(
    fixture: _load.Fixture, backend: str
) -> None:
    """One cell of the (construct × backend) matrix."""
    expected_bad = _expected_unsupported_kinds(fixture, backend)
    module = parse(fixture.source)

    if expected_bad:
        with pytest.raises(UnsupportedConstruct) as exc_info:
            _transpile_with_tier_check(module, target=backend)
        # The error must name at least one of the kinds we predicted.
        reported = set(exc_info.value.kinds)
        overlap = reported & expected_bad
        assert overlap, (
            f"backend {backend!r} on {fixture.name!r}: expected "
            f"UnsupportedConstruct mentioning {expected_bad}, got "
            f"{reported}"
        )
        return

    cell = (backend, fixture.category, fixture.name)
    expected_orthogonal = _EXPECTED_ORTHOGONAL_RAISES.get(cell)
    if expected_orthogonal is not None:
        with pytest.raises(UnsupportedConstruct) as exc_info:
            _transpile_with_tier_check(module, target=backend)
        kinds = exc_info.value.kinds
        assert any(k.startswith(expected_orthogonal) for k in kinds), (
            f"{backend!r} on {fixture.category}/{fixture.name!r}: "
            f"expected an orthogonal-concern raise with kind prefix "
            f"{expected_orthogonal!r}, got kinds={kinds!r}. Either "
            f"the renderer changed (update the entry in "
            f"`_EXPECTED_ORTHOGONAL_RAISES`) or a different gap fired "
            f"(fix the renderer)."
        )
        return

    output = _transpile_with_tier_check(module, target=backend)
    assert output, (
        f"backend {backend!r} on {fixture.name!r}: transpile "
        f"returned empty bytes"
    )
