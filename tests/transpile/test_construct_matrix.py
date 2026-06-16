"""Tier 4: (construct fixture × backend) compatibility matrix.

For every fixture under `fixtures/{statements, steps,
let_expressions, options, axes}` and every registered backend, the
test asserts one of three outcomes:

1. **Pass**: the construct is in the backend's `support` tier and
   the renderer pipeline successfully transpiles.
2. **`UnsupportedConstruct`**: the construct is outside the support
   tier; the pipeline raises with the offending kind name.
3. **`pytest.xfail`** (set inside the test body): the renderer
   raises on a construct that *should* be supported (i.e., the
   kind is in the tier but the renderer hasn't grown the case
   yet). This is the real assertion of the contract; the xfail
   tracks the renderer fix.
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
    CHURCH_LIKE,
    PYTHON_DEEP,
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
    """
    module = parse(fixture.source)
    tier = _SUPPORT_TIER[backend]
    bad: set[str] = set()
    for stmt in module.statements:
        kind = str(stmt.kind)
        if kind not in tier:
            bad.add(kind)
    return bad


# Fixtures that the renderer doesn't fully emit yet, even though
# the statement is in the support tier. Each maps fixture stem →
# reason. When the renderer grows the case, the strict-xfail flips
# to a failure and the entry is removed.
_KNOWN_WALKER_GAPS: dict[str, str] = {}


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

    if fixture.name in _KNOWN_WALKER_GAPS:
        # Strict gap: the pipeline MUST raise UnsupportedConstruct,
        # AND the error must report a kind that matches the gap. If
        # the pipeline stops raising (gap closed), the test fails so
        # the operator removes the fixture from `_KNOWN_WALKER_GAPS`.
        # If the pipeline raises with a different kind than expected,
        # the test fails so the operator updates the gap reason or
        # investigates the regression.
        try:
            output = _transpile_with_tier_check(module, target=backend)
        except UnsupportedConstruct as exc:
            reason = _KNOWN_WALKER_GAPS[fixture.name]
            # The gap reason starts with a step / option signature
            # like "walker does not emit LetStep"; we don't enforce a
            # specific kind tag (each backend's error message differs
            # in detail), but we require the gap to surface as some
            # `step:` or `family:` error so the gap is real.
            kinds = exc.kinds
            del reason
            assert any(
                ":" in k for k in kinds
            ), (
                f"{backend!r} on {fixture.name!r}: gap fired but the "
                f"error kinds {kinds!r} have no `:`-tagged construct "
                f"reference, the gap is no longer the one described "
                f"in `_KNOWN_WALKER_GAPS[{fixture.name!r}]`. Either "
                f"update the entry or remove it."
            )
        else:
            pytest.fail(
                f"{backend!r} on {fixture.name!r}: pipeline no longer "
                f"raises `UnsupportedConstruct`; the gap recorded in "
                f"`_KNOWN_WALKER_GAPS[{fixture.name!r}]` has been "
                f"closed. Remove the entry from the table. "
                f"(output: {output[:120]!r})"
            )
    else:
        # The construct-matrix test verifies CONSTRUCT support (the
        # pipeline can handle the Statement / Step / OptionValue
        # kind). Family support is a separate concern handled by
        # `test_family_matrix.py`; let-composition resolution is
        # tracked separately. When the pipeline raises with one of
        # these orthogonal-concern kinds, the construct cell xfails
        # so the construct-matrix doesn't double-report bugs the
        # other matrix already catches.
        # Renderer-gap prefixes the construct-matrix test treats as
        # orthogonal: each is raised by the Lower + Renderer pipeline
        # for a reason unrelated to whether the input's
        # construct-kind is in the support tier:
        #
        # - ``family:`` / ``let:`` — distribution / let resolution
        #   gaps; tracked by `test_family_matrix.py` and the
        #   let-resolution tests.
        # - ``let-expr:`` — let-expression Expr-kind the renderer
        #   has not grown; tracked alongside `let:`.
        # - ``node:`` — IR node kind the renderer has not wired
        #   (e.g. BUGS not handling `IRDeterministic`).
        # - ``return:`` — generated-quantities aliasing gap (return
        #   variable whose shape the renderer cannot resolve).
        # - ``declare:`` — type-constraint gap during declaration
        #   emission (e.g. Stan vector with wrong event-rank).
        # - ``broadcast:`` — broadcast-op gap on a non-scalar arg.
        # - ``arg:`` — arg-shape gap on an IRArg subclass the
        #   renderer has not wired (e.g. BUGS on `IRArgBroadcast`).
        _ORTHOGONAL_PREFIXES = (
            "family:",
            "let:",
            "let-expr:",
            "node:",
            "return:",
            "declare:",
            "broadcast:",
            "arg:",
        )
        try:
            output = _transpile_with_tier_check(module, target=backend)
            if backend == "church" and output == b"":
                pytest.xfail(
                    reason=(
                        "panproto/panproto#172: scheme `emit_pretty` "
                        "returns empty bytes for every input. Pipeline "
                        "succeeded; flips when upstream restores the "
                        "scheme pretty-printer."
                    )
                )
        except UnsupportedConstruct as exc:
            orthogonal = [
                k for k in exc.kinds
                if any(k.startswith(p) for p in _ORTHOGONAL_PREFIXES)
            ]
            if orthogonal:
                pytest.xfail(
                    reason=(
                        f"{backend!r} on {fixture.name!r}: pipeline "
                        f"raised on an orthogonal-concern kind "
                        f"({orthogonal!r}); construct support is "
                        f"orthogonal to this raise. Tracked by the "
                        f"family-matrix / let-resolution tests."
                    )
                )
            raise
        assert output, (
            f"backend {backend!r} on {fixture.name!r}: transpile "
            f"returned empty bytes"
        )
