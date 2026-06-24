"""Mutation tests for the IR-to-renderer contract.

For each known-good (fixture, backend) cell, mutate the lowered
[`IRProgram`][quivers.transpile.ir.IRProgram] in a way that flips a
semantic field (family name, arg order, plate batch_dims, observed
flag, sample/observe swap, etc.) and assert that the renderer either:

* raises [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
  (the mutation produces an unrepresentable shape, which the
  renderer must refuse rather than emit silently), OR
* emits source bytes that, when re-parsed through the target's
  syntax check, differ from the unmutated baseline.

If the mutated emit is **byte-identical** to the unmutated emit, the
renderer is ignoring the mutated field; that's a real bug because the
field carries semantics the user can observe (a Beta(2,5) call where
the user wrote Beta(5,2) is not the same model).

The assertion catches: silent field-drop in a walker, accidental
hard-coding of an arg index, copy-paste between similar code paths
that scrambles the dispatch, an unused-attr import that hides a real
read.

Run only against backends with no Docker dependency (target's emit is
the test surface). The numeric-equivalence tier covers the cross-
backend correctness contract; this tier covers the
field-by-field IR-renderer interface.
"""

from __future__ import annotations

from importlib import import_module

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.transpile._pipeline import EmitPretty
from quivers.transpile.ir import (
    IRArgNumber,
    IRObserve,
    IRProgram,
    IRSample,
    Plate,
)
from quivers.transpile.lower import Lower


_FIXTURE_SOURCES: dict[str, str] = {
    "beta_bernoulli": """object Obs : FinSet 30
program beta_bernoulli : Obs -> Obs
    sample theta <- Beta(2.0, 5.0)
    observe y : Obs <- Bernoulli(theta)
    return theta
export beta_bernoulli
""",
    "bayes_linear_regression": """object Obs : FinSet 60
program bayes_linear_regression : Obs -> Obs
    sample a <- Normal(0.0, 1.0)
    sample b <- Normal(0.0, 1.0)
    let mu = a + b * x_design
    observe y : Obs <- Normal(mu, 0.3)
    return mu
export bayes_linear_regression
""",
}


# Backends with target_protocol bindings present. Mutation tests
# don't need Docker (they compare emitted bytes), so every backend
# is in scope. WebPPL gallery quirks are excluded per-fixture only
# when they would `UnsupportedConstruct` on the unmutated baseline.
_BACKENDS: tuple[str, ...] = (
    "stan", "numpyro", "pyro", "pymc", "edward2",
    "turing", "gen", "webppl",
)


def _baseline_emit(fixture_source: str, backend: str) -> bytes | None:
    """Transpile `fixture_source` to `backend`; return the emitted
    bytes, or `None` if the cell is a known unsupported combination
    (in which case the mutation test is vacuous and skipped via
    `pytest.xfail`).
    """
    module = parse(fixture_source)
    try:
        return transpile(module, target=backend)
    except UnsupportedConstruct:
        return None


def _lower(fixture_source: str) -> IRProgram:
    return Lower().forward(parse(fixture_source))


def _mutate_swap_beta_args(ir: IRProgram) -> IRProgram:
    """Swap the first two positional args of the first
    [`IRSample`][quivers.transpile.ir.IRSample] whose call site has
    two numeric args. For asymmetric families (Beta, Normal, Gamma,
    LogNormal, ...) the swap produces a distinct distribution; any
    backend that emits the right call site must reflect it. Named
    `swap_beta_args` for legacy reasons but applies to every 2-arg
    family with numeric literals at the first two positions."""
    new_body = []
    swapped = False
    for node in ir.body:
        if (
            isinstance(node, IRSample)
            and len(node.args) >= 2
            and isinstance(node.args[0], IRArgNumber)
            and isinstance(node.args[1], IRArgNumber)
            and not swapped
        ):
            new_args = (
                node.args[1],
                node.args[0],
                *node.args[2:],
            )
            new_body.append(IRSample(
                name=node.name,
                family=node.family,
                args=new_args,
                arg_names=node.arg_names,
                constraint=node.constraint,
                plate=node.plate,
            ))
            swapped = True
            continue
        new_body.append(node)
    if not swapped:
        return ir
    return IRProgram(
        name=ir.name, inputs=ir.inputs, body=tuple(new_body),
        cards=ir.cards,
    )


def _mutate_normal_to_cauchy(ir: IRProgram) -> IRProgram:
    """Rename the first sample / observe to a same-arity, same-support
    alternative family. Normal -> Cauchy and Beta -> Kumaraswamy are
    the canonical pairs; any backend that emits the family name must
    reflect the rename. Named `normal_to_cauchy` for legacy reasons
    but applies to every renameable family pair."""
    pairs = {
        "Normal": "Cauchy",
        "Beta": "Kumaraswamy",
        "Gamma": "InverseGamma",
        "Bernoulli": "ContinuousBernoulli",
        "Poisson": "Geometric",
    }
    new_body = []
    changed = False
    for node in ir.body:
        if isinstance(node, IRSample) and node.family in pairs and not changed:
            new_body.append(IRSample(
                name=node.name,
                family=pairs[node.family],
                args=node.args,
                arg_names=node.arg_names,
                constraint=node.constraint,
                plate=node.plate,
            ))
            changed = True
            continue
        if isinstance(node, IRObserve) and node.family in pairs and not changed:
            new_body.append(IRObserve(
                name=node.name,
                family=pairs[node.family],
                args=node.args,
                arg_names=node.arg_names,
                constraint=node.constraint,
                plate=node.plate,
                via=node.via,
            ))
            changed = True
            continue
        new_body.append(node)
    if not changed:
        return ir
    return IRProgram(
        name=ir.name, inputs=ir.inputs, body=tuple(new_body),
        cards=ir.cards,
    )


def _mutate_drop_observe(ir: IRProgram) -> IRProgram:
    """Remove the first IRObserve. Any backend that emits the
    observation must reflect the drop (the call site disappears)."""
    new_body = []
    dropped = False
    for node in ir.body:
        if isinstance(node, IRObserve) and not dropped:
            dropped = True
            continue
        new_body.append(node)
    if not dropped:
        return ir
    return IRProgram(
        name=ir.name, inputs=ir.inputs, body=tuple(new_body),
        cards=ir.cards,
    )


def _mutate_drop_sample(ir: IRProgram) -> IRProgram:
    """Remove the LAST [`IRSample`][quivers.transpile.ir.IRSample]. The
    dropped sample's name may be referenced by a later `let` /
    `observe` / `return`; in that case the renderer either raises
    [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
    (because an arg references an undeclared name) or emits source
    that still differs from the baseline by the missing sample call
    site. Either outcome counts as `the mutation was observed`."""
    body = list(ir.body)
    for i in range(len(body) - 1, -1, -1):
        if isinstance(body[i], IRSample):
            body.pop(i)
            return IRProgram(
                name=ir.name, inputs=ir.inputs, body=tuple(body),
                cards=ir.cards,
            )
    return ir


def _mutate_flip_plate_dims(ir: IRProgram) -> IRProgram:
    """Empty the first non-scalar plate's batch_dims (changes a
    plated observe into a scalar observe). Any backend that emits
    the plate loop must drop the loop."""
    new_body = []
    flipped = False
    for node in ir.body:
        if (
            isinstance(node, IRObserve)
            and node.plate.batch_dims
            and not flipped
        ):
            new_body.append(IRObserve(
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                constraint=node.constraint,
                plate=Plate(event_dims=node.plate.event_dims, batch_dims=()),
                via=node.via,
            ))
            flipped = True
            continue
        new_body.append(node)
    if not flipped:
        return ir
    return IRProgram(
        name=ir.name, inputs=ir.inputs, body=tuple(new_body),
        cards=ir.cards,
    )


_MUTATIONS: dict[str, object] = {
    "swap_beta_args": _mutate_swap_beta_args,
    "normal_to_cauchy": _mutate_normal_to_cauchy,
    "drop_observe": _mutate_drop_observe,
    "drop_sample": _mutate_drop_sample,
    "flip_plate_dims": _mutate_flip_plate_dims,
}


#: ``(backend, mutation_name)`` cells where the renderer legitimately
#: produces identical bytes for the mutated and unmutated IR because
#: the backend uses array-broadcast semantics that doesn't need the
#: mutated field directly:
#:
#: * ``edward2`` lowers Bernoulli/Normal observations through
#:   ``edward2.Distribution(...)`` whose runtime shape comes from the
#:   ``y`` data tensor rather than from a per-plate loop in the
#:   emitted source. Flipping `IRObserve.plate.batch_dims` doesn't
#:   change the emit because the emit doesn't refer to the plate
#:   shape at the call site.
#:
#: * ``turing`` lowers plated observations through
#:   ``product_distribution(Family.(args))``; the broadcast `.`
#:   notation accepts any-length args at runtime. Flipping the IR's
#:   plate.batch_dims doesn't change the emit, only the runtime data
#:   shape.
#:
#: These cells are tracked but xfailed. The mutation test still has
#: value: it catches any new renderer that *should* read the field
#: but doesn't.
_BROADCAST_INSENSITIVE_CELLS: frozenset[tuple[str, str]] = frozenset({
    ("edward2", "flip_plate_dims"),
    ("turing", "flip_plate_dims"),
})


#: ``(mutation_name, fixture_name)`` pairs where the mutator's
#: selector criterion cannot match any node in the fixture's lowered
#: IR (so the mutation is a structural no-op for that fixture). The
#: value is a fixture-specific reason printed in the `pytest.xfail`
#: call, replacing the generic `no matching IR node found` message
#: that buries the actual cause.
_MUTATION_INAPPLICABLE: dict[tuple[str, str], str] = {}


def _render(ir: IRProgram, backend: str) -> bytes | None:
    """Run the renderer for `backend` against an arbitrary
    [`IRProgram`][quivers.transpile.ir.IRProgram]. Returns the
    emitted bytes or `None` on `UnsupportedConstruct` (the mutation
    produces a shape the backend refuses, which counts as 'the
    mutation was observed')."""
    mod = _import_renderer_module(backend)
    renderer_cls = next(
        cls for name, cls in vars(mod).items()
        if name.endswith("Renderer") and isinstance(cls, type)
    )
    renderer = renderer_cls()
    try:
        schema = renderer.render(ir)
    except UnsupportedConstruct:
        return None
    grammar = renderer.target_protocol().name
    return EmitPretty(grammar).forward(schema)


def _import_renderer_module(backend: str):
    """Import the per-backend renderer module by name. Wrapped in a
    helper to keep the dynamic import isolated from the call site
    (and to keep the `from importlib import import_module` cost
    paid once per pytest collection rather than per test)."""
    return import_module(f"quivers.transpile.renderers.{backend}")


@pytest.mark.parametrize("fixture_name", sorted(_FIXTURE_SOURCES))
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("mutation_name", sorted(_MUTATIONS))
def test_mutation_changes_emit(
    fixture_name: str, backend: str, mutation_name: str
) -> None:
    """For each (fixture, backend, mutation) triple, assert the
    mutated IR's emit either:

    * raises `UnsupportedConstruct` (the renderer refused the new
      shape), OR
    * differs in bytes from the unmutated baseline.

    A mutated emit that's byte-identical to the baseline means the
    renderer silently dropped the mutated field; that's a contract
    violation.
    """
    src = _FIXTURE_SOURCES[fixture_name]
    baseline_bytes = _baseline_emit(src, backend)
    if baseline_bytes is None:
        pytest.xfail(
            f"{backend} cannot transpile baseline {fixture_name!r}; "
            f"mutation test vacuous"
        )

    ir_baseline = _lower(src)
    mutator = _MUTATIONS[mutation_name]
    ir_mutated = mutator(ir_baseline)
    if ir_mutated is ir_baseline:
        reason = _MUTATION_INAPPLICABLE.get(
            (mutation_name, fixture_name),
            f"mutation {mutation_name!r} selector matched no node in "
            f"{fixture_name!r}",
        )
        pytest.xfail(reason)

    try:
        mutated_bytes = _render(ir_mutated, backend)
    except UnsupportedConstruct:
        mutated_bytes = None

    if mutated_bytes is None:
        return  # Renderer refused the mutated shape; mutation was observed.

    if mutated_bytes == baseline_bytes:
        if (backend, mutation_name) in _BROADCAST_INSENSITIVE_CELLS:
            pytest.xfail(
                f"{backend} legitimately produces identical bytes for "
                f"`{mutation_name}` because it uses array-broadcast "
                f"semantics that doesn't reference the mutated IR field "
                f"in the emit; see _BROADCAST_INSENSITIVE_CELLS"
            )
        pytest.fail(
            f"{backend} renderer silently dropped the mutated field "
            f"(mutation={mutation_name!r}, fixture={fixture_name!r}); "
            f"both the baseline and the mutated IR produced identical "
            f"emit bytes:\n{baseline_bytes.decode('utf-8', errors='replace')}"
        )
