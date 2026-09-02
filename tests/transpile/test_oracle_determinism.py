"""Tier-4 guarantee: the QVR reference oracle is bitwise deterministic.

The numeric-equivalence tier compares a transpiled backend program
against the in-process QVR oracle and asserts that the difference
`qvr_i - backend_i` is the same constant across the point set. That
contract presupposes a fact it never checks: that `qvr_i` is a
*number*, reproducible on demand, rather than one draw from an
estimator of the joint. Against a resampled reference the comparison
measures Monte-Carlo noise and its verdict is an accident of the
generator state.

[`assert_all_latents_clamped`][tests.transpile.probes.qvr.assert_all_latents_clamped]
was the only thing standing behind that presupposition, and it cannot
carry it. The guard reads
[`Trace.latent_sites`][quivers.effects.trace_types.Trace], which
enumerates the entries of `Trace.sites`, and a latent internal to a
`SampledComposition` is recorded at no site: an RNN scan cell's
per-step gates, a decoder's attention chain, every intermediate a
composition integrates over without opening a site. Let any of those
be integrated by drawing rather than by a rule, and the guard reports
"no free latents" on a joint that is being redrawn on every call.
That is a structural proxy standing in for a behavioural property,
and the proxy does not hold.

This module tests the property itself. Determinism is invariance of
the computation to the generator state, so the oracle is evaluated
once per entry of a fixed multi-seed sweep, each run under a freshly
seeded global torch RNG, and every joint (with every per-site
log-density summand feeding it) is required to agree bit for bit.
A hidden latent breaks that equality wherever it lives and whatever
it is called, because a redrawn quantity moves the bits it feeds.

Five things are asserted, in the order a reader needs them:

1. **The guarantee.** Every gallery example that scores a joint is
   bitwise deterministic at *every* point of the multi-point set, not
   merely at ground truth. The distinction is load-bearing: a latent
   can be clamped by the ground-truth payload and left free by a
   perturbed one, and the equivalence tier evaluates all six.
2. **The blind spot, measured.** Every example in
   `_SKIP_QVR_INCOMPATIBLE` draws a latent from a
   `SampledComposition` whose intermediates are recorded at no site,
   which is precisely the shape `Trace.latent_sites` cannot see.
   Their joints are nonetheless bit-identical across the sweep at
   every point, because the composition integrates those
   intermediates against a deterministic rule rather than by drawing
   them. This tier asserts that measurement instead of the prose the
   registry carries: one of the eight starting to redraw fails here.
   Passing here is not a route out of `_SKIP_QVR_INCOMPATIBLE`,
   though, and the assertion is deliberately narrow for that reason.
   Determinism says the oracle returns a number rather than a draw;
   it says nothing about whether the number is the model's density.
   What keeps the eight exempt is measured against the density, in
   `test_oracle_reference_strength.py`: five report a value the
   quadrature rule chose, and three report a likelihood counted once
   per plate row with no factor at all for the latent the program
   declares. Both defects are perfectly reproducible, which is
   precisely why this tier cannot see them.
3. **The contrast.** The structural guard passing where the
   behavioural guard rejects is the whole reason the behavioural one
   is worth its cost, and no gallery example exhibits that gap any
   more (item 2 is the measurement that says so). The contrast is
   therefore drawn against a program built for the purpose: a
   two-step `MonadicProgram` whose likelihood marginalises an
   internal latent by *drawing* it instead of integrating it against
   a rule, which is the defect class the guard exists to catch. Both
   of its recorded sites are clamped, so the structural guard is
   silent; its joint moves by nats between two seeds, so the
   behavioural guard fires. Running both over the same pair of
   traces, produced by the same tracing machinery the gallery uses,
   is what makes the contrast evidence rather than assertion.
4. **The strength.** Each way the guard could be weakened back into
   something that reads the same and asserts less is pinned by the
   case it would start admitting: a one-ULP drift (which a tolerance
   accepts), a summand that moves while the joint stands still (which
   a joint-only comparison accepts), and a site recorded under one
   seed alone (which a comparison over shared names accepts).
5. **The guard-rails.** A determinism sweep degenerates the moment
   its seed set stops containing two distinct seeds, so the sweep
   rejects such a request rather than passing vacuously; and the
   probe restores the caller's RNG state, so nothing it seeds can
   leak into a neighbouring test.

Comparison is on raw bytes throughout. A tolerance here would be a
second, quieter threshold under the equivalence tolerance the tier
actually asserts on, and a resampled latent whose effect happened to
land inside it would pass unnoticed. Bytes also settle the two edge
cases float equality gets wrong in opposite directions: two `nan`
results with the same payload agree, and two floats a tolerance would
merge do not.
"""

from __future__ import annotations

import functools
import math
import pathlib

import pytest
import torch
from torch import distributions as td

from quivers.continuous.families import ConditionalNormal
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet
from quivers.effects.trace_types import SampleSite, Trace
from tests.transpile import _gallery_data
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import (
    DETERMINISM_SEEDS,
    QvrProbe,
    assert_all_latents_clamped,
    assert_reference_joint_deterministic,
    reference_traces,
)
from tests.transpile.test_gallery_numeric_equivalence import (
    _SKIP_DATASET_LOAD_FAILED,
    _SKIP_QVR_INCOMPATIBLE,
)


# Seeds the per-model sweep traces under. Wider than the two the probe
# path uses on every evaluation, because this tier runs once per model
# rather than once per (backend, model) cell and can afford the extra
# passes. The width buys protection against a coincidence: a free
# *discrete* latent over a small support can draw the same value under
# two generator states by luck, and each additional seed multiplies
# that luck by the collision probability again. The values are fixed
# constants, so the sweep is reproducible and carries no wall-clock or
# unseeded input.
_SWEEP_SEEDS: tuple[int, ...] = (
    0,
    1,
    0xA5A5A5A5,
    0x5EED0001,
    0x7FFFFFFF,
)

# ---------------------------------------------------------------------
# The constructed contrast.
#
# No gallery example is left whose joint the structural guard passes
# and the behavioural guard rejects, so the pair of guards is run
# against a program written to have exactly that shape. Nothing below
# stands in for a trace: it is a real `MonadicProgram`, walked by the
# same `trace` the gallery walks, recording real sites. The one thing
# built for the occasion is the likelihood kernel, which integrates
# its internal latent by drawing it rather than against a rule.
# ---------------------------------------------------------------------

_CONSTRUCTED_FIXTURE = "drawn_marginal"
"""Name the constructed program is reported under.

Deliberately not a gallery stem: a reader who sees it in a failure
message should not go looking for a `.qvr` file."""

_CONSTRUCTED_SEED = 0xD8A21
"""Seed the constructed program's prior weights are drawn under.

`ConditionalNormal` initialises an affine head at construction, which
consumes the global RNG. Seeding around that, and restoring the
caller's state afterwards, keeps the fixture reproducible run to run
and stops building it from shifting any draw the surrounding test
makes, which is the same discipline
[`test_probe_evaluation_leaves_the_global_rng_state_untouched`][tests.transpile.test_oracle_determinism.test_probe_evaluation_leaves_the_global_rng_state_untouched]
holds the probe to."""

_DRAWN_MARGINAL_DRAWS = 16
"""Draws the constructed likelihood marginalises its latent over.

Small on purpose. A Monte-Carlo marginal converges as the draw count
grows, so a large count would make the disagreement between two
generator states shrink toward round-off and the contrast would rest
on a coincidence of magnitudes rather than on the redraw."""

_UNIT = FinSet(name="Unit", cardinality=1)
_R1 = Euclidean(name="R1", dim=1)


class _DrawnMarginal(ContinuousMorphism):
    """A kernel that integrates its internal latent by drawing it.

    Denotes

        p(y | x) = integral N(u; x, 1) N(y; u, 1) du

    and evaluates it as the Monte-Carlo average over `n_draws` draws
    of `u`, which is the estimator shape the structural guard cannot
    see: `u` is internal to the kernel, so it is recorded at no site,
    appears in no `Trace.latent_sites`, and can be clamped by no
    point. Every call therefore returns a different number for the
    same arguments.

    A plain `ContinuousMorphism` subclass rather than a `dx.Model`: it
    is a `torch.nn.Module` participating in a compiled program, not a
    structured value.
    """

    def __init__(self, n_draws: int = _DRAWN_MARGINAL_DRAWS) -> None:
        super().__init__(_R1, _R1)
        self.n_draws = n_draws

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Ancestral draw: `u ~ N(x, 1)`, then `y ~ N(u, 1)`."""
        shape = torch.Size((*sample_shape, *x.shape))
        return x.expand(shape) + torch.randn(shape) + torch.randn(shape)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """`log (1/n) sum_i N(y; u_i, 1)` over `u_i ~ N(x, 1)`."""
        u = x.unsqueeze(0) + torch.randn((self.n_draws, *x.shape))
        per_draw = td.Normal(u, 1.0).log_prob(y.unsqueeze(0)).sum(-1)
        return torch.logsumexp(per_draw, dim=0) - math.log(self.n_draws)


def _drawn_marginal_program() -> MonadicProgram:
    """A two-step program whose likelihood redraws inside `log_prob`.

        sample  z ~ prior()
        observe y ~ drawn_marginal(z)

    Both sites are recorded and both are clamped by the point, so
    `Trace.latent_sites` is empty and the structural guard has nothing
    to report. The quantity that moves lives one level below either
    site, inside the likelihood's own marginalisation.
    """
    saved_rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(_CONSTRUCTED_SEED)
        prior = ConditionalNormal(_UNIT, _R1)
        likelihood = _DrawnMarginal()
    finally:
        torch.set_rng_state(saved_rng_state)
    return MonadicProgram(
        _UNIT,
        _R1,
        steps=[
            (("z",), prior, None),
            (("y",), likelihood, ("z",), True),
        ],
        return_vars=("y",),
    )


_CONSTRUCTED_POINT = Point(params={"z": [0.25]}, data={"y": [1.5]})
"""Ground truth for the constructed program: both sites clamped."""

_CONSTRUCTED_INPUT = torch.zeros(1, 1)
"""The one-row bracket token a `FinSet 1` domain reads."""

# Model the RNG-neutrality assertions run against. Small, fully
# clamped, and free of template instantiation, so the assertion is
# about the probe's RNG discipline rather than about the example.
_RNG_NEUTRALITY_EXAMPLE = "bayesian_regression"


def _scored_examples() -> list[pathlib.Path]:
    """Gallery examples the oracle is expected to score to a joint."""
    return [
        example
        for example in _gallery_data.gallery_examples_with_data()
        if example.stem not in _SKIP_DATASET_LOAD_FAILED
        and example.stem not in _SKIP_QVR_INCOMPATIBLE
    ]


def _composition_marginalised_examples() -> list[pathlib.Path]:
    """Gallery examples whose oracle integrates a composition's insides.

    `_SKIP_QVR_INCOMPATIBLE` is the registry of examples that draw a
    latent from a `SampledComposition`: the four recurrent language
    models, the transformer, the deep Markov state-space model, the
    encoder-decoder, and the variational autoencoder. Their
    intermediates are recorded at no trace site, so they are exactly
    the programs `Trace.latent_sites` cannot speak about, and the
    behavioural guard is the only thing that can.
    """
    return [
        example
        for example in _gallery_data.gallery_examples_with_data()
        if example.stem in _SKIP_QVR_INCOMPATIBLE
    ]


@functools.cache
def _dataset(example: pathlib.Path) -> _gallery_data.GalleryDataset:
    """Load one example's synthetic data, once per session.

    The `.md` snippet costs a full `exec` plus a compile, and three
    tests in this module walk the same models; caching keeps the tier
    proportional to the number of examples rather than to the number
    of assertions over them.
    """
    dataset = _gallery_data.load_gallery_data(example)
    if dataset is None:
        raise AssertionError(
            f"{example.stem!r}: `load_gallery_data` returned None even "
            f"though the example is not in `_SKIP_DATASET_LOAD_FAILED`. "
            f"Add the example's stem to that registry, or fix the `.md` "
            f"snippet so it produces an `observations` dict."
        )
    if dataset.monadic is None:
        raise AssertionError(
            f"{example.stem!r}: the synthetic-data snippet bound no "
            f"compiled program, so there is no joint to check for "
            f"determinism."
        )
    return dataset


@functools.cache
def _points(example: pathlib.Path) -> tuple[Point, ...]:
    """The multi-point evaluation set for one example, once per session."""
    return tuple(_gallery_data.points_from_dataset(_dataset(example)))


def _bit_pattern(tensor: torch.Tensor) -> tuple[int, ...]:
    """Raw byte sequence of `tensor`'s storage, as a hashable tuple.

    Reinterpreting through `uint8` compares the values exactly: two
    `nan` results with the same payload count as agreeing, where
    `torch.equal` would call them different, and two distinct floats
    that a tolerance would call equal count as differing. The `clone`
    makes the dtype view total by dropping any storage offset the
    tensor inherited from a larger buffer.
    """
    flat = tensor.detach().cpu().reshape(-1).clone()
    return tuple(int(byte) for byte in flat.view(torch.uint8).tolist())


def _joint_of(trace_result: Trace, context: str, seed: int) -> torch.Tensor:
    """The trace's `log_joint`, or a failure naming the model and seed."""
    joint = trace_result.log_joint
    assert joint is not None, (
        f"{context}: the trace under global RNG seed {seed} returned a "
        f"None log_joint, so the reference carries no density at all."
    )
    return joint


def _sweep(
    example: pathlib.Path,
    point: Point,
    seeds: tuple[int, ...] = _SWEEP_SEEDS,
) -> list[Trace]:
    """Trace one example at one point, once per seed."""
    dataset = _dataset(example)
    assert dataset.monadic is not None
    return reference_traces(
        dataset.monadic,
        point,
        x_input=dataset.x_input,
        observations=_gallery_data.observations_for_point(dataset, point),
        seeds=seeds,
    )


@pytest.mark.parametrize(
    "example", _scored_examples(), ids=lambda p: p.stem
)
def test_gallery_reference_joint_is_bitwise_deterministic(
    example: pathlib.Path,
) -> None:
    """Every scored example's joint is bit-identical across the seed sweep.

    Asserted at every point of the multi-point set rather than at
    ground truth alone. A latent clamped by the ground-truth payload
    can be left free by a perturbation (a site the snippet binds under
    one spelling and the perturber rewrites under another), and the
    equivalence tier scores all six points, so a reference that is
    deterministic only at point 0 still poisons five of the six
    differences the constant-spread check reads.

    Per-site `log_prob` is compared alongside the joint, because the
    joint is their sum and two moving summands can cancel at one point
    while diverging at the next.
    """
    points = _points(example)
    labels = _gallery_data.perturbation_labels(len(points))
    assert len(points) >= 2, (
        f"{example.stem!r}: {len(points)} point(s); the determinism "
        f"sweep is meant to cover the same set the constant-spread "
        f"check reads, which needs at least two."
    )

    for index, point in enumerate(points):
        context = f"{example.stem!r} at point {index} ({labels[index]})"
        traces = _sweep(example, point)
        joints = [
            _joint_of(trace_result, context, seed)
            for trace_result, seed in zip(traces, _SWEEP_SEEDS)
        ]
        patterns = {_bit_pattern(joint) for joint in joints}
        totals = [float(joint.sum().item()) for joint in joints]
        assert len(patterns) == 1, (
            f"{context}: the reference joint is not deterministic. "
            f"Tracing under global torch RNG seeds {list(_SWEEP_SEEDS)} "
            f"produced {totals!r} (hex "
            f"{[total.hex() for total in totals]!r}), so some quantity "
            f"in this program is redrawn on every call and the value "
            f"the equivalence tier compares a backend against is a "
            f"sample rather than a density. Either a `sample` site "
            f"lost its ground-truth clamp, or a composition inside the "
            f"program marginalises by redrawing; neither is a "
            f"tolerable reference."
        )

        names = {
            frozenset(trace_result.sites) for trace_result in traces
        }
        assert len(names) == 1, (
            f"{context}: the set of recorded trace sites depends on the "
            f"generator state ({[sorted(entry) for entry in names]!r}), "
            f"so the program takes a different control-flow path under "
            f"a different seed."
        )
        for name in sorted(traces[0].sites):
            summands = [
                trace_result.sites[name].log_prob for trace_result in traces
            ]
            site_patterns = {
                _bit_pattern(summand) for summand in summands
            }
            site_totals = [
                float(summand.sum().item()) for summand in summands
            ]
            assert len(site_patterns) == 1, (
                f"{context}: site {name!r} contributes a different "
                f"log-density under different generator states "
                f"({site_totals!r}). The joint happens to agree at this "
                f"point, which means two moving summands cancelled "
                f"here; they need not cancel at the next point, and "
                f"neither summand is a density."
            )


@pytest.mark.parametrize(
    "example", _composition_marginalised_examples(), ids=lambda p: p.stem
)
def test_composition_marginalised_models_are_bitwise_deterministic(
    example: pathlib.Path,
) -> None:
    """A latent drawn from a composition still scores to a fixed number.

    These eight are the hardest cases the guard has: the quantities
    their joints integrate over are internal to a `SampledComposition`
    and are recorded at no site, so no point can clamp them and
    `assert_all_latents_clamped` has nothing to look at. Whether the
    joint is a number or a draw is decided entirely by how the
    composition performs that integral, which is exactly the question
    a structural check cannot reach and this one can.

    The answer is measured here rather than taken from
    `_SKIP_QVR_INCOMPATIBLE`'s prose, at every point of the set and
    across the full seed sweep, per site as well as per joint. An
    example whose composition starts drawing its intermediates fails
    here, which is what makes this the tier's live statement about the
    eight rather than a transcription of the registry.

    The probe is exercised alongside the raw sweep, and its reported
    float is required to match the swept joint bit for bit. The two
    run over *different* seed sets, so the agreement says something
    the sweep alone does not: the value is a property of the program
    and not of whichever seeds happened to be asked for.
    """
    points = _points(example)
    labels = _gallery_data.perturbation_labels(len(points))
    dataset = _dataset(example)
    probe = QvrProbe()
    scratch = pathlib.Path(__file__).parent

    for index, point in enumerate(points):
        context = f"{example.stem!r} at point {index} ({labels[index]})"
        traces = _sweep(example, point)
        joints = [
            _joint_of(trace_result, context, seed)
            for trace_result, seed in zip(traces, _SWEEP_SEEDS)
        ]
        totals = [float(joint.sum().item()) for joint in joints]
        assert len({_bit_pattern(joint) for joint in joints}) == 1, (
            f"{context}: the joint moved across global torch RNG seeds "
            f"{list(_SWEEP_SEEDS)}, producing {totals!r} (hex "
            f"{[total.hex() for total in totals]!r}). The composition "
            f"this program draws its latent from is integrating its "
            f"intermediates by drawing them, so the reference is a "
            f"draw from an estimator rather than a density, and "
            f"nothing here or in the equivalence tier can compare a "
            f"backend against it. No point can clamp the redrawn "
            f"quantity, because it is recorded at no site, so the "
            f"integral itself has to be a rule rather than a draw."
        )

        names = {frozenset(trace_result.sites) for trace_result in traces}
        assert len(names) == 1, (
            f"{context}: the set of recorded trace sites depends on the "
            f"generator state ({[sorted(entry) for entry in names]!r}), "
            f"so the program takes a different control-flow path under "
            f"a different seed."
        )
        for name in sorted(traces[0].sites):
            summands = [
                trace_result.sites[name].log_prob for trace_result in traces
            ]
            site_totals = [
                float(summand.sum().item()) for summand in summands
            ]
            assert len({_bit_pattern(summand) for summand in summands}) == 1, (
                f"{context}: site {name!r} contributes a different "
                f"log-density under different generator states "
                f"({site_totals!r}). The joint agrees here only because "
                f"two moving summands cancelled, and they need not "
                f"cancel at the next point."
            )

        result = probe.evaluate(
            example.read_bytes(),
            example.stem,
            [point],
            scratch=scratch,
            monadic=dataset.monadic,
            x_input=dataset.x_input,
            observations=_gallery_data.observations_for_point(
                dataset, point,
            ),
        )
        assert len(result.log_densities) == 1
        reported = result.log_densities[0]
        assert math.isfinite(reported), (
            f"{context}: the probe reported {reported!r}. A "
            f"non-finite reference is reproducible and still useless."
        )
        assert reported.hex() == totals[0].hex(), (
            f"{context}: the probe reports {reported!r} (hex "
            f"{reported.hex()}) where the sweep traced {totals[0]!r} "
            f"(hex {totals[0].hex()}). The two differ only in which "
            f"seeds they ran under, so the oracle's value depends on "
            f"the generator state after all."
        )


def _constructed_traces() -> list[Trace]:
    """Trace the constructed program once per entry of `DETERMINISM_SEEDS`."""
    return reference_traces(
        _drawn_marginal_program(),
        _CONSTRUCTED_POINT,
        x_input=_CONSTRUCTED_INPUT,
        seeds=DETERMINISM_SEEDS,
    )


def test_structural_guard_passes_where_behavioural_guard_rejects() -> None:
    """The two guards disagree on a program built to make them disagree,
    and that gap is the whole reason the behavioural one exists.

    Every site
    [`trace`][quivers.inference.trace.trace] records for
    [`_drawn_marginal_program`][tests.transpile.test_oracle_determinism._drawn_marginal_program]
    is clamped by the point, so `Trace.latent_sites` is empty and
    [`assert_all_latents_clamped`][tests.transpile.probes.qvr.assert_all_latents_clamped]
    reports a clean joint. Its likelihood integrates an internal
    latent by drawing it, and that latent is recorded at no site, so
    the joint moves by nats between two seeds and
    [`assert_reference_joint_deterministic`][tests.transpile.probes.qvr.assert_reference_joint_deterministic]
    rejects it. Running both guards over the *same* pair of traces is
    what makes the contrast evidence rather than assertion: nothing
    about the inputs differs between the silent guard and the firing
    one.

    The program is constructed rather than drawn from the gallery
    because no gallery example has this shape any longer;
    [`test_composition_marginalised_models_are_bitwise_deterministic`][tests.transpile.test_oracle_determinism.test_composition_marginalised_models_are_bitwise_deterministic]
    is the measurement that establishes it, and would fail the moment
    a gallery example reacquired it. What is constructed is one
    kernel; the program,
    the trace, the site records, and both guards are the same
    machinery the gallery runs through.

    The rejection is required to come from the *joint*, not merely
    from a moving summand, and the prior site is required to hold
    still. Together those say the disagreement is localised in the
    kernel that draws, rather than being a program that is unstable
    everywhere and would reject under any guard at all.
    """
    traces = _constructed_traces()
    first, second = traces[0], traces[1]

    assert sorted(first.sites) == ["y", "z"], (
        f"the constructed program recorded {sorted(first.sites)!r}; "
        f"the contrast needs the clamped prior site and the clamped "
        f"observation, and nothing else."
    )

    # The structural guard is silent: every recorded site is clamped,
    # so it has nothing to report.
    assert not first.latent_sites, (
        f"`Trace.latent_sites` is {sorted(first.latent_sites)!r}, so "
        f"the structural guard already rejects the constructed "
        f"program and it demonstrates nothing about the gap. The "
        f"point must clamp every recorded site."
    )
    assert_all_latents_clamped(first, _CONSTRUCTED_FIXTURE)
    assert_all_latents_clamped(second, _CONSTRUCTED_FIXTURE)

    # The behavioural guard, over those same two traces, fires.
    with pytest.raises(RuntimeError) as exc_info:
        assert_reference_joint_deterministic(
            traces, _CONSTRUCTED_FIXTURE, DETERMINISM_SEEDS,
        )

    message = str(exc_info.value)
    left = float(
        _joint_of(first, _CONSTRUCTED_FIXTURE, DETERMINISM_SEEDS[0])
        .sum()
        .item()
    )
    right = float(
        _joint_of(second, _CONSTRUCTED_FIXTURE, DETERMINISM_SEEDS[1])
        .sum()
        .item()
    )
    assert left != right, (
        f"the two joints agree ({left!r}), so the raise came from a "
        f"moving per-site summand rather than from the joint. That is "
        f"still a rejection, but this test claims the joint itself "
        f"moves."
    )
    assert _CONSTRUCTED_FIXTURE in message, (
        f"the rejection does not name the program: {message!r}."
    )
    assert repr(left) in message and repr(right) in message, (
        f"the rejection does not show both differing joints "
        f"({left!r}, {right!r}): {message!r}."
    )
    assert "'y'" in message, (
        f"the rejection does not name the site the drawn "
        f"marginalisation sits under: {message!r}."
    )

    prior_patterns = {
        _bit_pattern(trace_result.sites["z"].log_prob)
        for trace_result in traces
    }
    assert len(prior_patterns) == 1, (
        f"the prior site 'z' also moved between seeds, so the "
        f"constructed program is non-deterministic everywhere rather "
        f"than at the one kernel that draws, and the contrast says "
        f"less than it claims. Prior log-densities: "
        f"{[float(t.sites['z'].log_prob.sum().item()) for t in traces]!r}."
    )


def test_constructed_contrast_is_a_function_of_the_generator_state() -> None:
    """The constructed program's disagreement is about the seed alone.

    A program that returned a fresh number on every call for reasons
    *other* than the generator state (a clock, an address, an
    uninitialised buffer) would also make the behavioural guard fire,
    and would make it fire for a reason the guard does not claim to
    detect. Tracing the same seed pair twice and requiring both runs
    to reproduce their own joints bit for bit rules that out: the
    program is a deterministic function of the seed, and the contrast
    above is a statement about two different seeds.
    """
    first_pass = _constructed_traces()
    second_pass = _constructed_traces()

    for index, seed in enumerate(DETERMINISM_SEEDS):
        earlier = _joint_of(first_pass[index], _CONSTRUCTED_FIXTURE, seed)
        later = _joint_of(second_pass[index], _CONSTRUCTED_FIXTURE, seed)
        assert _bit_pattern(earlier) == _bit_pattern(later), (
            f"tracing the constructed program twice under seed {seed} "
            f"produced {float(earlier.sum().item())!r} and then "
            f"{float(later.sum().item())!r}. Its value depends on "
            f"something other than the global torch RNG, so it is the "
            f"wrong witness for a guard that varies the RNG alone."
        )


def _observed_site(name: str, log_prob: float) -> SampleSite:
    """One clamped, observed site carrying `log_prob`.

    `is_observed=True` and `is_deterministic=False` is the shape the
    structural guard treats as fully pinned: it appears in neither
    `Trace.latent_sites` nor the deterministic exclusion.
    """
    return SampleSite(
        name=name,
        morphism=None,
        value=torch.zeros(1),
        log_prob=torch.tensor([log_prob]),
        is_observed=True,
        is_deterministic=False,
    )


def _trace_with(joint: float, site_log_prob: float) -> Trace:
    """A single-site trace whose recorded site is fully clamped."""
    return Trace(
        sites={"y": _observed_site("y", site_log_prob)},
        output=torch.zeros(1),
        log_joint=torch.tensor([joint]),
    )


def test_structural_guard_admits_a_trace_the_behavioural_guard_rejects() -> None:
    """A hand-built pair pins the contrast at the function level.

    The gallery contrast above depends on two examples staying the way
    they are; this one depends on nothing. Two traces record the same
    single site, clamped and observed, so `Trace.latent_sites` is
    empty and the structural guard is silent. Their joints differ, so
    the behavioural guard rejects. That is exactly the failure mode
    `Trace.latent_sites` cannot see: a quantity that moves without
    being recorded at any site.
    """
    first = _trace_with(joint=-1.5, site_log_prob=-1.5)
    second = _trace_with(joint=-2.25, site_log_prob=-2.25)

    assert not first.latent_sites
    assert_all_latents_clamped(first, "handbuilt")
    assert_all_latents_clamped(second, "handbuilt")

    with pytest.raises(RuntimeError) as exc_info:
        assert_reference_joint_deterministic(
            [first, second], "handbuilt", DETERMINISM_SEEDS,
        )
    message = str(exc_info.value)
    assert "handbuilt" in message
    assert repr(-1.5) in message and repr(-2.25) in message
    assert "'y'" in message, (
        f"the rejection does not name the site whose log-density "
        f"moved: {message!r}"
    )


def test_behavioural_guard_rejects_a_joint_moving_under_frozen_sites() -> None:
    """A joint can move while every recorded summand stands still.

    `log_joint` carries contributions a `SampledComposition`'s
    marginalisation makes without opening a site, so per-site
    comparison alone would miss them. The guard compares the joint as
    well, and says so: with no moving site to name it points the
    reader at the composition rather than at a `sample` binder that
    does not exist.
    """
    first = _trace_with(joint=-10.0, site_log_prob=-4.0)
    second = _trace_with(joint=-11.5, site_log_prob=-4.0)

    with pytest.raises(RuntimeError) as exc_info:
        assert_reference_joint_deterministic(
            [first, second], "handbuilt", DETERMINISM_SEEDS,
        )
    message = str(exc_info.value)
    assert "[]" in message, (
        f"with no moving site the rejection should report an empty "
        f"list and blame the composition: {message!r}"
    )
    assert "composition" in message


def test_behavioural_guard_rejects_a_one_ulp_difference() -> None:
    """The comparison is bitwise, and a tolerance would not do.

    This is the mutation the guard has to survive to stay a
    determinism check: replacing the byte comparison with
    `torch.allclose` (or any tolerance at all) reads as the same
    assertion, passes every other test in this module, and quietly
    admits a resampled latent whose effect happens to be small. The
    two joints here differ by a single unit in the last place, so the
    tolerance version accepts and the byte version rejects, and the
    test asserts both halves of that so the contrast cannot be read as
    an accident.

    A one-ULP drift is not a rounding artefact to be forgiven. The
    reference is traced from identical inputs under two generator
    states; identical inputs through a deterministic computation
    return identical bits. Any movement at all means randomness
    entered, and a latent that moves the joint by one ULP at this
    point is free to move it by nats at the next.
    """
    joint = torch.tensor([-3.75])
    neighbour = torch.nextafter(joint, torch.zeros_like(joint))
    assert not torch.equal(joint, neighbour), (
        "torch.nextafter returned the same value, so this test has no "
        "one-ULP pair to assert on."
    )
    assert torch.allclose(joint, neighbour), (
        "the two joints are not within the default allclose tolerance, "
        "so they do not demonstrate what a tolerance-based comparison "
        "would wave through."
    )

    first = Trace(
        sites={"y": _observed_site("y", -3.75)},
        output=torch.zeros(1),
        log_joint=joint,
    )
    second = Trace(
        sites={"y": _observed_site("y", -3.75)},
        output=torch.zeros(1),
        log_joint=neighbour,
    )

    with pytest.raises(RuntimeError) as exc_info:
        assert_reference_joint_deterministic(
            [first, second], "handbuilt", DETERMINISM_SEEDS,
        )
    message = str(exc_info.value)
    left = float(joint.item())
    right = float(neighbour.item())
    assert left.hex() != right.hex()
    assert repr(left) in message and repr(right) in message, (
        f"the rejection does not show both joints ({left!r}, "
        f"{right!r}): {message!r}"
    )


def test_behavioural_guard_rejects_a_moving_site_under_a_frozen_joint() -> None:
    """A summand may move while the joint stands still, and that counts.

    Two redrawn per-site densities can cancel to the same total: the
    joint is their sum, and a sum is blind to a pair of equal and
    opposite shifts. Comparing the joint alone would therefore accept
    a program whose sites are being resampled, and it would accept it
    only at the points where the cancellation happens to hold. The
    guard compares every summand as well, so this pair is rejected and
    the moving site is named.
    """
    first = _trace_with(joint=-3.75, site_log_prob=-1.0)
    second = _trace_with(joint=-3.75, site_log_prob=-2.0)
    first_joint = first.log_joint
    second_joint = second.log_joint
    assert first_joint is not None and second_joint is not None
    assert torch.equal(first_joint, second_joint), (
        "the joints differ, so this pair does not isolate the per-site "
        "comparison from the joint comparison."
    )

    with pytest.raises(RuntimeError) as exc_info:
        assert_reference_joint_deterministic(
            [first, second], "handbuilt", DETERMINISM_SEEDS,
        )
    assert "'y'" in str(exc_info.value), (
        f"the rejection does not name the site whose log-density "
        f"moved: {str(exc_info.value)!r}"
    )


def test_behavioural_guard_rejects_a_site_recorded_under_only_one_seed() -> None:
    """A site that exists under one generator state and not the other.

    This is nondeterminism of the sharpest kind: the program took a
    different control-flow path, so the two "references" are densities
    of different models. Neither a joint comparison nor a summand
    comparison restricted to shared names would see it, because there
    is no shared name to compare, so the guard treats presence itself
    as a moving quantity.
    """
    first = Trace(
        sites={"y": _observed_site("y", -1.0)},
        output=torch.zeros(1),
        log_joint=torch.tensor([-1.0]),
    )
    second = Trace(
        sites={
            "y": _observed_site("y", -1.0),
            "branch": _observed_site("branch", 0.0),
        },
        output=torch.zeros(1),
        log_joint=torch.tensor([-1.0]),
    )

    with pytest.raises(RuntimeError) as exc_info:
        assert_reference_joint_deterministic(
            [first, second], "handbuilt", DETERMINISM_SEEDS,
        )
    assert "'branch'" in str(exc_info.value), (
        f"the rejection does not name the site recorded under only one "
        f"seed: {str(exc_info.value)!r}"
    )


def test_behavioural_guard_reads_bytes_rather_than_float_equality() -> None:
    """Two `nan` joints with the same payload are in agreement.

    The guard answers one question, "did this computation reproduce
    itself", and a computation that produced the same bits twice did.
    Whether those bits are a usable density is a different contract,
    asserted where the equivalence tier requires each reported
    log-density to be finite; folding the two together here would make
    a determinism failure and a degenerate-density failure
    indistinguishable in the message.

    The test also pins the implementation choice underneath that
    answer: `torch.equal` calls this pair unequal, because IEEE says
    `nan != nan`, so an implementation built on float equality would
    raise a determinism failure at a program that is perfectly
    reproducible.
    """
    payload = torch.tensor([float("nan")])
    assert not torch.equal(payload, payload.clone()), (
        "torch.equal now reports nan == nan, so this pair no longer "
        "distinguishes byte comparison from float equality."
    )

    first = Trace(
        sites={"y": _observed_site("y", -1.0)},
        output=torch.zeros(1),
        log_joint=payload,
    )
    second = Trace(
        sites={"y": _observed_site("y", -1.0)},
        output=torch.zeros(1),
        log_joint=payload.clone(),
    )
    assert_reference_joint_deterministic(
        [first, second], "handbuilt", DETERMINISM_SEEDS,
    )


@pytest.mark.parametrize(
    ("label", "seeds"),
    [
        pytest.param("DETERMINISM_SEEDS", DETERMINISM_SEEDS, id="probe-path"),
        pytest.param("_SWEEP_SEEDS", _SWEEP_SEEDS, id="sweep-tier"),
    ],
)
def test_seed_sets_can_observe_a_disagreement(
    label: str, seeds: tuple[int, ...],
) -> None:
    """Both seed sets carry at least two pairwise-distinct seeds.

    Every determinism assertion in this module and in the probe path
    is a comparison across generator states, so a seed set that
    shrinks to one entry, or whose entries coincide, turns all of them
    into tautologies that nothing else in the suite would notice.
    `reference_traces` refuses such a set at call time; this pins the
    constants themselves, so the degeneracy is caught at collection
    rather than as a confusing raise from deep inside a gallery
    parametrisation.
    """
    assert len(seeds) >= 2, (
        f"{label} holds {len(seeds)} seed(s); a single trace has "
        f"nothing to disagree with, so every determinism assertion "
        f"reading it would hold vacuously."
    )
    assert len(set(seeds)) == len(seeds), (
        f"{label} repeats a seed ({seeds!r}); comparing a computation "
        f"against itself under the same generator state passes for a "
        f"non-deterministic program too."
    )


def test_behavioural_guard_accepts_a_bit_identical_pair() -> None:
    """The guard is not a blanket rejection.

    A determinism check that raised on every input would pass every
    positive control in this module while proving nothing, so the
    accepting case is asserted as explicitly as the rejecting ones.
    """
    first = _trace_with(joint=-3.75, site_log_prob=-3.75)
    second = _trace_with(joint=-3.75, site_log_prob=-3.75)
    assert_reference_joint_deterministic(
        [first, second], "handbuilt", DETERMINISM_SEEDS,
    )


@pytest.mark.parametrize(
    "seeds",
    [
        pytest.param((7,), id="single-seed"),
        pytest.param((7, 7), id="repeated-seed"),
        pytest.param((7, 11, 7), id="repeat-within-sweep"),
    ],
)
def test_sweep_rejects_a_degenerate_seed_set(seeds: tuple[int, ...]) -> None:
    """A sweep that cannot observe a disagreement is refused, not run.

    One seed gives one trace and nothing to compare it against; a
    repeated seed compares a computation against itself under the same
    generator state, which a non-deterministic program passes just as
    easily as a deterministic one. Both shapes would turn every
    assertion in this module into a tautology, so the sweep raises
    instead of returning traces that cannot disagree.
    """
    example = _scored_examples()[0]
    dataset = _dataset(example)
    assert dataset.monadic is not None
    point = _points(example)[0]
    with pytest.raises(ValueError):
        reference_traces(
            dataset.monadic,
            point,
            x_input=dataset.x_input,
            observations=_gallery_data.observations_for_point(
                dataset, point,
            ),
            seeds=seeds,
        )


def test_probe_evaluation_leaves_the_global_rng_state_untouched() -> None:
    """Seeding inside the guard cannot leak into a neighbouring test.

    The determinism sweep seeds the global torch RNG, which is shared
    process-wide. Were the state left where the sweep put it, every
    draw a later test made would depend on how many probe evaluations
    ran before it, and this module would have traded an oracle bug for
    a suite-wide reproducibility bug. The probe therefore restores the
    caller's state, and the property is asserted twice: the state
    bytes come back unchanged, and the next draw is the draw the
    caller would have got with no probe call in between.
    """
    example = next(
        candidate
        for candidate in _scored_examples()
        if candidate.stem == _RNG_NEUTRALITY_EXAMPLE
    )
    dataset = _dataset(example)
    point = _points(example)[0]
    probe = QvrProbe()
    scratch = pathlib.Path(__file__).parent

    def evaluate() -> list[float]:
        return probe.evaluate(
            example.read_bytes(),
            example.stem,
            [point],
            scratch=scratch,
            monadic=dataset.monadic,
            x_input=dataset.x_input,
            observations=_gallery_data.observations_for_point(
                dataset, point,
            ),
        ).log_densities

    torch.manual_seed(4242)
    before = torch.get_rng_state()
    evaluate()
    after = torch.get_rng_state()
    assert torch.equal(before, after), (
        "QvrProbe.evaluate moved the global torch RNG state; the "
        "determinism sweep must restore what it seeded."
    )

    torch.manual_seed(4242)
    expected = torch.randn(8)
    torch.manual_seed(4242)
    evaluate()
    observed = torch.randn(8)
    assert torch.equal(expected, observed), (
        f"a draw taken after a probe evaluation differs from the draw "
        f"taken without one ({expected.tolist()!r} vs "
        f"{observed.tolist()!r}); the probe is consuming randomness "
        f"the caller owns."
    )


@pytest.mark.parametrize(
    "example", _scored_examples(), ids=lambda p: p.stem
)
def test_probe_log_density_is_invariant_to_the_ambient_rng(
    example: pathlib.Path,
) -> None:
    """The number the probe reports does not depend on the caller's seed.

    This is the determinism guarantee restated at the surface the
    equivalence tier consumes. It is asserted on the returned Python
    float, bit for bit through `float.hex`, because that float is what
    the constant-spread difference is computed from: an oracle whose
    reported value shifted with the ambient generator state would make
    `qvr_i - backend_i` depend on test ordering.
    """
    dataset = _dataset(example)
    point = _points(example)[0]
    probe = QvrProbe()
    scratch = pathlib.Path(__file__).parent
    observations = _gallery_data.observations_for_point(dataset, point)

    reported: list[str] = []
    for ambient in (0, 0x0BADC0DE):
        torch.manual_seed(ambient)
        result = probe.evaluate(
            example.read_bytes(),
            example.stem,
            [point],
            scratch=scratch,
            monadic=dataset.monadic,
            x_input=dataset.x_input,
            observations=observations,
        )
        assert len(result.log_densities) == 1
        reported.append(result.log_densities[0].hex())

    assert len(set(reported)) == 1, (
        f"{example.stem!r}: the probe reported {reported!r} under "
        f"ambient global seeds 0 and {0x0BADC0DE}, so the oracle's "
        f"value depends on whatever the surrounding test last seeded."
    )
