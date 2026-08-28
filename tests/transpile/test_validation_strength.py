"""Pin the strength of the transpile equivalence check itself.

Theorem 4.1 of
[docs/semantics/transpile-correctness.md](../../docs/semantics/transpile-correctness.md)
says a transpiled program is correct when its log-density agrees with
the QVR reference measure at **every** point of the model's support,
up to an additive constant that does not depend on the point. The
gallery suite turns that into an operational check: evaluate both
sides on a point set, subtract the mean difference, and require the
residual spread to sit under a tight tolerance.

That operationalisation has two independent failure surfaces, and
both have already fired in this repository:

1. **The tolerance can drift.** Widening the atol until a failing
   fixture goes green converts a detected measure-inequivalence back
   into a pass.
2. **The point set can collapse.** The spread of a difference
   sequence is a statement about *variation*. A one-point set has
   spread identically zero whatever the two evaluators computed, and
   a set whose data section is byte-identical throughout is the same
   vacuity restricted to the data coordinates: a backend that drops a
   data-dependent summand holds a perfectly constant offset as the
   latents move and passes.

Each of those decays left a green test that proved nothing. The
principle this module enforces is that a check is not validated until
what it rejects has been demonstrated, and that the demonstration is
made by the suite on every run rather than by hand once. The tests
below therefore assert the *properties that give the equivalence
assertion its teeth*, not the equivalence itself:

1. `test_tolerance_constants_are_pinned` and
   `test_adaptive_atol_stays_under_ceiling` pin the tolerance model.
2. `test_gallery_point_set_size_is_pinned`,
   `test_single_point_comparison_is_vacuous`, and
   `test_min_points_rejects_a_collapsed_point_set` pin the point
   count.
3. `test_every_quantified_coordinate_varies` pins per-coordinate
   coverage of the dimensions Theorem 4.1 quantifies over.
4. `test_reference_joint_is_in_support_and_varies` pins that every
   point is in the support and that the reference measure genuinely
   moves.
5. The `test_spread_*` family pins the behaviour of the spread
   statistic on synthetic difference sequences of known magnitude,
   which is the direct measurement of what the assertion rejects.
"""

from __future__ import annotations

import inspect
import math
import pathlib
from typing import Literal

import didactic.api as dx
import pytest

from quivers.continuous.programs import _LetSpec, _ScoreSpec
from tests.transpile import (
    _equivalence,
    _gallery_data,
    test_gallery_numeric_equivalence as _gallery_equivalence,
)
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import QvrProbe


# ---------------------------------------------------------------------------
# Pinned values. Every constant here mirrors a value the equivalence
# check reads at run time; the tests assert the mirror still matches.
# ---------------------------------------------------------------------------

_PINNED_DEFAULT_ATOL = 5e-4
"""Mirror of
[`_DEFAULT_ATOL`][tests.transpile._equivalence._DEFAULT_ATOL]."""

_PINNED_PER_OBS_ROUNDOFF_ESTIMATE = 5e-16
"""Mirror of
[`_PER_OBS_ROUNDOFF_ESTIMATE`][tests.transpile._equivalence._PER_OBS_ROUNDOFF_ESTIMATE]."""

_PINNED_TOLERANCE_HEADROOM = 100.0
"""Mirror of
[`_TOLERANCE_HEADROOM`][tests.transpile._equivalence._TOLERANCE_HEADROOM]."""

_ADAPTIVE_ATOL_CEILING = 1e-3
"""Hard ceiling the adaptive tolerance must stay under for every
fixture size the gallery can realistically reach.

The round-off term of
[`adaptive_atol`][tests.transpile._equivalence.adaptive_atol] is
``n_obs * condition_number * 5e-16 * 100``, so at
``condition_number = 1`` it only overtakes the 5e-4 floor once
``n_obs`` passes 1e10. Well-conditioned fixtures therefore evaluate
at the floor, and the ceiling asserts exactly that: no realistic
observation count may buy a fixture a looser tolerance."""

_LARGEST_REALISTIC_N_OBS = 100_000
"""Two orders of magnitude above the largest gallery fixture
(``zip_regression`` scores 1200 observed entries), so the ceiling is
asserted well past any size the corpus can grow into."""

_SMALLEST_SEMANTIC_DISCREPANCY = 1e-2
"""Per-point log-density discrepancy of the cheapest real bug the
check must catch: a swapped distribution argument on the gallery's
parameter ranges. The tolerance model is only meaningful while it
sits orders of magnitude below this, which
`test_adaptive_atol_stays_under_ceiling` asserts."""

_BUG_DETECTION_MARGIN = 10.0
"""Required ratio between
[`_SMALLEST_SEMANTIC_DISCREPANCY`][tests.transpile.test_validation_strength._SMALLEST_SEMANTIC_DISCREPANCY]
and the tolerance ceiling. A ratio near 1 would mean the tolerance
and the smallest real bug are the same size, so a rounding
difference and a semantic error would be indistinguishable."""

_MIN_GALLERY_POINTS = 6
"""Smallest point count the gallery equivalence check may run on.

Point 0 is the ground truth and the schedule cycles latents-only,
data-only, latents+data. Six points give two latents-only, two
data-only, and one joint perturbation, so a broken constancy
localises to the section that moved instead of merely reporting that
some point disagreed. Below six the schedule stops covering each
mode twice, and at one point the assertion is a tautology."""

_MIN_JOINT_VARIATION_IN_ATOL = 100.0
"""Required ratio between the range of the reference joint across the
point set and the equivalence tolerance.

A point set whose reference measure moves by less than the tolerance
cannot separate a correct backend from a wrong one: any discrepancy
the perturbation could expose is smaller than the noise the check
already forgives. Requiring two orders of magnitude of headroom keeps
"the points moved" from degenerating into "the points moved within
round-off"."""


# ---------------------------------------------------------------------------
# Registry of coordinates that are outside the quantified space.
# ---------------------------------------------------------------------------


class UnperturbableCoordinate(dx.Model):
    """One (example, coordinate) pair the point set holds fixed, with
    the reason the coordinate is not a dimension Theorem 4.1
    quantifies over.

    `kind` is the structural class of the coordinate and
    `justification` is the written argument for the exemption. An
    entry is a claim the suite checks in both directions: a
    coordinate listed here must be frozen across the whole point set,
    and a coordinate not listed here must move. An unexplained frozen
    coordinate fails.
    """

    example: str
    coordinate: str
    kind: Literal["plate-subscript"]
    justification: str


_PLATE_SUBSCRIPT_ARGUMENT = (
    "A plate subscript is fixed by the experimental design rather "
    "than drawn from the model, so it is a structural input on the "
    "same footing as a FinSet size: it is not a coordinate of the "
    "(theta, y) support Theorem 4.1 quantifies over, and the "
    "constant c is allowed to depend on it. Freezing it does not "
    "blind the check to a mis-gathered index, because the gathered "
    "latent and the scored response both move across the point set, "
    "so a permuted or off-by-one gather shifts the per-row mean by "
    "an amount that varies with the latents and breaks constancy."
)

_UNPERTURBABLE_COORDINATES: tuple[UnperturbableCoordinate, ...] = tuple(
    UnperturbableCoordinate(
        example=_example,
        coordinate=_coordinate,
        kind="plate-subscript",
        justification=f"{_detail}. {_PLATE_SUBSCRIPT_ARGUMENT}",
    )
    for _example, _coordinate, _detail in (
        (
            "beta_binomial_ab_test", "arm_idx",
            "Gathers conc1 and conc0 over the Arm plate",
        ),
        (
            "beta_regression", "out_idx",
            "Gathers beta_0, beta_1 and phi over the Out plate",
        ),
        (
            "factor_analysis", "item_idx",
            "Gathers the per-item latent code row of Z_mat over the "
            "Item plate",
        ),
        (
            "factor_analysis", "obs_idx",
            "Gathers the loading row of W_mat over the ObsDim plate",
        ),
        (
            "gamma_regression", "cat_idx",
            "Gathers beta_0 and beta_1 over the Cat plate",
        ),
        (
            "half_student_t_hierarchical", "group_idx",
            "Gathers the per-group offset u over the Group plate",
        ),
        (
            "horseshoe_regression", "coef_idx",
            "Gathers lambda_local and z_raw over the Coef plate",
        ),
        (
            "irt_2pl", "item_idx",
            "Gathers difficulty and discrim over the Item plate",
        ),
        (
            "irt_2pl", "person_idx",
            "Gathers ability over the Person plate",
        ),
        (
            "lda", "word_idx",
            "Names the fibration from each word position into its "
            "document, selecting the theta row the per-word mixture "
            "marginal is scored under",
        ),
        (
            "negbin_regression", "out_idx",
            "Gathers beta_0, beta_1 and dispersion over the Out plate",
        ),
        (
            "ppca", "item_idx",
            "Gathers the per-item latent code row of Z_mat over the "
            "Item plate",
        ),
        (
            "ppca", "obs_idx",
            "Gathers the loading row of W_mat over the ObsDim plate",
        ),
        (
            "zip_regression", "out_idx",
            "Gathers alpha_zero, beta_zero, alpha_rate and beta_rate "
            "over the Out plate",
        ),
    )
)
"""Every coordinate the gallery point set holds fixed on purpose.

Populate an entry only for a coordinate that is genuinely outside the
quantified space. A covariate whose value enters the density is
inside it: `x` is perturbed in every regression example, so any other
covariate that the harness merely fails to move is a coverage gap to
close in the perturber, not an entry to add here.

Which of the two a given coordinate is, is not left to the entry's
prose. `test_every_quantified_coordinate_varies` checks every
`plate-subscript` entry against
[`structural_subscript_names`][tests.transpile._gallery_data.structural_subscript_names],
the classifier the point builder itself consults, so the registry can
only ever exempt a coordinate the perturber independently reads as a
plate subscript. A frozen coordinate the classifier does not name,
`n_trials` being the count parameter of a Beta-Binomial rather than
an index into a plate, cannot be registered at all and has to be
fixed in the perturber."""


_MIN_JUSTIFICATION_CHARS = 80
"""Length floor on an exemption's written justification. A one-word
reason is how a registry of arguments decays into a list of names."""


# ---------------------------------------------------------------------------
# Cached per-example evaluation.
# ---------------------------------------------------------------------------


class _EvaluatedPointSet(dx.Model):
    """One gallery example's point set and its reference log-densities.

    Held in a module-level cache so the per-example tests below share
    one QVR evaluation pass. Every input is deterministic (the point
    builder seeds its own generator and never touches the global RNG),
    so the cached values are reproducible run to run.
    """

    stem: str
    dataset: _gallery_data.GalleryDataset = dx.field(opaque=True)
    points: tuple[Point, ...] = dx.field(opaque=True)
    log_densities: tuple[float, ...]


_EVALUATION_CACHE: dict[str, _EvaluatedPointSet] = {}


def _gallery_cells() -> list[pathlib.Path]:
    """Every gallery example whose `.md` ships a synthetic-data block
    and whose point set the reference oracle can score."""
    return [
        example
        for example in _gallery_data.gallery_examples_with_data()
        if example.stem
        not in _gallery_equivalence._SKIP_DATASET_LOAD_FAILED
        and example.stem
        not in _gallery_equivalence._SKIP_QVR_INCOMPATIBLE
    ]


def _evaluate(example: pathlib.Path) -> _EvaluatedPointSet:
    """Load `example`, build its point set, and score the reference
    joint at every point."""
    cached = _EVALUATION_CACHE.get(example.stem)
    if cached is not None:
        return cached
    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example is in neither "
        f"`_SKIP_DATASET_LOAD_FAILED` nor `_SKIP_QVR_INCOMPATIBLE`."
    )
    points = _gallery_data.points_from_dataset(dataset)
    probe = QvrProbe()
    scratch = pathlib.Path("/tmp") / f"qvr_strength_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    source = example.read_bytes()
    log_densities: list[float] = []
    for point in points:
        # One probe call per point. The probe's `observations` keyword
        # overrides the flat per-point payload, so a perturbed point
        # needs its own pre-shaped observation dict; passing the
        # dataset's ground-truth observations for the whole set would
        # score the reference at unperturbed data.
        log_densities.extend(
            probe.evaluate(
                source,
                example.stem,
                [point],
                scratch=scratch,
                monadic=dataset.monadic,
                x_input=dataset.x_input,
                observations=_gallery_data.observations_for_point(
                    dataset, point,
                ),
            ).log_densities
        )
    evaluated = _EvaluatedPointSet(
        stem=example.stem,
        dataset=dataset,
        points=tuple(points),
        log_densities=tuple(float(lp) for lp in log_densities),
    )
    _EVALUATION_CACHE[example.stem] = evaluated
    return evaluated


def _latent_site_names(
    dataset: _gallery_data.GalleryDataset,
) -> tuple[str, ...]:
    """Post-inline names of the compiled program's latent sample
    sites.

    Derived here rather than imported so this module measures the
    quantified latent dimensions independently of the helper the
    point builder uses to fill them: a bug that dropped a site from
    the ground-truth capture would otherwise also drop it from the
    coverage check.
    """
    monadic = dataset.monadic
    if monadic is None:
        return ()
    names: list[str] = []
    for spec in monadic._step_specs:
        if isinstance(spec, (_LetSpec, _ScoreSpec)) or spec.is_observed:
            continue
        names.extend(spec.vars)
    return tuple(names)


def _coordinate_key(
    value: float | int | list[float] | list[int],
) -> tuple[float, ...]:
    """Hashable identity of one point-section entry, for counting
    distinct values across the point set."""
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(entry) for entry in value)


def _distinct_values(
    points: tuple[Point, ...], name: str, *, latent: bool,
) -> int:
    """Number of distinct values `name` takes across `points`."""
    seen: set[tuple[float, ...]] = set()
    for point in points:
        section = point.params if latent else point.data
        if name in section:
            seen.add(_coordinate_key(section[name]))
    return len(seen)


def _registered(stem: str) -> dict[str, UnperturbableCoordinate]:
    """Registry entries for one example, keyed by coordinate name."""
    return {
        entry.coordinate: entry
        for entry in _UNPERTURBABLE_COORDINATES
        if entry.example == stem
    }


def _diff_pair(
    diffs: list[float], base: float = -137.25,
) -> tuple[list[float], list[float]]:
    """A (qvr, target) sequence pair realising `diffs` exactly.

    The reference side walks a spread of realistic joint values so the
    constructed case exercises the same subtraction the gallery check
    performs, rather than differencing two constants.
    """
    qvr = [base - 3.5 * index for index in range(len(diffs))]
    return qvr, [q + d for q, d in zip(qvr, diffs)]


# ---------------------------------------------------------------------------
# 1. The tolerance model cannot be silently loosened.
# ---------------------------------------------------------------------------


def test_tolerance_constants_are_pinned() -> None:
    """The three constants of the tolerance model hold their derived
    values.

    Each is a claim about floating-point behaviour, not a knob:

    * `_DEFAULT_ATOL = 5e-4` is the floor, roughly an order of
      magnitude above the worst cross-backend agreement measured on a
      60-observation fixture and two orders below the cheapest
      semantic discrepancy a real bug produces.
    * `_PER_OBS_ROUNDOFF_ESTIMATE = 5e-16` is the measured float64
      round-off one observation-site `log_prob` contributes, which is
      why the estimator grows linearly in the observation count.
    * `_TOLERANCE_HEADROOM = 100.0` is the multiplier that absorbs
      benign last-ULP disagreements between two algebraically equal
      formulations of the same density.

    Changing any of them requires re-deriving the round-off argument
    from theory and updating the pinned mirror deliberately. Adjusting
    a number until a failing fixture passes is the decay this test
    exists to block: it converts a detected measure-inequivalence into
    a green run.
    """
    assert _equivalence._DEFAULT_ATOL == _PINNED_DEFAULT_ATOL, (
        f"`_DEFAULT_ATOL` moved to "
        f"{_equivalence._DEFAULT_ATOL!r} from the pinned "
        f"{_PINNED_DEFAULT_ATOL!r}. The floor is derived from measured "
        f"cross-backend agreement, so re-derive it before changing it "
        f"and update this mirror in the same commit."
    )
    assert (
        _equivalence._PER_OBS_ROUNDOFF_ESTIMATE
        == _PINNED_PER_OBS_ROUNDOFF_ESTIMATE
    ), (
        f"`_PER_OBS_ROUNDOFF_ESTIMATE` moved to "
        f"{_equivalence._PER_OBS_ROUNDOFF_ESTIMATE!r} from the pinned "
        f"{_PINNED_PER_OBS_ROUNDOFF_ESTIMATE!r}. This is a measured "
        f"per-log_prob float64 round-off; re-measure it across the "
        f"backends before changing it."
    )
    assert (
        _equivalence._TOLERANCE_HEADROOM == _PINNED_TOLERANCE_HEADROOM
    ), (
        f"`_TOLERANCE_HEADROOM` moved to "
        f"{_equivalence._TOLERANCE_HEADROOM!r} from the pinned "
        f"{_PINNED_TOLERANCE_HEADROOM!r}. The multiplier is the "
        f"headroom above per-point round-off, not a dial to turn when "
        f"a fixture fails."
    )


def test_adaptive_atol_stays_under_ceiling() -> None:
    """The adaptive estimator cannot hand a realistic fixture a loose
    tolerance.

    `adaptive_atol` returns `max(floor, n_obs * condition_number *
    round_off * headroom)`. At `condition_number = 1` the round-off
    term is `n_obs * 5e-14`, so it only reaches the 5e-4 floor once
    `n_obs` passes 1e10, which no fixture will. The estimator must
    therefore evaluate to exactly the floor across the whole realistic
    range, stay monotone, and never fall below the floor even for
    degenerate inputs.

    The ceiling matters because Theorem 4.1 is only testable while the
    tolerance sits far below the smallest discrepancy a semantic bug
    produces. A swapped distribution argument moves the log-density by
    at least 1e-2 nats per point on the gallery's parameter ranges, so
    a tolerance within an order of magnitude of that would make a real
    bug and a rounding difference indistinguishable.
    """
    assert _ADAPTIVE_ATOL_CEILING >= _PINNED_DEFAULT_ATOL, (
        "the ceiling must sit at or above the floor, otherwise it "
        "asserts nothing about the adaptive term."
    )
    assert (
        _SMALLEST_SEMANTIC_DISCREPANCY
        >= _BUG_DETECTION_MARGIN * _ADAPTIVE_ATOL_CEILING
    ), (
        f"the tolerance ceiling {_ADAPTIVE_ATOL_CEILING!r} leaves less "
        f"than a factor of {_BUG_DETECTION_MARGIN!r} below the "
        f"cheapest semantic discrepancy "
        f"{_SMALLEST_SEMANTIC_DISCREPANCY!r}; at that ratio a real bug "
        f"and float round-off are the same size."
    )

    sizes = [1, 8, 64, 200, 1_200, 10_000, _LARGEST_REALISTIC_N_OBS]
    previous = 0.0
    for n_obs in sizes:
        atol = _equivalence.adaptive_atol(n_obs=n_obs)
        assert atol <= _ADAPTIVE_ATOL_CEILING, (
            f"adaptive_atol(n_obs={n_obs}) = {atol!r} exceeds the "
            f"ceiling {_ADAPTIVE_ATOL_CEILING!r}. Either the round-off "
            f"estimate or the headroom grew; re-derive the round-off "
            f"argument rather than raising the ceiling."
        )
        assert atol == _PINNED_DEFAULT_ATOL, (
            f"adaptive_atol(n_obs={n_obs}) = {atol!r} but the "
            f"round-off term at this size is "
            f"{n_obs * _PINNED_PER_OBS_ROUNDOFF_ESTIMATE * _PINNED_TOLERANCE_HEADROOM!r}, "
            f"far under the {_PINNED_DEFAULT_ATOL!r} floor, so the "
            f"floor must dominate."
        )
        assert atol >= previous, (
            f"adaptive_atol is not monotone in n_obs: {atol!r} at "
            f"{n_obs} follows {previous!r}. A non-monotone estimator "
            f"tightens the bug-detection threshold as coverage grows."
        )
        previous = atol

    for degenerate in (0, -1):
        assert (
            _equivalence.adaptive_atol(n_obs=degenerate)
            == _PINNED_DEFAULT_ATOL
        ), "a non-positive observation count must fall back to the floor."
    assert (
        _equivalence.adaptive_atol(n_obs=64, condition_number=0.0)
        == _PINNED_DEFAULT_ATOL
    ), "a degenerate condition number must never drop below the floor."


# ---------------------------------------------------------------------------
# 2. The point set cannot collapse.
# ---------------------------------------------------------------------------


def test_gallery_point_set_size_is_pinned() -> None:
    """The gallery evaluates at least six points, covering each
    perturbation mode.

    The constant-spread contract is a statement about how the
    difference varies across points, so the point count is the
    resolution of the check. Six is the smallest count that gives two
    latents-only, two data-only, and one joint perturbation, which is
    what lets a broken constancy localise to the section that moved
    rather than to "some point". A default that quietly dropped back
    toward one would restore the vacuity the multi-point set was built
    to remove.
    """
    default = inspect.signature(
        _gallery_data.points_from_dataset,
    ).parameters["n_points"].default
    assert default >= _MIN_GALLERY_POINTS, (
        f"`points_from_dataset` defaults to {default!r} points, below "
        f"the pinned minimum {_MIN_GALLERY_POINTS!r}. Every gallery "
        f"caller takes this default, so lowering it weakens every "
        f"equivalence cell at once."
    )

    labels = _gallery_data.perturbation_labels(default)
    assert labels[0] == _gallery_data.PERTURB_GROUND_TRUTH
    for mode in (
        _gallery_data.PERTURB_LATENTS,
        _gallery_data.PERTURB_DATA,
        _gallery_data.PERTURB_BOTH,
    ):
        assert labels.count(mode) >= 1, (
            f"the {default}-point schedule {labels!r} never runs the "
            f"{mode!r} perturbation, so that section of the point "
            f"space is untested."
        )
    assert labels.count(_gallery_data.PERTURB_LATENTS) >= 2, (
        f"the schedule {labels!r} carries fewer than two latents-only "
        f"points; one gives no cross-check on the prior-scoring path."
    )
    assert labels.count(_gallery_data.PERTURB_DATA) >= 2, (
        f"the schedule {labels!r} carries fewer than two data-only "
        f"points; one gives no cross-check on the dropped-data-term "
        f"path."
    )

    cells = _gallery_cells()
    assert cells, (
        "no gallery example is available to measure a real point set "
        "against; the corpus or its skip registries collapsed."
    )
    evaluated = _evaluate(cells[0])
    assert len(evaluated.points) >= _MIN_GALLERY_POINTS, (
        f"{evaluated.stem!r}: the realised point set holds "
        f"{len(evaluated.points)} points, below the pinned minimum "
        f"{_MIN_GALLERY_POINTS!r}."
    )


def test_single_point_comparison_is_vacuous() -> None:
    """A one-point comparison accepts an arbitrarily wrong backend.

    This is the failure mode that let a Stan renderer which dropped
    data-dependent terms through: with `n == 1` the mean of the
    difference sequence is the difference itself, so
    `max_i |d_i - mean(d)|` is exactly zero whatever the two
    evaluators computed. The demonstration is kept in the suite rather
    than in a commit message because it is the reason every gallery
    call site passes `min_points=2`, and a reader who does not see the
    vacuity has no way to know why that argument is load-bearing.
    """
    qvr, target = _diff_pair([1.0e6])
    constant = _equivalence.assert_log_density_match(
        qvr, target, context="vacuity-demonstration",
    )
    assert constant == pytest.approx(1.0e6), (
        "the one-point comparison should absorb the entire "
        "discrepancy into the constant, which is precisely why it "
        "cannot reject anything."
    )


def test_min_points_rejects_a_collapsed_point_set() -> None:
    """`assert_log_density_match` refuses a point sequence shorter than
    two when the caller declares it needs variation.

    The gallery equivalence cell passes `min_points=2`, so a point set
    that ever collapses fails loudly instead of passing
    unconditionally. The same one-point input that
    `test_single_point_comparison_is_vacuous` shows sailing through at
    the default must be rejected here.
    """
    qvr, target = _diff_pair([1.0e6])
    with pytest.raises(AssertionError) as exc_info:
        _equivalence.assert_log_density_match(
            qvr, target, context="collapsed", min_points=2,
        )
    message = str(exc_info.value)
    assert "at least 2" in message, (
        f"the rejection must name the required point count so the "
        f"failure is actionable; got {message!r}"
    )

    with pytest.raises(AssertionError):
        _equivalence.assert_log_density_match(
            [], [], context="empty", min_points=2,
        )

    # Pin the gallery call site, not just the helper's capability: the
    # `min_points` guard only protects a caller that asks for it, and
    # the gallery equivalence cell is the caller whose point set the
    # whole multi-point design exists to keep honest. Reading the
    # module source rather than one function's keeps the pin alive
    # across a refactor that moves the call between functions.
    source = inspect.getsource(_gallery_equivalence)
    assert "min_points=2" in source, (
        "the gallery equivalence module no longer passes "
        "`min_points=2` to `assert_log_density_match`. Without it a "
        "collapsed point set passes unconditionally, which is exactly "
        "how this check decayed before."
    )


# ---------------------------------------------------------------------------
# 3. Coverage of the quantified dimensions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
def test_every_quantified_coordinate_varies(
    example: pathlib.Path,
) -> None:
    """Every coordinate Theorem 4.1 quantifies over actually moves
    across the point set.

    The theorem quantifies over the whole support, so the operational
    check inherits a coverage obligation the aggregate statistics hide:
    a point set that never moves a given coordinate proves nothing
    about that coordinate. The joint can move a great deal while one
    observation array stays byte-identical throughout, and against
    such a set any backend error that is a function of that array
    alone is absorbed into the constant `c`. That is not a
    hypothetical: a data section frozen at ground truth is precisely
    what makes a dropped data-dependent summand invisible.

    Two families of coordinate are checked, and each must take at
    least two distinct values across the set:

    1. Every latent sample site of the compiled program. A site the
       point set never pins is worse than one it never moves, so a
       missing site fails here too.
    2. Every observed data array the point carries.

    A coordinate that is genuinely outside the quantified space, a
    plate subscript fixed by the experimental design, needs an entry
    in `_UNPERTURBABLE_COORDINATES` stating the argument. The registry
    is checked in three directions: an unexplained frozen coordinate
    fails, a registered coordinate that has started moving fails as a
    stale claim, and an entry whose `kind` the point builder's own
    classifier does not confirm fails as a misfiled one. None of the
    three may be settled by editing the assertion.
    """
    evaluated = _evaluate(example)
    dataset = evaluated.dataset
    points = evaluated.points
    registry = _registered(example.stem)

    latents = _latent_site_names(dataset)
    observed = sorted(_gallery_data.observed_data_names(dataset))
    known = frozenset(latents) | frozenset(observed)
    unknown = sorted(name for name in registry if name not in known)
    assert not unknown, (
        f"{example.stem!r}: `_UNPERTURBABLE_COORDINATES` names "
        f"{unknown!r}, which is neither a latent site nor an observed "
        f"array of this example. Drop the stale entry; a registry of "
        f"names that no longer exist stops describing the gap it "
        f"claims to."
    )

    subscripts = _gallery_data.structural_subscript_names(dataset)
    misfiled = sorted(
        name
        for name, entry in registry.items()
        if entry.kind == "plate-subscript" and name not in subscripts
    )
    assert not misfiled, (
        f"{example.stem!r}: `_UNPERTURBABLE_COORDINATES` files "
        f"{misfiled!r} as a plate subscript, but "
        f"`structural_subscript_names` does not classify it as one. "
        f"A subscript indexes a plate, so it is an integer vector "
        f"with one entry per row; anything else the harness leaves "
        f"frozen is a coordinate of the support whose value enters "
        f"the density, and the exemption would hide a coverage gap "
        f"instead of recording a structural one. Close the gap in "
        f"the perturber."
    )

    for name in latents:
        present = sum(1 for point in points if name in point.params)
        assert present == len(points), (
            f"{example.stem!r}: latent site {name!r} is carried by "
            f"{present} of {len(points)} points. A site the point set "
            f"does not pin is a coordinate the equivalence check never "
            f"fixes, so the two evaluators are free to score it at "
            f"different values and the difference is not a function of "
            f"the point at all."
        )
        distinct = _distinct_values(points, name, latent=True)
        entry = registry.get(name)
        if entry is None:
            assert distinct >= 2, (
                f"{example.stem!r}: latent site {name!r} takes a "
                f"single value across all {len(points)} points, so the "
                f"constant-spread check says nothing about how the "
                f"backend scores that coordinate. Give the site a "
                f"support the perturber can step in, or, if it truly "
                f"cannot move, record the argument in "
                f"`_UNPERTURBABLE_COORDINATES`."
            )
        else:
            assert distinct < 2, (
                f"{example.stem!r}: `_UNPERTURBABLE_COORDINATES` "
                f"claims latent site {name!r} cannot move "
                f"({entry.justification}), but it takes {distinct} "
                f"distinct values. Drop the entry: the coordinate is "
                f"covered and the claim is stale."
            )

    for name in observed:
        distinct = _distinct_values(points, name, latent=False)
        entry = registry.get(name)
        if entry is None:
            assert distinct >= 2, (
                f"{example.stem!r}: observed array {name!r} is "
                f"byte-identical at every one of the {len(points)} "
                f"points, so the check is blind to any backend error "
                f"that depends on {name!r} alone: such an error is a "
                f"constant across the set and the mean subtraction "
                f"absorbs it. Give the array a support the perturber "
                f"can step in, or, if it is a structural index rather "
                f"than a coordinate of the support, record the "
                f"argument in `_UNPERTURBABLE_COORDINATES`."
            )
        else:
            assert distinct < 2, (
                f"{example.stem!r}: `_UNPERTURBABLE_COORDINATES` "
                f"claims observed array {name!r} cannot move "
                f"({entry.justification}), but it takes {distinct} "
                f"distinct values. Drop the entry: the coordinate is "
                f"covered and the claim is stale."
            )


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
def test_point_set_exposes_a_planted_coordinate_defect(
    example: pathlib.Path,
) -> None:
    """Each example's own point set provably rejects a defect confined
    to any single coordinate.

    Counting distinct values shows a coordinate *moved*; this test
    shows the movement is enough to be *detected*, and it shows it on
    the real point set rather than on an abstract sequence. For every
    non-exempt coordinate the test plants the shape of the historical
    bug: a per-point term of the documented smallest-bug magnitude
    that fires exactly on the points where that coordinate left its
    ground-truth value, added on top of a legitimate additive constant
    standing in for a Jacobian or normaliser difference. A dropped
    data-dependent summand has precisely this shape, constant wherever
    the perturbation did not reach it.

    The assertion must reject every one of those planted sequences
    while the constant alone is tolerated. A coordinate whose planted
    defect slips through is a coordinate the equivalence cell cannot
    police, whatever its distinct-value count says, so the sensitivity
    of the point set is measured here on every run instead of being
    established once by hand.
    """
    evaluated = _evaluate(example)
    points = evaluated.points
    labels = _gallery_data.perturbation_labels(len(points))
    reference = list(evaluated.log_densities)
    registry = _registered(example.stem)
    constant = 4.25

    coordinates = [
        (name, True) for name in _latent_site_names(evaluated.dataset)
    ] + [
        (name, False)
        for name in sorted(
            _gallery_data.observed_data_names(evaluated.dataset)
        )
    ]
    planted_any = False
    for name, latent in coordinates:
        if name in registry:
            continue
        keys = [
            _coordinate_key(
                (point.params if latent else point.data)[name]
            )
            for point in points
        ]
        moved_at = [key != keys[0] for key in keys]
        planted = [
            value + constant
            + (_SMALLEST_SEMANTIC_DISCREPANCY if moved else 0.0)
            for value, moved in zip(reference, moved_at)
        ]
        planted_any = True
        with pytest.raises(AssertionError) as exc_info:
            _equivalence.assert_log_density_match(
                reference,
                planted,
                context=f"planted@{example.stem}:{name}",
                labels=labels,
                min_points=2,
            )
        assert "spread" in str(exc_info.value), (
            f"{example.stem!r}: the rejection of the defect planted on "
            f"{name!r} does not report the spread statistic, so a real "
            f"failure would not be diagnosable; got "
            f"{str(exc_info.value)!r}"
        )
        # The same sequence without the defect must pass, so the
        # rejection above is attributable to the planted term and not
        # to the additive constant the contract allows.
        assert _equivalence.assert_log_density_match(
            reference,
            [value + constant for value in reference],
            context=f"constant-only@{example.stem}:{name}",
            labels=labels,
            min_points=2,
        ) == pytest.approx(constant, abs=1e-9)

    assert planted_any, (
        f"{example.stem!r}: every coordinate of this example is "
        f"registered as unperturbable, so no defect can be planted "
        f"anywhere and the equivalence cell for it tests nothing."
    )


def test_unperturbable_registry_is_well_formed() -> None:
    """Every exemption names a live example and carries a written
    argument.

    The registry is the only sanctioned way to hold a coordinate
    fixed, which makes it the obvious place for the check to decay: an
    entry with a one-word reason, or an entry for an example that no
    longer exists, silently widens the exempt set. Requiring a real
    argument of substantial length keeps each exemption reviewable,
    and requiring the example to exist keeps the registry describing
    the corpus it is exempting.
    """
    stems = {example.stem for example in _gallery_data.gallery_examples_with_data()}
    seen: set[tuple[str, str]] = set()
    for entry in _UNPERTURBABLE_COORDINATES:
        assert entry.example in stems, (
            f"`_UNPERTURBABLE_COORDINATES` names example "
            f"{entry.example!r}, which is not a gallery example with "
            f"synthetic data. Drop the entry."
        )
        key = (entry.example, entry.coordinate)
        assert key not in seen, (
            f"duplicate registry entry for {key!r}; two arguments for "
            f"one exemption means one of them is unreviewed."
        )
        seen.add(key)
        assert len(entry.justification) >= _MIN_JUSTIFICATION_CHARS, (
            f"the exemption for {key!r} carries a "
            f"{len(entry.justification)}-character justification, "
            f"under the {_MIN_JUSTIFICATION_CHARS}-character floor. "
            f"State why the coordinate is outside the (theta, y) "
            f"support Theorem 4.1 quantifies over."
        )


# ---------------------------------------------------------------------------
# 4. Every point is in the support, and the reference measure moves.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
def test_reference_joint_is_in_support_and_varies(
    example: pathlib.Path,
) -> None:
    """Every point scores a finite reference joint, and the joint
    genuinely varies across the set.

    Theorem 4.1 quantifies over points *of the support*, so a
    perturbation that steps outside it is not a witness of anything:
    both evaluators return `-inf`, their difference is `nan`, and a
    check that compared them would be testing a floating-point
    convention rather than a measure. Finiteness at every point is
    what makes the set admissible.

    Variation is what makes it informative. The check requires the
    joint's range across the set to clear the equivalence tolerance by
    two orders of magnitude: a set whose reference measure moves by
    less than the tolerance cannot separate a correct backend from a
    wrong one, because every discrepancy such a perturbation could
    expose is smaller than the noise the assertion already forgives.
    """
    evaluated = _evaluate(example)
    labels = _gallery_data.perturbation_labels(len(evaluated.points))
    log_densities = list(evaluated.log_densities)

    for index, lp in enumerate(log_densities):
        assert math.isfinite(lp), (
            f"{example.stem!r}: point {index} ({labels[index]}) scores "
            f"a non-finite reference joint ({lp!r}), so it left the "
            f"model's support and cannot witness the equivalence at "
            f"any tolerance."
        )

    span = max(log_densities) - min(log_densities)
    required = _MIN_JOINT_VARIATION_IN_ATOL * _PINNED_DEFAULT_ATOL
    assert span >= required, (
        f"{example.stem!r}: the reference joint spans only "
        f"{span:.6e} nats across {len(log_densities)} points, under "
        f"the {required:.6e} nats this check requires "
        f"({_MIN_JOINT_VARIATION_IN_ATOL!r} times the "
        f"{_PINNED_DEFAULT_ATOL!r} tolerance). A point set that moves "
        f"the measure by less than the tolerance cannot reject a "
        f"wrong backend. Per-point values: "
        f"{[round(lp, 6) for lp in log_densities]!r}"
    )


# ---------------------------------------------------------------------------
# 5. The spread statistic itself behaves.
# ---------------------------------------------------------------------------


def test_spread_accepts_a_genuinely_constant_offset() -> None:
    """A constant additive offset is accepted, and reported as `c`.

    This is the permissive half of Theorem 4.1 and it is not
    negotiable: a backend may legitimately differ from the reference
    by a fixed constant (a Jacobian term from a different
    parameterisation, a normaliser one framing drops), and the
    equivalence relation quotients that out. A check that rejected a
    constant offset would fail correct transpilations, which is the
    mirror-image failure of the vacuity this module guards against.
    """
    offset = 12.3456789
    qvr, target = _diff_pair([offset] * 6)
    constant = _equivalence.assert_log_density_match(
        qvr, target, context="constant-offset", min_points=2,
    )
    assert constant == pytest.approx(offset, abs=1e-12)

    # The same offset must still be accepted when the reference joints
    # are far apart: the statistic is a deviation from the mean, so it
    # may not grow with the magnitude of the values being differenced.
    wide_qvr = [-1.0e4, -1.0, 5.0e3, 12.5, -7.5e3, 0.25]
    wide_target = [value + offset for value in wide_qvr]
    assert _equivalence.assert_log_density_match(
        wide_qvr, wide_target, context="constant-offset-wide", min_points=2,
    ) == pytest.approx(offset, abs=1e-9)


def test_spread_rejects_a_non_constant_offset() -> None:
    """A difference that drifts by the smallest real-bug magnitude is
    rejected.

    The documented detection threshold is 1e-2 nats per point: a
    swapped distribution argument on the gallery's parameter ranges
    moves the log-density by at least that much. A drift of that size
    across a six-point set produces a spread of roughly 2.5e-2, fifty
    times the 5e-4 tolerance, so the check has to reject it with
    plenty of margin. If this test ever passes, the tolerance has
    grown past the point where a real bug is distinguishable from
    round-off.
    """
    drift = _SMALLEST_SEMANTIC_DISCREPANCY
    diffs = [7.0 + index * drift for index in range(6)]
    qvr, target = _diff_pair(diffs)
    with pytest.raises(AssertionError) as exc_info:
        _equivalence.assert_log_density_match(
            qvr, target, context="drifting-offset", min_points=2,
        )
    message = str(exc_info.value)
    assert "spread" in message, (
        f"the rejection must report the spread statistic so the "
        f"failure is diagnosable; got {message!r}"
    )

    mean = sum(diffs) / len(diffs)
    spread = max(abs(d - mean) for d in diffs)
    assert spread >= 10.0 * _PINNED_DEFAULT_ATOL, (
        f"a {drift!r}-per-point drift produces a spread of "
        f"{spread:.6e}, less than ten times the {_PINNED_DEFAULT_ATOL!r} "
        f"tolerance. The margin between the cheapest real bug and the "
        f"tolerance has collapsed."
    )


def test_spread_rejects_a_single_deviating_point() -> None:
    """A difference that is constant on every point but one is
    rejected.

    This is the shape a partially-correct backend produces: it agrees
    with the reference wherever the perturbation did not reach the
    broken term, and disagrees at the one point that moved the
    coordinate the term depends on. Because the mean absorbs `1/n` of
    a lone deviation, the surviving spread is `(n-1)/n` of it, so a
    1e-2 outlier over six points still leaves 8.3e-3, more than an
    order of magnitude above tolerance. A check that smoothed a lone
    outlier away would pass exactly the backends whose error is
    confined to one section of the point space.
    """
    outlier = _SMALLEST_SEMANTIC_DISCREPANCY
    for outlier_index in (0, 3, 5):
        diffs = [-2.5] * 6
        diffs[outlier_index] += outlier
        qvr, target = _diff_pair(diffs)
        with pytest.raises(AssertionError) as exc_info:
            _equivalence.assert_log_density_match(
                qvr,
                target,
                context=f"outlier-{outlier_index}",
                labels=_gallery_data.perturbation_labels(6),
                min_points=2,
            )
        message = str(exc_info.value)
        assert f"index {outlier_index}" in message, (
            f"the rejection must localise the worst point so the "
            f"failure names the perturbation that broke constancy; "
            f"got {message!r}"
        )


def test_spread_boundary_matches_the_pinned_tolerance() -> None:
    """The accept / reject boundary sits at the pinned tolerance.

    The statistic is `max_i |d_i - mean(d)|` compared against `atol`,
    so a sequence engineered to a known spread must be accepted just
    below the floor and rejected just above it. Pinning both sides
    keeps a change to the comparison itself, a `>=` turned into a `>`,
    a spread computed before the mean subtraction, from passing as a
    refactor.
    """
    n = 6
    for factor, should_pass in ((0.5, True), (0.9, True), (2.0, False)):
        # A lone deviation of `delta` over `n` points leaves a spread
        # of `delta * (n - 1) / n` after the mean subtraction.
        target_spread = factor * _PINNED_DEFAULT_ATOL
        delta = target_spread * n / (n - 1)
        diffs = [4.0] * n
        diffs[2] += delta
        qvr, target = _diff_pair(diffs)
        if should_pass:
            _equivalence.assert_log_density_match(
                qvr, target, context=f"boundary-{factor}", min_points=2,
            )
        else:
            with pytest.raises(AssertionError):
                _equivalence.assert_log_density_match(
                    qvr, target, context=f"boundary-{factor}", min_points=2,
                )


def test_spread_rejects_degenerate_sequences() -> None:
    """Non-finite differences and mismatched lengths are errors, not
    passes.

    Two `-inf` log-densities differ by `nan`, and `nan` compares false
    against any tolerance, so a naive spread check would report
    agreement for a pair of points that both left the support. A
    length mismatch is the other degenerate shape: it means the two
    evaluators were scored on different point sets, and pairing them
    by position compares unrelated points.
    """
    qvr, target = _diff_pair([1.0] * 4)
    target[2] = math.inf
    with pytest.raises(AssertionError) as exc_info:
        _equivalence.assert_log_density_match(
            qvr, target, context="non-finite", min_points=2,
        )
    assert "non-finite" in str(exc_info.value)

    both_infinite = [-math.inf, -math.inf, -math.inf]
    with pytest.raises(AssertionError):
        _equivalence.assert_log_density_match(
            both_infinite,
            list(both_infinite),
            context="both-outside-support",
            min_points=2,
        )

    with pytest.raises(AssertionError) as exc_info:
        _equivalence.assert_log_density_match(
            [1.0, 2.0], [1.0, 2.0, 3.0], context="length", min_points=2,
        )
    assert "length mismatch" in str(exc_info.value)
