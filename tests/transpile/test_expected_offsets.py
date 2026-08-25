"""Named-constant equivalence: the offset a backend is *entitled* to.

Theorem 4.1 of
[docs/semantics/transpile-correctness](../../docs/semantics/transpile-correctness/index.md)
declares a transpiled program correct when its log-density agrees with
the QVR reference measure up to an additive constant that does not
depend on the evaluation point. The gallery suite enforces exactly that
statement through
[`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]:
it subtracts the mean of the pointwise differences and bounds the
residual spread.

That statement has a soft spot, which this module closes. Quantifying
existentially over the constant ("there exists some `c`") makes any
systematically wrong but point-independent term invisible. A renderer
that drops a whole prior factor whose value happens not to move with
the point, that double-counts a normalizer, or that scores a `Beta` as
a `Kumaraswamy` at coincidentally matched moments, all shift `c` and
leave the spread untouched. The suite would stay green while the
emitted program denoted a different measure. This module therefore
replaces the existential with a **named-constant criterion** (NCC):
for each `(backend, example)` cell the expected offset is derived in
closed form ahead of the measurement, pinned in
[`_EXPECTED_OFFSET`][tests.transpile.test_expected_offsets._EXPECTED_OFFSET],
and asserted as an equality rather than as a mere constancy.

Derivation
----------

Write `c(T, M) = mean_i (log p_QVR(z_i) - log p_T(z_i))` over the
gallery point set for target `T` and example `M`. A positive `c` means
the backend scores *lower* than the reference, so it dropped a term.
The question the registry answers is which terms each target is
entitled to drop.

**1. Normalizers of the target's own scoring API.** Every backend in
the Docker matrix is probed through an API that returns a fully
normalized joint. Stan is probed with
`cmdstanpy.CmdStanModel.log_prob(..., jacobian=False)`, so its value
is the constrained-space `target` accumulator with the
change-of-variables term removed, and the Stan renderer emits
`target += <family>_lpdf(<variate> | <args>);` increments rather than
`~` sampling statements, which is the form that retains every
normalizing constant Stan would otherwise be free to drop. NumPyro,
Pyro, PyMC, Edward2, Turing and Gen are probed through
`log_density` / `log_prob_sum` / `compile_logp` / `logjoint` /
`assess`, each documented to return normalized per-site densities.
JAGS and BUGS are scored through the JAGS graph, and WebPPL through
each distribution object's `score` method. Thus no target is entitled
to a constant on this account, and the API contributes zero.

**2. Truncation renormalizers of composed folded families.** This is
the one term that is genuinely dropped, and it is dropped by exactly
the targets that lack a native folded family. QVR's `HalfCauchy(gamma)`,
`HalfNormal(sigma)` and `HalfStudentT(df, scale)` are the folded
densities

    f_half(v) = 2 * f_base(v)    for v >= 0,

so `log f_half = log 2 + log f_base` at every point of the support.
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] records how
each target spells each family:

* `numpyro`, `pyro`, `pymc`, `edward2` name the native `HalfCauchy` /
  `HalfNormal` / `HalfStudentT` class, which carries the factor of two.
* `turing` composes `truncated(Cauchy(0, gamma), 0, Inf)`, and
  Distributions.jl's `truncated` divides by `1 - F(0) = 1/2`, which
  restores the same `log 2`.
* `bugs` and `jags` emit the symmetric base (`dt(0, tau, 1)`,
  `dnorm(0, tau)`) with a one-sided truncation suffix, and JAGS
  renormalizes over the truncation interval.
* `stan` emits the **symmetric base density alone** for all three
  (`cauchy_lpdf`, `normal_lpdf`, `student_t_lpdf`), with the
  non-negativity carried by a `real<lower=0>` declaration rather than
  by a renormalization. Each such site scores `log 2` below the
  reference.
* `gen` and `webppl` do the same for `HalfCauchy` and `HalfNormal`
  (`Gen.cauchy`, `Cauchy({...})`), but resolve `HalfStudentT` to a
  grafted runtime helper (`half_student_t`, `HalfStudentT`) whose
  scorer adds `log 2` explicitly, so they keep the renormalizer for
  that family alone. `turing` grafts the same helper.

The drop is therefore a property of the pair `(target, family)`, not of
the target alone, which is what
[`_DROPS_HALF_NORMALIZER`][tests.transpile.test_expected_offsets._DROPS_HALF_NORMALIZER]
records.

`log 2` does not depend on the parameter point or on the data, so the
drop is legitimate under Theorem 4.1. It is also *countable*: the
number of scalar folded-density factors of a given family a program
contains is a syntactic property of its QVR source, namely the sum,
over every `sample` / `observe` step naming that family, of the product
of the cardinalities of the axes attached to that step, counting steps
nested in a `marginalize` scope as well. Hence

    c(T, M) = sum_f drops_half(T, f) * n_f(M) * log 2,

with `drops_half(T, f) = 1` when `T` spells `f` as its bare symmetric
base and `0` otherwise.

**3. Everything else is zero.** Argument aliasing (`loc -> mu`,
`concentration -> a`) and parameterization substitution (BUGS and JAGS
precision `tau = 1/sigma^2`) are algebraic identities on the density,
not on a normalizer, so they contribute nothing. Plate expansion,
`filldist` / `arraydist`, `sample_shape`, and per-index `for` loops
all denote the same product measure. Change-of-variables terms are
either absent (identity `Psi` on every non-Stan target) or removed by
the probe (`jacobian=False`).

Validation legs
---------------

The pinned numbers rest on two independent legs, neither of which is
the other's restatement.

1. `test_registry_offset_matches_the_closed_form_derivation` recomputes
   `sum_f drops_half(T, f) * n_f(M) * log 2` from the example's `.qvr`
   source, in process and without Docker, and requires the registry to
   equal it exactly. A registry entry cannot be quietly retuned to
   whatever a container returned; it has to agree with a count taken
   off the program text.
2. `test_backend_offset_matches_registry` measures the offset in the
   pinned runtime and requires it to match the registry within the
   suite's equivalence tolerance. The tolerance floor is `5e-4`,
   roughly 1400 times smaller than a single dropped `log 2`, so one
   unaccounted truncation renormalizer, one missing prior factor, or
   any other point-independent term above a milli-nat fails the cell.

Unexplained offsets are never absorbed. An entry may be registered as
[`Unexplained`][tests.transpile.test_expected_offsets.Unexplained] to
record a measurement the derivation does not account for, and
`test_no_unexplained_offsets_are_registered` then fails until the cell
appears in
[`_ACKNOWLEDGED_UNEXPLAINED`][tests.transpile.test_expected_offsets._ACKNOWLEDGED_UNEXPLAINED],
whose sole purpose is to make such an admission loud.
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Sequence
from typing import Literal

import didactic.api as dx
import pytest

from quivers.dsl.ast_nodes import (
    DiscreteConstructor,
    ExportDecl,
    ExprIdent,
    MarginalizeStep,
    Module,
    ObjectDecl,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    SampleStep,
    TypeFromExpr,
)
from quivers.dsl.parser import parse
from quivers.transpile import transpile
from quivers.transpile.family_meta import FAMILY_META
from tests.transpile import _docker, _equivalence, _gallery_data
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import QvrProbe
from tests.transpile.test_gallery_numeric_equivalence import (
    _BACKENDS_WITH_IMAGES,
    _EXPECTED_TRANSPILE_RAISES,
    _SKIP_DATASET_LOAD_FAILED,
    _SKIP_PROBE_INCOMPATIBLE,
    _SKIP_QVR_INCOMPATIBLE,
    _dtypes_from_dataset,
    _gallery_cells,
    _shapes_from_dataset,
)


_LOG_2 = math.log(2.0)
"""The truncation renormalizer of a folded half-family, `log 2`.

`HalfCauchy`, `HalfNormal` and `HalfStudentT` are the base density
restricted to the non-negative half-line and rescaled by two. A target
that emits the symmetric base without a renormalization scores this
much below the reference at every point of the support."""

_HALF_FAMILIES: frozenset[str] = frozenset(
    {"HalfCauchy", "HalfNormal", "HalfStudentT"}
)
"""QVR families whose density carries a folding factor of two.

Membership is a statement about the QVR-side density, independent of
how any target spells the family. A new folded family added to
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] belongs here
and needs its own row in the drop table below.

`HalfStudentT` folds `student_t(df, 0, scale)` over the non-negative
half-line exactly as `HalfCauchy` folds `cauchy(0, gamma)`, so it
carries the same `log 2`, and it is spelled as the bare `student_t` on
Stan; its drop set is therefore narrower than the other two, which is
why the table below is keyed by family.

`Truncated` / `TruncatedNormal` stay outside the set: they renormalize
by `log(F(b) - F(a))`, which is a function of the bounds and of the
family's own arguments rather than a constant, so a target that drops
it fails the constant-spread leg whenever those arguments are latent
and needs a per-family term here when they are not."""

_DROPS_HALF_NORMALIZER: dict[str, frozenset[str]] = {
    "HalfCauchy": frozenset({"stan", "gen", "webppl"}),
    "HalfNormal": frozenset({"stan", "gen", "webppl"}),
    "HalfStudentT": frozenset({"stan"}),
}
"""Per folded family, the targets whose emit is the symmetric base
density with no truncation renormalization.

Read off
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META].
`HalfCauchy.target_names` maps `stan -> "cauchy"`, `gen -> "cauchy"`,
`webppl -> "Cauchy"`, all symmetric bases emitted without a truncation
wrapper, against `numpyro / pyro / pymc / edward2 -> "HalfCauchy"`
(native folded class), `turing -> "truncated"` (Distributions.jl
renormalizes) and `bugs / jags -> "dt"` carrying a one-sided truncation
suffix that JAGS renormalizes over.

`HalfStudentT` splits differently: only `stan -> "student_t"` is a bare
base. `gen -> "half_student_t"`, `turing -> "HalfStudentT"` and
`webppl -> "HalfStudentT"` all resolve to a grafted runtime helper
whose scorer adds `log 2` explicitly, so those three targets keep the
renormalizer for this family while dropping it for the other two. A
single per-target flag cannot express that, hence the keying by family.

`test_drop_table_agrees_with_the_family_registry` re-derives each set
from the family registry so a renderer that switches to a native or
truncated spelling cannot leave a stale entry behind."""


class OffsetJustification(dx.TaggedUnion, discriminator="kind"):
    """Why a registered cell carries the offset it carries.

    Two variants, and the distinction is load-bearing.
    [`Derived`][tests.transpile.test_expected_offsets.Derived] states
    that the module docstring's closed form accounts for the value, and
    the derivation test checks that claim against the QVR source.
    [`Unexplained`][tests.transpile.test_expected_offsets.Unexplained]
    records a measurement nobody has accounted for, which is a finding
    rather than a fact about the contract.
    """

    @property
    def summary(self) -> str:
        """One-line rendering for a failure message."""
        raise NotImplementedError


class Derived(OffsetJustification):
    """The docstring's closed form accounts for this offset.

    Attributes
    ----------
    kind : Literal["derived"]
        Discriminator.
    half_sites : int
        Number of scalar folded-family density factors in the example's
        QVR source, summed over every family in
        [`_HALF_FAMILIES`][tests.transpile.test_expected_offsets._HALF_FAMILIES].
        A property of the source alone, so it is identical across the
        targets of one example and is recorded even for a target that
        drops nothing: it pins the count the derivation test recomputes
        from the source.
    dropped_sites : int
        How many of those factors this target emits as a bare symmetric
        base, i.e. the subset whose family lists the target in
        [`_DROPS_HALF_NORMALIZER`][tests.transpile.test_expected_offsets._DROPS_HALF_NORMALIZER].
        The expected offset is `dropped_sites * log 2`. The two counts
        are kept apart rather than collapsed into one flag because a
        single example may mix families whose drop sets differ, as
        `HalfCauchy` and `HalfStudentT` do on Gen, Turing and WebPPL.
    """

    kind: Literal["derived"] = "derived"
    half_sites: int
    dropped_sites: int

    @property
    def summary(self) -> str:
        return (
            f"derived from {self.half_sites} folded-family density "
            f"factor(s) in the QVR source, {self.dropped_sites} of them "
            f"emitted without a truncation renormalizer"
        )


class Unexplained(OffsetJustification):
    """A measured offset the derivation does not account for.

    Attributes
    ----------
    kind : Literal["unexplained"]
        Discriminator.
    measured : float
        The offset observed in the pinned runtime.
    missing : str
        What it would take to explain the value: the term that would
        have to be identified in the renderer's emit, or the scoring-API
        normalizer that would have to be shown absent.
    """

    kind: Literal["unexplained"] = "unexplained"
    measured: float
    missing: str

    @property
    def summary(self) -> str:
        return (
            f"UNEXPLAINED: measured {self.measured:+.6f}, unaccounted "
            f"for by the derivation ({self.missing})"
        )


class ExpectedOffset(dx.Model):
    """One registry entry: the constant a cell is entitled to.

    Attributes
    ----------
    offset : float
        Expected value of `mean_i (log p_QVR(z_i) - log p_T(z_i))`. A
        positive value means the target scores below the reference,
        which is what dropping a term looks like.
    justification : OffsetJustification
        Why the value is what it is.
    """

    offset: float
    justification: OffsetJustification


def _derived(*, half_sites: int, dropped_sites: int) -> ExpectedOffset:
    """Registry entry for a cell the derivation accounts for.

    Parameters
    ----------
    half_sites
        Scalar folded-family density factors in the example's QVR
        source, over every family in
        [`_HALF_FAMILIES`][tests.transpile.test_expected_offsets._HALF_FAMILIES].
    dropped_sites
        How many of those the target emits as a bare symmetric base,
        per
        [`_DROPS_HALF_NORMALIZER`][tests.transpile.test_expected_offsets._DROPS_HALF_NORMALIZER].
    """
    return ExpectedOffset(
        offset=dropped_sites * _LOG_2,
        justification=Derived(
            half_sites=half_sites, dropped_sites=dropped_sites,
        ),
    )


# ----------------------------------------------------------------------
# The registry.
#
# Grouped by example. Within a group the `half_sites` count is a
# property of the QVR source and is therefore identical across targets;
# only `dropped_sites` varies, and it varies exactly with the family's
# entry in `_DROPS_HALF_NORMALIZER`.
#
# Population workflow for a newly-live cell: derive `half_sites` from
# the example's `.qvr` (every `sample` / `observe` step naming a folded
# family contributes the product of the cardinalities of the axes
# attached to it, nested `marginalize` scopes included), restrict to the
# families whose drop set contains the target to get `dropped_sites`,
# add the entry, and confirm the measured offset agrees. Never read the
# measurement first and write it down: the derivation is the authority
# and the measurement is its test.
# ----------------------------------------------------------------------
_EXPECTED_OFFSET: dict[tuple[str, str], ExpectedOffset] = {
    # ar1: 1 HalfCauchy folded factor(s).
    ('bugs', 'ar1'): _derived(half_sites=1, dropped_sites=0),
    ('edward2', 'ar1'): _derived(half_sites=1, dropped_sites=0),
    ('gen', 'ar1'): _derived(half_sites=1, dropped_sites=1),
    ('jags', 'ar1'): _derived(half_sites=1, dropped_sites=0),
    ('numpyro', 'ar1'): _derived(half_sites=1, dropped_sites=0),
    ('pymc', 'ar1'): _derived(half_sites=1, dropped_sites=0),
    ('pyro', 'ar1'): _derived(half_sites=1, dropped_sites=0),
    ('stan', 'ar1'): _derived(half_sites=1, dropped_sites=1),
    ('turing', 'ar1'): _derived(half_sites=1, dropped_sites=0),
    ('webppl', 'ar1'): _derived(half_sites=1, dropped_sites=1),
    # bayesian_regression: 1 HalfCauchy folded factor(s).
    ('bugs', 'bayesian_regression'): _derived(half_sites=1, dropped_sites=0),
    ('edward2', 'bayesian_regression'): _derived(half_sites=1, dropped_sites=0),
    ('gen', 'bayesian_regression'): _derived(half_sites=1, dropped_sites=1),
    ('jags', 'bayesian_regression'): _derived(half_sites=1, dropped_sites=0),
    ('numpyro', 'bayesian_regression'): _derived(half_sites=1, dropped_sites=0),
    ('pymc', 'bayesian_regression'): _derived(half_sites=1, dropped_sites=0),
    ('pyro', 'bayesian_regression'): _derived(half_sites=1, dropped_sites=0),
    ('stan', 'bayesian_regression'): _derived(half_sites=1, dropped_sites=1),
    ('turing', 'bayesian_regression'): _derived(half_sites=1, dropped_sites=0),
    ('webppl', 'bayesian_regression'): _derived(half_sites=1, dropped_sites=1),
    # beta_regression: 3 HalfCauchy folded factor(s).
    ('bugs', 'beta_regression'): _derived(half_sites=3, dropped_sites=0),
    ('edward2', 'beta_regression'): _derived(half_sites=3, dropped_sites=0),
    ('gen', 'beta_regression'): _derived(half_sites=3, dropped_sites=3),
    ('jags', 'beta_regression'): _derived(half_sites=3, dropped_sites=0),
    ('numpyro', 'beta_regression'): _derived(half_sites=3, dropped_sites=0),
    ('pymc', 'beta_regression'): _derived(half_sites=3, dropped_sites=0),
    ('pyro', 'beta_regression'): _derived(half_sites=3, dropped_sites=0),
    ('stan', 'beta_regression'): _derived(half_sites=3, dropped_sites=3),
    ('turing', 'beta_regression'): _derived(half_sites=3, dropped_sites=0),
    ('webppl', 'beta_regression'): _derived(half_sites=3, dropped_sites=3),
    # changepoint: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ('bugs', 'changepoint'): _derived(half_sites=0, dropped_sites=0),
    ('edward2', 'changepoint'): _derived(half_sites=0, dropped_sites=0),
    ('gen', 'changepoint'): _derived(half_sites=0, dropped_sites=0),
    ('jags', 'changepoint'): _derived(half_sites=0, dropped_sites=0),
    ('numpyro', 'changepoint'): _derived(half_sites=0, dropped_sites=0),
    ('pymc', 'changepoint'): _derived(half_sites=0, dropped_sites=0),
    ('pyro', 'changepoint'): _derived(half_sites=0, dropped_sites=0),
    ('stan', 'changepoint'): _derived(half_sites=0, dropped_sites=0),
    ('turing', 'changepoint'): _derived(half_sites=0, dropped_sites=0),
    ('webppl', 'changepoint'): _derived(half_sites=0, dropped_sites=0),
    # factor_analysis: 1 HalfCauchy folded factor(s).
    ('bugs', 'factor_analysis'): _derived(half_sites=1, dropped_sites=0),
    ('edward2', 'factor_analysis'): _derived(half_sites=1, dropped_sites=0),
    ('gen', 'factor_analysis'): _derived(half_sites=1, dropped_sites=1),
    ('jags', 'factor_analysis'): _derived(half_sites=1, dropped_sites=0),
    ('numpyro', 'factor_analysis'): _derived(half_sites=1, dropped_sites=0),
    ('pymc', 'factor_analysis'): _derived(half_sites=1, dropped_sites=0),
    ('pyro', 'factor_analysis'): _derived(half_sites=1, dropped_sites=0),
    ('stan', 'factor_analysis'): _derived(half_sites=1, dropped_sites=1),
    ('turing', 'factor_analysis'): _derived(half_sites=1, dropped_sites=0),
    # gamma_regression: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ('bugs', 'gamma_regression'): _derived(half_sites=0, dropped_sites=0),
    ('edward2', 'gamma_regression'): _derived(half_sites=0, dropped_sites=0),
    ('gen', 'gamma_regression'): _derived(half_sites=0, dropped_sites=0),
    ('jags', 'gamma_regression'): _derived(half_sites=0, dropped_sites=0),
    ('numpyro', 'gamma_regression'): _derived(half_sites=0, dropped_sites=0),
    ('pymc', 'gamma_regression'): _derived(half_sites=0, dropped_sites=0),
    ('pyro', 'gamma_regression'): _derived(half_sites=0, dropped_sites=0),
    ('stan', 'gamma_regression'): _derived(half_sites=0, dropped_sites=0),
    ('turing', 'gamma_regression'): _derived(half_sites=0, dropped_sites=0),
    ('webppl', 'gamma_regression'): _derived(half_sites=0, dropped_sites=0),
    # hmm: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ('numpyro', 'hmm'): _derived(half_sites=0, dropped_sites=0),
    ('pyro', 'hmm'): _derived(half_sites=0, dropped_sites=0),
    # horseshoe_regression: 6 HalfCauchy folded factor(s).
    ('bugs', 'horseshoe_regression'): _derived(half_sites=6, dropped_sites=0),
    ('edward2', 'horseshoe_regression'): _derived(half_sites=6, dropped_sites=0),
    ('gen', 'horseshoe_regression'): _derived(half_sites=6, dropped_sites=6),
    ('jags', 'horseshoe_regression'): _derived(half_sites=6, dropped_sites=0),
    ('numpyro', 'horseshoe_regression'): _derived(half_sites=6, dropped_sites=0),
    ('pymc', 'horseshoe_regression'): _derived(half_sites=6, dropped_sites=0),
    ('pyro', 'horseshoe_regression'): _derived(half_sites=6, dropped_sites=0),
    ('stan', 'horseshoe_regression'): _derived(half_sites=6, dropped_sites=6),
    ('turing', 'horseshoe_regression'): _derived(half_sites=6, dropped_sites=0),
    ('webppl', 'horseshoe_regression'): _derived(half_sites=6, dropped_sites=6),
    # irt_2pl: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ('bugs', 'irt_2pl'): _derived(half_sites=0, dropped_sites=0),
    ('edward2', 'irt_2pl'): _derived(half_sites=0, dropped_sites=0),
    ('gen', 'irt_2pl'): _derived(half_sites=0, dropped_sites=0),
    ('jags', 'irt_2pl'): _derived(half_sites=0, dropped_sites=0),
    ('numpyro', 'irt_2pl'): _derived(half_sites=0, dropped_sites=0),
    ('pymc', 'irt_2pl'): _derived(half_sites=0, dropped_sites=0),
    ('pyro', 'irt_2pl'): _derived(half_sites=0, dropped_sites=0),
    ('stan', 'irt_2pl'): _derived(half_sites=0, dropped_sites=0),
    ('turing', 'irt_2pl'): _derived(half_sites=0, dropped_sites=0),
    ('webppl', 'irt_2pl'): _derived(half_sites=0, dropped_sites=0),
    # lda: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ('edward2', 'lda'): _derived(half_sites=0, dropped_sites=0),
    ('numpyro', 'lda'): _derived(half_sites=0, dropped_sites=0),
    ('pymc', 'lda'): _derived(half_sites=0, dropped_sites=0),
    ('pyro', 'lda'): _derived(half_sites=0, dropped_sites=0),
    # negbin_regression: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ('bugs', 'negbin_regression'): _derived(half_sites=0, dropped_sites=0),
    ('edward2', 'negbin_regression'): _derived(half_sites=0, dropped_sites=0),
    ('gen', 'negbin_regression'): _derived(half_sites=0, dropped_sites=0),
    ('jags', 'negbin_regression'): _derived(half_sites=0, dropped_sites=0),
    ('numpyro', 'negbin_regression'): _derived(half_sites=0, dropped_sites=0),
    ('pymc', 'negbin_regression'): _derived(half_sites=0, dropped_sites=0),
    ('pyro', 'negbin_regression'): _derived(half_sites=0, dropped_sites=0),
    ('stan', 'negbin_regression'): _derived(half_sites=0, dropped_sites=0),
    ('turing', 'negbin_regression'): _derived(half_sites=0, dropped_sites=0),
    ('webppl', 'negbin_regression'): _derived(half_sites=0, dropped_sites=0),
    # ppca: 1 HalfCauchy folded factor(s).
    ('bugs', 'ppca'): _derived(half_sites=1, dropped_sites=0),
    ('edward2', 'ppca'): _derived(half_sites=1, dropped_sites=0),
    ('gen', 'ppca'): _derived(half_sites=1, dropped_sites=1),
    ('jags', 'ppca'): _derived(half_sites=1, dropped_sites=0),
    ('numpyro', 'ppca'): _derived(half_sites=1, dropped_sites=0),
    ('pymc', 'ppca'): _derived(half_sites=1, dropped_sites=0),
    ('pyro', 'ppca'): _derived(half_sites=1, dropped_sites=0),
    ('stan', 'ppca'): _derived(half_sites=1, dropped_sites=1),
    ('turing', 'ppca'): _derived(half_sites=1, dropped_sites=0),
    # stochastic_volatility: 1 HalfCauchy folded factor(s).
    ('bugs', 'stochastic_volatility'): _derived(half_sites=1, dropped_sites=0),
    ('edward2', 'stochastic_volatility'): _derived(half_sites=1, dropped_sites=0),
    ('gen', 'stochastic_volatility'): _derived(half_sites=1, dropped_sites=1),
    ('jags', 'stochastic_volatility'): _derived(half_sites=1, dropped_sites=0),
    ('numpyro', 'stochastic_volatility'): _derived(half_sites=1, dropped_sites=0),
    ('pymc', 'stochastic_volatility'): _derived(half_sites=1, dropped_sites=0),
    ('pyro', 'stochastic_volatility'): _derived(half_sites=1, dropped_sites=0),
    ('stan', 'stochastic_volatility'): _derived(half_sites=1, dropped_sites=1),
    ('turing', 'stochastic_volatility'): _derived(half_sites=1, dropped_sites=0),
    # survival_weibull: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ('bugs', 'survival_weibull'): _derived(half_sites=0, dropped_sites=0),
    ('edward2', 'survival_weibull'): _derived(half_sites=0, dropped_sites=0),
    ('gen', 'survival_weibull'): _derived(half_sites=0, dropped_sites=0),
    ('jags', 'survival_weibull'): _derived(half_sites=0, dropped_sites=0),
    ('numpyro', 'survival_weibull'): _derived(half_sites=0, dropped_sites=0),
    ('pymc', 'survival_weibull'): _derived(half_sites=0, dropped_sites=0),
    ('pyro', 'survival_weibull'): _derived(half_sites=0, dropped_sites=0),
    ('stan', 'survival_weibull'): _derived(half_sites=0, dropped_sites=0),
    ('turing', 'survival_weibull'): _derived(half_sites=0, dropped_sites=0),
    ('webppl', 'survival_weibull'): _derived(half_sites=0, dropped_sites=0),
    # zip_regression: no folded-family site, so every target is
    # entitled to nothing and scores the reference exactly.
    ('edward2', 'zip_regression'): _derived(half_sites=0, dropped_sites=0),
    ('pymc', 'zip_regression'): _derived(half_sites=0, dropped_sites=0),
    ('pyro', 'zip_regression'): _derived(half_sites=0, dropped_sites=0),
    ('turing', 'zip_regression'): _derived(half_sites=0, dropped_sites=0),
}


_ACKNOWLEDGED_UNEXPLAINED: frozenset[tuple[str, str]] = frozenset()
"""Cells whose registered offset the derivation does not explain.

An entry here is an admission that a backend adds or drops a term
nobody has accounted for, which is a correctness finding and not a
property of the contract. The set is empty, and
`test_no_unexplained_offsets_are_registered` keeps it that way unless
someone deliberately writes a cell into both this set and the registry.
"""


def _live_cells() -> list[tuple[str, str]]:
    """Every `(backend, example)` cell the gallery equivalence test
    currently evaluates numerically.

    Derived from the same registries
    [`test_gallery_numeric_equivalence`][tests.transpile.test_gallery_numeric_equivalence]
    consults, so a cell that becomes live there becomes live here on the
    same commit and
    `test_every_live_cell_has_a_registered_offset` fails until its
    offset is derived and registered.
    """
    out: list[tuple[str, str]] = []
    for example in _gallery_cells():
        stem = example.stem
        if stem in _SKIP_DATASET_LOAD_FAILED:
            continue
        if stem in _SKIP_QVR_INCOMPATIBLE:
            continue
        for backend in sorted(_BACKENDS_WITH_IMAGES):
            if (backend, stem) in _EXPECTED_TRANSPILE_RAISES:
                continue
            if (backend, stem) in _SKIP_PROBE_INCOMPATIBLE:
                continue
            out.append((backend, stem))
    return out


def _example_path(stem: str) -> pathlib.Path:
    """The gallery `.qvr` source for `stem`."""
    for example in _gallery_cells():
        if example.stem == stem:
            return example
    raise AssertionError(
        f"{stem!r} is not a gallery example with a synthetic-data "
        f"block; `_live_cells` should never have produced it"
    )


def _finset_cardinalities(module: Module) -> dict[str, int]:
    """Cardinality of every `object <name> : FinSet <n>` declaration.

    Only plain `FinSet` declarations are collected. An axis declared as
    a product, coproduct or free monoid is absent from the map, and
    [`half_family_factor_counts`][tests.transpile.test_expected_offsets.half_family_factor_counts]
    raises rather than guess when a folded site indexes one.
    """
    out: dict[str, int] = {}
    for statement in module.statements:
        if not isinstance(statement, ObjectDecl):
            continue
        init = statement.init
        if not isinstance(init, TypeFromExpr):
            continue
        expr = init.expr
        if not isinstance(expr, DiscreteConstructor):
            continue
        if expr.constructor != "FinSet" or len(expr.args) != 1:
            continue
        for name in statement.names:
            out[name] = int(expr.args[0])
    return out


def _exported_program(module: Module, stem: str) -> ProgramDecl:
    """The `ProgramDecl` the module's `export` names.

    The exported program's name need not match the file stem
    (`ppca.qvr` exports `ppca_program`), so the export declaration is
    the authority. A module may export more than one name, since the
    probabilistic surface often sits alongside the pure composite it
    scores (`hmm.qvr` exports both `hmm` and `hmm_program`); only the
    exports naming a `program` declaration are candidates, and exactly
    one of those must exist for the count to be well defined.
    """
    programs = {
        statement.name: statement
        for statement in module.statements
        if isinstance(statement, ProgramDecl)
    }
    exported: list[str] = []
    for statement in module.statements:
        if not isinstance(statement, ExportDecl):
            continue
        expr = statement.expr
        if isinstance(expr, ExprIdent) and expr.name in programs:
            exported.append(expr.name)
    if len(exported) != 1:
        raise AssertionError(
            f"{stem!r}: expected exactly one `export <ident>` naming a "
            f"`program` declaration, found {exported!r}. The "
            f"folded-family count is taken off the exported program and "
            f"cannot be resolved without one."
        )
    return programs[exported[0]]


def _axis_names(step: SampleStep | ObserveStep) -> tuple[str, ...]:
    """Every axis attached to a draw step, batch and event alike.

    A folded family is scored elementwise, so the folding factor of two
    applies once per coordinate whether the coordinate sits on a batch
    axis (`sample s : Coef <- HalfCauchy(1.0)`) or on an event axis
    (`[over=...]`). Both therefore multiply the factor count. When an
    `[over=..., iid_over=...]` clause is present it is authoritative and
    the bare index is not counted again, since the index axis reappears
    inside `iid_over`.
    """
    axes = step.axes
    if axes is not None:
        return tuple(str(axis) for axis in axes.over) + tuple(
            str(axis) for axis in axes.iid_over
        )
    index = step.index
    if index is None:
        return ()
    return (str(index.name),)


def _flatten_steps(
    steps: Sequence[ProgramStep],
) -> list[ProgramStep]:
    """Every step of a program body, including those nested in a
    `marginalize` scope.

    A folded-family site inside a marginalized scope still contributes
    its `log 2`. Under a `logsumexp` reduction the constant factor is
    common to every branch of the sum and so factors straight back out
    of the reduction; under a fibred reduction it factors out of each
    fibre alike. Either way the site counts exactly once per coordinate,
    as it would at the top level, so a flat walk of the nested steps is
    the right count.
    """
    out: list[ProgramStep] = []
    for step in steps:
        out.append(step)
        if isinstance(step, MarginalizeStep):
            out.extend(_flatten_steps(step.scope))
    return out


def half_family_factor_counts(example: pathlib.Path) -> dict[str, int]:
    """Scalar folded-family density factors in a QVR source, by family.

    Sums, over every `sample` / `observe` step of the exported program
    that names a family in
    [`_HALF_FAMILIES`][tests.transpile.test_expected_offsets._HALF_FAMILIES],
    the number of variables the step binds times the product of the
    cardinalities of the axes attached to it. Steps nested in a
    `marginalize` scope are walked too. Families with no site in the
    source are absent from the returned map.

    The breakdown is per family rather than a single total because the
    targets that drop the renormalizer differ between families: Gen,
    Turing and WebPPL emit a bare base for `HalfCauchy` and `HalfNormal`
    but a folded runtime helper for `HalfStudentT`.

    Raises
    ------
    AssertionError
        If the exported program draws from another program declared in
        the same module (a folded site could hide inside the callee, and
        the count would silently miss it), if a `marginalize` head names
        a folded family (the reduction's own treatment of the constant
        would need its own derivation), or if a folded site indexes an
        axis that is not a plain `FinSet`.
    """
    module = parse(example.read_text())
    program = _exported_program(module, example.stem)
    cardinalities = _finset_cardinalities(module)
    sub_programs = {
        statement.name
        for statement in module.statements
        if isinstance(statement, ProgramDecl)
        and statement.name != program.name
    }

    counts: dict[str, int] = {}
    for step in _flatten_steps(program.draws):
        if isinstance(step, MarginalizeStep):
            if step.morphism in _HALF_FAMILIES:
                raise AssertionError(
                    f"{example.stem!r}: `marginalize {step.var}` draws "
                    f"from the folded family {step.morphism!r}. Whether "
                    f"the folding factor survives the reduction depends "
                    f"on the reduction, so the flat per-site count does "
                    f"not apply; derive the term for this step before "
                    f"registering an offset for this example."
                )
            continue
        if not isinstance(step, (SampleStep, ObserveStep)):
            continue
        if step.morphism in sub_programs:
            raise AssertionError(
                f"{example.stem!r}: step {step.vars!r} draws from the "
                f"sub-program {step.morphism!r}, so a folded-family site "
                f"may sit inside the callee and the syntactic count "
                f"would miss it. Extend "
                f"`half_family_factor_counts` to descend into the "
                f"callee before registering an offset for this example."
            )
        if step.morphism not in _HALF_FAMILIES:
            continue
        multiplicity = len(step.vars)
        for axis in _axis_names(step):
            if axis not in cardinalities:
                raise AssertionError(
                    f"{example.stem!r}: folded step {step.vars!r} "
                    f"indexes axis {axis!r}, which is not declared as a "
                    f"plain `FinSet`. Its cardinality decides how many "
                    f"`log 2` factors the site carries, so the "
                    f"derivation cannot proceed without it."
                )
            multiplicity *= cardinalities[axis]
        counts[step.morphism] = counts.get(step.morphism, 0) + multiplicity
    return counts


def _dropped_factor_count(backend: str, stem: str) -> int:
    """Folded factors `backend` emits without a renormalizer, for
    `stem`.

    The per-family counts of the source restricted to the families whose
    drop set contains the target.
    """
    counts = half_family_factor_counts(_example_path(stem))
    return sum(
        count
        for family, count in counts.items()
        if backend in _DROPS_HALF_NORMALIZER[family]
    )


def _derived_offset(backend: str, stem: str) -> float:
    """The closed-form expected offset for a cell.

    `sum_f drops_half(T, f) * n_f(M) * log 2` per the module docstring.
    """
    return _dropped_factor_count(backend, stem) * _LOG_2


def _offset_atol(dataset: _gallery_data.GalleryDataset) -> float:
    """Tolerance on the measured-versus-registered offset comparison.

    The same estimator the constant-spread check uses, driven by the
    example's observed-data count. The offset is a mean of pointwise
    differences, so it inherits the per-observation round-off the
    estimator models plus a small systematic bias between the
    reference's and the backend's summation order, which is what the
    `5e-4` floor absorbs. The floor is roughly `log(2) / 1386`, so a
    single unaccounted truncation renormalizer, and any other
    point-independent term above a milli-nat, still fails the cell.
    """
    n_obs = sum(
        int(dataset.observations[name].numel())
        for name in _gallery_data.observed_data_names(dataset)
    )
    return _equivalence.adaptive_atol(n_obs=n_obs)


_QVR_LOG_DENSITY_CACHE: dict[str, list[float]] = {}
"""Per-example QVR reference values at the gallery point set.

The reference does not depend on the backend, so the ten cells of one
example share a single in-process evaluation. Keyed by example stem;
the point set is deterministic, so the cache is a pure memoisation.
"""


def _qvr_log_densities(
    example: pathlib.Path,
    dataset: _gallery_data.GalleryDataset,
    points: list[Point],
    scratch: pathlib.Path,
) -> list[float]:
    """QVR reference log-densities at every point of `points`.

    One probe call per point. The probe's `observations` keyword
    overrides the flat per-point payload, so a perturbed point needs its
    own pre-shaped observation dict; passing the dataset's ground-truth
    observations once for the whole set would score the reference at the
    unperturbed data while the container scored the perturbed data, and
    manufacture an offset out of the mismatch.
    """
    cached = _QVR_LOG_DENSITY_CACHE.get(example.stem)
    if cached is not None:
        return cached
    probe = QvrProbe()
    source = example.read_bytes()
    values: list[float] = []
    for point in points:
        values.extend(
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
    _QVR_LOG_DENSITY_CACHE[example.stem] = values
    return values


def test_drop_table_agrees_with_the_family_registry() -> None:
    """Every target in
    [`_DROPS_HALF_NORMALIZER`][tests.transpile.test_expected_offsets._DROPS_HALF_NORMALIZER]
    spells the half-families as a symmetric base, and no target outside
    it does.

    The drop table is the hinge of the whole derivation: it decides
    which cells expect `n * log 2` and which expect zero. Pinning it as
    a literal set and never re-checking it would let a renderer switch
    `HalfCauchy` from `cauchy` to a native or truncated spelling while
    the registry kept charging the cell a `log 2` it no longer drops.
    The reconciliation here is against
    [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META], which is
    the single source of truth for target spellings.
    """
    symmetric_bases = {
        "cauchy", "Cauchy", "normal", "Normal", "gaussian", "Gaussian",
        "student_t", "StudentT",
    }
    assert set(_DROPS_HALF_NORMALIZER) == set(_HALF_FAMILIES), (
        f"the drop table keys {sorted(_DROPS_HALF_NORMALIZER)} but the "
        f"folded families are {sorted(_HALF_FAMILIES)}. Every folded "
        f"family needs its own drop set; the two differ."
    )
    for family in sorted(_HALF_FAMILIES):
        meta = FAMILY_META[family]
        bare_base: set[str] = set()
        for backend in sorted(_BACKENDS_WITH_IMAGES):
            target_name = meta.target_names.get(backend)
            assert target_name is not None, (
                f"{family!r} has no {backend!r} target name in "
                f"`FAMILY_META`, yet {backend!r} is in the Docker "
                f"matrix. Either the family lost its spelling or the "
                f"matrix gained a backend; the derivation cannot "
                f"decide whether the cell drops a `log 2` without it."
            )
            if target_name in symmetric_bases:
                bare_base.add(backend)

        # BUGS and JAGS name the symmetric `dt` / `dnorm` but attach a
        # one-sided truncation suffix the renderer emits at the site,
        # which JAGS renormalizes over; the spelling alone therefore
        # does not settle them and the `dt` / `dnorm` names keep them
        # out of `bare_base` above. Gen, Turing and WebPPL name
        # `half_student_t` / `HalfStudentT`, each a grafted runtime
        # helper whose scorer adds the `log 2` itself.
        assert bare_base == _DROPS_HALF_NORMALIZER[family], (
            f"targets spelling {family!r} as a bare symmetric base are "
            f"{sorted(bare_base)}, but `_DROPS_HALF_NORMALIZER` claims "
            f"{sorted(_DROPS_HALF_NORMALIZER[family])}. A renderer "
            f"changed its emit for this family; re-derive the entitled "
            f"constant for every affected cell before editing either "
            f"side."
        )


@pytest.mark.parametrize(
    "cell", _live_cells(), ids=lambda c: f"{c[0]}-{c[1]}"
)
def test_every_live_cell_has_a_registered_offset(
    cell: tuple[str, str],
) -> None:
    """Every numerically-evaluated gallery cell names its constant.

    A cell that starts passing without an entry here would be checked
    only for constant spread, which is the existential statement this
    module exists to strengthen. Growing the coverage of
    [`test_gallery_numeric_equivalence`][tests.transpile.test_gallery_numeric_equivalence]
    therefore fails this test until the new cell's offset is derived
    and registered, and new coverage cannot bypass the named-constant
    criterion by arriving unannounced.
    """
    backend, stem = cell
    assert cell in _EXPECTED_OFFSET, (
        f"{backend!r} on {stem!r} is evaluated numerically by the "
        f"gallery equivalence suite but has no entry in "
        f"`_EXPECTED_OFFSET`, so its additive constant is unnamed and "
        f"only its spread is checked. Derive the constant: count the "
        f"scalar folded-family density factors in "
        f"`docs/examples/source/{stem}.qvr` (each `sample` / `observe` "
        f"step naming a family in `_HALF_FAMILIES` contributes the "
        f"product of the cardinalities of its axes), keep the ones "
        f"whose family lists {backend!r} in `_DROPS_HALF_NORMALIZER`, "
        f"multiply that subtotal by `log 2`, and "
        f"register the result with a `Derived` justification. If the "
        f"measured offset disagrees with that number, register it as "
        f"`Unexplained` and report the discrepancy rather than tuning "
        f"the entry to match."
    )


def test_registry_has_no_entries_for_dead_cells() -> None:
    """The registry names no cell the gallery suite stopped evaluating.

    A stale entry is a claim about a measurement nobody takes any more.
    It would survive a renderer regression that pushed the cell into
    `_EXPECTED_TRANSPILE_RAISES` or `_SKIP_PROBE_INCOMPATIBLE`, and
    would read as coverage that no longer exists.
    """
    live = set(_live_cells())
    stale = sorted(set(_EXPECTED_OFFSET) - live)
    assert not stale, (
        f"`_EXPECTED_OFFSET` registers {stale!r}, which the gallery "
        f"equivalence suite no longer evaluates (the cell moved into "
        f"`_EXPECTED_TRANSPILE_RAISES`, one of the `_SKIP_*` "
        f"registries, or the example lost its synthetic-data block). "
        f"Drop the entries; a registered constant nobody measures is "
        f"not coverage."
    )


@pytest.mark.parametrize(
    "cell", sorted(_EXPECTED_OFFSET), ids=lambda c: f"{c[0]}-{c[1]}"
)
def test_registry_offset_matches_the_closed_form_derivation(
    cell: tuple[str, str],
) -> None:
    """A registered constant equals what the derivation predicts.

    This is the leg that keeps the registry honest without a container.
    The prediction is recomputed from the example's QVR source, so the
    pinned number cannot drift toward whatever a runtime happened to
    return: it has to stay equal to `drops_half(T) * n_half(M) * log 2`.
    An entry the derivation does not cover is registered as
    `Unexplained` and skipped here, and
    `test_no_unexplained_offsets_are_registered` then makes it visible.
    """
    backend, stem = cell
    entry = _EXPECTED_OFFSET[cell]
    justification = entry.justification
    if isinstance(justification, Unexplained):
        pytest.skip(
            f"{backend!r} on {stem!r}: registered as unexplained "
            f"({justification.missing}); "
            f"`test_no_unexplained_offsets_are_registered` reports it."
        )
    assert isinstance(justification, Derived), (
        f"{backend!r} on {stem!r}: justification "
        f"{type(justification).__name__!r} is neither `Derived` nor "
        f"`Unexplained`; the registry admits no third status."
    )

    counts = half_family_factor_counts(_example_path(stem))
    counted = sum(counts.values())
    assert counted == justification.half_sites, (
        f"{backend!r} on {stem!r}: the registry records "
        f"{justification.half_sites} folded-family density factor(s), "
        f"but `docs/examples/source/{stem}.qvr` now carries {counted} "
        f"({counts!r}). The example changed; re-derive the entitled "
        f"constant for every cell of this example."
    )
    dropped = _dropped_factor_count(backend, stem)
    assert dropped == justification.dropped_sites, (
        f"{backend!r} on {stem!r}: the registry records "
        f"{justification.dropped_sites} factor(s) emitted without a "
        f"renormalizer, but the drop table gives {dropped} for the "
        f"source's {counts!r}. Re-derive the entitled constant."
    )
    predicted = _derived_offset(backend, stem)
    assert entry.offset == pytest.approx(predicted, abs=1e-12), (
        f"{backend!r} on {stem!r}: registry pins offset "
        f"{entry.offset!r} but the derivation predicts {predicted!r} "
        f"({counted} folded-family factor(s) of which {dropped} lose "
        f"the truncation renormalizer on {backend!r}). The registry "
        f"must state the constant the target is entitled to, not the "
        f"constant it happened to produce."
    )


def test_no_unexplained_offsets_are_registered() -> None:
    """No cell carries an offset the derivation cannot account for.

    An `Unexplained` entry means the backend adds or drops a term
    nobody has identified, which is a correctness finding: the emitted
    program may denote a different measure and the constant-spread
    check would never say so. Recording one is allowed, so the finding
    can be pinned with its measured value while it is investigated, but
    it must also be written into `_ACKNOWLEDGED_UNEXPLAINED`, and this
    test fails until it is. The acknowledgement set is empty today.
    """
    unexplained = {
        cell: entry.justification
        for cell, entry in _EXPECTED_OFFSET.items()
        if isinstance(entry.justification, Unexplained)
    }
    surprises = sorted(set(unexplained) - _ACKNOWLEDGED_UNEXPLAINED)
    assert not surprises, (
        "cells registered with an offset the derivation does not "
        "explain: "
        + "; ".join(
            f"{backend}@{stem} {unexplained[(backend, stem)].summary}"
            for backend, stem in surprises
        )
        + ". Each is a backend term nobody has accounted for. Identify "
        "the term in the renderer's emit and re-register the cell as "
        "`Derived`, or, to carry the finding while it is investigated, "
        "add the cell to `_ACKNOWLEDGED_UNEXPLAINED`."
    )
    resolved = sorted(_ACKNOWLEDGED_UNEXPLAINED - set(unexplained))
    assert not resolved, (
        f"`_ACKNOWLEDGED_UNEXPLAINED` still lists {resolved!r}, but "
        f"those cells no longer carry an `Unexplained` justification. "
        f"Drop them from the acknowledgement set so it keeps naming "
        f"only live findings."
    )


@pytest.mark.requires_docker
@pytest.mark.parametrize(
    "cell", sorted(_EXPECTED_OFFSET), ids=lambda c: f"{c[0]}-{c[1]}"
)
def test_backend_offset_matches_registry(cell: tuple[str, str]) -> None:
    """The measured constant equals the registered one.

    Runs the pinned runtime over the same deterministic point set the
    gallery equivalence test uses, takes the mean of
    `log p_QVR - log p_T`, and requires it to equal the registry value.
    This upgrades the cell's contract from "some constant" to "this
    constant": a backend that drops a prior factor, double-counts a
    normalizer, or scores a family at a shifted parameterization moves
    the mean without moving the spread, and passes the existential
    check while failing this one.

    The spread is asserted first, since a non-constant difference makes
    the mean meaningless.
    """
    backend, stem = cell
    image, source_ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.docker_available():
        raise RuntimeError(
            "docker daemon not reachable; the session-scope "
            "`_ensure_docker_environment` autouse fixture should have "
            "started it"
        )
    if not _docker.image_available(image):
        raise RuntimeError(
            f"docker image {image!r} not available; the session-scope "
            f"`_ensure_docker_environment` autouse fixture should have "
            f"built it"
        )

    example = _example_path(stem)
    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{stem!r}: `load_gallery_data` returned None even though the "
        f"example is not in `_SKIP_DATASET_LOAD_FAILED`."
    )

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    scratch = pathlib.Path("/tmp") / f"qvr_offset_{stem}_{backend}"
    scratch.mkdir(exist_ok=True, parents=True)

    qvr_lps = _qvr_log_densities(example, dataset, points, scratch)
    emitted = transpile(parse(example.read_text()), target=backend)
    script_path = (
        pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    )
    raw_result = _docker.run_probe(
        image=image,
        script=script_path,
        source=emitted,
        source_ext=source_ext,
        points=[
            {"params": point.params, "data": point.data}
            for point in points
        ],
        scratch=scratch,
        shapes=_shapes_from_dataset(dataset),
        dtypes=_dtypes_from_dataset(dataset),
        timeout=600.0,
    )
    backend_lps = [float(x) for x in raw_result["log_densities"]]

    atol = _offset_atol(dataset)
    # `assert_log_density_match` returns mean(target - qvr); the
    # registry states the reference-minus-target sign, so negate.
    measured = -_equivalence.assert_log_density_match(
        qvr_lps,
        backend_lps,
        atol=atol,
        context=f"{backend}@{stem} (named-constant)",
        labels=labels,
        min_points=2,
    )

    entry = _EXPECTED_OFFSET[cell]
    assert abs(measured - entry.offset) <= atol, (
        f"{backend}@{stem}: measured additive constant "
        f"{measured:+.6f} but the cell is entitled to "
        f"{entry.offset:+.6f} ({entry.justification.summary}); "
        f"discrepancy {measured - entry.offset:+.6e} exceeds "
        f"{atol:.3e}. The difference is point-independent, so the "
        f"constant-spread check cannot see it: the backend scores a "
        f"term the reference does not, or drops one the derivation "
        f"does not entitle it to drop. In units of a truncation "
        f"renormalizer the discrepancy is "
        f"{(measured - entry.offset) / _LOG_2:+.4f} x log 2."
    )
