"""Tier-4 numeric equivalence on the documentation gallery.

For every `docs/examples/source/<example>.qvr` that ships a `###
Generating synthetic data` block in its sibling `.md`, this test:

1. Extracts and executes the data-gen snippet to recover an
   `observations` dict plus every captured `true_*` ground-truth
   parameter value.
2. Builds a multi-point
   [`Point`][tests.transpile.probes._protocol.Point] set via
   [`points_from_dataset`][tests.transpile._gallery_data.points_from_dataset]:
   the ground truth first, then deterministic in-support
   perturbations of the latents, of the observed data, and of both.
3. Runs the in-process
   [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe] to compute
   `log p_QVR(θ, y) = sum_i log f_i(...)` at every point.
4. For every backend whose Docker image is locally built, runs the
   target's native log-density probe inside the container over the
   same point set and asserts constant-spread equivalence
   (`max_i | δ_i − mean δ | < atol`) per Theorem 4.1 of
   [docs/semantics/transpile-correctness.md](../../docs/semantics/transpile-correctness.md).

The point set is what gives the constant-spread assertion its teeth.
Evaluated at one point the spread is identically zero, so the check
passes whatever the backend computed. Varying the latents catches a
mis-scored prior; varying the *data* catches a backend that drops a
data-dependent term while keeping a stable offset as the latents move
(a Stan `~` sampling statement discards data-only summands, for
instance). Both must vary before the assertion means anything.

Each cell resolves to one of three pre-declared outcomes:

1. `(backend, example) in _EXPECTED_TRANSPILE_RAISES`: the
   pipeline MUST `pytest.raises(UnsupportedConstruct)` with the
   pinned kind-prefix.
2. Cell falls in one of the four `_SKIP_*` registries: a known
   environmental gap (missing data block, QVR-probe incompatibility,
   backend probe script lacking shape registration for arbitrary
   gallery datasets). `pytest.skip` with the diagnostic.
3. Neither: the pipeline MUST emit non-empty bytes, the QVR probe
   MUST evaluate to a finite log-density, and the in-container
   probe MUST return a vector whose constant-spread offset from
   the QVR reference is below the equivalence tolerance.

When neither Docker images nor backend Python runtimes are
available (the typical local-dev state), the test still exercises
the QVR-side trace: every gallery example whose `.md` ships data
gets a verified `log p_QVR` value, which is the strongest in-process
correctness signal we can produce without target runtimes.

The constant-spread contract is blind by construction to a
*point-independent* error in the reference itself: adding the same
constant to every `log p_QVR` leaves every difference `δ_i` shifted
by that constant and every spread unchanged. The oracle therefore
carries a second, independent obligation, and this module holds it:
[`_QVR_REFERENCE_JOINT`][tests.transpile.test_gallery_numeric_equivalence._QVR_REFERENCE_JOINT]
pins the reference joint at **every** point of the set, the pin is
**mandatory** for every example that scores a joint (an example
without one is an assertion failure, never a silent pass), and the
pin tolerance is
[`reference_pin_atol`][tests.transpile.test_gallery_numeric_equivalence.reference_pin_atol],
which is never looser than the equivalence tolerance it underwrites.
The pinned numbers themselves are re-derived from raw
`torch.distributions` by
[`test_oracle_reference_strength`][tests.transpile.test_oracle_reference_strength]
for every example the backend tier cannot reach.
"""

from __future__ import annotations

import math
import pathlib

import pytest
import torch

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence, _gallery_data
from tests.transpile.probes.qvr import QvrProbe


_BACKENDS_WITH_IMAGES = {
    "stan": ("panproto-test-stan", "stan", "stan.py"),
    "numpyro": ("panproto-test-numpyro", "py", "numpyro.py"),
    "pyro": ("panproto-test-pyro", "py", "pyro.py"),
    "pymc": ("panproto-test-pymc", "py", "pymc.py"),
    "edward2": ("panproto-test-edward2", "py", "edward2.py"),
    "turing": ("panproto-test-julia", "jl", "turing.jl"),
    "gen": ("panproto-test-julia", "jl", "gen.jl"),
    "webppl": ("panproto-test-node", "js", "webppl.py"),
    "jags": ("panproto-test-jags", "jags", "jags.py"),
    "bugs": ("panproto-test-bugs", "bugs", "jags.py"),
}


def _gallery_cells() -> list[pathlib.Path]:
    return _gallery_data.gallery_examples_with_data()


# ----------------------------------------------------------------------
# Pre-declared cell outcomes.
#
# Population workflow when a previously-skipped infrastructure gap
# closes (probe script gains shape registration; QVR trace grows a
# primitive): drop the cell from the corresponding registry, re-run,
# pass.
#
# Population workflow when a renderer regression introduces a
# transpile gap: pin the cell in `_EXPECTED_TRANSPILE_RAISES` with
# the new kind-prefix; the test then makes that raise a contract
# instead of a silent xfail.
# ----------------------------------------------------------------------

# Cells where the pipeline raises `UnsupportedConstruct` and the
# raise is the documented support boundary. `pytest.raises` asserts
# the kind-prefix; a closed gap surfaces as a test failure and a
# regression (different kind) surfaces as a mismatch. Each group
# carries a one-line reason for the boundary.
_EXPECTED_TRANSPILE_RAISES: dict[tuple[str, str], str] = {
    # mixture_model names the `MixtureNormal` likelihood, absent from
    # every backend's lower family registry.
    ("bugs", "mixture_model"): "family:MixtureNormal",
    ("edward2", "mixture_model"): "family:MixtureNormal",
    ("gen", "mixture_model"): "family:MixtureNormal",
    ("jags", "mixture_model"): "family:MixtureNormal",
    ("numpyro", "mixture_model"): "family:MixtureNormal",
    ("pymc", "mixture_model"): "family:MixtureNormal",
    ("pyro", "mixture_model"): "family:MixtureNormal",
    ("stan", "mixture_model"): "family:MixtureNormal",
    ("turing", "mixture_model"): "family:MixtureNormal",
    ("webppl", "mixture_model"): "family:MixtureNormal",
    # parametric_pooling samples the `school_effects` sub-program
    # (program-as-distribution); no backend resolves it to a target
    # family.
    ("bugs", "parametric_pooling"): "family:school_effects",
    ("edward2", "parametric_pooling"): "family:school_effects",
    ("gen", "parametric_pooling"): "family:school_effects",
    ("jags", "parametric_pooling"): "family:school_effects",
    ("numpyro", "parametric_pooling"): "family:school_effects",
    ("pymc", "parametric_pooling"): "family:school_effects",
    ("pyro", "parametric_pooling"): "family:school_effects",
    ("stan", "parametric_pooling"): "family:school_effects",
    ("turing", "parametric_pooling"): "family:school_effects",
    ("webppl", "parametric_pooling"): "family:school_effects",
    # pmf / tensor_contraction carry `composition` (and `contraction`)
    # declarations; no PPL backend has a surface for them.
    ("bugs", "pmf"): "composition_decl",
    ("bugs", "tensor_contraction"): "composition_decl",
    ("edward2", "pmf"): "composition_decl",
    ("edward2", "tensor_contraction"): "composition_decl",
    ("gen", "pmf"): "composition_decl",
    ("gen", "tensor_contraction"): "composition_decl",
    ("jags", "pmf"): "composition_decl",
    ("jags", "tensor_contraction"): "composition_decl",
    ("numpyro", "pmf"): "composition_decl",
    ("numpyro", "tensor_contraction"): "composition_decl",
    ("pymc", "pmf"): "composition_decl",
    ("pymc", "tensor_contraction"): "composition_decl",
    ("pyro", "pmf"): "composition_decl",
    ("pyro", "tensor_contraction"): "composition_decl",
    ("stan", "pmf"): "composition_decl",
    ("stan", "tensor_contraction"): "composition_decl",
    ("turing", "pmf"): "composition_decl",
    ("turing", "tensor_contraction"): "composition_decl",
    ("webppl", "pmf"): "composition_decl",
    ("webppl", "tensor_contraction"): "composition_decl",
    # hmm / lda broadcast a literal scalar concentration to a vector.
    # JAGS spells this with `rep(v, K)`, so the JAGS cells transpile;
    # the BUGS renderer does not emit the broadcast and raises.
    ("bugs", "hmm"): "arg:broadcast",
    ("bugs", "lda"): "arg:broadcast",
    # zip_regression names `ContinuousBernoulli`, which has no JAGS /
    # BUGS target name.
    ("bugs", "zip_regression"): "family:",
    ("jags", "zip_regression"): "family:",
}

# bnn's `net` morphism draws its mean from an `mlp` param source. The
# network weights are model-internal, absent from both the wire form
# and the sample sites, so no backend can reconstruct the mean; the
# transpiler raises on every backend.
for _neural_model in (
    "bnn",
    "bidirectional_rnn_lm",
    "deep_markov",
    "seq2seq",
    "transformer_lm",
    "vae",
):
    for _neural_backend in _BACKENDS_WITH_IMAGES:
        _EXPECTED_TRANSPILE_RAISES[(_neural_backend, _neural_model)] = (
            "param-source:mlp"
        )

# Gallery examples whose `.md` synthetic-data snippet fails to
# `exec` (raises at runtime, or never sets the `observations`
# dict). Numeric evaluation has no point set, so the cell skips.
_SKIP_DATASET_LOAD_FAILED: frozenset[str] = frozenset({
    # Structural algebra examples: `pmf` carries a `composition` and
    # `tensor_contraction` a `composition` + `contraction`, with no
    # `sample` / `observe` sites. Each exports a composition morphism
    # rather than a probabilistic program, so `load_gallery_data`
    # builds no `observations` dict and there is no joint density to
    # trace.
    "pmf",
    "tensor_contraction",
})

# Gallery examples the in-process [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe]
# cannot score to a deterministic, correct joint.
#
# Each of these carries a `sample h <- backbone` latent whose backbone
# is a `SampledComposition` over continuous intermediate objects
# (RNN/LSTM/GRU scan cells, attention and feed-forward Kleisli chains).
# Its `log_prob` marginalizes those intermediates by Monte-Carlo
# importance sampling, redrawing on every call with no fixed seed, so
# the joint is non-deterministic even with `h` clamped. Worse, the
# per-timestep gates and per-layer composition latents are internal to
# the `SampledComposition` and are never surfaced as trace sites, so no
# point entry can pin them and the endpoint `h` carries zero density.
# The oracle for these models needs the composition marginalization made
# deterministic and its inner latents exposed as sites before the
# joint can be validated; until then a deterministic reference does not
# exist to compare a backend against.
_SKIP_QVR_INCOMPATIBLE: frozenset[str] = frozenset({
    "bidirectional_rnn_lm",
    "deep_markov",
    "gru_lm",
    "lstm_lm",
    "seq2seq",
    "transformer_lm",
    "vae",
    "vanilla_rnn_lm",
})

# ----------------------------------------------------------------------
# The reference pin.
#
# Theorem 4.1's quotient by an additive constant is what makes the
# backend comparison robust to differing base measures, and it is also
# what makes it blind: an oracle that is wrong by the same amount at
# every point produces exactly the spread of a correct one. Nothing on
# the backend side can see such an error, so the reference carries its
# own pin, and the pin has to hold everywhere the comparison does.
#
# Three properties, each asserted rather than assumed:
#
# 1. **Total.** Every gallery example that scores a joint has an entry
#    here or a justified row in `_REFERENCE_PIN_EXEMPT`, and
#    `test_gallery_reference_pin_registry_is_total` fails when the two
#    registries stop covering the gallery exactly. A deleted row, or a
#    new example, is a failure rather than a silently weaker suite.
# 2. **Per point.** An entry pins the joint at every point of
#    `points_from_dataset`, in schedule order, not only at the ground
#    truth. A single-point pin cannot see a *data-dependent* oracle
#    error, and neither can the constant-spread check when the same
#    error rides on both sides, so between them that class would be
#    invisible.
# 3. **Tight.** The comparison runs at `reference_pin_atol`, which is
#    never looser than the equivalence tolerance it underwrites.
#
# Every value here is reproduced from an independent computation on
# each run: an example with live backend cells is re-derived by those
# containers, and every example without one is re-derived from raw
# `torch.distributions` in `test_oracle_reference_strength.py`.
# `test_oracle_reference_strength.py::test_every_pinned_example_has_an_independent_witness`
# asserts that split covers the registry with nothing left over.
#
# Update an entry only after re-deriving the joint independently; a
# drift here is either an oracle regression or a deliberate
# ground-truth change, and the two are told apart by the raw-torch
# reconstruction, not by the number's plausibility.
# ----------------------------------------------------------------------
_QVR_REFERENCE_JOINT: dict[str, tuple[float, ...]] = {
    "ar1": (
        -24.030813217163086,
        -121.17807006835938,
        -44.41229248046875,
        -37.35548400878906,
        -44.10717010498047,
        -38.02667236328125,
    ),
    "bayesian_regression": (
        -54.560791015625,
        -67.99243927001953,
        -81.71551513671875,
        -117.940673828125,
        -164.50860595703125,
        -83.58819580078125,
    ),
    "beta_regression": (
        26.09986114501953,
        12.179271697998047,
        21.649688720703125,
        4.467376708984375,
        20.713706970214844,
        16.414939880371094,
    ),
    "bnn": (
        -412.09527587890625,
        -412.09527587890625,
        -433.6674499511719,
        -431.0032653808594,
        -412.09527587890625,
        -416.7749938964844,
    ),
    "changepoint": (
        -131.08474731445312,
        -155.9451446533203,
        -133.2001953125,
        -145.42013549804688,
        -188.2016143798828,
        -135.9104766845703,
    ),
    "continuous_hmm": (
        -716.6048583984375,
        -733.0679321289062,
        -721.8971557617188,
        -736.2081909179688,
        -722.6180419921875,
        -721.2566528320312,
    ),
    "factor_analysis": (
        -132.61660766601562,
        -214.58114624023438,
        -162.3575897216797,
        -226.3590087890625,
        -187.85678100585938,
        -168.95579528808594,
    ),
    "gamma_regression": (
        -69.07034301757812,
        -80.96452331542969,
        -76.0668716430664,
        -93.28099822998047,
        -72.47667694091797,
        -76.60382080078125,
    ),
    "hmm": (
        335.3818359375,
        335.1047668457031,
        336.9115295410156,
        337.01318359375,
        334.794677734375,
        334.94793701171875,
    ),
    "horseshoe_regression": (
        -64.92990112304688,
        -85.11199951171875,
        -82.23753356933594,
        -145.4669647216797,
        -135.35128784179688,
        -84.14229583740234,
    ),
    "irt_2pl": (
        -69.23272705078125,
        -76.04976654052734,
        -85.58287811279297,
        -78.32916259765625,
        -71.45870971679688,
        -80.12108612060547,
    ),
    "lda": (
        2550.3076171875,
        2563.206298828125,
        1729.403564453125,
        1705.196044921875,
        2534.7998046875,
        1779.01318359375,
    ),
    "linear_gaussian_ssm": (
        -218.77745056152344,
        -224.5797119140625,
        -219.4437255859375,
        -218.57310485839844,
        -223.74766540527344,
        -219.547607421875,
    ),
    "mixture_model": (
        -189.3568115234375,
        -197.78488159179688,
        -197.35861206054688,
        -214.3572998046875,
        -197.0948486328125,
        -196.84445190429688,
    ),
    "negbin_regression": (
        -203.68475341796875,
        -225.95242309570312,
        -270.5653381347656,
        -302.58197021484375,
        -221.177734375,
        -268.622314453125,
    ),
    "parametric_pooling": (
        -16.053003311157227,
        -21.250812530517578,
        -15.52226448059082,
        -19.705469131469727,
        -26.930932998657227,
        -18.749011993408203,
    ),
    "ppca": (
        -67.73426818847656,
        -231.67868041992188,
        -105.8104476928711,
        -274.6122741699219,
        -188.58070373535156,
        -141.64019775390625,
    ),
    "stochastic_volatility": (
        -307.8003845214844,
        -497.64605712890625,
        -374.3330078125,
        -449.9366760253906,
        -448.69561767578125,
        -387.8696594238281,
    ),
    "survival_weibull": (
        -28.67302131652832,
        -39.528465270996094,
        -37.15167236328125,
        -47.73482894897461,
        -234.78810119628906,
        -36.604496002197266,
    ),
    "tree_categorical": (
        -14.377462387084961,
        -14.70197868347168,
        -14.261514663696289,
        -16.662046432495117,
        -17.16188621520996,
        -14.015467643737793,
    ),
    "zip_regression": (
        -651.6888427734375,
        -664.6071166992188,
        -755.0150146484375,
        -759.4091186523438,
        -665.1041259765625,
        -747.2142333984375,
    ),
}

# Gallery examples that carry synthetic data but score no joint the
# pin could hold, each with the reason. An entry is an assertion, not
# an escape hatch: `test_gallery_reference_pin_registry_is_total`
# requires the reason to name a registry that independently agrees the
# example has no reference, so an exemption cannot outlive the gap it
# describes.
_REFERENCE_PIN_EXEMPT: dict[str, str] = {
    # No `program` block, no `sample` site, no `observe` site: the
    # module exports a `define`d composition morphism (`U.dagger >> V`,
    # `bilinear_score(...)`), which denotes a linear map rather than a
    # measure. `load_gallery_data` builds no `observations` dict for
    # either, both sit in `_SKIP_DATASET_LOAD_FAILED`, and there is no
    # joint log-density to pin.
    # `test_oracle_reference_strength.py::test_structurally_exempt_examples_declare_no_probabilistic_program`
    # asserts that structural claim against the `.qvr` text itself.
    "pmf": "structural: composition morphism, no stochastic site",
    "tensor_contraction": (
        "structural: contraction morphism, no stochastic site"
    ),
    # Sequence models carrying a `SampledComposition` latent. The
    # oracle marginalises the composition's internal states by
    # importance sampling and redraws on every call, so its "joint" is
    # a sample from an estimator rather than a density and there is no
    # value a pin could hold. Each sits in `_SKIP_QVR_INCOMPATIBLE`,
    # whose comment carries the full diagnosis.
    "bidirectional_rnn_lm": "no deterministic oracle joint to pin",
    "deep_markov": "no deterministic oracle joint to pin",
    "gru_lm": "no deterministic oracle joint to pin",
    "lstm_lm": "no deterministic oracle joint to pin",
    "seq2seq": "no deterministic oracle joint to pin",
    "transformer_lm": "no deterministic oracle joint to pin",
    "vae": "no deterministic oracle joint to pin",
    "vanilla_rnn_lm": "no deterministic oracle joint to pin",
}

_REFERENCE_PIN_ULP_BUDGET = 8
"""Round-off budget for
[`reference_pin_atol`][tests.transpile.test_gallery_numeric_equivalence.reference_pin_atol],
in units of the float32 grid spacing at the pinned magnitude.

The reference joint reaches the harness as
``float(trace.log_joint.sum().item())``: the exact float64 widening of
a float32 accumulator. Two evaluations of the *same* density therefore
agree exactly whenever they accumulate in the same order, and differ
only in the low bits of that accumulator when they do not (a different
reduction kernel, SIMD width, or BLAS). The budget is calibrated
against a direct measurement of that re-association effect rather than
against a guess: the raw-`torch.distributions` reconstructions in
`test_oracle_reference_strength.py` sum the same per-site terms in a
different order and in different groupings, and across all six models
and all six points their largest disagreement with the trace is
**one** float32 ULP (`continuous_hmm`, 6.10e-05 at magnitude 736;
`linear_gaussian_ssm`, 1.53e-05 at magnitude 219). Eight ULPs is three
bits of headroom above the measured worst case.

This is a headroom figure, not a necessity: the oracle is bit-exact
run to run and across `torch.set_num_threads`, both measured. A
failure at this budget is a real change in the density, not noise."""


def _float32_ulp(value: float) -> float:
    """Spacing of the float32 grid at `value`.

    `math.frexp` returns ``(m, e)`` with ``|value| = m * 2**e`` and
    ``m`` in ``[0.5, 1)``, so the leading significand bit has weight
    ``2**(e - 1)`` and the trailing bit of a 24-bit significand has
    weight ``2**(e - 24)``.
    """
    magnitude = abs(value)
    if not math.isfinite(magnitude):
        raise ValueError(
            f"_float32_ulp needs a finite value; got {value!r}. A "
            f"non-finite reference is a broken pin, not a tolerance "
            f"question."
        )
    if magnitude == 0.0:
        return math.ldexp(1.0, -24)
    _, exponent = math.frexp(magnitude)
    return math.ldexp(1.0, exponent - 24)


def reference_pin_atol(reference: float) -> float:
    """Absolute tolerance for the reference pin at `reference`.

    Two bounds, and the tighter one wins.

    The first is
    [`_REFERENCE_PIN_ULP_BUDGET`][tests.transpile.test_gallery_numeric_equivalence._REFERENCE_PIN_ULP_BUDGET]
    ULPs of the float32 grid at the pinned magnitude, floored at the
    same budget taken at magnitude 1. The floor keeps a joint that
    happens to land near zero from demanding bit equality: such a
    joint is still a sum of order-one terms, and the grid at 1 is the
    finest resolution those terms carry.

    The second is the equivalence tolerance from
    [`adaptive_atol`][tests.transpile._equivalence.adaptive_atol] at
    its floor, which is the tolerance
    [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
    holds the backends to. Taking the minimum is the whole point of
    the function: a constant oracle error is invisible on the backend
    side, so the pin is the only defence against it, and a defence
    looser than the check it underwrites defends nothing. It also ties
    the two together in code, so the pin cannot be left behind if the
    equivalence floor ever moves.

    Across the 126 pinned values the ULP bound binds for 120 and the
    equivalence floor caps the remaining six, all of them `lda`
    (magnitudes 1705 to 2563, where one float32 ULP is already
    1.22e-04 to 2.44e-04). The loosest pin in the registry is thus
    5e-04 and the tightest is 3.81e-06 (`beta_regression` at its
    latents+data point, magnitude 4.47). Measured at the ground-truth
    point of every registry entry, the band is between 1238 times
    (`changepoint`) and 5141 times (`lda`) tighter than the
    `1e-3 * |reference| + 2e-2` relative band it replaces.
    """
    ulp_bound = _REFERENCE_PIN_ULP_BUDGET * max(
        _float32_ulp(reference), _float32_ulp(1.0),
    )
    return min(_equivalence.adaptive_atol(n_obs=0), ulp_bound)

# Gallery examples that genuinely carry no perturbable observation, so
# their point set moves the latents alone. Each entry states why the
# example's observed data cannot move, and
# `test_gallery_multipoint_set_is_in_support_and_varies` asserts the
# claim: an example listed here must have a frozen data section, and an
# example not listed here must have a data section that moves.
#
# Populate an entry only for a model whose every observation is an
# index-valued covariate or a degenerate single-valued response, never
# to quiet a data section the harness merely fails to perturb. A frozen
# data section makes the constant-spread check blind to a backend that
# drops a data-dependent term, which is the single failure mode the
# multi-point set exists to catch.
_NO_PERTURBABLE_OBSERVATION: dict[str, str] = {}

# (backend, example) cells whose measured log-density cannot yet be
# compared against the QVR reference, each grouped under the single
# defect that blocks it. Every entry has been run against its
# container and carries the error the run actually produced, so the
# comment names a live blocker rather than a historical one.
#
# Closure path: fix the named defect in its owning file, re-measure
# the cell with the six-point set, drop it from this registry. A cell
# whose spread is constant to within `adaptive_atol` belongs outside
# the registry; nothing else does.
_SKIP_PROBE_INCOMPATIBLE: frozenset[tuple[str, str]] = frozenset({
    # Sequence-model examples carried by `_SKIP_QVR_INCOMPATIBLE`,
    # which the test consults first: the in-process oracle redraws
    # the `SampledComposition` intermediates on every call, so there
    # is no deterministic reference to compare a container against.
    # The entries stay so that closing the oracle gap surfaces the
    # backend-side state of each cell rather than 30 fresh failures.
    ('bugs', 'gru_lm'),
    ('bugs', 'lstm_lm'),
    ('bugs', 'vanilla_rnn_lm'),
    ('edward2', 'gru_lm'),
    ('edward2', 'lstm_lm'),
    ('edward2', 'vanilla_rnn_lm'),
    ('gen', 'gru_lm'),
    ('gen', 'lstm_lm'),
    ('gen', 'vanilla_rnn_lm'),
    ('jags', 'gru_lm'),
    ('jags', 'lstm_lm'),
    ('jags', 'vanilla_rnn_lm'),
    ('numpyro', 'gru_lm'),
    ('numpyro', 'lstm_lm'),
    ('numpyro', 'vanilla_rnn_lm'),
    ('pymc', 'gru_lm'),
    ('pymc', 'lstm_lm'),
    ('pymc', 'vanilla_rnn_lm'),
    ('pyro', 'gru_lm'),
    ('pyro', 'lstm_lm'),
    ('pyro', 'vanilla_rnn_lm'),
    ('stan', 'gru_lm'),
    ('stan', 'lstm_lm'),
    ('stan', 'vanilla_rnn_lm'),
    ('turing', 'gru_lm'),
    ('turing', 'lstm_lm'),
    ('turing', 'vanilla_rnn_lm'),
    ('webppl', 'gru_lm'),
    ('webppl', 'lstm_lm'),
    ('webppl', 'vanilla_rnn_lm'),
    # continuous_hmm / linear_gaussian_ssm: a Kleisli morphism
    # declared with a `~ Family` init and no `[param_source=...]`
    # option takes the default linear source, whose weights are
    # initialised at compile time, appear in no sample site and in no
    # line of the `.qvr` text. The emitted program therefore degrades
    # to the bare family at its defaults and binds a State-width mean
    # to an Obs-width site, which no point payload can bridge. The
    # program's domain object (the previous state, and the LGSSM
    # driver) also has no wire channel: `dataset.x_input` is a single
    # concatenated matrix that only the in-process `QvrProbe`
    # consumes, so every container reports the missing argument or
    # its rank. The raise belongs beside the existing
    # `param-source:mlp` rejection in
    # `src/quivers/transpile/_resolve.py`; the wire split belongs in
    # `tests/transpile/_gallery_data.py`.
    #   numpyro / pyro: model() missing 'state' (and 'driver')
    #   stan: dims declared=(16), dims found=() for `state`
    #   pymc: ShapeError, actual 2 != expected 1
    #   edward2: cannot convert None to a Tensor
    #   turing / gen: no method matching model(::Matrix{Float64})
    #   jags / bugs: dimension mismatch in subset expression of `o`
    #   webppl: Parameter "mu" should be of type "real"
    ('bugs', 'continuous_hmm'),
    ('bugs', 'linear_gaussian_ssm'),
    ('edward2', 'continuous_hmm'),
    ('edward2', 'linear_gaussian_ssm'),
    ('gen', 'continuous_hmm'),
    ('gen', 'linear_gaussian_ssm'),
    ('jags', 'continuous_hmm'),
    ('jags', 'linear_gaussian_ssm'),
    ('numpyro', 'continuous_hmm'),
    ('numpyro', 'linear_gaussian_ssm'),
    ('pymc', 'continuous_hmm'),
    ('pymc', 'linear_gaussian_ssm'),
    ('pyro', 'continuous_hmm'),
    ('pyro', 'linear_gaussian_ssm'),
    ('stan', 'continuous_hmm'),
    ('stan', 'linear_gaussian_ssm'),
    ('turing', 'continuous_hmm'),
    ('turing', 'linear_gaussian_ssm'),
    ('webppl', 'continuous_hmm'),
    ('webppl', 'linear_gaussian_ssm'),
    # tree_categorical: the synthetic-data snippet in
    # `docs/examples/tree-categorical.md` emits a single response and
    # clamps every scalar latent at rank-2 singleton shape, while the
    # model declares `object Resp : FinSet 200`. Stan reports `y`
    # declared (200) against data found (1,1), pymc raises
    # `ShapeError`, JAGS and BUGS report a dimension mismatch, the
    # Julia backends meet a `Matrix{Float64}` where a scalar belongs,
    # and the Python backends score a shape-broadcast joint that
    # drifts by ~178 nats across the point set. Rebuilt against a
    # corrected 200-response dataset the renderer sides measure a
    # constant spread (numpyro / pymc 6.83e-05, edward2 2.29e-05,
    # pyro 1.94e-05, jags / bugs 1.14e-06), so the fix is the data
    # snippet plus the rank-0 clamp path in
    # `src/quivers/continuous/inline.py`, with the
    # `_QVR_REFERENCE_JOINT` entry re-derived in the same change.
    ('bugs', 'tree_categorical'),
    ('edward2', 'tree_categorical'),
    ('gen', 'tree_categorical'),
    ('jags', 'tree_categorical'),
    ('numpyro', 'tree_categorical'),
    ('pymc', 'tree_categorical'),
    ('pyro', 'tree_categorical'),
    ('stan', 'tree_categorical'),
    ('turing', 'tree_categorical'),
    ('webppl', 'tree_categorical'),
    # hmm: `sample initial_row : State <- Dirichlet(1.0)
    # [over=State]` lowers to one `simplex[8]` under an empty plate
    # while the QVR runtime produces an (8, 8) batch of simplices,
    # and the gallery data ships `obs` with 8 entries against the
    # model's `object Obs : FinSet 16`. The axis-role derivation in
    # `src/quivers/transpile/lower.py::_build_plate` owns the first
    # half and `tests/transpile/_gallery_data.py` the second. numpyro
    # and pyro reproduce the reference through untyped broadcasting;
    # every typed backend reports the clash directly (stan: `obs`
    # declared (16), found (8); pymc: cannot convert Matrix(8, 8)
    # into Vector(8,); gen: Vector{Float64}(::Matrix{Float64});
    # turing: +(::Float64, ::Vector{Float64}); jags: compilation
    # error on line 2). edward2 leaves the mis-ranked `initial_row`
    # unconditioned and drifts 0.92 nats; webppl additionally passes
    # the Dirichlet concentration as a plain JS array, which its
    # `Dirichlet` rejects as not a vector.
    ('edward2', 'hmm'),
    ('gen', 'hmm'),
    ('jags', 'hmm'),
    ('pymc', 'hmm'),
    ('stan', 'hmm'),
    ('turing', 'hmm'),
    ('webppl', 'hmm'),
    # lda: the four backends that integrate the topic latent
    # correctly are out of this registry; these five each carry a
    # distinct blocker.
    #   gen: `Gen.assess` requires every traced address to be
    #     constrained and the `@gen` DSL has no log-weight primitive,
    #     so the marginalized `z` surfaces as KeyError (:z, 1).
    #   jags: `RendererBase.explicit_latent_scope` lowers the
    #     marginalize to a live `IRSample(z)` and drops the
    #     reduction, so the emitted measure lives on a strictly
    #     larger space and the spread runs to 409 nats. The zeros
    #     trick `jags.py::_emit_score` already uses is the closure.
    #   stan: the point payload ships `theta` / `phi` rows as float32
    #     summing to 1.00000006, which `stan::math::simplex_free`
    #     rejects; the harness must renormalize simplex-typed
    #     parameters. The renderer's log_sum_exp accumulator is
    #     fixed and measures 3.18e-04 once that holds.
    #   turing: the gathered per-word topic weights index a scalar,
    #     raising BoundsError at index [2].
    #   webppl: the Dirichlet concentration reaches WebPPL as a plain
    #     JS array rather than a vector.
    ('gen', 'lda'),
    ('jags', 'lda'),
    ('stan', 'lda'),
    ('turing', 'lda'),
    ('webppl', 'lda'),
    # zip_regression: the backends whose Poisson uses an `xlogy` form
    # integrate the zero-inflation indicator correctly and are out of
    # this registry; these four do not.
    #   numpyro: its `Poisson` computes log(rate) * value directly,
    #     so the z = 0 atom's rate of exactly 0 yields nan at every
    #     y == 0 observation. The emitted expression is faithful; the
    #     deficiency is upstream in numpyro.
    #   stan: `_is_continuous_support` routes `ContinuousBernoulli`
    #     to the continuous marginalization, declaring a live
    #     400-dim `z` parameter the point payload cannot fill
    #     (dims declared=(400), found=()), where the QVR compiler
    #     enumerates the hard support {0, 1}.
    #   gen: same missing log-weight primitive as gen/lda; the
    #     reduced density has no address to ride on, and
    #     `Gen.logpdf(Gen.poisson, 0, 0.0)` is NaN where the
    #     reference scores the point mass exactly.
    #   webppl: the atom reduction emits `factor(...)`, which the
    #     probe's rewrite does not turn into `globalStore.lp`
    #     accumulation, so WebPPL raises `factor allowed only inside
    #     inference`.
    ('gen', 'zip_regression'),
    ('numpyro', 'zip_regression'),
    ('stan', 'zip_regression'),
    ('webppl', 'zip_regression'),
    # webppl/ppca and webppl/factor_analysis: the renderer emits the
    # residual event axis as a nested
    # `repeat(32, function () { return repeat(2, ...); })`, but the
    # probe's `_lift_iid_plate` matches only a callback whose body is
    # a bare `return sample(...)`. The nested plate falls through
    # unlifted and the probe raises rather than let the site be
    # redrawn. Measured against a locally patched probe the cells are
    # constant to 6.75e-06 and 2.10e-05, so the closure is a
    # recursive branch in
    # `tests/transpile/probes/_scripts/webppl.py`.
    ('webppl', 'factor_analysis'),
    ('webppl', 'ppca'),
    # webppl/stochastic_volatility: the latent trajectory emits as
    # `mapIndexed(..., repeat(200, 0))` and WebPPL's `repeat`
    # requires its second argument to be a function. The probe's
    # `mapIndexed` lift is otherwise ready for this cell.
    ('webppl', 'stochastic_volatility'),
})


def test_gallery_reference_pin_registry_is_total() -> None:
    """Every gallery example is either pinned or justifiably exempt,
    and neither registry may carry a row the gallery does not.

    This is the test that makes the pin a guarantee rather than an
    opt-in. The failure mode it exists to prevent is silent: a
    `dict.get` that returns `None` and a guard that skips the
    assertion turn a deleted row, or a newly-added example, into a
    check that passes while asserting nothing about the value it was
    written to protect. Requiring the two registries to *partition*
    the gallery makes both directions loud. Deleting a row fails here.
    Adding an example without deriving its reference fails here.
    Retiring an example without dropping its row fails here too, so a
    stale pin cannot sit in the registry looking like coverage.

    The exemption side is checked against the registries that
    independently agree the example has no reference, not taken on its
    word: an exempt example must appear in `_SKIP_DATASET_LOAD_FAILED`
    (no dataset at all) or in `_SKIP_QVR_INCOMPATIBLE` (no
    deterministic joint). An exemption whose gap has closed therefore
    fails rather than quietly suppressing a pin the example could now
    carry.
    """
    gallery = {example.stem for example in _gallery_cells()}
    pinned = set(_QVR_REFERENCE_JOINT)
    exempt = set(_REFERENCE_PIN_EXEMPT)

    overlap = sorted(pinned & exempt)
    assert not overlap, (
        f"{overlap!r} are both pinned and exempt. An example carries a "
        f"reference or it does not; the two registries must be "
        f"disjoint so a reader can tell which examples are actually "
        f"guarded."
    )

    unpinned = sorted(gallery - pinned - exempt)
    assert not unpinned, (
        f"{unpinned!r} ship synthetic data but appear in neither "
        f"`_QVR_REFERENCE_JOINT` nor `_REFERENCE_PIN_EXEMPT`, so "
        f"nothing asserts their oracle joint is right. Derive the "
        f"reference from raw `torch.distributions` and pin every "
        f"point, or record why the example has no reference."
    )

    stale = sorted((pinned | exempt) - gallery)
    assert not stale, (
        f"{stale!r} are registered but are no longer gallery examples "
        f"with synthetic data. Drop the rows: a pin on an example that "
        f"no longer runs reads as coverage and is not."
    )

    expected_points = len(_gallery_data.perturbation_labels())
    for stem, values in sorted(_QVR_REFERENCE_JOINT.items()):
        assert len(values) == expected_points, (
            f"{stem!r}: pinned at {len(values)} point(s) against a "
            f"{expected_points}-point set. The pin has to cover every "
            f"point the equivalence check evaluates, or a "
            f"data-dependent oracle error hides in the unpinned tail."
        )
        for index, value in enumerate(values):
            assert math.isfinite(value), (
                f"{stem!r}: pinned value at point {index} is "
                f"{value!r}. A non-finite pin asserts nothing: every "
                f"comparison against it is `nan`, which no `<=` "
                f"rejects."
            )

    for stem, reason in sorted(_REFERENCE_PIN_EXEMPT.items()):
        assert reason.strip(), (
            f"{stem!r}: exempt from the reference pin with an empty "
            f"reason. An exemption without a stated cause is an "
            f"unexplained hole in the guarantee."
        )
        assert (
            stem in _SKIP_DATASET_LOAD_FAILED
            or stem in _SKIP_QVR_INCOMPATIBLE
        ), (
            f"{stem!r}: claims exemption from the reference pin "
            f"({reason}) but is in neither `_SKIP_DATASET_LOAD_FAILED` "
            f"nor `_SKIP_QVR_INCOMPATIBLE`, so the gallery tier does "
            f"score a joint for it. Either the gap closed and the row "
            f"belongs in `_QVR_REFERENCE_JOINT`, or the skip registry "
            f"is missing the entry that justifies the exemption."
        )


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
def test_gallery_qvr_reference_pin_holds_at_every_point(
    example: pathlib.Path,
) -> None:
    """The QVR reference reproduces its pinned value at **every**
    point of the set, not only at the ground truth.

    A ground-truth-only pin and the constant-spread check have
    complementary blind spots that overlap exactly on the class of
    error that matters most here. The pin at point 0 sees a constant
    offset but says nothing about points 1..5. The spread check sees a
    varying offset but is invariant to a constant one. An oracle error
    that is *zero at the ground truth and constant across the
    perturbed points* therefore passes both: the pin holds where it
    looks, and the spread is unchanged because the same wrong value
    feeds both sides of every difference. Pinning per point removes
    that overlap, because a per-point pin is violated by any error
    that moves at all.

    The concrete shape of that error class in this codebase is a
    data-dependent term the oracle drops. The dropped term is zero at
    the ground truth for a fixture whose ground truth sits at the
    term's zero (a centred residual, a sum-to-zero score, a
    log-normaliser that cancels at the generating parameters) and
    non-zero once the data moves.
    """
    if example.stem in _SKIP_DATASET_LOAD_FAILED:
        pytest.skip(
            f"{example.stem!r}: synthetic-data snippet in the `.md` "
            f"file fails to load; populate / drop from "
            f"`_SKIP_DATASET_LOAD_FAILED`."
        )
    if example.stem in _SKIP_QVR_INCOMPATIBLE:
        pytest.skip(
            f"{example.stem!r}: in-process QVR trace cannot evaluate "
            f"this program; populate / drop from `_SKIP_QVR_INCOMPATIBLE`."
        )

    reference = _QVR_REFERENCE_JOINT.get(example.stem)
    assert reference is not None, (
        f"{example.stem!r}: reaches the multi-point scoring path with "
        f"no `_QVR_REFERENCE_JOINT` entry. See "
        f"`test_gallery_reference_pin_registry_is_total`."
    )

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`."
    )

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    assert len(points) == len(reference), (
        f"{example.stem!r}: {len(points)} point(s) against "
        f"{len(reference)} pinned value(s). The point schedule moved "
        f"under the pin; re-derive every value before re-pinning."
    )

    probe = QvrProbe()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_pin_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    source = example.read_bytes()
    for index, point in enumerate(points):
        measured = probe.evaluate(
            source,
            example.stem,
            [point],
            scratch=scratch,
            monadic=dataset.monadic,
            x_input=dataset.x_input,
            observations=_gallery_data.observations_for_point(
                dataset, point,
            ),
        ).log_densities[0]
        expected = reference[index]
        atol = reference_pin_atol(expected)
        assert abs(measured - expected) <= atol, (
            f"{example.stem!r} point {index} ({labels[index]}): QVR "
            f"joint {measured!r} against pinned reference "
            f"{expected!r}, a gap of {abs(measured - expected):.6g} "
            f"nats past the {atol:.6g} round-off budget. The oracle's "
            f"density changed at this point. Re-derive it from raw "
            f"`torch.distributions` before touching the pin; widening "
            f"the tolerance would restore exactly the blindness the "
            f"per-point pin exists to remove."
        )


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
def test_gallery_qvr_logdensity_finite(example: pathlib.Path) -> None:
    """The QVR-side log-density at the ground-truth (θ_true, y)
    point evaluates to a finite real number.

    This is the always-on Tier-4 invariant: regardless of backend
    Docker availability, every gallery example with synthetic data
    must produce a well-defined joint log-density under the
    reference QVR trace. A non-finite value (`-inf`, `nan`)
    indicates that either (a) the data-gen snippet's ground-truth
    parameters fall outside the model's support, or (b) the QVR
    program has a structural defect that the trace surfaces only
    on real data.
    """
    if example.stem in _SKIP_DATASET_LOAD_FAILED:
        pytest.skip(
            f"{example.stem!r}: synthetic-data snippet in the `.md` "
            f"file fails to load; populate / drop from "
            f"`_SKIP_DATASET_LOAD_FAILED`."
        )
    if example.stem in _SKIP_QVR_INCOMPATIBLE:
        pytest.skip(
            f"{example.stem!r}: in-process QVR trace cannot evaluate "
            f"this program; populate / drop from `_SKIP_QVR_INCOMPATIBLE`."
        )

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`. "
        f"Add the example's stem to that registry, or fix the "
        f"`.md` snippet so it produces an `observations` dict."
    )

    point = _gallery_data.point_from_dataset(dataset)
    probe = QvrProbe()
    source = example.read_bytes()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    result = probe.evaluate(
        source,
        example.stem,
        [point],
        scratch=scratch,
        monadic=dataset.monadic,
        x_input=dataset.x_input,
        observations=dataset.observations,
    )
    assert len(result.log_densities) == 1, (
        f"{example.stem!r}: expected exactly one log-density, got "
        f"{result.log_densities!r}"
    )
    lp = result.log_densities[0]
    assert math.isfinite(lp), (
        f"{example.stem!r}: QVR log p(θ_true, y) is non-finite ({lp!r}); "
        f"observations={list(dataset.observations)} params={list(dataset.params)}"
    )

    # Finiteness alone is a weak oracle check: a measure bug can return a
    # finite-but-wrong joint (an unclamped latent resampling, a
    # marginalized site double-counted, a plate broadcast inflating the
    # sum). The reference pin is what turns it into a correctness
    # check, and it is mandatory: an example that reaches this line has
    # a joint, so a missing entry is a hole in the guarantee and fails
    # here rather than reverting to the finiteness-only check.
    reference = _QVR_REFERENCE_JOINT.get(example.stem)
    assert reference is not None, (
        f"{example.stem!r}: scores a finite QVR joint ({lp!r}) but has "
        f"no `_QVR_REFERENCE_JOINT` entry, so nothing checks that the "
        f"value is *right*. Theorem 4.1's constant-spread quotient "
        f"cannot see a point-independent oracle error, which makes "
        f"this pin the only thing that can. Re-derive the joint from "
        f"raw `torch.distributions` (see "
        f"`test_oracle_reference_strength.py`), pin every point of "
        f"the set, and add the row. If this example genuinely has no "
        f"reference, record the reason in `_REFERENCE_PIN_EXEMPT` "
        f"instead."
    )
    atol = reference_pin_atol(reference[0])
    assert abs(lp - reference[0]) <= atol, (
        f"{example.stem!r}: QVR joint {lp!r} drifted from its "
        f"independently-verified reference {reference[0]!r} by "
        f"{abs(lp - reference[0]):.6g} nats, past the "
        f"{atol:.6g} round-off budget. Either the oracle regressed or "
        f"the ground-truth point changed; re-derive the joint "
        f"independently before updating the `_QVR_REFERENCE_JOINT` "
        f"entry, and never widen the tolerance to absorb the drift."
    )


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
def test_gallery_multipoint_set_is_in_support_and_varies(
    example: pathlib.Path,
) -> None:
    """The multi-point set stays in support and actually moves.

    Three properties make
    [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
    a real test rather than a tautology, and each is asserted here so
    a regression surfaces without needing a Docker image:

    1. Every point scores a **finite** QVR joint. A perturbation that
       steps outside the support sends both evaluators to `-inf`, and
       two `-inf` values differ by `nan` rather than by a constant, so
       the comparison would be meaningless.
    2. The joint **varies** across the set. A point set that collapses
       to repeats of the ground truth restores the single-point
       vacuity the multi-point check exists to remove: the spread of a
       constant difference sequence is zero whatever the backend
       computed. Variation in the joint is the observable form of
       "the latents or the data really moved", and it holds for every
       gallery shape, including examples that capture no latents (only
       the data moves) and examples whose data is entirely
       integer-valued covariates (only the latents move).
    3. The **observed data** varies across the set. This is strictly
       stronger than (2) and it is the property the constant-spread
       contract actually needs: a latents-only perturbation moves the
       joint while leaving every observation at ground truth, so (2)
       passes on a point set whose data section is byte-identical
       throughout. Against such a set a backend that drops a
       data-dependent term keeps a perfectly constant offset and the
       equivalence assertion is vacuous. An example whose data
       genuinely cannot move states so in `_NO_PERTURBABLE_OBSERVATION`
       and has the frozen data section asserted rather than assumed.
    """
    if example.stem in _SKIP_DATASET_LOAD_FAILED:
        pytest.skip(
            f"{example.stem!r}: synthetic-data snippet in the `.md` "
            f"file fails to load; populate / drop from "
            f"`_SKIP_DATASET_LOAD_FAILED`."
        )
    if example.stem in _SKIP_QVR_INCOMPATIBLE:
        pytest.skip(
            f"{example.stem!r}: in-process QVR trace cannot evaluate "
            f"this program; populate / drop from `_SKIP_QVR_INCOMPATIBLE`."
        )

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`."
    )

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    assert len(points) >= 2, (
        f"{example.stem!r}: {len(points)} point(s); the constant-spread "
        f"contract needs at least two to be testable."
    )

    probe = QvrProbe()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_points_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    source = example.read_bytes()
    lps: list[float] = []
    for point in points:
        lps.extend(
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

    for index, lp in enumerate(lps):
        assert math.isfinite(lp), (
            f"{example.stem!r}: point {index} ({labels[index]}) scores a "
            f"non-finite QVR joint ({lp!r}), so it left the model's "
            f"support. The perturbation for one of its sites does not "
            f"respect the site's declared constraint."
        )

    assert len({round(lp, 6) for lp in lps}) > 1, (
        f"{example.stem!r}: every point scores the same QVR joint "
        f"({lps[0]!r}), so no site moved and the constant-spread check "
        f"would pass unconditionally. Either the dataset captured "
        f"nothing perturbable, or a constraint this example needs is "
        f"missing from `_perturb_by_support`."
    )

    observed = _gallery_data.observed_data_names(dataset)
    moved = _gallery_data.varying_observation_names(dataset, points)
    frozen_reason = _NO_PERTURBABLE_OBSERVATION.get(example.stem)
    if frozen_reason is None:
        assert moved, (
            f"{example.stem!r}: the observed data is byte-identical at "
            f"every point ({sorted(observed)} all frozen), so the "
            f"constant-spread check for this example is a "
            f"latents-only test: a backend that drops a "
            f"data-dependent term would keep a constant offset and "
            f"pass. Give the data section a support the perturber can "
            f"step in (an observed count moves inside its attested "
            f"range, a simplex-valued observation renormalises, a "
            f"bounded one moves in its own space), or, if this "
            f"example truly has no perturbable observation, record "
            f"the reason in `_NO_PERTURBABLE_OBSERVATION`."
        )
    else:
        assert not moved, (
            f"{example.stem!r}: `_NO_PERTURBABLE_OBSERVATION` claims "
            f"this example has no perturbable observation "
            f"({frozen_reason}), but {sorted(moved)} moved across the "
            f"point set. Drop the entry: the data section is testable "
            f"and the claim is stale."
        )


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
@pytest.mark.parametrize("backend", sorted(_BACKENDS_WITH_IMAGES))
def test_gallery_backend_logdensity_matches_qvr(
    example: pathlib.Path, backend: str
) -> None:
    """Constant-spread equivalence between the QVR reference and the
    backend's native log-density at the ground-truth (θ_true, y)
    point set for `example`."""
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
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

    cell = (backend, example.stem)

    expected_raise = _EXPECTED_TRANSPILE_RAISES.get(cell)
    if expected_raise is not None:
        source = example.read_bytes()
        with pytest.raises(UnsupportedConstruct) as exc_info:
            transpile(parse(source.decode("utf-8")), target=backend)
        kinds = exc_info.value.kinds
        assert any(k.startswith(expected_raise) for k in kinds), (
            f"{backend!r} on {example.stem!r}: expected raise with "
            f"kind prefix {expected_raise!r}, got kinds={kinds!r}. "
            f"Either the renderer changed (update the entry in "
            f"`_EXPECTED_TRANSPILE_RAISES`) or a different gap fired."
        )
        return

    if example.stem in _SKIP_DATASET_LOAD_FAILED:
        pytest.skip(
            f"{example.stem!r}: synthetic-data snippet in the `.md` "
            f"file fails to load; populate / drop from "
            f"`_SKIP_DATASET_LOAD_FAILED`."
        )
    if example.stem in _SKIP_QVR_INCOMPATIBLE:
        pytest.skip(
            f"{example.stem!r}: in-process QVR trace cannot evaluate "
            f"this program; populate / drop from `_SKIP_QVR_INCOMPATIBLE`."
        )
    if cell in _SKIP_PROBE_INCOMPATIBLE:
        pytest.skip(
            f"{backend!r} on {example.stem!r}: in-container probe "
            f"script has no shape registration for this example's "
            f"dataset; populate / drop from `_SKIP_PROBE_INCOMPATIBLE`."
        )

    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None, (
        f"{example.stem!r}: `load_gallery_data` returned None even "
        f"though the example was not in `_SKIP_DATASET_LOAD_FAILED`. "
        f"Add the example's stem to that registry, or fix the "
        f"`.md` snippet so it produces an `observations` dict."
    )

    source = example.read_bytes()
    emitted = transpile(parse(source.decode("utf-8")), target=backend)

    points = _gallery_data.points_from_dataset(dataset)
    labels = _gallery_data.perturbation_labels(len(points))
    qvr_probe = QvrProbe()
    # Isolate the bind-mounted scratch per (backend, example) so a
    # probe never mounts a source or helper file another backend's run
    # left behind in a shared directory.
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_eq_{example.stem}_{backend}"
    scratch.mkdir(exist_ok=True, parents=True)
    # One probe call per point: the probe's `observations` keyword
    # overrides the flat per-point payload (it is the only channel that
    # preserves multi-axis shapes), so a perturbed point needs its own
    # pre-shaped observation dict. Passing the dataset's ground-truth
    # observations once for the whole set would score the QVR side at
    # the unperturbed data while the container scored the perturbed
    # data.
    qvr_lps: list[float] = []
    for point in points:
        qvr_lps.extend(
            qvr_probe.evaluate(
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

    script_path = (
        pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    )
    raw_result = _docker.run_probe(
        image=image,
        script=script_path,
        source=emitted,
        source_ext=ext,
        points=[
            {"params": point.params, "data": point.data}
            for point in points
        ],
        scratch=scratch,
        shapes=_shapes_from_dataset(dataset),
        dtypes=_dtypes_from_dataset(dataset),
    )

    backend_lps = [float(x) for x in raw_result["log_densities"]]

    _equivalence.assert_log_density_match(
        qvr_lps,
        backend_lps,
        context=f"{backend}@{example.stem}",
        labels=labels,
        min_points=2,
    )


def _shapes_from_dataset(
    dataset: _gallery_data.GalleryDataset,
) -> dict[str, list[int]]:
    """Per-name shape table the per-backend probe scripts read to
    rebuild nested arrays from the row-major flat lists in `Point`.

    Scalar tensors (zero-dim) map to ``[]`` rather than ``[1]``;
    `point_from_dataset` collapses length-1 lists to bare scalars
    and `_reshape.reshape_value` reads the empty shape as a no-op
    after the dtype cast.
    """
    shapes: dict[str, list[int]] = {}
    for k, v in dataset.observations.items():
        shapes[k] = list(v.shape)
    for k, v in dataset.params.items():
        shapes[k] = list(v.shape)
    # A scalar type-parameter reaches the container as a bare float in
    # the point's data section; the empty shape casts it through the
    # dtype table without rebuilding a nested container around it.
    for name in dataset.scalar_params:
        shapes[name] = []
    return shapes


def _dtypes_from_dataset(
    dataset: _gallery_data.GalleryDataset,
) -> dict[str, str]:
    """Per-name dtype tag (``"int"`` or ``"float"``) so backends that
    distinguish integer and real declarations (Stan, JAGS, BUGS, PyMC)
    get the right native type after reshape.

    Resolution order, most authoritative first:

    1. **The declared family.** When the name answers to a stochastic
       site of the compiled program, its
       [`site_supports`][tests.transpile._gallery_data.site_supports]
       constraint decides: an integer-supported site (a Poisson /
       Binomial / Categorical / Bernoulli observation) is ``"int"``
       and a continuous-supported one is ``"float"``, whatever its
       ground-truth value happens to look like. This is what keeps a
       continuous latent whose ground truth is integer-valued (a
       horseshoe global scale pinned at ``tau = 1.0``, unit local
       scales) out of the ``int`` bucket: the in-container reshape
       casts every ``int``-tagged leaf through Python's `int`, which
       truncates the value and makes the backend's density piecewise
       constant in integer buckets rather than equal to the QVR
       reference.
    2. **The declared torch dtype**, for a name with no site: an
       integer-kinded covariate (a plate subscript such as
       ``coef_idx``) is ``"int"``.
    3. **The value domain**, for a float-kinded covariate: an
       all-integer-valued one is ``"int"``. This last rule is the only
       place the value heuristic survives, and it applies to
       covariates alone: an observation's kind now comes from its
       family instead.
    """
    integer_dtypes = (
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, torch.bool,
    )
    supports = _gallery_data.site_supports(dataset)
    out: dict[str, str] = {}
    for section in (dataset.observations, dataset.params):
        for name, tensor in section.items():
            support = supports.get(name)
            if support is not None:
                out[name] = (
                    "int"
                    if _gallery_data.is_discrete_support(support)
                    else "float"
                )
            elif tensor.dtype in integer_dtypes:
                out[name] = "int"
            elif tensor.numel() > 0 and torch.equal(
                tensor, tensor.round(),
            ):
                out[name] = "int"
            else:
                out[name] = "float"
    # A scalar type-parameter is declared `Real` in the program header
    # and rendered as a real input (`real alpha;` in Stan's data block,
    # `rep_vector(alpha, 3)` as a Dirichlet concentration), so it is
    # float-tagged whatever its instantiated value looks like: an
    # integer cast of `alpha = 1.0` would still parse and still be
    # wrong the moment the snippet instantiates at a fractional value.
    for name in dataset.scalar_params:
        out[name] = "float"
    return out
