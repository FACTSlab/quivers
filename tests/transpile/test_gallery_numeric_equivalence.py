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

# QVR reference joint log-densities, each verified against an
# independent raw-torch reconstruction of the model's joint at the
# ground-truth point. `test_gallery_qvr_logdensity_finite` asserts the
# QVR trace still reproduces these, turning a finite-only check into a
# correctness check. Update an entry only after re-deriving the joint
# independently.
_QVR_REFERENCE_JOINT: dict[str, float] = {
    "ar1": -24.0308,
    "bayesian_regression": -54.5608,
    "beta_regression": 26.0999,
    "bnn": -412.0953,
    "changepoint": -131.0847,
    "continuous_hmm": -716.6049,
    "factor_analysis": -132.6166,
    "gamma_regression": -69.0704,
    "hmm": 335.3818,
    "horseshoe_regression": -64.9299,
    "irt_2pl": -69.2327,
    "lda": 2550.3076,
    "linear_gaussian_ssm": -218.7775,
    "mixture_model": -189.3568,
    "negbin_regression": -203.6848,
    "parametric_pooling": -16.0530,
    "ppca": -67.7343,
    "stochastic_volatility": -307.8004,
    "survival_weibull": -28.6730,
    "tree_categorical": -14.3775,
    "zip_regression": -651.6888,
}

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
    # sum). For every model whose joint was verified against an
    # independent raw-torch reconstruction, assert the QVR reference
    # still equals that value. A drift here is either a real oracle
    # regression or a deliberate ground-truth change that must be
    # re-verified and the reference updated.
    reference = _QVR_REFERENCE_JOINT.get(example.stem)
    if reference is not None:
        assert abs(lp - reference) <= 1e-3 * abs(reference) + 2e-2, (
            f"{example.stem!r}: QVR joint {lp:.5f} drifted from its "
            f"independently-verified reference {reference:.5f}. Either the "
            f"oracle regressed or the ground-truth point changed; "
            f"re-derive the joint independently before updating the "
            f"`_QVR_REFERENCE_JOINT` entry."
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
