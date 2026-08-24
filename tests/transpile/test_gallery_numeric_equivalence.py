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

# (backend, example) cells whose in-container probe script does not
# register the per-example data shapes the gallery example needs.
# The backend-side probe scripts in
# `tests/transpile/probes/_scripts/` were written for the dedicated
# composition fixtures; arbitrary gallery datasets need per-example
# shape registration to reshape the flat `Point` lists back into
# the tensors the model expects. Generalising the probe scripts is
# the closure path.
_SKIP_PROBE_INCOMPATIBLE: frozenset[tuple[str, str]] = frozenset({
    ('bugs', 'continuous_hmm'),
    ('bugs', 'factor_analysis'),
    ('bugs', 'gru_lm'),
    ('bugs', 'linear_gaussian_ssm'),
    ('bugs', 'lstm_lm'),
    ('bugs', 'ppca'),
    ('bugs', 'tree_categorical'),
    ('bugs', 'vanilla_rnn_lm'),
    ('edward2', 'beta_regression'),
    ('edward2', 'changepoint'),
    ('edward2', 'continuous_hmm'),
    ('edward2', 'gamma_regression'),
    ('edward2', 'gru_lm'),
    ('edward2', 'hmm'),
    ('edward2', 'horseshoe_regression'),
    ('edward2', 'lda'),
    ('edward2', 'linear_gaussian_ssm'),
    ('edward2', 'lstm_lm'),
    ('edward2', 'negbin_regression'),
    ('edward2', 'tree_categorical'),
    ('edward2', 'vanilla_rnn_lm'),
    ('edward2', 'zip_regression'),
    ('gen', 'continuous_hmm'),
    ('gen', 'factor_analysis'),
    ('gen', 'gru_lm'),
    ('gen', 'hmm'),
    ('gen', 'lda'),
    ('gen', 'linear_gaussian_ssm'),
    ('gen', 'lstm_lm'),
    ('gen', 'ppca'),
    ('gen', 'tree_categorical'),
    ('gen', 'vanilla_rnn_lm'),
    ('gen', 'zip_regression'),
    ('jags', 'continuous_hmm'),
    ('jags', 'factor_analysis'),
    ('jags', 'gru_lm'),
    ('jags', 'hmm'),
    ('jags', 'lda'),
    ('jags', 'linear_gaussian_ssm'),
    ('jags', 'lstm_lm'),
    ('jags', 'ppca'),
    ('jags', 'tree_categorical'),
    ('jags', 'vanilla_rnn_lm'),
    ('numpyro', 'continuous_hmm'),
    ('numpyro', 'gru_lm'),
    ('numpyro', 'hmm'),
    ('numpyro', 'lda'),
    ('numpyro', 'linear_gaussian_ssm'),
    ('numpyro', 'lstm_lm'),
    ('numpyro', 'tree_categorical'),
    ('numpyro', 'vanilla_rnn_lm'),
    ('numpyro', 'zip_regression'),
    ('pymc', 'continuous_hmm'),
    ('pymc', 'gru_lm'),
    ('pymc', 'hmm'),
    ('pymc', 'lda'),
    ('pymc', 'linear_gaussian_ssm'),
    ('pymc', 'lstm_lm'),
    ('pymc', 'tree_categorical'),
    ('pymc', 'vanilla_rnn_lm'),
    ('pymc', 'zip_regression'),
    ('pyro', 'beta_regression'),
    ('pyro', 'continuous_hmm'),
    ('pyro', 'gamma_regression'),
    ('pyro', 'gru_lm'),
    ('pyro', 'hmm'),
    ('pyro', 'horseshoe_regression'),
    ('pyro', 'lda'),
    ('pyro', 'linear_gaussian_ssm'),
    ('pyro', 'lstm_lm'),
    ('pyro', 'negbin_regression'),
    ('pyro', 'stochastic_volatility'),
    ('pyro', 'tree_categorical'),
    ('pyro', 'vanilla_rnn_lm'),
    ('pyro', 'zip_regression'),
    ('stan', 'continuous_hmm'),
    ('stan', 'gru_lm'),
    ('stan', 'hmm'),
    ('stan', 'lda'),
    ('stan', 'linear_gaussian_ssm'),
    ('stan', 'lstm_lm'),
    ('stan', 'tree_categorical'),
    ('stan', 'vanilla_rnn_lm'),
    ('stan', 'zip_regression'),
    ('turing', 'continuous_hmm'),
    ('turing', 'factor_analysis'),
    ('turing', 'gru_lm'),
    ('turing', 'hmm'),
    ('turing', 'lda'),
    ('turing', 'linear_gaussian_ssm'),
    ('turing', 'lstm_lm'),
    ('turing', 'ppca'),
    ('turing', 'tree_categorical'),
    ('turing', 'vanilla_rnn_lm'),
    ('turing', 'zip_regression'),
    ('webppl', 'beta_regression'),
    ('webppl', 'continuous_hmm'),
    ('webppl', 'factor_analysis'),
    ('webppl', 'gamma_regression'),
    ('webppl', 'gru_lm'),
    ('webppl', 'hmm'),
    ('webppl', 'horseshoe_regression'),
    ('webppl', 'irt_2pl'),
    ('webppl', 'lda'),
    ('webppl', 'linear_gaussian_ssm'),
    ('webppl', 'lstm_lm'),
    ('webppl', 'negbin_regression'),
    ('webppl', 'ppca'),
    ('webppl', 'stochastic_volatility'),
    ('webppl', 'survival_weibull'),
    ('webppl', 'tree_categorical'),
    ('webppl', 'vanilla_rnn_lm'),
    ('webppl', 'zip_regression'),
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

    Two properties make
    [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
    a real test rather than a tautology, and both are asserted here so
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
    return out
