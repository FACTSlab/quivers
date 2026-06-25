"""Tier-4 numeric equivalence on the documentation gallery.

For every `docs/examples/source/<example>.qvr` that ships a `###
Generating synthetic data` block in its sibling `.md`, this test:

1. Extracts and executes the data-gen snippet to recover an
   `observations` dict plus every captured `true_*` ground-truth
   parameter value.
2. Builds a single test
   [`Point`][tests.transpile.probes._protocol.Point] anchoring
   the ground-truth params and observations.
3. Runs the in-process
   [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe] to compute
   `log p_QVR(θ_true, y) = sum_i log f_i(...)` at that point.
4. For every backend whose Docker image is locally built, runs the
   target's native log-density probe inside the container and
   asserts constant-spread equivalence (`max_i | δ_i − mean δ | <
   1e-6`) per Theorem 4.1 of
   [docs/semantics/transpile-correctness.md](../../docs/semantics/transpile-correctness.md).

Each cell resolves to one of three pre-declared outcomes:

1. `(backend, example) in _EXPECTED_TRANSPILE_RAISES` — the
   pipeline MUST `pytest.raises(UnsupportedConstruct)` with the
   pinned kind-prefix.
2. Cell falls in one of the four `_SKIP_*` registries — known
   environmental gap (missing data block, QVR-probe incompatibility,
   backend probe script lacking shape registration for arbitrary
   gallery datasets). `pytest.skip` with the diagnostic.
3. Neither — the pipeline MUST emit non-empty bytes, the QVR probe
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

import pathlib

import pytest
import torch

from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _gallery_data
from tests.transpile.probes import _protocol
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
# the kind-prefix; a closed gap surfaces as a test failure.
_EXPECTED_TRANSPILE_RAISES: dict[tuple[str, str], str] = {}

# Gallery examples whose `.md` synthetic-data snippet fails to
# `exec` (raises at runtime, or never sets the `observations`
# dict). Numeric evaluation has no point set, so the cell skips.
_SKIP_DATASET_LOAD_FAILED: frozenset[str] = frozenset()

# Gallery examples the in-process [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe]
# cannot evaluate (missing categorical / monadic primitives in the
# in-process trace; the trace runtime grows the case and the entry
# comes out).
_SKIP_QVR_INCOMPATIBLE: frozenset[str] = frozenset()

# (backend, example) cells whose in-container probe script does not
# register the per-example data shapes the gallery example needs.
# The backend-side probe scripts in
# `tests/transpile/probes/_scripts/` were written for the dedicated
# composition fixtures; arbitrary gallery datasets need per-example
# shape registration to reshape the flat `Point` lists back into
# the tensors the model expects. Generalising the probe scripts is
# the closure path.
_SKIP_PROBE_INCOMPATIBLE: frozenset[tuple[str, str]] = frozenset({
    ("bugs", "ar1"),
    ("bugs", "bayesian_regression"),
    ("bugs", "beta_regression"),
    ("bugs", "bidirectional_rnn_lm"),
    ("bugs", "changepoint"),
    ("bugs", "continuous_hmm"),
    ("bugs", "deep_markov"),
    ("bugs", "factor_analysis"),
    ("bugs", "gamma_regression"),
    ("bugs", "gru_lm"),
    ("bugs", "hmm"),
    ("bugs", "horseshoe_regression"),
    ("bugs", "irt_2pl"),
    ("bugs", "lda"),
    ("bugs", "linear_gaussian_ssm"),
    ("bugs", "lstm_lm"),
    ("bugs", "mixture_model"),
    ("bugs", "negbin_regression"),
    ("bugs", "pmf"),
    ("bugs", "ppca"),
    ("bugs", "seq2seq"),
    ("bugs", "stochastic_volatility"),
    ("bugs", "survival_weibull"),
    ("bugs", "transformer_lm"),
    ("bugs", "tree_categorical"),
    ("bugs", "vae"),
    ("bugs", "vanilla_rnn_lm"),
    ("bugs", "zip_regression"),
    ("edward2", "ar1"),
    ("edward2", "bayesian_regression"),
    ("edward2", "beta_regression"),
    ("edward2", "bidirectional_rnn_lm"),
    ("edward2", "changepoint"),
    ("edward2", "continuous_hmm"),
    ("edward2", "deep_markov"),
    ("edward2", "factor_analysis"),
    ("edward2", "gamma_regression"),
    ("edward2", "gru_lm"),
    ("edward2", "hmm"),
    ("edward2", "horseshoe_regression"),
    ("edward2", "irt_2pl"),
    ("edward2", "lda"),
    ("edward2", "linear_gaussian_ssm"),
    ("edward2", "lstm_lm"),
    ("edward2", "mixture_model"),
    ("edward2", "negbin_regression"),
    ("edward2", "pmf"),
    ("edward2", "ppca"),
    ("edward2", "seq2seq"),
    ("edward2", "stochastic_volatility"),
    ("edward2", "survival_weibull"),
    ("edward2", "transformer_lm"),
    ("edward2", "tree_categorical"),
    ("edward2", "vae"),
    ("edward2", "vanilla_rnn_lm"),
    ("edward2", "zip_regression"),
    ("gen", "beta_regression"),
    ("gen", "bidirectional_rnn_lm"),
    ("gen", "changepoint"),
    ("gen", "continuous_hmm"),
    ("gen", "deep_markov"),
    ("gen", "factor_analysis"),
    ("gen", "gamma_regression"),
    ("gen", "gru_lm"),
    ("gen", "hmm"),
    ("gen", "horseshoe_regression"),
    ("gen", "irt_2pl"),
    ("gen", "lda"),
    ("gen", "linear_gaussian_ssm"),
    ("gen", "lstm_lm"),
    ("gen", "mixture_model"),
    ("gen", "negbin_regression"),
    ("gen", "pmf"),
    ("gen", "ppca"),
    ("gen", "seq2seq"),
    ("gen", "survival_weibull"),
    ("gen", "transformer_lm"),
    ("gen", "tree_categorical"),
    ("gen", "vae"),
    ("gen", "vanilla_rnn_lm"),
    ("gen", "zip_regression"),
    ("jags", "ar1"),
    ("jags", "bayesian_regression"),
    ("jags", "beta_regression"),
    ("jags", "bidirectional_rnn_lm"),
    ("jags", "changepoint"),
    ("jags", "continuous_hmm"),
    ("jags", "deep_markov"),
    ("jags", "factor_analysis"),
    ("jags", "gamma_regression"),
    ("jags", "gru_lm"),
    ("jags", "hmm"),
    ("jags", "horseshoe_regression"),
    ("jags", "irt_2pl"),
    ("jags", "lda"),
    ("jags", "linear_gaussian_ssm"),
    ("jags", "lstm_lm"),
    ("jags", "mixture_model"),
    ("jags", "negbin_regression"),
    ("jags", "pmf"),
    ("jags", "ppca"),
    ("jags", "seq2seq"),
    ("jags", "stochastic_volatility"),
    ("jags", "survival_weibull"),
    ("jags", "transformer_lm"),
    ("jags", "tree_categorical"),
    ("jags", "vae"),
    ("jags", "vanilla_rnn_lm"),
    ("jags", "zip_regression"),
    ("numpyro", "ar1"),
    ("numpyro", "bayesian_regression"),
    ("numpyro", "beta_regression"),
    ("numpyro", "bidirectional_rnn_lm"),
    ("numpyro", "changepoint"),
    ("numpyro", "continuous_hmm"),
    ("numpyro", "deep_markov"),
    ("numpyro", "factor_analysis"),
    ("numpyro", "gamma_regression"),
    ("numpyro", "gru_lm"),
    ("numpyro", "hmm"),
    ("numpyro", "horseshoe_regression"),
    ("numpyro", "irt_2pl"),
    ("numpyro", "lda"),
    ("numpyro", "linear_gaussian_ssm"),
    ("numpyro", "lstm_lm"),
    ("numpyro", "mixture_model"),
    ("numpyro", "negbin_regression"),
    ("numpyro", "pmf"),
    ("numpyro", "ppca"),
    ("numpyro", "seq2seq"),
    ("numpyro", "stochastic_volatility"),
    ("numpyro", "survival_weibull"),
    ("numpyro", "transformer_lm"),
    ("numpyro", "tree_categorical"),
    ("numpyro", "vae"),
    ("numpyro", "vanilla_rnn_lm"),
    ("numpyro", "zip_regression"),
    ("pymc", "beta_regression"),
    ("pymc", "bidirectional_rnn_lm"),
    ("pymc", "changepoint"),
    ("pymc", "continuous_hmm"),
    ("pymc", "deep_markov"),
    ("pymc", "factor_analysis"),
    ("pymc", "gamma_regression"),
    ("pymc", "gru_lm"),
    ("pymc", "hmm"),
    ("pymc", "horseshoe_regression"),
    ("pymc", "irt_2pl"),
    ("pymc", "lda"),
    ("pymc", "linear_gaussian_ssm"),
    ("pymc", "lstm_lm"),
    ("pymc", "mixture_model"),
    ("pymc", "negbin_regression"),
    ("pymc", "pmf"),
    ("pymc", "ppca"),
    ("pymc", "seq2seq"),
    ("pymc", "stochastic_volatility"),
    ("pymc", "survival_weibull"),
    ("pymc", "transformer_lm"),
    ("pymc", "tree_categorical"),
    ("pymc", "vae"),
    ("pymc", "vanilla_rnn_lm"),
    ("pymc", "zip_regression"),
    ("pyro", "beta_regression"),
    ("pyro", "bidirectional_rnn_lm"),
    ("pyro", "changepoint"),
    ("pyro", "continuous_hmm"),
    ("pyro", "deep_markov"),
    ("pyro", "factor_analysis"),
    ("pyro", "gamma_regression"),
    ("pyro", "gru_lm"),
    ("pyro", "hmm"),
    ("pyro", "horseshoe_regression"),
    ("pyro", "irt_2pl"),
    ("pyro", "lda"),
    ("pyro", "linear_gaussian_ssm"),
    ("pyro", "lstm_lm"),
    ("pyro", "mixture_model"),
    ("pyro", "negbin_regression"),
    ("pyro", "pmf"),
    ("pyro", "ppca"),
    ("pyro", "seq2seq"),
    ("pyro", "stochastic_volatility"),
    ("pyro", "survival_weibull"),
    ("pyro", "transformer_lm"),
    ("pyro", "tree_categorical"),
    ("pyro", "vae"),
    ("pyro", "vanilla_rnn_lm"),
    ("pyro", "zip_regression"),
    ("stan", "beta_regression"),
    ("stan", "bidirectional_rnn_lm"),
    ("stan", "continuous_hmm"),
    ("stan", "deep_markov"),
    ("stan", "factor_analysis"),
    ("stan", "gamma_regression"),
    ("stan", "gru_lm"),
    ("stan", "hmm"),
    ("stan", "horseshoe_regression"),
    ("stan", "irt_2pl"),
    ("stan", "lda"),
    ("stan", "linear_gaussian_ssm"),
    ("stan", "lstm_lm"),
    ("stan", "mixture_model"),
    ("stan", "negbin_regression"),
    ("stan", "ppca"),
    ("stan", "seq2seq"),
    ("stan", "transformer_lm"),
    ("stan", "tree_categorical"),
    ("stan", "vae"),
    ("stan", "vanilla_rnn_lm"),
    ("stan", "zip_regression"),
    ("turing", "ar1"),
    ("turing", "bayesian_regression"),
    ("turing", "beta_regression"),
    ("turing", "bidirectional_rnn_lm"),
    ("turing", "changepoint"),
    ("turing", "continuous_hmm"),
    ("turing", "deep_markov"),
    ("turing", "factor_analysis"),
    ("turing", "gamma_regression"),
    ("turing", "gru_lm"),
    ("turing", "hmm"),
    ("turing", "horseshoe_regression"),
    ("turing", "irt_2pl"),
    ("turing", "lda"),
    ("turing", "linear_gaussian_ssm"),
    ("turing", "lstm_lm"),
    ("turing", "mixture_model"),
    ("turing", "negbin_regression"),
    ("turing", "pmf"),
    ("turing", "ppca"),
    ("turing", "seq2seq"),
    ("turing", "stochastic_volatility"),
    ("turing", "survival_weibull"),
    ("turing", "transformer_lm"),
    ("turing", "tree_categorical"),
    ("turing", "vae"),
    ("turing", "vanilla_rnn_lm"),
    ("turing", "zip_regression"),
    ("webppl", "ar1"),
    ("webppl", "bayesian_regression"),
    ("webppl", "beta_regression"),
    ("webppl", "bidirectional_rnn_lm"),
    ("webppl", "changepoint"),
    ("webppl", "continuous_hmm"),
    ("webppl", "deep_markov"),
    ("webppl", "factor_analysis"),
    ("webppl", "gamma_regression"),
    ("webppl", "gru_lm"),
    ("webppl", "hmm"),
    ("webppl", "horseshoe_regression"),
    ("webppl", "irt_2pl"),
    ("webppl", "lda"),
    ("webppl", "linear_gaussian_ssm"),
    ("webppl", "lstm_lm"),
    ("webppl", "mixture_model"),
    ("webppl", "negbin_regression"),
    ("webppl", "pmf"),
    ("webppl", "ppca"),
    ("webppl", "seq2seq"),
    ("webppl", "stochastic_volatility"),
    ("webppl", "survival_weibull"),
    ("webppl", "transformer_lm"),
    ("webppl", "tree_categorical"),
    ("webppl", "vae"),
    ("webppl", "vanilla_rnn_lm"),
    ("webppl", "zip_regression"),
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
    import math
    assert math.isfinite(lp), (
        f"{example.stem!r}: QVR log p(θ_true, y) is non-finite ({lp!r}); "
        f"observations={list(dataset.observations)} params={list(dataset.params)}"
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
            transpile(
                __import__("quivers.dsl.parser", fromlist=["parse"]).parse(
                    source.decode("utf-8")
                ),
                target=backend,
            )
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
    emitted = transpile(
        __import__("quivers.dsl.parser", fromlist=["parse"]).parse(
            source.decode("utf-8")
        ),
        target=backend,
    )

    point = _gallery_data.point_from_dataset(dataset)
    qvr_probe = QvrProbe()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_eq_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    qvr_result = qvr_probe.evaluate(
        source,
        example.stem,
        [point],
        scratch=scratch,
        monadic=dataset.monadic,
        x_input=dataset.x_input,
        observations=dataset.observations,
    )

    script_path = (
        pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    )
    raw_result = _docker.run_probe(
        image=image,
        script=script_path,
        source=emitted,
        source_ext=ext,
        points=[{"params": point.params, "data": point.data}],
        scratch=scratch,
        shapes=_shapes_from_dataset(dataset),
        dtypes=_dtypes_from_dataset(dataset),
        helpers=list(_PROBE_HELPERS),
    )

    backend_lps = [float(x) for x in raw_result["log_densities"]]

    from tests.transpile import _equivalence

    _equivalence.assert_log_density_match(
        qvr_result.log_densities,
        backend_lps,
        context=f"{backend}@{example.stem}",
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

    A tensor is tagged ``"int"`` when its declared dtype is one of
    the torch integer kinds OR when every element is integer-valued
    (no fractional part). The latter rule covers the common gallery
    pattern where `torch.poisson` / `torch.distributions.Binomial`
    returns ``float32`` counts that the model nevertheless declares
    as ``int`` (Stan / JAGS / BUGS reject floats for int variables).
    """
    integer_dtypes = (
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, torch.bool,
    )
    out: dict[str, str] = {}
    for section in (dataset.observations, dataset.params):
        for name, tensor in section.items():
            if tensor.dtype in integer_dtypes:
                out[name] = "int"
            elif tensor.numel() > 0 and torch.equal(
                tensor, tensor.round(),
            ):
                out[name] = "int"
            else:
                out[name] = "float"
    return out


_PROBE_HELPERS: tuple[pathlib.Path, ...] = (
    pathlib.Path(__file__).parent
    / "probes" / "_scripts" / "_reshape.py",
    pathlib.Path(__file__).parent
    / "probes" / "_scripts" / "_reshape.jl",
)
