"""Regression tests for the audit-confirmed PyMC transpilation fixes.

Each test transpiles a model exercising one family to the PyMC
backend and asserts on the emitted text: the corrected distribution
name, keyword, argument order, and value. The emitted program's joint
log-density must match the QVR model's up to an additive constant, so
these assertions pin the exact call shape that carries that density.

The fixes live in
[`renderers/pymc.py`][quivers.transpile.renderers.pymc] and the
grafted [`runtime_pymc.py`][quivers.transpile.runtime_pymc] helper.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types

import numpy as np
import pymc
import pytensor
import pytensor.tensor as pt
import torch
from pymc.distributions.transforms import CholeskyCorrTransform

from quivers.dsl.parser import parse
from quivers.transpile import transpile
from quivers.transpile.family_meta import FAMILY_META
from quivers.transpile.renderers.pymc import _merged_aliases


def _emit(source: str) -> str:
    return transpile(parse(source), target="pymc").decode("utf-8")


def _stripped(source: str) -> str:
    """Emit `source` with all whitespace removed, for spacing-robust
    substring checks."""
    return "".join(_emit(source).split())


def test_poisson_rate_renamed_to_mu() -> None:
    """PyMC `Poisson` takes `mu`, not the torch `rate`."""
    src = (
        "object O : Real 4\n"
        "program g : O -> O\n"
        "    sample t <- Poisson(3.0)\n"
        "    return t\n"
        "export g"
    )
    out = _emit(src)
    assert "pymc.Poisson(" in out
    assert "mu=3" in out
    assert "rate=" not in out


def test_chi2_df_renamed_to_nu() -> None:
    """PyMC `ChiSquared` takes `nu`, not the torch `df`."""
    src = (
        "object Obs : Real 4\n"
        "morphism k : Obs -> Obs [role=kernel] ~ Chi2(3.0)\n"
        "program g : Obs -> Obs\n"
        "    sample x <- k\n"
        "    return x\n"
        "export g"
    )
    out = _emit(src)
    assert "pymc.ChiSquared(" in out
    assert "nu=3" in out
    assert "df=" not in out


def test_kumaraswamy_concentrations_renamed_to_a_b() -> None:
    """PyMC `Kumaraswamy` takes `a` / `b`, not `concentration1` /
    `concentration0`."""
    src = (
        "object Obs : Real 4\n"
        "morphism k : Obs -> Obs [role=kernel] ~ Kumaraswamy(2.0, 3.0)\n"
        "program g : Obs -> Obs\n"
        "    sample x <- k\n"
        "    return x\n"
        "export g"
    )
    out = _emit(src)
    assert "pymc.Kumaraswamy(" in out
    assert "a=2" in out
    assert "b=3" in out
    assert "concentration1" not in out
    assert "concentration0" not in out


def test_negative_binomial_probs_complemented() -> None:
    """PyMC's `NegativeBinomial(n, p)` uses `p == 1 - probs` relative
    to torch's success probability."""
    out = _emit(open("docs/examples/source/negbin_regression.qvr").read())
    assert "pymc.NegativeBinomial(" in out
    assert "p=(1-probs)" in out
    # The count parameter passes through unchanged.
    assert "n=disp" in out


def test_geometric_latent_shifted_off_by_one() -> None:
    """torch `Geometric` counts failures on `{0, 1, ...}`; PyMC counts
    trials on `{1, 2, ...}`. A latent sample keeps the density on the
    PyMC RV and exposes the torch-convention value via a shifted
    `Deterministic`."""
    src = (
        "object O : FinSet 4\n"
        "program g : O -> O\n"
        "    sample t : O <- Geometric(0.3)\n"
        "    return t\n"
        "export g"
    )
    out = _emit(src)
    assert 'pymc.Geometric("t__geom"' in out
    assert 'pymc.Deterministic("t"' in out
    assert "-1" in out
    # The success probability is shared, not complemented.
    assert "p=0.3" in out


def test_geometric_observed_shifted_off_by_one() -> None:
    """An observed `Geometric` feeds `observed + 1` so torch-convention
    data land in PyMC's `{1, 2, ...}` support."""
    src = (
        "object O : FinSet 4\n"
        "program g : O -> O\n"
        "    observe t : O <- Geometric(0.3)\n"
        "    return t\n"
        "export g"
    )
    out = _emit(src)
    assert 'pymc.Geometric("t"' in out
    assert "observed=(t+1)" in out


def test_continuous_bernoulli_grafts_custom_dist_helper() -> None:
    """PyMC ships no `ContinuousBernoulli`; the renderer grafts a
    `CustomDist`-backed helper and calls it by bare name."""
    out = _emit(open("tests/transpile/fixtures/families/continuousbernoulli.qvr").read())
    assert "def ContinuousBernoulli(" in out
    assert "pymc.CustomDist(" in out
    assert 'ContinuousBernoulli("x"' in out
    # The nonexistent `pymc.ContinuousBernoulli` name is never emitted.
    assert "pymc.ContinuousBernoulli(" not in out


def test_wishart_alias_maps_covariance_to_capital_v() -> None:
    """PyMC `Wishart(nu, V)`: the scale matrix is `V`, not the
    torch `covariance_matrix` nor PyMC's `scale_matrix`."""
    aliases = _merged_aliases(FAMILY_META["Wishart"])
    assert aliases["covariance_matrix"] == "V"
    assert aliases["df"] == "nu"
    assert "scale_matrix" not in aliases.values()


def test_wishart_call_site_emits_nu_and_capital_v() -> None:
    """The alias reaches the emitted call: `Wishart("s", nu=..., V=...)`."""
    src = (
        "object Dim : FinSet 2\n"
        "program g : Dim -> Dim\n"
        "    sample s : Dim <- Wishart(5.0, [[1.0, 0.0], [0.0, 1.0]])\n"
        "    return s\n"
        "export g"
    )
    out = _emit(src)
    assert "nu=5" in out
    assert "V=np.array([[1,0],[0,1]])" in "".join(out.split())
    assert "scale_matrix" not in out
    assert "covariance_matrix" not in out


# ---------------------------------------------------------------------------
# LKJCholesky
# ---------------------------------------------------------------------------


_LKJ_SOURCE = (
    "object Dim : FinSet 4\n"
    "program correlation_model : Dim -> Dim\n"
    "    sample eta <- HalfNormal(2.0)\n"
    "    sample chol : Dim <- LKJCholesky(eta)\n"
    "    return chol\n"
    "export correlation_model"
)


#: A three-by-three correlation factor at fixed concentration `eta = 1`,
#: whose LKJ off-diagonal marginal has the analytic standard deviation
#: `0.5`. A literal concentration keeps the marginal fixed, so the
#: sampler's draws compare against a closed-form target.
_LKJ_SOURCE_D3 = (
    "object Dim : FinSet 3\n"
    "program correlation_model : Dim -> Dim\n"
    "    sample chol : Dim <- LKJCholesky(1.0)\n"
    "    return chol\n"
    "export correlation_model"
)


def test_lkj_cholesky_calls_runtime_helper_with_dimension() -> None:
    """`LKJCholesky` over a `FinSet 4` event axis emits the grafted
    helper call `LKJCholesky("chol", n=4, eta=eta)`: the matrix
    dimension leads, the concentration follows under PyMC's `eta`."""
    out = _emit(_LKJ_SOURCE)
    assert 'LKJCholesky("chol",n=4,eta=eta)' in "".join(out.split()), out


def test_lkj_cholesky_never_emits_lkjcholeskycov() -> None:
    """`LKJCholeskyCov` multiplies in a standard-deviation prior the
    QVR model does not have, and demands `n` / `eta` / `sd_dist`, so it
    is the wrong target. The rejected `concentration=` keyword and the
    single-name `dims` tuple for a square-matrix variable must be gone
    too."""
    out = _emit(_LKJ_SOURCE)
    assert "LKJCholeskyCov" not in out
    assert "concentration=" not in out
    assert 'LKJCholesky("chol",n=4,eta=eta,dims=' not in "".join(out.split())


def test_lkj_cholesky_grafts_runtime_helper_definitions() -> None:
    """The emitted module carries the helper and its log-density, so
    the bare-name call resolves."""
    out = _emit(_LKJ_SOURCE)
    assert "def LKJCholesky(name, n, eta, **kwargs):" in out
    assert "def _lkj_cholesky_logp(value, n, eta):" in out
    assert "def _lkj_cholesky_log_normalizer(n, eta):" in out


def _load_emitted_module(
    source: str, path: pathlib.Path, name: str
) -> types.ModuleType:
    """Write the emitted PyMC source under the imports its grafted
    helpers assume, import it, and return the module."""
    path.write_text(
        "import numpy as np\n"
        "import pymc\n"
        "import pytensor.tensor as pt\n\n" + _emit(source)
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def _free_lower(matrix: np.ndarray) -> np.ndarray:
    """The strictly-lower-triangular entries of `matrix`, the free
    coordinates of a correlation Cholesky factor."""
    dim = int(matrix.shape[-1])
    return np.array([matrix[i, j] for i in range(1, dim) for j in range(i)])


def test_lkj_cholesky_log_density_matches_torch(
    tmp_path: pathlib.Path,
) -> None:
    """The emitted model's log-density over the LKJ factor equals
    `torch.distributions.LKJCholesky.log_prob` up to an additive
    constant, which is the transpile correctness contract.

    Both sides are compared in PyMC's unconstrained coordinates: the
    torch density on the factor's free coordinates gains the log
    determinant of the Jacobian of the unconstraining map, computed by
    central differences."""
    module = _load_emitted_module(
        _LKJ_SOURCE, tmp_path / "emitted_lkj.py", "emitted_lkj_pymc"
    )
    model = module.build_model()
    # Only the LKJ term, so the `eta` prior's own density stays out of
    # the comparison.
    logp = model.compile_logp(vars=[model["chol"]], sum=True)

    dim = 4
    free = dim * (dim - 1) // 2
    transform = CholeskyCorrTransform(n=dim, upper=False)
    z_sym = pt.vector("z")
    compiled = pytensor.function([z_sym], transform.backward(z_sym))

    def backward(z: np.ndarray) -> np.ndarray:
        return np.asarray(compiled(z), dtype="float64")

    # `eta` is held fixed, so only the LKJ term varies across draws.
    eta_unconstrained = 0.4
    reference = torch.distributions.LKJCholesky(
        dim, torch.tensor(math.exp(eta_unconstrained))
    )

    def jacobian_logdet(z: np.ndarray, step: float = 1e-6) -> float:
        jac = np.zeros((free, free))
        for k in range(free):
            plus, minus = z.copy(), z.copy()
            plus[k] += step
            minus[k] -= step
            jac[:, k] = (
                _free_lower(backward(plus)) - _free_lower(backward(minus))
            ) / (2.0 * step)
        return float(np.linalg.slogdet(jac)[1])

    rng = np.random.default_rng(0)
    offsets: list[float] = []
    for _ in range(4):
        z = rng.normal(size=free) * 0.8
        emitted = float(
            logp({"eta_log__": eta_unconstrained, "chol_cholesky_corr__": z})
        )
        expected = (
            reference.log_prob(torch.tensor(backward(z))).item()
            + jacobian_logdet(z)
        )
        offsets.append(emitted - expected)

    # Agreement up to an additive constant, and in fact outright: the
    # helper carries the normalising constant.
    assert max(offsets) - min(offsets) < 1e-5, offsets
    assert abs(float(np.mean(offsets))) < 1e-5, offsets


def test_lkj_cholesky_draws_are_correlation_factors(
    tmp_path: pathlib.Path,
) -> None:
    """Prior draws of the emitted variable are lower-triangular with
    unit row norms, the support of a correlation Cholesky factor, and
    their reconstructed correlations are exchangeable across
    off-diagonals.

    Support membership alone (triangular shape, unit row norms) passes
    for any wrong-but-valid law over factors, so this test also checks
    exchangeability: every off-diagonal of the reconstructed
    correlation `L @ L.T` must share one marginal, which the LKJ law
    guarantees under any concentration (and holds after mixing over the
    `HalfNormal` `eta` prior). A per-off-diagonal sampler defect breaks
    this even when the support is respected."""
    module = _load_emitted_module(
        _LKJ_SOURCE,
        tmp_path / "emitted_lkj_draw.py",
        "emitted_lkj_draw_pymc",
    )
    model = module.build_model()
    with model:
        draws = np.asarray(
            pymc.draw(model["chol"], draws=120000, random_seed=11),
            dtype="float64",
        )
    assert draws.shape == (120000, 4, 4)
    assert np.allclose(np.triu(draws, 1), 0.0)
    assert np.allclose(np.sum(draws**2, axis=-1), 1.0)
    corr = draws @ np.transpose(draws, (0, 2, 1))
    off_diag_sds = np.array(
        [corr[:, i, j].std() for i in range(1, 4) for j in range(i)]
    )
    # All six off-diagonals share the LKJ marginal, so their sample
    # standard deviations agree.
    assert off_diag_sds.max() - off_diag_sds.min() < 0.02, off_diag_sds


def test_lkj_cholesky_draws_match_analytic_marginals(
    tmp_path: pathlib.Path,
) -> None:
    """Emitted draws for `d = 3`, `eta = 1` reproduce the LKJ
    off-diagonal marginal, whose analytic standard deviation is exactly
    `0.5` for every off-diagonal.

    This is the distributional correctness contract for the forward
    sampler: a factor law with the correct support but wrong
    off-diagonal marginals (the failure mode of PyMC's own `LKJCorr`
    onion sampler) would satisfy the support checks yet miss this
    target. The C-vine sampler grafted from `runtime_pymc` returns the
    Cholesky factor directly, so its reconstructed correlations carry
    the LKJ marginal. The analytic `sd = 0.5` ground truth is used
    rather than any library draw, since both `pymc.LKJCorr` and
    `torch.distributions.LKJCholesky.sample` mis-sample the
    off-diagonals."""
    module = _load_emitted_module(
        _LKJ_SOURCE_D3,
        tmp_path / "emitted_lkj_marginal.py",
        "emitted_lkj_marginal_pymc",
    )
    model = module.build_model()
    with model:
        draws = np.asarray(
            pymc.draw(model["chol"], draws=200000, random_seed=7),
            dtype="float64",
        )
    assert draws.shape == (200000, 3, 3)
    corr = draws @ np.transpose(draws, (0, 2, 1))
    pairs = [(1, 0), (2, 0), (2, 1)]
    off_diag_sds = np.array([corr[:, i, j].std() for (i, j) in pairs])
    off_diag_means = np.array([corr[:, i, j].mean() for (i, j) in pairs])
    # Every off-diagonal has analytic marginal sd 0.5 under LKJ(1) at
    # d = 3, and mean 0 by sign symmetry.
    assert np.allclose(off_diag_sds, 0.5, atol=0.01), off_diag_sds
    assert np.allclose(off_diag_means, 0.0, atol=0.01), off_diag_means
    # Exchangeability: the three off-diagonals share one marginal.
    assert off_diag_sds.max() - off_diag_sds.min() < 0.01, off_diag_sds
