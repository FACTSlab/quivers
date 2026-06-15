"""Static transpile-only metadata for the registered distribution families.

The torch distribution class supplies `arg_constraints`, `.support`,
`event_shape`, `batch_shape`, and the natural parameterisation;
[`FamilyMeta`][quivers.transpile.family_meta.FamilyMeta] carries
only the transpile-specific facts that torch doesn't publish:

* `qvr_name`: the DSL-facing family name.
* `distribution_class`: the underlying
  [`torch.distributions.Distribution`][torch.distributions.Distribution]
  subclass.
* `target_names`: per-backend distribution-name mapping. The single
  source of truth for backend-to-distribution-name resolution. No
  per-renderer `_FAMILIES` dict.
* `arg_aliases`: per-backend per-arg renames. Most families have
  empty `arg_aliases`. Renderers that apply parameterisation-converting
  arithmetic (BUGS Normal mean/scale to mean/precision) key the
  arithmetic on the alias's target name.

Phase B tier-one families (`BetaBinomial`, `OrderedLogistic`,
`OrderedProbit`, `Logistic`, `HalfStudentT`) lack direct
[`torch.distributions`][torch.distributions] classes; this module
defines minimal `Distribution` subclasses carrying the right
`arg_constraints` and `support` so the lower pipeline can introspect
them. Several existing wrappers (`Horseshoe`, `GP`, `InverseWishart`,
`MatrixNormal`, `LogitNormal`, `TruncatedNormal`) also use this shim
mechanism for the metadata that the renderer pipeline reads. The
Phase A `Conditional*` classes from
[`quivers.continuous.families`][quivers.continuous.families] expose
their own class-level `arg_constraints` and `support` and serve as
their own `distribution_class`.

`finite_enumerable_at_call_site` is a per-call predicate (not a
per-family flag); it dispatches on the family name and the IR-form
of the user's args. Bernoulli, Categorical, OrderedLogistic, and
OrderedProbit are always finite-enumerable; Binomial is only when
its `total_count` is a literal `IRArgNumber`.
"""

from __future__ import annotations

import didactic.api as dx
import torch
import torch.distributions as td
import torch.distributions.constraints as c
from torch.distributions.distribution import Distribution

from quivers.continuous.families import (
    ConditionalBinomial,
    ConditionalGeometric,
    ConditionalIndependent,
    ConditionalLKJCholesky,
    ConditionalLogisticNormal,
    ConditionalMixture,
    ConditionalNegativeBinomial,
    ConditionalOneHotCategorical,
    ConditionalPoisson,
    ConditionalTransformed,
    ConditionalVonMises,
    LKJCorrelationFactor,
    Truncated,
)
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.transpile.ir import (
    IRArg,
    IRArgNumber,
)


class FamilyMeta(dx.Model):
    """Static transpile-only metadata for one distribution family.

    `distribution_class` accepts either a
    [`torch.distributions.Distribution`][torch.distributions.Distribution]
    subclass or a
    [`quivers.continuous.morphisms.ContinuousMorphism`][quivers.continuous.morphisms.ContinuousMorphism]
    subclass; both expose class-level `arg_constraints` and `support`
    attributes that the transpile lower pipeline reads."""

    qvr_name: str
    distribution_class: type[Distribution] | type[ContinuousMorphism] = dx.field(
        opaque=True
    )
    target_names: dict[str, str]
    arg_aliases: dict[str, dict[str, str]] = dx.field(
        default_factory=lambda: {}
    )


# ---------------------------------------------------------------------------
# Phase B tier 1: shim Distribution subclasses for families torch lacks.
# ---------------------------------------------------------------------------


class _BetaBinomial(Distribution):
    """Beta-Binomial: `Binomial(n, p)` with `p ~ Beta(c1, c0)`."""

    arg_constraints: dict[str, c.Constraint] = {
        "total_count": c.nonnegative_integer,
        "concentration1": c.positive,
        "concentration0": c.positive,
    }
    support: c.Constraint = c.nonnegative_integer
    has_rsample: bool = False

    def __init__(
        self,
        total_count: torch.Tensor,
        concentration1: torch.Tensor,
        concentration0: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.total_count = total_count
        self.concentration1 = concentration1
        self.concentration0 = concentration0
        super().__init__(validate_args=validate_args)


class _OrderedLogistic(Distribution):
    """Ordered-logistic over `len(cutpoints) + 1` ordered categories."""

    arg_constraints: dict[str, c.Constraint] = {
        "eta": c.real,
        "cutpoints": c.real_vector,
    }
    support: c.Constraint = c.nonnegative_integer
    has_rsample: bool = False

    def __init__(
        self,
        eta: torch.Tensor,
        cutpoints: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.eta = eta
        self.cutpoints = cutpoints
        super().__init__(validate_args=validate_args)


class _OrderedProbit(Distribution):
    """Ordered-probit over `len(cutpoints) + 1` ordered categories."""

    arg_constraints: dict[str, c.Constraint] = {
        "eta": c.real,
        "cutpoints": c.real_vector,
    }
    support: c.Constraint = c.nonnegative_integer
    has_rsample: bool = False

    def __init__(
        self,
        eta: torch.Tensor,
        cutpoints: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.eta = eta
        self.cutpoints = cutpoints
        super().__init__(validate_args=validate_args)


class _Logistic(Distribution):
    """Logistic location-scale distribution on the real line."""

    arg_constraints: dict[str, c.Constraint] = {
        "loc": c.real,
        "scale": c.positive,
    }
    support: c.Constraint = c.real
    has_rsample: bool = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.loc = loc
        self.scale = scale
        super().__init__(validate_args=validate_args)


class _HalfStudentT(Distribution):
    """Half-StudentT: a StudentT folded around zero (support on the
    nonnegative reals)."""

    arg_constraints: dict[str, c.Constraint] = {
        "df": c.positive,
        "scale": c.positive,
    }
    support: c.Constraint = c.positive
    has_rsample: bool = True

    def __init__(
        self,
        df: torch.Tensor,
        scale: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.df = df
        self.scale = scale
        super().__init__(validate_args=validate_args)


# ---------------------------------------------------------------------------
# Shim Distribution subclasses for existing QVR families torch lacks.
# ---------------------------------------------------------------------------


class _LogitNormal(Distribution):
    """`sigmoid(Normal(loc, scale))` on the open unit interval."""

    arg_constraints: dict[str, c.Constraint] = {
        "loc": c.real,
        "scale": c.positive,
    }
    support: c.Constraint = c.unit_interval
    has_rsample: bool = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.loc = loc
        self.scale = scale
        super().__init__(validate_args=validate_args)


class _TruncatedNormal(Distribution):
    """Normal truncated to a (low, high) interval."""

    arg_constraints: dict[str, c.Constraint] = {
        "loc": c.real,
        "scale": c.positive,
        "low": c.real,
        "high": c.real,
    }
    support: c.Constraint = c.real
    has_rsample: bool = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        super().__init__(validate_args=validate_args)


class _InverseWishart(Distribution):
    """Inverse-Wishart over positive-definite matrices."""

    arg_constraints: dict[str, c.Constraint] = {
        "df": c.positive,
        "scale_tril": c.lower_cholesky,
    }
    support: c.Constraint = c.positive_definite
    has_rsample: bool = False

    def __init__(
        self,
        df: torch.Tensor,
        scale_tril: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.df = df
        self.scale_tril = scale_tril
        super().__init__(validate_args=validate_args)


class _MatrixNormal(Distribution):
    """Matrix-Normal: a normal over real matrices with Kronecker covariance."""

    arg_constraints: dict[str, c.Constraint] = {
        "loc": c.independent(c.real, 2),
        "row_covariance": c.positive_definite,
        "col_covariance": c.positive_definite,
    }
    support: c.Constraint = c.independent(c.real, 2)
    has_rsample: bool = True

    def __init__(
        self,
        loc: torch.Tensor,
        row_covariance: torch.Tensor,
        col_covariance: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.loc = loc
        self.row_covariance = row_covariance
        self.col_covariance = col_covariance
        super().__init__(validate_args=validate_args)


class _GaussianProcess(Distribution):
    """Gaussian process realised at a finite collection of points."""

    arg_constraints: dict[str, c.Constraint] = {
        "mean": c.real_vector,
        "kernel": c.positive_definite,
    }
    support: c.Constraint = c.real_vector
    has_rsample: bool = True

    def __init__(
        self,
        mean: torch.Tensor,
        kernel: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.mean = mean
        self.kernel = kernel
        super().__init__(validate_args=validate_args)


class _Horseshoe(Distribution):
    """Horseshoe prior: `Normal(0, tau * lambda)` with sparsity scales."""

    arg_constraints: dict[str, c.Constraint] = {
        "scale": c.positive,
    }
    support: c.Constraint = c.real
    has_rsample: bool = True

    def __init__(
        self,
        scale: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.scale = scale
        super().__init__(validate_args=validate_args)


# ---------------------------------------------------------------------------
# FAMILY_META: the single transpile-time family registry.
# ---------------------------------------------------------------------------


FAMILY_META: dict[str, FamilyMeta] = {
    # ----- continuous scalar -----
    "Normal": FamilyMeta(
        qvr_name="Normal",
        distribution_class=td.Normal,
        target_names={
            "stan": "normal", "numpyro": "Normal", "pyro": "Normal",
            "pymc": "Normal", "edward2": "Normal",
            "turing": "Normal", "gen": "normal",
            "church": "gaussian", "webppl": "Gaussian",
            "bugs": "dnorm", "jags": "dnorm",
        },
        arg_aliases={
            "bugs": {"scale": "tau"},
            "jags": {"scale": "tau"},
        },
    ),
    "LogitNormal": FamilyMeta(
        qvr_name="LogitNormal",
        distribution_class=_LogitNormal,
        target_names={
            "stan": "logit_normal", "numpyro": "LogitNormal",
            "pyro": "LogitNormal", "pymc": "LogitNormal",
            "edward2": "LogitNormal",
        },
    ),
    "Beta": FamilyMeta(
        qvr_name="Beta",
        distribution_class=td.Beta,
        target_names={
            "stan": "beta", "numpyro": "Beta", "pyro": "Beta",
            "pymc": "Beta", "edward2": "Beta",
            "turing": "Beta", "gen": "beta",
            "church": "beta", "webppl": "Beta",
            "bugs": "dbeta", "jags": "dbeta",
        },
    ),
    "TruncatedNormal": FamilyMeta(
        qvr_name="TruncatedNormal",
        distribution_class=_TruncatedNormal,
        target_names={
            "numpyro": "TruncatedNormal", "pyro": "TruncatedNormal",
            "pymc": "TruncatedNormal", "edward2": "TruncatedNormal",
        },
    ),
    "Dirichlet": FamilyMeta(
        qvr_name="Dirichlet",
        distribution_class=td.Dirichlet,
        target_names={
            "stan": "dirichlet", "numpyro": "Dirichlet", "pyro": "Dirichlet",
            "pymc": "Dirichlet", "edward2": "Dirichlet",
            "turing": "Dirichlet", "gen": "dirichlet",
            "church": "dirichlet", "webppl": "Dirichlet",
            "bugs": "ddirch", "jags": "ddirich",
        },
        arg_aliases={
            "pymc": {"concentration": "a"},
        },
    ),
    "Cauchy": FamilyMeta(
        qvr_name="Cauchy",
        distribution_class=td.Cauchy,
        target_names={
            "stan": "cauchy", "numpyro": "Cauchy", "pyro": "Cauchy",
            "pymc": "Cauchy", "edward2": "Cauchy",
            "turing": "Cauchy", "gen": "cauchy",
            "church": "cauchy",
            "bugs": "dt", "jags": "dt",
        },
        arg_aliases={
            "bugs": {"scale": "tau"},
            "jags": {"scale": "tau"},
        },
    ),
    "Laplace": FamilyMeta(
        qvr_name="Laplace",
        distribution_class=td.Laplace,
        target_names={
            "stan": "double_exponential", "numpyro": "Laplace",
            "pyro": "Laplace", "pymc": "Laplace",
            "edward2": "Laplace", "turing": "Laplace",
            "gen": "laplace",
            "bugs": "ddexp", "jags": "ddexp",
        },
        arg_aliases={
            "bugs": {"scale": "tau"},
            "jags": {"scale": "tau"},
        },
    ),
    "Gumbel": FamilyMeta(
        qvr_name="Gumbel",
        distribution_class=td.Gumbel,
        target_names={
            "stan": "gumbel", "numpyro": "Gumbel", "pyro": "Gumbel",
            "pymc": "Gumbel", "edward2": "Gumbel",
            "turing": "Gumbel",
        },
    ),
    "LogNormal": FamilyMeta(
        qvr_name="LogNormal",
        distribution_class=td.LogNormal,
        target_names={
            "stan": "lognormal", "numpyro": "LogNormal", "pyro": "LogNormal",
            "pymc": "LogNormal", "edward2": "LogNormal",
            "turing": "LogNormal", "gen": "lognormal",
            "church": "lognormal",
            "bugs": "dlnorm", "jags": "dlnorm",
        },
    ),
    "StudentT": FamilyMeta(
        qvr_name="StudentT",
        distribution_class=td.StudentT,
        target_names={
            "stan": "student_t", "numpyro": "StudentT", "pyro": "StudentT",
            "pymc": "StudentT", "edward2": "StudentT",
            "turing": "TDist",
            "church": "student-t", "webppl": "StudentT",
            "bugs": "dt", "jags": "dt",
        },
    ),
    "Exponential": FamilyMeta(
        qvr_name="Exponential",
        distribution_class=td.Exponential,
        target_names={
            "stan": "exponential", "numpyro": "Exponential",
            "pyro": "Exponential", "pymc": "Exponential",
            "edward2": "Exponential", "turing": "Exponential",
            "gen": "exponential", "church": "exponential",
            "webppl": "Exponential",
            "bugs": "dexp", "jags": "dexp",
        },
    ),
    "Gamma": FamilyMeta(
        qvr_name="Gamma",
        distribution_class=td.Gamma,
        target_names={
            "stan": "gamma", "numpyro": "Gamma", "pyro": "Gamma",
            "pymc": "Gamma", "edward2": "Gamma",
            "turing": "Gamma", "gen": "gamma",
            "church": "gamma", "webppl": "Gamma",
            "bugs": "dgamma", "jags": "dgamma",
        },
    ),
    "Chi2": FamilyMeta(
        qvr_name="Chi2",
        distribution_class=td.Chi2,
        target_names={
            "stan": "chi_square", "numpyro": "Chi2", "pyro": "Chi2",
            "pymc": "ChiSquared", "edward2": "Chi2",
            "bugs": "dchisqr", "jags": "dchisqr",
        },
    ),
    "HalfCauchy": FamilyMeta(
        qvr_name="HalfCauchy",
        distribution_class=td.HalfCauchy,
        target_names={
            "stan": "cauchy", "numpyro": "HalfCauchy", "pyro": "HalfCauchy",
            "pymc": "HalfCauchy", "edward2": "HalfCauchy",
            "turing": "truncated",
            "bugs": "dt", "jags": "dt",
        },
    ),
    "HalfNormal": FamilyMeta(
        qvr_name="HalfNormal",
        distribution_class=td.HalfNormal,
        target_names={
            "stan": "normal", "numpyro": "HalfNormal", "pyro": "HalfNormal",
            "pymc": "HalfNormal", "edward2": "HalfNormal",
            "turing": "truncated", "webppl": "Gaussian",
            "bugs": "dnorm", "jags": "dnorm",
        },
    ),
    "InverseGamma": FamilyMeta(
        qvr_name="InverseGamma",
        distribution_class=td.InverseGamma,
        target_names={
            "stan": "inv_gamma", "numpyro": "InverseGamma",
            "pyro": "InverseGamma", "pymc": "InverseGamma",
            "edward2": "InverseGamma", "turing": "InverseGamma",
        },
    ),
    "Weibull": FamilyMeta(
        qvr_name="Weibull",
        distribution_class=td.Weibull,
        target_names={
            "stan": "weibull", "numpyro": "Weibull", "pyro": "Weibull",
            "pymc": "Weibull", "edward2": "Weibull",
            "turing": "Weibull",
            "bugs": "dweib", "jags": "dweib",
        },
    ),
    "Pareto": FamilyMeta(
        qvr_name="Pareto",
        distribution_class=td.Pareto,
        target_names={
            "stan": "pareto", "numpyro": "Pareto", "pyro": "Pareto",
            "pymc": "Pareto", "edward2": "Pareto",
            "turing": "Pareto", "church": "pareto",
            "bugs": "dpar", "jags": "dpar",
        },
    ),
    "Kumaraswamy": FamilyMeta(
        qvr_name="Kumaraswamy",
        distribution_class=td.Kumaraswamy,
        target_names={
            "numpyro": "Kumaraswamy", "pyro": "Kumaraswamy",
            "pymc": "Kumaraswamy", "edward2": "Kumaraswamy",
        },
    ),
    "ContinuousBernoulli": FamilyMeta(
        qvr_name="ContinuousBernoulli",
        distribution_class=td.ContinuousBernoulli,
        target_names={
            "stan": "continuous_bernoulli", "numpyro": "ContinuousBernoulli",
            "pyro": "ContinuousBernoulli", "edward2": "ContinuousBernoulli",
        },
    ),
    "FisherSnedecor": FamilyMeta(
        qvr_name="FisherSnedecor",
        distribution_class=td.FisherSnedecor,
        target_names={
            "numpyro": "FisherSnedecor", "pyro": "FisherSnedecor",
        },
    ),
    "Uniform": FamilyMeta(
        qvr_name="Uniform",
        distribution_class=td.Uniform,
        target_names={
            "stan": "uniform", "numpyro": "Uniform", "pyro": "Uniform",
            "pymc": "Uniform", "edward2": "Uniform",
            "turing": "Uniform", "gen": "uniform",
            "church": "uniform", "webppl": "Uniform",
            "bugs": "dunif", "jags": "dunif",
        },
    ),
    # ----- continuous multivariate -----
    "MultivariateNormal": FamilyMeta(
        qvr_name="MultivariateNormal",
        distribution_class=td.MultivariateNormal,
        target_names={
            "stan": "multi_normal", "numpyro": "MultivariateNormal",
            "pyro": "MultivariateNormal", "pymc": "MvNormal",
            "edward2": "MultivariateNormalFullCovariance",
            "turing": "MvNormal", "gen": "mvnormal",
            "church": "multivariate-gaussian",
            "webppl": "MultivariateGaussian",
            "bugs": "dmnorm", "jags": "dmnorm",
        },
    ),
    "LowRankMVN": FamilyMeta(
        qvr_name="LowRankMVN",
        distribution_class=td.LowRankMultivariateNormal,
        target_names={
            "numpyro": "LowRankMultivariateNormal",
            "pyro": "LowRankMultivariateNormal",
        },
    ),
    # ----- continuous discrete-relaxation -----
    "RelaxedBernoulli": FamilyMeta(
        qvr_name="RelaxedBernoulli",
        distribution_class=td.RelaxedBernoulli,
        target_names={
            "numpyro": "RelaxedBernoulli", "pyro": "RelaxedBernoulli",
        },
    ),
    "RelaxedOneHotCategorical": FamilyMeta(
        qvr_name="RelaxedOneHotCategorical",
        distribution_class=td.RelaxedOneHotCategorical,
        target_names={
            "numpyro": "RelaxedOneHotCategorical",
            "pyro": "RelaxedOneHotCategorical",
        },
    ),
    # ----- matrix-valued -----
    "Wishart": FamilyMeta(
        qvr_name="Wishart",
        distribution_class=td.Wishart,
        target_names={
            "numpyro": "Wishart", "pyro": "Wishart",
            "pymc": "Wishart", "edward2": "Wishart",
            "bugs": "dwish", "jags": "dwish",
        },
    ),
    "InverseWishart": FamilyMeta(
        qvr_name="InverseWishart",
        distribution_class=_InverseWishart,
        target_names={
            "numpyro": "InverseWishart", "pyro": "InverseWishart",
        },
    ),
    "MatrixNormal": FamilyMeta(
        qvr_name="MatrixNormal",
        distribution_class=_MatrixNormal,
        target_names={
            "pymc": "MatrixNormal",
            "edward2": "MatrixNormalLinearOperator",
        },
    ),
    "GP": FamilyMeta(
        qvr_name="GP",
        distribution_class=_GaussianProcess,
        target_names={
            "edward2": "GaussianProcess",
        },
    ),
    "Horseshoe": FamilyMeta(
        qvr_name="Horseshoe",
        distribution_class=_Horseshoe,
        target_names={
            "stan": "normal", "numpyro": "Normal", "pyro": "Normal",
            "pymc": "Normal", "edward2": "Normal",
        },
    ),
    # ----- discrete -----
    "Bernoulli": FamilyMeta(
        qvr_name="Bernoulli",
        distribution_class=td.Bernoulli,
        target_names={
            "stan": "bernoulli", "numpyro": "Bernoulli", "pyro": "Bernoulli",
            "pymc": "Bernoulli", "edward2": "Bernoulli",
            "turing": "Bernoulli", "gen": "bernoulli",
            "church": "flip",
            "bugs": "dbern", "jags": "dbern",
        },
    ),
    "Categorical": FamilyMeta(
        qvr_name="Categorical",
        distribution_class=td.Categorical,
        target_names={
            "stan": "categorical", "numpyro": "Categorical",
            "pyro": "Categorical", "pymc": "Categorical",
            "edward2": "Categorical", "turing": "Categorical",
            "gen": "categorical", "church": "categorical",
            "webppl": "Categorical",
            "bugs": "dcat", "jags": "dcat",
        },
    ),
    # ----- Phase A: implementations not yet exposed via DSL -----
    "Poisson": FamilyMeta(
        qvr_name="Poisson",
        distribution_class=ConditionalPoisson,
        target_names={
            "stan": "poisson", "numpyro": "Poisson", "pyro": "Poisson",
            "pymc": "Poisson", "edward2": "Poisson",
            "turing": "Poisson", "gen": "poisson",
            "bugs": "dpois", "jags": "dpois",
        },
    ),
    "NegativeBinomial": FamilyMeta(
        qvr_name="NegativeBinomial",
        distribution_class=ConditionalNegativeBinomial,
        target_names={
            "stan": "neg_binomial_2", "numpyro": "NegativeBinomial2",
            "pyro": "NegativeBinomial", "pymc": "NegativeBinomial",
            "edward2": "NegativeBinomial",
            "bugs": "dnegbin", "jags": "dnegbin",
        },
    ),
    "Geometric": FamilyMeta(
        qvr_name="Geometric",
        distribution_class=ConditionalGeometric,
        target_names={
            "numpyro": "Geometric", "pyro": "Geometric",
            "pymc": "Geometric", "edward2": "Geometric",
            "gen": "geometric",
        },
    ),
    "Binomial": FamilyMeta(
        qvr_name="Binomial",
        distribution_class=ConditionalBinomial,
        target_names={
            "stan": "binomial", "numpyro": "Binomial", "pyro": "Binomial",
            "pymc": "Binomial", "edward2": "Binomial",
            "bugs": "dbin", "jags": "dbin",
        },
    ),
    "VonMises": FamilyMeta(
        qvr_name="VonMises",
        distribution_class=ConditionalVonMises,
        target_names={
            "stan": "von_mises", "numpyro": "VonMises", "pyro": "VonMises",
        },
    ),
    "LogisticNormal": FamilyMeta(
        qvr_name="LogisticNormal",
        distribution_class=ConditionalLogisticNormal,
        target_names={
            "numpyro": "LogisticNormal", "pyro": "LogisticNormal",
        },
    ),
    "OneHotCategorical": FamilyMeta(
        qvr_name="OneHotCategorical",
        distribution_class=ConditionalOneHotCategorical,
        target_names={
            "numpyro": "OneHotCategorical", "pyro": "OneHotCategorical",
            "edward2": "OneHotCategorical",
        },
    ),
    "LKJCholesky": FamilyMeta(
        qvr_name="LKJCholesky",
        distribution_class=ConditionalLKJCholesky,
        target_names={
            "stan": "lkj_corr_cholesky", "numpyro": "LKJCholesky",
            "pyro": "LKJCorrCholesky", "pymc": "LKJCholeskyCov",
            "edward2": "LKJ",
        },
    ),
    "Mixture": FamilyMeta(
        qvr_name="Mixture",
        distribution_class=ConditionalMixture,
        target_names={
            "numpyro": "MixtureSameFamily", "pyro": "MixtureSameFamily",
            "pymc": "Mixture",
        },
    ),
    "Independent": FamilyMeta(
        qvr_name="Independent",
        distribution_class=ConditionalIndependent,
        target_names={
            "numpyro": "Independent", "pyro": "Independent",
        },
    ),
    "Transformed": FamilyMeta(
        qvr_name="Transformed",
        distribution_class=ConditionalTransformed,
        target_names={
            "numpyro": "TransformedDistribution",
            "pyro": "TransformedDistribution",
        },
    ),
    "Truncated": FamilyMeta(
        qvr_name="Truncated",
        distribution_class=Truncated,
        target_names={
            "pymc": "Truncated",
        },
    ),
    "LKJCorrelationFactor": FamilyMeta(
        qvr_name="LKJCorrelationFactor",
        distribution_class=LKJCorrelationFactor,
        target_names={
            "pymc": "LKJCorr",
        },
    ),
    # ----- Phase B tier 1 -----
    "BetaBinomial": FamilyMeta(
        qvr_name="BetaBinomial",
        distribution_class=_BetaBinomial,
        target_names={
            "stan": "beta_binomial", "numpyro": "BetaBinomial",
            "pyro": "BetaBinomial", "pymc": "BetaBinomial",
            "bugs": "dbetabin",
        },
    ),
    "OrderedLogistic": FamilyMeta(
        qvr_name="OrderedLogistic",
        distribution_class=_OrderedLogistic,
        target_names={
            "stan": "ordered_logistic", "numpyro": "OrderedLogistic",
            "pyro": "OrderedLogistic", "pymc": "OrderedLogistic",
        },
    ),
    "OrderedProbit": FamilyMeta(
        qvr_name="OrderedProbit",
        distribution_class=_OrderedProbit,
        target_names={
            "stan": "ordered_probit", "numpyro": "OrderedProbit",
            "pyro": "OrderedProbit", "pymc": "OrderedProbit",
        },
    ),
    "Logistic": FamilyMeta(
        qvr_name="Logistic",
        distribution_class=_Logistic,
        target_names={
            "stan": "logistic", "numpyro": "Logistic",
            "pyro": "Logistic", "pymc": "Logistic",
            "bugs": "dlogis", "jags": "dlogis",
        },
    ),
    "HalfStudentT": FamilyMeta(
        qvr_name="HalfStudentT",
        distribution_class=_HalfStudentT,
        target_names={
            "stan": "student_t", "numpyro": "HalfStudentT",
            "pyro": "HalfStudentT", "pymc": "HalfStudentT",
        },
    ),
}


# Families whose call sites are always finite-enumerable over the
# latent regardless of arg shape. Marginalize-eligibility for any
# other family folds in arg-form inspection (see Binomial below).
_ALWAYS_ENUMERABLE: frozenset[str] = frozenset(
    {"Bernoulli", "Categorical", "OrderedLogistic", "OrderedProbit"}
)


def finite_enumerable_at_call_site(
    family_meta: FamilyMeta,
    args: tuple[IRArg, ...],
) -> bool:
    """True iff the call site has finite enumerable support over the latent.

    Returns True for Bernoulli, Categorical, OrderedLogistic,
    OrderedProbit unconditionally. For Binomial returns True only
    when `args[0]` (`total_count`) is a literal
    [`IRArgNumber`][quivers.transpile.ir.IRArgNumber]; False when it
    is a reference.
    """
    name = family_meta.qvr_name
    if name in _ALWAYS_ENUMERABLE:
        return True
    if name == "Binomial":
        return bool(args) and isinstance(args[0], IRArgNumber)
    return False


__all__ = [
    "FAMILY_META",
    "FamilyMeta",
    "finite_enumerable_at_call_site",
]
