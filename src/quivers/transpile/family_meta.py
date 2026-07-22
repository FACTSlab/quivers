"""Static transpile-only metadata for the registered distribution families.

The torch distribution class supplies `arg_constraints`, `.support`,
`event_shape`, `batch_shape`, and the natural parameterisation;
[`FamilyMeta`][quivers.transpile.family_meta.FamilyMeta] carries
only the transpile-specific facts that torch doesn't publish:

* `qvr_name`: the DSL-facing family name.
* `distribution_class`: the underlying
  [`torch.distributions.Distribution`][torch.distributions.Distribution]
  subclass. The transpile layer reads `arg_constraints`, `support`,
  and `event_shape` from this class.
* `quivers_class`: the
  [`ContinuousMorphism`][quivers.continuous.morphisms.ContinuousMorphism]
  subclass used by the inference layer for runtime ``log_prob`` and
  ``rsample`` evaluation. ``None`` for wrapper or aggregate families
  whose runtime morphism is built from a referenced inner morphism
  rather than constructed directly.
* `target_names`: per-backend distribution-name mapping. The single
  source of truth for backend-to-distribution-name resolution. No
  per-renderer `_FAMILIES` dict.
* `arg_aliases`: per-backend per-arg renames. Most families have
  empty `arg_aliases`. Renderers that apply parameterisation-converting
  arithmetic (BUGS Normal mean/scale to mean/precision) key the
  arithmetic on the alias's target name.

Families without a direct
[`torch.distributions`][torch.distributions] class (`BetaBinomial`,
`OrderedLogistic`, `OrderedProbit`, `Logistic`, `HalfStudentT`,
plus the wrappers `Truncated`, `Mixture`, `Independent`,
`Transformed`, `LKJCorrelationFactor`, `Horseshoe`, `GP`,
`InverseWishart`, `MatrixNormal`, `LogitNormal`,
`TruncatedNormal`) get minimal `Distribution` subclasses defined
in this module, carrying the right `arg_constraints` and `support`
so the lower pipeline can introspect them.

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
    ConditionalBernoulli,
    ConditionalBeta,
    ConditionalBetaBinomial,
    ConditionalBinomial,
    ConditionalCategorical,
    ConditionalCauchy,
    ConditionalChi2,
    ConditionalContinuousBernoulli,
    ConditionalDirichlet,
    ConditionalExponential,
    ConditionalFisherSnedecor,
    ConditionalGamma,
    ConditionalGaussianProcess,
    ConditionalGeometric,
    ConditionalGumbel,
    ConditionalHalfCauchy,
    ConditionalHalfNormal,
    ConditionalHalfStudentT,
    ConditionalHorseshoe,
    ConditionalIndependent,
    ConditionalInverseGamma,
    ConditionalInverseWishart,
    ConditionalKumaraswamy,
    ConditionalLKJCholesky,
    ConditionalLaplace,
    ConditionalLogNormal,
    ConditionalLogistic,
    ConditionalLogisticNormal,
    ConditionalLogitNormal,
    ConditionalLowRankMVN,
    ConditionalMatrixNormal,
    ConditionalMixture,
    ConditionalMultivariateNormal,
    ConditionalNegativeBinomial,
    ConditionalNormal,
    ConditionalOneHotCategorical,
    ConditionalPareto,
    ConditionalPoisson,
    ConditionalRelaxedBernoulli,
    ConditionalRelaxedOneHotCategorical,
    ConditionalStudentT,
    ConditionalTransformed,
    ConditionalTruncatedNormal,
    ConditionalUniform,
    ConditionalVonMises,
    ConditionalWeibull,
    ConditionalWishart,
    LKJCorrelationFactor,
    Truncated,
)
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.ordered import (
    ConditionalOrderedLogistic,
    ConditionalOrderedProbit,
)
from quivers.transpile.ir import (
    DomainGridAxis,
    IRArg,
    IRArgNumber,
    OverOrCodomainAxes,
    StructuredDataArg,
    StructuredKernelArg,
    StructuredSampleLowering,
    StructuredZeroVectorArg,
)


class FamilyMeta(dx.Model):
    """Static transpile-only metadata for one distribution family."""

    qvr_name: str
    distribution_class: type[Distribution] = dx.field(opaque=True)
    target_names: dict[str, str]
    arg_aliases: dict[str, dict[str, str]] = dx.field(default_factory=lambda: {})
    quivers_class: type[ContinuousMorphism] | None = dx.field(default=None, opaque=True)
    structured_lowering: StructuredSampleLowering | None = None


# ---------------------------------------------------------------------------
# Shim Distribution subclasses for families that torch does not ship.
# Their `__name__`, `arg_constraints`, and `support` are the
# transpile-layer contract that `Lower` and every renderer read. The
# runtime behaviour lives in the `quivers_class` `ContinuousMorphism`
# subclass paired in FAMILY_META.
# ---------------------------------------------------------------------------


class BetaBinomial(Distribution):
    """Beta-Binomial: `Binomial(n, p)` with `p ~ Beta(c1, c0)`.

    The marginal pmf is

        p(k; n, a, b) = C(n, k) * B(a + k, b + n - k) / B(a, b),

    where `B(.,.)` is the beta function. `log_prob` evaluates this
    in log space via `torch.lgamma`. `sample` draws p ~ Beta(a, b)
    then k ~ Binomial(n, p), matching the generative definition.
    """

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

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """``log p(value; n, a, b)`` via the closed-form Beta-Binomial pmf."""
        n = self.total_count.to(value.dtype)
        a = self.concentration1.to(value.dtype)
        b = self.concentration0.to(value.dtype)
        k = value.to(value.dtype)
        log_comb = (
            torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0)
        )
        log_beta_post = (
            torch.lgamma(a + k) + torch.lgamma(b + n - k) - torch.lgamma(a + b + n)
        )
        log_beta_prior = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        return log_comb + log_beta_post - log_beta_prior

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Two-stage draw: ``p ~ Beta(a, b)``; ``k ~ Binomial(n, p)``."""
        p = torch.distributions.Beta(self.concentration1, self.concentration0).sample(
            sample_shape
        )
        return torch.distributions.Binomial(
            total_count=self.total_count, probs=p
        ).sample()

    @property
    def mean(self) -> torch.Tensor:
        """``E[K] = n * a / (a + b)``."""
        return (
            self.total_count
            * self.concentration1
            / (self.concentration1 + self.concentration0)
        )

    @property
    def variance(self) -> torch.Tensor:
        """Closed-form Beta-Binomial variance."""
        n = self.total_count
        a = self.concentration1
        b = self.concentration0
        return n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1.0))


class OrderedLogistic(Distribution):
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


class OrderedProbit(Distribution):
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


class Logistic(Distribution):
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


class HalfStudentT(Distribution):
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
        # `scale_tril` is `(..., dim, dim)`; carry the trailing two
        # axes as the matrix event shape so transpile shape inference
        # picks up the square dimension.
        event_shape = scale_tril.shape[-2:]
        batch_shape = scale_tril.shape[:-2]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )


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


class _Truncated(Distribution):
    """Generic truncation wrapper: a base distribution restricted to
    `[low, high]`."""

    arg_constraints: dict[str, c.Constraint] = {
        "low": c.real,
        "high": c.real,
    }
    support: c.Constraint = c.real
    has_rsample: bool = False

    def __init__(
        self,
        base: Distribution,
        low: torch.Tensor,
        high: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.base = base
        self.low = low
        self.high = high
        super().__init__(validate_args=validate_args)


class _Mixture(Distribution):
    """A mixture: per-component weights plus a component distribution."""

    arg_constraints: dict[str, c.Constraint] = {
        "weights": c.simplex,
    }
    support: c.Constraint = c.real
    has_rsample: bool = False

    def __init__(
        self,
        weights: torch.Tensor,
        component: Distribution,
        validate_args: bool | None = None,
    ) -> None:
        self.weights = weights
        self.component = component
        super().__init__(validate_args=validate_args)


class _LKJCorrelationFactor(Distribution):
    """LKJ on the full correlation matrix (non-Cholesky form)."""

    arg_constraints: dict[str, c.Constraint] = {
        "concentration": c.positive,
    }
    support: c.Constraint = c.positive_definite
    has_rsample: bool = False

    def __init__(
        self,
        concentration: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.concentration = concentration
        super().__init__(validate_args=validate_args)


# ---------------------------------------------------------------------------
# FAMILY_META: the single transpile-time family registry.
# ---------------------------------------------------------------------------


FAMILY_META: dict[str, FamilyMeta] = {
    # ----- continuous scalar -----
    "Normal": FamilyMeta(
        qvr_name="Normal",
        distribution_class=td.Normal,
        quivers_class=ConditionalNormal,
        target_names={
            "stan": "normal",
            "numpyro": "Normal",
            "pyro": "Normal",
            "pymc": "Normal",
            "edward2": "Normal",
            "turing": "Normal",
            "gen": "normal",
            "church": "gaussian",
            "webppl": "Gaussian",
            "bugs": "dnorm",
            "jags": "dnorm",
        },
        arg_aliases={
            "bugs": {"scale": "tau"},
            "jags": {"scale": "tau"},
            "pymc": {"loc": "mu", "scale": "sigma"},
            "webppl": {"loc": "mu", "scale": "sigma"},
        },
    ),
    "LogitNormal": FamilyMeta(
        qvr_name="LogitNormal",
        distribution_class=_LogitNormal,
        quivers_class=ConditionalLogitNormal,
        target_names={
            "stan": "logit_normal",
            "numpyro": "LogitNormal",
            "pyro": "LogitNormal",
            "pymc": "LogitNormal",
            "edward2": "LogitNormal",
        },
        arg_aliases={
            "pymc": {"loc": "mu", "scale": "sigma"},
        },
    ),
    "Beta": FamilyMeta(
        qvr_name="Beta",
        distribution_class=td.Beta,
        quivers_class=ConditionalBeta,
        target_names={
            "stan": "beta",
            "numpyro": "Beta",
            "pyro": "Beta",
            "pymc": "Beta",
            "edward2": "Beta",
            "turing": "Beta",
            "gen": "beta",
            "church": "beta",
            "webppl": "Beta",
            "bugs": "dbeta",
            "jags": "dbeta",
        },
        arg_aliases={
            "pymc": {"concentration1": "alpha", "concentration0": "beta"},
            "webppl": {"concentration1": "a", "concentration0": "b"},
        },
    ),
    "TruncatedNormal": FamilyMeta(
        qvr_name="TruncatedNormal",
        distribution_class=_TruncatedNormal,
        quivers_class=ConditionalTruncatedNormal,
        # Stan / BUGS / JAGS use truncation-suffix syntax on the
        # sampling statement (`theta ~ normal(loc, scale) T[low, high]`
        # for Stan; `I(low, high)` for BUGS; `T(low, high)` for JAGS).
        # The `target_name` is the underlying base family; the per-
        # renderer sample path detects `family == "TruncatedNormal"`
        # and emits the suffix after the family call.
        #
        # Gen.jl has no built-in `truncated_normal`. The renderer
        # grafts a `Gen.Distribution` subclass plus a callable instance
        # named `truncated_normal` (defined in
        # [`runtime_gen.jl`][quivers.transpile.runtime_gen]) onto the
        # module above the `@gen function model`; the call site emits
        # as `truncated_normal(loc, scale, low, high)`.
        target_names={
            "numpyro": "TruncatedNormal",
            "pyro": "TruncatedNormal",
            "pymc": "TruncatedNormal",
            "edward2": "TruncatedNormal",
            "turing": "truncated",
            "gen": "truncated_normal",
            "stan": "normal",
            "bugs": "dnorm",
            "jags": "dnorm",
        },
        arg_aliases={
            "pymc": {
                "loc": "mu",
                "scale": "sigma",
                "low": "lower",
                "high": "upper",
            },
            "bugs": {"scale": "tau"},
            "jags": {"scale": "tau"},
        },
    ),
    "Dirichlet": FamilyMeta(
        qvr_name="Dirichlet",
        distribution_class=td.Dirichlet,
        quivers_class=ConditionalDirichlet,
        target_names={
            "stan": "dirichlet",
            "numpyro": "Dirichlet",
            "pyro": "Dirichlet",
            "pymc": "Dirichlet",
            "edward2": "Dirichlet",
            "turing": "Dirichlet",
            "gen": "dirichlet",
            "church": "dirichlet",
            "webppl": "Dirichlet",
            "bugs": "ddirch",
            "jags": "ddirich",
        },
        arg_aliases={
            "pymc": {"concentration": "a"},
            "webppl": {"concentration": "alpha"},
        },
    ),
    "Cauchy": FamilyMeta(
        qvr_name="Cauchy",
        distribution_class=td.Cauchy,
        quivers_class=ConditionalCauchy,
        target_names={
            "stan": "cauchy",
            "numpyro": "Cauchy",
            "pyro": "Cauchy",
            "pymc": "Cauchy",
            "edward2": "Cauchy",
            "turing": "Cauchy",
            "gen": "cauchy",
            "church": "cauchy",
            "webppl": "Cauchy",
            "bugs": "dt",
            "jags": "dt",
        },
        arg_aliases={
            # JAGS / BUGS `dt(mu, tau, k)` parameterise by precision
            # and degrees of freedom; the renderer's `_APPEND_DF_ONE`
            # injection appends ``k=1`` after this alias map renames
            # ``scale -> tau`` (triggering the inv_square arithmetic
            # transform, so the emitted tau is ``1/(scale*scale)``).
            "bugs": {"scale": "tau"},
            "jags": {"scale": "tau"},
            "pymc": {"loc": "alpha", "scale": "beta"},
            "webppl": {"loc": "location"},
        },
    ),
    "Laplace": FamilyMeta(
        qvr_name="Laplace",
        distribution_class=td.Laplace,
        quivers_class=ConditionalLaplace,
        target_names={
            "stan": "double_exponential",
            "numpyro": "Laplace",
            "pyro": "Laplace",
            "pymc": "Laplace",
            "edward2": "Laplace",
            "turing": "Laplace",
            "gen": "laplace",
            "webppl": "Laplace",
            "bugs": "ddexp",
            "jags": "ddexp",
        },
        arg_aliases={
            "bugs": {"scale": "tau"},
            "jags": {"scale": "tau"},
            "pymc": {"loc": "mu", "scale": "b"},
            "webppl": {"loc": "location", "scale": "scale"},
        },
    ),
    "Gumbel": FamilyMeta(
        qvr_name="Gumbel",
        distribution_class=td.Gumbel,
        quivers_class=ConditionalGumbel,
        target_names={
            "stan": "gumbel",
            "numpyro": "Gumbel",
            "pyro": "Gumbel",
            "pymc": "Gumbel",
            "edward2": "Gumbel",
            "turing": "Gumbel",
        },
        arg_aliases={
            "pymc": {"loc": "mu", "scale": "beta"},
        },
    ),
    "LogNormal": FamilyMeta(
        qvr_name="LogNormal",
        distribution_class=td.LogNormal,
        quivers_class=ConditionalLogNormal,
        target_names={
            "stan": "lognormal",
            "numpyro": "LogNormal",
            "pyro": "LogNormal",
            "pymc": "LogNormal",
            "edward2": "LogNormal",
            "turing": "LogNormal",
            "gen": "lognormal",
            "church": "lognormal",
            "webppl": "LogNormal",
            "bugs": "dlnorm",
            "jags": "dlnorm",
        },
        arg_aliases={
            "pymc": {"loc": "mu", "scale": "sigma"},
        },
    ),
    "StudentT": FamilyMeta(
        qvr_name="StudentT",
        distribution_class=td.StudentT,
        quivers_class=ConditionalStudentT,
        target_names={
            "stan": "student_t",
            "numpyro": "StudentT",
            "pyro": "StudentT",
            "pymc": "StudentT",
            "edward2": "StudentT",
            "turing": "TDist",
            "church": "student-t",
            "webppl": "StudentT",
            "bugs": "dt",
            "jags": "dt",
        },
        arg_aliases={
            "pymc": {"df": "nu", "loc": "mu", "scale": "sigma"},
        },
    ),
    "Exponential": FamilyMeta(
        qvr_name="Exponential",
        distribution_class=td.Exponential,
        quivers_class=ConditionalExponential,
        target_names={
            "stan": "exponential",
            "numpyro": "Exponential",
            "pyro": "Exponential",
            "pymc": "Exponential",
            "edward2": "Exponential",
            "turing": "Exponential",
            "gen": "exponential",
            "church": "exponential",
            "webppl": "Exponential",
            "bugs": "dexp",
            "jags": "dexp",
        },
        arg_aliases={
            "pymc": {"rate": "lam"},
            "webppl": {"rate": "a"},
        },
    ),
    "Gamma": FamilyMeta(
        qvr_name="Gamma",
        distribution_class=td.Gamma,
        quivers_class=ConditionalGamma,
        target_names={
            "stan": "gamma",
            "numpyro": "Gamma",
            "pyro": "Gamma",
            "pymc": "Gamma",
            "edward2": "Gamma",
            "turing": "Gamma",
            "gen": "gamma",
            "church": "gamma",
            "webppl": "Gamma",
            "bugs": "dgamma",
            "jags": "dgamma",
        },
        arg_aliases={
            "pymc": {"concentration": "alpha", "rate": "beta"},
            "webppl": {"concentration": "shape", "rate": "scale"},
        },
    ),
    "Chi2": FamilyMeta(
        qvr_name="Chi2",
        distribution_class=td.Chi2,
        quivers_class=ConditionalChi2,
        target_names={
            "stan": "chi_square",
            "numpyro": "Chi2",
            "pyro": "Chi2",
            "pymc": "ChiSquared",
            "edward2": "Chi2",
            "turing": "Chisq",
            "bugs": "dchisqr",
            "jags": "dchisqr",
        },
        arg_aliases={
            "pymc": {"df": "nu"},
        },
    ),
    "HalfCauchy": FamilyMeta(
        qvr_name="HalfCauchy",
        distribution_class=td.HalfCauchy,
        quivers_class=ConditionalHalfCauchy,
        target_names={
            "stan": "cauchy",
            "numpyro": "HalfCauchy",
            "pyro": "HalfCauchy",
            "pymc": "HalfCauchy",
            "edward2": "HalfCauchy",
            "turing": "truncated",
            "gen": "cauchy",
            "church": "cauchy",
            "webppl": "Cauchy",
            "bugs": "dt",
            "jags": "dt",
        },
        arg_aliases={
            "pymc": {"scale": "beta"},
            # WebPPL renders HalfCauchy as ``Cauchy({location: 0,
            # scale: scale})``; the renderer prepends a ``loc=0``
            # argument before this alias map is consulted, so the
            # keyword for the injected zero is renamed `loc ->
            # location` here.
            "webppl": {"scale": "scale", "loc": "location"},
            # JAGS / BUGS `dt(mu, tau, k)` parameterise by precision
            # and degrees of freedom; the renderer prepends ``loc=0``
            # and the `_APPEND_DF_ONE` injection appends ``k=1`` after
            # this alias map renames ``scale -> tau`` (triggering the
            # inv_square arithmetic transform). A latent draw carries
            # the one-sided truncation suffix that restricts the
            # symmetric ``dt`` back to the non-negative reals.
            "bugs": {"scale": "tau"},
            "jags": {"scale": "tau"},
        },
    ),
    "HalfNormal": FamilyMeta(
        qvr_name="HalfNormal",
        distribution_class=td.HalfNormal,
        quivers_class=ConditionalHalfNormal,
        target_names={
            "stan": "normal",
            "numpyro": "HalfNormal",
            "pyro": "HalfNormal",
            "pymc": "HalfNormal",
            "edward2": "HalfNormal",
            "turing": "truncated",
            "gen": "normal",
            "church": "gaussian",
            "webppl": "Gaussian",
            "bugs": "dnorm",
            "jags": "dnorm",
        },
        arg_aliases={
            "pymc": {"scale": "sigma"},
            # WebPPL renders HalfNormal as ``Gaussian({mu: 0, sigma:
            # scale})``; the renderer prepends a `loc=0` argument
            # before this alias map is consulted, so the keyword for
            # the injected zero is renamed `loc -> mu` here.
            "webppl": {"scale": "sigma", "loc": "mu"},
            # JAGS / BUGS `dnorm(mu, tau)` parameterise by precision;
            # the renderer prepends ``loc=0`` and this alias renames
            # ``scale -> tau`` (triggering the inv_square arithmetic
            # transform). A latent draw carries the one-sided
            # truncation suffix that restricts the symmetric ``dnorm``
            # back to the non-negative reals.
            "bugs": {"scale": "tau"},
            "jags": {"scale": "tau"},
        },
    ),
    "InverseGamma": FamilyMeta(
        qvr_name="InverseGamma",
        distribution_class=td.InverseGamma,
        quivers_class=ConditionalInverseGamma,
        target_names={
            "stan": "inv_gamma",
            "numpyro": "InverseGamma",
            "pyro": "InverseGamma",
            "pymc": "InverseGamma",
            "edward2": "InverseGamma",
            "turing": "InverseGamma",
        },
        arg_aliases={
            "pymc": {"concentration": "alpha", "rate": "beta"},
        },
    ),
    "Weibull": FamilyMeta(
        qvr_name="Weibull",
        distribution_class=td.Weibull,
        quivers_class=ConditionalWeibull,
        target_names={
            "stan": "weibull",
            "numpyro": "Weibull",
            "pyro": "Weibull",
            "pymc": "Weibull",
            "edward2": "Weibull",
            "turing": "Weibull",
            "gen": "weibull",
            "church": "weibull",
            "webppl": "Weibull",
            "bugs": "dweib",
            "jags": "dweib",
        },
        arg_aliases={
            "pymc": {"concentration": "alpha", "scale": "beta"},
        },
    ),
    "Pareto": FamilyMeta(
        qvr_name="Pareto",
        distribution_class=td.Pareto,
        quivers_class=ConditionalPareto,
        target_names={
            "stan": "pareto",
            "numpyro": "Pareto",
            "pyro": "Pareto",
            "pymc": "Pareto",
            "edward2": "Pareto",
            "turing": "Pareto",
            "church": "pareto",
            "bugs": "dpar",
            "jags": "dpar",
        },
        arg_aliases={
            "pymc": {"alpha": "alpha", "scale": "m"},
        },
    ),
    "Kumaraswamy": FamilyMeta(
        qvr_name="Kumaraswamy",
        distribution_class=td.Kumaraswamy,
        quivers_class=ConditionalKumaraswamy,
        target_names={
            "stan": "kumaraswamy",
            "numpyro": "Kumaraswamy",
            "pyro": "Kumaraswamy",
            "pymc": "Kumaraswamy",
            "edward2": "Kumaraswamy",
            "gen": "kumaraswamy",
            "turing": "Kumaraswamy",
            "webppl": "Kumaraswamy",
        },
        arg_aliases={
            "pymc": {"concentration1": "a", "concentration0": "b"},
        },
    ),
    "ContinuousBernoulli": FamilyMeta(
        qvr_name="ContinuousBernoulli",
        distribution_class=td.ContinuousBernoulli,
        quivers_class=ConditionalContinuousBernoulli,
        target_names={
            "stan": "continuous_bernoulli",
            "numpyro": "ContinuousBernoulli",
            "pyro": "ContinuousBernoulli",
            "edward2": "ContinuousBernoulli",
            "pymc": "ContinuousBernoulli",
            "turing": "ContinuousBernoulli",
            "gen": "continuous_bernoulli",
            "webppl": "ContinuousBernoulli",
        },
    ),
    "FisherSnedecor": FamilyMeta(
        qvr_name="FisherSnedecor",
        distribution_class=td.FisherSnedecor,
        quivers_class=ConditionalFisherSnedecor,
        target_names={
            "numpyro": "FisherSnedecor",
            "pyro": "FisherSnedecor",
            "turing": "FDist",
            "jags": "df",
        },
        arg_aliases={
            "jags": {"df1": "n", "df2": "m"},
        },
    ),
    "Uniform": FamilyMeta(
        qvr_name="Uniform",
        distribution_class=td.Uniform,
        quivers_class=ConditionalUniform,
        target_names={
            "stan": "uniform",
            "numpyro": "Uniform",
            "pyro": "Uniform",
            "pymc": "Uniform",
            "edward2": "Uniform",
            "turing": "Uniform",
            "gen": "uniform",
            "church": "uniform",
            "webppl": "Uniform",
            "bugs": "dunif",
            "jags": "dunif",
        },
        arg_aliases={
            "pymc": {"low": "lower", "high": "upper"},
            "webppl": {"low": "a", "high": "b"},
        },
    ),
    # ----- continuous multivariate -----
    "MultivariateNormal": FamilyMeta(
        qvr_name="MultivariateNormal",
        distribution_class=td.MultivariateNormal,
        quivers_class=ConditionalMultivariateNormal,
        target_names={
            "stan": "multi_normal",
            "numpyro": "MultivariateNormal",
            "pyro": "MultivariateNormal",
            "pymc": "MvNormal",
            "edward2": "MultivariateNormalFullCovariance",
            "turing": "MvNormal",
            "gen": "mvnormal",
            "church": "multivariate-gaussian",
            "webppl": "MultivariateGaussian",
            "bugs": "dmnorm",
            "jags": "dmnorm",
        },
        arg_aliases={
            "pymc": {"loc": "mu", "covariance_matrix": "cov"},
            "webppl": {"loc": "mu", "covariance_matrix": "cov"},
        },
        structured_lowering=StructuredSampleLowering(
            args=(
                StructuredDataArg(
                    arg_name="loc",
                    axis_indices=(0,),
                    constraint_kind="real_vector",
                ),
                StructuredDataArg(
                    arg_name="covariance_matrix",
                    axis_indices=(0, 0),
                    constraint_kind="positive_definite",
                ),
            ),
            event_axis_source=OverOrCodomainAxes(axis_count=1),
            sample_constraint_kind="real_vector",
        ),
    ),
    "LowRankMVN": FamilyMeta(
        qvr_name="LowRankMVN",
        distribution_class=td.LowRankMultivariateNormal,
        quivers_class=ConditionalLowRankMVN,
        target_names={
            "numpyro": "LowRankMultivariateNormal",
            "pyro": "LowRankMultivariateNormal",
        },
        arg_aliases={
            "pymc": {
                "loc": "mu",
                "cov_factor": "W",
                "cov_diag": "diag",
            },
        },
    ),
    # ----- continuous discrete-relaxation -----
    "RelaxedBernoulli": FamilyMeta(
        qvr_name="RelaxedBernoulli",
        distribution_class=td.RelaxedBernoulli,
        quivers_class=ConditionalRelaxedBernoulli,
        target_names={
            "numpyro": "RelaxedBernoulli",
            "pyro": "RelaxedBernoulli",
            "edward2": "RelaxedBernoulli",
        },
    ),
    "RelaxedOneHotCategorical": FamilyMeta(
        qvr_name="RelaxedOneHotCategorical",
        distribution_class=td.RelaxedOneHotCategorical,
        quivers_class=ConditionalRelaxedOneHotCategorical,
        target_names={
            "numpyro": "RelaxedOneHotCategorical",
            "pyro": "RelaxedOneHotCategorical",
            "edward2": "RelaxedOneHotCategorical",
        },
    ),
    # ----- matrix-valued -----
    "Wishart": FamilyMeta(
        qvr_name="Wishart",
        distribution_class=td.Wishart,
        quivers_class=ConditionalWishart,
        target_names={
            "stan": "wishart",
            "numpyro": "Wishart",
            "pyro": "Wishart",
            "pymc": "Wishart",
            "edward2": "Wishart",
            "turing": "Wishart",
            "bugs": "dwish",
            "jags": "dwish",
        },
        arg_aliases={
            "pymc": {"df": "nu", "covariance_matrix": "V"},
        },
    ),
    "InverseWishart": FamilyMeta(
        qvr_name="InverseWishart",
        distribution_class=_InverseWishart,
        quivers_class=ConditionalInverseWishart,
        target_names={
            "stan": "inv_wishart",
            "numpyro": "InverseWishart",
            "pyro": "InverseWishart",
            "turing": "InverseWishart",
        },
    ),
    "MatrixNormal": FamilyMeta(
        qvr_name="MatrixNormal",
        distribution_class=_MatrixNormal,
        quivers_class=ConditionalMatrixNormal,
        target_names={
            "stan": "matrix_normal",
            "numpyro": "MatrixNormal",
            "pyro": "MatrixNormal",
            "pymc": "MatrixNormal",
            "edward2": "MatrixNormalLinearOperator",
            "turing": "MatrixNormal",
            "gen": "matrix_normal",
            "webppl": "MatrixNormal",
            "church": "matrix-normal",
        },
        arg_aliases={
            "pymc": {
                "loc": "mu",
                "row_covariance": "rowcov",
                "col_covariance": "colcov",
            },
        },
        structured_lowering=StructuredSampleLowering(
            args=(
                StructuredDataArg(
                    arg_name="loc",
                    axis_indices=(0, 1),
                    constraint_kind="real_matrix",
                ),
                StructuredDataArg(
                    arg_name="row_covariance",
                    axis_indices=(0, 0),
                    constraint_kind="positive_definite",
                ),
                StructuredDataArg(
                    arg_name="col_covariance",
                    axis_indices=(1, 1),
                    constraint_kind="positive_definite",
                ),
            ),
            event_axis_source=OverOrCodomainAxes(axis_count=2),
            sample_constraint_kind="real_matrix",
        ),
    ),
    "GP": FamilyMeta(
        qvr_name="GP",
        distribution_class=_GaussianProcess,
        quivers_class=ConditionalGaussianProcess,
        target_names={
            "edward2": "MultivariateNormalFullCovariance",
            "stan": "multi_normal",
            "numpyro": "MultivariateNormal",
            "pyro": "MultivariateNormal",
            "pymc": "MvNormal",
            "turing": "MvNormal",
            "gen": "mvnormal",
            "webppl": "MultivariateGaussian",
            "church": "multivariate-gaussian",
            "bugs": "dmnorm",
            "jags": "dmnorm",
        },
        structured_lowering=StructuredSampleLowering(
            args=(
                StructuredZeroVectorArg(arg_name="mean"),
                StructuredKernelArg(
                    arg_name="covariance_matrix",
                    x_input_name="x",
                ),
            ),
            event_axis_source=DomainGridAxis(),
            sample_constraint_kind="real_vector",
            always_apply=True,
        ),
    ),
    "Horseshoe": FamilyMeta(
        qvr_name="Horseshoe",
        distribution_class=_Horseshoe,
        quivers_class=ConditionalHorseshoe,
        target_names={
            "stan": "normal",
            "numpyro": "Normal",
            "pyro": "Normal",
            "pymc": "Normal",
            "edward2": "Normal",
            "turing": "Normal",
            "gen": "normal",
            "church": "gaussian",
            "webppl": "Gaussian",
            "bugs": "dnorm",
            "jags": "dnorm",
        },
    ),
    # ----- discrete -----
    "Bernoulli": FamilyMeta(
        qvr_name="Bernoulli",
        distribution_class=td.Bernoulli,
        quivers_class=ConditionalBernoulli,
        target_names={
            "stan": "bernoulli",
            "numpyro": "Bernoulli",
            "pyro": "Bernoulli",
            "pymc": "Bernoulli",
            "edward2": "Bernoulli",
            "turing": "Bernoulli",
            "gen": "bernoulli",
            "church": "flip",
            "webppl": "Bernoulli",
            "bugs": "dbern",
            "jags": "dbern",
        },
        arg_aliases={
            "pymc": {"probs": "p"},
            "webppl": {"probs": "p"},
        },
    ),
    "Categorical": FamilyMeta(
        qvr_name="Categorical",
        distribution_class=td.Categorical,
        quivers_class=ConditionalCategorical,
        target_names={
            "stan": "categorical",
            "numpyro": "Categorical",
            "pyro": "Categorical",
            "pymc": "Categorical",
            "edward2": "Categorical",
            "turing": "Categorical",
            "gen": "categorical",
            "church": "categorical",
            "webppl": "Categorical",
            "bugs": "dcat",
            "jags": "dcat",
        },
        arg_aliases={
            "pymc": {"probs": "p"},
            "webppl": {"probs": "ps"},
        },
    ),
    # ----- count / rate families -----
    "Poisson": FamilyMeta(
        qvr_name="Poisson",
        distribution_class=td.Poisson,
        quivers_class=ConditionalPoisson,
        target_names={
            "stan": "poisson",
            "numpyro": "Poisson",
            "pyro": "Poisson",
            "pymc": "Poisson",
            "edward2": "Poisson",
            "turing": "Poisson",
            "gen": "poisson",
            "church": "poisson",
            "webppl": "Poisson",
            "bugs": "dpois",
            "jags": "dpois",
        },
        arg_aliases={
            "pymc": {"rate": "mu"},
            "webppl": {"rate": "mu"},
        },
    ),
    "NegativeBinomial": FamilyMeta(
        qvr_name="NegativeBinomial",
        distribution_class=td.NegativeBinomial,
        quivers_class=ConditionalNegativeBinomial,
        target_names={
            "stan": "neg_binomial_2",
            "numpyro": "NegativeBinomial2",
            "pyro": "NegativeBinomial",
            "pymc": "NegativeBinomial",
            "edward2": "NegativeBinomial",
            "turing": "NegativeBinomial",
            "gen": "neg_binom",
            "church": "negative-binomial",
            "webppl": "NegativeBinomial",
            "bugs": "dnegbin",
            "jags": "dnegbin",
        },
        arg_aliases={
            "pymc": {"probs": "p", "total_count": "n"},
        },
    ),
    "Geometric": FamilyMeta(
        qvr_name="Geometric",
        distribution_class=td.Geometric,
        quivers_class=ConditionalGeometric,
        target_names={
            "numpyro": "Geometric",
            "pyro": "Geometric",
            "pymc": "Geometric",
            "edward2": "Geometric",
            "turing": "Geometric",
            "gen": "geometric",
            "church": "geometric",
        },
        arg_aliases={
            "pymc": {"probs": "p"},
        },
    ),
    "Binomial": FamilyMeta(
        qvr_name="Binomial",
        distribution_class=td.Binomial,
        quivers_class=ConditionalBinomial,
        target_names={
            "stan": "binomial",
            "numpyro": "Binomial",
            "pyro": "Binomial",
            "pymc": "Binomial",
            "edward2": "Binomial",
            "turing": "Binomial",
            "webppl": "Binomial",
            "bugs": "dbin",
            "jags": "dbin",
        },
        arg_aliases={
            "pymc": {"probs": "p", "total_count": "n"},
            "webppl": {"probs": "p", "total_count": "n"},
        },
    ),
    "VonMises": FamilyMeta(
        qvr_name="VonMises",
        distribution_class=td.VonMises,
        quivers_class=ConditionalVonMises,
        target_names={
            "stan": "von_mises",
            "numpyro": "VonMises",
            "pyro": "VonMises",
            "pymc": "VonMises",
            "edward2": "VonMises",
            "turing": "VonMises",
        },
        arg_aliases={
            "pymc": {"loc": "mu", "concentration": "kappa"},
        },
    ),
    "LogisticNormal": FamilyMeta(
        qvr_name="LogisticNormal",
        distribution_class=td.LogisticNormal,
        quivers_class=ConditionalLogisticNormal,
        target_names={
            "numpyro": "LogisticNormal",
            "pyro": "LogisticNormal",
            "webppl": "LogisticNormal",
        },
        arg_aliases={
            "webppl": {"loc": "mu", "scale": "sigma"},
        },
    ),
    "OneHotCategorical": FamilyMeta(
        qvr_name="OneHotCategorical",
        distribution_class=td.OneHotCategorical,
        quivers_class=ConditionalOneHotCategorical,
        target_names={
            "numpyro": "OneHotCategorical",
            "pyro": "OneHotCategorical",
            "edward2": "OneHotCategorical",
        },
        arg_aliases={
            "pymc": {"probs": "p"},
        },
    ),
    "LKJCholesky": FamilyMeta(
        qvr_name="LKJCholesky",
        distribution_class=td.LKJCholesky,
        quivers_class=ConditionalLKJCholesky,
        target_names={
            "stan": "lkj_corr_cholesky",
            "numpyro": "LKJCholesky",
            "pyro": "LKJCorrCholesky",
            "pymc": "LKJCholesky",
            "edward2": "LKJ",
            "turing": "LKJCholesky",
            "gen": "lkj_cholesky",
            "webppl": "LKJCholesky",
        },
        arg_aliases={
            "pymc": {"concentration": "eta"},
        },
    ),
    "Mixture": FamilyMeta(
        qvr_name="Mixture",
        distribution_class=td.MixtureSameFamily,
        quivers_class=ConditionalMixture,
        target_names={
            "numpyro": "MixtureSameFamily",
            "pyro": "MixtureSameFamily",
            "pymc": "Mixture",
            "edward2": "MixtureSameFamily",
            "turing": "MixtureModel",
            "webppl": "Mixture",
        },
        arg_aliases={
            "pymc": {
                "mixture_distribution": "w",
                "component_distribution": "comp_dists",
            },
        },
    ),
    "Independent": FamilyMeta(
        qvr_name="Independent",
        distribution_class=td.Independent,
        quivers_class=ConditionalIndependent,
        target_names={
            "numpyro": "Independent",
            "pyro": "Independent",
            "edward2": "Independent",
        },
    ),
    "Transformed": FamilyMeta(
        qvr_name="Transformed",
        distribution_class=td.TransformedDistribution,
        quivers_class=ConditionalTransformed,
        target_names={
            "numpyro": "TransformedDistribution",
            "pyro": "TransformedDistribution",
            "edward2": "TransformedDistribution",
        },
    ),
    "Truncated": FamilyMeta(
        qvr_name="Truncated",
        distribution_class=_Truncated,
        quivers_class=Truncated,
        target_names={
            "pymc": "Truncated",
            "numpyro": "TruncatedDistribution",
            "pyro": "TruncatedDistribution",
            "turing": "truncated",
        },
        arg_aliases={
            "pymc": {"base_distribution": "dist"},
        },
    ),
    "LKJCorrelationFactor": FamilyMeta(
        qvr_name="LKJCorrelationFactor",
        distribution_class=_LKJCorrelationFactor,
        quivers_class=LKJCorrelationFactor,
        target_names={
            "pymc": "LKJCorr",
            "stan": "lkj_corr",
            "numpyro": "LKJ",
            "pyro": "LKJ",
        },
    ),
    # ----- compound / shim families -----
    "BetaBinomial": FamilyMeta(
        qvr_name="BetaBinomial",
        distribution_class=BetaBinomial,
        quivers_class=ConditionalBetaBinomial,
        target_names={
            "stan": "beta_binomial",
            "numpyro": "BetaBinomial",
            "pyro": "BetaBinomial",
            "pymc": "BetaBinomial",
            "edward2": "BetaBinomial",
            "turing": "BetaBinomial",
            "bugs": "dbetabin",
            "jags": "dbetabin",
            "gen": "beta_binomial",
            "webppl": "BetaBinomial",
        },
        arg_aliases={
            "pymc": {
                "concentration1": "alpha",
                "concentration0": "beta",
                "total_count": "n",
            },
        },
    ),
    "OrderedLogistic": FamilyMeta(
        qvr_name="OrderedLogistic",
        distribution_class=OrderedLogistic,
        quivers_class=ConditionalOrderedLogistic,
        target_names={
            "stan": "ordered_logistic",
            "numpyro": "OrderedLogistic",
            "pyro": "OrderedLogistic",
            "pymc": "OrderedLogistic",
        },
    ),
    "OrderedProbit": FamilyMeta(
        qvr_name="OrderedProbit",
        distribution_class=OrderedProbit,
        quivers_class=ConditionalOrderedProbit,
        target_names={
            "stan": "ordered_probit",
            "numpyro": "OrderedProbit",
            "pyro": "OrderedProbit",
            "pymc": "OrderedProbit",
        },
    ),
    "Logistic": FamilyMeta(
        qvr_name="Logistic",
        distribution_class=Logistic,
        quivers_class=ConditionalLogistic,
        target_names={
            "stan": "logistic",
            "numpyro": "Logistic",
            "pyro": "Logistic",
            "pymc": "Logistic",
            "edward2": "Logistic",
            "turing": "Logistic",
            "bugs": "dlogis",
            "jags": "dlogis",
            "gen": "logistic",
            "webppl": "Logistic",
        },
        arg_aliases={
            "pymc": {"loc": "mu", "scale": "s"},
        },
    ),
    "HalfStudentT": FamilyMeta(
        qvr_name="HalfStudentT",
        distribution_class=HalfStudentT,
        quivers_class=ConditionalHalfStudentT,
        target_names={
            "stan": "student_t",
            "numpyro": "HalfStudentT",
            "pyro": "HalfStudentT",
            "pymc": "HalfStudentT",
            "edward2": "HalfStudentT",
            "bugs": "dt",
            "jags": "dt",
            "gen": "half_student_t",
            "turing": "HalfStudentT",
            "webppl": "HalfStudentT",
        },
        arg_aliases={
            "pymc": {"df": "nu", "scale": "sigma"},
            "bugs": {"scale": "tau"},
            "jags": {"scale": "tau"},
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
