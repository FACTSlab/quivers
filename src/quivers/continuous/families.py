"""Parameterized distribution families as continuous morphisms.

Each family is a ContinuousMorphism whose codomain is a continuous
space and whose conditional distribution p(y | x) belongs to a
specific parametric family. The parameters are learnable functions
of x:

- For discrete domains (FinSet): parameters are looked up from a table.
- For continuous domains (ContinuousSpace): parameters are produced
  by a small neural network.

This module wraps every reparameterizable distribution in
torch.distributions as a conditional morphism, plus custom
families (TruncatedNormal, MultivariateNormal, etc.).

Architecture
------------
Most per-dimension-independent distributions are built on a shared
generic base ``_IndependentConditional`` that handles the parameter
source, transform, and torch.distributions plumbing. The
``_make_family`` class factory generates named classes from a
specification. Distributions that need special handling
(MultivariateNormal, Dirichlet, TruncatedNormal, etc.) are
implemented as standalone classes.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import constraints as _constraints

from quivers.continuous.family_spec import (
    FamilySpec,
    ParamSpec,
    _RAW_TRANSFORMS as _TRANSFORMS,
    register as _register_family,
)
from quivers.continuous.spaces import (
    ContinuousSpace,
    Euclidean,
)
from quivers.continuous.morphisms import (
    AnySpace,
    ContinuousMorphism,
    _make_source,
)
from quivers.core._util import EPS


def _lower_bounded(bound: float) -> Callable:
    """Create a transform that ensures output > bound."""

    def transform(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + bound

    return transform


def _resolve_support(
    dist_class: type, override: _constraints.Constraint | None
) -> _constraints.Constraint:
    """Look up the family's output support from the torch class, or
    use the user-supplied override. Falls back to ``real`` when
    neither is available."""
    if override is not None:
        return override
    support = getattr(dist_class, "support", None)
    if isinstance(support, _constraints.Constraint):
        return support
    return _constraints.real


def _resolve_discrete(dist_class: type, override: bool | None) -> bool:
    """Discrete iff the torch distribution lacks reparameterized
    sampling (``has_rsample == False``)."""
    if override is not None:
        return override
    return not bool(getattr(dist_class, "has_rsample", True))


# ============================================================================
# generic per-dimension-independent conditional
# ============================================================================


class _IndependentConditional(ContinuousMorphism):
    """Generic conditional morphism for per-dimension-independent distributions.

    Wraps any torch.distributions.Distribution class that operates
    independently on each dimension of the codomain. Parameters are
    produced by a learnable source (lookup table or neural net) and
    transformed to satisfy distribution constraints.

    Parameters
    ----------
    domain : AnySpace
        Source space.
    codomain : ContinuousSpace
        Target space.
    dist_class : type
        A torch.distributions.Distribution subclass.
    param_specs : list of (str, Callable)
        Each entry is (parameter_name, transform_function). One raw
        scalar per codomain dimension is allocated for each spec.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        dist_class: type,
        param_specs: list[tuple[str, Callable]],
        hidden_dim: int = 64,
        discrete: bool = False,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self._dist_class = dist_class
        self._param_specs = param_specs
        self._discrete = discrete

        # total raw parameters: one scalar per spec per codomain dim
        total_raw = len(param_specs) * d
        self.param_source = _make_source(domain, total_raw, hidden_dim)

    def _get_dist(self, x: torch.Tensor) -> D.Distribution:
        """Build the torch distribution for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.distributions.Distribution
            Parameterized distribution with batch_shape (batch, d).
        """
        raw = self.param_source(x)  # (batch, n_specs * d)
        params = {}
        offset = 0

        for name, transform in self._param_specs:
            chunk = raw[..., offset : offset + self._d]
            params[name] = transform(chunk)
            offset += self._d

        return self._dist_class(**params)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        y_typed = y.float() if self._discrete else y
        return dist.log_prob(y_typed).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        if self._discrete:
            return dist.sample(sample_shape).long()
        if getattr(dist, "has_rsample", True):
            return dist.rsample(sample_shape)
        # Continuous but non-reparameterizable (e.g. VonMises). Fall
        # back to ``.sample()`` — gradients won't flow through the
        # sample but the distribution still scores log-probabilities.
        return dist.sample(sample_shape)


# ============================================================================
# class factory
# ============================================================================


def _make_family(
    name: str,
    dist_class: type,
    param_specs: list[tuple[str, str]],
    doc: str,
    *,
    dsl_name: str | None = None,
    support: _constraints.Constraint | None = None,
    discrete: bool | None = None,
) -> type:
    """Generate a named conditional distribution class.

    Each call also installs a :class:`FamilySpec` into the unified
    catalog so the inline (fixed / mixed) call paths and the DSL
    compiler discover the family automatically.

    Parameters
    ----------
    name : str
        Class name (e.g. ``"ConditionalCauchy"``).
    dist_class : type
        The :mod:`torch.distributions` class.
    param_specs : list of (str, str)
        Ordered ``(parameter_name, transform_name)`` pairs.
    doc : str
        Class docstring.
    dsl_name : str, optional
        DSL key. Defaults to ``name.removeprefix("Conditional")``.
    support : Constraint, optional
        Output support. Auto-detected from ``dist_class.support`` if
        absent.
    discrete : bool, optional
        Whether the family is discrete. Auto-detected from
        ``not dist_class.has_rsample`` if absent.

    Returns
    -------
    type
        A new :class:`ContinuousMorphism` subclass.
    """
    resolved_specs = [(pname, _TRANSFORMS[tname]) for pname, tname in param_specs]
    is_discrete = _resolve_discrete(dist_class, discrete)
    out_support = _resolve_support(dist_class, support)

    class _Cls(_IndependentConditional):
        __doc__ = doc

        def __init__(
            self,
            domain: AnySpace,
            codomain: ContinuousSpace,
            hidden_dim: int = 64,
        ) -> None:
            super().__init__(
                domain,
                codomain,
                dist_class,
                resolved_specs,
                hidden_dim,
                discrete=is_discrete,
            )

    _Cls.__name__ = name
    _Cls.__qualname__ = name

    if dsl_name is None:
        dsl_name = name.removeprefix("Conditional")
    _register_family(
        FamilySpec(
            name=dsl_name,
            dist_class=dist_class,
            params=tuple(
                ParamSpec(name=pname, transform=tname) for pname, tname in param_specs
            ),
            support=out_support,
            discrete=is_discrete,
            output_kind="independent",
            docstring=doc,
            conditional_class_override=_Cls,
        )
    )
    return _Cls


# ============================================================================
# hand-written families (backward compatible, tested)
# ============================================================================


class ConditionalNormal(ContinuousMorphism):
    """Conditional normal (Gaussian) distribution.

    For each input x, produces an independent normal distribution
    on each dimension of the codomain:

        y_i ~ Normal(mu_i(x), sigma_i(x))

    Parameters are learnable: mu and log(sigma) are functions of x,
    implemented as lookup tables (discrete domain) or neural networks
    (continuous domain).

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : Euclidean
        Target space.
    hidden_dim : int
        Hidden layer width for neural parameter source.

    Examples
    --------
    >>> from quivers import FinSet
    >>> from quivers.continuous.spaces import Euclidean
    >>> A = FinSet(name="context", cardinality=5)
    >>> Y = Euclidean(name="response", dim=3)
    >>> f = ConditionalNormal(A, Y)
    >>> x = torch.tensor([0, 1, 2])
    >>> samples = f.rsample(x)  # shape (3, 3)
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim

        # param_dim = d (mu) + d (log_sigma)
        self.param_source = _make_source(domain, 2 * d, hidden_dim)
        self._d = d

    def _get_params(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract mu and sigma from the parameter source.

        Returns
        -------
        mu : torch.Tensor
            Mean. Shape (batch, d).
        sigma : torch.Tensor
            Standard deviation (positive). Shape (batch, d).
        """
        raw = self.param_source(x)  # (batch, 2*d)
        mu = raw[..., : self._d]
        log_sigma = raw[..., self._d :]
        sigma = log_sigma.exp().clamp(min=EPS)
        return mu, sigma

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        # independent normal log-density, summed over dimensions
        log_p = (
            -0.5 * ((y - mu) / sigma) ** 2 - sigma.log() - 0.5 * math.log(2 * math.pi)
        )

        return log_p.sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        # reparameterization trick
        eps = torch.randn(
            *sample_shape,
            *mu.shape,
            device=mu.device,
            dtype=mu.dtype,
        )

        return mu + sigma * eps


class ConditionalLogitNormal(ContinuousMorphism):
    """Conditional logit-normal distribution on (0, 1)^d.

    If z ~ Normal(mu(x), sigma(x)), then y = sigmoid(z) ~ LogitNormal.
    Useful for modeling probabilities and bounded quantities.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : Euclidean
        Target space (should have bounds [0, 1]).
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(domain, 2 * d, hidden_dim)
        self._d = d

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.unit_interval

    def _get_params(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.param_source(x)
        mu = raw[..., : self._d]
        log_sigma = raw[..., self._d :]
        sigma = log_sigma.exp().clamp(min=EPS)
        return mu, sigma

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        # logit transform
        y_clamped = y.clamp(min=EPS, max=1.0 - EPS)
        logit_y = torch.log(y_clamped / (1.0 - y_clamped))

        # normal log-density at logit(y)
        log_normal = (
            -0.5 * ((logit_y - mu) / sigma) ** 2
            - sigma.log()
            - 0.5 * math.log(2 * math.pi)
        )

        # jacobian correction: -log(y * (1-y))
        log_jacobian = -torch.log(y_clamped * (1.0 - y_clamped))

        return (log_normal + log_jacobian).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        eps = torch.randn(
            *sample_shape,
            *mu.shape,
            device=mu.device,
            dtype=mu.dtype,
        )
        z = mu + sigma * eps

        return torch.sigmoid(z)


class ConditionalBeta(ContinuousMorphism):
    """Conditional beta distribution on (0, 1)^d.

    For each input x, produces an independent Beta(alpha_i(x), beta_i(x))
    on each dimension of the codomain.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : Euclidean
        Target space (should have bounds [0, 1]).
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(domain, 2 * d, hidden_dim)
        self._d = d

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.unit_interval

    def _get_params(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.param_source(x)
        log_alpha = raw[..., : self._d]
        log_beta = raw[..., self._d :]
        alpha = F.softplus(log_alpha) + 0.1
        beta = F.softplus(log_beta) + 0.1
        return alpha, beta

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha, beta = self._get_params(x)
        y_clamped = y.clamp(min=EPS, max=1.0 - EPS)

        dist = D.Beta(alpha, beta)
        return dist.log_prob(y_clamped).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        alpha, beta = self._get_params(x)
        dist = D.Beta(alpha, beta)

        return dist.rsample(sample_shape)


class ConditionalTruncatedNormal(ContinuousMorphism):
    """Conditional truncated normal on [low, high]^d.

    A normal distribution restricted to a bounded interval. Uses
    rejection-free sampling via the inverse CDF (Phi-based) method.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : Euclidean
        Target space (must be bounded).
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: Euclidean,
        hidden_dim: int = 64,
    ) -> None:
        if codomain.low is None or codomain.high is None:
            raise ValueError("ConditionalTruncatedNormal requires a bounded codomain")

        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(domain, 2 * d, hidden_dim)
        self._d = d
        self._low = codomain.low
        self._high = codomain.high

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.interval(float(self._low), float(self._high))

    def _get_params(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.param_source(x)
        # initialize mu near center of interval
        mu = raw[..., : self._d] + (self._low + self._high) / 2.0
        log_sigma = raw[..., self._d :]
        sigma = log_sigma.exp().clamp(min=EPS)
        return mu, sigma

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        log_phi = (
            -0.5 * ((y - mu) / sigma) ** 2 - sigma.log() - 0.5 * math.log(2 * math.pi)
        )

        normal = D.Normal(0, 1)
        log_Z = torch.log(
            (
                normal.cdf((self._high - mu) / sigma)
                - normal.cdf((self._low - mu) / sigma)
            ).clamp(min=EPS)
        )

        return (log_phi - log_Z).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        normal = D.Normal(0, 1)
        alpha = normal.cdf((self._low - mu) / sigma)
        beta = normal.cdf((self._high - mu) / sigma)

        u = torch.rand(
            *sample_shape,
            *mu.shape,
            device=mu.device,
            dtype=mu.dtype,
        )
        u_scaled = alpha + u * (beta - alpha)
        u_scaled = u_scaled.clamp(min=EPS, max=1.0 - EPS)

        return normal.icdf(u_scaled) * sigma + mu


class ConditionalDirichlet(ContinuousMorphism):
    """Conditional Dirichlet distribution on a probability simplex.

    For each input x, produces a Dirichlet(alpha(x)) distribution
    on the simplex.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : Simplex
        Target simplex.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(domain, d, hidden_dim)
        self._d = d

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.simplex

    def _get_concentration(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.param_source(x)
        return F.softplus(raw) + 0.1

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        concentration = self._get_concentration(x)
        dist = D.Dirichlet(concentration)
        return dist.log_prob(y.clamp(min=EPS))

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        concentration = self._get_concentration(x)
        dist = D.Dirichlet(concentration)
        return dist.rsample(sample_shape)


# ============================================================================
# factory-generated families (all remaining torch.distributions with rsample)
# ============================================================================

# -- loc-scale family --------------------------------------------------------

ConditionalCauchy = _make_family(
    "ConditionalCauchy",
    D.Cauchy,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional Cauchy(loc(x), scale(x)). Heavy-tailed, no finite moments.",
)

ConditionalLaplace = _make_family(
    "ConditionalLaplace",
    D.Laplace,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional Laplace(loc(x), scale(x)). Sharp peak, heavier tails than normal.",
)

ConditionalGumbel = _make_family(
    "ConditionalGumbel",
    D.Gumbel,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional Gumbel(loc(x), scale(x)). Extreme value distribution (type I).",
)

ConditionalLogNormal = _make_family(
    "ConditionalLogNormal",
    D.LogNormal,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional LogNormal(loc(x), scale(x)). Positive-valued, right-skewed.",
)

ConditionalStudentT = _make_family(
    "ConditionalStudentT",
    D.StudentT,
    [("df", "softplus_shifted"), ("loc", "id"), ("scale", "softplus")],
    "Conditional StudentT(df(x), loc(x), scale(x)). Heavy-tailed with learnable df.",
)

# -- positive-valued distributions -------------------------------------------

ConditionalExponential = _make_family(
    "ConditionalExponential",
    D.Exponential,
    [("rate", "softplus")],
    "Conditional Exponential(rate(x)). Memoryless, positive-valued.",
)

ConditionalGamma = _make_family(
    "ConditionalGamma",
    D.Gamma,
    [("concentration", "softplus_shifted"), ("rate", "softplus")],
    "Conditional Gamma(concentration(x), rate(x)). Positive-valued, flexible shape.",
)

ConditionalChi2 = _make_family(
    "ConditionalChi2",
    D.Chi2,
    [("df", "softplus_shifted")],
    "Conditional Chi2(df(x)). Positive-valued, sum of squared normals.",
)

ConditionalHalfCauchy = _make_family(
    "ConditionalHalfCauchy",
    D.HalfCauchy,
    [("scale", "softplus")],
    "Conditional HalfCauchy(scale(x)). Heavy-tailed positive prior.",
)

ConditionalHalfNormal = _make_family(
    "ConditionalHalfNormal",
    D.HalfNormal,
    [("scale", "softplus")],
    "Conditional HalfNormal(scale(x)). Folded normal, positive-valued.",
)

ConditionalInverseGamma = _make_family(
    "ConditionalInverseGamma",
    D.InverseGamma,
    [("concentration", "softplus_shifted"), ("rate", "softplus")],
    "Conditional InverseGamma(concentration(x), rate(x)). Conjugate prior for normal variance.",
)

ConditionalWeibull = _make_family(
    "ConditionalWeibull",
    D.Weibull,
    [("scale", "softplus"), ("concentration", "softplus")],
    "Conditional Weibull(scale(x), concentration(x)). Survival analysis, reliability.",
)

ConditionalPareto = _make_family(
    "ConditionalPareto",
    D.Pareto,
    [("scale", "softplus"), ("alpha", "softplus")],
    "Conditional Pareto(scale(x), alpha(x)). Power-law tail.",
)

# -- (0, 1)-valued distributions ---------------------------------------------

ConditionalKumaraswamy = _make_family(
    "ConditionalKumaraswamy",
    D.Kumaraswamy,
    [("concentration1", "softplus_shifted"), ("concentration0", "softplus_shifted")],
    "Conditional Kumaraswamy(a(x), b(x)). Beta-like on (0,1), closed-form CDF.",
)

ConditionalContinuousBernoulli = _make_family(
    "ConditionalContinuousBernoulli",
    D.ContinuousBernoulli,
    [("logits", "id")],
    "Conditional ContinuousBernoulli(logits(x)). Continuous relaxation of Bernoulli.",
)

# -- two-df distributions ----------------------------------------------------

ConditionalFisherSnedecor = _make_family(
    "ConditionalFisherSnedecor",
    D.FisherSnedecor,
    [("df1", "softplus_shifted"), ("df2", "softplus_shifted")],
    "Conditional FisherSnedecor(df1(x), df2(x)). F-distribution, ratio of chi-squared.",
)

# -- discrete count distributions --------------------------------------------

ConditionalPoisson = _make_family(
    "ConditionalPoisson",
    D.Poisson,
    [("rate", "softplus")],
    "Conditional Poisson(rate(x)). Discrete, non-negative integer counts.",
)

ConditionalGeometric = _make_family(
    "ConditionalGeometric",
    D.Geometric,
    [("probs", "sigmoid")],
    "Conditional Geometric(probs(x)). Number of failures before first success.",
)

ConditionalNegativeBinomial = _make_family(
    "ConditionalNegativeBinomial",
    D.NegativeBinomial,
    [("total_count", "softplus_shifted"), ("probs", "sigmoid")],
    "Conditional NegativeBinomial(total_count(x), probs(x)). "
    "Discrete, over-dispersed count distribution.",
)

# -- circular distributions --------------------------------------------------

# VonMises is continuous (angles on the real line) but torch flags it
# ``has_rsample=False`` because reparameterized gradients are not
# implemented. Override the discrete-detection so the conditional
# class uses ``.sample()`` without casting to long.
ConditionalVonMises = _make_family(
    "ConditionalVonMises",
    D.VonMises,
    [("loc", "id"), ("concentration", "softplus")],
    "Conditional VonMises(loc(x), concentration(x)). "
    "Circular analogue of the Normal distribution.",
    discrete=False,
)

# -- uniform distribution (special parameterization) -------------------------


class ConditionalUniform(ContinuousMorphism):
    """Conditional uniform distribution on a learnable interval.

    Parameterized as Uniform(loc - width/2, loc + width/2) where
    loc is unconstrained and width is positive. This ensures
    low < high is always satisfied.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        # param_dim = d (loc) + d (raw_width)
        self.param_source = _make_source(domain, 2 * d, hidden_dim)
        self._d = d
        # The bounds are data-dependent, so we cannot pin a single
        # interval at construction time; advertise the codomain's
        # declared bounds when available, otherwise fall back to the
        # real line.
        low, high = getattr(codomain, "low", None), getattr(codomain, "high", None)
        if low is not None and high is not None:
            self._support_cache: _constraints.Constraint = _constraints.interval(
                float(low), float(high)
            )
        else:
            self._support_cache = _constraints.real

    @property
    def support(self) -> _constraints.Constraint:
        return self._support_cache

    def _get_dist(self, x: torch.Tensor) -> D.Uniform:
        raw = self.param_source(x)
        loc = raw[..., : self._d]
        width = F.softplus(raw[..., self._d :]) + EPS
        low = loc - width / 2.0
        high = loc + width / 2.0
        return D.Uniform(low, high)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


# ============================================================================
# multivariate distributions (special parameter structures)
# ============================================================================


class ConditionalMultivariateNormal(ContinuousMorphism):
    """Conditional multivariate normal with full covariance.

    Parameterized via Cholesky factor: the parameter source outputs
    loc (d values) and the lower-triangular entries of L (d*(d+1)/2
    values), where Sigma = L @ L^T.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space (d-dimensional).
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        n_tril = d * (d + 1) // 2
        self.param_source = _make_source(domain, d + n_tril, hidden_dim)
        self._d = d
        self._n_tril = n_tril

    def _get_dist(self, x: torch.Tensor) -> D.MultivariateNormal:
        raw = self.param_source(x)
        loc = raw[..., : self._d]
        tril_raw = raw[..., self._d :]

        # build lower-triangular matrix
        batch_shape = tril_raw.shape[:-1]
        L = torch.zeros(
            *batch_shape,
            self._d,
            self._d,
            device=tril_raw.device,
            dtype=tril_raw.dtype,
        )

        # fill lower triangle
        idx = torch.tril_indices(self._d, self._d)
        L[..., idx[0], idx[1]] = tril_raw

        # ensure positive diagonal (for positive definiteness)
        diag_idx = torch.arange(self._d)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx]) + EPS

        return D.MultivariateNormal(loc, scale_tril=L)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


class ConditionalLowRankMVN(ContinuousMorphism):
    """Conditional low-rank multivariate normal.

    Parameterized as loc + low-rank factor + diagonal:
        Sigma = W @ W^T + diag(d)

    This is more parameter-efficient than full MVN for high dimensions.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space (d-dimensional).
    rank : int
        Rank of the low-rank factor W.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        rank: int = 2,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self._rank = rank

        # loc (d) + factor (d * rank) + diag (d)
        total = d + d * rank + d
        self.param_source = _make_source(domain, total, hidden_dim)

    def _get_dist(self, x: torch.Tensor) -> D.LowRankMultivariateNormal:
        raw = self.param_source(x)
        d = self._d
        r = self._rank

        loc = raw[..., :d]
        factor_raw = raw[..., d : d + d * r]
        diag_raw = raw[..., d + d * r :]

        cov_factor = factor_raw.reshape(*raw.shape[:-1], d, r)
        cov_diag = F.softplus(diag_raw) + EPS

        return D.LowRankMultivariateNormal(loc, cov_factor, cov_diag)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


# ============================================================================
# relaxed discrete distributions (continuous outputs)
# ============================================================================


class ConditionalRelaxedBernoulli(ContinuousMorphism):
    """Conditional relaxed Bernoulli (concrete) distribution.

    Outputs continuous values in (0, 1) that approximate Bernoulli
    samples. The temperature controls the relaxation: lower temperature
    = closer to discrete.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space (should be 1-d per Bernoulli component).
    temperature : float
        Relaxation temperature.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        temperature: float = 0.5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(domain, d, hidden_dim)
        self._d = d
        self._temperature = temperature

    def _get_dist(self, x: torch.Tensor) -> D.RelaxedBernoulli:
        logits = self.param_source(x)
        temp = torch.tensor(
            self._temperature,
            device=logits.device,
            dtype=logits.dtype,
        )
        return D.RelaxedBernoulli(temperature=temp, logits=logits)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y.clamp(min=EPS, max=1.0 - EPS)).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


class ConditionalRelaxedOneHotCategorical(ContinuousMorphism):
    """Conditional relaxed one-hot categorical (Gumbel-Softmax).

    Outputs continuous vectors on the simplex that approximate
    one-hot categorical samples.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space (simplex or d-dimensional).
    temperature : float
        Relaxation temperature.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        temperature: float = 0.5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(domain, d, hidden_dim)
        self._d = d
        self._temperature = temperature

    def _get_dist(self, x: torch.Tensor) -> D.RelaxedOneHotCategorical:
        logits = self.param_source(x)
        temp = torch.tensor(
            self._temperature,
            device=logits.device,
            dtype=logits.dtype,
        )
        return D.RelaxedOneHotCategorical(temperature=temp, logits=logits)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y.clamp(min=EPS))

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


# ============================================================================
# wishart (matrix-valued)
# ============================================================================


class ConditionalWishart(ContinuousMorphism):
    """Conditional Wishart distribution over positive-definite matrices.

    Produces random d x d positive-definite matrices. Parameterized
    by degrees of freedom df(x) and a scale matrix V(x).

    The codomain dimension is interpreted as d, and outputs are
    d x d matrices flattened to d^2.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space. dim is the matrix size d (output is d x d).
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        n_tril = d * (d + 1) // 2
        # df (1) + lower-triangular scale (n_tril)
        self.param_source = _make_source(domain, 1 + n_tril, hidden_dim)
        self._d = d
        self._n_tril = n_tril

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.positive_definite

    def _get_dist(self, x: torch.Tensor) -> D.Wishart:
        raw = self.param_source(x)
        d = self._d

        # df must be > d - 1
        df = F.softplus(raw[..., 0]) + d

        tril_raw = raw[..., 1:]
        batch_shape = tril_raw.shape[:-1]
        L = torch.zeros(
            *batch_shape,
            d,
            d,
            device=tril_raw.device,
            dtype=tril_raw.dtype,
        )
        idx = torch.tril_indices(d, d)
        L[..., idx[0], idx[1]] = tril_raw
        diag_idx = torch.arange(d)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx]) + EPS

        return D.Wishart(df=df, scale_tril=L)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        d = self._d
        y_mat = y.reshape(*y.shape[:-1], d, d)
        return dist.log_prob(y_mat)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        mat = dist.rsample(sample_shape)
        # flatten d x d to d^2
        return mat.reshape(*mat.shape[:-2], self._d * self._d)


# ============================================================================
# optional: GeneralizedPareto (may not be in all torch versions)
# ============================================================================

# ============================================================================
# discrete-valued conditional distributions
# ============================================================================


class ConditionalBernoulli(ContinuousMorphism):
    """Conditional Bernoulli: continuous probability -> discrete truth value.

    Takes a continuous input x and produces learnable logits that
    parameterize a Bernoulli distribution. The output is a discrete
    sample in {0, 1}, returned as a LongTensor.

    This is the key bridge used in PDS (Grove & White) for the
    ``Bern x`` pattern, where a LogitNormal draw x in (0,1)
    parameterizes a Bernoulli over truth values.

    The codomain must be a FinSet of size 2 (representing {False, True}
    or {0, 1}).

    Note
    ----
    Sampling from Bernoulli is NOT reparameterizable. Gradients
    do not flow through the discrete samples. Use score function
    estimators (REINFORCE) or the Gumbel-Softmax trick if
    differentiable samples are needed.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space (typically UnitInterval or a FinSet).
    codomain : SetObject
        Target FinSet of size 2.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        hidden_dim: int = 64,
    ) -> None:
        from quivers.core.objects import SetObject

        if not isinstance(codomain, SetObject) or codomain.size != 2:
            raise ValueError(
                f"ConditionalBernoulli requires a FinSet(2) codomain, got {codomain!r}"
            )

        super().__init__(domain, codomain)

        # one logit per input
        self.param_source = _make_source(domain, 1, hidden_dim)

    def _get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Bernoulli probabilities from input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Probabilities. Shape (batch,).
        """
        logits = self.param_source(x).squeeze(-1)  # (batch,)
        return torch.sigmoid(logits)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability of discrete output y given input x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Discrete output in {0, 1}. Shape (batch,).

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape (batch,).
        """
        probs = self._get_probs(x)
        dist = D.Bernoulli(probs=probs)
        return dist.log_prob(y.float())

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Sample from Bernoulli (not reparameterizable).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Discrete samples in {0, 1}. Shape (*sample_shape, batch).
        """
        probs = self._get_probs(x)
        dist = D.Bernoulli(probs=probs)
        return dist.sample(sample_shape).long()


class ConditionalCategorical(ContinuousMorphism):
    """Conditional Categorical: continuous input -> discrete category.

    Generalizes ConditionalBernoulli to k > 2 categories.
    Takes a continuous input and produces learnable logits over k
    categories. The output is a discrete sample in {0, ..., k-1}.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : SetObject
        Target FinSet of size k.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        hidden_dim: int = 64,
    ) -> None:
        from quivers.core.objects import SetObject

        if not isinstance(codomain, SetObject):
            raise ValueError(
                f"ConditionalCategorical requires a FinSet codomain, got {codomain!r}"
            )

        super().__init__(domain, codomain)
        self._k = codomain.size
        self.param_source = _make_source(domain, self._k, hidden_dim)

    def _get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute category logits from input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Logits. Shape (batch, k).
        """
        return self.param_source(x)  # (batch, k)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability of discrete output y given input x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Discrete output in {0, ..., k-1}. Shape (batch,).

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape (batch,).
        """
        logits = self._get_logits(x)
        dist = D.Categorical(logits=logits)
        return dist.log_prob(y.long())

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Sample from Categorical (not reparameterizable).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Discrete samples in {0, ..., k-1}. Shape (*sample_shape, batch).
        """
        logits = self._get_logits(x)
        dist = D.Categorical(logits=logits)
        return dist.sample(sample_shape).long()


class ConditionalBinomial(ContinuousMorphism):
    """Conditional Binomial(total_count, probs(x)).

    The ``total_count`` (number of trials) is a fixed
    hyperparameter set at construction time — typical for binomial
    likelihoods where ``n`` is known per observation. Only the
    ``probs`` parameter is learnable.

    Outputs integer counts in ``{0, 1, ..., total_count}``.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space.
    total_count : int
        Number of Bernoulli trials per observation.
    hidden_dim : int
        Hidden layer width for the parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        total_count: int = 1,
        hidden_dim: int = 64,
    ) -> None:
        if total_count < 1:
            raise ValueError(
                f"ConditionalBinomial: total_count must be >= 1, got {total_count}"
            )
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self._total_count = int(total_count)
        self.param_source = _make_source(domain, d, hidden_dim)

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.integer_interval(0, self._total_count)

    def _get_dist(self, x: torch.Tensor) -> D.Binomial:
        logits = self.param_source(x)
        return D.Binomial(total_count=self._total_count, logits=logits)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y.float()).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.sample(sample_shape).long()


class ConditionalLogisticNormal(ContinuousMorphism):
    """Conditional LogisticNormal on the simplex.

    Pushes a Normal(loc(x), scale(x)) draw through the softmax
    transform to produce a simplex-valued sample. Multivariate
    analogue of :class:`ConditionalLogitNormal`. Useful as an
    alternative to :class:`ConditionalDirichlet` when the
    underlying simplex distribution should be Gaussian in
    logit space rather than Beta-shaped.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space; ``codomain.dim`` is the simplex dimension.
    hidden_dim : int
        Hidden layer width for the parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        # We use a Normal in (d-1)-dim space and the
        # StickBreakingTransform to land on the d-simplex.
        # torch.distributions.LogisticNormal handles this.
        self.param_source = _make_source(domain, 2 * (d - 1), hidden_dim)
        self._d = d

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.simplex

    def _get_dist(self, x: torch.Tensor) -> D.LogisticNormal:
        raw = self.param_source(x)
        d_minus_1 = self._d - 1
        loc = raw[..., :d_minus_1]
        scale = F.softplus(raw[..., d_minus_1:]) + EPS
        return D.LogisticNormal(loc, scale)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


class ConditionalOneHotCategorical(ContinuousMorphism):
    """Conditional OneHotCategorical(probs(x)).

    Generalises :class:`ConditionalCategorical` to one-hot
    encoded outputs (vector of zeros with a single one). Useful
    as a discrete-output observation kernel where downstream
    code wants a vector rather than an integer index.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space; ``codomain.dim`` is the number of categories.
    hidden_dim : int
        Hidden layer width for the parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self.param_source = _make_source(domain, d, hidden_dim)

    @property
    def support(self) -> _constraints.Constraint:
        # torch's OneHotCategorical.support is OneHot(); the
        # variational guide treats it as a simplex bijector target.
        return D.OneHotCategorical.support  # type: ignore[return-value]

    def _get_dist(self, x: torch.Tensor) -> D.OneHotCategorical:
        logits = self.param_source(x)
        return D.OneHotCategorical(logits=logits)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y.float())

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.sample(sample_shape).float()


class ConditionalMixture(ContinuousMorphism):
    """K-component mixture of a conditional family.

    Wraps a single conditional family class (one of the registered
    ``ConditionalX`` types) and gives it K independent
    parameterizations plus learnable mixture logits, producing

        p(y | x) = sum_k pi_k(x) * f_k(y | x)

    where each ``f_k`` is an instance of the component class and
    ``pi`` is the softmax of ``K`` learnable logits.

    Sampling is via ancestral simulation (Categorical pick + the
    chosen component's ``rsample``). ``log_prob`` evaluates the
    log-sum-exp of the per-component log-densities. The Categorical
    pick is non-reparameterizable; gradient flow through the
    weights uses the score-function path (which higher-level
    objectives like IWAE can route through).

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space; matches the component family's codomain.
    component_class : type
        A ``ConditionalX`` class accepting ``(domain, codomain,
        hidden_dim)`` constructor args.
    num_components : int
        Number of mixture components.
    hidden_dim : int
        Hidden width for both the mixture-logit MLP and each
        component's parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        component_class: type,
        num_components: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        if num_components < 2:
            raise ValueError(
                f"ConditionalMixture: num_components must be >= 2, "
                f"got {num_components}"
            )
        super().__init__(domain, codomain)
        self._K = int(num_components)
        self._components = torch.nn.ModuleList(
            [component_class(domain, codomain, hidden_dim) for _ in range(self._K)]
        )
        self.mixture_logits = _make_source(domain, self._K, hidden_dim)

    @property
    def support(self):  # type: ignore[override]
        return self._components[0].support

    def _log_weights(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mixture_logits(x)
        return torch.log_softmax(logits, dim=-1)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_w = self._log_weights(x)
        per_comp = torch.stack(
            [comp.log_prob(x, y) for comp in self._components], dim=-1
        )
        return torch.logsumexp(log_w + per_comp, dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        log_w = self._log_weights(x)
        cat = D.Categorical(logits=log_w)
        comp_idx = cat.sample(sample_shape)  # (*sample_shape, batch)
        # Draw from every component, then gather along a new K axis.
        comp_samples = torch.stack(
            [comp.rsample(x, sample_shape) for comp in self._components],
            dim=-1,
        )
        # ``comp_samples`` has shape ``(*sample_shape, batch, *event,
        # K)``. Broadcast ``comp_idx`` (shape ``(*sample_shape,
        # batch)``) to match every event axis with size 1 then 1 on
        # the K axis, so the final gather picks one element per
        # batch row.
        idx = comp_idx
        for _ in range(comp_samples.dim() - 1 - idx.dim()):
            idx = idx.unsqueeze(-1)
        idx = idx.unsqueeze(-1)  # K-axis broadcast dim
        idx = idx.expand(*comp_samples.shape[:-1], 1)
        return comp_samples.gather(-1, idx).squeeze(-1)


class ConditionalIndependent(ContinuousMorphism):
    """Reinterpret the trailing batch dimension of a base conditional
    family as an event dimension.

    Equivalent to wrapping the base distribution in
    :class:`torch.distributions.Independent` with
    ``reinterpreted_batch_ndims = 1``. Used to make per-element
    independence explicit when a downstream guide wants to score
    a vector-valued observation as a single event.

    Parameters
    ----------
    base : ContinuousMorphism
        The base conditional family. The wrapped distribution sums
        the base's per-element log-probabilities along the last
        axis to score a vector-valued observation.
    """

    def __init__(self, base: ContinuousMorphism) -> None:
        super().__init__(base.domain, base.codomain)
        self._base = base

    @property
    def support(self):  # type: ignore[override]
        return self._base.support

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # The base's log_prob already sums along the last axis for
        # ``_IndependentConditional``; ``Independent`` adds the
        # explicit reinterpretation for any base.
        lp = self._base.log_prob(x, y)
        if lp.dim() > 1:
            return lp.sum(dim=-1)
        return lp

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        return self._base.rsample(x, sample_shape)


class ConditionalTransformed(ContinuousMorphism):
    """A base conditional family composed with a chain of bijectors.

    Equivalent to :class:`torch.distributions.TransformedDistribution`
    lifted to ContinuousMorphism. The ``transforms`` are applied in
    forward order to ``rsample`` outputs; ``log_prob`` includes the
    log-determinant Jacobian correction.

    Parameters
    ----------
    base : ContinuousMorphism
        Base conditional family.
    transforms : list of torch.distributions.Transform
        Bijectors applied in forward order. Each must implement
        ``__call__``, ``inv``, and ``log_abs_det_jacobian``.
    """

    def __init__(
        self,
        base: ContinuousMorphism,
        transforms: list,
    ) -> None:
        super().__init__(base.domain, base.codomain)
        self._base = base
        self._transforms = list(transforms)

    @property
    def support(self):  # type: ignore[override]
        # The composed support is the codomain of the final transform.
        if self._transforms:
            return self._transforms[-1].codomain
        return self._base.support

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        z = self._base.rsample(x, sample_shape)
        for tf in self._transforms:
            z = tf(z)
        return z

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Invert the transforms to recover the base sample, then
        # accumulate the log-det-Jacobian corrections.
        z = y
        log_det = torch.zeros(
            y.shape[:-1] if y.dim() > 1 else (1,),
            device=y.device,
            dtype=y.dtype,
        )
        for tf in reversed(self._transforms):
            z_prev = tf.inv(z)
            ldj = tf.log_abs_det_jacobian(z_prev, z)
            if ldj.dim() > log_det.dim():
                ldj = ldj.sum(dim=tuple(range(log_det.dim(), ldj.dim())))
            log_det = log_det + ldj
            z = z_prev
        return self._base.log_prob(x, z) - log_det


class ConditionalLKJCholesky(ContinuousMorphism):
    """Conditional LKJCholesky(dim, concentration(x)).

    Produces lower-triangular Cholesky factors of correlation
    matrices on the LKJ distribution (Lewandowski-Kurowicka-Joe
    2009, doi:10.1016/j.jmva.2009.04.008). The matrix dimension
    is taken from ``codomain.dim``; only the concentration parameter
    is learnable.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space; ``codomain.dim`` is the correlation-matrix size.
    hidden_dim : int
        Hidden layer width for the parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(domain, codomain)
        self._matrix_dim = codomain.dim
        self.param_source = _make_source(domain, 1, hidden_dim)

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.corr_cholesky

    def _get_dist(self, x: torch.Tensor) -> D.LKJCholesky:
        raw = self.param_source(x)
        concentration = F.softplus(raw.squeeze(-1)) + 0.1
        return D.LKJCholesky(dim=self._matrix_dim, concentration=concentration)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.sample(sample_shape)


# ============================================================================
# optional: generalized Pareto (requires recent torch)
# ============================================================================

try:
    _GPD = D.GeneralizedPareto

    class ConditionalGeneralizedPareto(ContinuousMorphism):
        """Conditional generalized Pareto distribution.

        Parameters
        ----------
        domain : SetObject or ContinuousSpace
            Source space.
        codomain : ContinuousSpace
            Target space.
        hidden_dim : int
            Hidden layer width for neural parameter source.
        """

        def __init__(
            self,
            domain: AnySpace,
            codomain: ContinuousSpace,
            hidden_dim: int = 64,
        ) -> None:
            super().__init__(domain, codomain)
            d = codomain.dim
            # loc + scale + concentration
            self.param_source = _make_source(domain, 3 * d, hidden_dim)
            self._d = d

        def _get_dist(self, x: torch.Tensor) -> D.GeneralizedPareto:
            raw = self.param_source(x)
            d = self._d
            loc = raw[..., :d]
            scale = F.softplus(raw[..., d : 2 * d]) + EPS
            concentration = raw[..., 2 * d :]
            return _GPD(loc, scale, concentration)

        def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            dist = self._get_dist(x)
            return dist.log_prob(y).sum(dim=-1)

        def rsample(
            self,
            x: torch.Tensor,
            sample_shape: torch.Size = torch.Size(),
        ) -> torch.Tensor:
            dist = self._get_dist(x)
            return dist.rsample(sample_shape)

    _HAS_GPD = True

except AttributeError:
    _HAS_GPD = False


# ============================================================================
# FamilySpec registrations for hand-written classes
# ============================================================================
#
# The factory-generated families (Cauchy, Laplace, ..., VonMises) are
# registered inside :func:`_make_family`. The hand-written specials
# below (Normal, LogitNormal, Beta, TruncatedNormal, Dirichlet, Uniform,
# MultivariateNormal, LowRankMVN, RelaxedBernoulli,
# RelaxedOneHotCategorical, Wishart, Bernoulli, Categorical) require
# explicit registration. Each :class:`FamilySpec` records the same
# canonical info the factory-generated ones do; the
# ``conditional_class_override`` field carries the hand-written class
# so the unified catalog can build a ConditionalX from any DSL name.


_register_family(
    FamilySpec(
        name="Normal",
        dist_class=D.Normal,
        params=(ParamSpec("loc", "id"), ParamSpec("scale", "softplus")),
        support=_constraints.real,
        output_kind="independent",
        docstring="Conditional Normal(loc(x), scale(x)).",
        conditional_class_override=ConditionalNormal,
    )
)

_register_family(
    FamilySpec(
        name="LogitNormal",
        dist_class=D.TransformedDistribution,
        params=(ParamSpec("mu", "id"), ParamSpec("sigma", "softplus")),
        support=_constraints.unit_interval,
        output_kind="independent",
        docstring="LogitNormal(mu(x), sigma(x)) — Normal pushed through sigmoid.",
        conditional_class_override=ConditionalLogitNormal,
    )
)

_register_family(
    FamilySpec(
        name="Beta",
        dist_class=D.Beta,
        params=(
            ParamSpec("concentration1", "softplus_shifted"),
            ParamSpec("concentration0", "softplus_shifted"),
        ),
        support=_constraints.unit_interval,
        output_kind="independent",
        docstring="Conditional Beta(alpha(x), beta(x)) on (0, 1).",
        conditional_class_override=ConditionalBeta,
    )
)

_register_family(
    FamilySpec(
        name="TruncatedNormal",
        dist_class=D.Normal,  # surrogate; inline path handles truncation
        params=(
            ParamSpec("mu", "id"),
            ParamSpec("sigma", "softplus"),
        ),
        support=_constraints.real,  # interval set per-call when bounds are literal
        output_kind="independent",
        docstring="TruncatedNormal(mu(x), sigma(x), low, high) — Normal truncated to [low, high].",
        conditional_class_override=ConditionalTruncatedNormal,
    )
)

_register_family(
    FamilySpec(
        name="Dirichlet",
        dist_class=D.Dirichlet,
        params=(ParamSpec("concentration", "softplus_shifted", kind="vector"),),
        support=_constraints.simplex,
        output_kind="vector",
        docstring="Conditional Dirichlet(alpha(x)) on the simplex.",
        conditional_class_override=ConditionalDirichlet,
    )
)

_register_family(
    FamilySpec(
        name="Uniform",
        dist_class=D.Uniform,
        params=(ParamSpec("low", "id"), ParamSpec("high", "id")),
        support=_constraints.real,  # interval set per-call when bounds are literal
        output_kind="independent",
        docstring="Conditional Uniform(low(x), high(x)).",
        conditional_class_override=ConditionalUniform,
    )
)

_register_family(
    FamilySpec(
        name="MultivariateNormal",
        dist_class=D.MultivariateNormal,
        params=(ParamSpec("loc", "id"), ParamSpec("scale_tril", "id")),
        support=_constraints.real_vector,
        output_kind="mvn",
        docstring="Conditional MultivariateNormal(loc(x), scale_tril(x)).",
        conditional_class_override=ConditionalMultivariateNormal,
    )
)

_register_family(
    FamilySpec(
        name="LowRankMVN",
        dist_class=D.LowRankMultivariateNormal,
        params=(
            ParamSpec("loc", "id"),
            ParamSpec("cov_factor", "id"),
            ParamSpec("cov_diag", "softplus"),
        ),
        support=_constraints.real_vector,
        output_kind="mvn",
        docstring="Conditional LowRankMVN(loc(x), W(x), D(x)) — Σ = W W^T + diag(D).",
        conditional_class_override=ConditionalLowRankMVN,
    )
)

_register_family(
    FamilySpec(
        name="RelaxedBernoulli",
        dist_class=D.RelaxedBernoulli,
        params=(ParamSpec("temperature", "softplus"), ParamSpec("logits", "id")),
        support=_constraints.unit_interval,
        output_kind="independent",
        docstring="RelaxedBernoulli(temperature, logits(x)) — Gumbel-softmax / continuous Bernoulli.",
        conditional_class_override=ConditionalRelaxedBernoulli,
    )
)

_register_family(
    FamilySpec(
        name="RelaxedOneHotCategorical",
        dist_class=D.RelaxedOneHotCategorical,
        params=(ParamSpec("temperature", "softplus"), ParamSpec("logits", "id", kind="vector")),
        support=_constraints.simplex,
        output_kind="vector",
        docstring="RelaxedOneHotCategorical(temperature, logits(x)) — Gumbel-softmax on the simplex.",
        conditional_class_override=ConditionalRelaxedOneHotCategorical,
    )
)

_register_family(
    FamilySpec(
        name="Wishart",
        dist_class=D.Wishart,
        params=(
            ParamSpec("df", "softplus_shifted"),
            ParamSpec("scale_tril", "id"),
        ),
        support=_constraints.positive_definite,
        output_kind="matrix",
        docstring="Conditional Wishart(df(x), scale_tril(x)) — positive-definite matrices.",
        conditional_class_override=ConditionalWishart,
    )
)

_register_family(
    FamilySpec(
        name="Bernoulli",
        dist_class=D.Bernoulli,
        params=(ParamSpec("probs", "sigmoid"),),
        support=_constraints.boolean,
        discrete=True,
        output_kind="categorical",
        docstring="Conditional Bernoulli(probs(x)).",
        conditional_class_override=ConditionalBernoulli,
    )
)

_register_family(
    FamilySpec(
        name="Categorical",
        dist_class=D.Categorical,
        params=(ParamSpec("logits", "id", kind="vector"),),
        support=_constraints.nonnegative_integer,
        discrete=True,
        output_kind="categorical",
        docstring="Conditional Categorical(logits(x)) over {0, ..., k-1}.",
        conditional_class_override=ConditionalCategorical,
    )
)

_register_family(
    FamilySpec(
        name="Binomial",
        dist_class=D.Binomial,
        params=(ParamSpec("probs", "sigmoid"),),
        support=_constraints.nonnegative_integer,
        discrete=True,
        output_kind="categorical",
        docstring="Conditional Binomial(n=total_count, probs(x)) with fixed total_count.",
        conditional_class_override=ConditionalBinomial,
    )
)


_register_family(
    FamilySpec(
        name="LogisticNormal",
        dist_class=D.LogisticNormal,
        params=(ParamSpec("loc", "id"), ParamSpec("scale", "softplus")),
        support=_constraints.simplex,
        output_kind="vector",
        docstring="Conditional LogisticNormal(loc(x), scale(x)) on the simplex.",
        conditional_class_override=ConditionalLogisticNormal,
    )
)


_register_family(
    FamilySpec(
        name="OneHotCategorical",
        dist_class=D.OneHotCategorical,
        params=(ParamSpec("logits", "id", kind="vector"),),
        support=D.OneHotCategorical.support,
        discrete=False,  # vector output, not integer-scalar
        output_kind="vector",
        docstring="Conditional OneHotCategorical(logits(x)) — one-hot vector output.",
        conditional_class_override=ConditionalOneHotCategorical,
    )
)


_register_family(
    FamilySpec(
        name="LKJCholesky",
        dist_class=D.LKJCholesky,
        params=(ParamSpec("concentration", "softplus_shifted"),),
        support=_constraints.corr_cholesky,
        output_kind="matrix",
        docstring="Conditional LKJCholesky(matrix_dim, concentration(x)) for correlation Cholesky factors.",
        conditional_class_override=ConditionalLKJCholesky,
    )
)


if _HAS_GPD:
    _register_family(
        FamilySpec(
            name="GeneralizedPareto",
            dist_class=_GPD,  # type: ignore[name-defined]
            params=(
                ParamSpec("loc", "id"),
                ParamSpec("scale", "softplus"),
                ParamSpec("concentration", "id"),
            ),
            support=_constraints.real,
            output_kind="independent",
            docstring="Conditional GeneralizedPareto(loc(x), scale(x), concentration(x)).",
            conditional_class_override=ConditionalGeneralizedPareto,  # type: ignore[name-defined]
        )
    )
