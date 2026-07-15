"""Inline distribution morphisms for monadic programs.

These ContinuousMorphisms parameterize distributions using either
fixed literal values or direct variable values, without learned
neural-network transformations. Used by the DSL compiler for inline
draw steps like::

    draw x ~ LogitNormal(0.0, 1.0)                 # all fixed
    draw b ~ Bernoulli(x)                           # direct variable
    draw r ~ TruncatedNormal(mu, sigma, 0.0, 1.0)  # mixed

Terminology
-----------
- **Fixed**: all distribution parameters are float literals known at
  compile time. The morphism has domain ``Unit`` (terminal object)
  and ignores its input.
- **Direct**: some or all parameters come from bound variables at
  runtime. The morphism uses the input tensor directly as
  distribution parameters (no learned neural-net transformation).

Architecture
------------
Each family is described once by a `FamilySpec` in
`quivers.continuous.family_spec`. The factories and builders
below dispatch on the registered spec rather than maintaining
per-family hard-coded functions; specials (truncated, Dirichlet,
uniform, transformed) override the generic dispatch by setting
``fixed_factory_override`` / ``mixed_builder_override`` on their
FamilySpec entry.
"""

from __future__ import annotations
import math
from collections.abc import Callable
import torch
import torch.distributions as D
from torch.distributions import constraints as _constraints
from quivers.core.objects import Unit
from quivers.continuous._ordered import OrderedLogistic
from quivers.continuous._zip_hurdle import (
    HurdlePoisson,
    MixtureNormal,
    ZeroInflatedPoisson,
)
from quivers.continuous.measure import (
    Independent as _MeasureIndependent,
    Mixture as _MeasureMixture,
    Normalize as _MeasureNormalize,
    PointMass as _MeasurePointMass,
    Pushforward as _MeasurePushforward,
    Restrict as _MeasureRestrict,
)
from quivers.continuous.spaces import Euclidean
from quivers.continuous.morphisms import ContinuousMorphism, AnySpace
from quivers.continuous.family_spec import (
    FAMILY_REGISTRY,
    FamilySpec,
)
from quivers.core._util import EPS
from quivers.dsl.ast_nodes import (
    DrawArgDist,
    DrawArgList,
    DrawArgName,
    DrawArgScalar,
)


class FixedDistribution(ContinuousMorphism):
    """A distribution with all parameters fixed at construction time.

    The ``rsample`` input is used only for batch size and device
    inference; the distribution parameters themselves are constants.

    Parameters
    ----------
    codomain : AnySpace
        The output space.
    make_dist : callable
        ``(batch_size: int, device: torch.device) -> Distribution``.
    discrete : bool
        Whether the output is discrete (returns LongTensor).
    support : _constraints.Constraint, optional
        The support of the underlying distribution. Used by variational
        guides to apply the right bijector for unconstrained→constrained
        sampling. Defaults to ``constraints.real``; the factory
        functions for each family supply the correct constraint.
    """

    def __init__(
        self,
        codomain: AnySpace,
        make_dist: Callable,
        discrete: bool = False,
        support: _constraints.Constraint | None = None,
    ) -> None:
        super().__init__(Unit, codomain)
        self._make_dist_fn = make_dist
        self._discrete = discrete
        self._support = support if support is not None else D.constraints.real

    @property
    def support(self) -> _constraints.Constraint:
        return self._support

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from the fixed distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input (used only for batch size and device).
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples from the distribution.
        """
        batch = x.shape[0]
        dist = self._make_dist_fn(batch, x.device)
        if self._discrete:
            return dist.sample(sample_shape).long()
        if getattr(dist, "has_rsample", True):
            return dist.rsample(sample_shape)
        # Continuous but non-reparameterizable (e.g. VonMises).
        return dist.sample(sample_shape)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability under the fixed distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input (used only for batch size and device).
        y : torch.Tensor
            Output values.

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape ``(batch,)``.
        """
        batch = x.shape[0]
        dist = self._make_dist_fn(batch, x.device)
        lp = dist.log_prob(y.float() if self._discrete else y)
        if lp.dim() > 1:
            return lp.sum(dim=-1)
        return lp


class MixedInlineDistribution(ContinuousMorphism):
    """General inline distribution with arbitrary literal/variable mix.

    Handles any combination of literal float values and runtime variable
    inputs for any distribution family. At construction time, receives a
    specification of which parameter positions are fixed literals vs.
    variable inputs, plus a builder function that creates the PyTorch
    distribution from fully-resolved parameter tensors.

    This is the general mechanism underlying all inline distribution
    patterns in the DSL::

        draw x ~ Normal(0.0, 1.0)           # all fixed (0 variable inputs)
        draw y ~ Normal(mu, sigma)           # all variable (2 variable inputs)
        draw z ~ Normal(mu, 0.5)             # mixed (1 variable input)
        draw w ~ TruncatedNormal(mu, 0.5, 0.0, 1.0)  # mixed

    Parameters
    ----------
    domain : AnySpace
        Source space (stacked variable parameters from program env).
    codomain : AnySpace
        Target space.
    param_spec : list of tuple
        For each distribution parameter position, one of:
        - ``('var', dim)``   — variable from input; ``dim`` is its width
        - ``('lit', value)`` — fixed literal float value
    dist_builder : callable
        ``(list[torch.Tensor]) -> torch.distributions.Distribution``.
        Receives one 1-D tensor per parameter (all same batch size).
    discrete : bool
        Whether the output is discrete (returns LongTensor).
    support : Constraint, optional
        Support of the underlying distribution; used by variational
        guides. Defaults to ``constraints.real``.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        param_spec: list[tuple[str, int | float]],
        dist_builder: Callable,
        discrete: bool = False,
        support: _constraints.Constraint | None = None,
        param_event_ranks: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        self._param_spec = param_spec
        self._dist_builder = dist_builder
        self._discrete = discrete
        self._support = support if support is not None else _constraints.real
        # Per-position event rank. Position 0 = per-row scalar; >= 1 =
        # vector-shaped distribution parameter (cutpoints, mixture
        # weights / locations / scales). Any number of vector-typed
        # positions is supported: each consumes the number of columns
        # recorded in its ``param_spec`` dim, so the stacked input is
        # split by offset. The one exception is a vector whose event
        # dimension was not resolved at construction (dim recorded as
        # 1 while the runtime tensor is wider); a single such vector,
        # in the last slot, absorbs the surplus columns.
        self._param_event_ranks: tuple[int, ...] = (
            param_event_ranks
            if param_event_ranks is not None
            else tuple(0 for _ in param_spec)
        )
        if len(self._param_event_ranks) != len(param_spec):
            raise ValueError(
                "MixedInlineDistribution: param_event_ranks length "
                f"{len(self._param_event_ranks)} disagrees with "
                f"param_spec length {len(param_spec)}"
            )
        vec_positions = [i for i, r in enumerate(self._param_event_ranks) if r >= 1]
        self._has_vector_param: bool = bool(vec_positions)
        self._last_vector_pos: int | None = vec_positions[-1] if vec_positions else None
        self._n_vector_params: int = len(vec_positions)

    @property
    def support(self) -> _constraints.Constraint:
        return self._support

    def _resolve_params(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Reconstruct full parameter list from input + stored literals.

        Parameters
        ----------
        x : torch.Tensor
            Stacked variable parameters. Shape ``(batch, total_var_dim)``.
            May be ``(batch,)`` if total variable dimension is 1. When
            the family declares a vector-typed final parameter, the
            trailing columns of ``x`` are the vector values.

        Returns
        -------
        list of torch.Tensor
            One tensor per parameter, each shape ``(batch,)`` or
            ``(batch, dim)``.
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        params: list[torch.Tensor] = []
        var_offset = 0
        # Columns the recorded dims account for; any surplus belongs to
        # a single trailing vector whose event dim was not resolved at
        # construction. With more than one vector param the surplus is
        # unattributable, so the dims must be exact.
        known_total = sum(int(v) for k, v in self._param_spec if k == "var")
        surplus = x.shape[-1] - known_total
        if surplus > 0 and self._n_vector_params > 1:
            raise ValueError(
                "MixedInlineDistribution: cannot resolve parameters: the "
                f"stacked width {x.shape[-1]} exceeds the declared "
                f"variable dimension {known_total}, but there is more than "
                "one vector-typed parameter, so the surplus columns cannot "
                "be attributed to a single parameter"
            )
        for pos, (kind, value) in enumerate(self._param_spec):
            if kind == "lit":
                params.append(
                    torch.full(
                        (x.shape[0],), float(value), device=x.device, dtype=x.dtype
                    )
                )
                continue
            dim = int(value)
            if surplus > 0 and pos == self._last_vector_pos:
                # The single trailing under-counted vector absorbs the
                # surplus columns.
                dim += surplus
            is_vector = self._param_event_ranks[pos] >= 1
            if dim == 1 and not is_vector:
                params.append(x[..., var_offset])
            else:
                params.append(x[..., var_offset : var_offset + dim])
            var_offset += dim
        return params

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from the distribution.

        Parameters
        ----------
        x : torch.Tensor
            Stacked variable parameters.
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples from the distribution.
        """
        params = self._resolve_params(x)
        dist = self._dist_builder(params)
        if self._discrete:
            # Discrete samples are kept at the distribution's native
            # batch shape; downstream code unpacks as needed.
            return dist.sample(sample_shape).long()
        if getattr(dist, "has_rsample", True):
            result = dist.rsample(sample_shape)
        else:
            # Continuous but non-reparameterizable (e.g. VonMises).
            result = dist.sample(sample_shape)
        if result.dim() == 1:
            result = result.unsqueeze(-1)
        return result

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability under the distribution.

        Parameters
        ----------
        x : torch.Tensor
            Stacked variable parameters.
        y : torch.Tensor
            Observed values.

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape matches the broadcast of the
            distribution's batch shape with ``y``'s shape, with any
            event dimension summed out.
        """
        params = self._resolve_params(x)
        dist = self._dist_builder(params)
        # For discrete distributions over an integer support,
        # categorical-valued samples must remain integer-typed so
        # PyTorch's ``IntegerInterval`` validator accepts them.
        # Bernoulli / Boolean families accept float-valued samples,
        # but the integer family suite (Categorical, Binomial,
        # Geometric, NegativeBinomial, Poisson) refuses floats. Cast
        # only when the validator's support is a continuous
        # constraint.
        if self._discrete and y.dtype.is_floating_point:
            sup = self._support
            if isinstance(sup, _constraints._Boolean):
                y_in = y.float()
            else:
                y_in = y.long()
        else:
            y_in = y
        # Reconcile y's shape with the distribution's expected
        # event shape: append a trailing singleton if the
        # distribution carries an event dim that y lacks; drop a
        # trailing singleton from y when the distribution is scalar
        # and y carries an extra unit dim from upstream reshapes.
        event_dim = len(dist.event_shape)
        batch_dim = len(dist.batch_shape)
        target_dim = event_dim + batch_dim
        if not self._discrete:
            while y_in.dim() < target_dim:
                y_in = y_in.unsqueeze(-1)
            while y_in.dim() > target_dim and y_in.shape[-1] == 1:
                y_in = y_in.squeeze(-1)
        lp = dist.log_prob(y_in)
        # Sum out any explicit event axes that the distribution
        # already accounts for; PyTorch returns ``log_prob`` of
        # shape ``batch_shape`` so no extra reduction is needed.
        return lp


class DirectBernoulli(ContinuousMorphism):
    """Bernoulli using the input value directly as the probability.

    Unlike ``ConditionalBernoulli`` (which learns a mapping
    ``x -> logit -> prob`` via a neural net), this uses the input
    directly: ``Bernoulli(probs=x)``.

    This implements the PDS pattern ``Bern x`` where ``x`` is a
    continuous value in ``(0, 1)`` drawn from a prior like
    ``LogitNormal``.

    Parameters
    ----------
    domain : AnySpace
        Source space (typically UnitInterval).
    codomain : AnySpace
        Target FinSet of size 2.
    """

    def __init__(self, domain: AnySpace, codomain: AnySpace) -> None:
        super().__init__(domain, codomain)

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.boolean

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from Bernoulli(probs=x).

        Parameters
        ----------
        x : torch.Tensor
            Probabilities. Shape ``(batch,)`` or ``(batch, 1)``.
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Discrete samples in {0, 1}. Shape ``(*sample_shape, batch)``.
        """
        probs = x.squeeze(-1) if x.dim() > 1 else x
        probs = probs.clamp(EPS, 1.0 - EPS)
        dist = D.Bernoulli(probs=probs)
        return dist.sample(sample_shape).long()

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability of y under Bernoulli(probs=x).

        Parameters
        ----------
        x : torch.Tensor
            Probabilities.
        y : torch.Tensor
            Discrete outcomes in {0, 1}.

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape ``(batch,)``.
        """
        probs = x.squeeze(-1) if x.dim() > 1 else x
        probs = probs.clamp(EPS, 1.0 - EPS)
        dist = D.Bernoulli(probs=probs)
        return dist.log_prob(y.float())


class DirectNormal(ContinuousMorphism):
    """Normal using input values directly as (loc, scale).

    Input tensor has shape ``(batch, 2)`` where column 0 is loc and
    column 1 is scale.

    Parameters
    ----------
    domain : AnySpace
        Source space (provides loc and scale stacked).
    codomain : AnySpace
        Target space.
    """

    def __init__(self, domain: AnySpace, codomain: AnySpace) -> None:
        super().__init__(domain, codomain)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from Normal(loc, scale).

        Parameters
        ----------
        x : torch.Tensor
            Stacked ``(loc, scale)`` input. Shape ``(batch, 2)``.
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples. Shape ``(*sample_shape, batch, 1)``.
        """
        loc = x[..., 0]
        scale = x[..., 1].clamp(min=EPS)
        dist = D.Normal(loc, scale)
        result = dist.rsample(sample_shape)
        if result.dim() == 1:
            result = result.unsqueeze(-1)
        return result

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability under Normal(loc, scale).

        Parameters
        ----------
        x : torch.Tensor
            Stacked ``(loc, scale)`` input. Shape ``(batch, 2)``.
        y : torch.Tensor
            Observed values.

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape ``(batch,)``.
        """
        loc = x[..., 0]
        scale = x[..., 1].clamp(min=EPS)
        y_flat = y.squeeze(-1) if y.dim() > 1 else y
        dist = D.Normal(loc, scale)
        return dist.log_prob(y_flat)


class DirectTruncatedNormal(ContinuousMorphism):
    """TruncatedNormal with variable ``(mu, sigma)`` and fixed bounds.

    Input tensor has shape ``(batch, 2)`` where column 0 is mu and
    column 1 is sigma. The truncation bounds ``[low, high]`` are
    fixed at construction time.

    This implements the PDS response kernel where a noisy observation
    is drawn from a truncated normal centered on the latent state.

    Parameters
    ----------
    domain : AnySpace
        Source space (provides mu and sigma stacked).
    codomain : AnySpace
        Target space (bounded continuous).
    low : float
        Lower truncation bound.
    high : float
        Upper truncation bound.
    """

    def __init__(
        self, domain: AnySpace, codomain: AnySpace, low: float, high: float
    ) -> None:
        super().__init__(domain, codomain)
        self._low = low
        self._high = high

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.interval(self._low, self._high)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from TruncatedNormal(mu, sigma, low, high).

        Parameters
        ----------
        x : torch.Tensor
            Stacked ``(mu, sigma)`` input. Shape ``(batch, 2)``.
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples in ``[low, high]``. Shape ``(*sample_shape, batch, 1)``.
        """
        mu = x[..., 0]
        sigma = x[..., 1].clamp(min=EPS)
        normal = D.Normal(0, 1)
        alpha = normal.cdf((self._low - mu) / sigma)
        beta_cdf = normal.cdf((self._high - mu) / sigma)
        u = torch.rand(*sample_shape, *mu.shape, device=mu.device, dtype=mu.dtype)
        u_scaled = alpha + u * (beta_cdf - alpha)
        u_scaled = u_scaled.clamp(min=EPS, max=1.0 - EPS)
        result = normal.icdf(u_scaled) * sigma + mu
        if result.dim() == 1:
            result = result.unsqueeze(-1)
        return result

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability under TruncatedNormal(mu, sigma, low, high).

        Parameters
        ----------
        x : torch.Tensor
            Stacked ``(mu, sigma)`` input. Shape ``(batch, 2)``.
        y : torch.Tensor
            Observed values.

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape ``(batch,)``.
        """
        mu = x[..., 0]
        sigma = x[..., 1].clamp(min=EPS)
        y_flat = y.squeeze(-1) if y.dim() > 1 else y
        log_phi = (
            -0.5 * ((y_flat - mu) / sigma) ** 2
            - sigma.log()
            - 0.5 * math.log(2 * math.pi)
        )
        normal = D.Normal(0, 1)
        log_Z = torch.log(
            (
                normal.cdf((self._high - mu) / sigma)
                - normal.cdf((self._low - mu) / sigma)
            ).clamp(min=EPS)
        )
        return log_phi - log_Z


def make_fixed_logitnormal(
    mu: float, sigma: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed LogitNormal(mu, sigma) distribution.

    Parameters
    ----------
    mu : float
        Location parameter of the underlying normal.
    sigma : float
        Scale parameter of the underlying normal (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from LogitNormal(mu, sigma).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        mu_t = torch.full((batch, d), mu, device=device)
        sigma_t = torch.full((batch, d), sigma, device=device)
        base = D.Normal(mu_t, sigma_t)
        return D.TransformedDistribution(base, [D.SigmoidTransform()])

    return FixedDistribution(codomain, builder, support=_constraints.unit_interval)


def make_fixed_uniform(
    low: float, high: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed Uniform(low, high) distribution.

    Parameters
    ----------
    low : float
        Lower bound.
    high : float
        Upper bound.
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Uniform(low, high).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        low_t = torch.full((batch, d), low, device=device)
        high_t = torch.full((batch, d), high, device=device)
        return D.Uniform(low_t, high_t)

    return FixedDistribution(
        codomain, builder, support=_constraints.interval(low, high)
    )


def make_fixed_normal(
    loc: float, scale: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed Normal(loc, scale) distribution.

    Parameters
    ----------
    loc : float
        Mean.
    scale : float
        Standard deviation (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Normal(loc, scale).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        loc_t = torch.full((batch, d), loc, device=device)
        scale_t = torch.full((batch, d), scale, device=device)
        return D.Normal(loc_t, scale_t)

    return FixedDistribution(codomain, builder)


def make_fixed_bernoulli(prob: float, codomain: AnySpace) -> FixedDistribution:
    """Create a fixed Bernoulli(prob) distribution.

    Parameters
    ----------
    prob : float
        Success probability.
    codomain : AnySpace
        Output space (FinSet of size 2).

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Bernoulli(prob).
    """

    def builder(batch: int, device: torch.device) -> D.Distribution:
        probs_t = torch.full((batch,), prob, device=device)
        return D.Bernoulli(probs=probs_t)

    return FixedDistribution(
        codomain, builder, discrete=True, support=_constraints.boolean
    )


def make_fixed_beta(
    concentration1: float, concentration0: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed Beta(concentration1, concentration0) distribution.

    Parameters
    ----------
    concentration1 : float
        Alpha parameter.
    concentration0 : float
        Beta parameter.
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Beta(a, b).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        a = torch.full((batch, d), concentration1, device=device)
        b = torch.full((batch, d), concentration0, device=device)
        return D.Beta(a, b)

    return FixedDistribution(codomain, builder, support=_constraints.unit_interval)


def make_fixed_exponential(rate: float, codomain: AnySpace) -> FixedDistribution:
    """Create a fixed Exponential(rate) distribution.

    Parameters
    ----------
    rate : float
        Rate parameter (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Exponential(rate).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        rate_t = torch.full((batch, d), rate, device=device)
        return D.Exponential(rate_t)

    return FixedDistribution(codomain, builder, support=_constraints.positive)


def make_fixed_halfcauchy(scale: float, codomain: AnySpace) -> FixedDistribution:
    """Create a fixed HalfCauchy(scale) distribution.

    Parameters
    ----------
    scale : float
        Scale parameter (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from HalfCauchy(scale).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        scale_t = torch.full((batch, d), scale, device=device)
        return D.HalfCauchy(scale_t)

    return FixedDistribution(codomain, builder, support=_constraints.positive)


def make_fixed_halfnormal(scale: float, codomain: AnySpace) -> FixedDistribution:
    """Create a fixed HalfNormal(scale) distribution.

    Parameters
    ----------
    scale : float
        Scale parameter (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from HalfNormal(scale).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        scale_t = torch.full((batch, d), scale, device=device)
        return D.HalfNormal(scale_t)

    return FixedDistribution(codomain, builder, support=_constraints.positive)


def make_fixed_lognormal(
    loc: float, scale: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed LogNormal(loc, scale) distribution.

    Parameters
    ----------
    loc : float
        Mean of the underlying normal.
    scale : float
        Standard deviation of the underlying normal (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from LogNormal(loc, scale).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        loc_t = torch.full((batch, d), loc, device=device)
        scale_t = torch.full((batch, d), scale, device=device)
        return D.LogNormal(loc_t, scale_t)

    return FixedDistribution(codomain, builder, support=_constraints.positive)


def make_fixed_gamma(
    concentration: float, rate: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed Gamma(concentration, rate) distribution.

    Parameters
    ----------
    concentration : float
        Shape parameter (positive).
    rate : float
        Rate parameter (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Gamma(concentration, rate).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        conc_t = torch.full((batch, d), concentration, device=device)
        rate_t = torch.full((batch, d), rate, device=device)
        return D.Gamma(conc_t, rate_t)

    return FixedDistribution(codomain, builder, support=_constraints.positive)


def make_fixed_dirichlet(
    concentration: float | list[float], codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed Dirichlet(concentration) distribution on the simplex
    over the codomain's element axis.

    Parameters
    ----------
    concentration : float or list of float
        Concentration parameter. A scalar is broadcast to all simplex
        components (symmetric Dirichlet). A list / tuple of length
        equal to the codomain's dimension specifies per-component
        concentrations. Each entry must be positive.
    codomain : AnySpace
        Output space; its ``dim`` attribute gives the simplex
        dimension (the number of components).

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from the chosen Dirichlet on
        the (d-1)-simplex embedded in ``R^d``.
    """
    d = getattr(codomain, "dim", 1)
    if d < 2:
        raise ValueError(f"Dirichlet codomain must have dim >= 2, got dim={d}")
    if isinstance(concentration, (list, tuple)):
        if len(concentration) == 1:
            # Single-element sequence is treated as a scalar
            # (symmetric Dirichlet); broadcast to the codomain's
            # simplex dimension.
            conc_values = [float(concentration[0])] * d
        elif len(concentration) != d:
            raise ValueError(
                f"Dirichlet concentration vector has length "
                f"{len(concentration)} but codomain has dim={d}"
            )
        else:
            conc_values = [float(c) for c in concentration]
    else:
        conc_values = [float(concentration)] * d
    if any(c <= 0.0 for c in conc_values):
        raise ValueError(
            f"Dirichlet concentration must be positive componentwise, got {conc_values}"
        )
    conc_tuple = tuple(conc_values)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        conc_t = torch.tensor(conc_tuple, device=device).expand(batch, d)
        return D.Dirichlet(conc_t)

    return FixedDistribution(codomain, builder, support=_constraints.simplex)


# Families whose all-literal factory takes a single vector
# argument (a ``list[float]`` or ``tuple[float, ...]``) rather than
# one positional float per scalar parameter. The inline call site in
# ``make_inline_distribution`` re-bundles the splat-flattened
# literals into a list before invoking the factory.
_VECTOR_PARAM_FAMILIES: frozenset[str] = frozenset({"Dirichlet"})


_FIXED_FACTORIES: dict[str, tuple[tuple[str, ...], Callable]] = {
    "LogitNormal": (("mu", "sigma"), make_fixed_logitnormal),
    "Normal": (("loc", "scale"), make_fixed_normal),
    "Uniform": (("low", "high"), make_fixed_uniform),
    "Bernoulli": (("probs",), make_fixed_bernoulli),
    "Beta": (("concentration1", "concentration0"), make_fixed_beta),
    "Exponential": (("rate",), make_fixed_exponential),
    "HalfCauchy": (("scale",), make_fixed_halfcauchy),
    "HalfNormal": (("scale",), make_fixed_halfnormal),
    "LogNormal": (("loc", "scale"), make_fixed_lognormal),
    "Gamma": (("concentration", "rate"), make_fixed_gamma),
    "Dirichlet": (("concentration",), make_fixed_dirichlet),
}

# Per-family support constraints for inline distributions. Used when
# constructing a `MixedInlineDistribution` (which has at least one
# variable-bound parameter) so the resulting morphism advertises the
# correct constrained support to variational guides.
_FAMILY_SUPPORTS: dict[str, _constraints.Constraint] = {
    "Normal": _constraints.real,
    "Bernoulli": _constraints.boolean,
    "TruncatedNormal": _constraints.real,  # interval is set per-call
    "LogitNormal": _constraints.unit_interval,
    "Uniform": _constraints.real,  # interval is set per-call
    "Beta": _constraints.unit_interval,
    "Exponential": _constraints.positive,
    "HalfCauchy": _constraints.positive,
    "HalfNormal": _constraints.positive,
    "LogNormal": _constraints.positive,
    "Gamma": _constraints.positive,
    "Dirichlet": _constraints.simplex,
    "OrderedLogistic": _constraints.nonnegative_integer,
    "ZeroInflatedPoisson": _constraints.nonnegative_integer,
    "HurdlePoisson": _constraints.nonnegative_integer,
    "MixtureNormal": _constraints.real,
}


_OPERATOR_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "PointMass": ("value",),
    "Restrict": ("base", "low", "high"),
    "Truncate": ("base", "low", "high"),
    "Pushforward": ("base", "bijector"),
    "Mixture": ("weights", "components"),
    "Independent": ("base", "reinterpreted_batch_ndims"),
    "Normalize": ("base",),
}


def get_inline_param_names(family: str) -> tuple[str, ...] | None:
    """Get the ordered parameter names for an inline family.

    Parameters
    ----------
    family : str
        Distribution family name.

    Returns
    -------
    tuple[str, ...] or None
        Parameter names, or None if not an inline family.
    """
    if family in _OPERATOR_PARAM_NAMES:
        return _OPERATOR_PARAM_NAMES[family]
    if family in _FIXED_FACTORIES:
        return _FIXED_FACTORIES[family][0]
    if family in _FAMILY_BUILDERS:
        return _FAMILY_BUILDERS[family][0]
    if family == "TruncatedNormal":
        return ("mu", "sigma", "low", "high")
    return None


def _normal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Normal from [loc, scale]."""
    return D.Normal(params[0], params[1].clamp(min=EPS))


def _bernoulli_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Bernoulli from [probs]."""
    return D.Bernoulli(probs=params[0].clamp(EPS, 1.0 - EPS))


def _truncated_normal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build TruncatedNormal from [mu, sigma, low, high].

    Returns a proxy object with rsample and log_prob that performs
    inverse-CDF truncated normal sampling.
    """
    mu, sigma, low_t, high_t = params
    sigma = sigma.clamp(min=EPS)
    low = float(low_t.flatten()[0])
    high = float(high_t.flatten()[0])

    class _TruncNorm:
        """Minimal truncated-normal distribution interface.

        Exposes the same ``event_shape`` and ``batch_shape``
        attributes the generic `torch.distributions` interface
        carries, so callers walking the family registry's
        ``log_prob`` path treat it uniformly with stock families.
        """

        event_shape: torch.Size = torch.Size()
        batch_shape: torch.Size = mu.shape

        def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
            normal = D.Normal(0, 1)
            alpha = normal.cdf((low - mu) / sigma)
            beta_cdf = normal.cdf((high - mu) / sigma)
            u = torch.rand(*sample_shape, *mu.shape, device=mu.device, dtype=mu.dtype)
            u_scaled = alpha + u * (beta_cdf - alpha)
            u_scaled = u_scaled.clamp(min=EPS, max=1.0 - EPS)
            return normal.icdf(u_scaled) * sigma + mu

        def log_prob(self, y: torch.Tensor) -> torch.Tensor:
            log_phi = (
                -0.5 * ((y - mu) / sigma) ** 2
                - sigma.log()
                - 0.5 * math.log(2 * math.pi)
            )
            normal = D.Normal(0, 1)
            log_Z = torch.log(
                (
                    normal.cdf((high - mu) / sigma) - normal.cdf((low - mu) / sigma)
                ).clamp(min=EPS)
            )
            return log_phi - log_Z

        def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
            with torch.no_grad():
                return self.rsample(sample_shape)

    return _TruncNorm()


def _logitnormal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build LogitNormal from [mu, sigma]."""
    base = D.Normal(params[0], params[1].clamp(min=EPS))
    return D.TransformedDistribution(base, [D.SigmoidTransform()])


def _uniform_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Uniform from [low, high]."""
    return D.Uniform(params[0], params[1])


def _beta_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Beta from [concentration1, concentration0]."""
    return D.Beta(params[0].clamp(min=EPS), params[1].clamp(min=EPS))


def _exponential_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Exponential from [rate]."""
    return D.Exponential(params[0].clamp(min=EPS))


def _halfcauchy_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build HalfCauchy from [scale]."""
    return D.HalfCauchy(params[0].clamp(min=EPS))


def _halfnormal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build HalfNormal from [scale]."""
    return D.HalfNormal(params[0].clamp(min=EPS))


def _lognormal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build LogNormal from [loc, scale]."""
    return D.LogNormal(params[0], params[1].clamp(min=EPS))


def _gamma_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Gamma from [concentration, rate]."""
    return D.Gamma(params[0].clamp(min=EPS), params[1].clamp(min=EPS))


def _dirichlet_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Dirichlet from [concentration]. The concentration tensor is
    ``(batch,)`` for scalar input (broadcast to the simplex dimension at
    sample time) or ``(batch, d)`` for a per-fibre vector."""
    return D.Dirichlet(params[0].clamp(min=EPS))


def _zero_inflated_poisson_builder(
    params: list[torch.Tensor],
) -> D.Distribution:
    """Build `ZeroInflatedPoisson` from ``[zero_prob, rate]``."""
    return ZeroInflatedPoisson(
        params[0].clamp(EPS, 1.0 - EPS),
        params[1].clamp(min=EPS),
    )


def _hurdle_poisson_builder(
    params: list[torch.Tensor],
) -> D.Distribution:
    """Build `HurdlePoisson` from ``[zero_prob, rate]``."""
    return HurdlePoisson(
        params[0].clamp(EPS, 1.0 - EPS),
        params[1].clamp(min=EPS),
    )


def _mixture_normal_builder(
    params: list[torch.Tensor],
) -> D.Distribution:
    """Build `MixtureNormal` from ``[weights, loc, scale]``.

    Each parameter is a per-row vector of length ``K`` (number of
    mixture components). The DSL surfaces this as:

        observe y : Resp <- MixtureNormal(weights, locs, scales)

    with `weights`, `locs`, `scales` declared as ``K``-dim variables
    (e.g. sampled per-row vectors or host-supplied data tensors).
    """
    return MixtureNormal(
        params[0],
        params[1],
        params[2].clamp(min=EPS),
    )


def _ordered_logistic_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build OrderedLogistic from ``[predictor, cutpoints]``.

    ``predictor`` is shape ``(batch,)``; ``cutpoints`` is either
    ``(batch, K-1)`` (per-row thresholds, the ordinal-mixed-model
    case) or a length-``K-1`` vector broadcast over the batch.

    Cutpoints are not sorted here; the caller (typically a learned
    monotonic transform or a fixed list) is responsible for ordering.
    Unsorted cutpoints produce negative differences and the
    distribution's log-prob clamp will return ``log(0)`` for the
    affected rows, surfacing the bug at training time.
    """
    return OrderedLogistic(params[0], params[1])


_FAMILY_BUILDERS: dict[str, tuple[tuple[str, ...], Callable, bool]] = {
    "Normal": (("loc", "scale"), _normal_builder, False),
    "Bernoulli": (("probs",), _bernoulli_builder, True),
    "TruncatedNormal": (
        ("mu", "sigma", "low", "high"),
        _truncated_normal_builder,
        False,
    ),
    "LogitNormal": (("mu", "sigma"), _logitnormal_builder, False),
    "Uniform": (("low", "high"), _uniform_builder, False),
    "Beta": (("concentration1", "concentration0"), _beta_builder, False),
    "Exponential": (("rate",), _exponential_builder, False),
    "HalfCauchy": (("scale",), _halfcauchy_builder, False),
    "HalfNormal": (("scale",), _halfnormal_builder, False),
    "LogNormal": (("loc", "scale"), _lognormal_builder, False),
    "Gamma": (("concentration", "rate"), _gamma_builder, False),
    "Dirichlet": (("concentration",), _dirichlet_builder, False),
    "OrderedLogistic": (
        ("predictor", "cutpoints"),
        _ordered_logistic_builder,
        True,
    ),
    "ZeroInflatedPoisson": (
        ("zero_prob", "rate"),
        _zero_inflated_poisson_builder,
        True,
    ),
    "HurdlePoisson": (
        ("zero_prob", "rate"),
        _hurdle_poisson_builder,
        True,
    ),
    "MixtureNormal": (
        ("weights", "loc", "scale"),
        _mixture_normal_builder,
        False,
    ),
}


# Per-family per-parameter event ranks for the inline observe / draw
# surface. A rank-0 parameter is a per-row scalar (participates in
# the standard `_stack_tensors` per-row concatenation). A rank-1
# parameter is a vector whose length is a distribution property
# (cutpoints for OrderedLogistic, weights for MixtureNormal, ...).
# Vector params may arrive as either a shared 1D tensor of shape
# ``(D,)`` broadcast across the plate, or a per-row 2D tensor of
# shape ``(batch, D)``. Families not listed default to all-rank-0.
_PARAM_EVENT_RANKS: dict[str, tuple[int, ...]] = {
    "OrderedLogistic": (0, 1),  # predictor scalar, cutpoints vector
    "MixtureNormal": (1, 1, 1),  # per-component vectors
    "ZeroInflatedPoisson": (0, 0),
    "HurdlePoisson": (0, 0),
}


# Operator families that take Distribution-valued / list-valued args
# rather than scalar / vector parameters. Dispatched separately by
# `make_inline_distribution`; see the compositional measure algebra
# in `quivers.continuous.measure`.
_OPERATOR_FAMILIES: frozenset[str] = frozenset(
    {
        "Mixture",
        "Restrict",
        "Truncate",  # alias of Restrict
        "Pushforward",
        "PointMass",
        "Independent",
        "Normalize",
    }
)


def _unwrap_draw_arg(arg):
    """Translate a `DrawArg` tagged variant back into the plain
    Python representation the inline machinery already understands.

    `DrawArgName` -> `str`, `DrawArgScalar` -> `float`, `DrawArgIndex`
    -> `DrawArgIndex` (passed through structurally so downstream
    var-name collection can pattern-match on the `kind`).
    `DrawArgDist` and `DrawArgList` round-trip unchanged; the
    operator dispatch above handles them.
    """
    if isinstance(arg, DrawArgName):
        return arg.text
    if isinstance(arg, DrawArgScalar):
        return arg.value
    return arg


def _normalize_inline_args(args):
    """Map a tuple of (possibly tagged) draw args to the plain
    `str | float | DrawArgDist | DrawArgList` form. Operator
    dispatch sees the tagged compound shapes; the rest of the
    inline path sees bare strings and floats.
    """
    return tuple(_unwrap_draw_arg(a) for a in args)


def _eval_draw_arg_value(
    arg,
    variable_types,
):
    """Evaluate a draw-arg to a Python value usable inside the
    measure-algebra constructors. Scalars become floats; numeric
    `DrawArgList`s become lists of floats; `DrawArgDist`s become
    `Distribution` instances built recursively.

    Identifiers stay as strings; the operator family decides what
    to do with them (e.g. `PointMass(x)` where `x` is an identifier
    must be resolved at trace time, which the operator dispatch
    handles by raising for the current implementation).
    """
    if isinstance(arg, DrawArgScalar):
        return arg.value
    if isinstance(arg, DrawArgName):
        return arg.text
    if isinstance(arg, DrawArgList):
        return [_eval_draw_arg_value(item, variable_types) for item in arg.items]
    if isinstance(arg, DrawArgDist):
        return _build_inner_distribution(arg, variable_types)
    if isinstance(arg, (int, float)):
        return float(arg)
    if isinstance(arg, str):
        return arg
    raise TypeError(f"_eval_draw_arg_value: unexpected arg type {type(arg).__name__}")


def _build_inner_distribution(
    arg: DrawArgDist,
    variable_types,
):
    """Recursively build a `torch.distributions.Distribution` (or
    `Measure`) from a `DrawArgDist`, dispatching on the family name.
    Used by the compositional measure algebra to evaluate
    distribution-valued arguments before passing them to the outer
    family's constructor.

    Recognises every operator in `_OPERATOR_FAMILIES` plus the base
    families registered in `_FIXED_FACTORIES` and `_FAMILY_BUILDERS`
    (with all-literal args; variable-bound base families inside a
    compositional expression are deferred to the operator's runtime
    evaluation).
    """
    family = arg.family
    if family in _OPERATOR_FAMILIES:
        return _build_operator_value(family, arg.args, variable_types)
    if family in _BASE_DISTRIBUTION_BUILDERS:
        evaluated_args = [_eval_draw_arg_value(a, variable_types) for a in arg.args]
        if any(isinstance(v, str) for v in evaluated_args):
            raise ValueError(
                f"Inline distribution {family}({arg.args}): "
                "variable-bound base families inside a compositional "
                "expression require the operator's runtime evaluation; "
                "use the standalone form or a let-binding."
            )
        return _BASE_DISTRIBUTION_BUILDERS[family](evaluated_args)
    raise ValueError(
        f"unknown inline family {family!r} in compositional "
        f"expression; operators: {sorted(_OPERATOR_FAMILIES)}; bases: "
        f"{sorted(_BASE_DISTRIBUTION_BUILDERS)}"
    )


# Closed-form constructors for base distributions used inside
# compositional expressions. Each takes a list of evaluated args
# (floats, after recursing through `_eval_draw_arg_value`) and
# returns a `torch.distributions.Distribution` instance. Only the
# families that compose naturally with the operator algebra are
# listed; user-supplied families can be registered here at runtime.
_BASE_DISTRIBUTION_BUILDERS: dict = {
    "Normal": lambda a: D.Normal(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "Cauchy": lambda a: D.Cauchy(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "Laplace": lambda a: D.Laplace(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "StudentT": lambda a: D.StudentT(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]) if len(a) > 1 else torch.tensor(0.0),
        torch.as_tensor(a[2]) if len(a) > 2 else torch.tensor(1.0),
    ),
    "Beta": lambda a: D.Beta(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "Gamma": lambda a: D.Gamma(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "Exponential": lambda a: D.Exponential(
        torch.as_tensor(a[0]),
    ),
    "Poisson": lambda a: D.Poisson(torch.as_tensor(a[0])),
    "Bernoulli": lambda a: D.Bernoulli(probs=torch.as_tensor(a[0])),
    "Uniform": lambda a: D.Uniform(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "HalfNormal": lambda a: D.HalfNormal(torch.as_tensor(a[0])),
    "HalfCauchy": lambda a: D.HalfCauchy(torch.as_tensor(a[0])),
    "LogNormal": lambda a: D.LogNormal(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
}


def _build_operator_value(
    family: str,
    args: tuple,
    variable_types,
):
    """Evaluate an operator-family call inside a compositional
    expression. Returns a `Distribution` / `Measure` value usable as
    a parameter to an enclosing operator.
    """
    evaluated = [_eval_draw_arg_value(a, variable_types) for a in args]
    if family == "PointMass":
        if len(evaluated) != 1:
            raise ValueError(f"PointMass: expected 1 arg, got {len(evaluated)}")
        return _MeasurePointMass(torch.as_tensor(evaluated[0]))
    if family in ("Restrict", "Truncate"):
        if len(evaluated) < 2 or len(evaluated) > 3:
            raise ValueError(
                f"{family}: expected (base[, low, high]), got {len(evaluated)} args"
            )
        base = evaluated[0]
        if not hasattr(base, "log_prob"):
            raise ValueError(
                f"{family}: first arg must be a Distribution, got {type(base).__name__}"
            )
        low = evaluated[1] if len(evaluated) >= 2 else None
        high = evaluated[2] if len(evaluated) >= 3 else None
        low_t = torch.as_tensor(low) if low is not None else None
        high_t = torch.as_tensor(high) if high is not None else None
        return _MeasureRestrict(base, low=low_t, high=high_t)
    if family == "Pushforward":
        if len(evaluated) != 2:
            raise ValueError(
                f"Pushforward: expected (base, bijector), got {len(evaluated)} args"
            )
        from quivers.continuous.bijectors import Bijector

        base, bij = evaluated
        if isinstance(bij, str):
            from quivers.continuous import bijectors as _bj

            bij_cls = getattr(_bj, bij, None)
            if bij_cls is None:
                raise ValueError(
                    f"Pushforward: unknown bijector {bij!r}; "
                    f"register or use the named operators."
                )
            bij = bij_cls()
        if not isinstance(bij, Bijector):
            raise ValueError(
                f"Pushforward: second arg must be a Bijector, got {type(bij).__name__}"
            )
        return _MeasurePushforward(base, bij)
    if family == "Mixture":
        if len(evaluated) != 2:
            raise ValueError(
                f"Mixture: expected (weights, components), got {len(evaluated)} args"
            )
        weights_raw, components_raw = evaluated
        weights = torch.as_tensor(weights_raw, dtype=torch.float32)
        if not isinstance(components_raw, list):
            raise ValueError("Mixture: components must be a list literal")
        return _MeasureMixture(weights, components_raw)
    if family == "Independent":
        if len(evaluated) != 2:
            raise ValueError(
                "Independent: expected (base, reinterpreted_batch_ndims), "
                f"got {len(evaluated)} args"
            )
        base, n = evaluated
        return _MeasureIndependent(base, int(n))
    if family == "Normalize":
        if len(evaluated) != 1:
            raise ValueError(f"Normalize: expected 1 arg, got {len(evaluated)}")
        return _MeasureNormalize(evaluated[0])
    raise ValueError(f"unknown operator family {family!r}")


def _make_operator_distribution(
    family: str,
    args,
    codomain: AnySpace,
    variable_types,
):
    """Build a `FixedDistribution` wrapper around a compositional
    measure expression. The expression has no free QVR variables;
    all arguments are evaluated at compile time to a closed-form
    `Distribution` / `Measure` instance.
    """
    fake_dist = _build_operator_value(family, tuple(args), variable_types)
    support = getattr(fake_dist, "support", _constraints.real)
    discrete = isinstance(support, type(_constraints.nonnegative_integer))

    def builder(batch: int, device: torch.device):
        return fake_dist

    morph = FixedDistribution(
        codomain,
        builder,
        discrete=discrete,
        support=support,
    )
    return (morph, None)


def make_inline_distribution(
    family: str,
    args: tuple,
    codomain: AnySpace,
    variable_types: dict[str, AnySpace] | None = None,
) -> tuple[ContinuousMorphism, tuple[str, ...] | None]:
    """Create an inline distribution from family name and mixed args.

    Handles any combination of literal and variable arguments for any
    registered distribution family. The general mechanism:

    1. All literals → ``FixedDistribution`` (no variable input)
    2. Any variables → ``MixedInlineDistribution`` with a param_spec
       that records which positions are literals vs. variable slices

    Parameters
    ----------
    family : str
        Distribution family name.
    args : tuple of str | float
        Arguments from the DSL. Strings are variable names,
        floats are literal values.
    codomain : AnySpace
        The output space for the distribution.
    variable_types : dict or None
        Mapping of variable names to their space types (for domain
        construction of direct distributions).

    Returns
    -------
    tuple of (ContinuousMorphism, tuple[str, ...] | None)
        The inline distribution morphism, and the variable names
        to pass as step input (None = use program input).
    """
    if family in _OPERATOR_FAMILIES:
        return _make_operator_distribution(
            family,
            args,
            codomain,
            variable_types,
        )
    args = _normalize_inline_args(args)
    # A `DrawArgIndex` (structural bracket-indexed ref) counts as a
    # variable reference for the purposes of picking the fixed vs
    # mixed factory dispatch, alongside bare identifier strings.
    var_names = [
        a for a in args if isinstance(a, str) or getattr(a, "kind", None) == "index"
    ]
    if not var_names:
        all_floats = [float(a) for a in args]
        if family in _FIXED_FACTORIES:
            _, factory = _FIXED_FACTORIES[family]
            # Vector-parameter families take a single ``list[float]``
            # / ``tuple[float, ...]`` argument rather than splatting
            # the literals. The parser surfaces ``Dirichlet([1, 2, 3])``
            # as three positional float args (the grammar's draw-arg
            # list flattens any bracket-bounded numeric sequence), so
            # we re-bundle here when the factory's documented contract
            # is a single vector argument.
            if family in _VECTOR_PARAM_FAMILIES:
                morph = factory(all_floats, codomain)
            else:
                morph = factory(*all_floats, codomain)
            return (morph, None)
        raise ValueError(f"no fixed factory for inline family {family!r}")
    if family not in _FAMILY_BUILDERS:
        raise ValueError(
            f"no builder for inline family {family!r} with variable arguments"
        )
    param_names, dist_builder, discrete = _FAMILY_BUILDERS[family]
    if len(args) != len(param_names):
        raise ValueError(
            f"inline {family} expects {len(param_names)} args ({', '.join(param_names)}), got {len(args)}"
        )
    param_spec: list[tuple[str, int | float]] = []
    var_name_order: list[str] = []
    total_var_dim = 0
    for i, arg in enumerate(args):
        if isinstance(arg, (int, float)):
            param_spec.append(("lit", float(arg)))
        else:
            var_dim = 1
            # `DrawArgIndex` carries the base identifier under
            # `arg.name`; the compiled step spec still references
            # the base tensor via `_lookup_arg`, so the variable-
            # type lookup uses that name.
            lookup_name = arg.name if getattr(arg, "kind", None) == "index" else arg
            if variable_types and lookup_name in variable_types:
                vtype = variable_types[lookup_name]
                var_dim = getattr(vtype, "dim", 1)
            param_spec.append(("var", var_dim))
            var_name_order.append(arg)
            total_var_dim += var_dim
    if total_var_dim == 0:
        domain = Euclidean(name="_inline_domain", dim=1)
    elif len(var_name_order) == 1 and variable_types:
        vtype = variable_types.get(var_name_order[0])
        domain = (
            vtype
            if vtype is not None
            else Euclidean(name="_inline_domain", dim=total_var_dim)
        )
    else:
        domain = _infer_domain(var_name_order, variable_types)
    fam_support: _constraints.Constraint = _FAMILY_SUPPORTS.get(
        family, _constraints.real
    )
    # TruncatedNormal / Uniform encode bounded supports via two literal
    # arguments; if both bounds are literal we can specialize the
    # constraint to the matching `interval`. Otherwise we fall
    # back to ``real`` (the closest correct guide-side approximation).
    if family in ("TruncatedNormal", "Uniform"):
        lit_args: list[float] = []
        for kind, value in param_spec:
            if kind == "lit":
                lit_args.append(float(value))
        if family == "Uniform" and len(lit_args) == 2:
            fam_support = _constraints.interval(lit_args[0], lit_args[1])
        elif family == "TruncatedNormal" and len(lit_args) >= 2:
            fam_support = _constraints.interval(lit_args[-2], lit_args[-1])
    fam_event_ranks = _PARAM_EVENT_RANKS.get(
        family,
        tuple(0 for _ in param_names),
    )
    morph = MixedInlineDistribution(
        domain,
        codomain,
        param_spec,
        dist_builder,
        discrete,
        support=fam_support,
        param_event_ranks=fam_event_ranks,
    )
    return (morph, tuple(var_name_order))


def _infer_domain(
    var_names: list[str], variable_types: dict[str, AnySpace] | None
) -> AnySpace:
    """Infer a domain space from variable types.

    Parameters
    ----------
    var_names : list[str]
        Variable names used as input.
    variable_types : dict or None
        Known variable types.

    Returns
    -------
    AnySpace
        The inferred domain.
    """
    if variable_types is None or not var_names:
        return Euclidean(name="_inline_domain", dim=len(var_names))
    if len(var_names) == 1:
        vtype = variable_types.get(var_names[0])
        if vtype is not None:
            return vtype
        return Euclidean(name="_inline_domain", dim=1)
    from quivers.core.objects import ProductSet
    from quivers.continuous.spaces import ContinuousSpace, ProductSpace

    components = []
    for vn in var_names:
        vtype = variable_types.get(vn)
        if vtype is not None:
            components.append(vtype)
        else:
            components.append(Euclidean(name=f"_inline_{vn}", dim=1))
    if any((isinstance(c, ContinuousSpace) for c in components)):
        return ProductSpace(components=tuple(components))
    return ProductSet(components=tuple(components))


# ---------------------------------------------------------------------------
# Auto-generated inline support from `FAMILY_REGISTRY`.
#
# For every registered family with ``output_kind == "independent"``
# that isn't already in the hand-written dicts, build a generic
# `FixedDistribution` factory and a matching mixed-mode
# builder.  This is the architectural seam that closes the gap
# between the conditional path and the inline path: a family
# declared once via [`quivers.continuous.family_spec.register`][quivers.continuous.family_spec.register]
# automatically becomes usable in DSL ``F(args)`` syntax.
# ---------------------------------------------------------------------------


def _build_generic_fixed_factory(spec: FamilySpec) -> Callable:
    """Return a fixed-factory callable matching the
    ``make_fixed_X(literal_args..., codomain)`` calling convention.

    The factory broadcasts each scalar literal to ``(batch, d)``
    where ``d = getattr(codomain, "dim", 1)`` and clamps the value
    by the per-parameter inline clamp before passing it to the
    underlying torch distribution.  Discrete families have their
    output cast to ``long`` by `FixedDistribution`.
    """

    def factory(*all_args) -> FixedDistribution:
        # Match the existing factory convention from
        # ``make_inline_distribution``: positional literals followed
        # by the codomain as the final positional argument.
        if len(all_args) != len(spec.params) + 1:
            raise ValueError(
                f"inline {spec.name} expects "
                f"{len(spec.params)} literal args "
                f"({', '.join(spec.param_names)}) plus codomain; "
                f"got {len(all_args)} args"
            )
        *literal_args, codomain = all_args
        d = getattr(codomain, "dim", 1)
        clamped_floats = []
        for param, val in zip(spec.params, literal_args):
            f = float(val)
            clamped_floats.append(float(param.inline_clamp(torch.tensor(f)).item()))

        def builder(batch: int, device: torch.device) -> D.Distribution:
            kwargs = {}
            for param, val in zip(spec.params, clamped_floats):
                kwargs[param.name] = torch.full((batch, d), val, device=device)
            return spec.dist_class(**kwargs)

        return FixedDistribution(
            codomain,
            builder,
            discrete=spec.discrete,
            support=spec.support,
        )

    return factory


def _build_generic_mixed_builder(spec: FamilySpec) -> Callable:
    """Return a mixed-mode builder accepting a list of resolved
    parameter tensors and returning a torch distribution.

    Each tensor is passed through the per-parameter inline clamp
    before construction.
    """

    def builder(params: list[torch.Tensor]) -> D.Distribution:
        if len(params) != len(spec.params):
            raise ValueError(
                f"inline {spec.name} mixed builder expects "
                f"{len(spec.params)} tensors; got {len(params)}"
            )
        kwargs = {}
        for param, t in zip(spec.params, params):
            kwargs[param.name] = param.inline_clamp(t)
        return spec.dist_class(**kwargs)

    return builder


def _auto_register_inline(spec: FamilySpec) -> None:
    """Populate the inline dicts for ``spec`` unless a hand-written
    entry is already present.

    Output-kind handling:

    * ``independent`` families register both a fixed factory (all
      literal scalar args broadcast to ``(batch, dim)``) and a mixed
      builder.
    * ``categorical`` / ``vector`` / ``mvn`` / ``matrix`` families
      register only the mixed builder. Their parameters are vector-
      or matrix-shaped, so the literal-broadcast path doesn't make
      sense; calling ``F(args)`` with at least one variable argument
      goes through the mixed path, which forwards the resolved
      parameter tensor straight to the underlying torch distribution
      constructor.
    """
    if spec.name in _FIXED_FACTORIES or spec.name in _FAMILY_BUILDERS:
        return  # hand-written entry takes precedence
    if spec.output_kind not in {
        "independent",
        "categorical",
        "vector",
        "mvn",
        "matrix",
    }:
        return
    if spec.output_kind == "independent":
        factory = _build_generic_fixed_factory(spec)
        _FIXED_FACTORIES[spec.name] = (spec.param_names, factory)
    builder = _build_generic_mixed_builder(spec)
    _FAMILY_BUILDERS[spec.name] = (spec.param_names, builder, spec.discrete)
    _FAMILY_SUPPORTS[spec.name] = spec.support


# Ensure [`quivers.continuous.families`][quivers.continuous.families] has registered every
# family before we walk the registry. Imported here rather than at
# module top to break the import cycle (families.py imports inline
# indirectly via the DSL compiler).
from quivers.continuous import families as _families  # noqa: E402, F401

for _spec in FAMILY_REGISTRY.values():
    _auto_register_inline(_spec)


def reload_inline_registry() -> None:
    """Refresh the inline dicts from
    `quivers.continuous.family_spec.FAMILY_REGISTRY`. Useful
    when new families are registered after `inline` has
    imported (test plugins, downstream extensions)."""
    for spec in FAMILY_REGISTRY.values():
        _auto_register_inline(spec)
