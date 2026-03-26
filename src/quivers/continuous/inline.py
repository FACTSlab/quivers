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
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.distributions as D

from quivers.core.objects import Unit
from quivers.continuous.spaces import Euclidean
from quivers.continuous.morphisms import ContinuousMorphism, AnySpace
from quivers.core._util import EPS


# ============================================================================
# fixed-parameter distributions
# ============================================================================


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
    """

    def __init__(
        self,
        codomain: AnySpace,
        make_dist: Callable,
        discrete: bool = False,
    ) -> None:
        super().__init__(Unit, codomain)
        self._make_dist_fn = make_dist
        self._discrete = discrete

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
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

        return dist.rsample(sample_shape)

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

        # sum over event dimensions if present
        if lp.dim() > 1:
            return lp.sum(dim=-1)

        return lp


# ============================================================================
# direct-parameter distributions
# ============================================================================


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
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        param_spec: list[tuple[str, int | float]],
        dist_builder: Callable,
        discrete: bool = False,
    ) -> None:
        super().__init__(domain, codomain)
        self._param_spec = param_spec
        self._dist_builder = dist_builder
        self._discrete = discrete

    def _resolve_params(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Reconstruct full parameter list from input + stored literals.

        Parameters
        ----------
        x : torch.Tensor
            Stacked variable parameters. Shape ``(batch, total_var_dim)``.
            May be ``(batch,)`` if total variable dimension is 1.

        Returns
        -------
        list of torch.Tensor
            One tensor per parameter, each shape ``(batch,)`` or
            ``(batch, dim)``.
        """
        # ensure x is at least 2D for consistent slicing
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        params = []
        var_offset = 0

        for kind, value in self._param_spec:
            if kind == "lit":
                # broadcast literal to match batch dimension
                lit_val = torch.full(
                    (x.shape[0],),
                    float(value),
                    device=x.device,
                    dtype=x.dtype,
                )
                params.append(lit_val)

            else:
                # variable: slice from stacked input
                dim = int(value)

                if dim == 1:
                    params.append(x[..., var_offset])

                else:
                    params.append(x[..., var_offset : var_offset + dim])

                var_offset += dim

        return params

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
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
            return dist.sample(sample_shape).long()

        result = dist.rsample(sample_shape)

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
            Log-probabilities. Shape ``(batch,)``.
        """
        params = self._resolve_params(x)
        dist = self._dist_builder(params)
        lp = dist.log_prob(
            y.float() if self._discrete else (y.squeeze(-1) if y.dim() > 1 else y)
        )

        if lp.dim() > 1:
            return lp.sum(dim=-1)

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

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
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
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
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
        self,
        domain: AnySpace,
        codomain: AnySpace,
        low: float,
        high: float,
    ) -> None:
        super().__init__(domain, codomain)
        self._low = low
        self._high = high

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
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

        u = torch.rand(
            *sample_shape,
            *mu.shape,
            device=mu.device,
            dtype=mu.dtype,
        )
        u_scaled = alpha + u * (beta_cdf - alpha)
        u_scaled = u_scaled.clamp(min=EPS, max=1.0 - EPS)

        result = normal.icdf(u_scaled) * sigma + mu

        # ensure 2D output for continuous codomain consistency
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


# ============================================================================
# factory functions for creating inline distributions from DSL args
# ============================================================================


def make_fixed_logitnormal(
    mu: float,
    sigma: float,
    codomain: AnySpace,
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

    return FixedDistribution(codomain, builder)


def make_fixed_uniform(
    low: float,
    high: float,
    codomain: AnySpace,
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

    return FixedDistribution(codomain, builder)


def make_fixed_normal(
    loc: float,
    scale: float,
    codomain: AnySpace,
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


def make_fixed_bernoulli(
    prob: float,
    codomain: AnySpace,
) -> FixedDistribution:
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

    return FixedDistribution(codomain, builder, discrete=True)


def make_fixed_beta(
    concentration1: float,
    concentration0: float,
    codomain: AnySpace,
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

    return FixedDistribution(codomain, builder)


def make_fixed_exponential(
    rate: float,
    codomain: AnySpace,
) -> FixedDistribution:
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

    return FixedDistribution(codomain, builder)


def make_fixed_halfcauchy(
    scale: float,
    codomain: AnySpace,
) -> FixedDistribution:
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

    return FixedDistribution(codomain, builder)


def make_fixed_halfnormal(
    scale: float,
    codomain: AnySpace,
) -> FixedDistribution:
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

    return FixedDistribution(codomain, builder)


def make_fixed_lognormal(
    loc: float,
    scale: float,
    codomain: AnySpace,
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

    return FixedDistribution(codomain, builder)


def make_fixed_gamma(
    concentration: float,
    rate: float,
    codomain: AnySpace,
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

    return FixedDistribution(codomain, builder)


# ============================================================================
# inline family registry: maps family names -> (param_names, factory)
# ============================================================================

# maps family name -> (ordered param names, factory function)
# factory is called with (*float_values, codomain) for all-fixed case
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
    if family in _FIXED_FACTORIES:
        return _FIXED_FACTORIES[family][0]

    # families with direct-variable support but no all-fixed factory
    if family == "TruncatedNormal":
        return ("mu", "sigma", "low", "high")

    return None


def _normal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Normal from [loc, scale]."""
    return D.Normal(params[0], params[1].clamp(min=EPS))


def _bernoulli_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Bernoulli from [probs]."""
    return D.Bernoulli(probs=params[0].clamp(EPS, 1.0 - EPS))


def _truncated_normal_builder(
    params: list[torch.Tensor],
) -> D.Distribution:
    """Build TruncatedNormal from [mu, sigma, low, high].

    Returns a proxy object with rsample and log_prob that performs
    inverse-CDF truncated normal sampling.
    """
    mu, sigma, low_t, high_t = params
    sigma = sigma.clamp(min=EPS)
    # extract scalar bounds from broadcast tensors
    low = float(low_t.flatten()[0])
    high = float(high_t.flatten()[0])

    class _TruncNorm:
        """Minimal truncated-normal distribution interface."""

        def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
            normal = D.Normal(0, 1)
            alpha = normal.cdf((low - mu) / sigma)
            beta_cdf = normal.cdf((high - mu) / sigma)
            u = torch.rand(
                *sample_shape,
                *mu.shape,
                device=mu.device,
                dtype=mu.dtype,
            )
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

    return _TruncNorm()  # type: ignore[return-value]


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


# maps family name -> (param_names, builder, discrete)
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
    "Beta": (
        ("concentration1", "concentration0"),
        _beta_builder,
        False,
    ),
    "Exponential": (("rate",), _exponential_builder, False),
    "HalfCauchy": (("scale",), _halfcauchy_builder, False),
    "HalfNormal": (("scale",), _halfnormal_builder, False),
    "LogNormal": (("loc", "scale"), _lognormal_builder, False),
    "Gamma": (("concentration", "rate"), _gamma_builder, False),
}


def make_inline_distribution(
    family: str,
    args: tuple[str | float, ...],
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
    var_names = [a for a in args if isinstance(a, str)]

    # all literals: create fixed distribution (unchanged)
    if not var_names:
        all_floats = [float(a) for a in args]

        if family in _FIXED_FACTORIES:
            _, factory = _FIXED_FACTORIES[family]
            morph = factory(*all_floats, codomain)
            return morph, None

        raise ValueError(f"no fixed factory for inline family {family!r}")

    # has variable args: use the general MixedInlineDistribution
    if family not in _FAMILY_BUILDERS:
        raise ValueError(
            f"no builder for inline family {family!r} with variable arguments"
        )

    param_names, dist_builder, discrete = _FAMILY_BUILDERS[family]

    if len(args) != len(param_names):
        raise ValueError(
            f"inline {family} expects {len(param_names)} args "
            f"({', '.join(param_names)}), got {len(args)}"
        )

    # build param_spec and compute variable domain dimension
    param_spec: list[tuple[str, int | float]] = []
    var_name_order: list[str] = []
    total_var_dim = 0

    for i, arg in enumerate(args):
        if isinstance(arg, (int, float)):
            param_spec.append(("lit", float(arg)))

        else:
            # variable: determine its dimension from type info
            var_dim = 1

            if variable_types and arg in variable_types:
                vtype = variable_types[arg]
                var_dim = getattr(vtype, "dim", 1)

            param_spec.append(("var", var_dim))
            var_name_order.append(arg)
            total_var_dim += var_dim

    # build the domain from variable dimensions
    if total_var_dim == 0:
        domain = Euclidean("_inline_domain", 1)

    elif len(var_name_order) == 1 and variable_types:
        vtype = variable_types.get(var_name_order[0])
        domain = (
            vtype if vtype is not None else Euclidean("_inline_domain", total_var_dim)
        )

    else:
        domain = _infer_domain(var_name_order, variable_types)

    morph = MixedInlineDistribution(
        domain,
        codomain,
        param_spec,
        dist_builder,
        discrete,
    )
    return morph, tuple(var_name_order)


def _infer_domain(
    var_names: list[str],
    variable_types: dict[str, AnySpace] | None,
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
        # fallback: use a generic euclidean space
        return Euclidean("_inline_domain", len(var_names))

    if len(var_names) == 1:
        vtype = variable_types.get(var_names[0])

        if vtype is not None:
            return vtype

        return Euclidean("_inline_domain", 1)

    # multiple variables: create a product
    from quivers.core.objects import ProductSet

    components = []

    for vn in var_names:
        vtype = variable_types.get(vn)

        if vtype is not None:
            components.append(vtype)

        else:
            components.append(Euclidean(f"_inline_{vn}", 1))

    return ProductSet(*components)
