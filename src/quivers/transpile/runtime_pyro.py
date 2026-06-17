"""Runtime helpers for transpiled Pyro source.

Pyro does not ship a `TruncatedNormal` distribution. The
[`PyroRenderer`][quivers.transpile.renderers.pyro.PyroRenderer]
embeds this module's source as a triple-quoted literal at the
top of the emitted file and runs it through `exec(..., globals())`
to register the `TruncatedNormal` class in the emitted module's
namespace. The emit is self-contained: a user reading the emitted
source can see the class definition directly, no external import
is required to run the result.

The class subclasses
[`pyro.distributions.torch_distribution.TorchDistribution`][pyro.distributions.torch_distribution.TorchDistribution]
and implements the truncated-normal log density

    log p(x; loc, scale, low, high)
        = log Normal(x; loc, scale) - log(CDF(high) - CDF(low))
        for x in [low, high],
        = -inf otherwise.

`sample` uses inverse-CDF sampling on a uniform draw rescaled to
the truncated quantile interval ``(CDF(low), CDF(high))``.

The class lives in the quivers package (not embedded as an
exec'd string inside the emit) so that:

* the emitted Pyro source is a normal Python module that
  imports its dependencies the way every other Python file
  does, with no `exec` at module load time;
* the math here is statically analysable and unit-testable;
* the class is reusable across multiple emitted models without
  duplicating its source.
"""

from __future__ import annotations

import pyro
import torch


class TruncatedNormal(pyro.distributions.torch_distribution.TorchDistribution):
    """Normal distribution truncated to ``[low, high]``.

    Parameters
    ----------
    loc, scale
        Centre and (positive) scale of the underlying Normal.
    low, high
        Closed-interval support endpoints. ``high > low`` must
        hold; the constructor does not validate this (the QVR
        compile path ensures the constraint) so that variational
        guides that briefly violate the bound during optimisation
        do not raise at the boundary.
    """

    arg_constraints: dict[str, object] = {}
    has_rsample: bool = False

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: torch.Tensor | float,
        high: torch.Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.base_dist = pyro.distributions.Normal(loc, scale)
        self.low = torch.as_tensor(low, dtype=torch.get_default_dtype())
        self.high = torch.as_tensor(high, dtype=torch.get_default_dtype())
        super().__init__(
            self.base_dist.batch_shape,
            self.base_dist.event_shape,
            validate_args=validate_args,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """``log Normal(value) - log(CDF(high) - CDF(low))`` on the
        support; ``-inf`` outside.
        """
        base_lp = self.base_dist.log_prob(value)
        log_z = torch.log(
            self.base_dist.cdf(self.high) - self.base_dist.cdf(self.low)
        )
        in_bounds = (value >= self.low) & (value <= self.high)
        out = base_lp - log_z
        return torch.where(
            in_bounds, out, torch.full_like(out, float("-inf"))
        )

    def sample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Inverse-CDF sampling on the rescaled uniform draw
        ``CDF(low) + u * (CDF(high) - CDF(low))``."""
        shape = (
            torch.Size(sample_shape)
            + self.base_dist.batch_shape
            + self.base_dist.event_shape
        )
        u = torch.rand(shape)
        f_low = self.base_dist.cdf(self.low)
        f_high = self.base_dist.cdf(self.high)
        return self.base_dist.icdf(f_low + u * (f_high - f_low))


__all__ = ["TruncatedNormal"]
