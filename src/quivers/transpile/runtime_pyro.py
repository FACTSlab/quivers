"""Runtime helpers for transpiled Pyro source.

Pyro ships no `TruncatedNormal`, `LogitNormal`, `HalfStudentT`, or
`MatrixNormal` distribution. The
[`PyroRenderer`][quivers.transpile.renderers.pyro.PyroRenderer]
grafts the relevant class definition from this module onto the top
of the emitted file (parsed once through panproto's Python
tree-sitter grammar) and calls it by its bare name in the model
body. The emit is self-contained: a user reading the emitted source
sees the class definition directly, no external import is required
to run the result.

Each class subclasses
[`pyro.distributions.torch_distribution.TorchDistribution`][pyro.distributions.torch_distribution.TorchDistribution]:

* `TruncatedNormal` implements the truncated-normal log density

      log p(x; loc, scale, low, high)
          = log Normal(x; loc, scale) - log(CDF(high) - CDF(low))
          for x in [low, high],
          = -inf otherwise,

  with inverse-CDF sampling on a uniform draw rescaled to the
  truncated quantile interval ``(CDF(low), CDF(high))``.
* `LogitNormal` is `sigmoid(Normal(loc, scale))` on the open unit
  interval, realised as a
  [`TransformedDistribution`][pyro.distributions.TransformedDistribution]
  of a `Normal` under a
  [`SigmoidTransform`][pyro.distributions.transforms.SigmoidTransform].
* `HalfStudentT` is `|StudentT(df, 0, scale)|` on the nonnegative
  reals, realised as a
  [`FoldedDistribution`][pyro.distributions.FoldedDistribution].
* `MatrixNormal` is the matrix-variate normal with Kronecker
  covariance ``col_covariance ⊗ row_covariance``, with the closed
  form log density and a reparameterised sampler.

The classes live in the quivers package (not embedded as an exec'd
string inside the emit) so that:

* the emitted Pyro source is a normal Python module that imports
  its dependencies the way every other Python file does, with no
  `exec` at module load time;
* the math here is statically analysable and unit-testable;
* the classes are reusable across multiple emitted models without
  duplicating their source.
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


class LogitNormal(pyro.distributions.torch_distribution.TorchDistribution):
    """`sigmoid(Normal(loc, scale))` on the open unit interval.

    Pyro ships no `LogitNormal`; this realises it as a
    [`TransformedDistribution`][pyro.distributions.TransformedDistribution]
    of a `Normal(loc, scale)` under a
    [`SigmoidTransform`][pyro.distributions.transforms.SigmoidTransform],
    so the log density carries the transform's Jacobian correction.
    """

    arg_constraints: dict[str, object] = {}
    support = pyro.distributions.constraints.unit_interval
    has_rsample: bool = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        base = pyro.distributions.Normal(loc, scale)
        self.base_dist = base
        self._transformed = pyro.distributions.TransformedDistribution(
            base, pyro.distributions.transforms.SigmoidTransform()
        )
        super().__init__(
            base.batch_shape,
            base.event_shape,
            validate_args=validate_args,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._transformed.log_prob(value)

    def rsample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        return self._transformed.rsample(sample_shape)

    def sample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        return self._transformed.sample(sample_shape)


class HalfStudentT(pyro.distributions.torch_distribution.TorchDistribution):
    """`|StudentT(df, 0, scale)|` on the nonnegative reals.

    Pyro ships no `HalfStudentT`; this realises it as a
    [`FoldedDistribution`][pyro.distributions.FoldedDistribution] of
    a zero-location `StudentT(df, 0, scale)`, folding the symmetric
    density about zero so the mass on ``[0, inf)`` doubles.
    """

    arg_constraints: dict[str, object] = {}
    support = pyro.distributions.constraints.positive
    has_rsample: bool = False

    def __init__(
        self,
        df: torch.Tensor,
        scale: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        base = pyro.distributions.StudentT(df, 0.0, scale)
        self._folded = pyro.distributions.FoldedDistribution(base)
        super().__init__(
            self._folded.batch_shape,
            self._folded.event_shape,
            validate_args=validate_args,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._folded.log_prob(value)

    def sample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        return self._folded.sample(sample_shape)


class MatrixNormal(pyro.distributions.torch_distribution.TorchDistribution):
    """Matrix-variate normal with Kronecker covariance.

    Parameters
    ----------
    loc
        The ``(..., n, p)`` mean matrix.
    row_covariance
        The ``(..., n, n)`` among-row covariance ``U``.
    col_covariance
        The ``(..., p, p)`` among-column covariance ``V``.

    Pyro ships no `MatrixNormal`; this implements the matrix-variate
    normal ``MN(loc, U, V)`` whose vectorisation ``vec(X)`` is
    ``Normal(vec(loc), V ⊗ U)``. The log density is the closed form

        log p(X) = -0.5 * [ tr(V^{-1} (X-M)^T U^{-1} (X-M))
                            + n*p*log(2*pi)
                            + p*log|U| + n*log|V| ],

    evaluated through the Cholesky factors of ``U`` and ``V``.
    `sample` draws ``X = M + A Z B^T`` for ``Z`` standard normal and
    ``U = A A^T``, ``V = B B^T``.
    """

    arg_constraints: dict[str, object] = {}
    has_rsample: bool = True

    def __init__(
        self,
        loc: torch.Tensor,
        row_covariance: torch.Tensor,
        col_covariance: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        dtype = torch.get_default_dtype()
        self.loc = torch.as_tensor(loc, dtype=dtype)
        self.row_covariance = torch.as_tensor(row_covariance, dtype=dtype)
        self.col_covariance = torch.as_tensor(col_covariance, dtype=dtype)
        self._row_tril = torch.linalg.cholesky(self.row_covariance)
        self._col_tril = torch.linalg.cholesky(self.col_covariance)
        n = self.loc.shape[-2]
        p = self.loc.shape[-1]
        super().__init__(
            self.loc.shape[:-2],
            (n, p),
            validate_args=validate_args,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        n = self.loc.shape[-2]
        p = self.loc.shape[-1]
        diff = value - self.loc
        row_solve = torch.cholesky_solve(diff, self._row_tril)
        gram = diff.transpose(-1, -2) @ row_solve
        col_solve = torch.cholesky_solve(gram, self._col_tril)
        trace = col_solve.diagonal(dim1=-2, dim2=-1).sum(-1)
        logdet_row = 2.0 * torch.log(
            self._row_tril.diagonal(dim1=-2, dim2=-1)
        ).sum(-1)
        logdet_col = 2.0 * torch.log(
            self._col_tril.diagonal(dim1=-2, dim2=-1)
        ).sum(-1)
        const = n * p * torch.log(
            torch.as_tensor(2.0 * torch.pi, dtype=self.loc.dtype)
        )
        return -0.5 * (trace + const + p * logdet_row + n * logdet_col)

    def sample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        shape = (
            torch.Size(sample_shape) + self.batch_shape + self.event_shape
        )
        z = torch.randn(shape, dtype=self.loc.dtype)
        return (
            self.loc
            + self._row_tril @ z @ self._col_tril.transpose(-1, -2)
        )


__all__ = [
    "TruncatedNormal",
    "LogitNormal",
    "HalfStudentT",
    "MatrixNormal",
]
