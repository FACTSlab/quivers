"""Deterministic morphisms in :math:`\\mathbf{Kern}`.

Each builder returns a :class:`ContinuousMorphism` whose runtime
realisation is a Dirac kernel concentrated on a deterministic
image: ``cumsum`` for ordinal monotone splines, ``softmax`` for
the standard simplex projection, ``cholesky_quad_form`` for
covariance reconstruction from a correlation Cholesky factor and
a positive-scale vector.

Categorically a deterministic morphism :math:`f : X \\to Y` is the
unit-Kleisli image of a measurable map; its log-density is
:math:`-\\infty` everywhere except at :math:`y = f(x)` and zero
there. The variational backend treats the sampling path as an
identity reparameterization of the deterministic image and the
log-density as a no-op (the surrounding KL terms absorb the
Dirac indicator).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from quivers.continuous.morphisms import ContinuousMorphism, AnySpace
from quivers.continuous.spaces import (
    CholeskyFactor,
    Euclidean,
    PositiveReals,
    ProductSpace,
    Simplex,
)


class _DeterministicMorphism(ContinuousMorphism):
    """Helper base for deterministic morphisms ``f : X → Y``.

    A deterministic morphism in :math:`\\mathbf{Kern}` is a Dirac
    kernel concentrated on :math:`f(x)`. Its log-density is
    :math:`-\\infty` everywhere except at :math:`y = f(x)` and
    :math:`0` (= :math:`\\log 1`) there; for gradient-based inference
    we treat the sampling path as an identity reparameterization
    of the deterministic image.

    Subclasses supply ``_apply(x) -> y``.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        apply: Callable[[torch.Tensor], torch.Tensor],
        name: str,
    ) -> None:
        super().__init__(domain, codomain)
        self._apply_fn = apply
        self._name = name

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        del sample_shape
        return self._apply_fn(x)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Deterministic kernels: log-prob is 0 at y == f(x), -inf otherwise.
        # In the variational backend we drop the indicator and treat
        # the morphism's effective log-density as zero (it's a Dirac
        # kernel; the inference target's gradient is uninformed by
        # this density and the surrounding KL terms absorb it).
        expected = self._apply_fn(x)
        return torch.zeros(expected.shape[:1], device=x.device, dtype=x.dtype)

    def __repr__(self) -> str:
        return f"{self._name}({self.domain!s} → {self.codomain!s})"


def cumsum(dim: int) -> _DeterministicMorphism:
    """Cumulative sum ``cumsum : Euclidean(dim) → Euclidean(dim)``.

    Maps :math:`(x_1, \\dots, x_K)` to
    :math:`(x_1, x_1 + x_2, \\dots, \\sum_i x_i)`. The canonical
    monotonic-spline parameterization for ordinal covariates
    (``coef ~ Normal(0, sigma)`` followed by ``cumsum(coef - coef[0])``
    gives a 0-anchored, monotonically-increasing effect across the
    ordered levels).
    """
    space = Euclidean(name="cumsum", dim=dim)
    return _DeterministicMorphism(
        space, space, lambda x: torch.cumsum(x, dim=-1), name="cumsum"
    )


def softmax(dim: int) -> _DeterministicMorphism:
    """Softmax ``softmax : Euclidean(dim) → Simplex(dim)``.

    The standard exponential normalizer onto the probability
    simplex. Useful for closed-form class-probability computations
    in posterior blocks where one would otherwise need a
    log-sum-exp aggregation by hand.
    """
    src = Euclidean(name="softmax_in", dim=dim)
    tgt = Simplex(name="softmax_out", dim=dim)
    return _DeterministicMorphism(
        src, tgt, lambda x: torch.softmax(x, dim=-1), name="softmax"
    )


def cholesky_quad_form(dim: int) -> ContinuousMorphism:
    """Covariance reconstruction ``(L, s) -> diag(s) L L^T diag(s)``.

    Given a Cholesky factor :math:`L` of a :math:`K \\times K`
    correlation matrix and a positive-scale vector
    :math:`s \\in (0, \\infty)^K`, returns the corresponding
    covariance matrix :math:`\\Sigma = D R D^T` with
    :math:`R = L L^T` and :math:`D = \\mathrm{diag}(s)`.

    Domain is the product
    :class:`CholeskyFactor(K) <quivers.continuous.spaces.CholeskyFactor>` ``*``
    :class:`PositiveReals(K) <quivers.continuous.spaces.PositiveReals>`;
    codomain is :class:`Euclidean(K * K) <quivers.continuous.spaces.Euclidean>`
    flattened in row-major order so the result composes with
    downstream
    :class:`~quivers.continuous.families.ConditionalMultivariateNormal`
    consumers that accept a flat covariance vector.
    """
    cholesky = CholeskyFactor(name="L", dim=dim)
    scale = PositiveReals(name="scale", dim=dim)
    source = ProductSpace(components=(cholesky, scale))
    target = Euclidean(name="cov", dim=dim * dim)

    def _apply(xs: torch.Tensor) -> torch.Tensor:
        # xs : (batch, K*K + K) — the cholesky's flat lower-tri plus
        # the K-vector of scales, concatenated in that order. We
        # rehydrate L from the lower-triangular packing, apply the
        # quadratic form, then flatten.
        batch = xs.shape[0]
        chol_flat = xs[:, : dim * dim]
        scale_vec = xs[:, dim * dim :]
        L = chol_flat.reshape(batch, dim, dim)
        mask = torch.tril(torch.ones(dim, dim, device=xs.device, dtype=xs.dtype))
        L = L * mask
        R = L @ L.transpose(-1, -2)
        D = scale_vec.unsqueeze(-1) * torch.eye(
            dim, device=xs.device, dtype=xs.dtype
        )
        cov = D @ R @ D
        return cov.reshape(batch, dim * dim)

    return _DeterministicMorphism(
        source, target, _apply, name="cholesky_quad_form"
    )
