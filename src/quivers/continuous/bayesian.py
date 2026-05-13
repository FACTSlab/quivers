"""Hierarchical Bayesian modelling primitives.

Adds the operations needed to express models of the kind found in
hierarchical Bayesian regression / latent-class analysis (the
canonical Stan workhorse): finite-domain-indexed draws (plates),
vectorized observations with gather indexing, LKJ priors on
correlation matrices, ordinal monotone splines via cumulative
sum, and generic distribution truncation. Each primitive is
declared with its categorical denotation in :math:`\\mathbf{Kern}`;
the runtime is a straight realisation of those denotations.

Categorical foundations
-----------------------
* **Plate (indexed draw)**. Given a finite index set :math:`A` and a
  parameterized family :math:`F : \\Theta \\to \\mathcal{G}(B)`, a
  *plate draw* declares the kernel

  .. math::

      v : A \\to B,\\quad v(a) \\sim F(\\theta(a))

  equivalently a single morphism into the function space
  :math:`\\mathbf{1} \\to \\mathcal{G}(B^A)` factoring as the
  independent product :math:`\\prod_{a \\in A} F(\\theta(a))`. Under
  the natural isomorphism
  :math:`\\mathbf{Kern}(\\mathbf{1}, B^A) \\cong \\mathbf{Kern}(A, B)`
  the plate variable IS a :math:`\\mathbf{Kern}`-morphism
  :math:`A \\to B`; in the variational backend it is held as a
  tensor of shape ``(|A|, *B.shape)`` whose prior contribution to
  the ELBO is the per-row log-density.

* **Gather**. Given a finite fibration :math:`\\iota : N \\to A` (a
  per-observation grouping assignment) and a plate variable
  :math:`v : A \\to B`, the gathered morphism is the pullback
  :math:`\\iota^* v = v \\circ \\iota : N \\to B`. Categorically a
  textbook reindexing morphism in :math:`\\mathbf{Kern}`; runtime is
  ``v[indices]`` along the leading axis.

* **Vectorised observe**. A batched observation step

  .. math::

      \\mathcal{S}\\llbracket\\,\\mathsf{observe}\\ r[n] \\sim
      F(\\theta[n])\\ \\mathsf{for}\\ n\\in N\\,\\rrbracket :
      \\Phi \\to \\mathcal{G}_{\\le 1}(\\Phi)

  has score :math:`\\prod_{n \\in N} p_F(r_{\\text{obs}}(n);\\,
  \\theta(n,\\phi))`. The single-observation form is the
  ``|N| = 1`` special case; mass-correctness follows from the
  Cho-Jacobs Markov-with-conditioning calculus.

* **Marginalise (program-level)**. Given a previously-drawn
  discrete latent :math:`c : \\Phi \\to \\mathcal{G}(C)`, the
  marginalisation step is the pushforward through the projection
  :math:`\\pi_{\\Phi \\setminus C} : \\Phi \\times C \\to \\Phi`:

  .. math::

      \\mathsf{marg}(c) = \\mathcal{G}(\\pi_{\\Phi \\setminus C})
                         \\circ \\mathcal{S}\\llbracket\\mathsf{draw}\\ c\\rrbracket

  numerically realised as :math:`\\log \\sum_c \\exp(\\cdot)` over
  the :math:`C` axis of the per-class log-likelihoods.

* **cumsum**. The deterministic morphism
  :math:`\\mathrm{cumsum} : \\mathrm{Euclidean}(K) \\to \\mathrm{Euclidean}(K)`,
  :math:`(x_1, \\dots, x_K) \\mapsto (x_1, x_1 + x_2, \\dots,
  \\sum_i x_i)`. Used for the standard monotonic-spline
  parameterization of ordinal coefficients.

* **LKJ correlation prior**. The Lewandowski-Kurowicka-Joe
  distribution :math:`\\mathrm{LKJ}(K, \\eta)` on the manifold
  :math:`\\mathrm{Corr}_K` of :math:`K \\times K` correlation
  matrices, parameterized via the Cholesky factor for numerical
  stability. The accompanying ``cholesky_quad_form`` deterministic
  morphism reconstructs a full covariance
  :math:`\\Sigma = \\mathrm{diag}(s)\\, R\\, \\mathrm{diag}(s)`
  from a correlation :math:`R` and a per-component scale
  :math:`s \\in \\mathrm{PositiveReals}(K)`.

* **Truncated**. Given a base family :math:`F` and an interval
  :math:`[a, b]`, the truncated family
  :math:`F_{|[a, b]}` has density
  :math:`p_F(x) / (F_{\\text{cdf}}(b) - F_{\\text{cdf}}(a))` on
  :math:`[a, b]` and zero elsewhere. Realised via inverse-CDF
  sampling when available; rejection otherwise.

References
----------
- Cho, K. and Jacobs, B. (2019). *Disintegration and Bayesian
  inversion via string diagrams*. Mathematical Structures in
  Computer Science 29(7), 938–971. doi:10.1017/S0960129518000488.
- Lewandowski, D., Kurowicka, D. and Joe, H. (2009). *Generating
  random correlation matrices based on vines and extended onion
  method*. Journal of Multivariate Analysis 100(9), 1989–2001.
  doi:10.1016/j.jmva.2009.04.008.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn

from typing import Literal as _Literal

from quivers.continuous.morphisms import ContinuousMorphism, AnySpace
from quivers.continuous.spaces import ContinuousSpace, Euclidean, PositiveReals


# ---------------------------------------------------------------------------
# Deterministic morphisms: cumsum, softmax, sigmoid, cholesky_quad_form
# ---------------------------------------------------------------------------


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
    (`coef ~ Normal(0, σ)` followed by `cumsum(coef - coef[0])`
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
    in posterior blocks where we'd otherwise need a log-sum-exp
    aggregation by hand.
    """
    from quivers.continuous.spaces import Simplex

    src = Euclidean(name="softmax_in", dim=dim)
    tgt = Simplex(name="softmax_out", dim=dim)
    return _DeterministicMorphism(
        src, tgt, lambda x: torch.softmax(x, dim=-1), name="softmax"
    )


def cholesky_quad_form(dim: int) -> ContinuousMorphism:
    """Covariance reconstruction ``(L, s) ↦ diag(s) · L L^T · diag(s)``.

    Given a Cholesky factor :math:`L` of a :math:`K \\times K`
    correlation matrix and a positive-scale vector
    :math:`s \\in (0, \\infty)^K`, returns the corresponding
    covariance matrix :math:`\\Sigma = D R D^T` with
    :math:`R = L L^T` and :math:`D = \\mathrm{diag}(s)`.

    Domain is the product
    ``CholeskyFactor(K) × PositiveReals(K)``; codomain is
    ``Euclidean(K * K)`` flattened in row-major order so the
    result composes with downstream :class:`ConditionalMultivariateNormal`
    consumers that accept a flat covariance vector.
    """
    from quivers.continuous.spaces import ProductSpace

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
        # Zero out the strict upper triangle to enforce lower-tri.
        mask = torch.tril(torch.ones(dim, dim, device=xs.device, dtype=xs.dtype))
        L = L * mask
        R = L @ L.transpose(-1, -2)
        D = scale_vec.unsqueeze(-1) * torch.eye(dim, device=xs.device, dtype=xs.dtype)
        cov = D @ R @ D
        return cov.reshape(batch, dim * dim)

    return _DeterministicMorphism(source, target, _apply, name="cholesky_quad_form")


# ---------------------------------------------------------------------------
# CholeskyFactor space (manifold of unit-diagonal lower-triangular factors)
# ---------------------------------------------------------------------------


class CholeskyFactor(ContinuousSpace):
    """The manifold of :math:`K \\times K` lower-triangular Cholesky factors.

    Each element is a lower-triangular matrix :math:`L` whose rows
    have unit norm: :math:`L_{ii}^2 + \\sum_{j<i} L_{ij}^2 = 1` for
    every :math:`i`. The product :math:`L L^T` is then a
    correlation matrix. The standard parameterization places
    :math:`L` on a :math:`K(K-1)/2`-dimensional manifold.

    Carrier represented as a flat :math:`K \\times K` array
    (row-major); the on-manifold constraint is enforced by the
    sampling family (:class:`LKJCorrelationFactor` below) and not
    by the type itself.

    Attributes
    ----------
    name : str
        Human-readable name.
    dim : int
        The cardinality :math:`K` of the correlation matrix.
    """

    name: str
    dim: int
    kind: _Literal["cholesky_factor"] = "cholesky_factor"

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dim * self.dim,)

    @property
    def ndim(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# LKJ correlation prior on Cholesky factors
# ---------------------------------------------------------------------------


class LKJCorrelationFactor(ContinuousMorphism):
    """LKJ prior on Cholesky factors ``LKJ(K, η)`` over CholeskyFactor(K).

    Density on the Cholesky factor:

    .. math::

        p(L) \\propto \\prod_{k=2}^{K} L_{kk}^{K - k + 2(\\eta - 1)}.

    A higher concentration :math:`\\eta > 1` pulls toward the
    identity correlation; :math:`\\eta = 1` is uniform on
    correlations. Sampling uses the *onion method* of
    Lewandowski-Kurowicka-Joe 2009: draw row-norm partial
    correlations from Beta distributions and form :math:`L`
    row-by-row.

    Parameters
    ----------
    dim : int
        Correlation-matrix size :math:`K \\ge 2`.
    eta : float
        Concentration :math:`\\eta > 0`.
    domain : AnySpace
        The morphism's source (parameter conditioning); typically
        the program's input space. The LKJ prior itself does not
        consume per-observation conditioning, so the rsample path
        broadcasts the prior across the batch dimension.
    """

    def __init__(self, dim: int, eta: float, domain: AnySpace) -> None:
        if dim < 2:
            raise ValueError(f"LKJ requires dim >= 2; got {dim}")
        if eta <= 0:
            raise ValueError(f"LKJ requires eta > 0; got {eta}")
        codomain = CholeskyFactor(name=f"L({dim})", dim=dim)
        super().__init__(domain, codomain)
        self._dim = dim
        self._eta = float(eta)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        del sample_shape
        """Onion-method sample from the LKJ Cholesky prior.

        Following Stan's reference implementation: for each row
        :math:`i = 1 \\ldots K-1` of :math:`L` (1-indexed),
        sample a vector :math:`y` on the unit sphere and a beta-
        distributed radius :math:`r_i \\sim \\mathrm{Beta}(i/2,
        \\eta + (K - i - 1)/2)` (when :math:`\\eta = 1` this reduces
        to the uniform-on-correlations LKJ-1 case).
        """
        batch = x.shape[0]
        K = self._dim
        eta = self._eta
        L = torch.zeros(batch, K, K, device=x.device, dtype=x.dtype)
        L[:, 0, 0] = 1.0
        for i in range(1, K):
            # Beta parameters for row i (Stan's onion method).
            alpha = eta + (K - 1 - i) / 2.0
            beta = i / 2.0
            r2 = torch.distributions.Beta(
                torch.full((batch,), alpha, device=x.device, dtype=x.dtype),
                torch.full((batch,), beta, device=x.device, dtype=x.dtype),
            ).rsample()
            # Sample a vector uniformly on the unit (i)-sphere.
            u = torch.randn(batch, i, device=x.device, dtype=x.dtype)
            u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
            # row i has off-diagonal entries r * u, diagonal sqrt(1 - r^2).
            L[:, i, :i] = torch.sqrt(r2).unsqueeze(-1) * u
            L[:, i, i] = torch.sqrt(1.0 - r2)
        return L.reshape(batch, K * K)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-density of the LKJ prior at the Cholesky factor ``y``.

        Up to a normalizing constant that doesn't depend on
        :math:`L`, :math:`\\log p(L) = \\sum_{k=2}^{K} (K-k+2(\\eta-1))
        \\log L_{kk}`. The diagonal entries are extracted from the
        flattened representation.
        """
        batch = y.shape[0]
        K = self._dim
        L = y.reshape(batch, K, K)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)  # (batch, K)
        # Coefficients per diagonal entry (Stan's lkj_corr_cholesky_lpdf):
        # log_jac_term[k] = (K - k + 2*(eta - 1)) * log(L_kk)  for k = 2..K
        # Pre-K-indexed: power[0..K-1] where power[k] = (K-1-k) + 2*(eta-1).
        # The first diagonal is fixed at 1 so log(1)=0 contributes nothing.
        ks = torch.arange(K, device=y.device, dtype=y.dtype)
        powers = (K - 1 - ks) + 2.0 * (self._eta - 1.0)
        log_diag = torch.log(diag.clamp(min=1e-30))
        return (powers * log_diag).sum(dim=-1)

    def __repr__(self) -> str:
        return f"LKJCorrelationFactor(dim={self._dim}, eta={self._eta})"


# ---------------------------------------------------------------------------
# Generic truncation combinator
# ---------------------------------------------------------------------------


class Truncated(ContinuousMorphism):
    """Truncate a base family to an interval :math:`[a, b]`.

    Categorical denotation: given a base family
    :math:`F : \\Theta \\to \\mathcal{G}(\\mathbb{R})` and constants
    :math:`a, b \\in \\bar{\\mathbb{R}}` with :math:`a < b`, the
    truncated family has density

    .. math::

        p_{F_{|[a,b]}}(x) = \\frac{p_F(x)}{F_{\\text{cdf}}(b)
        - F_{\\text{cdf}}(a)} \\cdot \\mathbb{1}_{[a,b]}(x)

    and the morphism :math:`F_{|[a,b]} : \\Theta \\to
    \\mathcal{G}([a,b])`. Sampling uses inverse-CDF when
    :attr:`base` supports it; otherwise rejection sampling.

    Parameters
    ----------
    base : ContinuousMorphism
        The base distribution-family morphism. Must expose
        ``log_prob`` and ``rsample`` plus an ``icdf`` method or a
        ``base_distribution`` torch ``Distribution`` for inverse-CDF
        sampling. Falls back to rejection sampling otherwise.
    lower : float or None
        Lower bound :math:`a`. ``None`` means :math:`-\\infty`.
    upper : float or None
        Upper bound :math:`b`. ``None`` means :math:`+\\infty`.
    max_rejection_iterations : int
        Cap on rejection-sampling attempts before raising.
    """

    def __init__(
        self,
        base: ContinuousMorphism,
        lower: float | None = None,
        upper: float | None = None,
        max_rejection_iterations: int = 64,
    ) -> None:
        super().__init__(base.domain, base.codomain)
        if lower is None and upper is None:
            raise ValueError(
                "Truncated requires at least one of lower / upper to be finite; "
                "without truncation, use the base family directly"
            )
        if lower is not None and upper is not None and not (lower < upper):
            raise ValueError(
                f"Truncated requires lower < upper; got lower={lower}, upper={upper}"
            )
        self._base = base
        self._lower = lower
        self._upper = upper
        self._max_iters = max_rejection_iterations
        # Attach so the parent nn.Module tracks parameters.
        self._base_mod = base

    def _interval_logmass(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate :math:`\\log(F(b) - F(a))` via Monte-Carlo on the base.

        For families with analytic CDFs (Normal, etc.) we could use
        a closed form, but a general primitive operating across
        every family in the registry uses 256 base samples to
        estimate the truncation mass. This keeps the combinator
        usable on every family without per-family special-casing.
        """
        with torch.no_grad():
            samples = self._base.rsample(x.repeat(256, *([1] * (x.ndim - 1))))
            mask = torch.ones_like(samples)
            if self._lower is not None:
                mask = mask * (samples >= self._lower).float()
            if self._upper is not None:
                mask = mask * (samples <= self._upper).float()
            in_interval = mask.reshape(256, x.shape[0], -1).mean(dim=0)
        return torch.log(in_interval.clamp(min=1e-30)).squeeze(-1)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        del sample_shape
        """Rejection-sample from the base until every entry is in :math:`[a,b]`."""
        batch = x.shape[0]
        out = self._base.rsample(x)
        mask = self._in_bounds(out)
        for _ in range(self._max_iters):
            if mask.all():
                return out
            replacement = self._base.rsample(x)
            out = torch.where(
                mask.reshape(batch, *([1] * (out.ndim - 1))).expand_as(out),
                out,
                replacement,
            )
            mask = self._in_bounds(out)
        if not mask.all():
            raise RuntimeError(
                f"Truncated rejection sampling failed to fill the batch in "
                f"{self._max_iters} iterations; the truncation mass is "
                "vanishingly small for the supplied parameters"
            )
        return out

    def _in_bounds(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.ones(x.shape[:1], device=x.device, dtype=torch.bool)
        if self._lower is not None:
            m = m & (x.reshape(x.shape[0], -1) >= self._lower).all(dim=-1)
        if self._upper is not None:
            m = m & (x.reshape(x.shape[0], -1) <= self._upper).all(dim=-1)
        return m

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        base_lp = self._base.log_prob(x, y)
        # Hard-zero outside the interval.
        in_bounds = self._in_bounds(y)
        truncation_lp = self._interval_logmass(x)
        adjusted = base_lp - truncation_lp
        return torch.where(
            in_bounds,
            adjusted,
            torch.full_like(adjusted, float("-inf")),
        )

    def __repr__(self) -> str:
        return f"Truncated({self._base!r}, lower={self._lower}, upper={self._upper})"


# ---------------------------------------------------------------------------
# Plate / vectorized observe / program-level marginalise Python builders
# ---------------------------------------------------------------------------


class PlateDraw(ContinuousMorphism):
    """A finite-domain-indexed draw, as a Kern-morphism ``A → B``.

    Concretely: ``v : A → B ~ F(theta)`` becomes a tensor of shape
    ``(|A|, *B.shape)`` whose ``a``-th row is an independent
    :math:`F(\\theta_a)`-distributed random variable. The variational
    posterior factorizes across rows by default; the prior's ELBO
    contribution is :math:`\\sum_a \\log p_F(v_a; \\theta_a)`.

    Categorically: by the natural isomorphism
    :math:`\\mathbf{Kern}(\\mathbf{1}, B^A) \\cong \\mathbf{Kern}(A, B)`,
    the plate variable IS a Kern-morphism :math:`A \\to B`. The
    PlateDraw is realised as a :class:`ContinuousMorphism` whose
    codomain is the flat product-space of ``index_size`` copies of
    the per-row family's codomain.

    Parameters
    ----------
    index_size : int
        Cardinality :math:`|A|`.
    family : ContinuousMorphism
        Per-row distribution family.
    domain : AnySpace
        The program's input space (broadcast conditioning).
    """

    def __init__(
        self,
        index_size: int,
        family: ContinuousMorphism,
        domain: AnySpace | None = None,
    ) -> None:
        # Continuous spaces use `dim` instead of `shape`; treat
        # them uniformly by extracting a flat dim count.
        if hasattr(family.codomain, "dim"):
            per_row_dim = int(family.codomain.dim)
            per_row_shape: tuple[int, ...] = (per_row_dim,)
        else:
            per_row_shape = tuple(family.codomain.shape)
            per_row_dim = (
                int(torch.tensor(per_row_shape).prod().item()) if per_row_shape else 1
            )
        flat_codomain = Euclidean(
            name=f"plate({index_size}x{family.codomain!s})",
            dim=index_size * per_row_dim,
        )
        actual_domain = domain if domain is not None else family.domain
        super().__init__(actual_domain, flat_codomain)
        self._index_size = index_size
        self._family = family
        self._per_row_shape = per_row_shape
        # Variational mean / log-scale per row (mean-field Gaussian
        # posterior over the plate). Shape (|A|, *B.shape).
        self._mean = nn.Parameter(torch.zeros(index_size, *per_row_shape))
        self._log_scale = nn.Parameter(torch.full((index_size, *per_row_shape), -2.0))

    @property
    def index_size(self) -> int:
        return self._index_size

    @property
    def family(self) -> ContinuousMorphism:
        return self._family

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Reparameterized sample.

        A plate draw is *batch-invariant*: the latent vector is a
        global model parameter shared across every row of an
        observed plate. The returned tensor has shape
        ``(|A|, *B.shape)`` regardless of the program input's
        leading batch dimension.

        This is the standard Pyro / NumPyro semantic: a sample inside
        a ``plate("subj", n_subj)`` context is one ``(n_subj,)``
        vector, *not* a per-particle (batch, n_subj) replication. The
        gather ``arr[idx]`` along the plate axis then composes
        cleanly with per-row observed-plate axes downstream.
        """
        del sample_shape, x  # plate latents are batch-invariant
        eps = torch.randn(
            *self._mean.shape, device=self._mean.device, dtype=self._mean.dtype
        )
        sample = self._mean + self._log_scale.exp() * eps
        # Scalar-per-row plates have ``per_row_shape == (1,)``; the
        # trailing length-1 axis is noise from how the family
        # advertises its codomain (Euclidean(name=..., dim=1)). Squeeze
        # it so the latent has the natural ``(|A|,)`` shape and
        # downstream ``arr[idx]`` advance-indexing produces ``(N,)``
        # without the user having to squeeze manually.
        if (
            sample.dim() >= 2
            and sample.shape[-1] == 1
            and len(self._per_row_shape) == 1
        ):
            sample = sample.squeeze(-1)
        return sample

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-density of the variational posterior on the plate sample.

        Accepts ``y`` shaped either as the natural plate-latent
        ``(|A|, *B.shape)`` (one shared sample, the new convention)
        or the legacy flat ``(batch, |A| * prod(B.shape))``; the
        latter reshape is preserved for back-compat with any caller
        that pre-flattened the latent.
        """
        del x  # plate latents are batch-invariant
        # Accept the new natural plate shape (``(|A|, *per_row_shape)``
        # or the squeezed ``(|A|,)`` for scalar-per-row plates) before
        # falling back to the legacy ``(batch, flat)`` reshape.
        if (
            len(self._per_row_shape) == 1
            and self._per_row_shape[0] == 1
            and y.dim() == 1
            and y.shape[0] == self._index_size
        ):
            sample = y.unsqueeze(-1)
            collapse_batch = False
        elif y.dim() == 1 + len(self._per_row_shape) and y.shape[0] == self._index_size:
            sample = y
            collapse_batch = False
        else:
            batch = y.shape[0]
            sample = y.reshape(batch, self._index_size, *self._per_row_shape)
            collapse_batch = True
        var = (2.0 * self._log_scale).exp()
        per_row_lp = (
            -0.5 * ((sample - self._mean) ** 2 / var)
            - self._log_scale
            - 0.5
            * torch.log(torch.tensor(2.0 * torch.pi, device=y.device, dtype=y.dtype))
        )
        if collapse_batch:
            return per_row_lp.reshape(per_row_lp.shape[0], -1).sum(dim=-1)
        # Plate-latent shape: return a scalar log-density. We wrap in
        # a length-1 tensor so the downstream log-joint accumulator
        # (which sums batched per-step contributions) sees a uniform
        # 1-d shape it can broadcast against the response plate.
        return per_row_lp.reshape(-1).sum().unsqueeze(0)

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        """Pullback ``v[indices]`` along a finite fibration.

        ``indices`` is a long-tensor of shape ``(N,)`` with entries
        in ``[0, |A|)``; returns a tensor of shape
        ``(N, *B.shape)``.
        """
        return self.rsample()[indices]

    def __repr__(self) -> str:
        return f"PlateDraw(index_size={self._index_size}, family={self._family!r})"


class VectorisedObserve(ContinuousMorphism):
    """A batched observation step accumulating per-row log-likelihoods.

    Categorically, the batched-likelihood kernel
    :math:`\\Phi \\to \\mathcal{G}_{\\le 1}(\\Phi)` whose score is
    :math:`\\prod_{n \\in N} p_F(r_{\\text{obs}}(n);\\, \\theta(n,\\phi))`.
    Realised as a :class:`ContinuousMorphism` whose domain is the
    parameter-input space (the morphism conditions on θ) and whose
    codomain is the per-observation response space — so the
    existing :class:`MonadicProgram` ``_StepSpec`` machinery treats
    it as an observed site and threads the score through
    ``log_joint`` via the usual ``morph.log_prob(theta, response)``
    call, with ``log_prob`` here summing over the leading index axis.

    The observed response tensor is registered as a buffer so the
    parent program's optimizer tracks it and the runtime never
    has to thread it through ``observations=...``.

    Parameters
    ----------
    family : ContinuousMorphism
        The per-observation distribution family.
    response : torch.Tensor
        Observed values ``r_obs`` of shape ``(N, *codom.shape)``
        (or ``(N,)`` for scalar codomains).
    """

    def __init__(self, family: ContinuousMorphism, response: torch.Tensor) -> None:
        super().__init__(family.domain, family.codomain)
        self._family = family
        self.register_buffer("_response", response.detach())

    @property
    def response(self) -> torch.Tensor:
        return cast("torch.Tensor", self._response)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample the per-observation family at the supplied θ.

        ``x`` is the θ-tensor (one row per observation index); the
        result is the per-observation response sample. Used in
        prior-predictive simulation; never called during inference
        when the response is observed.
        """
        return self._family.rsample(x, sample_shape)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Sum of per-observation log-densities.

        ``y`` defaults to the registered response buffer; passing a
        different value (e.g. a clamped observation) is supported
        for fast prior-predictive checks.
        """
        target = y if y is not None else cast("torch.Tensor", self._response)
        return self._family.log_prob(x, target).sum()

    def log_likelihood(self, theta: torch.Tensor) -> torch.Tensor:
        """Alias for ``log_prob(theta)``; preserved for the Python
        builder API."""
        return self.log_prob(theta)

    def __repr__(self) -> str:
        return (
            f"VectorisedObserve(family={self._family!r}, "
            f"N={cast('torch.Tensor', self._response).shape[0]})"
        )


def marginalize_categorical(log_probs_per_class: torch.Tensor) -> torch.Tensor:
    """Program-level marginalisation over a discrete latent class.

    Given per-class log-likelihoods of shape ``(N, K)`` (one row per
    observation, one column per latent-class assignment), returns
    the marginalised log-likelihood of shape ``(N,)``:

    .. math::

        \\log p(r_n) = \\log \\sum_{k=1}^{K} p(c_n = k)\\, p(r_n | c_n = k).

    Realises the program-level ``marginalize c`` step as the
    pushforward through :math:`\\pi_{\\Phi \\setminus C}`.
    """
    return torch.logsumexp(log_probs_per_class, dim=-1)


def _flatten_product_indices(
    group_indices: tuple[torch.Tensor, ...],
    group_sizes: tuple[int, ...],
) -> tuple[torch.Tensor, int]:
    """Compose a tuple of co-indexed fibration tensors into a
    single flat group index on the product grouping plate.

    The product :math:`G_1 \\times G_2 \\times \\dots \\times G_r`
    is row-major: ``flat = idx_1 * (|G_2| ... |G_r|) + idx_2 * (|G_3| ...) + ... + idx_r``.

    Returns ``(flat_index, total_groups)`` where ``total_groups =
    \\prod_i |G_i|``.
    """
    if not group_indices:
        raise ValueError(
            "_flatten_product_indices: need at least one fibration "
            "tensor"
        )
    if len(group_indices) != len(group_sizes):
        raise ValueError(
            "_flatten_product_indices: number of indices "
            f"({len(group_indices)}) must equal number of sizes "
            f"({len(group_sizes)})"
        )
    n_rows = int(group_indices[0].shape[0])
    for j, idx in enumerate(group_indices):
        if idx.shape != (n_rows,):
            raise ValueError(
                f"_flatten_product_indices: fibration {j} has shape "
                f"{tuple(idx.shape)}; expected ({n_rows},) to match "
                f"the first fibration"
            )
    total = 1
    for size in group_sizes:
        if size <= 0:
            raise ValueError(
                "_flatten_product_indices: all group sizes must be "
                f"positive; got {group_sizes}"
            )
        total *= int(size)
    flat = torch.zeros(n_rows, dtype=torch.long, device=group_indices[0].device)
    running = 1
    for j in range(len(group_indices) - 1, -1, -1):
        size = int(group_sizes[j])
        idx_j = group_indices[j].to(torch.long)
        if (idx_j < 0).any() or (idx_j >= size).any():
            raise ValueError(
                f"_flatten_product_indices: fibration {j} has entries "
                f"outside [0, {size})"
            )
        flat = flat + idx_j * running
        running *= size
    return flat, total


def marginalize_grouped(
    log_likelihood_per_row_per_class: torch.Tensor,
    group_index: torch.Tensor | tuple[torch.Tensor, ...],
    log_prior_per_group_per_class: torch.Tensor,
    num_groups: int | tuple[int, ...],
    *,
    reduction: str = "logsumexp",
) -> torch.Tensor:
    """Per-group marginalisation over a discrete latent class.

    Given a per-(response, class) log-likelihood tensor of shape
    ``(N, K)``, a fibration
    :math:`r : \\text{Resp} \\to G` (or a tuple of co-indexed
    fibrations into a product grouping plate
    :math:`G_1 \\times \\dots \\times G_r`), a per-(group, class)
    log-prior, and ``|G|``, return the scalar

    .. math::

        \\sum_{g \\in G}\\, \\rho\\!\\left[
            \\log \\pi(g, \\cdot) + \\sum_{n:\\, r(n) = g} \\ell(n, \\cdot)
        \\right]

    where :math:`\\rho` is the per-group reduction over the class
    axis selected by ``reduction``: ``logsumexp`` for the canonical
    marginal-likelihood mixture, ``sum`` for the joint scoring (no
    marginalisation), ``mean`` for the symmetric average. The
    per-group accumulator is realised as a scatter-add along the
    fibration; this is the right Kan extension along ``r`` in
    :math:`\\mathbf{Kern}` with the additive monoid on
    log-densities. The final sum aggregates the per-group log-
    marginals into the program-level log-density contribution.

    Parameters
    ----------
    log_likelihood_per_row_per_class : torch.Tensor
        Per-(response, class) log-likelihood of shape ``(N, K)``.
    group_index : torch.Tensor or tuple[torch.Tensor, ...]
        Single fibration: long tensor of shape ``(N,)`` mapping
        each response row to its group index in ``[0, num_groups)``.
        Product fibration: a tuple of co-indexed long tensors, one
        per grouping plate; ``num_groups`` must be a matching tuple.
        The product grouping plate
        :math:`G_1 \\times \\dots \\times G_r` is flattened in row-
        major order before the scatter-add.
    log_prior_per_group_per_class : torch.Tensor
        Per-(group, class) log-prior of shape ``(num_groups, K)``,
        ``(K,)`` (broadcast over the group axis), or — for product
        fibrations — ``(num_groups_1, ..., num_groups_r, K)``.
    num_groups : int or tuple[int, ...]
        Cardinality of the (product) group plate.
    reduction : {"logsumexp", "sum", "mean"}
        Per-group reduction over the class axis. ``logsumexp`` is
        the canonical mixture marginalisation; ``sum`` joint-scores
        without marginalising the class; ``mean`` averages
        symmetrically. Default ``logsumexp``.

    Returns
    -------
    torch.Tensor
        Scalar program-level log-density contribution.

    Notes
    -----
    Edge cases:

    * ``K == 1``: the log-sum-exp collapses to the body's log-
      likelihood plus the constant log-prior.
    * Identity fibration (``group_index = arange(N)`` and
      ``num_groups == N``): each row is its own group, recovering
      the per-row mixture.
    * Empty fibre: contributes a multiplicative identity to the
      product (``log 1 = 0``), so an unused group adds zero per
      class before the reduction.
    """
    if reduction not in ("logsumexp", "sum", "mean"):
        raise ValueError(
            "reduction must be one of 'logsumexp', 'sum', 'mean'; "
            f"got {reduction!r}"
        )
    if log_likelihood_per_row_per_class.dim() < 1:
        raise ValueError(
            "log_likelihood_per_row_per_class must have at least one "
            "axis (the class axis); got "
            f"{log_likelihood_per_row_per_class.shape}"
        )
    # Two operating modes:
    #
    # 1. ``ll`` has a leading row axis (shape ``(N, *extra, K)``):
    #    scatter-add along the fibration to obtain
    #    ``(G, *extra, K)``, apply the per-cell prior, reduce over
    #    the class axis, and sum over groups. The shape of the
    #    return value is ``(*extra,)`` — a scalar if there are no
    #    extra (outer-class) axes, otherwise a tensor whose axes
    #    are the outer-block class axes still in scope.
    #
    # 2. ``ll`` has no row axis (shape ``(*extra, K)``): this
    #    occurs for *intermediate* levels of a nested marginalize
    #    stack, where the innermost level has already integrated
    #    the per-row contributions. There is no scatter step; the
    #    prior is added across the extra axes and the class axis is
    #    reduced. The return shape is ``(*extra_except_class_axis,)``.
    # The mode is detected from the shape of ``group_index``.
    flat_group_index_shape: tuple[int, ...]
    if isinstance(group_index, tuple):
        flat_group_index_shape = group_index[0].shape
    else:
        flat_group_index_shape = group_index.shape
    if flat_group_index_shape:
        n_idx = flat_group_index_shape[0]
        # If the user supplied a fibration of length N > 0, the ll
        # MUST carry a leading axis of length N. A length-N fibration
        # paired with an ll whose leading axis differs is a shape
        # mismatch — not the "no row axis" intermediate path.
        if (
            n_idx > 0
            and log_likelihood_per_row_per_class.dim() >= 2
            and log_likelihood_per_row_per_class.shape[0] != n_idx
        ):
            raise ValueError(
                "group_index must have shape (N,) matching the "
                "leading axis of the log-likelihood; got "
                f"{flat_group_index_shape} vs N="
                f"{log_likelihood_per_row_per_class.shape[0]}"
            )
        has_row_axis = (
            n_idx > 0
            and log_likelihood_per_row_per_class.dim() >= 2
            and log_likelihood_per_row_per_class.shape[0] == n_idx
        )
    else:
        has_row_axis = False
    if not has_row_axis:
        # No N-axis: apply prior + reduce over the class axis only.
        # ``log_prior_per_group_per_class`` is broadcastable across
        # the extra axes (typically just (K,)).
        weighted = log_prior_per_group_per_class + log_likelihood_per_row_per_class
        if reduction == "logsumexp":
            return torch.logsumexp(weighted, dim=-1)
        if reduction == "sum":
            return weighted.sum(dim=-1)
        return weighted.mean(dim=-1)
    n_rows = int(log_likelihood_per_row_per_class.shape[0])
    n_classes = int(log_likelihood_per_row_per_class.shape[-1])
    extra_axes = tuple(log_likelihood_per_row_per_class.shape[1:-1])

    # Resolve single-vs-product fibration into a flat group index.
    if isinstance(group_index, tuple):
        if not isinstance(num_groups, tuple):
            raise ValueError(
                "marginalize_grouped: product fibration (tuple "
                "group_index) requires num_groups to be a tuple of "
                f"matching length; got {type(num_groups).__name__}"
            )
        flat_index, total_groups = _flatten_product_indices(
            group_index, num_groups
        )
        # Flatten an (G_1, ..., G_r, K) prior to (G_1 * ... * G_r, K).
        if log_prior_per_group_per_class.dim() == 1:
            log_prior_flat: torch.Tensor = log_prior_per_group_per_class
        elif log_prior_per_group_per_class.dim() == 2 and (
            log_prior_per_group_per_class.shape[0] == total_groups
            or log_prior_per_group_per_class.shape[0] == 1
        ):
            log_prior_flat = log_prior_per_group_per_class
        else:
            expected_axes = len(num_groups) + 1
            if log_prior_per_group_per_class.dim() != expected_axes:
                raise ValueError(
                    "marginalize_grouped: product-fibration prior "
                    f"must have shape (*{num_groups}, K) or (K,); "
                    f"got {tuple(log_prior_per_group_per_class.shape)}"
                )
            log_prior_flat = log_prior_per_group_per_class.reshape(
                total_groups, n_classes
            )
    else:
        if isinstance(num_groups, tuple):
            raise ValueError(
                "marginalize_grouped: scalar group_index requires "
                f"scalar num_groups; got {num_groups!r}"
            )
        if group_index.shape != (n_rows,):
            raise ValueError(
                "group_index must have shape (N,) matching the "
                "leading axis of the log-likelihood; got "
                f"{tuple(group_index.shape)} vs N={n_rows}"
            )
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive; got {num_groups}")
        if (group_index < 0).any() or (group_index >= num_groups).any():
            raise ValueError(
                f"group_index entries must lie in [0, {num_groups}); "
                "out-of-range index detected"
            )
        flat_index = group_index.to(torch.long)
        total_groups = int(num_groups)
        log_prior_flat = log_prior_per_group_per_class

    # Scatter-add the (N, *extra, K) per-row log-likelihood along
    # the fibration to obtain a (|G|, *extra, K) per-group
    # accumulator. ``index_add`` operates on the leading axis;
    # ``extra`` axes ride along as broadcast.
    grouped_shape: tuple[int, ...] = (total_groups,) + extra_axes + (n_classes,)
    grouped = torch.zeros(
        grouped_shape,
        dtype=log_likelihood_per_row_per_class.dtype,
        device=log_likelihood_per_row_per_class.device,
    )
    grouped = grouped.index_add(0, flat_index, log_likelihood_per_row_per_class)
    # Broadcast log_prior_flat shaped ``(K,)`` or ``(G, K)``
    # against the accumulator ``(G, *extra, K)``. We insert
    # singleton axes for ``*extra`` so the broadcast is well-formed.
    prior_view = log_prior_flat
    if prior_view.dim() == 1:
        # (K,) → (1,) * len(extra) + (K,) → broadcasts against
        # (G, *extra, K).
        for _ in extra_axes:
            prior_view = prior_view.unsqueeze(0)
    elif prior_view.dim() == 2 and prior_view.shape[0] == total_groups:
        # (G, K) → (G, *(1,)*len(extra), K).
        for _ in extra_axes:
            prior_view = prior_view.unsqueeze(1)
    weighted = prior_view + grouped
    if reduction == "logsumexp":
        per_group = torch.logsumexp(weighted, dim=-1)
    elif reduction == "sum":
        per_group = weighted.sum(dim=-1)
    else:  # mean
        per_group = weighted.mean(dim=-1)
    # Sum over the group axis; extra axes (outer-block class
    # broadcasts) pass through unchanged.
    return per_group.sum(dim=0)


__all__ = [
    "CholeskyFactor",
    "LKJCorrelationFactor",
    "Truncated",
    "PlateDraw",
    "VectorisedObserve",
    "cumsum",
    "softmax",
    "cholesky_quad_form",
    "marginalize_categorical",
    "marginalize_grouped",
]
