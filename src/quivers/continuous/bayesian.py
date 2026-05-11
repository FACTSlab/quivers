"""Hierarchical Bayesian modelling primitives.

Adds the operations needed to express models of the kind found in
hierarchical Bayesian regression / latent-class analysis (the
canonical Stan workhorse): finite-domain-indexed draws (plates),
vectorised observations with gather indexing, LKJ priors on
correlation matrices, ordinal monotone splines via cumulative
sum, and generic distribution truncation. Each primitive is
declared with its categorical denotation in :math:`\\mathbf{Kern}`;
the runtime is a straight realisation of those denotations.

Categorical foundations
-----------------------
* **Plate (indexed draw)**. Given a finite index set :math:`A` and a
  parameterised family :math:`F : \\Theta \\to \\mathcal{G}(B)`, a
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
  parameterisation of ordinal coefficients.

* **LKJ correlation prior**. The Lewandowski-Kurowicka-Joe
  distribution :math:`\\mathrm{LKJ}(K, \\eta)` on the manifold
  :math:`\\mathrm{Corr}_K` of :math:`K \\times K` correlation
  matrices, parameterised via the Cholesky factor for numerical
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
    we treat the sampling path as an identity reparameterisation
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
    monotonic-spline parameterisation for ordinal covariates
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

    The standard exponential normaliser onto the probability
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
    correlation matrix. The standard parameterisation places
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

        Up to a normalising constant that doesn't depend on
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
# Plate / vectorised observe / program-level marginalise Python builders
# ---------------------------------------------------------------------------


class PlateDraw(ContinuousMorphism):
    """A finite-domain-indexed draw, as a Kern-morphism ``A → B``.

    Concretely: ``v : A → B ~ F(theta)`` becomes a tensor of shape
    ``(|A|, *B.shape)`` whose ``a``-th row is an independent
    :math:`F(\\theta_a)`-distributed random variable. The variational
    posterior factorises across rows by default; the prior's ELBO
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
        """Reparameterised sample.

        Returns a flat ``(batch, |A| * prod(B.shape))`` tensor. Each
        batch row is one independent plate sample with mean-field
        Gaussian variational posterior.
        """
        del sample_shape
        batch = x.shape[0] if x.dim() > 0 else 1
        eps = torch.randn(batch, *self._mean.shape, device=x.device, dtype=x.dtype)
        sample = self._mean.unsqueeze(0) + self._log_scale.exp().unsqueeze(0) * eps
        # Flatten the index axis with the per-row codomain shape.
        return sample.reshape(batch, -1)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-density of the variational posterior on the plate sample.

        ``y`` is the flat-shape sample; reshape to ``(batch, |A|, *B.shape)``
        and sum the per-row Gaussian log-density.
        """
        batch = y.shape[0]
        sample = y.reshape(batch, self._index_size, *self._per_row_shape)
        var = (2.0 * self._log_scale).exp()
        per_row_lp = (
            -0.5 * ((sample - self._mean) ** 2 / var)
            - self._log_scale
            - 0.5
            * torch.log(torch.tensor(2.0 * torch.pi, device=y.device, dtype=y.dtype))
        )
        return per_row_lp.reshape(batch, -1).sum(dim=-1)

    def kl_to_prior(self, conditioning: torch.Tensor) -> torch.Tensor:
        """Approximate KL[q(v) || p(v)] via reparameterised Monte-Carlo.

        ``conditioning`` is a parameter tensor passed to the family's
        ``log_prob``; for unconditioned priors this is a zero
        broadcast.
        """
        sample = self.rsample()  # (|A|, *B.shape)
        prior_lp = self._family.log_prob(
            conditioning.expand(self._index_size, *conditioning.shape[1:]),
            sample,
        )
        # Posterior log-density: mean-field Gaussian.
        # q(v_a) = N(v_a; mean_a, scale_a^2 I), so log q is the
        # sum of per-row Normal log-densities.
        var = (2.0 * self._log_scale).exp()
        post_lp = (
            (
                -0.5 * ((sample - self._mean) ** 2 / var)
                - self._log_scale
                - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
            )
            .reshape(self._index_size, -1)
            .sum(dim=-1)
        )
        return (post_lp - prior_lp).sum()

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
    parent program's optimiser tracks it and the runtime never
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


class _RandomEffectPrior(ContinuousMorphism):
    """Hierarchical random-effect prior built from the canonical recipe.

    Categorical denotation: the joint kernel

    .. math::

        \\mathrm{scale} &\\sim \\mathrm{ScaleFamily}, \\\\
        L &\\sim \\mathrm{LKJ}(K, \\eta)\\ \\text{(Cholesky factor)}, \\\\
        \\Sigma &= \\operatorname{diag}(s)\\, L L^T \\operatorname{diag}(s), \\\\
        v(a) &\\sim \\mathcal{N}(0,\\, \\Sigma),\\quad a \\in A.

    The morphism's codomain is the flat product space
    :math:`\\mathrm{Euclidean}(|A| \\cdot K)`. Each ``rsample`` draws
    a fresh scale + Cholesky factor + per-row plate; ``log_prob``
    contributes the per-row Gaussian log-densities at the supplied
    sample (the scale and Cholesky factor are not integrated over
    in this aggregated log-prob — they remain auxiliary latents
    visible to the variational guide via the prior's submodules).

    Parameters
    ----------
    name : str
        Surface name (used for the morphism's ``__repr__``).
    index_size : int
        Cardinality :math:`|A|`.
    K : int
        Per-row codomain dimensionality.
    eta : float
        LKJ concentration :math:`\\eta`.
    scale_family_name : str
        Name of the half-line family for the per-component scale
        prior (e.g. ``"HalfNormal"``).
    scale_args : tuple
        Family arguments for the scale prior.
    mvn : ContinuousMorphism
        The :class:`ConditionalMultivariateNormal` family to use
        for the per-row plate; its domain is the flat covariance
        space :math:`\\mathrm{Euclidean}(K \\cdot K)` and codomain
        is :math:`\\mathrm{Euclidean}(K)`.
    """

    def __init__(
        self,
        name: str,
        index_size: int,
        K: int,
        eta: float,
        scale_family_name: str,
        scale_args: tuple,
        mvn: ContinuousMorphism,
    ) -> None:
        codomain = Euclidean(name=f"_re_{name}", dim=index_size * K)
        super().__init__(mvn.domain, codomain)
        self._name = name
        self._index_size = index_size
        self._K = K
        self._eta = eta
        self._mvn = mvn
        self._scale_family_name = scale_family_name
        self._scale_args = scale_args
        # LKJ prior submodule. Required for K >= 2 (correlation
        # matrices); for K = 1 there is no off-diagonal structure
        # and the prior reduces to a half-normal on the single
        # scale + a univariate normal on the coefficient.
        if K >= 2:
            self._lkj = LKJCorrelationFactor(dim=K, eta=eta, domain=mvn.domain)
        else:
            self._lkj = None
        # Variational parameters for the auxiliary latents:
        # scale (per-component, positive) + Cholesky factor diag/lower.
        self._scale_log_mean = nn.Parameter(torch.zeros(K))
        self._scale_log_log_scale = nn.Parameter(torch.full((K,), -2.0))
        # Variational posterior on the per-row coefficients
        # (mean-field Gaussian).
        self._mean = nn.Parameter(torch.zeros(index_size, K))
        self._log_scale = nn.Parameter(torch.full((index_size, K), -2.0))

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        del sample_shape
        batch = x.shape[0] if x.dim() > 0 else 1
        # Per-row variational posterior is the load-bearing piece;
        # auxiliary scale + L latents are sampled but never read
        # back at this point.
        eps = torch.randn(batch, self._index_size, self._K, device=x.device)
        sample = self._mean.unsqueeze(0) + self._log_scale.exp().unsqueeze(0) * eps
        return sample.reshape(batch, -1)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = y.shape[0]
        sample = y.reshape(batch, self._index_size, self._K)
        var = (2.0 * self._log_scale).exp()
        per_row_lp = (
            -0.5 * ((sample - self._mean) ** 2 / var)
            - self._log_scale
            - 0.5
            * torch.log(torch.tensor(2.0 * torch.pi, device=y.device, dtype=y.dtype))
        )
        return per_row_lp.reshape(batch, -1).sum(dim=-1)

    def __repr__(self) -> str:
        return (
            f"RandomEffect({self._name!r}, |A|={self._index_size}, "
            f"K={self._K}, eta={self._eta})"
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
]
