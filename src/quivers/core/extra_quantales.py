"""Additional quantales for V-enriched categories.

This module extends the base quantales (ProductFuzzy, BooleanQuantale)
with three additional enrichment algebras:

    LukasiewiczQuantale — [0,1] with Łukasiewicz t-norm
    GodelQuantale       — [0,1] with Gödel (min) t-norm
    TropicalQuantale    — [0,∞] with + as tensor, inf as join

Each quantale gives a different category of relations:

    - Łukasiewicz: Resource-sensitive fuzzy relations.
      ⊗ = max(a + b - 1, 0), good for reasoning about bounded resources.

    - Gödel: Possibilistic relations with min semantics.
      ⊗ = min(a, b), giving the weakest fuzzy logic.

    - Tropical: Lawvere metric spaces (generalized metrics).
      ⊗ = a + b (distances add), ⋁ = inf (shortest path).
      Note: values are in [0, ∞], unit = 0, zero = ∞.
"""

from __future__ import annotations

import itertools

import torch

from quivers.core.quantales import Quantale


class LukasiewiczQuantale(Quantale):
    """[0,1] with Łukasiewicz t-norm and bounded sum.

    The Łukasiewicz t-norm is the strongest continuous t-norm:

        ⊗ = Łukasiewicz:   a ⊗ b = max(a + b - 1, 0)
        ⋁ = bounded sum:   ⋁_i x_i = min(1, ∑_i x_i)
        ⋀ = min:           ⋀_i x_i = min_i x_i
        ¬ = strong neg:    ¬a = 1 - a
        I = 1.0
        ⊥ = 0.0

    This quantale is useful for resource-sensitive reasoning where
    combining evidence can "cancel out" (unlike product t-norm).
    """

    @property
    def name(self) -> str:
        return "Lukasiewicz"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Łukasiewicz t-norm: max(a + b - 1, 0)."""
        return (a + b - 1.0).clamp(min=0.0)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Bounded sum: min(1, ∑_i x_i)."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t.sum(dim=dim).clamp(max=1.0)
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Min: ⋀_i x_i = min_i x_i."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values

        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Strong negation: ¬a = 1 - a."""
        return 1.0 - t

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0


class GodelQuantale(Quantale):
    """[0,1] with Gödel (min) t-norm.

    The weakest continuous t-norm:

        ⊗ = min:       a ⊗ b = min(a, b)
        ⋁ = max:       ⋁_i x_i = max_i x_i
        ⋀ = min:       ⋀_i x_i = min_i x_i
        ¬ = Gödel neg: ¬a = 1 if a = 0, else 0
        I = 1.0
        ⊥ = 0.0

    In a Gödel-enriched category, composition computes the
    "best worst-case" path — the minimax composition familiar
    from fuzzy graph theory.
    """

    @property
    def name(self) -> str:
        return "Godel"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Gödel t-norm: min(a, b)."""
        return torch.min(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Max: ⋁_i x_i = max_i x_i."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.max(dim=d).values

        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Min: ⋀_i x_i = min_i x_i."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values

        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Gödel negation: ¬a = 1 if a == 0, else 0."""
        return (t == 0.0).float()

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0


class TropicalQuantale(Quantale):
    """[0, ∞] with addition and infimum (tropical semiring).

    This is the Lawvere enrichment for generalized metric spaces:

        ⊗ = addition:     a ⊗ b = a + b (distances compose additively)
        ⋁ = infimum:      ⋁_i x_i = min_i x_i (shortest path)
        ⋀ = supremum:     ⋀_i x_i = max_i x_i (longest path)
        ¬ = n/a:          negation is not well-defined for metrics
        I = 0.0           (zero distance)
        ⊥ = ∞             (infinite distance / unreachable)

    Composition computes shortest-path distances:

        (g ∘ f)(a, c) = inf_b [f(a, b) + g(b, c)]

    This is the tropical matrix multiplication, a.k.a. the
    (min, +) semiring product.

    Note
    ----
    We use torch.inf for ⊥ (unreachable) and 0.0 for I (identity).
    The identity tensor has 0 on the diagonal and ∞ elsewhere.
    """

    @property
    def name(self) -> str:
        return "Tropical"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Tropical tensor: a + b."""
        return a + b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Infimum (min): shortest path."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values

        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Supremum (max): longest path."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.max(dim=d).values

        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Negation is not meaningful for the tropical quantale.

        Returns the additive inverse as a best-effort approximation,
        but note this is outside [0, ∞] for positive values.

        Raises
        ------
        NotImplementedError
            Always, since tropical negation is not well-defined.
        """
        raise NotImplementedError(
            "negation is not well-defined for the tropical quantale"
        )

    @property
    def unit(self) -> float:
        return 0.0

    @property
    def zero(self) -> float:
        return float("inf")

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        """Identity with 0 on diagonal and ∞ elsewhere.

        Override because the default uses self.zero for off-diagonal
        and self.unit for diagonal, which is correct here (0 on diag,
        ∞ off), but we use torch.inf explicitly for clarity.

        Parameters
        ----------
        obj_shape : tuple[int, ...]
            Shape of the object.

        Returns
        -------
        torch.Tensor
            Identity tensor.
        """
        full_shape = obj_shape + obj_shape
        result = torch.full(full_shape, float("inf"))
        ndim = len(obj_shape)

        if ndim == 1:
            n = obj_shape[0]

            for i in range(n):
                result[i, i] = 0.0

        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 0.0

        return result


class MaxPlusQuantale(Quantale):
    """Tropical *max-plus* semiring on :math:`(-\\infty, \\infty]`.

    Distinct from :class:`TropicalQuantale` (which is *min-plus*,
    suited to shortest-path aggregations): the join here is
    :math:`\\max` and the tensor is :math:`+`. This is the Viterbi
    / best-path semiring — the canonical algebra for MAP decoding
    in HMMs, CRFs, and weighted automata.
    """

    @property
    def name(self) -> str:
        return "MaxPlus"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Max-plus tensor: ``a + b`` (real-valued addition)."""
        return a + b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Supremum (max): best path."""
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.max(dim=d).values
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Infimum (min): worst path."""
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "negation is not well-defined for the max-plus quantale"
        )

    @property
    def unit(self) -> float:
        """Monoidal unit: 0 (a + 0 = a)."""
        return 0.0

    @property
    def zero(self) -> float:
        """Join unit: -inf (max(-inf, a) = a)."""
        return -float("inf")

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        full_shape = obj_shape + obj_shape
        result = torch.full(full_shape, -float("inf"))
        ndim = len(obj_shape)
        if ndim == 1:
            n = obj_shape[0]
            for i in range(n):
                result[i, i] = 0.0
        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 0.0
        return result


class LogProbQuantale(Quantale):
    """Log-space sum-product semiring on :math:`(-\\infty, 0]`.

    Tensor is real addition (probability multiplication in log-
    space) and join is :func:`torch.logsumexp` (probability
    summation in log-space). Pairs naturally with float32
    numerics for hierarchical-Bayes log-likelihood pipelines: the
    "Markov" quantale's sum-product is numerically delicate for
    long chains; LogProb is the same algebra moved to log-space
    where addition and logsumexp are both well-conditioned.
    """

    @property
    def name(self) -> str:
        return "LogProb"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Log-tensor: ``a + b`` (probability multiplication in
        log-space)."""
        return a + b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Logsumexp: numerically stable log-of-sum-of-exp."""
        if isinstance(dim, int):
            dim = (dim,)
        return torch.logsumexp(t, dim=dim)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Min in log-space — meaningful for "least likely" paths."""
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "negation is not well-defined for the log-prob quantale"
        )

    @property
    def unit(self) -> float:
        """Monoidal unit: 0 (log 1 = 0)."""
        return 0.0

    @property
    def zero(self) -> float:
        """Join unit: -inf (log 0 = -inf)."""
        return -float("inf")

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        full_shape = obj_shape + obj_shape
        result = torch.full(full_shape, -float("inf"))
        ndim = len(obj_shape)
        if ndim == 1:
            n = obj_shape[0]
            for i in range(n):
                result[i, i] = 0.0
        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 0.0
        return result


class RealQuantale(Quantale):
    """Sum-product semiring on the real numbers
    :math:`(\\mathbb{R}, +, \\cdot)`.

    The canonical numeric semiring: addition is the lattice join,
    multiplication the monoidal tensor. Distinct from
    :class:`ProductFuzzy` (whose join is noisy-OR on ``[0, 1]``)
    and from :class:`MarkovQuantale` (which constrains rows to
    sum to 1). Use when entries are unbounded real weights with
    no probability interpretation — adjacency-matrix weights,
    bilinear scores, signed similarities, regression
    coefficients. Mirrors the ``RealWeight`` semiring shipped by
    ``arcweight``.
    """

    @property
    def name(self) -> str:
        return "Real"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Real multiplication: ``a · b``."""
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Sum along the contracted axes."""
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.sum(dim=d)
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Min along the contracted axes (the meet of the real-
        line lattice ordered by ≤)."""
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Real-line additive inverse: ``-t``."""
        return -t

    @property
    def unit(self) -> float:
        """Monoidal unit: 1 (a · 1 = a)."""
        return 1.0

    @property
    def zero(self) -> float:
        """Join unit: 0 (a + 0 = a)."""
        return 0.0

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        """Identity matrix: ones on the diagonal, zeros elsewhere."""
        full_shape = obj_shape + obj_shape
        result = torch.zeros(full_shape)
        ndim = len(obj_shape)
        if ndim == 1:
            n = obj_shape[0]
            for i in range(n):
                result[i, i] = 1.0
        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 1.0
        return result


class ProbabilityQuantale(Quantale):
    """Sum-product semiring on ``[0, 1]``.

    Same operations as :class:`RealQuantale` but restricted to the
    unit interval: entries are clamped to ``[0, 1]`` at every
    tensor op so the result is interpretable as a probability.
    Distinct from :class:`ProductFuzzy` (whose join is noisy-OR
    rather than sum) and :class:`MarkovQuantale` (which enforces
    row-stochasticity). Use when entries are *unnormalised*
    probabilities — confusion-matrix entries, soft co-occurrence
    counts, fuzzy-set membership with additive aggregation.
    Mirrors the ``ProbabilityWeight`` semiring shipped by
    ``arcweight``.
    """

    @property
    def name(self) -> str:
        return "Probability"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Probability multiplication: ``a · b`` clamped to
        ``[0, 1]``."""
        return (a * b).clamp(min=0.0, max=1.0)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Sum along the contracted axes, clamped to ``[0, 1]``."""
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.sum(dim=d)
        return result.clamp(min=0.0, max=1.0)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Min along the contracted axes."""
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Complement on the unit interval: ``1 - t``."""
        return (1.0 - t).clamp(min=0.0, max=1.0)

    @property
    def unit(self) -> float:
        """Monoidal unit: 1."""
        return 1.0

    @property
    def zero(self) -> float:
        """Join unit: 0."""
        return 0.0

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        full_shape = obj_shape + obj_shape
        result = torch.zeros(full_shape)
        ndim = len(obj_shape)
        if ndim == 1:
            n = obj_shape[0]
            for i in range(n):
                result[i, i] = 1.0
        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 1.0
        return result


class CountingQuantale(Quantale):
    """Sum-product semiring on the non-negative integers
    :math:`(\\mathbb{N}, +, \\cdot)`.

    Counting algebra: composition counts the number of distinct
    paths through a structure. Distinct from
    :class:`BooleanQuantale` (which collapses to existence) and
    from :class:`RealQuantale` (which allows non-integer / signed
    weights). Used in weighted parsing, derivation counting,
    enumeration over discrete structures. Mirrors the
    ``IntegerWeight`` semiring shipped by ``arcweight``.

    The underlying tensor is float-typed (PyTorch's autograd
    requires it) but operations are integer-respecting: the join
    is plain summation and the tensor product is multiplication.
    """

    @property
    def name(self) -> str:
        return "Counting"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Integer multiplication: ``a · b``."""
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Sum along the contracted axes."""
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.sum(dim=d)
        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Min along the contracted axes."""
        if isinstance(dim, int):
            dim = (dim,)
        result = t
        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values
        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "negation is not well-defined for the counting "
            "(non-negative integer) quantale"
        )

    @property
    def unit(self) -> float:
        """Monoidal unit: 1 (the empty product of paths)."""
        return 1.0

    @property
    def zero(self) -> float:
        """Join unit: 0 (no paths)."""
        return 0.0

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        full_shape = obj_shape + obj_shape
        result = torch.zeros(full_shape)
        ndim = len(obj_shape)
        if ndim == 1:
            n = obj_shape[0]
            for i in range(n):
                result[i, i] = 1.0
        else:
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = 1.0
        return result


# -- module-level singletons ------------------------------------------------

LUKASIEWICZ = LukasiewiczQuantale()
GODEL = GodelQuantale()
TROPICAL = TropicalQuantale()
MAX_PLUS = MaxPlusQuantale()
LOG_PROB = LogProbQuantale()
REAL = RealQuantale()
PROBABILITY = ProbabilityQuantale()
COUNTING = CountingQuantale()
