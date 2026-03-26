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


# -- module-level singletons ------------------------------------------------

LUKASIEWICZ = LukasiewiczQuantale()
GODEL = GodelQuantale()
TROPICAL = TropicalQuantale()
