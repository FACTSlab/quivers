"""Sum-product composition quantale for stochastic matrices.

Implements the MarkovQuantale which governs composition of finite
stochastic matrices via sum-product (matrix multiplication).
"""

from __future__ import annotations

import torch

from quivers.core.quantales import Quantale


class MarkovQuantale(Quantale):
    """Sum-product composition for stochastic matrices.

    Implements the composition rule of FinStoch:

        (g ∘ f)(a, c) = Σ_b f(a, b) · g(b, c)

    This is standard matrix multiplication. Formally:

        ⊗ = product:     a ⊗ b = a · b
        ⋁ = sum:         ⋁_i x_i = Σ_i x_i
        ⋀ = product:     ⋀_i x_i = Π_i x_i
        ¬ = complement:  ¬a = 1 - a
        I = 1.0
        ⊥ = 0.0

    Note
    ----
    This is not a true quantale in the lattice-theoretic sense
    because Σ is not idempotent (a ⋁ a = 2a ≠ a). It is a
    commutative semiring ([0,∞), +, ×, 0, 1) whose composition
    formula matches the quantale interface. The important property
    is that composition of row-stochastic matrices yields
    row-stochastic matrices.
    """

    @property
    def name(self) -> str:
        return "Markov"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Pointwise product."""
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Summation (marginalization in probability)."""
        if isinstance(dim, int):
            dim = (dim,)

        return t.sum(dim=dim)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Product (joint probability under independence)."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.prod(dim=d)

        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Complement: ¬p = 1 - p."""
        return 1.0 - t

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0

    def compose(
        self,
        m: torch.Tensor,
        n: torch.Tensor,
        n_contract: int,
    ) -> torch.Tensor:
        """Sum-product composition (matrix multiplication).

        Computes: result[d..., c...] = Σ_{s...} m[d..., s...] · n[s..., c...]

        Uses the default quantale compose, which expands + tensor_op + join.
        For 2D tensors this is equivalent to torch.mm.

        Parameters
        ----------
        m : torch.Tensor
            Left stochastic matrix of shape (*domain, *shared).
        n : torch.Tensor
            Right stochastic matrix of shape (*shared, *codomain).
        n_contract : int
            Number of shared dimensions.

        Returns
        -------
        torch.Tensor
            Composed matrix of shape (*domain, *codomain).
        """
        return super().compose(m, n, n_contract)


MARKOV = MarkovQuantale()
