"""Quantales: enrichment algebras for V-enriched categories.

A commutative quantale Q = (L, ⊗, ⋁, ⋀, ¬, I, ⊥) provides the algebraic
structure that parameterizes composition in a V-enriched category:

    (g ∘ f)(a, c) = ⋁_b f(a, b) ⊗ g(b, c)

Different quantales yield different categories of relations:

    - BooleanQuantale:  {0,1} with ∧, ∨         → Rel (crisp relations)
    - ProductFuzzy:     [0,1] with ×, noisy-OR   → FuzzyRel (product t-norm)

The enrichment determines composition, identity, marginalization, and
quantification, all derived from the quantale's operations.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import torch

from quivers.core._util import clamp_probs


class Quantale(ABC):
    """Abstract commutative quantale for V-enriched categories.

    Subclasses must implement the six primitive operations.
    Composition and identity are derived but overridable.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this quantale."""
        ...

    # -- primitive operations ------------------------------------------------

    @abstractmethod
    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Monoidal product ⊗ (elementwise).

        Parameters
        ----------
        a : torch.Tensor
            Left operand.
        b : torch.Tensor
            Right operand (broadcastable with a).

        Returns
        -------
        torch.Tensor
            a ⊗ b, elementwise.
        """
        ...

    @abstractmethod
    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Join ⋁ — reduction for composition and existential (∃).

        Parameters
        ----------
        t : torch.Tensor
            Input tensor with values in L.
        dim : int or tuple[int, ...]
            Dimension(s) to reduce.

        Returns
        -------
        torch.Tensor
            Reduced tensor.
        """
        ...

    @abstractmethod
    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Meet ⋀ — reduction for universal quantification (∀).

        Parameters
        ----------
        t : torch.Tensor
            Input tensor with values in L.
        dim : int or tuple[int, ...]
            Dimension(s) to reduce.

        Returns
        -------
        torch.Tensor
            Reduced tensor.
        """
        ...

    @abstractmethod
    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Complement / negation ¬.

        Parameters
        ----------
        t : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            ¬t, elementwise.
        """
        ...

    @property
    @abstractmethod
    def unit(self) -> float:
        """Unit element I of the monoidal product ⊗."""
        ...

    @property
    @abstractmethod
    def zero(self) -> float:
        """Bottom element ⊥ (identity for ⋁)."""
        ...

    # -- derived operations --------------------------------------------------

    def compose(
        self,
        m: torch.Tensor,
        n: torch.Tensor,
        n_contract: int,
    ) -> torch.Tensor:
        """V-enriched composition.

        Computes: result[d..., c...] = ⋁_{s...} m[d..., s...] ⊗ n[s..., c...]

        Override for numerical stability in specific quantales.

        Parameters
        ----------
        m : torch.Tensor
            Left tensor of shape (*domain, *shared).
        n : torch.Tensor
            Right tensor of shape (*shared, *codomain).
        n_contract : int
            Number of shared dimensions to contract.

        Returns
        -------
        torch.Tensor
            Composed tensor of shape (*domain, *codomain).
        """
        if n_contract < 1:
            raise ValueError(f"n_contract must be >= 1, got {n_contract}")

        # validate shared dimensions
        shared_m = m.shape[-n_contract:]
        shared_n = n.shape[:n_contract]

        if shared_m != shared_n:
            raise ValueError(
                f"shared dimensions do not match: "
                f"m trailing {shared_m} != n leading {shared_n}"
            )

        n_domain = m.ndim - n_contract
        n_codomain = n.ndim - n_contract

        # broadcast for element-wise tensor_op
        m_expanded = m.reshape(*m.shape, *([1] * n_codomain))
        n_expanded = n.reshape(*([1] * n_domain), *n.shape)

        product = self.tensor_op(m_expanded, n_expanded)

        # join over shared dims
        contract_dims = tuple(range(n_domain, n_domain + n_contract))
        return self.join(product, dim=contract_dims)

    def identity_tensor(self, obj_shape: tuple[int, ...]) -> torch.Tensor:
        """Identity morphism tensor for an object with given shape.

        Returns a tensor of shape (*obj_shape, *obj_shape) with
        the unit value on the diagonal and zero elsewhere.

        Parameters
        ----------
        obj_shape : tuple[int, ...]
            Shape of the object (e.g., (n,) for FinSet(n)).

        Returns
        -------
        torch.Tensor
            Identity tensor.
        """
        full_shape = obj_shape + obj_shape
        result = torch.full(full_shape, self.zero)
        ndim = len(obj_shape)

        if ndim == 1:
            # simple case: (n, n) matrix
            n = obj_shape[0]

            for i in range(n):
                result[i, i] = self.unit

        else:
            # multi-dimensional: iterate over all index tuples
            for idx in itertools.product(*(range(s) for s in obj_shape)):
                result[idx + idx] = self.unit

        return result

    def is_compatible(self, other: Quantale) -> bool:
        """Check if two quantales are compatible for composition.

        Parameters
        ----------
        other : Quantale
            The other quantale.

        Returns
        -------
        bool
            True if morphisms from these quantales can compose.
        """
        return type(self) is type(other)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ProductFuzzy(Quantale):
    """[0,1] with product t-norm and probabilistic sum (noisy-OR).

    This is the enrichment for the Kleisli category of the fuzzy
    powerset monad with the product t-norm:

        ⊗ = product:      a ⊗ b = a * b
        ⋁ = noisy-OR:     ⋁_i x_i = 1 - ∏_i (1 - x_i)
        ⋀ = product:      ⋀_i x_i = ∏_i x_i
        ¬ = complement:    ¬a = 1 - a
        I = 1.0
        ⊥ = 0.0

    Composition uses log-space for numerical stability.
    """

    @property
    def name(self) -> str:
        return "ProductFuzzy"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Noisy-OR in log-space: 1 - exp(∑ log(1 - t))."""
        if isinstance(dim, int):
            dim = (dim,)

        t_clamped = clamp_probs(t)
        log_complement = torch.log1p(-t_clamped)
        sum_log = log_complement.sum(dim=dim)

        return -torch.expm1(sum_log)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Product (fuzzy AND): ∏_i t_i."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.prod(dim=d)

        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
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
        """Override for log-space numerical stability.

        Computes noisy-OR contraction matching the existing
        noisy_or_contract implementation exactly.
        """
        if n_contract < 1:
            raise ValueError(f"n_contract must be >= 1, got {n_contract}")

        shared_m = m.shape[-n_contract:]
        shared_n = n.shape[:n_contract]

        if shared_m != shared_n:
            raise ValueError(
                f"shared dimensions do not match: "
                f"m trailing {shared_m} != n leading {shared_n}"
            )

        n_domain = m.ndim - n_contract
        n_codomain = n.ndim - n_contract
        n_shared = n_contract

        m_expanded = m.reshape(*m.shape, *([1] * n_codomain))
        n_expanded = n.reshape(*([1] * n_domain), *n.shape)

        product = m_expanded * n_expanded

        # log-space noisy-OR for stability
        product_clamped = clamp_probs(product)
        log_complement = torch.log1p(-product_clamped)

        contract_dims = tuple(range(n_domain, n_domain + n_shared))
        sum_log = log_complement.sum(dim=contract_dims)

        return -torch.expm1(sum_log)


class BooleanQuantale(Quantale):
    """{0, 1} with logical AND and OR.

    The enrichment for the category Rel of crisp binary relations:

        ⊗ = AND:     a ⊗ b = a ∧ b
        ⋁ = OR:      ⋁_i x_i = max_i x_i
        ⋀ = AND:     ⋀_i x_i = min_i x_i
        ¬ = NOT:     ¬a = 1 - a
        I = 1.0
        ⊥ = 0.0

    Works on float tensors with values in {0.0, 1.0}. Intermediate
    fuzzy values are rounded.
    """

    @property
    def name(self) -> str:
        return "Boolean"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Logical AND via product (exact for {0,1} inputs)."""
        return a * b

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Logical OR via iterated max."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.max(dim=d).values

        return result

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Logical AND via iterated min."""
        if isinstance(dim, int):
            dim = (dim,)

        result = t

        for d in sorted(dim, reverse=True):
            result = result.min(dim=d).values

        return result

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    @property
    def unit(self) -> float:
        return 1.0

    @property
    def zero(self) -> float:
        return 0.0


# -- module-level singletons ------------------------------------------------

PRODUCT_FUZZY = ProductFuzzy()
BOOLEAN = BooleanQuantale()
