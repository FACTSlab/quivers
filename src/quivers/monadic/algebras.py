"""Algebras and coalgebras for monads and comonads.

An algebra (A, α) for a monad (T, η, μ) consists of an object A
and a morphism α: T(A) → A satisfying:

    α ∘ η_A = id_A           (unit law)
    α ∘ μ_A = α ∘ T(α)       (associativity)

Dually, a coalgebra (A, γ) for a comonad (W, ε, δ) consists of an
object A and a morphism γ: A → W(A) satisfying:

    ε_A ∘ γ = id_A           (counit law)
    δ_A ∘ γ = W(γ) ∘ γ       (coassociativity)

The Eilenberg-Moore category of a monad T has T-algebras as objects
and algebra homomorphisms as morphisms. The comparison functor
K: Kl(T) → EM(T) relates the Kleisli and Eilenberg-Moore categories.

This module provides:

    Algebra (abstract)
    ├── FreeAlgebra            — free T-algebra μ_A: T(T(A)) → T(A)
    └── ObservedAlgebra        — algebra from a concrete structure map

    Coalgebra (abstract)
    ├── CofreeCoalgebra        — cofree W-coalgebra δ_A: W(A) → W(W(A))
    └── ObservedCoalgebra      — coalgebra from a concrete structure map

    EilenbergMooreCategory     — the category of algebras for a monad
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.objects import SetObject
from quivers.core.morphisms import Morphism, observed, identity
from quivers.core.quantales import PRODUCT_FUZZY, Quantale
from quivers.monadic.monads import Monad


class Algebra(ABC):
    """Abstract algebra for a monad.

    An algebra (A, α) for monad T consists of a carrier object A
    and a structure map α: T(A) → A.

    Parameters
    ----------
    monad : Monad
        The monad T.
    carrier : SetObject
        The carrier object A.
    """

    def __init__(self, monad: Monad, carrier: SetObject) -> None:
        self._monad = monad
        self._carrier = carrier

    @property
    def monad(self) -> Monad:
        """The monad T."""
        return self._monad

    @property
    def carrier(self) -> SetObject:
        """The carrier object A."""
        return self._carrier

    @abstractmethod
    def structure_map(self) -> Morphism:
        """The structure map α: T(A) → A.

        Returns
        -------
        Morphism
            The algebra structure map.
        """
        ...

    def verify_unit_law(self, atol: float = 1e-5) -> bool:
        """Verify α ∘ η_A = id_A.

        Parameters
        ----------
        atol : float
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            True if the unit law holds within tolerance.
        """
        alpha = self.structure_map()
        eta = self._monad.unit(self._carrier)
        result = (eta >> alpha).tensor
        expected = identity(self._carrier).tensor

        return torch.allclose(result, expected, atol=atol)

    def verify_associativity(self, atol: float = 1e-5) -> bool:
        """Verify α ∘ μ_A = α ∘ T(α).

        Parameters
        ----------
        atol : float
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            True if associativity holds within tolerance.
        """
        alpha = self.structure_map()
        mu = self._monad.multiply(self._carrier)

        # left side: α ∘ μ_A
        left = (mu >> alpha).tensor

        # right side: α ∘ T(α)
        t_alpha = self._monad.endofunctor.map_morphism(alpha)
        right = (t_alpha >> alpha).tensor

        return torch.allclose(left, right, atol=atol)

    def __repr__(self) -> str:
        return f"Algebra({self._monad!r}, carrier={self._carrier!r})"


class FreeAlgebra(Algebra):
    """The free T-algebra on an object A.

    The carrier is T(A) and the structure map is μ_A: T(T(A)) → T(A).
    This is the algebra that the comparison functor assigns to
    each object in the Kleisli category.

    Parameters
    ----------
    monad : Monad
        The monad T.
    obj : SetObject
        The base object A (the carrier will be T(A)).
    """

    def __init__(self, monad: Monad, obj: SetObject) -> None:
        carrier = monad.endofunctor.map_object(obj)
        super().__init__(monad, carrier)
        self._base = obj

    @property
    def base(self) -> SetObject:
        """The base object A (carrier is T(A))."""
        return self._base

    def structure_map(self) -> Morphism:
        """μ_A: T(T(A)) → T(A)."""
        return self._monad.multiply(self._base)


class ObservedAlgebra(Algebra):
    """An algebra defined by an explicit structure map tensor.

    Parameters
    ----------
    monad : Monad
        The monad T.
    carrier : SetObject
        The carrier object A.
    structure_tensor : torch.Tensor
        The tensor for α: T(A) → A.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(
        self,
        monad: Monad,
        carrier: SetObject,
        structure_tensor: torch.Tensor,
        quantale: Quantale | None = None,
    ) -> None:
        super().__init__(monad, carrier)
        q = quantale if quantale is not None else PRODUCT_FUZZY
        ta = monad.endofunctor.map_object(carrier)
        self._morphism = observed(ta, carrier, structure_tensor, quantale=q)

    def structure_map(self) -> Morphism:
        """The stored structure map α: T(A) → A."""
        return self._morphism


class Coalgebra(ABC):
    """Abstract coalgebra for a comonad.

    A coalgebra (A, γ) for comonad W consists of a carrier object A
    and a structure map γ: A → W(A).

    Parameters
    ----------
    comonad : object
        The comonad W (a Comonad instance).
    carrier : SetObject
        The carrier object A.
    """

    def __init__(self, comonad: object, carrier: SetObject) -> None:
        self._comonad = comonad
        self._carrier = carrier

    @property
    def comonad(self) -> object:
        """The comonad W."""
        return self._comonad

    @property
    def carrier(self) -> SetObject:
        """The carrier object A."""
        return self._carrier

    @abstractmethod
    def structure_map(self) -> Morphism:
        """The structure map γ: A → W(A).

        Returns
        -------
        Morphism
            The coalgebra structure map.
        """
        ...

    def verify_counit_law(self, atol: float = 1e-5) -> bool:
        """Verify ε_A ∘ γ = id_A.

        Parameters
        ----------
        atol : float
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            True if the counit law holds within tolerance.
        """

        gamma = self.structure_map()
        eps = self._comonad.counit(self._carrier)  # type: ignore[union-attr]
        result = (gamma >> eps).tensor
        expected = identity(self._carrier).tensor

        return torch.allclose(result, expected, atol=atol)

    def verify_coassociativity(self, atol: float = 1e-5) -> bool:
        """Verify δ_A ∘ γ = W(γ) ∘ γ.

        Parameters
        ----------
        atol : float
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            True if coassociativity holds within tolerance.
        """

        gamma = self.structure_map()
        delta = self._comonad.comultiply(self._carrier)  # type: ignore[union-attr]

        # left side: δ_A ∘ γ
        left = (gamma >> delta).tensor

        # right side: W(γ) ∘ γ
        w_gamma = self._comonad.endofunctor.map_morphism(gamma)  # type: ignore[union-attr]
        right = (gamma >> w_gamma).tensor

        return torch.allclose(left, right, atol=atol)

    def __repr__(self) -> str:
        return f"Coalgebra({self._comonad!r}, carrier={self._carrier!r})"


class CofreeCoalgebra(Coalgebra):
    """The cofree W-coalgebra on an object A.

    The carrier is W(A) and the structure map is δ_A: W(A) → W(W(A)).

    Parameters
    ----------
    comonad : object
        The comonad W.
    obj : SetObject
        The base object A (the carrier will be W(A)).
    """

    def __init__(self, comonad: object, obj: SetObject) -> None:
        carrier = comonad.endofunctor.map_object(obj)  # type: ignore[union-attr]
        super().__init__(comonad, carrier)
        self._base = obj

    @property
    def base(self) -> SetObject:
        """The base object A (carrier is W(A))."""
        return self._base

    def structure_map(self) -> Morphism:
        """δ_A: W(A) → W(W(A))."""
        return self._comonad.comultiply(self._base)  # type: ignore[union-attr]


class ObservedCoalgebra(Coalgebra):
    """A coalgebra defined by an explicit structure map tensor.

    Parameters
    ----------
    comonad : object
        The comonad W.
    carrier : SetObject
        The carrier object A.
    structure_tensor : torch.Tensor
        The tensor for γ: A → W(A).
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(
        self,
        comonad: object,
        carrier: SetObject,
        structure_tensor: torch.Tensor,
        quantale: Quantale | None = None,
    ) -> None:
        super().__init__(comonad, carrier)
        q = quantale if quantale is not None else PRODUCT_FUZZY
        wa = comonad.endofunctor.map_object(carrier)  # type: ignore[union-attr]
        self._morphism = observed(carrier, wa, structure_tensor, quantale=q)

    def structure_map(self) -> Morphism:
        """The stored structure map γ: A → W(A)."""
        return self._morphism


class EilenbergMooreCategory:
    """The Eilenberg-Moore category of a monad.

    Objects are T-algebras. Morphisms (A, α) → (B, β) are base
    morphisms f: A → B satisfying β ∘ T(f) = f ∘ α, i.e., algebra
    homomorphisms.

    Parameters
    ----------
    monad : Monad
        The underlying monad T.
    """

    def __init__(self, monad: Monad) -> None:
        self._monad = monad

    @property
    def monad(self) -> Monad:
        """The underlying monad."""
        return self._monad

    def free_algebra(self, obj: SetObject) -> FreeAlgebra:
        """Construct the free T-algebra on an object.

        Parameters
        ----------
        obj : SetObject
            The base object A.

        Returns
        -------
        FreeAlgebra
            The free algebra with carrier T(A).
        """
        return FreeAlgebra(self._monad, obj)

    def is_homomorphism(
        self,
        f: Morphism,
        source_alg: Algebra,
        target_alg: Algebra,
        atol: float = 1e-5,
    ) -> bool:
        """Check if f: A → B is an algebra homomorphism.

        Verifies β ∘ T(f) = f ∘ α.

        Parameters
        ----------
        f : Morphism
            A morphism A → B between the carriers.
        source_alg : Algebra
            The source algebra (A, α).
        target_alg : Algebra
            The target algebra (B, β).
        atol : float
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            True if f is a homomorphism within tolerance.
        """
        alpha = source_alg.structure_map()
        beta = target_alg.structure_map()
        tf = self._monad.endofunctor.map_morphism(f)

        # left: β ∘ T(f)
        left = (tf >> beta).tensor

        # right: f ∘ α
        right = (alpha >> f).tensor

        return torch.allclose(left, right, atol=atol)

    def __repr__(self) -> str:
        return f"EilenbergMooreCategory({self._monad!r})"
