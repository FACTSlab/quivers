"""Natural transformations between endofunctors.

A natural transformation η: F ⇒ G is a family of morphisms
{η_A: F(A) → G(A)} indexed by objects, satisfying the naturality
condition:

    η_B ∘ F(f) = G(f) ∘ η_A   for all f: A → B

This module provides:

    NaturalTransformation (abstract)
    └── ComponentwiseNT — defined by a callable producing components
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from quivers.categorical.functors import Functor
    from quivers.core.morphisms import Morphism
    from quivers.core.objects import SetObject


class NaturalTransformation(ABC):
    """Abstract natural transformation η: F ⇒ G.

    Subclasses must implement ``component`` to produce η_A for
    each object A.

    Parameters
    ----------
    source : Functor
        The source functor F.
    target : Functor
        The target functor G.
    """

    def __init__(self, source: Functor, target: Functor) -> None:
        self._source = source
        self._target = target

    @property
    def source(self) -> Functor:
        """The source functor F."""
        return self._source

    @property
    def target(self) -> Functor:
        """The target functor G."""
        return self._target

    @abstractmethod
    def component(self, obj: SetObject) -> Morphism:
        """Return the component η_A: F(A) → G(A).

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The morphism η_A: F(A) → G(A).
        """
        ...

    def verify_naturality(self, f: Morphism, atol: float = 1e-5) -> bool:
        """Verify the naturality condition for a given morphism.

        Checks that η_B ∘ F(f) ≈ G(f) ∘ η_A for f: A → B.

        Parameters
        ----------
        f : Morphism
            A morphism f: A → B to check naturality against.
        atol : float
            Absolute tolerance for tensor comparison.

        Returns
        -------
        bool
            True if the naturality square commutes within tolerance.
        """
        a = f.domain
        b = f.codomain

        eta_a = self.component(a)
        eta_b = self.component(b)

        # left path: η_B ∘ F(f)
        f_mapped = self._source.map_morphism(f)
        left = (f_mapped >> eta_b).tensor

        # right path: G(f) ∘ η_A
        g_mapped = self._target.map_morphism(f)
        right = (eta_a >> g_mapped).tensor

        return torch.allclose(left, right, atol=atol)

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self._source!r} ⇒ {self._target!r})"


class ComponentwiseNT(NaturalTransformation):
    """A natural transformation defined by a callable.

    Wraps a function obj ↦ morphism into a NaturalTransformation.
    Useful for constructing natural transformations from explicit
    component definitions.

    Parameters
    ----------
    source : Functor
        The source functor F.
    target : Functor
        The target functor G.
    component_fn : Callable[[SetObject], Morphism]
        A function that, given an object A, returns η_A: F(A) → G(A).
    """

    def __init__(
        self,
        source: Functor,
        target: Functor,
        component_fn: Callable[[SetObject], Morphism],
    ) -> None:
        super().__init__(source, target)
        self._component_fn = component_fn

    def component(self, obj: SetObject) -> Morphism:
        """Return η_A by calling the stored function."""
        return self._component_fn(obj)
