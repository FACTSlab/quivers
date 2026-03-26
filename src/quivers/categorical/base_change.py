"""Change of enrichment (change of base) for V-enriched categories.

A lax monoidal functor F: V → W between quantales induces a
change-of-base 2-functor from V-Cat to W-Cat. This transforms
all morphism tensors by applying F elementwise, changing the
enrichment algebra from V to W.

This module provides:

    BaseChange (abstract)
    ├── BoolToFuzzy   — inclusion {0,1} ↪ [0,1]
    └── FuzzyToBool   — thresholding [0,1] → {0,1}
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.objects import SetObject
from quivers.core.morphisms import Morphism, ObservedMorphism, observed
from quivers.core.quantales import (
    Quantale,
    ProductFuzzy,
    BooleanQuantale,
    PRODUCT_FUZZY,
    BOOLEAN,
)


class BaseChange(ABC):
    """Abstract change of enrichment from quantale V to quantale W.

    Subclasses must implement source, target, and apply_to_values.
    apply_to_morphism is derived.
    """

    @property
    @abstractmethod
    def source(self) -> Quantale:
        """The source quantale V."""
        ...

    @property
    @abstractmethod
    def target(self) -> Quantale:
        """The target quantale W."""
        ...

    @abstractmethod
    def apply_to_values(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the value-level map V → W elementwise.

        Parameters
        ----------
        tensor : torch.Tensor
            A V-valued tensor.

        Returns
        -------
        torch.Tensor
            The corresponding W-valued tensor.
        """
        ...

    def apply_to_morphism(self, morph: Morphism) -> ObservedMorphism:
        """Transform a V-morphism to a W-morphism.

        Applies the value-level map to the morphism's tensor and
        returns an observed morphism with the target quantale.

        Parameters
        ----------
        morph : Morphism
            A morphism enriched over V.

        Returns
        -------
        ObservedMorphism
            The same morphism re-enriched over W.
        """
        new_tensor = self.apply_to_values(morph.tensor.detach())

        return observed(
            morph.domain,
            morph.codomain,
            new_tensor,
            quantale=self.target,
        )

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self.source.name} → {self.target.name})"


class BoolToFuzzy(BaseChange):
    """Inclusion of the boolean quantale into the fuzzy quantale.

    The identity map {0, 1} ↪ [0, 1]. Boolean-valued tensors are
    already valid fuzzy-valued tensors, so this is the identity
    on tensor values.
    """

    @property
    def source(self) -> BooleanQuantale:
        return BOOLEAN

    @property
    def target(self) -> ProductFuzzy:
        return PRODUCT_FUZZY

    def apply_to_values(self, tensor: torch.Tensor) -> torch.Tensor:
        """Identity: {0,1} values are already in [0,1]."""
        return tensor.clone()


class FuzzyToBool(BaseChange):
    """Thresholding from the fuzzy quantale to the boolean quantale.

    Applies a threshold: values >= threshold become 1, others become 0.
    This is a right adjoint to BoolToFuzzy.

    Parameters
    ----------
    threshold : float
        The threshold value. Must be in (0, 1). Default 0.5.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"threshold must be in (0, 1), got {threshold}"
            )

        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """The threshold value."""
        return self._threshold

    @property
    def source(self) -> ProductFuzzy:
        return PRODUCT_FUZZY

    @property
    def target(self) -> BooleanQuantale:
        return BOOLEAN

    def apply_to_values(self, tensor: torch.Tensor) -> torch.Tensor:
        """Threshold: values >= threshold → 1, else → 0."""
        return (tensor >= self._threshold).float()
