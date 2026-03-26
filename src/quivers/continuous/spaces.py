"""Continuous measurable spaces for the hybrid architecture.

This module provides continuous space objects that serve as domains
and codomains for continuous morphisms. These complement the finite
set objects (FinSet, ProductSet, etc.) used by discrete morphisms.

Spaces provided
---------------
ContinuousSpace   — abstract base for all continuous spaces
Euclidean         — R^d with optional bounds [low, high]^d
Simplex           — probability simplex {x in R^d : x_i >= 0, sum = 1}
PositiveReals     — (0, inf)^d
ProductSpace      — cartesian product of continuous spaces
UnitInterval      — convenience constructor for bounded [0,1]^d
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


class ContinuousSpace(ABC):
    """Abstract base for continuous measurable spaces.

    Unlike SetObject (which has a finite cardinality and a tensor shape),
    a ContinuousSpace is characterized by its event dimensionality and
    support constraints.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Event dimensionality (number of real-valued components)."""
        ...

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of a single sample from this space."""
        return (self.dim,)

    @abstractmethod
    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Check whether points lie in the support.

        Parameters
        ----------
        x : torch.Tensor
            Points to check. Shape (..., dim).

        Returns
        -------
        torch.Tensor
            Boolean tensor. Shape (...).
        """
        ...

    def sample_uniform(self, n: int) -> torch.Tensor:
        """Sample n points uniformly from the space.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        torch.Tensor
            Samples of shape (n, dim).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support uniform sampling"
        )

    def __mul__(self, other: ContinuousSpace) -> ProductSpace:
        """Cartesian product via * operator."""
        if not isinstance(other, ContinuousSpace):
            return NotImplemented

        return ProductSpace(self, other)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {self.dim})"


@dataclass(frozen=True)
class Euclidean(ContinuousSpace):
    """Euclidean space R^d, optionally bounded to [low, high]^d.

    Parameters
    ----------
    _name : str
        Human-readable name.
    _dim : int
        Dimensionality.
    low : float or None
        Lower bound per dimension (None for unbounded below).
    high : float or None
        Upper bound per dimension (None for unbounded above).

    Examples
    --------
    >>> R3 = Euclidean("position", 3)
    >>> R3.dim
    3
    >>> bounded = Euclidean("unit", 2, low=0.0, high=1.0)
    >>> bounded.is_bounded
    True
    """

    _name: str
    _dim: int
    low: float | None = None
    high: float | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_bounded(self) -> bool:
        """Whether the space has finite bounds on all sides."""
        return self.low is not None and self.high is not None

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.ones(
            x.shape[:-1], dtype=torch.bool, device=x.device
        )

        if self.low is not None:
            result = result & (x >= self.low).all(dim=-1)

        if self.high is not None:
            result = result & (x <= self.high).all(dim=-1)

        return result

    def sample_uniform(self, n: int) -> torch.Tensor:
        if not self.is_bounded:
            raise ValueError(
                "cannot sample uniformly from unbounded Euclidean space"
            )

        return (
            torch.rand(n, self._dim) * (self.high - self.low) + self.low
        )

    def __repr__(self) -> str:
        bounds = ""

        if self.low is not None or self.high is not None:
            bounds = f", low={self.low}, high={self.high}"

        return f"Euclidean({self._name!r}, {self._dim}{bounds})"


def UnitInterval(name: str, dim: int = 1) -> Euclidean:
    """Create a [0, 1]^d bounded Euclidean space.

    Parameters
    ----------
    name : str
        Human-readable name.
    dim : int
        Dimensionality (default 1).

    Returns
    -------
    Euclidean
        A Euclidean space bounded to [0, 1]^d.
    """
    return Euclidean(name, dim, low=0.0, high=1.0)


@dataclass(frozen=True)
class Simplex(ContinuousSpace):
    """The probability simplex {x in R^d : x_i >= 0, sum(x_i) = 1}.

    Parameters
    ----------
    _name : str
        Human-readable name.
    _dim : int
        Number of components.

    Examples
    --------
    >>> S = Simplex("probs", 3)
    >>> S.dim
    3
    >>> S.contains(torch.tensor([[0.3, 0.3, 0.4]]))
    tensor([True])
    """

    _name: str
    _dim: int

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        nonneg = (x >= -1e-7).all(dim=-1)
        sums_to_one = (x.sum(dim=-1) - 1.0).abs() < 1e-5
        return nonneg & sums_to_one

    def sample_uniform(self, n: int) -> torch.Tensor:
        # uniform on simplex via exponential distribution
        e = torch.distributions.Exponential(1.0).sample((n, self._dim))
        return e / e.sum(dim=-1, keepdim=True)

    def __repr__(self) -> str:
        return f"Simplex({self._name!r}, {self._dim})"


@dataclass(frozen=True)
class PositiveReals(ContinuousSpace):
    """The positive reals (0, inf)^d.

    Parameters
    ----------
    _name : str
        Human-readable name.
    _dim : int
        Dimensionality.

    Examples
    --------
    >>> R_plus = PositiveReals("variance", 1)
    >>> R_plus.contains(torch.tensor([[0.5]]))
    tensor([True])
    >>> R_plus.contains(torch.tensor([[-1.0]]))
    tensor([False])
    """

    _name: str
    _dim: int

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0.0).all(dim=-1)

    def __repr__(self) -> str:
        return f"PositiveReals({self._name!r}, {self._dim})"


@dataclass(frozen=True)
class ProductSpace(ContinuousSpace):
    """Cartesian product of continuous spaces.

    Flattens nested products so that ProductSpace(A, ProductSpace(B, C))
    becomes ProductSpace(A, B, C).

    The total event dimension is the sum of component dimensions.
    Containment checks dispatch to each component on the appropriate
    slice of the input vector.

    Examples
    --------
    >>> A = Euclidean("x", 2)
    >>> B = PositiveReals("sigma", 1)
    >>> P = A * B
    >>> P.dim
    3
    """

    components: tuple[ContinuousSpace, ...]

    def __init__(self, *spaces: ContinuousSpace) -> None:
        flat: list[ContinuousSpace] = []

        for s in spaces:
            if isinstance(s, ProductSpace):
                flat.extend(s.components)

            else:
                flat.append(s)

        object.__setattr__(self, "components", tuple(flat))

    @property
    def name(self) -> str:
        return " × ".join(s.name for s in self.components)

    @property
    def dim(self) -> int:
        return sum(s.dim for s in self.components)

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.ones(
            x.shape[:-1], dtype=torch.bool, device=x.device
        )
        offset = 0

        for s in self.components:
            chunk = x[..., offset:offset + s.dim]
            result = result & s.contains(chunk)
            offset += s.dim

        return result

    def __repr__(self) -> str:
        inner = " × ".join(repr(s) for s in self.components)
        return f"({inner})"

    def __hash__(self) -> int:
        return hash(("ProductSpace", self.components))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProductSpace):
            return NotImplemented

        return self.components == other.components
