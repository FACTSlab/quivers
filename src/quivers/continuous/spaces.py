"""Continuous measurable spaces for the hybrid architecture.

Continuous space objects serve as domains and codomains for continuous
morphisms. They complement :mod:`quivers.core.objects` (FinSet etc.) used
by discrete morphisms.

The space family is a sum type:

- :class:`Euclidean` — :math:`\\mathbb{R}^d` with optional bounds
- :class:`Simplex` — probability simplex over ``d`` components
- :class:`PositiveReals` — :math:`(0, \\infty)^d`
- :class:`ProductSpace` — cartesian product of continuous spaces

:func:`UnitInterval` is a convenience factory for ``[0, 1]^d``.
"""

from typing import Literal

import didactic.api as dx
import torch

from quivers.core.objects import SetObject


# ---------------------------------------------------------------------------
# the ContinuousSpace sum
# ---------------------------------------------------------------------------


class ContinuousSpace(dx.TaggedUnion, discriminator="kind"):
    """Continuous measurable space (Euclidean, Simplex, PositiveReals, Product).

    Variants expose ``name: str`` and ``dim: int`` either as fields (the
    atomic variants) or as derived properties (:class:`ProductSpace`),
    plus a :meth:`contains` predicate over the support.
    """

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of a single sample."""
        return (self.dim,)  # type: ignore[attr-defined]

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Check whether points lie in the support."""
        raise NotImplementedError(f"{type(self).__name__}.contains")

    def sample_uniform(self, n: int) -> torch.Tensor:
        """Sample n points uniformly from the space."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support uniform sampling"
        )

    def __mul__(self, other: "ContinuousSpace") -> "ProductSpace":
        if not isinstance(other, ContinuousSpace):
            return NotImplemented
        return ProductSpace(components=(self, other))


# ---------------------------------------------------------------------------
# Euclidean
# ---------------------------------------------------------------------------


class Euclidean(ContinuousSpace):
    """Euclidean space :math:`\\mathbb{R}^d`, optionally bounded."""

    name: str
    dim: int
    low: float | None = None
    high: float | None = None
    kind: Literal["euclidean"] = "euclidean"

    @property
    def is_bounded(self) -> bool:
        """Whether the space has finite bounds on all sides."""
        return self.low is not None and self.high is not None

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        if self.low is not None:
            result = result & (x >= self.low).all(dim=-1)
        if self.high is not None:
            result = result & (x <= self.high).all(dim=-1)
        return result

    def sample_uniform(self, n: int) -> torch.Tensor:
        if not self.is_bounded:
            raise ValueError("cannot sample uniformly from unbounded Euclidean space")
        assert self.low is not None and self.high is not None
        return torch.rand(n, self.dim) * (self.high - self.low) + self.low

    def __str__(self) -> str:
        bounds = ""
        if self.low is not None or self.high is not None:
            bounds = f", low={self.low}, high={self.high}"
        return f"Euclidean({self.name!r}, {self.dim}{bounds})"


def UnitInterval(name: str, dim: int = 1) -> Euclidean:
    """Create a :math:`[0, 1]^d` bounded Euclidean space."""
    return Euclidean(name=name, dim=dim, low=0.0, high=1.0)


# ---------------------------------------------------------------------------
# Simplex
# ---------------------------------------------------------------------------


class Simplex(ContinuousSpace):
    """The probability simplex :math:`\\{x \\in \\mathbb{R}^d : x_i \\geq 0, \\sum x_i = 1\\}`."""

    name: str
    dim: int
    kind: Literal["simplex"] = "simplex"

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        nonneg = (x >= -1e-7).all(dim=-1)
        sums_to_one = (x.sum(dim=-1) - 1.0).abs() < 1e-5
        return nonneg & sums_to_one

    def sample_uniform(self, n: int) -> torch.Tensor:
        e = torch.distributions.Exponential(1.0).sample((n, self.dim))
        return e / e.sum(dim=-1, keepdim=True)

    def __str__(self) -> str:
        return f"Simplex({self.name!r}, {self.dim})"


# ---------------------------------------------------------------------------
# PositiveReals
# ---------------------------------------------------------------------------


class PositiveReals(ContinuousSpace):
    """The positive reals :math:`(0, \\infty)^d`."""

    name: str
    dim: int
    kind: Literal["positive_reals"] = "positive_reals"

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0.0).all(dim=-1)

    def __str__(self) -> str:
        return f"PositiveReals({self.name!r}, {self.dim})"


# ---------------------------------------------------------------------------
# CholeskyFactor
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
    sampling family
    (:class:`~quivers.continuous.families.LKJCorrelationFactor`)
    and not by the type itself.
    """

    name: str
    dim: int
    kind: Literal["cholesky_factor"] = "cholesky_factor"

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dim * self.dim,)

    @property
    def ndim(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# ProductSpace
# ---------------------------------------------------------------------------


def _flatten_spaces(
    items: tuple["ContinuousSpace | SetObject", ...],
) -> tuple["ContinuousSpace | SetObject", ...]:
    """Flatten nested ProductSpace so that P(A, P(B, C)) collapses to P(A, B, C)."""
    out: list[ContinuousSpace | SetObject] = []
    for s in items:
        if isinstance(s, ProductSpace):
            out.extend(s.components)
        else:
            out.append(s)
    return tuple(out)


def _component_name(c: "ContinuousSpace | SetObject") -> str:
    return c.name if isinstance(c, ContinuousSpace) else str(c)


def _component_dim(c: "ContinuousSpace | SetObject") -> int:
    """Return the event-shape width of a product component.

    ContinuousSpace components contribute :attr:`dim`; SetObject
    components contribute the length of their tensor :attr:`shape`
    (1 for FinSet, len(components) for ProductSet, 1 for CoproductSet).
    """
    return c.dim if isinstance(c, ContinuousSpace) else len(c.shape)


def _product_name(components: tuple["ContinuousSpace | SetObject", ...]) -> str:
    return " × ".join(_component_name(c) for c in components)


def _product_dim(components: tuple["ContinuousSpace | SetObject", ...]) -> int:
    return sum(_component_dim(c) for c in components)


class ProductSpace(ContinuousSpace):
    """Cartesian product of continuous spaces (and discrete objects).

    Components may be a mix of :class:`ContinuousSpace` variants and
    :class:`~quivers.core.objects.SetObject` variants — programs whose
    domain or codomain combines discrete and continuous variables produce
    such a ProductSpace at compile time. Nested products are flattened
    on construction; :attr:`name` and :attr:`dim` are derived from
    :attr:`components` (for SetObject components, :attr:`dim` falls back
    to ``len(component.shape)``).
    """

    components: tuple[ContinuousSpace | SetObject, ...] = dx.field(
        default=(), converter=_flatten_spaces
    )
    kind: Literal["product_space"] = "product_space"

    @property
    def name(self) -> str:
        return _product_name(self.components)

    @property
    def dim(self) -> int:
        return _product_dim(self.components)

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        offset = 0
        for s in self.components:
            width = _component_dim(s)
            if isinstance(s, ContinuousSpace):
                chunk = x[..., offset : offset + width]
                result = result & s.contains(chunk)
            else:
                # SetObject component (mixed-domain product): the slice is a
                # discrete index. Check it falls inside the index range.
                chunk = x[..., offset : offset + width]
                in_range = (chunk >= 0).all(dim=-1) & (chunk < s.size).all(dim=-1)
                result = result & in_range
            offset += width
        return result

    def __str__(self) -> str:
        inner = " × ".join(str(s) for s in self.components)
        return f"({inner})"
