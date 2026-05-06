"""Categorical objects: finite sets, products, coproducts, and free monoids.

The set-object family is a sum type with four variants — :class:`FinSet`,
:class:`ProductSet`, :class:`CoproductSet`, :class:`FreeMonoid` — discriminated
by ``kind``. ``SetObject`` is the :class:`dx.TaggedUnion` root that gathers
them; ``X * Y`` and ``X + Y`` work on any pair of variants.
"""

from math import prod
from typing import Literal
import didactic.api as dx


class SetObject(dx.TaggedUnion, discriminator="kind"):
    """The category of finite sets, product sets, and coproduct sets.

    Variants share two derived properties (:attr:`size`, :attr:`shape`) and
    the ``*`` / ``+`` operators for product and coproduct construction.
    """

    @property
    def size(self) -> int:
        """Total cardinality of the set."""
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor dimension shape."""
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        """Number of tensor dimensions."""
        return len(self.shape)

    def __mul__(self, other: "SetObject") -> "ProductSet":
        if not isinstance(other, SetObject):
            return NotImplemented
        return ProductSet(components=(self, other))

    def __add__(self, other: "SetObject") -> "CoproductSet":
        if not isinstance(other, SetObject):
            return NotImplemented
        return CoproductSet(components=(self, other))


def _check_cardinality(value: int) -> int:
    if value < 1:
        raise ValueError(f"cardinality must be >= 1, got {value}")
    return value


class FinSet(SetObject):
    """A named finite set with a fixed cardinality.

    Attributes
    ----------
    name : str
        Human-readable name for the set.
    cardinality : int
        Number of elements (must be ``>= 1``).
    """

    name: str
    cardinality: int = dx.field(converter=_check_cardinality)
    kind: Literal["finset"] = "finset"

    @property
    def size(self) -> int:
        return self.cardinality

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.cardinality,)

    def __str__(self) -> str:
        return f"FinSet(name={self.name!r}, cardinality={self.cardinality})"


Unit: FinSet = FinSet(name="1", cardinality=1)


def _flatten_products(items: tuple["SetObject", ...]) -> tuple["SetObject", ...]:
    """Flatten nested ProductSet so that P(A, P(B, C)) collapses to P(A, B, C)."""
    out: list[SetObject] = []
    for c in items:
        if isinstance(c, ProductSet):
            out.extend(c.components)
        else:
            out.append(c)
    return tuple(out)


class ProductSet(SetObject):
    """Cartesian product of finite sets.

    Nested products are flattened: ``ProductSet(components=(A, ProductSet(components=(B, C))))``
    constructs to ``ProductSet`` with ``components == (A, B, C)``.
    """

    components: tuple[SetObject, ...] = dx.field(
        default=(), converter=_flatten_products
    )
    kind: Literal["product_set"] = "product_set"

    @property
    def size(self) -> int:
        return prod((c.size for c in self.components))

    @property
    def shape(self) -> tuple[int, ...]:
        result: list[int] = []
        for c in self.components:
            result.extend(c.shape)
        return tuple(result)

    def __str__(self) -> str:
        inner = " × ".join((str(c) for c in self.components))
        return f"({inner})"


def _flatten_coproducts(items: tuple["SetObject", ...]) -> tuple["SetObject", ...]:
    """Flatten nested CoproductSet so that C(A, C(B, C)) collapses to C(A, B, C)."""
    out: list[SetObject] = []
    for c in items:
        if isinstance(c, CoproductSet):
            out.extend(c.components)
        else:
            out.append(c)
    return tuple(out)


class CoproductSet(SetObject):
    """Tagged union (coproduct) of finite sets.

    The flat cardinality is the sum of component cardinalities; the shape is
    a single dimension of that total size with offsets recoverable from
    :attr:`offsets`.
    """

    components: tuple[SetObject, ...] = dx.field(
        default=(), converter=_flatten_coproducts
    )
    kind: Literal["coproduct_set"] = "coproduct_set"

    @dx.derived
    def offsets(self) -> tuple[int, ...]:
        """Per-component flat offsets, computed from cumulative sizes."""
        out: list[int] = []
        running = 0
        for c in self.components:
            out.append(running)
            running += c.size
        return tuple(out)

    @property
    def size(self) -> int:
        return sum((c.size for c in self.components))

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.size,)

    def offset(self, index: int) -> int:
        """Return the flat offset for the i-th component."""
        return self.offsets[index]

    def component_range(self, index: int) -> tuple[int, int]:
        """Return ``(start, end)`` flat indices for the i-th component."""
        start = self.offsets[index]
        end = start + self.components[index].size
        return (start, end)

    def __str__(self) -> str:
        inner = " + ".join((str(c) for c in self.components))
        return f"({inner})"


def _build_free_monoid_components(
    generators: FinSet, max_length: int
) -> tuple[SetObject, ...]:
    if max_length < 0:
        raise ValueError(f"max_length must be >= 0, got {max_length}")
    components: list[SetObject] = [Unit]
    for k in range(1, max_length + 1):
        if k == 1:
            components.append(generators)
        else:
            components.append(ProductSet(components=tuple([generators] * k)))
    return tuple(components)


class FreeMonoid(SetObject):
    """Free monoid on a generator set, truncated to ``max_length``.

    Represents all tuples (strings) from elements of ``generators`` with
    length 0 through ``max_length``. Internally a coproduct
    ``Unit + G + G×G + ... + G^max_length``; obtain it via
    :meth:`as_coproduct`.

    Attributes
    ----------
    generators : FinSet
        The generator set.
    max_length : int
        Maximum string length (inclusive).
    """

    generators: FinSet
    max_length: int
    kind: Literal["free_monoid"] = "free_monoid"

    @dx.derived
    def _coproduct(self) -> CoproductSet:
        return CoproductSet(
            components=_build_free_monoid_components(self.generators, self.max_length)
        )

    def as_coproduct(self) -> CoproductSet:
        """Return the underlying coproduct view."""
        return self._coproduct

    @property
    def components(self) -> tuple[SetObject, ...]:
        return self._coproduct.components

    @property
    def size(self) -> int:
        return self._coproduct.size

    @property
    def shape(self) -> tuple[int, ...]:
        return self._coproduct.shape

    def offset(self, index: int) -> int:
        return self._coproduct.offset(index)

    def component_range(self, index: int) -> tuple[int, int]:
        return self._coproduct.component_range(index)

    def encode(self, word: tuple[int, ...]) -> int:
        """Encode a word (tuple of generator indices) to a flat index."""
        k = len(word)
        if k > self.max_length:
            raise ValueError(f"word length {k} exceeds max_length {self.max_length}")
        g = self.generators.cardinality
        base = self.offset(k)
        if k == 0:
            return base
        idx = 0
        for w in word:
            if not 0 <= w < g:
                raise ValueError(f"generator index {w} out of range [0, {g})")
            idx = idx * g + w
        return base + idx

    def decode(self, flat_index: int) -> tuple[int, ...]:
        """Decode a flat index back to a word."""
        if not 0 <= flat_index < self.size:
            raise ValueError(f"flat_index {flat_index} out of range [0, {self.size})")
        g = self.generators.cardinality
        cop = self._coproduct
        for k in range(len(cop.components)):
            start, end = cop.component_range(k)
            if start <= flat_index < end:
                local = flat_index - start
                if k == 0:
                    return ()
                digits: list[int] = []
                for _ in range(k):
                    digits.append(local % g)
                    local //= g
                digits.reverse()
                return tuple(digits)
        raise RuntimeError("unreachable")

    def __str__(self) -> str:
        return (
            f"FreeMonoid(generators={self.generators!s}, max_length={self.max_length})"
        )
