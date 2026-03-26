"""Categorical objects: finite sets, products, coproducts, and free monoids."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import prod


class SetObject(ABC):
    """Abstract base for all categorical objects (finite sets)."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Total cardinality of the set."""
        ...

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Tensor dimension shape.

        For a FinSet of size n, this is (n,).
        For a ProductSet of FinSets with sizes n1, n2, ..., this is (n1, n2, ...).
        """
        ...

    @property
    def ndim(self) -> int:
        """Number of tensor dimensions."""
        return len(self.shape)

    def __mul__(self, other: SetObject) -> ProductSet:
        """Cartesian product via * operator."""
        if not isinstance(other, SetObject):
            return NotImplemented

        return ProductSet(self, other)

    def __add__(self, other: SetObject) -> CoproductSet:
        """Coproduct (tagged union) via + operator."""
        if not isinstance(other, SetObject):
            return NotImplemented

        return CoproductSet(self, other)


@dataclass(frozen=True)
class FinSet(SetObject):
    """A named finite set with a fixed cardinality.

    Parameters
    ----------
    name : str
        Human-readable name for the set.
    cardinality : int
        Number of elements.

    Examples
    --------
    >>> X = FinSet("phoneme", 40)
    >>> X.size
    40
    >>> X.shape
    (40,)
    """

    name: str
    cardinality: int

    def __post_init__(self) -> None:
        if self.cardinality < 1:
            raise ValueError(f"cardinality must be >= 1, got {self.cardinality}")

    @property
    def size(self) -> int:
        return self.cardinality

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.cardinality,)

    def __repr__(self) -> str:
        return f"FinSet({self.name!r}, {self.cardinality})"


# module-level singleton for the terminal object
Unit: FinSet = FinSet("1", 1)


@dataclass(frozen=True)
class ProductSet(SetObject):
    """Cartesian product of finite sets.

    Flattens nested products so that ProductSet(A, ProductSet(B, C))
    becomes ProductSet(A, B, C).

    Parameters
    ----------
    *components : SetObject
        Component sets.

    Examples
    --------
    >>> X = FinSet("X", 3)
    >>> Y = FinSet("Y", 4)
    >>> P = ProductSet(X, Y)
    >>> P.shape
    (3, 4)
    >>> P.size
    12
    """

    components: tuple[SetObject, ...] = field(default_factory=tuple)

    def __init__(self, *components: SetObject) -> None:
        # flatten nested products
        flat: list[SetObject] = []

        for c in components:
            if isinstance(c, ProductSet):
                flat.extend(c.components)

            else:
                flat.append(c)

        object.__setattr__(self, "components", tuple(flat))

    @property
    def size(self) -> int:
        return prod(c.size for c in self.components)

    @property
    def shape(self) -> tuple[int, ...]:
        result: list[int] = []

        for c in self.components:
            result.extend(c.shape)

        return tuple(result)

    def __repr__(self) -> str:
        inner = " × ".join(repr(c) for c in self.components)
        return f"({inner})"

    def __hash__(self) -> int:
        return hash(("ProductSet", self.components))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProductSet):
            return NotImplemented

        return self.components == other.components


@dataclass(frozen=True)
class CoproductSet(SetObject):
    """Tagged union (coproduct) of finite sets.

    The flat cardinality is the sum of component cardinalities.
    The shape is (total_size,) — a single flat dimension with an
    internal offset table for locating each component.

    Parameters
    ----------
    *components : SetObject
        Component sets.

    Examples
    --------
    >>> X = FinSet("X", 3)
    >>> Y = FinSet("Y", 4)
    >>> C = CoproductSet(X, Y)
    >>> C.size
    7
    """

    components: tuple[SetObject, ...] = field(default_factory=tuple)
    _offsets: tuple[int, ...] = field(default_factory=tuple, repr=False)

    def __init__(self, *components: SetObject) -> None:
        # flatten nested coproducts
        flat: list[SetObject] = []

        for c in components:
            if isinstance(c, CoproductSet):
                flat.extend(c.components)

            else:
                flat.append(c)

        offsets: list[int] = []
        running = 0

        for c in flat:
            offsets.append(running)
            running += c.size

        object.__setattr__(self, "components", tuple(flat))
        object.__setattr__(self, "_offsets", tuple(offsets))

    @property
    def size(self) -> int:
        return sum(c.size for c in self.components)

    @property
    def shape(self) -> tuple[int, ...]:
        # coproducts are represented as a single flat dimension
        return (self.size,)

    def offset(self, index: int) -> int:
        """Return the flat offset for the i-th component.

        Parameters
        ----------
        index : int
            Component index.

        Returns
        -------
        int
            Flat offset.
        """
        return self._offsets[index]

    def component_range(self, index: int) -> tuple[int, int]:
        """Return (start, end) flat indices for the i-th component.

        Parameters
        ----------
        index : int
            Component index.

        Returns
        -------
        tuple[int, int]
            Start and end indices (exclusive end).
        """
        start = self._offsets[index]
        end = start + self.components[index].size
        return start, end

    def __repr__(self) -> str:
        inner = " + ".join(repr(c) for c in self.components)
        return f"({inner})"

    def __hash__(self) -> int:
        return hash(("CoproductSet", self.components))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoproductSet):
            return NotImplemented

        return self.components == other.components


class FreeMonoid(CoproductSet):
    """Free monoid on a generator set, truncated to a maximum string length.

    Represents the set of all tuples (strings) from elements of
    ``generators`` with length 0 through ``max_length``.  Internally
    this is a CoproductSet(Unit, G, G×G, ..., G^max_length).

    Parameters
    ----------
    generators : FinSet
        The generator set.
    max_length : int
        Maximum string length (inclusive).

    Examples
    --------
    >>> G = FinSet("abc", 3)
    >>> FM = FreeMonoid(G, max_length=2)
    >>> FM.size  # 1 + 3 + 9 = 13
    13
    """

    generators: FinSet  # type: ignore[assignment]
    max_length: int  # type: ignore[assignment]

    def __init__(self, generators: FinSet, max_length: int) -> None:
        if max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}")

        # build components: Unit, G, G^2, ..., G^max_length
        components: list[SetObject] = [Unit]

        for k in range(1, max_length + 1):
            components.append(ProductSet(*([generators] * k)) if k > 1 else generators)

        super().__init__(*components)
        object.__setattr__(self, "generators", generators)
        object.__setattr__(self, "max_length", max_length)

    def encode(self, word: tuple[int, ...]) -> int:
        """Encode a word (tuple of generator indices) to a flat index.

        Parameters
        ----------
        word : tuple[int, ...]
            Tuple of generator indices. Empty tuple for the empty string.

        Returns
        -------
        int
            Flat index into the coproduct.
        """
        k = len(word)

        if k > self.max_length:
            raise ValueError(f"word length {k} exceeds max_length {self.max_length}")

        g = self.generators.cardinality
        base = self.offset(k)

        if k == 0:
            return base

        # mixed-radix encoding: word = (w_0, w_1, ..., w_{k-1})
        # index = w_0 * g^(k-1) + w_1 * g^(k-2) + ... + w_{k-1}
        idx = 0

        for w in word:
            if not (0 <= w < g):
                raise ValueError(f"generator index {w} out of range [0, {g})")

            idx = idx * g + w

        return base + idx

    def decode(self, flat_index: int) -> tuple[int, ...]:
        """Decode a flat index back to a word.

        Parameters
        ----------
        flat_index : int
            Flat index into the coproduct.

        Returns
        -------
        tuple[int, ...]
            Tuple of generator indices.
        """
        if not (0 <= flat_index < self.size):
            raise ValueError(f"flat_index {flat_index} out of range [0, {self.size})")

        g = self.generators.cardinality

        # find which component (word length)
        for k in range(len(self.components)):
            start, end = self.component_range(k)

            if start <= flat_index < end:
                local = flat_index - start

                if k == 0:
                    return ()

                # mixed-radix decoding
                digits: list[int] = []

                for _ in range(k):
                    digits.append(local % g)
                    local //= g

                digits.reverse()
                return tuple(digits)

        raise RuntimeError("unreachable")

    def __repr__(self) -> str:
        return f"FreeMonoid({self.generators!r}, max_length={self.max_length})"
