"""Kinds, closed index sorts, and declaration telescopes for QIEC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TypeKind:
    tag: Literal["type"] = "type"


@dataclass(frozen=True, slots=True)
class EffectKind:
    tag: Literal["effect"] = "effect"


@dataclass(frozen=True, slots=True)
class RowKind:
    tag: Literal["row"] = "row"


@dataclass(frozen=True, slots=True)
class ArrowKind:
    domain: Kind
    codomain: Kind
    tag: Literal["arrow"] = "arrow"


type Kind = TypeKind | EffectKind | RowKind | ArrowKind

TYPE = TypeKind()
EFFECT = EffectKind()
ROW = RowKind()


@dataclass(frozen=True, slots=True)
class NatSort:
    tag: Literal["nat"] = "nat"


@dataclass(frozen=True, slots=True)
class ShapeSort:
    """The sort of shapes, optionally restricted to a fixed rank.

    Parameters
    ----------
    rank : int or None
        The number of dimensions, or None for any rank. A binder of
        unspecified rank accepts a shape of any rank, which is what makes
        a rank-polymorphic signature expressible.
    tag : Literal["shape"]
        Discriminator for the index-sort union.
    """

    rank: int | None = None
    tag: Literal["shape"] = "shape"

    def __post_init__(self) -> None:
        """Reject a negative rank.

        Raises
        ------
        ValueError
            If `rank` is negative.
        """
        if self.rank is not None and self.rank < 0:
            raise ValueError("shape rank cannot be negative")


@dataclass(frozen=True, slots=True)
class ContextSort:
    """A logic-program context indexed by a stable signature name."""

    signature: str
    tag: Literal["context"] = "context"


@dataclass(frozen=True, slots=True)
class UserIndexSort:
    """A closed user-defined index datatype.

    Constructor names are part of the sort identity.  Open index sorts are
    deliberately absent from the first kernel because they make coverage
    unstable under separate compilation.

    Parameters
    ----------
    name : str
        The sort's name, part of its identity.
    constructors : tuple[str, ...]
        Constructor names, unique and part of the identity. Two sorts
        with the same name and different constructors are different
        sorts, which is what keeps coverage decidable.
    arities : tuple[int, ...]
        Argument count per constructor, positionally aligned with
        `constructors`. Empty defaults to all-nullary.
    tag : Literal["user"]
        Discriminator for the index-sort union.
    """

    name: str
    constructors: tuple[str, ...]
    arities: tuple[int, ...] = ()
    tag: Literal["user"] = "user"

    def __post_init__(self) -> None:
        """Default the arities and reject a malformed sort.

        Raises
        ------
        ValueError
            If the name is empty, there are no constructors, a
            constructor name repeats, the arities do not align with the
            constructors, or an arity is negative.
        """
        if not self.name:
            raise ValueError("user index sort name cannot be empty")
        if not self.constructors:
            raise ValueError("a closed user index sort needs a constructor")
        if len(set(self.constructors)) != len(self.constructors):
            raise ValueError(f"duplicate constructors in index sort {self.name!r}")
        if not self.arities:
            object.__setattr__(self, "arities", (0,) * len(self.constructors))
        if len(self.arities) != len(self.constructors):
            raise ValueError("index constructor arities must align with constructors")
        if any(arity < 0 for arity in self.arities):
            raise ValueError("index constructor arities cannot be negative")

    def constructor_arity(self, name: str) -> int:
        """The number of arguments one constructor takes.

        Parameters
        ----------
        name : str
            The constructor name.

        Returns
        -------
        int
            Its arity, zero for a nullary constructor.

        Raises
        ------
        ValueError
            If the sort has no such constructor. The sort is closed, so
            an unknown name is an error rather than an open extension.
        """
        try:
            position = self.constructors.index(name)
        except ValueError as error:
            raise ValueError(
                f"{name!r} is not a constructor of index sort {self.name!r}"
            ) from error
        return self.arities[position]


type IndexSort = NatSort | ShapeSort | ContextSort | UserIndexSort

NAT = NatSort()
SHAPE = ShapeSort()


@dataclass(frozen=True, slots=True)
class TypeBinder:
    name: str
    kind: Kind = TYPE
    refinable: bool = False
    tag: Literal["type"] = "type"


@dataclass(frozen=True, slots=True)
class IndexBinder:
    name: str
    sort: IndexSort
    refinable: bool = False
    tag: Literal["index"] = "index"


@dataclass(frozen=True, slots=True)
class EffectBinder:
    name: str
    refinable: bool = False
    tag: Literal["effect"] = "effect"


type TelescopeBinder = TypeBinder | IndexBinder | EffectBinder
type Telescope = tuple[TelescopeBinder, ...]


def validate_telescope(telescope: Telescope) -> None:
    """Reject duplicate names in one dependent telescope.

    Parameters
    ----------
    telescope : Telescope
        The binders to check, in order.

    Raises
    ------
    ValueError
        If a binder name is empty or repeats. Later binders may depend on
        earlier ones, so a repeated name would make a reference ambiguous.
    """
    seen: set[str] = set()
    for binder in telescope:
        if not binder.name:
            raise ValueError("telescope binder name cannot be empty")
        if binder.name in seen:
            raise ValueError(f"duplicate telescope binder {binder.name!r}")
        seen.add(binder.name)


__all__ = [
    "ArrowKind",
    "ContextSort",
    "EFFECT",
    "EffectBinder",
    "EffectKind",
    "IndexBinder",
    "IndexSort",
    "Kind",
    "NAT",
    "NatSort",
    "ROW",
    "RowKind",
    "SHAPE",
    "ShapeSort",
    "TYPE",
    "Telescope",
    "TelescopeBinder",
    "TypeBinder",
    "TypeKind",
    "UserIndexSort",
    "validate_telescope",
]
