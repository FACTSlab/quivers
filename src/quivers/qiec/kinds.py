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
    """The sort of shapes, optionally restricted to a fixed rank."""

    rank: int | None = None
    tag: Literal["shape"] = "shape"

    def __post_init__(self) -> None:
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
    """

    name: str
    constructors: tuple[str, ...]
    arities: tuple[int, ...] = ()
    tag: Literal["user"] = "user"

    def __post_init__(self) -> None:
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
    """Reject duplicate names in one dependent telescope."""
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
