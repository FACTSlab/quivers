"""Static terms for the Quivers Indexed Effect Core (QIEC).

The module intentionally models only the kernel's small static language.
Didactic and Panproto check a first-order projection of its indexed-family
declarations, while these frozen records retain the QIEC-specific distinction
between uniform parameters and refinable indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quivers.qiec.identifiers import EffectId, StaticVariableId, TypeId
from quivers.qiec.kinds import IndexSort, Kind, Telescope, TYPE, TypeBinder


@dataclass(frozen=True, slots=True, eq=False)
class IndexVariable:
    name: str
    sort: IndexSort
    identity: StaticVariableId | None = None
    tag: Literal["index_variable"] = "index_variable"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexVariable) or self.sort != other.sort:
            return False
        if self.identity is not None or other.identity is not None:
            return self.identity is not None and self.identity == other.identity
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(
            (self.sort, self.identity if self.identity is not None else self.name)
        )


@dataclass(frozen=True, slots=True)
class IndexLiteral:
    value: int | str
    sort: IndexSort
    tag: Literal["index_literal"] = "index_literal"

    def __post_init__(self) -> None:
        from quivers.qiec.kinds import NatSort, UserIndexSort

        if isinstance(self.sort, NatSort):
            if not isinstance(self.value, int) or self.value < 0:
                raise ValueError("Nat literals must be nonnegative integers")
        if (
            isinstance(self.sort, UserIndexSort)
            and self.value not in self.sort.constructors
        ):
            raise ValueError(
                f"{self.value!r} is not a constructor of index sort {self.sort.name!r}"
            )
        if (
            isinstance(self.sort, UserIndexSort)
            and isinstance(self.value, str)
            and self.sort.constructor_arity(self.value) != 0
        ):
            raise ValueError(f"index constructor {self.value!r} is not nullary")


@dataclass(frozen=True, slots=True)
class IndexConstructor:
    name: str
    arguments: tuple[IndexTerm, ...]
    sort: IndexSort
    tag: Literal["index_constructor"] = "index_constructor"

    def __post_init__(self) -> None:
        from quivers.qiec.kinds import UserIndexSort

        if not isinstance(self.sort, UserIndexSort):
            raise ValueError("named index constructors require a user index sort")
        expected = self.sort.constructor_arity(self.name)
        if len(self.arguments) != expected:
            raise ValueError(
                f"index constructor {self.name!r} expects {expected} arguments, "
                f"got {len(self.arguments)}"
            )
        for argument in self.arguments:
            if index_sort(argument) != self.sort:
                raise ValueError(
                    f"argument of index constructor {self.name!r} has the wrong sort"
                )


@dataclass(frozen=True, slots=True)
class ShapeIndex:
    dimensions: tuple[IndexTerm, ...]
    tag: Literal["shape_index"] = "shape_index"


type IndexTerm = IndexVariable | IndexLiteral | IndexConstructor | ShapeIndex


@dataclass(frozen=True, slots=True, eq=False)
class TypeVariable:
    name: str
    kind: Kind = TYPE
    identity: StaticVariableId | None = None
    tag: Literal["type_variable"] = "type_variable"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeVariable) or self.kind != other.kind:
            return False
        if self.identity is not None or other.identity is not None:
            return self.identity is not None and self.identity == other.identity
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(
            (self.kind, self.identity if self.identity is not None else self.name)
        )


@dataclass(frozen=True, slots=True, eq=False)
class TypeConstructorRef:
    """A fully qualified type constructor and its kinding telescope."""

    id: TypeId
    name: str
    telescope: Telescope = ()

    @classmethod
    def builtin(cls, name: str) -> TypeConstructorRef:
        return cls(TypeId.derive("builtin", name), name)

    def __eq__(self, other: object) -> bool:
        """Compare semantic identity, excluding diagnostic presentation data."""
        return isinstance(other, TypeConstructorRef) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True, slots=True)
class TypeApplication:
    constructor: TypeConstructorRef
    arguments: tuple[StaticArgument, ...] = ()
    tag: Literal["type_application"] = "type_application"


@dataclass(frozen=True, slots=True)
class FunctionType:
    """A pure function type.

    Effectful codomains are represented explicitly by
    :class:`quivers.qiec.effects.ComputationType`, keeping the value and
    computation strata separate.
    """

    parameter: TypeExpr
    result: TypeExpr
    tag: Literal["function_type"] = "function_type"


@dataclass(frozen=True, slots=True)
class EqualityType:
    kind: Kind | IndexSort
    left: StaticArgument
    right: StaticArgument
    tag: Literal["equality_type"] = "equality_type"


type TypeExpr = TypeVariable | TypeApplication | FunctionType | EqualityType


@dataclass(frozen=True, slots=True, eq=False)
class EffectVariable:
    name: str
    identity: StaticVariableId | None = None
    tag: Literal["effect_variable"] = "effect_variable"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EffectVariable):
            return False
        if self.identity is not None or other.identity is not None:
            return self.identity is not None and self.identity == other.identity
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.identity if self.identity is not None else self.name)


@dataclass(frozen=True, slots=True, eq=False)
class EffectRef:
    """One closed effect interface application."""

    id: EffectId
    name: str
    arguments: tuple[StaticArgument, ...] = ()
    tag: Literal["effect_ref"] = "effect_ref"

    def __eq__(self, other: object) -> bool:
        """Compare a concrete interface application by stable identity."""
        return (
            isinstance(other, EffectRef)
            and self.id == other.id
            and self.arguments == other.arguments
        )

    def __hash__(self) -> int:
        return hash((self.id, self.arguments))


type StaticArgument = TypeExpr | IndexTerm | EffectRef | EffectVariable


UNIT = TypeApplication(TypeConstructorRef.builtin("Unit"))
BOOL = TypeApplication(TypeConstructorRef.builtin("Bool"))
INT = TypeApplication(TypeConstructorRef.builtin("Int"))
REAL = TypeApplication(TypeConstructorRef.builtin("Real"))
STRING = TypeApplication(TypeConstructorRef.builtin("String"))


def product_type(*components: TypeExpr) -> TypeApplication:
    """Construct the canonical finite-product type of the given arity."""

    arity = len(components)
    constructor = TypeConstructorRef(
        TypeId.derive("builtin", "Product", arity),
        f"Product{arity}",
        tuple(TypeBinder(f"item{position}") for position in range(arity)),
    )
    return TypeApplication(constructor, components)


def static_kind(term: StaticArgument) -> Kind:
    """Return the outer kind of a static term.

    Index sorts are deliberately not collapsed into ``Type``: callers that
    validate an index binder compare the term's concrete ``sort`` separately.
    """
    from quivers.qiec.kinds import EFFECT

    if isinstance(term, (EffectRef, EffectVariable)):
        return EFFECT
    if isinstance(term, TypeVariable):
        return term.kind
    if isinstance(term, (TypeApplication, FunctionType, EqualityType)):
        return TYPE
    raise TypeError("index terms have a sort, not a QIEC kind")


def index_sort(term: IndexTerm) -> IndexSort:
    """Return an index term's sort, checking shape dimensions."""
    from quivers.qiec.kinds import NAT, ShapeSort

    if isinstance(term, ShapeIndex):
        for dimension in term.dimensions:
            if index_sort(dimension) != NAT:
                raise TypeError("shape dimensions must have Nat sort")
        return ShapeSort(rank=len(term.dimensions))
    return term.sort


__all__ = [
    "BOOL",
    "EffectRef",
    "EffectVariable",
    "EqualityType",
    "FunctionType",
    "INT",
    "IndexConstructor",
    "IndexLiteral",
    "IndexTerm",
    "IndexVariable",
    "REAL",
    "STRING",
    "ShapeIndex",
    "StaticArgument",
    "TypeApplication",
    "TypeConstructorRef",
    "TypeExpr",
    "TypeVariable",
    "UNIT",
    "index_sort",
    "product_type",
    "static_kind",
]
