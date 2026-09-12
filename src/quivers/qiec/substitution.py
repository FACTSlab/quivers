"""Capture-free substitution and telescope instantiation for QIEC."""

from __future__ import annotations

from dataclasses import dataclass

from quivers.qiec.effects import EffectRow, OperationDef, RowEntry
from quivers.qiec.kinds import (
    ContextSort,
    EffectBinder,
    IndexBinder,
    NatSort,
    ShapeSort,
    Telescope,
    TYPE,
    TypeBinder,
    UserIndexSort,
)
from quivers.qiec.types import (
    EffectRef,
    EffectVariable,
    EqualityType,
    FunctionType,
    IndexConstructor,
    IndexLiteral,
    IndexTerm,
    IndexVariable,
    ShapeIndex,
    StaticArgument,
    TypeApplication,
    TypeExpr,
    TypeVariable,
    index_sort,
    static_kind,
)


def _sort_matches(expected: object, actual: object) -> bool:
    if isinstance(expected, ShapeSort) and isinstance(actual, ShapeSort):
        return expected.rank is None or expected.rank == actual.rank
    return expected == actual


def validate_static_argument(term: StaticArgument) -> None:
    """Recursively validate one intrinsically kinded static argument."""
    if isinstance(term, TypeVariable | EffectVariable | IndexVariable | IndexLiteral):
        return
    if isinstance(term, TypeApplication):
        instantiate_telescope(term.constructor.telescope, term.arguments)
        return
    if isinstance(term, FunctionType):
        validate_static_argument(term.parameter)
        validate_static_argument(term.result)
        if static_kind(term.parameter) != TYPE:
            raise TypeError("function parameter must have Type kind")
        if static_kind(term.result) != TYPE:
            raise TypeError("function result must have Type kind")
        return
    if isinstance(term, EqualityType):
        validate_static_argument(term.left)
        validate_static_argument(term.right)
        index_kinds = (NatSort, ShapeSort, ContextSort, UserIndexSort)
        index_nodes = (IndexVariable, IndexLiteral, IndexConstructor, ShapeIndex)
        if isinstance(term.kind, index_kinds):
            if not isinstance(term.left, index_nodes) or not isinstance(
                term.right, index_nodes
            ):
                raise TypeError("index equality endpoints must be index terms")
            if not _sort_matches(term.kind, index_sort(term.left)) or not _sort_matches(
                term.kind, index_sort(term.right)
            ):
                raise TypeError("index equality endpoint has the wrong sort")
        elif (
            static_kind(term.left) != term.kind or static_kind(term.right) != term.kind
        ):
            raise TypeError("equality endpoint has the wrong kind")
        return
    if isinstance(term, EffectRef):
        for argument in term.arguments:
            validate_static_argument(argument)
        return
    if isinstance(term, IndexConstructor):
        for argument in term.arguments:
            validate_static_argument(argument)
        index_sort(term)
        return
    if isinstance(term, ShapeIndex):
        index_sort(term)
        for dimension in term.dimensions:
            validate_static_argument(dimension)
        return
    raise TypeError(f"unknown static argument {term!r}")


@dataclass(frozen=True, slots=True)
class StaticSubstitution:
    """A deterministic substitution split by static namespace."""

    types: tuple[tuple[str, TypeExpr], ...] = ()
    indices: tuple[tuple[str, IndexTerm], ...] = ()
    effects: tuple[tuple[str, EffectRef | EffectVariable], ...] = ()

    def __post_init__(self) -> None:
        for namespace in (self.types, self.indices, self.effects):
            names = [name for name, _ in namespace]
            if len(set(names)) != len(names):
                raise ValueError("duplicate substitution binding")

    def type(self, name: str) -> TypeExpr | None:
        return next((value for key, value in self.types if key == name), None)

    def index(self, name: str) -> IndexTerm | None:
        return next((value for key, value in self.indices if key == name), None)

    def effect(self, name: str) -> EffectRef | EffectVariable | None:
        return next((value for key, value in self.effects if key == name), None)


def instantiate_telescope(
    telescope: Telescope,
    arguments: tuple[StaticArgument, ...],
) -> StaticSubstitution:
    """Kind-check static arguments and construct their substitution."""
    if len(telescope) != len(arguments):
        raise TypeError(
            f"expected {len(telescope)} static arguments, got {len(arguments)}"
        )

    types: list[tuple[str, TypeExpr]] = []
    indices: list[tuple[str, IndexTerm]] = []
    effects: list[tuple[str, EffectRef | EffectVariable]] = []
    for binder, argument in zip(telescope, arguments, strict=True):
        validate_static_argument(argument)
        if isinstance(binder, TypeBinder):
            if not isinstance(
                argument,
                (TypeVariable, TypeApplication, FunctionType, EqualityType),
            ):
                raise TypeError(f"{binder.name!r} expects a type argument")
            if static_kind(argument) != binder.kind:
                raise TypeError(f"type argument for {binder.name!r} has wrong kind")
            types.append((binder.name, argument))
        elif isinstance(binder, IndexBinder):
            if not isinstance(
                argument,
                (IndexVariable, IndexConstructor, ShapeIndex),
            ) and not hasattr(argument, "sort"):
                raise TypeError(f"{binder.name!r} expects an index argument")
            actual_sort = index_sort(argument)  # type: ignore[arg-type]
            if not _sort_matches(binder.sort, actual_sort):
                raise TypeError(
                    f"index argument for {binder.name!r} has sort {actual_sort!r}, "
                    f"expected {binder.sort!r}"
                )
            indices.append((binder.name, argument))  # type: ignore[arg-type]
        elif isinstance(binder, EffectBinder):
            if not isinstance(argument, (EffectRef, EffectVariable)):
                raise TypeError(f"{binder.name!r} expects an effect argument")
            effects.append((binder.name, argument))
        else:  # pragma: no cover - closed binder union
            raise TypeError(f"unknown telescope binder {binder!r}")
    return StaticSubstitution(tuple(types), tuple(indices), tuple(effects))


def substitute_index(term: IndexTerm, substitution: StaticSubstitution) -> IndexTerm:
    if isinstance(term, IndexVariable) and term.identity is None:
        return substitution.index(term.name) or term
    if isinstance(term, IndexConstructor):
        return IndexConstructor(
            term.name,
            tuple(substitute_index(arg, substitution) for arg in term.arguments),
            term.sort,
        )
    if isinstance(term, ShapeIndex):
        return ShapeIndex(
            tuple(
                substitute_index(dimension, substitution)
                for dimension in term.dimensions
            )
        )
    return term


def substitute_effect(effect: EffectRef, substitution: StaticSubstitution) -> EffectRef:
    return EffectRef(
        effect.id,
        effect.name,
        effect.interface_version,
        tuple(substitute_static(arg, substitution) for arg in effect.arguments),
    )


def substitute_type(type_: TypeExpr, substitution: StaticSubstitution) -> TypeExpr:
    if isinstance(type_, TypeVariable) and type_.identity is None:
        return substitution.type(type_.name) or type_
    if isinstance(type_, TypeVariable):
        return type_
    if isinstance(type_, TypeApplication):
        return TypeApplication(
            type_.constructor,
            tuple(substitute_static(arg, substitution) for arg in type_.arguments),
        )
    if isinstance(type_, FunctionType):
        return FunctionType(
            substitute_type(type_.parameter, substitution),
            substitute_type(type_.result, substitution),
        )
    if isinstance(type_, EqualityType):
        return EqualityType(
            type_.kind,
            substitute_static(type_.left, substitution),
            substitute_static(type_.right, substitution),
        )
    raise TypeError(f"unknown type term {type_!r}")


def substitute_static(
    term: StaticArgument,
    substitution: StaticSubstitution,
) -> StaticArgument:
    if isinstance(term, EffectVariable) and term.identity is None:
        return substitution.effect(term.name) or term
    if isinstance(term, EffectVariable):
        return term
    if isinstance(term, EffectRef):
        return substitute_effect(term, substitution)
    if isinstance(term, (IndexVariable, IndexConstructor, ShapeIndex)) or hasattr(
        term, "sort"
    ):
        return substitute_index(term, substitution)  # type: ignore[arg-type]
    return substitute_type(term, substitution)  # type: ignore[arg-type]


def substitute_row(row: EffectRow, substitution: StaticSubstitution) -> EffectRow:
    return EffectRow(
        tuple(
            RowEntry(entry.instance, substitute_effect(entry.effect, substitution))
            for entry in row.entries
        ),
        row.tail,
    )


def instantiate_operation(
    operation: OperationDef,
    arguments: tuple[StaticArgument, ...],
    outer: StaticSubstitution = StaticSubstitution(),
) -> tuple[tuple[TypeExpr, ...], TypeExpr]:
    local = instantiate_telescope(operation.telescope, arguments)
    substitution = StaticSubstitution(
        (*outer.types, *local.types),
        (*outer.indices, *local.indices),
        (*outer.effects, *local.effects),
    )
    parameter_types = tuple(
        substitute_type(argument.type, substitution) for argument in operation.arguments
    )
    return parameter_types, substitute_type(operation.result_type, substitution)


__all__ = [
    "StaticSubstitution",
    "instantiate_operation",
    "instantiate_telescope",
    "substitute_effect",
    "substitute_index",
    "substitute_row",
    "substitute_static",
    "substitute_type",
    "validate_static_argument",
]
