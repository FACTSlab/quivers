"""Indexed-family and constructor declarations for the QIEC kernel."""

from __future__ import annotations

from dataclasses import dataclass

from quivers.qiec.identifiers import ConstructorId, FamilyId, TypeId
from quivers.qiec.kinds import Telescope, validate_telescope
from quivers.qiec.types import StaticArgument, TypeConstructorRef, TypeExpr


@dataclass(frozen=True, slots=True)
class FieldDef:
    name: str
    type: TypeExpr


@dataclass(frozen=True, slots=True)
class FamilyDecl:
    """A closed or opaque indexed data family.

    ``parameters`` are uniform across every constructor.  ``indices`` may be
    refined by constructor results and therefore are kept separate even when
    both are represented by index binders.
    """

    id: FamilyId
    name: str
    parameters: Telescope
    indices: Telescope
    constructors: tuple[ConstructorId, ...]
    closed: bool = True

    def __post_init__(self) -> None:
        validate_telescope(self.parameters)
        validate_telescope(self.indices)
        names = [binder.name for binder in (*self.parameters, *self.indices)]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate binder in family {self.name!r}")
        if any(not binder.refinable for binder in self.indices):
            raise ValueError("family indices must be marked refinable")
        if len(set(self.constructors)) != len(self.constructors):
            raise ValueError(f"duplicate constructor in family {self.name!r}")

    @property
    def type_constructor(self) -> TypeConstructorRef:
        return TypeConstructorRef(
            TypeId.derive("family", str(self.id)),
            self.name,
            (*self.parameters, *self.indices),
        )


@dataclass(frozen=True, slots=True)
class ConstructorDecl:
    """One GADT constructor.

    Constructor-local telescope binders are rigid skolems while a branch is
    checked.  ``result_indices`` may mention those binders and thereby refine
    the scrutinee indices.
    """

    id: ConstructorId
    family: FamilyId
    name: str
    telescope: Telescope
    fields: tuple[FieldDef, ...]
    result_indices: tuple[StaticArgument, ...]

    def __post_init__(self) -> None:
        validate_telescope(self.telescope)
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field in constructor {self.name!r}")


__all__ = [
    "ConstructorDecl",
    "FamilyDecl",
    "FieldDef",
]
