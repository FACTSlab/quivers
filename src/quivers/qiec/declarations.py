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

    Parameters
    ----------
    id : FamilyId
        Stable identity.
    name : str
        Source name, used for display and for the derived type
        constructor.
    parameters : Telescope
        Binders uniform across every constructor.
    indices : Telescope
        Binders a constructor result may refine. Each must be marked
        refinable, which is what separates them from parameters.
    constructors : tuple[ConstructorId, ...]
        The constructors this family declares, without repetition.
    closed : bool
        Whether the constructor list is exhaustive. A closed family
        admits coverage checking; an opaque one does not.
    """

    id: FamilyId
    name: str
    parameters: Telescope
    indices: Telescope
    constructors: tuple[ConstructorId, ...]
    closed: bool = True

    def __post_init__(self) -> None:
        """Reject a family whose binders or constructors are malformed.

        Raises
        ------
        ValueError
            If either telescope is malformed, a name is shared between a
            parameter and an index, an index is not marked refinable, or
            a constructor is listed twice.
        """
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
        """The type constructor this family introduces.

        Returns
        -------
        TypeConstructorRef
            A constructor whose identity is derived from the family's,
            taking the parameters and indices together as its telescope.
            Deriving rather than storing it keeps the two from drifting
            apart.
        """
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

    Parameters
    ----------
    id : ConstructorId
        Stable identity.
    family : FamilyId
        The family this constructor belongs to, which must list it.
    name : str
        Source name.
    telescope : Telescope
        Binders local to the constructor. These become rigid skolems
        while a branch matching it is checked.
    fields : tuple[FieldDef, ...]
        The values the constructor carries, uniquely named.
    result_indices : tuple[StaticArgument, ...]
        What the constructor fixes the family's indices to. One per
        family index, and may mention the local telescope, which is how
        matching refines the scrutinee's type.
    """

    id: ConstructorId
    family: FamilyId
    name: str
    telescope: Telescope
    fields: tuple[FieldDef, ...]
    result_indices: tuple[StaticArgument, ...]

    def __post_init__(self) -> None:
        """Reject a constructor whose telescope or fields are malformed.

        Raises
        ------
        ValueError
            If the telescope is malformed or two fields share a name.
        """
        validate_telescope(self.telescope)
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field in constructor {self.name!r}")


__all__ = [
    "ConstructorDecl",
    "FamilyDecl",
    "FieldDef",
]
