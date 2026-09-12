"""Typed effect interfaces, lexical rows, and handler signatures for QIEC."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from quivers.qiec.identifiers import (
    EffectInstanceId,
    HandlerId,
    OperationId,
    RowVariableId,
    SiteProvenance,
)
from quivers.qiec.kinds import Telescope, validate_telescope
from quivers.qiec.types import EffectRef, StaticArgument, TypeExpr

if TYPE_CHECKING:
    from quivers.qiec.terms import Value


@dataclass(frozen=True, slots=True, eq=False)
class RowVariable:
    name: str
    identity: RowVariableId
    lacks: tuple[EffectInstanceId, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("row variable name cannot be empty")
        ordered = tuple(sorted(self.lacks, key=str))
        if len(set(ordered)) != len(ordered):
            raise ValueError("a row lacks constraint cannot repeat an instance")
        object.__setattr__(self, "lacks", ordered)

    def proves_lacks(self, instance: EffectInstanceId) -> bool:
        return instance in self.lacks

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RowVariable)
            and self.identity == other.identity
            and self.lacks == other.lacks
        )

    def __hash__(self) -> int:
        return hash((self.identity, self.lacks))

    def merge(self, other: RowVariable) -> RowVariable:
        """Unify two occurrences, accumulating their lacks constraints."""
        if self.identity != other.identity:
            raise ValueError("cannot unify distinct row variables")
        merged = tuple(sorted(set((*self.lacks, *other.lacks)), key=str))
        return RowVariable(self.name, self.identity, merged)

    def with_lacks(self, instances: tuple[EffectInstanceId, ...]) -> RowVariable:
        merged = tuple(sorted(set((*self.lacks, *instances)), key=str))
        return RowVariable(self.name, self.identity, merged)


@dataclass(frozen=True, slots=True)
class RowEntry:
    """One lexical effect instance and the interface it implements."""

    instance: EffectInstanceId
    effect: EffectRef


@dataclass(frozen=True, slots=True)
class EffectRow:
    """A finite map from lexical instance IDs to interfaces plus a tail.

    Entries are sorted by stable identifier at construction time, so row
    equality is insensitive to source order.  Two instances of the same
    interface remain distinct entries.
    """

    entries: tuple[RowEntry, ...] = ()
    tail: RowVariable | None = None

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.entries, key=lambda entry: str(entry.instance)))
        instances = [entry.instance for entry in ordered]
        if len(set(instances)) != len(instances):
            raise ValueError("an effect row cannot contain a lexical instance twice")
        if self.tail is not None:
            unconstrained = [
                instance
                for instance in instances
                if not self.tail.proves_lacks(instance)
            ]
            if unconstrained:
                raise ValueError(
                    "an open row must prove its tail lacks every explicit instance"
                )
        object.__setattr__(self, "entries", ordered)

    def lookup(self, instance: EffectInstanceId) -> EffectRef | None:
        return next(
            (entry.effect for entry in self.entries if entry.instance == instance),
            None,
        )

    def contains(self, instance: EffectInstanceId) -> bool:
        return self.lookup(instance) is not None

    def add(self, entry: RowEntry) -> EffectRow:
        existing = self.lookup(entry.instance)
        if existing is not None and existing != entry.effect:
            raise ValueError("one lexical instance cannot implement two interfaces")
        if existing is not None:
            return self
        tail = self.tail
        if tail is not None and not tail.proves_lacks(entry.instance):
            tail = tail.with_lacks((entry.instance,))
        return EffectRow((*self.entries, entry), tail)

    def without(self, instance: EffectInstanceId) -> EffectRow:
        if not self.contains(instance):
            raise KeyError(str(instance))
        return EffectRow(
            tuple(entry for entry in self.entries if entry.instance != instance),
            self.tail,
        )

    def union(self, other: EffectRow) -> EffectRow:
        """Union two rows, preserving a compatible open tail."""
        if self.tail is not None and other.tail is not None:
            tail = self.tail.merge(other.tail)
        else:
            tail = self.tail or other.tail
        if tail is not None:
            tail = tail.with_lacks(
                tuple(entry.instance for entry in (*self.entries, *other.entries))
            )
        result = EffectRow(self.entries, tail)
        for entry in other.entries:
            result = result.add(entry)
        return result


EMPTY_ROW = EffectRow()


@dataclass(frozen=True, slots=True)
class RowSubstitution:
    """A finite, occurs-checked substitution for stable row variables."""

    bindings: tuple[tuple[RowVariableId, EffectRow], ...] = ()

    def __post_init__(self) -> None:
        identities = [identity for identity, _ in self.bindings]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate row substitution binding")
        for identity, row in self.bindings:
            if row.tail is not None and row.tail.identity == identity:
                raise ValueError("recursive row substitution")

    def lookup(self, variable: RowVariable) -> EffectRow | None:
        return next(
            (row for identity, row in self.bindings if identity == variable.identity),
            None,
        )

    def apply(self, row: EffectRow) -> EffectRow:
        return self._apply(row, frozenset())

    def _apply(
        self,
        row: EffectRow,
        visiting: frozenset[RowVariableId],
    ) -> EffectRow:
        if row.tail is None:
            return row
        replacement = self.lookup(row.tail)
        if replacement is None:
            return row
        if row.tail.identity in visiting:
            raise ValueError("recursive row substitution")
        resolved = self._apply(replacement, visiting | {row.tail.identity})
        forbidden = set(row.tail.lacks)
        if any(entry.instance in forbidden for entry in resolved.entries):
            raise ValueError("row substitution violates a lacks constraint")
        if resolved.tail is not None:
            resolved = EffectRow(
                resolved.entries,
                resolved.tail.with_lacks(row.tail.lacks),
            )
        return EffectRow(row.entries).union(resolved)


@dataclass(frozen=True, slots=True)
class RowUnification:
    row: EffectRow
    substitution: RowSubstitution = RowSubstitution()


def unify_effect_rows(left: EffectRow, right: EffectRow) -> RowUnification:
    """Unify two rows, returning their common row and tail substitution."""
    left_entries = {entry.instance: entry.effect for entry in left.entries}
    right_entries = {entry.instance: entry.effect for entry in right.entries}
    for instance in left_entries.keys() & right_entries.keys():
        if left_entries[instance] != right_entries[instance]:
            raise ValueError("row entries disagree on an effect interface")
    left_only = tuple(
        RowEntry(instance, effect)
        for instance, effect in left_entries.items()
        if instance not in right_entries
    )
    right_only = tuple(
        RowEntry(instance, effect)
        for instance, effect in right_entries.items()
        if instance not in left_entries
    )

    if left.tail is None and right.tail is None:
        if left_only or right_only:
            raise ValueError("closed effect rows do not unify")
        return RowUnification(left)

    if left.tail is not None and right.tail is None:
        if left_only:
            raise ValueError("an open row has entries absent from the closed row")
        replacement = EffectRow(right_only)
        substitution = RowSubstitution(((left.tail.identity, replacement),))
        if substitution.apply(left) != right:
            raise ValueError("row substitution violates a lacks constraint")
        return RowUnification(right, substitution)

    if left.tail is None and right.tail is not None:
        result = unify_effect_rows(right, left)
        return RowUnification(result.row, result.substitution)

    assert left.tail is not None and right.tail is not None
    if left.tail.identity == right.tail.identity:
        if left_only or right_only:
            raise ValueError(
                "two occurrences of one row variable have incompatible entries"
            )
        tail = left.tail.merge(right.tail)
        return RowUnification(EffectRow(left.entries, tail))

    if any(entry.instance in left.tail.lacks for entry in right_only):
        raise ValueError("left row lacks an entry required by the right row")
    if any(entry.instance in right.tail.lacks for entry in left_only):
        raise ValueError("right row lacks an entry required by the left row")
    all_instances = tuple((*left_entries.keys(), *right_entries.keys()))
    fresh_identity = RowVariableId.derive(
        "unify",
        *sorted((left.tail.identity, right.tail.identity), key=str),
        *sorted(all_instances, key=str),
    )
    fresh = RowVariable(
        "unified_tail",
        fresh_identity,
        tuple(
            sorted(
                set((*left.tail.lacks, *right.tail.lacks, *all_instances)),
                key=str,
            )
        ),
    )
    left_replacement = EffectRow(right_only, fresh)
    right_replacement = EffectRow(left_only, fresh)
    substitution = RowSubstitution(
        (
            (left.tail.identity, left_replacement),
            (right.tail.identity, right_replacement),
        )
    )
    unified_entries = tuple(
        RowEntry(instance, effect)
        for instance, effect in {**left_entries, **right_entries}.items()
    )
    unified = EffectRow(unified_entries, fresh)
    if substitution.apply(left) != unified or substitution.apply(right) != unified:
        raise ValueError("row unification produced an inconsistent substitution")
    return RowUnification(unified, substitution)


@dataclass(frozen=True, slots=True)
class ComputationType:
    effects: EffectRow
    result: TypeExpr


@dataclass(frozen=True, slots=True)
class ArgumentDef:
    name: str
    type: TypeExpr


@dataclass(frozen=True, slots=True)
class OperationDef:
    """One result-indexed request constructor in an effect interface."""

    id: OperationId
    name: str
    telescope: Telescope
    arguments: tuple[ArgumentDef, ...]
    result_type: TypeExpr

    def __post_init__(self) -> None:
        validate_telescope(self.telescope)
        names = [argument.name for argument in self.arguments]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate argument in operation {self.name!r}")


class InterfaceEvolution(str, Enum):
    """How adding operations to an interface affects existing handlers."""

    SEALED = "sealed"
    FORWARDING = "forwarding"


@dataclass(frozen=True, slots=True)
class EffectDef:
    """A versioned operation interface.

    A sealed interface requires a new version and identity when its operation
    set changes.  A forwarding interface permits extension only for handlers
    that explicitly forward unknown requests.
    """

    ref: EffectRef
    telescope: Telescope
    operations: tuple[OperationDef, ...]
    evolution: InterfaceEvolution = InterfaceEvolution.SEALED

    def __post_init__(self) -> None:
        validate_telescope(self.telescope)
        if self.ref.arguments:
            raise ValueError("an effect declaration ref must be unsaturated")
        ids = [operation.id for operation in self.operations]
        names = [operation.name for operation in self.operations]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate operation ID in effect {self.ref.name!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate operation name in effect {self.ref.name!r}")
        interface_names = {binder.name for binder in self.telescope}
        for operation in self.operations:
            overlap = interface_names & {binder.name for binder in operation.telescope}
            if overlap:
                raise ValueError(
                    f"operation {operation.name!r} shadows interface binders: {overlap!r}"
                )

    def apply(self, arguments: tuple[StaticArgument, ...]) -> EffectRef:
        """Instantiate the interface telescope as a concrete effect."""
        from quivers.qiec.substitution import instantiate_telescope

        instantiate_telescope(self.telescope, arguments)
        return EffectRef(
            self.ref.id,
            self.ref.name,
            self.ref.interface_version,
            arguments,
        )

    def matches(self, effect: EffectRef) -> bool:
        """Whether a concrete effect is a well-kinded application here."""
        if (
            effect.id != self.ref.id
            or effect.interface_version != self.ref.interface_version
        ):
            return False
        try:
            self.apply(effect.arguments)
        except TypeError, ValueError:
            return False
        return True

    def operation(self, operation: OperationId) -> OperationDef:
        for candidate in self.operations:
            if candidate.id == operation:
                return candidate
        raise KeyError(str(operation))


class ResumptionGrade(str, Enum):
    """Quantitative use allowed for a handler-clause resumption."""

    ZERO = "0"
    AFFINE = "aff"
    LINEAR = "1"
    UNRESTRICTED = "omega"


@dataclass(frozen=True, slots=True)
class HandlerClauseDef:
    operation: OperationId
    grade: ResumptionGrade


@dataclass(frozen=True, slots=True)
class HandlerDef:
    """Serializable type-and-coverage signature for a handler.

    Executable clause bodies are runtime attachments keyed by ``id``.  They
    never enter the stable core representation.
    """

    id: HandlerId
    name: str
    effect: EffectRef
    clauses: tuple[HandlerClauseDef, ...]
    input_type: TypeExpr
    output_type: TypeExpr
    introduced: EffectRow = EMPTY_ROW
    total: bool = True
    forwards_unknown: bool = False
    telescope: Telescope = ()

    def __post_init__(self) -> None:
        validate_telescope(self.telescope)
        operations = [clause.operation for clause in self.clauses]
        if len(set(operations)) != len(operations):
            raise ValueError(f"duplicate clause in handler {self.name!r}")
        if self.total and self.forwards_unknown:
            raise ValueError("a handler cannot be both total and forwarding")

    def clause(self, operation: OperationId) -> HandlerClauseDef | None:
        return next(
            (clause for clause in self.clauses if clause.operation == operation),
            None,
        )


@dataclass(frozen=True, slots=True)
class EffectRequest:
    """A typed request allocated before filtering or handler dispatch."""

    instance: EffectInstanceId
    effect: EffectRef
    operation: OperationId
    static_arguments: tuple[StaticArgument, ...]
    arguments: tuple[Value, ...]
    result_type: TypeExpr
    origin: SiteProvenance
    tag: Literal["effect_request"] = "effect_request"


def instantiate_effect(
    effect: EffectRef,
    *,
    module: str,
    lexical_path: tuple[str | int, ...],
) -> RowEntry:
    """Allocate a deterministic lexical instance of an effect interface."""
    instance = EffectInstanceId.derive(
        str(effect.id),
        effect.interface_version,
        effect.arguments,
        module,
        lexical_path,
    )
    return RowEntry(instance, effect)


__all__ = [
    "ArgumentDef",
    "ComputationType",
    "EMPTY_ROW",
    "EffectDef",
    "EffectRequest",
    "EffectRow",
    "HandlerClauseDef",
    "HandlerDef",
    "InterfaceEvolution",
    "OperationDef",
    "ResumptionGrade",
    "RowEntry",
    "RowSubstitution",
    "RowUnification",
    "RowVariable",
    "instantiate_effect",
    "unify_effect_rows",
]
