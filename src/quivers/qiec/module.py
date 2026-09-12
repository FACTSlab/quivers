"""Serializable, whole-module containers for the QIEC kernel.

The small records in this module are the boundary returned by source-language
extension lowerers.  They contain declarations and checked computation graphs,
but never parser nodes, compiler objects, callbacks, or backend-specific data.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass

from quivers.qiec.checking import (
    CheckContext,
    KernelError,
    KernelRegistry,
    infer_computation,
)
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl
from quivers.qiec.effects import (
    ComputationType,
    EffectDef,
    EffectRequest,
    EffectRow,
    HandlerDef,
    RowEntry,
)
from quivers.qiec.identifiers import (
    EffectInstanceId,
    QIEC_ABI,
    SiteProvenance,
    SourceOrigin,
)
from quivers.qiec.kinds import (
    EffectBinder,
    IndexBinder,
    Telescope,
    TypeBinder,
    UserIndexSort,
    validate_telescope,
)
from quivers.qiec.terms import CaseMotive, Computation, Handle, Local
from quivers.qiec.types import EffectRef, EffectVariable, IndexVariable, TypeVariable


@dataclass(frozen=True, slots=True)
class NamedEffectInstance:
    """A source name for one deterministically allocated lexical instance."""

    name: str
    entry: RowEntry
    origin: SourceOrigin

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an effect instance name cannot be empty")


@dataclass(frozen=True, slots=True)
class NamedComputation:
    """One named QVR computation and its checked QIEC type."""

    name: str
    telescope: Telescope
    parameters: tuple[Local, ...]
    body: Computation
    type: ComputationType
    origin: SourceOrigin

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a computation name cannot be empty")
        validate_telescope(self.telescope)
        static_names = {binder.name for binder in self.telescope}
        local_names = [parameter.name for parameter in self.parameters]
        if len(set(local_names)) != len(local_names):
            raise ValueError(f"duplicate parameter in computation {self.name!r}")
        overlap = static_names & set(local_names)
        if overlap:
            raise ValueError(
                f"computation {self.name!r} reuses static names as parameters: "
                f"{sorted(overlap)!r}"
            )


@dataclass(frozen=True, slots=True)
class QiecModule:
    """A complete, serializable QIEC compilation unit.

    ``source_protocol`` records the exact source route that produced the
    module.  The wire envelope separately records :data:`QIEC_ABI`, so neither
    endpoint can be guessed during deserialization or compiler integration.
    """

    module: str
    source_protocol: str
    index_sorts: tuple[UserIndexSort, ...] = ()
    families: tuple[FamilyDecl, ...] = ()
    constructors: tuple[ConstructorDecl, ...] = ()
    effects: tuple[EffectDef, ...] = ()
    instances: tuple[NamedEffectInstance, ...] = ()
    handlers: tuple[HandlerDef, ...] = ()
    computations: tuple[NamedComputation, ...] = ()
    abi: str = QIEC_ABI

    def __post_init__(self) -> None:
        if not isinstance(self.module, str) or not self.module:
            raise ValueError("a QIEC module name cannot be empty")
        if not isinstance(self.source_protocol, str) or not self.source_protocol:
            raise ValueError("a QIEC source protocol cannot be empty")
        if self.abi != QIEC_ABI:
            raise ValueError(f"QIEC module ABI must be {QIEC_ABI!r}")
        _require_unique(
            (sort.name for sort in self.index_sorts),
            subject="index sort name",
        )
        _require_unique((family.id for family in self.families), subject="family ID")
        _require_unique(
            (family.name for family in self.families), subject="family name"
        )
        _require_unique(
            (constructor.id for constructor in self.constructors),
            subject="constructor ID",
        )
        _require_unique((effect.ref.id for effect in self.effects), subject="effect ID")
        _require_unique(
            (effect.ref.name for effect in self.effects), subject="effect name"
        )
        _require_unique(
            (instance.entry.instance for instance in self.instances),
            subject="effect-instance ID",
        )
        _require_unique(
            (instance.name for instance in self.instances),
            subject="effect-instance name",
        )
        _require_unique((handler.id for handler in self.handlers), subject="handler ID")
        _require_unique(
            (handler.name for handler in self.handlers), subject="handler name"
        )
        _require_unique(
            (computation.name for computation in self.computations),
            subject="computation name",
        )

        declared_constructors = {
            constructor.id: constructor for constructor in self.constructors
        }
        family_ids = {family.id for family in self.families}
        expected_constructor_ids = {
            constructor
            for family in self.families
            for constructor in family.constructors
        }
        if set(declared_constructors) != expected_constructor_ids:
            raise ValueError(
                "module constructors must exactly match the constructors named by families"
            )
        if any(
            constructor.family not in family_ids
            for constructor in declared_constructors.values()
        ):
            raise ValueError("a module constructor names an unknown family")

        effect_ids = {effect.ref.id for effect in self.effects}
        if any(
            instance.entry.effect.id not in effect_ids for instance in self.instances
        ):
            raise ValueError("an effect instance names an unknown interface")
        if any(handler.effect.id not in effect_ids for handler in self.handlers):
            raise ValueError("a handler names an unknown effect interface")
        for item in (*self.instances, *self.computations):
            if item.origin.module != self.module:
                raise ValueError("source origin belongs to another QIEC module")
            if item.origin.source_protocol != self.source_protocol:
                raise ValueError("source origin uses another source protocol")


def _require_unique(values: object, *, subject: str) -> None:
    materialized = tuple(values)  # type: ignore[arg-type]
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"duplicate {subject} in QIEC module")


def validate_module(module: QiecModule) -> KernelRegistry:
    """Recheck a whole lowered module and return its resolved registry."""

    if module.abi != QIEC_ABI:
        raise KernelError(f"unsupported QIEC ABI {module.abi!r}")
    _validate_index_sorts(module)
    registry = KernelRegistry()
    for family in module.families:
        registry.register_family(family)
    for constructor in module.constructors:
        registry.register_constructor(constructor)
    for effect in module.effects:
        registry.register_effect(effect)
    for instance in module.instances:
        _validate_closed_effect_instance(instance)
        registry.validate_static(instance.entry.effect)
        declared = registry.effects.get(instance.entry.effect.id)
        if declared is None or not declared.matches(instance.entry.effect):
            raise KernelError(
                f"unknown effect interface for instance {instance.name!r}"
            )
    for handler in module.handlers:
        registry.register_handler(handler)
    instance_interfaces = {
        instance.entry.instance: instance.entry.effect for instance in module.instances
    }
    for handler in module.handlers:
        _validate_effect_instances(
            handler.introduced,
            instance_interfaces,
            subject=f"handler {handler.name!r}",
            module=module.module,
            source_protocol=module.source_protocol,
        )
    for computation in module.computations:
        _validate_named_static_scope(computation, computation.telescope)
        registry.validate_computation_type(computation.type)
        _validate_effect_instances(
            computation.type.effects,
            instance_interfaces,
            subject=f"declared row of computation {computation.name!r}",
            module=module.module,
            source_protocol=module.source_protocol,
        )
        _validate_effect_instances(
            computation.body,
            instance_interfaces,
            subject=f"body of computation {computation.name!r}",
            module=module.module,
            source_protocol=module.source_protocol,
        )
        context = CheckContext()
        for parameter in computation.parameters:
            registry.validate_type(parameter.type)
            context = context.extend(parameter)
        actual = infer_computation(computation.body, registry, context)
        if not computation_type_conforms(actual, computation.type):
            raise KernelError(
                f"computation {computation.name!r} has type {actual!r}, "
                f"not declared type {computation.type!r}"
            )
    return registry


def computation_type_conforms(
    actual: ComputationType,
    declared: ComputationType,
) -> bool:
    """Check exact rows or sound weakening into an explicitly open row.

    An inferred open tail can inhabit a declared open row only when it proves
    every absence guaranteed by the declaration. Row-variable names and IDs
    are lexical identities, so conformance compares their constraints rather
    than treating unrelated tails as identical.
    """

    if actual.result != declared.result:
        return False
    if declared.effects.tail is None:
        return actual.effects == declared.effects
    actual_entries = {entry.instance: entry.effect for entry in actual.effects.entries}
    declared_entries = {
        entry.instance: entry.effect for entry in declared.effects.entries
    }
    for instance, effect in actual_entries.items():
        expected = declared_entries.get(instance)
        if expected is not None:
            if expected != effect:
                return False
            continue
        if declared.effects.tail.proves_lacks(instance):
            return False
    if actual.effects.tail is None:
        return True
    forbidden = set(declared.effects.tail.lacks) - set(declared_entries)
    if any(instance in actual_entries for instance in forbidden):
        return False
    if any(not actual.effects.tail.proves_lacks(instance) for instance in forbidden):
        return False
    return True


def _validate_effect_instances(
    value: object,
    interfaces: dict[EffectInstanceId, EffectRef],
    *,
    subject: str,
    module: str,
    source_protocol: str,
) -> None:
    """Validate every concrete lexical instance in a module-owned object."""

    def require(
        instance: EffectInstanceId,
        effect: EffectRef | None = None,
    ) -> None:
        declared = interfaces.get(instance)
        if declared is None:
            raise KernelError(f"{subject} uses an undeclared lexical effect instance")
        if effect is not None and declared != effect:
            raise KernelError(
                f"{subject} assigns the wrong interface to a lexical effect instance"
            )

    def visit(node: object) -> None:
        if isinstance(node, EffectRow):
            for entry in node.entries:
                require(entry.instance, entry.effect)
            if node.tail is not None:
                for instance in node.tail.lacks:
                    require(instance)
            return
        if isinstance(node, EffectRequest):
            require(node.instance, node.effect)
            if not isinstance(node.origin, SiteProvenance):
                raise KernelError(f"{subject} has malformed request provenance")
            request_origin = node.origin.origin
            if not isinstance(request_origin, SourceOrigin):
                raise KernelError(f"{subject} has malformed request provenance")
            if request_origin.module != module:
                raise KernelError(
                    f"{subject} has request provenance from another QIEC module"
                )
            if request_origin.source_protocol != source_protocol:
                raise KernelError(
                    f"{subject} has request provenance from another source protocol"
                )
        elif isinstance(node, Handle):
            require(node.instance)
        if isinstance(node, tuple):
            for item in node:
                visit(item)
            return
        if is_dataclass(node) and not isinstance(node, type):
            for field in fields(node):
                visit(getattr(node, field.name))

    visit(value)


def _validate_closed_effect_instance(instance: NamedEffectInstance) -> None:
    """Require module-level effect applications to be fully closed."""

    def visit(node: object) -> None:
        if isinstance(node, (TypeVariable, IndexVariable, EffectVariable)):
            identity = "identity-bearing" if node.identity is not None else "free named"
            raise KernelError(
                f"module-level effect instance {instance.name!r} contains an "
                f"{identity} static variable {node.name!r}"
            )
        if isinstance(node, tuple):
            for item in node:
                visit(item)
            return
        if is_dataclass(node) and not isinstance(node, type):
            for field in fields(node):
                visit(getattr(node, field.name))

    visit(instance.entry.effect.arguments)


def _validate_index_sorts(module: QiecModule) -> None:
    """Require every embedded user index sort to name its module declaration."""

    declared = {sort.name: sort for sort in module.index_sorts}

    def visit(node: object) -> None:
        if isinstance(node, UserIndexSort):
            expected = declared.get(node.name)
            if expected is None:
                raise KernelError(f"undeclared user index sort {node.name!r}")
            if node != expected:
                raise KernelError(
                    f"embedded index sort {node.name!r} disagrees with its declaration"
                )
            return
        if isinstance(node, tuple):
            for item in node:
                visit(item)
            return
        if is_dataclass(node) and not isinstance(node, type):
            for field in fields(node):
                visit(getattr(node, field.name))

    for item in (
        *module.families,
        *module.constructors,
        *module.effects,
        *module.instances,
        *module.handlers,
        *module.computations,
    ):
        visit(item)


def _validate_named_static_scope(value: object, telescope: Telescope) -> None:
    """Reject source variables not introduced by a computation telescope."""

    def bindings(
        scope: Telescope,
    ) -> tuple[
        dict[str, TypeBinder],
        dict[str, IndexBinder],
        dict[str, EffectBinder],
    ]:
        return (
            {binder.name: binder for binder in scope if isinstance(binder, TypeBinder)},
            {
                binder.name: binder
                for binder in scope
                if isinstance(binder, IndexBinder)
            },
            {
                binder.name: binder
                for binder in scope
                if isinstance(binder, EffectBinder)
            },
        )

    def visit(
        node: object,
        type_binders: dict[str, TypeBinder],
        index_binders: dict[str, IndexBinder],
        effect_binders: dict[str, EffectBinder],
    ) -> None:
        if isinstance(node, TypeVariable) and node.identity is None:
            binder = type_binders.get(node.name)
            if binder is None or binder.kind != node.kind:
                raise KernelError(f"unbound or mistyped type variable {node.name!r}")
            return
        if isinstance(node, IndexVariable) and node.identity is None:
            binder = index_binders.get(node.name)
            if binder is None or binder.sort != node.sort:
                raise KernelError(f"unbound or mistyped index variable {node.name!r}")
            return
        if isinstance(node, EffectVariable) and node.identity is None:
            if node.name not in effect_binders:
                raise KernelError(f"unbound effect variable {node.name!r}")
            return
        if isinstance(node, CaseMotive):
            nested_types = dict(type_binders)
            nested_indices = dict(index_binders)
            nested_effects = dict(effect_binders)
            for binder in node.indices:
                if isinstance(binder, TypeBinder):
                    nested_types[binder.name] = binder
                elif isinstance(binder, IndexBinder):
                    nested_indices[binder.name] = binder
                else:
                    nested_effects[binder.name] = binder
            visit(
                node.result_type,
                nested_types,
                nested_indices,
                nested_effects,
            )
            return
        if isinstance(node, tuple):
            for item in node:
                visit(item, type_binders, index_binders, effect_binders)
            return
        if is_dataclass(node) and not isinstance(node, type):
            for field in fields(node):
                visit(
                    getattr(node, field.name),
                    type_binders,
                    index_binders,
                    effect_binders,
                )

    visit(value, *bindings(telescope))


__all__ = [
    "computation_type_conforms",
    "NamedComputation",
    "NamedEffectInstance",
    "QiecModule",
    "validate_module",
]
