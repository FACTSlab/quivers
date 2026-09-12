"""Validation and type/effect inference for the QIEC reference kernel."""

from __future__ import annotations

from dataclasses import dataclass, field

from quivers.qiec.declarations import ConstructorDecl, FamilyDecl
from quivers.qiec.effects import (
    ComputationType,
    EMPTY_ROW,
    EffectDef,
    EffectRequest,
    EffectRow,
    HandlerDef,
    InterfaceEvolution,
    OperationDef,
    RowEntry,
)
from quivers.qiec.coverage import (
    BranchPattern,
    CoverageStatus,
    Reachability,
    check_indexed_coverage,
    constructor_skolems,
    join_branch_rows,
    refine_branch,
)
from quivers.qiec.evidence import BranchGiven, EqualityEvidence, Reflexivity
from quivers.qiec.identifiers import (
    ConstructorId,
    EffectId,
    EqualityId,
    FamilyId,
    HandlerId,
    OperationId,
    StaticScopeId,
    StaticVariableId,
    TypeId,
)
from quivers.qiec.substitution import (
    StaticSubstitution,
    instantiate_operation,
    instantiate_telescope,
    substitute_effect,
    substitute_row,
    substitute_static,
    substitute_type,
)
from quivers.qiec.kinds import (
    TYPE,
    ContextSort,
    EffectBinder,
    IndexBinder,
    IndexSort,
    NatSort,
    ShapeSort,
    TypeBinder,
    UserIndexSort,
)
from quivers.qiec.terms import (
    AttachmentRef,
    Bind,
    Case,
    Computation,
    ConstructorValue,
    EvidenceValue,
    Handle,
    LiteralValue,
    Local,
    Perform,
    Return,
    TransportValue,
    Value,
    Var,
)
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    UNIT,
    EffectRef,
    EffectVariable,
    EqualityType,
    FunctionType,
    IndexConstructor,
    IndexLiteral,
    IndexVariable,
    ShapeIndex,
    StaticArgument,
    TypeApplication,
    TypeConstructorRef,
    TypeExpr,
    TypeVariable,
    index_sort,
    static_kind,
)


class KernelError(TypeError):
    """A static QIEC kernel rejection."""


@dataclass(frozen=True, slots=True)
class _StaticScope:
    """The three disjoint namespaces visible in a declaration body."""

    types: tuple[tuple[str, object], ...] = ()
    indices: tuple[tuple[str, IndexSort], ...] = ()
    effects: tuple[str, ...] = ()

    @property
    def names(self) -> frozenset[str]:
        return frozenset(
            (
                *[name for name, _ in self.types],
                *[name for name, _ in self.indices],
                *self.effects,
            )
        )

    def extend(self, telescope: tuple[object, ...], *, subject: str) -> _StaticScope:
        types = list(self.types)
        indices = list(self.indices)
        effects = list(self.effects)
        seen = set(self.names)
        for binder in telescope:
            name = getattr(binder, "name", None)
            if not isinstance(name, str):  # pragma: no cover - closed binder union
                raise KernelError(f"unknown binder in {subject}: {binder!r}")
            if name in seen:
                raise KernelError(f"static binder {name!r} is shadowed in {subject}")
            seen.add(name)
            if isinstance(binder, TypeBinder):
                types.append((binder.name, binder.kind))
            elif isinstance(binder, IndexBinder):
                indices.append((binder.name, binder.sort))
            elif isinstance(binder, EffectBinder):
                effects.append(binder.name)
            else:  # pragma: no cover - closed binder union
                raise KernelError(f"unknown binder in {subject}: {binder!r}")
        return _StaticScope(tuple(types), tuple(indices), tuple(effects))

    def type_kind(self, name: str) -> object | None:
        return next((kind for candidate, kind in self.types if candidate == name), None)

    def index_sort(self, name: str) -> IndexSort | None:
        return next(
            (sort for candidate, sort in self.indices if candidate == name), None
        )


@dataclass(slots=True)
class KernelRegistry:
    """Resolved declarations used by the reference checker."""

    families: dict[FamilyId, FamilyDecl] = field(default_factory=dict)
    constructors: dict[ConstructorId, ConstructorDecl] = field(default_factory=dict)
    effects: dict[EffectId, EffectDef] = field(default_factory=dict)
    operations: dict[OperationId, tuple[EffectDef, OperationDef]] = field(
        default_factory=dict
    )
    handlers: dict[HandlerId, HandlerDef] = field(default_factory=dict)
    type_constructors: dict[TypeId, TypeConstructorRef] = field(default_factory=dict)

    def _record_type_constructor(self, constructor: TypeConstructorRef) -> None:
        existing = self.type_constructors.get(constructor.id)
        if existing is not None and existing.telescope != constructor.telescope:
            raise KernelError(
                f"conflicting telescope metadata for type constructor {constructor.id}"
            )
        if existing is None:
            self.type_constructors[constructor.id] = constructor

    def _record_type_constructors(self, term: StaticArgument) -> None:
        if isinstance(term, TypeApplication):
            self._record_type_constructor(term.constructor)
            for argument in term.arguments:
                self._record_type_constructors(argument)
            return
        if isinstance(term, FunctionType):
            self._record_type_constructors(term.parameter)
            self._record_type_constructors(term.result)
            return
        if isinstance(term, EqualityType):
            self._record_type_constructors(term.left)
            self._record_type_constructors(term.right)
            return
        if isinstance(term, EffectRef | IndexConstructor):
            for argument in term.arguments:
                self._record_type_constructors(argument)
            return
        if isinstance(term, ShapeIndex):
            for dimension in term.dimensions:
                self._record_type_constructors(dimension)

    def _validate_effect_references(
        self,
        term: StaticArgument,
        additional: tuple[EffectDef, ...] = (),
    ) -> None:
        if isinstance(term, EffectRef):
            for argument in term.arguments:
                self._validate_effect_references(argument, additional)
            definition = self.effects.get(term.id) or next(
                (candidate for candidate in additional if candidate.ref.id == term.id),
                None,
            )
            if definition is not None and not definition.matches(term):
                raise KernelError(
                    f"invalid application or version of effect interface {term.name!r}"
                )
            return
        if isinstance(term, TypeApplication | IndexConstructor):
            for argument in term.arguments:
                self._validate_effect_references(argument, additional)
            return
        if isinstance(term, FunctionType):
            self._validate_effect_references(term.parameter, additional)
            self._validate_effect_references(term.result, additional)
            return
        if isinstance(term, EqualityType):
            self._validate_effect_references(term.left, additional)
            self._validate_effect_references(term.right, additional)
            return
        if isinstance(term, ShapeIndex):
            for dimension in term.dimensions:
                self._validate_effect_references(dimension, additional)

    def validate_static(
        self,
        term: StaticArgument,
        scope: _StaticScope | None = None,
    ) -> None:
        """Validate a static term against every declaration known to the registry."""
        check_static(term, scope, registry=self)
        self._record_type_constructors(term)

    def validate_type(
        self,
        type_: TypeExpr,
        scope: _StaticScope | None = None,
    ) -> None:
        """Validate a value type against every declaration known to the registry."""
        check_type(type_, scope, registry=self)
        self._record_type_constructors(type_)

    def validate_effect_row(self, row: EffectRow) -> None:
        """Validate all concrete interface applications in an effect row."""
        for entry in row.entries:
            self.validate_static(entry.effect)

    def validate_computation_type(self, type_: ComputationType) -> None:
        """Validate both strata of a computation type."""
        self.validate_effect_row(type_.effects)
        self.validate_type(type_.result)

    def register_family(self, family: FamilyDecl) -> None:
        if family.id in self.families:
            raise KernelError(f"family already registered: {family.name!r}")
        _StaticScope().extend(
            (*family.parameters, *family.indices),
            subject=f"family {family.name!r}",
        )
        self._record_type_constructor(family.type_constructor)
        self.families[family.id] = family

    def register_constructor(self, constructor: ConstructorDecl) -> None:
        if constructor.id in self.constructors:
            raise KernelError(f"constructor already registered: {constructor.name!r}")
        family = self.families.get(constructor.family)
        if family is None:
            raise KernelError(f"unknown family for constructor {constructor.name!r}")
        if constructor.id not in family.constructors:
            raise KernelError(
                f"constructor {constructor.name!r} is not declared by family {family.name!r}"
            )
        if len(constructor.result_indices) != len(family.indices):
            raise KernelError(
                f"constructor {constructor.name!r} returns "
                f"{len(constructor.result_indices)} indices; expected {len(family.indices)}"
            )
        family_names = {binder.name for binder in (*family.parameters, *family.indices)}
        constructor_names = {binder.name for binder in constructor.telescope}
        shadowed = family_names & constructor_names
        if shadowed:
            raise KernelError(
                f"constructor {constructor.name!r} shadows family binders: {shadowed!r}"
            )
        scope = (
            _StaticScope()
            .extend(
                family.parameters,
                subject=f"family parameters of {family.name!r}",
            )
            .extend(
                constructor.telescope,
                subject=f"constructor {constructor.name!r}",
            )
        )
        for field_definition in constructor.fields:
            self.validate_type(field_definition.type, scope)
        for result, binder in zip(
            constructor.result_indices,
            family.indices,
            strict=True,
        ):
            self.validate_static(result, scope)
            _check_binder_argument(
                binder,
                result,
                subject=f"result index of constructor {constructor.name!r}",
            )
        self.constructors[constructor.id] = constructor

    def register_effect(self, effect: EffectDef) -> None:
        if effect.ref.id in self.effects:
            raise KernelError(f"effect already registered: {effect.ref.name!r}")
        interface_scope = _StaticScope().extend(
            effect.telescope,
            subject=f"effect {effect.ref.name!r}",
        )
        for operation in effect.operations:
            if operation.id in self.operations:
                raise KernelError(f"operation already registered: {operation.name!r}")
            operation_scope = interface_scope.extend(
                operation.telescope,
                subject=f"operation {operation.name!r}",
            )
            for argument in operation.arguments:
                self.validate_type(argument.type, operation_scope)
                self._validate_effect_references(argument.type, (effect,))
            self.validate_type(operation.result_type, operation_scope)
            self._validate_effect_references(operation.result_type, (effect,))
        self.effects[effect.ref.id] = effect
        for operation in effect.operations:
            self.operations[operation.id] = (effect, operation)

    def register_handler(self, handler: HandlerDef) -> None:
        if handler.id in self.handlers:
            raise KernelError(f"handler already registered: {handler.name!r}")
        scope = _StaticScope().extend(
            handler.telescope,
            subject=f"handler {handler.name!r}",
        )
        self.validate_static(handler.effect, scope)
        self.validate_type(handler.input_type, scope)
        self.validate_type(handler.output_type, scope)
        for entry in handler.introduced.entries:
            self.validate_static(entry.effect, scope)
            introduced = self.effects.get(entry.effect.id)
            if introduced is None or not introduced.matches(entry.effect):
                raise KernelError(
                    f"handler {handler.name!r} introduces an unknown effect interface"
                )
        effect = self.effects.get(handler.effect.id)
        if effect is None or not effect.matches(handler.effect):
            raise KernelError(f"unknown effect interface for handler {handler.name!r}")
        declared = {operation.id for operation in effect.operations}
        clauses = {clause.operation for clause in handler.clauses}
        unknown = clauses - declared
        if unknown:
            raise KernelError(
                f"handler has clauses for unknown operations: {unknown!r}"
            )
        if handler.total and clauses != declared:
            missing = declared - clauses
            raise KernelError(f"total handler is missing operations: {missing!r}")
        if (
            handler.forwards_unknown
            and effect.evolution is not InterfaceEvolution.FORWARDING
        ):
            raise KernelError(
                "only a forwarding interface may forward future operations"
            )
        self.handlers[handler.id] = handler

    def constructor(self, constructor: ConstructorId) -> ConstructorDecl:
        try:
            return self.constructors[constructor]
        except KeyError as exc:
            raise KernelError(f"unknown constructor {constructor}") from exc

    def handler(self, handler: HandlerId) -> HandlerDef:
        try:
            return self.handlers[handler]
        except KeyError as exc:
            raise KernelError(f"unknown handler {handler}") from exc

    def operation(self, operation: OperationId) -> tuple[EffectDef, OperationDef]:
        try:
            return self.operations[operation]
        except KeyError as exc:
            raise KernelError(f"unknown operation {operation}") from exc

    def family_for_type(self, type_: TypeApplication) -> FamilyDecl:
        self.validate_static(type_)
        for family in self.families.values():
            if family.type_constructor == type_.constructor:
                return family
        raise KernelError(f"unknown indexed family type {type_.constructor.name!r}")


@dataclass(frozen=True, slots=True)
class CheckContext:
    locals: tuple[Local, ...] = ()
    givens: tuple[BranchGiven, ...] = ()
    static_scopes: tuple[StaticScopeId, ...] = ()
    static_variables: tuple[StaticVariableId, ...] = ()

    def extend(self, local: Local) -> CheckContext:
        if any(existing.name == local.name for existing in self.locals):
            raise KernelError(f"local already bound: {local.name!r}")
        return CheckContext(
            (*self.locals, local),
            self.givens,
            self.static_scopes,
            self.static_variables,
        )

    def with_givens(self, givens: tuple[BranchGiven, ...]) -> CheckContext:
        return CheckContext(
            self.locals,
            (*self.givens, *givens),
            self.static_scopes,
            self.static_variables,
        )

    def with_static_scope(
        self,
        scope: StaticScopeId,
        variables: tuple[StaticVariableId, ...] = (),
    ) -> CheckContext:
        if scope in self.static_scopes:
            raise KernelError("a case branch reused an enclosing static scope")
        if set(variables) & set(self.static_variables):
            raise KernelError("a case branch reused an active static variable")
        return CheckContext(
            self.locals,
            self.givens,
            (*self.static_scopes, scope),
            (*self.static_variables, *variables),
        )

    def local_type(self, name: str) -> TypeExpr | None:
        return next(
            (local.type for local in reversed(self.locals) if local.name == name),
            None,
        )

    def given(self, equality: EqualityId) -> BranchGiven | None:
        return next((given for given in self.givens if given.id == equality), None)


def _sort_matches(expected: IndexSort, actual: IndexSort) -> bool:
    if isinstance(expected, ShapeSort) and isinstance(actual, ShapeSort):
        return expected.rank is None or expected.rank == actual.rank
    return expected == actual


def check_static(
    term: StaticArgument,
    scope: _StaticScope | None = None,
    *,
    registry: KernelRegistry | None = None,
) -> None:
    """Check the intrinsic kind/sort structure of one static term."""
    if registry is not None:
        registry._validate_effect_references(term)
    if isinstance(term, TypeVariable):
        if scope is not None:
            if term.identity is not None:
                raise KernelError(
                    "a scoped type variable cannot occur in a declaration"
                )
            bound_kind = scope.type_kind(term.name)
            if bound_kind is None:
                raise KernelError(f"unbound type variable {term.name!r}")
            if bound_kind != term.kind:
                raise KernelError(f"type variable {term.name!r} has the wrong kind")
        return
    if isinstance(term, TypeApplication):
        if len(term.constructor.telescope) != len(term.arguments):
            raise KernelError(
                f"type constructor {term.constructor.name!r} expects "
                f"{len(term.constructor.telescope)} static arguments"
            )
        for binder, argument in zip(
            term.constructor.telescope,
            term.arguments,
            strict=True,
        ):
            _check_binder_argument(
                binder,
                argument,
                subject=f"argument of type constructor {term.constructor.name!r}",
            )
            check_static(argument, scope)
        return
    if isinstance(term, FunctionType):
        check_type(term.parameter, scope)
        check_type(term.result, scope)
        return
    if isinstance(term, EqualityType):
        check_static(term.left, scope)
        check_static(term.right, scope)
        if isinstance(term.kind, (NatSort, ShapeSort, ContextSort, UserIndexSort)):
            index_nodes = (IndexVariable, IndexLiteral, IndexConstructor, ShapeIndex)
            if not isinstance(term.left, index_nodes) or not isinstance(
                term.right, index_nodes
            ):
                raise KernelError("index equality endpoints must be index terms")
            if not _sort_matches(term.kind, index_sort(term.left)):  # type: ignore[arg-type]
                raise KernelError("left equality endpoint has the wrong index sort")
            if not _sort_matches(term.kind, index_sort(term.right)):  # type: ignore[arg-type]
                raise KernelError("right equality endpoint has the wrong index sort")
        else:
            if (
                static_kind(term.left) != term.kind
                or static_kind(term.right) != term.kind
            ):
                raise KernelError("equality endpoints have the wrong kind")
        return
    if isinstance(term, EffectRef):
        for argument in term.arguments:
            check_static(argument, scope)
        return
    if isinstance(term, EffectVariable):
        if scope is not None:
            if term.identity is not None:
                raise KernelError(
                    "a scoped effect variable cannot occur in a declaration"
                )
            if term.name not in scope.effects:
                raise KernelError(f"unbound effect variable {term.name!r}")
        return
    if isinstance(term, IndexConstructor):
        for argument in term.arguments:
            check_static(argument, scope)
        return
    if isinstance(term, ShapeIndex):
        index_sort(term)
        for dimension in term.dimensions:
            check_static(dimension, scope)
        return
    if isinstance(term, IndexVariable):
        if scope is not None:
            if term.identity is not None:
                raise KernelError(
                    "a scoped index variable cannot occur in a declaration"
                )
            bound_sort = scope.index_sort(term.name)
            if bound_sort is None:
                raise KernelError(f"unbound index variable {term.name!r}")
            if not _sort_matches(bound_sort, term.sort):
                raise KernelError(f"index variable {term.name!r} has the wrong sort")
        return
    if isinstance(term, IndexLiteral):
        return
    raise KernelError(f"unknown static term {term!r}")


def check_type(
    type_: TypeExpr,
    scope: _StaticScope | None = None,
    *,
    registry: KernelRegistry | None = None,
) -> None:
    """Check that a type expression is intrinsically well-kinded."""
    check_static(type_, scope, registry=registry)
    if static_kind(type_) != TYPE:
        raise KernelError(f"expected a type, got {type_!r}")


def _check_binder_argument(
    binder: TypeBinder | IndexBinder | EffectBinder,
    argument: StaticArgument,
    *,
    subject: str,
) -> None:
    if isinstance(binder, TypeBinder):
        if not isinstance(
            argument,
            (TypeVariable, TypeApplication, FunctionType, EqualityType),
        ):
            raise KernelError(f"{subject} must be a type argument")
        try:
            actual = static_kind(argument)
        except TypeError as error:
            raise KernelError(f"{subject} must be a type argument") from error
        if actual != binder.kind:
            raise KernelError(f"{subject} has the wrong kind")
        return
    if isinstance(binder, IndexBinder):
        if not isinstance(
            argument,
            (IndexVariable, IndexLiteral, IndexConstructor, ShapeIndex),
        ):
            raise KernelError(f"{subject} must be an index argument")
        if not _sort_matches(binder.sort, index_sort(argument)):
            raise KernelError(f"{subject} has the wrong index sort")
        return
    if not isinstance(argument, (EffectRef, EffectVariable)):
        raise KernelError(f"{subject} must be an effect argument")


def check_evidence(evidence: EqualityEvidence, context: CheckContext) -> EqualityType:
    check_type(evidence.equality)
    if isinstance(evidence, Reflexivity):
        if evidence.equality.left != evidence.equality.right:
            raise KernelError("reflexivity evidence has unequal endpoints")
        return evidence.equality
    if isinstance(evidence, BranchGiven):
        available = context.given(evidence.id)
        if available is None or available.equality != evidence.equality:
            raise KernelError("branch equality evidence escaped or was forged")
        return evidence.equality
    raise KernelError(f"unknown equality evidence {evidence!r}")


def _check_literal(value: object, type_: TypeExpr) -> None:
    if type_ == UNIT:
        if value is not None:
            raise KernelError("Unit literals must be None")
        return
    if type_ == BOOL:
        if not isinstance(value, bool):
            raise KernelError("Bool literal has a non-boolean value")
        return
    if type_ == INT:
        if not isinstance(value, int) or isinstance(value, bool):
            raise KernelError("Int literal has a non-integer value")
        return
    if type_ == REAL:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise KernelError("Real literal has a non-numeric value")
        return
    if type_ == STRING:
        if not isinstance(value, str):
            raise KernelError("String literal has a non-string value")
        return
    raise KernelError(
        "literal values are restricted to Unit, Bool, Int, Real, and String"
    )


def _combine_substitutions(
    first: StaticSubstitution,
    second: StaticSubstitution,
) -> StaticSubstitution:
    return StaticSubstitution(
        (*first.types, *second.types),
        (*first.indices, *second.indices),
        (*first.effects, *second.effects),
    )


def _computation_static_scopes(computation: Computation) -> tuple[StaticScopeId, ...]:
    if isinstance(computation, Return | Perform):
        return ()
    if isinstance(computation, Bind):
        return (
            *_computation_static_scopes(computation.first),
            *_computation_static_scopes(computation.then),
        )
    if isinstance(computation, Handle):
        return _computation_static_scopes(computation.computation)
    if isinstance(computation, Case):
        return tuple(
            scope
            for branch in computation.branches
            for scope in (
                branch.scope,
                *_computation_static_scopes(branch.body),
            )
        )
    raise KernelError(f"unknown computation term {computation!r}")


def _static_variable_identities(
    term: StaticArgument,
) -> tuple[StaticVariableId, ...]:
    if isinstance(term, TypeVariable | IndexVariable | EffectVariable):
        return (term.identity,) if term.identity is not None else ()
    if isinstance(term, TypeApplication | EffectRef | IndexConstructor):
        return tuple(
            identity
            for argument in term.arguments
            for identity in _static_variable_identities(argument)
        )
    if isinstance(term, FunctionType):
        return (
            *_static_variable_identities(term.parameter),
            *_static_variable_identities(term.result),
        )
    if isinstance(term, EqualityType):
        return (
            *_static_variable_identities(term.left),
            *_static_variable_identities(term.right),
        )
    if isinstance(term, ShapeIndex):
        return tuple(
            identity
            for dimension in term.dimensions
            for identity in _static_variable_identities(dimension)
        )
    return ()


def _check_static_variable_scope(
    term: StaticArgument,
    context: CheckContext,
    *,
    subject: str,
) -> None:
    escaped = set(_static_variable_identities(term)) - set(context.static_variables)
    if escaped:
        raise KernelError(
            f"{subject} contains a static variable that escapes its case branch"
        )


def _checked_computation_type(
    effects: EffectRow,
    result: TypeExpr,
    registry: KernelRegistry,
    context: CheckContext,
) -> ComputationType:
    type_ = ComputationType(effects, result)
    registry.validate_computation_type(type_)
    _check_static_variable_scope(result, context, subject="computation result type")
    for entry in effects.entries:
        _check_static_variable_scope(
            entry.effect,
            context,
            subject="computation effect row",
        )
    return type_


def infer_value(
    value: Value,
    registry: KernelRegistry,
    context: CheckContext = CheckContext(),
) -> TypeExpr:
    if isinstance(value, Var):
        actual = context.local_type(value.local.name)
        if actual is None or actual != value.local.type:
            raise KernelError(f"unbound or mistyped local {value.local.name!r}")
        registry.validate_type(actual)
        _check_static_variable_scope(actual, context, subject="local type")
        return actual
    if isinstance(value, LiteralValue):
        registry.validate_type(value.type)
        _check_static_variable_scope(value.type, context, subject="literal type")
        _check_literal(value.value, value.type)
        return value.type
    if isinstance(value, AttachmentRef):
        registry.validate_type(value.type)
        _check_static_variable_scope(value.type, context, subject="attachment type")
        return value.type
    if isinstance(value, EvidenceValue):
        equality = check_evidence(value.evidence, context)
        registry.validate_type(equality)
        _check_static_variable_scope(equality, context, subject="equality evidence")
        return equality
    if isinstance(value, TransportValue):
        equality = check_evidence(value.evidence, context)
        registry.validate_type(equality)
        _check_static_variable_scope(equality, context, subject="transport equality")
        source = infer_value(value.value, registry, context)
        registry.validate_type(value.target_type)
        _check_static_variable_scope(
            value.target_type,
            context,
            subject="transport target",
        )
        if equality.left != source or equality.right != value.target_type:
            raise KernelError("transport endpoints do not match value and target types")
        return value.target_type
    if isinstance(value, ConstructorValue):
        constructor = registry.constructor(value.constructor)
        family = registry.families[constructor.family]
        for argument in value.static_arguments:
            registry.validate_static(argument)
            _check_static_variable_scope(
                argument,
                context,
                subject="constructor static argument",
            )
        registry.validate_type(value.result_type)
        _check_static_variable_scope(
            value.result_type,
            context,
            subject="constructor result type",
        )
        parameter_count = len(family.parameters)
        local_count = len(constructor.telescope)
        if len(value.static_arguments) != parameter_count + local_count:
            raise KernelError(
                f"constructor {constructor.name!r} expects "
                f"{parameter_count + local_count} static arguments"
            )
        parameters = instantiate_telescope(
            family.parameters,
            value.static_arguments[:parameter_count],
        )
        locals_ = instantiate_telescope(
            constructor.telescope,
            value.static_arguments[parameter_count:],
        )
        substitution = _combine_substitutions(parameters, locals_)
        if len(value.fields) != len(constructor.fields):
            raise KernelError(
                f"constructor {constructor.name!r} expects {len(constructor.fields)} fields"
            )
        for field_value, field in zip(value.fields, constructor.fields, strict=True):
            actual = infer_value(field_value, registry, context)
            expected = substitute_type(field.type, substitution)
            if actual != expected:
                raise KernelError(
                    f"field {field.name!r} has type {actual!r}, expected {expected!r}"
                )
        parameter_arguments = value.static_arguments[:parameter_count]
        result_indices = tuple(
            substitute_static(index, substitution)
            for index in constructor.result_indices
        )
        expected_result = TypeApplication(
            family.type_constructor,
            (*parameter_arguments, *result_indices),
        )
        if value.result_type != expected_result:
            raise KernelError(
                f"constructor result is {value.result_type!r}, expected {expected_result!r}"
            )
        return expected_result
    raise KernelError(f"unknown value term {value!r}")


def check_request(
    request: EffectRequest,
    registry: KernelRegistry,
    context: CheckContext,
) -> tuple[EffectRef, TypeExpr]:
    registry.validate_static(request.effect)
    _check_static_variable_scope(request.effect, context, subject="request effect")
    for argument in request.static_arguments:
        registry.validate_static(argument)
        _check_static_variable_scope(
            argument,
            context,
            subject="request static argument",
        )
    registry.validate_type(request.result_type)
    _check_static_variable_scope(
        request.result_type,
        context,
        subject="request result type",
    )
    effect, operation_object = registry.operation(request.operation)
    if not effect.matches(request.effect):
        raise KernelError("request effect interface/version does not own the operation")
    interface_substitution = instantiate_telescope(
        effect.telescope,
        request.effect.arguments,
    )
    argument_types, result_type = instantiate_operation(
        operation_object,
        request.static_arguments,
        interface_substitution,
    )
    if len(request.arguments) != len(argument_types):
        raise KernelError(
            f"operation {operation_object.name!r} expects {len(argument_types)} arguments"
        )
    for argument, expected in zip(request.arguments, argument_types, strict=True):
        actual = infer_value(argument, registry, context)
        if actual != expected:
            raise KernelError(
                f"operation argument has type {actual!r}, expected {expected!r}"
            )
    if request.result_type != result_type:
        raise KernelError(
            f"request result is {request.result_type!r}, expected {result_type!r}"
        )
    return request.effect, result_type


def infer_computation(
    computation: Computation,
    registry: KernelRegistry,
    context: CheckContext = CheckContext(),
) -> ComputationType:
    static_scopes = _computation_static_scopes(computation)
    if len(set(static_scopes)) != len(static_scopes):
        raise KernelError("case branch static scopes must be globally fresh")
    if isinstance(computation, Return):
        return _checked_computation_type(
            EMPTY_ROW,
            infer_value(computation.value, registry, context),
            registry,
            context,
        )
    if isinstance(computation, Perform):
        effect, result = check_request(computation.request, registry, context)
        return _checked_computation_type(
            EffectRow((RowEntry(computation.request.instance, effect),)),
            result,
            registry,
            context,
        )
    if isinstance(computation, Bind):
        first = infer_computation(computation.first, registry, context)
        _check_static_variable_scope(
            computation.binder.type,
            context,
            subject="bind type",
        )
        if first.result != computation.binder.type:
            raise KernelError(
                f"bind expects {computation.binder.type!r}, got {first.result!r}"
            )
        then = infer_computation(
            computation.then,
            registry,
            context.extend(computation.binder),
        )
        return _checked_computation_type(
            first.effects.union(then.effects),
            then.result,
            registry,
            context,
        )
    if isinstance(computation, Case):
        scrutinee_type = infer_value(computation.scrutinee, registry, context)
        if not isinstance(scrutinee_type, TypeApplication):
            raise KernelError("case scrutinee is not an indexed family application")
        family = registry.family_for_type(scrutinee_type)
        registry.validate_type(computation.motive.result_type)
        _check_static_variable_scope(
            computation.motive.result_type,
            context,
            subject="case motive",
        )
        if len(computation.motive.indices) != len(family.indices):
            raise KernelError("case motive has the wrong number of index binders")
        for motive_index, family_index in zip(
            computation.motive.indices,
            family.indices,
            strict=True,
        ):
            same_index = (
                (
                    isinstance(motive_index, TypeBinder)
                    and isinstance(family_index, TypeBinder)
                    and motive_index.kind == family_index.kind
                )
                or (
                    isinstance(motive_index, IndexBinder)
                    and isinstance(family_index, IndexBinder)
                    and motive_index.sort == family_index.sort
                )
                or (
                    isinstance(motive_index, EffectBinder)
                    and isinstance(family_index, EffectBinder)
                )
            )
            if not same_index:
                raise KernelError("case motive index has the wrong sort")

        constructors = tuple(
            registry.constructor(constructor) for constructor in family.constructors
        )
        coverage = check_indexed_coverage(
            family,
            constructors,
            scrutinee_type,
            tuple(BranchPattern(branch.constructor) for branch in computation.branches),
        )
        if coverage.status is not CoverageStatus.COMPLETE:
            raise KernelError(
                f"case coverage is {coverage.status.value}; missing={coverage.missing!r}, "
                f"duplicates={coverage.duplicates!r}"
            )

        parameter_count = len(family.parameters)
        actual_parameters = scrutinee_type.arguments[:parameter_count]
        actual_indices = scrutinee_type.arguments[parameter_count:]
        motive_at_scrutinee = instantiate_telescope(
            computation.motive.indices,
            actual_indices,
        )
        result_type = substitute_type(
            computation.motive.result_type,
            motive_at_scrutinee,
        )
        _check_static_variable_scope(
            result_type,
            context,
            subject="case result type",
        )

        branch_scopes = [branch.scope for branch in computation.branches]
        if len(set(branch_scopes)) != len(branch_scopes):
            raise KernelError("case branches must have distinct static scopes")
        branch_rows: list[tuple[Reachability, EffectRow]] = []
        for branch in computation.branches:
            constructor = registry.constructor(branch.constructor)
            static_arguments = constructor_skolems(constructor, branch.scope)
            static_variables = tuple(
                identity
                for argument in static_arguments
                for identity in _static_variable_identities(argument)
            )
            branch_context = context.with_static_scope(
                branch.scope,
                static_variables,
            )
            if branch.static_arguments != static_arguments:
                raise KernelError(
                    f"branch {constructor.name!r} must use its canonical rigid skolems"
                )
            refinement = refine_branch(
                family,
                constructor,
                scrutinee_type,
                static_arguments,
            )
            if refinement.reachability is Reachability.IMPOSSIBLE:
                continue

            parameter_substitution = instantiate_telescope(
                family.parameters,
                actual_parameters,
            )
            local_substitution = instantiate_telescope(
                constructor.telescope,
                static_arguments,
            )
            substitution = _combine_substitutions(
                parameter_substitution,
                local_substitution,
            )
            if len(branch.fields) != len(constructor.fields):
                raise KernelError(
                    f"branch {constructor.name!r} binds the wrong number of fields"
                )
            branch_context = branch_context.with_givens(refinement.givens)
            for local, field in zip(branch.fields, constructor.fields, strict=True):
                expected = substitute_type(field.type, substitution)
                if local.type != expected:
                    raise KernelError(
                        f"branch field {local.name!r} has type {local.type!r}, "
                        f"expected {expected!r}"
                    )
                branch_context = branch_context.extend(local)
            body = infer_computation(branch.body, registry, branch_context)
            for entry in body.effects.entries:
                _check_static_variable_scope(
                    entry.effect,
                    context,
                    subject="case branch effect row",
                )
            constructor_indices = tuple(
                substitute_static(index, substitution)
                for index in constructor.result_indices
            )
            motive_at_constructor = instantiate_telescope(
                computation.motive.indices,
                constructor_indices,
            )
            expected_result = substitute_type(
                computation.motive.result_type,
                motive_at_constructor,
            )
            if body.result != expected_result:
                raise KernelError(
                    f"branch {constructor.name!r} returns {body.result!r}, "
                    f"expected {expected_result!r}"
                )
            branch_rows.append((refinement.reachability, body.effects))
        return _checked_computation_type(
            join_branch_rows(tuple(branch_rows)),
            result_type,
            registry,
            context,
        )
    if isinstance(computation, Handle):
        handler = registry.handler(computation.handler)
        for argument in computation.static_arguments:
            registry.validate_static(argument)
            _check_static_variable_scope(
                argument,
                context,
                subject="handler static argument",
            )
        substitution = instantiate_telescope(
            handler.telescope,
            computation.static_arguments,
        )
        handled_effect = substitute_effect(handler.effect, substitution)
        input_type = substitute_type(handler.input_type, substitution)
        output_type = substitute_type(handler.output_type, substitution)
        introduced = substitute_row(handler.introduced, substitution)
        inner = infer_computation(computation.computation, registry, context)
        if inner.result != input_type:
            raise KernelError(
                f"handler {handler.name!r} expects {input_type!r}, got {inner.result!r}"
            )
        interface = inner.effects.lookup(computation.instance)
        if interface != handled_effect:
            raise KernelError(
                f"handler {handler.name!r} does not match the lexical effect instance"
            )
        residual = (
            inner.effects.without(computation.instance)
            if handler.total
            else inner.effects
        )
        return _checked_computation_type(
            residual.union(introduced),
            output_type,
            registry,
            context,
        )
    raise KernelError(f"unknown computation term {computation!r}")


__all__ = [
    "CheckContext",
    "KernelError",
    "KernelRegistry",
    "check_evidence",
    "check_request",
    "infer_computation",
    "infer_value",
]
