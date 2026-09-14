"""Exact QVR v0.19 to QIEC v1alpha1 lowering.

Quivers owns this adapter. Didactic orders and negotiates its stages and checks
the first-order indexed-family projection. Name resolution, stable identity,
effect elaboration, runtime checking, and the QIEC ABI remain source-language
responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from didactic.extensions import LoweringRoute, lower_checked
from didactic.gadt import (
    GADT,
    Family as DidacticFamily,
    GADTDeclarationError,
    Operation as DidacticOperation,
    SortExpr as DidacticSortExpr,
    Term as DidacticTerm,
    param as didactic_param,
    var as didactic_var,
)
from panproto import GatError

from quivers.dsl import ast_nodes as surface
from quivers.qiec import (
    BOOL,
    EFFECT,
    INT,
    NAT,
    QIEC_ABI,
    REAL,
    STRING,
    TYPE,
    UNIT,
    ArgumentDef,
    Bind,
    Case,
    CaseBranch,
    CaseMotive,
    CheckContext,
    Computation,
    ComputationType,
    computation_type_conforms,
    ConstructorDecl,
    ConstructorId,
    ConstructorValue,
    ContextSort,
    EffectBinder,
    EffectDef,
    EffectId,
    EffectInstanceId,
    EffectRef,
    EffectRequest,
    EffectRow,
    EffectVariable,
    EqualityType,
    FamilyDecl,
    FamilyId,
    FieldDef,
    FunctionType,
    Handle,
    HandlerClauseDef,
    HandlerDef,
    HandlerId,
    IndexBinder,
    IndexConstructor,
    IndexLiteral,
    IndexSort,
    IndexVariable,
    Kind,
    KernelError,
    KernelRegistry,
    LiteralValue,
    Local,
    NamedComputation,
    NamedEffectInstance,
    NatSort,
    OperationDef,
    OperationId,
    Perform,
    QiecModule,
    ResumptionGrade,
    Return,
    RowEntry,
    RowVariable,
    RowVariableId,
    ShapeIndex,
    ShapeSort,
    SiteProvenance,
    SourceOrigin,
    StaticArgument,
    StaticScopeId,
    StaticSubstitution,
    Telescope,
    TelescopeBinder,
    TypeApplication,
    TypeBinder,
    TypeExpr,
    TypeVariable,
    UserIndexSort,
    Var,
    constructor_skolems,
    dumps,
    infer_computation,
    infer_value,
    index_sort,
    instantiate_effect,
    instantiate_operation,
    instantiate_telescope,
    loads,
    product_type,
    substitute_type,
    validate_module,
)


QVR_SOURCE_VERSION = "qvr-source/v0.19"

QIEC_STATEMENT_TYPES = (
    surface.QiecIndexDecl,
    surface.QiecFamilyDecl,
    surface.QiecEffectDecl,
    surface.QiecEffectInstanceDecl,
    surface.QiecHandlerDecl,
    surface.QiecComputationDecl,
)


class QiecDiagnosticError(ValueError):
    """A source-located, stable-code diagnostic from QIEC elaboration."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        file: str,
        line: int = 0,
        column: int = 0,
    ) -> None:
        self.message = message
        self.code = code
        self.file = file
        self.line = line
        self.column = column
        location = f"{file}:{line}:{column}" if line else file
        super().__init__(f"{location}: [{code}] {message}")


@dataclass(frozen=True, slots=True)
class QvrQiecSource:
    """A parsed QVR projection plus the identity absent from the AST root."""

    syntax: surface.Module
    module_name: str
    file_path: str = "<source>"


@dataclass(frozen=True, slots=True)
class CheckedQvrQiec:
    """The checked result passed between Didactic's ordered stages."""

    module: QiecModule


def has_qiec_surface(module: surface.Module) -> bool:
    """Return whether a parsed module contains any v0.19 QIEC declaration."""

    return any(
        isinstance(statement, QIEC_STATEMENT_TYPES) for statement in module.statements
    )


def qiec_projection(module: surface.Module) -> surface.Module:
    """Project QIEC declarations from a module with multiple neighborhoods."""

    return surface.Module(
        statements=tuple(
            statement
            for statement in module.statements
            if isinstance(statement, QIEC_STATEMENT_TYPES)
        )
    )


def non_qiec_projection(module: surface.Module) -> surface.Module:
    """Project declarations consumed by the categorical/probabilistic compiler."""

    return surface.Module(
        statements=tuple(
            statement
            for statement in module.statements
            if not isinstance(statement, QIEC_STATEMENT_TYPES)
        )
    )


class QvrQiecLowerer:
    """Quivers-owned implementation of Didactic's extension-lowering protocol."""

    route = LoweringRoute(QVR_SOURCE_VERSION, QIEC_ABI)

    def check(self, source: QvrQiecSource, /) -> CheckedQvrQiec:
        module = _Elaborator(source).elaborate()
        validate_module(module)
        return CheckedQvrQiec(module)

    def lower(self, checked: CheckedQvrQiec, /) -> QiecModule:
        return checked.module

    def validate(self, target: QiecModule, /) -> None:
        validate_module(target)
        decoded = loads(dumps(target))
        if decoded != target:
            raise KernelError("QIEC module changed during canonical serialization")


def lower_qvr_to_qiec(
    module: surface.Module,
    *,
    module_name: str | None = None,
    file_path: str = "<source>",
    source_version: str = QVR_SOURCE_VERSION,
    target_version: str = QIEC_ABI,
) -> QiecModule:
    """Check and lower the QIEC projection of one parsed QVR module."""

    if module_name is None:
        module_name = _module_name(file_path)
    source = QvrQiecSource(qiec_projection(module), module_name, file_path)
    return lower_checked(
        source,
        QvrQiecLowerer(),
        source_version=source_version,
        target_version=target_version,
    )


def _module_name(file_path: str) -> str:
    if file_path == "<source>":
        return "source"
    return Path(file_path).stem or "source"


def _didactic_name(category: str, identity: object) -> str:
    """Return an injective identifier for Didactic's global GAT namespace."""

    encoded = str(identity).encode().hex()
    return f"qiec_{category}_{encoded}"


class _DidacticGadtProjection:
    """Project QIEC indexed declarations into Didactic's checked GADT API.

    Didactic and Panproto use a first-order GAT signature, while QIEC admits
    static parameters of kind ``Type``.  The projection therefore uses the
    standard Tarski encoding: type parameters inhabit a code sort and values
    of a coded type inhabit an indexed carrier.  QIEC indexed families remain
    genuine Didactic families, so constructor result refinements and dependent
    field telescopes are checked without flattening their indices.
    """

    def __init__(
        self,
        module_name: str,
        index_sorts: tuple[UserIndexSort, ...],
        families: tuple[FamilyDecl, ...],
        constructors: tuple[ConstructorDecl, ...],
    ) -> None:
        self.language = GADT(_didactic_name("module", module_name))
        self.type_codes = self.language.sort(_didactic_name("sort", "Type"))
        self.effect_codes = self.language.sort(_didactic_name("sort", "Effect"))
        self.values = self.language.family(
            _didactic_name("family", "El"),
            parameters=(
                didactic_param(_didactic_name("parameter", "code"), self.type_codes()),
            ),
        )
        self._sorts: dict[IndexSort, DidacticFamily] = {}
        self._families: dict[FamilyId, DidacticFamily] = {}
        self._qiec_families = {family.id: family for family in families}
        self._family_by_type_id = {
            family.type_constructor.id: family for family in families
        }
        self._index_constructors: dict[
            tuple[UserIndexSort, str], DidacticOperation
        ] = {}
        self._code_operations: dict[tuple[str, str], DidacticOperation] = {}

        for sort in index_sorts:
            self._sort_for(sort)
        for family in families:
            self._declare_family(family)
        for constructor in constructors:
            self._declare_constructor(constructor)

    def compile(self) -> object:
        """Compile and check the indexed signature with Panproto."""

        return self.language.compile()

    def _declare_family(self, family: FamilyDecl) -> None:
        parameters = tuple(
            didactic_param(
                _didactic_name("family_parameter", f"{position}:{binder.name}"),
                self._binder_sort(binder),
            )
            for position, binder in enumerate((*family.parameters, *family.indices))
        )
        self._families[family.id] = self.language.family(
            _didactic_name("indexed_family", family.id),
            parameters=parameters,
            closed=family.closed,
        )

    def _declare_constructor(self, constructor: ConstructorDecl) -> None:
        family = self._families[constructor.family]
        qiec_family = self._qiec_families[constructor.family]
        scope: dict[str, DidacticTerm] = {}
        inputs = []
        for position, binder in enumerate(qiec_family.parameters):
            name = _didactic_name("constructor_parameter", f"{position}:{binder.name}")
            inputs.append(didactic_param(name, self._binder_sort(binder)))
            scope[binder.name] = didactic_var(name)
        for position, binder in enumerate(constructor.telescope):
            name = _didactic_name("constructor_static", f"{position}:{binder.name}")
            inputs.append(didactic_param(name, self._binder_sort(binder)))
            scope[binder.name] = didactic_var(name)
        for position, field in enumerate(constructor.fields):
            inputs.append(
                didactic_param(
                    _didactic_name("constructor_field", position),
                    self._value_sort(field.type, scope),
                )
            )
        result_arguments = (
            *(scope[binder.name] for binder in qiec_family.parameters),
            *(self._static_term(index, scope) for index in constructor.result_indices),
        )
        self.language.constructor(
            _didactic_name("constructor", constructor.id),
            inputs=inputs,
            result=family(*result_arguments),
        )

    def _binder_sort(self, binder: TelescopeBinder) -> DidacticSortExpr:
        if isinstance(binder, TypeBinder):
            if binder.kind == TYPE:
                return self.type_codes()
            if binder.kind == EFFECT:
                return self.effect_codes()
            raise GADTDeclarationError(
                f"unsupported QIEC binder kind in GADT projection: {binder.kind!r}"
            )
        if isinstance(binder, EffectBinder):
            return self.effect_codes()
        return self._sort_for(binder.sort)()

    def _sort_for(self, sort: IndexSort) -> DidacticFamily:
        existing = self._sorts.get(sort)
        if existing is not None:
            return existing
        family = self.language.sort(
            _didactic_name("index_sort", sort),
            closed=isinstance(sort, UserIndexSort),
        )
        self._sorts[sort] = family
        if isinstance(sort, UserIndexSort):
            for constructor_name, arity in zip(
                sort.constructors, sort.arities, strict=True
            ):
                operation = self.language.constructor(
                    _didactic_name(
                        "index_constructor", f"{sort.name}:{constructor_name}"
                    ),
                    inputs=tuple(
                        didactic_param(
                            _didactic_name("index_argument", position), family()
                        )
                        for position in range(arity)
                    ),
                    result=family(),
                )
                self._index_constructors[(sort, constructor_name)] = operation
        return family

    def _value_sort(
        self, type_: TypeExpr, scope: Mapping[str, DidacticTerm]
    ) -> DidacticSortExpr:
        if isinstance(type_, TypeApplication):
            family = self._family_by_type_id.get(type_.constructor.id)
            if family is not None:
                target = self._families[family.id]
                return target(
                    *(
                        self._static_term(argument, scope)
                        for argument in type_.arguments
                    )
                )
        return self.values(self._type_term(type_, scope))

    def _type_term(
        self, type_: TypeExpr, scope: Mapping[str, DidacticTerm]
    ) -> DidacticTerm:
        if isinstance(type_, TypeVariable):
            return scope[type_.name]
        if isinstance(type_, TypeApplication):
            operation = self._code_operation(
                "type_constructor",
                type_.constructor.id,
                tuple(
                    self._binder_sort(binder) for binder in type_.constructor.telescope
                ),
                self.type_codes(),
            )
            return operation(
                *(self._static_term(argument, scope) for argument in type_.arguments)
            )
        if isinstance(type_, FunctionType):
            operation = self._code_operation(
                "function_type",
                "function",
                (self.type_codes(), self.type_codes()),
                self.type_codes(),
            )
            return operation(
                self._type_term(type_.parameter, scope),
                self._type_term(type_.result, scope),
            )
        if isinstance(type_, EqualityType):
            operand_sort = self._static_sort(type_.kind)
            operation = self._code_operation(
                "equality_type",
                type_.kind,
                (operand_sort, operand_sort),
                self.type_codes(),
            )
            return operation(
                self._static_term(type_.left, scope),
                self._static_term(type_.right, scope),
            )
        raise GADTDeclarationError(f"unsupported QIEC type projection: {type_!r}")

    def _static_sort(self, kind: Kind | IndexSort) -> DidacticSortExpr:
        if kind == TYPE:
            return self.type_codes()
        if kind == EFFECT:
            return self.effect_codes()
        if isinstance(kind, (NatSort, ShapeSort, ContextSort, UserIndexSort)):
            return self._sort_for(kind)()
        raise GADTDeclarationError(f"unsupported QIEC static sort: {kind!r}")

    def _static_term(
        self, term: StaticArgument, scope: Mapping[str, DidacticTerm]
    ) -> DidacticTerm:
        if isinstance(
            term, (TypeVariable, TypeApplication, FunctionType, EqualityType)
        ):
            return self._type_term(term, scope)
        if isinstance(term, IndexVariable):
            return scope[term.name]
        if isinstance(term, IndexLiteral):
            if isinstance(term.sort, UserIndexSort):
                return self._index_constructors[(term.sort, str(term.value))]()
            operation = self._code_operation(
                "index_literal",
                f"{term.sort}:{term.value}",
                (),
                self._sort_for(term.sort)(),
            )
            return operation()
        if isinstance(term, IndexConstructor):
            constructor_sort = cast(UserIndexSort, term.sort)
            operation = self._index_constructors[(constructor_sort, term.name)]
            return operation(
                *(self._static_term(argument, scope) for argument in term.arguments)
            )
        if isinstance(term, ShapeIndex):
            sort = index_sort(term)
            operation = self._code_operation(
                "shape_index",
                len(term.dimensions),
                tuple(self._sort_for(NAT)() for _ in term.dimensions),
                self._sort_for(sort)(),
            )
            return operation(
                *(self._static_term(dimension, scope) for dimension in term.dimensions)
            )
        if isinstance(term, EffectVariable):
            return scope[term.name]
        if isinstance(term, EffectRef):
            operation = self._code_operation(
                "effect_interface",
                term.id,
                tuple(self._term_sort(argument) for argument in term.arguments),
                self.effect_codes(),
            )
            return operation(
                *(self._static_term(argument, scope) for argument in term.arguments)
            )
        raise GADTDeclarationError(f"unsupported QIEC static projection: {term!r}")

    def _term_sort(self, term: StaticArgument) -> DidacticSortExpr:
        if isinstance(
            term, (TypeVariable, TypeApplication, FunctionType, EqualityType)
        ):
            return self.type_codes()
        if isinstance(term, (EffectVariable, EffectRef)):
            return self.effect_codes()
        return self._sort_for(index_sort(term))()

    def _code_operation(
        self,
        category: str,
        identity: object,
        inputs: tuple[DidacticSortExpr, ...],
        output: DidacticSortExpr,
    ) -> DidacticOperation:
        key = (category, str(identity))
        operation = self._code_operations.get(key)
        if operation is not None:
            return operation
        operation = self.language.operation(
            _didactic_name(category, identity),
            inputs=tuple(
                didactic_param(_didactic_name("code_argument", position), sort)
                for position, sort in enumerate(inputs)
            ),
            result=output,
        )
        self._code_operations[key] = operation
        return operation


class _Elaborator:
    def __init__(self, source: QvrQiecSource) -> None:
        self.source = source
        self.statements = source.syntax.statements
        self.index_sorts: dict[str, UserIndexSort] = {}
        self.families: dict[str, FamilyDecl] = {}
        self.constructors: dict[str, ConstructorDecl] = {}
        self.effects: dict[str, EffectDef] = {}
        self.instances: dict[str, NamedEffectInstance] = {}
        self.handlers: dict[str, HandlerDef] = {}
        self.registry = KernelRegistry()

    def elaborate(self) -> QiecModule:
        self._declare_index_sorts()
        self._declare_family_headers()
        self._declare_effect_headers()
        self._declare_effect_operations()
        self._declare_constructors()
        self._validate_indexed_language()
        self._declare_instances()
        self._declare_handlers()
        self._build_registry()
        computations = self._declare_computations()
        return QiecModule(
            self.source.module_name,
            QVR_SOURCE_VERSION,
            tuple(self.index_sorts.values()),
            tuple(self.families.values()),
            tuple(self.constructors.values()),
            tuple(self.effects.values()),
            tuple(self.instances.values()),
            tuple(self.handlers.values()),
            computations,
        )

    def _validate_indexed_language(self) -> None:
        """Check the QVR GADT projection through Didactic's public language API."""

        if not self.index_sorts and not self.families:
            return
        try:
            _DidacticGadtProjection(
                self.source.module_name,
                tuple(self.index_sorts.values()),
                tuple(self.families.values()),
                tuple(self.constructors.values()),
            ).compile()
        except (GADTDeclarationError, GatError, TypeError, ValueError) as error:
            declarations = self._items(surface.QiecFamilyDecl) or self._items(
                surface.QiecIndexDecl
            )
            self._fail(
                declarations[0],
                f"Didactic rejected the indexed-family declaration: {error}",
                code="qiec-index",
            )

    def _items(self, type_: type[object]) -> tuple[object, ...]:
        return tuple(item for item in self.statements if isinstance(item, type_))

    def _declare_index_sorts(self) -> None:
        declarations = cast(
            tuple[surface.QiecIndexDecl, ...], self._items(surface.QiecIndexDecl)
        )
        for declaration in declarations:
            if declaration.name in self.index_sorts:
                self._fail(declaration, f"duplicate index sort {declaration.name!r}")
            constructor_names: set[str] = set()
            for constructor in declaration.constructors:
                if constructor.name in constructor_names:
                    self._fail(
                        constructor,
                        f"duplicate constructor {constructor.name!r} in index sort "
                        f"{declaration.name!r}",
                        code="qiec-index",
                    )
                constructor_names.add(constructor.name)
            try:
                self.index_sorts[declaration.name] = UserIndexSort(
                    declaration.name,
                    tuple(item.name for item in declaration.constructors),
                    tuple(len(item.arguments) for item in declaration.constructors),
                )
            except (TypeError, ValueError) as error:
                self._fail(declaration, str(error), code="qiec-index")
        for declaration in declarations:
            current = self.index_sorts[declaration.name]
            for constructor in declaration.constructors:
                for argument in constructor.arguments:
                    actual = self._lower_sort(argument)
                    if actual != current:
                        self._fail(
                            constructor,
                            "QIEC v1alpha1 index constructors must be self-recursive; "
                            f"expected {current.name!r}, got {actual!r}",
                            code="qiec-index",
                        )

    def _declare_family_headers(self) -> None:
        declarations = cast(
            tuple[surface.QiecFamilyDecl, ...], self._items(surface.QiecFamilyDecl)
        )
        for declaration in declarations:
            if declaration.name in self.families:
                self._fail(declaration, f"duplicate family {declaration.name!r}")
            if not isinstance(declaration.result_kind, surface.QiecTypeKind):
                self._fail(
                    declaration.result_kind,
                    "indexed families must return Type",
                    code="qiec-kind",
                )
            parameters = self._lower_telescope(declaration.parameters)
            indices = self._lower_telescope(declaration.indices, refinable=True)
            parameter_names = tuple(binder.name for binder in declaration.parameters)
            for binder in declaration.indices:
                if binder.name in parameter_names:
                    self._fail(
                        binder,
                        f"duplicate binder {binder.name!r} in family "
                        f"{declaration.name!r}",
                        code="qiec-kind",
                    )
            family_id = FamilyId.derive(
                self.source.module_name, "family", declaration.name
            )
            constructor_ids = tuple(
                ConstructorId.derive(family_id, "constructor", item.name)
                for item in declaration.constructors
            )
            self.families[declaration.name] = FamilyDecl(
                family_id,
                declaration.name,
                parameters,
                indices,
                constructor_ids,
            )

    def _declare_constructors(self) -> None:
        declarations = cast(
            tuple[surface.QiecFamilyDecl, ...], self._items(surface.QiecFamilyDecl)
        )
        for declaration in declarations:
            family = self.families[declaration.name]
            family_scope = family.parameters
            family_binder_names = {
                binder.name for binder in (*family.parameters, *family.indices)
            }
            for position, authored in enumerate(declaration.constructors):
                if authored.name in self.constructors:
                    self._fail(
                        authored,
                        f"duplicate constructor name {authored.name!r}",
                        code="qiec-index",
                    )
                for binder in authored.binders:
                    if binder.name in family_binder_names:
                        self._fail(
                            binder,
                            f"constructor {authored.name!r} shadows family binder "
                            f"{binder.name!r}",
                            code="qiec-index",
                        )
                telescope = self._lower_telescope(authored.binders)
                scope = (*family_scope, *telescope)
                fields = tuple(
                    FieldDef(f"arg{field_position}", self._lower_type(item, scope))
                    for field_position, item in enumerate(authored.arguments)
                )
                result = self._lower_type(authored.result, scope)
                if not isinstance(result, TypeApplication) or (
                    result.constructor != family.type_constructor
                ):
                    self._fail(
                        authored.result,
                        f"constructor {authored.name!r} must return {family.name!r}",
                        code="qiec-index",
                    )
                parameter_count = len(family.parameters)
                expected_parameters = tuple(
                    self._binder_variable(binder) for binder in family.parameters
                )
                if result.arguments[:parameter_count] != expected_parameters:
                    self._fail(
                        authored.result,
                        "GADT constructor results must preserve family parameters",
                        code="qiec-index",
                    )
                constructor = ConstructorDecl(
                    family.constructors[position],
                    family.id,
                    authored.name,
                    telescope,
                    fields,
                    result.arguments[parameter_count:],
                )
                self.constructors[authored.name] = constructor

    def _declare_effect_headers(self) -> None:
        declarations = cast(
            tuple[surface.QiecEffectDecl, ...], self._items(surface.QiecEffectDecl)
        )
        for declaration in declarations:
            if declaration.name in self.effects:
                self._fail(declaration, f"duplicate effect {declaration.name!r}")
            effect_id = EffectId.derive(
                self.source.module_name,
                "effect",
                declaration.name,
            )
            telescope = self._lower_telescope(declaration.binders)
            ref = EffectRef(effect_id, declaration.name)
            try:
                self.effects[declaration.name] = EffectDef(ref, telescope, ())
            except (TypeError, ValueError) as error:
                self._fail(declaration, str(error), code="qiec-handler")

    def _declare_effect_operations(self) -> None:
        declarations = cast(
            tuple[surface.QiecEffectDecl, ...], self._items(surface.QiecEffectDecl)
        )
        for declaration in declarations:
            header = self.effects[declaration.name]
            effect_id = header.ref.id
            telescope = header.telescope
            lowered_operations: list[OperationDef] = []
            for operation in declaration.operations:
                operation_telescope = self._lower_telescope(operation.binders)
                operation_scope = (*telescope, *operation_telescope)
                authored_arguments = operation.arguments
                # ``Unit -> A`` is the conventional nullary-operation surface.
                if (
                    len(authored_arguments) == 1
                    and isinstance(authored_arguments[0], surface.QiecTypeName)
                    and authored_arguments[0].name == "Unit"
                ):
                    authored_arguments = ()
                lowered_operations.append(
                    OperationDef(
                        OperationId.derive(
                            str(effect_id),
                            operation.name,
                        ),
                        operation.name,
                        operation_telescope,
                        tuple(
                            ArgumentDef(
                                f"arg{position}",
                                self._lower_type(argument, operation_scope),
                            )
                            for position, argument in enumerate(authored_arguments)
                        ),
                        self._lower_type(operation.result, operation_scope),
                    )
                )
            operations = tuple(lowered_operations)
            try:
                self.effects[declaration.name] = EffectDef(
                    header.ref,
                    telescope,
                    operations,
                )
            except (TypeError, ValueError) as error:
                self._fail(declaration, str(error), code="qiec-handler")

    def _declare_instances(self) -> None:
        declarations = cast(
            tuple[surface.QiecEffectInstanceDecl, ...],
            self._items(surface.QiecEffectInstanceDecl),
        )
        for declaration in declarations:
            if declaration.name in self.instances:
                self._fail(declaration, f"duplicate instance {declaration.name!r}")
            effect = self._lower_effect_ref(declaration.effect, ())
            entry = instantiate_effect(
                effect,
                module=self.source.module_name,
                lexical_path=("instances", declaration.name),
            )
            self.instances[declaration.name] = NamedEffectInstance(
                declaration.name,
                entry,
                self._origin(
                    declaration,
                    ("instances", declaration.name),
                    "effect-instance",
                ),
            )

    def _declare_handlers(self) -> None:
        declarations = cast(
            tuple[surface.QiecHandlerDecl, ...], self._items(surface.QiecHandlerDecl)
        )
        for declaration in declarations:
            if declaration.name in self.handlers:
                self._fail(declaration, f"duplicate handler {declaration.name!r}")
            if declaration.duplicate_options:
                self._fail(
                    declaration,
                    "duplicate handler option(s): "
                    f"{', '.join(declaration.duplicate_options)}",
                    code="qiec-handler",
                )
            if declaration.coverage == "total" and declaration.forwards_unknown:
                self._fail(
                    declaration,
                    "a handler cannot be both total and forwarding",
                    code="qiec-handler",
                )
            telescope = self._lower_telescope(declaration.binders)
            effect = self._lower_effect_ref(declaration.effect, telescope)
            definition = self.effects[declaration.effect.name]
            operations = {
                operation.name: operation for operation in definition.operations
            }
            clauses: list[HandlerClauseDef] = []
            seen_clauses: set[str] = set()
            for clause in declaration.clauses:
                if clause.operation in seen_clauses:
                    self._fail(
                        clause,
                        f"duplicate clause {clause.operation!r} in handler "
                        f"{declaration.name!r}",
                        code="qiec-handler",
                    )
                seen_clauses.add(clause.operation)
                operation = operations.get(clause.operation)
                if operation is None:
                    self._fail(
                        clause,
                        f"effect {effect.name!r} has no operation {clause.operation!r}",
                        code="qiec-handler",
                    )
                clauses.append(
                    HandlerClauseDef(operation.id, ResumptionGrade(clause.grade))
                )
            input_type = self._lower_type(declaration.input_type, telescope)
            output_type = self._lower_type(declaration.output_type, telescope)
            introduced = self._lower_row(
                declaration.introduced,
                ("handlers", declaration.name, "introduced"),
            )
            try:
                handler = HandlerDef(
                    HandlerId.derive(
                        self.source.module_name, "handler", declaration.name
                    ),
                    declaration.name,
                    effect,
                    tuple(clauses),
                    input_type,
                    output_type,
                    introduced,
                    total=declaration.coverage == "total",
                    forwards_unknown=declaration.forwards_unknown,
                    telescope=telescope,
                )
            except (TypeError, ValueError) as error:
                self._fail(declaration, str(error), code="qiec-handler")
            self.handlers[declaration.name] = handler

    def _build_registry(self) -> None:
        family_nodes = {
            item.name: item
            for item in cast(
                tuple[surface.QiecFamilyDecl, ...],
                self._items(surface.QiecFamilyDecl),
            )
        }
        effect_nodes = {
            item.name: item
            for item in cast(
                tuple[surface.QiecEffectDecl, ...],
                self._items(surface.QiecEffectDecl),
            )
        }
        handler_nodes = {
            item.name: item
            for item in cast(
                tuple[surface.QiecHandlerDecl, ...],
                self._items(surface.QiecHandlerDecl),
            )
        }
        for family in self.families.values():
            try:
                self.registry.register_family(family)
            except (KernelError, TypeError, ValueError) as error:
                self._fail(family_nodes[family.name], str(error), code="qiec-index")
        for constructor in self.constructors.values():
            try:
                self.registry.register_constructor(constructor)
            except (KernelError, TypeError, ValueError) as error:
                self._fail(
                    family_nodes[self.registry.families[constructor.family].name],
                    str(error),
                    code="qiec-index",
                )
        for effect in self.effects.values():
            try:
                self.registry.register_effect(effect)
            except (KernelError, TypeError, ValueError) as error:
                self._fail(effect_nodes[effect.ref.name], str(error), code="qiec-kind")
        for handler in self.handlers.values():
            try:
                self.registry.register_handler(handler)
            except (KernelError, TypeError, ValueError) as error:
                self._fail(handler_nodes[handler.name], str(error), code="qiec-handler")

    def _declare_computations(self) -> tuple[NamedComputation, ...]:
        declarations = cast(
            tuple[surface.QiecComputationDecl, ...],
            self._items(surface.QiecComputationDecl),
        )
        seen: set[str] = set()
        computations: list[NamedComputation] = []
        for declaration in declarations:
            if declaration.name in seen:
                self._fail(declaration, f"duplicate computation {declaration.name!r}")
            seen.add(declaration.name)
            telescope = self._lower_telescope(declaration.binders)
            parameter_names: set[str] = set()
            static_names = {binder.name for binder in telescope}
            for parameter in declaration.parameters:
                if parameter.name in parameter_names:
                    self._fail(
                        parameter,
                        f"duplicate parameter {parameter.name!r} in computation "
                        f"{declaration.name!r}",
                    )
                if parameter.name in static_names:
                    self._fail(
                        parameter,
                        f"computation {declaration.name!r} reuses static name "
                        f"{parameter.name!r} as a parameter",
                    )
                parameter_names.add(parameter.name)
            parameters = tuple(
                Local(parameter.name, self._lower_type(parameter.type_expr, telescope))
                for parameter in declaration.parameters
            )
            context = CheckContext(parameters)
            path = ("computations", declaration.name, "body")
            body = self._lower_computation(declaration.body, telescope, context, path)
            computation = NamedComputation(
                declaration.name,
                telescope,
                parameters,
                body,
                ComputationType(
                    self._lower_row(
                        declaration.effects,
                        ("computations", declaration.name, "effects"),
                    ),
                    self._lower_type(declaration.result_type, telescope),
                ),
                self._origin(
                    declaration,
                    ("computations", declaration.name),
                    "computation",
                ),
            )
            try:
                actual = infer_computation(body, self.registry, context)
            except (KernelError, TypeError, ValueError) as error:
                self._fail(declaration.body, str(error), code="qiec-kind")
            if actual.result != computation.type.result:
                self._fail(
                    declaration,
                    f"computation body returns {actual.result!r}, "
                    f"expected {computation.type.result!r}",
                    code="qiec-kind",
                )
            if not computation_type_conforms(actual, computation.type):
                self._fail(
                    declaration,
                    f"computation body has effect row {actual.effects!r}, "
                    f"not declared row {computation.type.effects!r}",
                    code="qiec-row",
                )
            computations.append(computation)
        return tuple(computations)

    def _lower_telescope(
        self,
        binders: tuple[surface.QiecBinder, ...],
        *,
        refinable: bool = False,
    ) -> Telescope:
        lowered: list[TelescopeBinder] = []
        seen: set[str] = set()
        for binder in binders:
            name = cast(str, binder.name)
            if name in seen:
                self._fail(
                    binder,
                    f"duplicate telescope binder {name!r}",
                    code="qiec-kind",
                )
            seen.add(name)
            if isinstance(binder, surface.QiecTypeBinder):
                kind = (
                    TYPE
                    if isinstance(binder.binder_kind, surface.QiecTypeKind)
                    else EFFECT
                )
                lowered.append(TypeBinder(name, kind, refinable))
            elif isinstance(binder, surface.QiecIndexBinder):
                lowered.append(
                    IndexBinder(name, self._lower_sort(binder.sort), refinable)
                )
            elif isinstance(binder, surface.QiecEffectBinder):
                lowered.append(EffectBinder(name, refinable))
            else:  # pragma: no cover - closed source union
                self._fail(binder, "unknown telescope binder", code="qiec-kind")
        return tuple(lowered)

    def _lower_sort(self, sort: surface.QiecIndexSort) -> IndexSort:
        if isinstance(sort, surface.QiecNatSort):
            return self.index_sorts.get("Nat", NAT)
        if isinstance(sort, surface.QiecShapeSort):
            return ShapeSort(sort.rank)
        if isinstance(sort, surface.QiecContextSort):
            return ContextSort(sort.signature)
        if isinstance(sort, surface.QiecUserIndexSort):
            resolved = self.index_sorts.get(sort.name)
            if resolved is None:
                self._fail(sort, f"unknown index sort {sort.name!r}", code="qiec-index")
            return resolved
        self._fail(sort, "unknown index sort", code="qiec-index")

    def _lower_type(
        self,
        authored: surface.QiecTypeExpr,
        scope: Telescope,
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> TypeExpr:
        if isinstance(authored, surface.QiecTypeName):
            bound = (static_bindings or {}).get(authored.name)
            if bound is not None:
                if isinstance(bound, TypeVariable):
                    return bound
                self._fail(
                    authored,
                    f"static binder {authored.name!r} is not a type",
                    code="qiec-kind",
                )
            binder = next(
                (
                    item
                    for item in reversed(scope)
                    if isinstance(item, TypeBinder) and item.name == authored.name
                ),
                None,
            )
            if binder is not None:
                return TypeVariable(binder.name, binder.kind)
            primitive = {
                "Unit": UNIT,
                "Bool": BOOL,
                "Int": INT,
                "Real": REAL,
                "String": STRING,
            }.get(authored.name)
            if primitive is not None:
                return primitive
            family = self.families.get(authored.name)
            if family is not None and not (*family.parameters, *family.indices):
                return TypeApplication(family.type_constructor)
            self._fail(authored, f"unknown or unsaturated type {authored.name!r}")
        if isinstance(authored, surface.QiecTypeApplication):
            family = self.families.get(authored.constructor)
            if family is None:
                self._fail(authored, f"unknown indexed family {authored.constructor!r}")
            if len(authored.static_arguments) != len(family.parameters):
                self._fail(
                    authored,
                    f"family {family.name!r} expects {len(family.parameters)} "
                    "static parameters",
                    code="qiec-kind",
                )
            if len(authored.indices) != len(family.indices):
                self._fail(
                    authored,
                    f"family {family.name!r} expects {len(family.indices)} indices",
                    code="qiec-index",
                )
            parameters = tuple(
                self._lower_static_argument(argument, binder, scope, static_bindings)
                for argument, binder in zip(
                    authored.static_arguments, family.parameters, strict=True
                )
            )
            indices = tuple(
                self._lower_index(argument, binder.sort, scope, static_bindings)
                for argument, binder in zip(
                    authored.indices, family.indices, strict=True
                )
                if isinstance(binder, IndexBinder)
            )
            if len(indices) != len(family.indices):
                self._fail(
                    authored,
                    "v0.19 family indices must use index binders",
                    code="qiec-index",
                )
            return TypeApplication(family.type_constructor, (*parameters, *indices))
        if isinstance(authored, surface.QiecProductType):
            return product_type(
                *(
                    self._lower_type(item, scope, static_bindings)
                    for item in authored.components
                )
            )
        if isinstance(authored, surface.QiecFunctionType):
            return FunctionType(
                self._lower_type(authored.parameter, scope, static_bindings),
                self._lower_type(authored.result, scope, static_bindings),
            )
        self._fail(authored, "unknown type expression", code="qiec-kind")

    def _lower_static_argument(
        self,
        authored: surface.QiecTypeExpr,
        binder: TelescopeBinder,
        scope: Telescope,
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> StaticArgument:
        if isinstance(binder, TypeBinder):
            return self._lower_type(authored, scope, static_bindings)
        if isinstance(binder, IndexBinder):
            return self._lower_index(
                self._type_syntax_as_index(authored),
                binder.sort,
                scope,
                static_bindings,
            )
        if isinstance(authored, surface.QiecTypeApplication):
            if authored.indices:
                self._fail(
                    authored,
                    "effect applications cannot carry family indices",
                    code="qiec-kind",
                )
            return self._lower_effect_ref(
                surface.QiecEffectRef(
                    name=authored.constructor,
                    arguments=authored.static_arguments,
                    line=authored.line,
                    col=authored.col,
                ),
                scope,
                static_bindings,
            )
        if isinstance(authored, surface.QiecTypeName):
            bound = (static_bindings or {}).get(authored.name)
            if bound is not None:
                if isinstance(bound, EffectVariable):
                    return bound
                self._fail(
                    authored,
                    f"static binder {authored.name!r} is not an effect",
                    code="qiec-kind",
                )
            if any(
                isinstance(item, EffectBinder) and item.name == authored.name
                for item in scope
            ):
                return EffectVariable(authored.name)
            definition = self.effects.get(authored.name)
            if definition is not None and not definition.telescope:
                return definition.ref
        self._fail(
            authored,
            f"argument for effect binder {binder.name!r} is not an effect",
            code="qiec-kind",
        )

    def _type_syntax_as_index(
        self, authored: surface.QiecTypeExpr
    ) -> surface.QiecIndexExpr:
        if isinstance(authored, surface.QiecTypeName):
            return surface.QiecIndexName(
                name=authored.name, line=authored.line, col=authored.col
            )
        if isinstance(authored, surface.QiecTypeApplication):
            arguments = tuple(
                self._type_syntax_as_index(argument)
                for argument in authored.static_arguments
            )
            return surface.QiecIndexApplication(
                constructor=authored.constructor,
                arguments=(*arguments, *authored.indices),
                line=authored.line,
                col=authored.col,
            )
        self._fail(
            authored,
            "this static argument is not valid index syntax",
            code="qiec-index",
        )

    def _lower_index(
        self,
        authored: surface.QiecIndexExpr,
        expected: IndexSort,
        scope: Telescope,
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> IndexVariable | IndexLiteral | IndexConstructor | ShapeIndex:
        if isinstance(authored, surface.QiecIndexName):
            bound = (static_bindings or {}).get(authored.name)
            if bound is not None:
                if isinstance(bound, IndexVariable):
                    if bound.sort != expected:
                        self._fail(
                            authored,
                            f"index {authored.name!r} has the wrong sort",
                            code="qiec-index",
                        )
                    return bound
                self._fail(
                    authored,
                    f"static binder {authored.name!r} is not an index",
                    code="qiec-index",
                )
            binder = next(
                (
                    item
                    for item in reversed(scope)
                    if isinstance(item, IndexBinder) and item.name == authored.name
                ),
                None,
            )
            if binder is not None:
                if binder.sort != expected:
                    self._fail(
                        authored,
                        f"index {authored.name!r} has the wrong sort",
                        code="qiec-index",
                    )
                return IndexVariable(authored.name, expected)
            if isinstance(expected, UserIndexSort):
                try:
                    if expected.constructor_arity(authored.name) == 0:
                        return IndexLiteral(authored.name, expected)
                except ValueError:
                    pass
            self._fail(authored, f"unknown index {authored.name!r}", code="qiec-index")
        if isinstance(authored, surface.QiecIndexLiteral):
            try:
                return IndexLiteral(authored.value, expected)
            except (TypeError, ValueError) as error:
                self._fail(authored, str(error), code="qiec-index")
        if isinstance(authored, surface.QiecIndexApplication):
            if not isinstance(expected, UserIndexSort):
                self._fail(
                    authored,
                    "named index constructors require a user-defined index sort",
                    code="qiec-index",
                )
            arguments = tuple(
                self._lower_index(argument, expected, scope, static_bindings)
                for argument in authored.arguments
            )
            try:
                return IndexConstructor(
                    authored.constructor,
                    arguments,
                    expected,
                )
            except (TypeError, ValueError) as error:
                self._fail(authored, str(error), code="qiec-index")
        if isinstance(authored, surface.QiecShapeIndex):
            return ShapeIndex(
                tuple(
                    self._lower_index(item, NAT, scope, static_bindings)
                    for item in authored.dimensions
                )
            )
        self._fail(authored, "unknown index expression", code="qiec-index")

    def _lower_effect_ref(
        self,
        authored: surface.QiecEffectRef,
        scope: Telescope,
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> EffectRef:
        definition = self.effects.get(authored.name)
        if definition is None:
            self._fail(authored, f"unknown effect interface {authored.name!r}")
        if len(authored.arguments) != len(definition.telescope):
            self._fail(
                authored,
                f"effect {authored.name!r} expects {len(definition.telescope)} arguments",
                code="qiec-kind",
            )
        arguments = tuple(
            self._lower_static_argument(argument, binder, scope, static_bindings)
            for argument, binder in zip(
                authored.arguments, definition.telescope, strict=True
            )
        )
        try:
            return definition.apply(arguments)
        except (KernelError, TypeError, ValueError) as error:
            self._fail(authored, str(error), code="qiec-kind")

    def _lower_row(
        self,
        authored: surface.QiecEffectRow,
        path: tuple[str | int, ...],
    ) -> EffectRow:
        entries: list[RowEntry] = []
        for item in authored.entries:
            instance = self.instances.get(item.instance)
            if instance is None:
                self._fail(
                    item,
                    f"unknown effect instance {item.instance!r}",
                    code="qiec-row",
                )
            entries.append(instance.entry)
        tail: RowVariable | None = None
        if authored.tail is not None:
            if len(set(authored.lacks)) != len(authored.lacks):
                self._fail(
                    authored,
                    "a row lacks constraint cannot repeat an instance",
                    code="qiec-row",
                )
            lacks_names = set(authored.lacks)
            explicit_names = {item.instance for item in authored.entries}
            missing = explicit_names - lacks_names
            if missing:
                self._fail(
                    authored,
                    "an open row tail must explicitly lack its entries: "
                    f"{sorted(missing)!r}",
                    code="qiec-row",
                )
            lacks: list[EffectInstanceId] = []
            for name in authored.lacks:
                instance = self.instances.get(name)
                if instance is None:
                    self._fail(
                        authored,
                        f"row lacks unknown instance {name!r}",
                        code="qiec-row",
                    )
                lacks.append(instance.entry.instance)
            try:
                tail = RowVariable(
                    authored.tail,
                    RowVariableId.derive(self.source.module_name, path, authored.tail),
                    tuple(lacks),
                )
            except ValueError as error:
                self._fail(authored, str(error), code="qiec-row")
        try:
            return EffectRow(tuple(entries), tail)
        except ValueError as error:
            self._fail(authored, str(error), code="qiec-row")

    def _lower_value(
        self,
        authored: surface.QiecValue,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ):
        if isinstance(authored, surface.QiecVariableValue):
            local = next(
                (
                    item
                    for item in reversed(context.locals)
                    if item.name == authored.name
                ),
                None,
            )
            if local is None:
                self._fail(authored, f"unbound local {authored.name!r}")
            return Var(local)
        if isinstance(authored, surface.QiecLiteralValue):
            value = authored.value
            if value is None:
                type_ = UNIT
            elif isinstance(value, bool):
                type_ = BOOL
            elif isinstance(value, int):
                type_ = INT
            elif isinstance(value, float):
                type_ = REAL
            else:
                type_ = STRING
            return LiteralValue(value, type_)
        if isinstance(authored, surface.QiecConstructorValue):
            constructor = self.constructors.get(authored.constructor)
            if constructor is None:
                self._fail(
                    authored,
                    f"unknown constructor {authored.constructor!r}",
                    code="qiec-index",
                )
            family = self.registry.families[constructor.family]
            binders = (*family.parameters, *constructor.telescope)
            if len(authored.static_arguments) != len(binders):
                self._fail(
                    authored,
                    f"constructor {constructor.name!r} expects {len(binders)} "
                    "static arguments",
                    code="qiec-kind",
                )
            static_arguments = tuple(
                self._lower_static_argument(argument, binder, scope, static_bindings)
                for argument, binder in zip(
                    authored.static_arguments, binders, strict=True
                )
            )
            return ConstructorValue(
                constructor.id,
                static_arguments,
                tuple(
                    self._lower_value(value, scope, context, static_bindings)
                    for value in authored.fields
                ),
                self._lower_type(authored.result_type, scope, static_bindings),
            )
        self._fail(authored, "unknown QIEC value")

    def _lower_computation(
        self,
        authored: surface.QiecComputation,
        scope: Telescope,
        context: CheckContext,
        path: tuple[str | int, ...],
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> Computation:
        if isinstance(authored, surface.QiecReturnComputation):
            return Return(
                self._lower_value(authored.value, scope, context, static_bindings)
            )
        if isinstance(authored, surface.QiecBindComputation):
            first = self._lower_computation(
                authored.first,
                scope,
                context,
                (*path, "first"),
                static_bindings,
            )
            try:
                inferred = infer_computation(first, self.registry, context)
            except (KernelError, TypeError, ValueError) as error:
                self._fail(authored.first, str(error))
            binder_type = (
                inferred.result
                if authored.binder.type_expr is None
                else self._lower_type(authored.binder.type_expr, scope, static_bindings)
            )
            binder = Local(authored.binder.name, binder_type)
            then_context = context.extend(binder)
            return Bind(
                binder,
                first,
                self._lower_computation(
                    authored.then,
                    scope,
                    then_context,
                    (*path, "then"),
                    static_bindings,
                ),
            )
        if isinstance(authored, surface.QiecSequenceComputation):
            first = self._lower_computation(
                authored.first,
                scope,
                context,
                (*path, "first"),
                static_bindings,
            )
            try:
                inferred = infer_computation(first, self.registry, context)
            except (KernelError, TypeError, ValueError) as error:
                self._fail(authored.first, str(error))
            name = self._fresh_sequence_name(context, path)
            binder = Local(name, inferred.result)
            return Bind(
                binder,
                first,
                self._lower_computation(
                    authored.then,
                    scope,
                    context.extend(binder),
                    (*path, "then"),
                    static_bindings,
                ),
            )
        if isinstance(authored, surface.QiecPerformComputation):
            request = authored.request
            instance = self.instances.get(request.instance)
            if instance is None:
                self._fail(
                    request,
                    f"unknown effect instance {request.instance!r}",
                    code="qiec-unhandled-effect",
                )
            definition = self.effects[instance.entry.effect.name]
            operation = next(
                (
                    item
                    for item in definition.operations
                    if item.name == request.operation
                ),
                None,
            )
            if operation is None:
                self._fail(
                    request,
                    f"effect {definition.ref.name!r} has no operation "
                    f"{request.operation!r}",
                    code="qiec-handler",
                )
            if len(request.static_arguments) != len(operation.telescope):
                self._fail(
                    request,
                    f"operation {operation.name!r} expects "
                    f"{len(operation.telescope)} static arguments",
                    code="qiec-kind",
                )
            static_arguments = tuple(
                self._lower_static_argument(argument, binder, scope, static_bindings)
                for argument, binder in zip(
                    request.static_arguments, operation.telescope, strict=True
                )
            )
            outer = instantiate_telescope(
                definition.telescope, instance.entry.effect.arguments
            )
            _, result_type = instantiate_operation(operation, static_arguments, outer)
            return Perform(
                EffectRequest(
                    instance.entry.instance,
                    instance.entry.effect,
                    operation.id,
                    static_arguments,
                    tuple(
                        self._lower_value(value, scope, context, static_bindings)
                        for value in request.arguments
                    ),
                    result_type,
                    SiteProvenance(self._origin(request, path, "effect-request")),
                )
            )
        if isinstance(authored, surface.QiecHandleComputation):
            instance = self.instances.get(authored.instance)
            if instance is None:
                self._fail(
                    authored,
                    f"unknown effect instance {authored.instance!r}",
                    code="qiec-handler",
                )
            handler = self.handlers.get(authored.handler.name)
            if handler is None:
                self._fail(
                    authored.handler,
                    f"unknown handler {authored.handler.name!r}",
                    code="qiec-handler",
                )
            if len(authored.handler.static_arguments) != len(handler.telescope):
                self._fail(
                    authored.handler,
                    f"handler {handler.name!r} expects {len(handler.telescope)} "
                    "static arguments",
                    code="qiec-kind",
                )
            static_arguments = tuple(
                self._lower_static_argument(argument, binder, scope, static_bindings)
                for argument, binder in zip(
                    authored.handler.static_arguments, handler.telescope, strict=True
                )
            )
            return Handle(
                instance.entry.instance,
                handler.id,
                self._lower_computation(
                    authored.body,
                    scope,
                    context,
                    (*path, "handled"),
                    static_bindings,
                ),
                static_arguments,
            )
        if isinstance(authored, surface.QiecCaseComputation):
            return self._lower_case(authored, scope, context, path, static_bindings)
        self._fail(authored, "unknown QIEC computation")

    def _lower_case(
        self,
        authored: surface.QiecCaseComputation,
        scope: Telescope,
        context: CheckContext,
        path: tuple[str | int, ...],
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> Case:
        scrutinee = self._lower_value(
            authored.scrutinee, scope, context, static_bindings
        )
        try:
            scrutinee_type = infer_value(scrutinee, self.registry, context)
        except (KernelError, TypeError, ValueError) as error:
            self._fail(authored.scrutinee, str(error), code="qiec-index")
        if not isinstance(scrutinee_type, TypeApplication):
            self._fail(
                authored.scrutinee,
                "case scrutinee is not an indexed-family value",
                code="qiec-index",
            )
        try:
            family = self.registry.family_for_type(scrutinee_type)
        except KernelError as error:
            self._fail(authored.scrutinee, str(error), code="qiec-index")
        motive_indices = self._lower_telescope(authored.motive.indices, refinable=True)
        motive_bindings = dict(static_bindings or {})
        for binder in motive_indices:
            motive_bindings.pop(binder.name, None)
        motive = CaseMotive(
            motive_indices,
            self._lower_type(
                authored.motive.result_type,
                (*scope, *motive_indices),
                motive_bindings,
            ),
        )
        parameter_count = len(family.parameters)
        actual_parameters = scrutinee_type.arguments[:parameter_count]
        parameter_substitution = instantiate_telescope(
            family.parameters, actual_parameters
        )
        branches: list[CaseBranch] = []
        for position, authored_branch in enumerate(authored.branches):
            constructor = self.constructors.get(authored_branch.constructor)
            if constructor is None or constructor.family != family.id:
                self._fail(
                    authored_branch,
                    f"constructor {authored_branch.constructor!r} does not belong "
                    f"to {family.name!r}",
                    code="qiec-coverage",
                )
            if len(authored_branch.static_arguments) != len(constructor.telescope):
                self._fail(
                    authored_branch,
                    f"branch {constructor.name!r} expects "
                    f"{len(constructor.telescope)} static binders",
                    code="qiec-index",
                )
            branch_scope = StaticScopeId.derive(
                self.source.module_name, path, "branch", position, constructor.id
            )
            skolems = constructor_skolems(constructor, branch_scope)
            authored_binders: list[TelescopeBinder] = []
            branch_bindings = dict(static_bindings or {})
            branch_names: set[str] = set()
            for authored_binder, declared_binder, skolem in zip(
                authored_branch.static_arguments,
                constructor.telescope,
                skolems,
                strict=True,
            ):
                if not isinstance(authored_binder, surface.QiecTypeName):
                    self._fail(
                        authored_binder,
                        "case branch static binders must be plain identifiers",
                        code="qiec-index",
                    )
                if authored_binder.name in branch_names:
                    self._fail(
                        authored_binder,
                        f"duplicate case branch static binder {authored_binder.name!r}",
                        code="qiec-index",
                    )
                branch_names.add(authored_binder.name)
                authored_binders.append(
                    self._rename_binder(declared_binder, authored_binder.name)
                )
                branch_bindings[authored_binder.name] = skolem
            local_substitution = instantiate_telescope(constructor.telescope, skolems)
            substitution = StaticSubstitution(
                (*parameter_substitution.types, *local_substitution.types),
                (*parameter_substitution.indices, *local_substitution.indices),
                (*parameter_substitution.effects, *local_substitution.effects),
            )
            if len(authored_branch.fields) != len(constructor.fields):
                self._fail(
                    authored_branch,
                    f"branch {constructor.name!r} binds "
                    f"{len(authored_branch.fields)} fields; expected "
                    f"{len(constructor.fields)}",
                    code="qiec-coverage",
                )
            branch_context = context.with_static_scope(
                branch_scope,
                tuple(
                    argument.identity
                    for argument in skolems
                    if isinstance(
                        argument, TypeVariable | IndexVariable | EffectVariable
                    )
                    and argument.identity is not None
                ),
            )
            fields: list[Local] = []
            seen_fields: set[str] = set()
            for authored_field, definition in zip(
                authored_branch.fields, constructor.fields, strict=True
            ):
                if authored_field.name in seen_fields:
                    self._fail(
                        authored_field,
                        f"duplicate case branch field {authored_field.name!r}",
                        code="qiec-kind",
                    )
                seen_fields.add(authored_field.name)
                expected = substitute_type(definition.type, substitution)
                if authored_field.type_expr is not None:
                    annotated = self._lower_type(
                        authored_field.type_expr,
                        (*scope, *authored_binders),
                        branch_bindings,
                    )
                    if annotated != expected:
                        self._fail(
                            authored_field,
                            f"branch field has type {annotated!r}, expected {expected!r}",
                            code="qiec-index",
                        )
                local = Local(authored_field.name, expected)
                fields.append(local)
                branch_context = branch_context.extend(local)
            branches.append(
                CaseBranch(
                    constructor.id,
                    skolems,
                    tuple(fields),
                    self._lower_computation(
                        authored_branch.body,
                        (*scope, *authored_binders),
                        branch_context,
                        (*path, "branch", position, "body"),
                        branch_bindings,
                    ),
                    branch_scope,
                )
            )
        return Case(scrutinee, motive, tuple(branches))

    def _rename_binder(
        self,
        binder: TelescopeBinder,
        name: str,
    ) -> TelescopeBinder:
        """Give an authored branch name the constructor binder's exact kind."""

        if isinstance(binder, TypeBinder):
            return TypeBinder(name, binder.kind, binder.refinable)
        if isinstance(binder, IndexBinder):
            return IndexBinder(name, binder.sort, binder.refinable)
        return EffectBinder(name, binder.refinable)

    def _binder_variable(self, binder: TelescopeBinder) -> StaticArgument:
        if isinstance(binder, TypeBinder):
            return TypeVariable(binder.name, binder.kind)
        if isinstance(binder, IndexBinder):
            return IndexVariable(binder.name, binder.sort)
        return EffectVariable(binder.name)

    def _fresh_sequence_name(
        self, context: CheckContext, path: tuple[str | int, ...]
    ) -> str:
        stem = "__qiec_sequence_" + "_".join(str(item) for item in path[-3:])
        name = stem
        suffix = 0
        existing = {local.name for local in context.locals}
        while name in existing:
            suffix += 1
            name = f"{stem}_{suffix}"
        return name

    def _origin(
        self,
        node: object,
        path: tuple[str | int, ...],
        role: str,
    ) -> SourceOrigin:
        return SourceOrigin(
            self.source.module_name,
            path,
            role,
            QVR_SOURCE_VERSION,
            self.source.file_path,
            getattr(node, "line", None),
            getattr(node, "col", None),
        )

    def _fail(
        self,
        node: object,
        message: str,
        *,
        code: str = "qiec-kind",
    ) -> NoReturn:
        raise QiecDiagnosticError(
            message,
            code=code,
            file=self.source.file_path,
            line=getattr(node, "line", 0),
            column=getattr(node, "col", 0),
        )


__all__ = [
    "CheckedQvrQiec",
    "QIEC_STATEMENT_TYPES",
    "QVR_SOURCE_VERSION",
    "QiecDiagnosticError",
    "QvrQiecLowerer",
    "QvrQiecSource",
    "has_qiec_surface",
    "lower_qvr_to_qiec",
    "non_qiec_projection",
    "qiec_projection",
]
