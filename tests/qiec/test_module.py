from __future__ import annotations

import pytest

from quivers.qiec import (
    ComputationId,
    EMPTY_ROW,
    INT,
    QIEC_ABI,
    ComputationType,
    EffectDef,
    EffectId,
    EffectInstanceId,
    EffectRef,
    EffectRequest,
    EffectRow,
    Handle,
    HandlerClauseDef,
    HandlerDef,
    HandlerId,
    LiteralValue,
    NamedComputation,
    NamedEffectInstance,
    OperationDef,
    OperationId,
    Perform,
    QiecModule,
    ResumptionGrade,
    RowEntry,
    RowVariable,
    RowVariableId,
    Return,
    SiteProvenance,
    SourceOrigin,
    TypeBinder,
    UserIndexSort,
    TypeVariable,
    StaticVariableId,
    dumps,
    loads,
    validate_module,
)


SOURCE_PROTOCOL = "qvr-source/v0.19"


def _origin() -> SourceOrigin:
    return SourceOrigin(
        module="example",
        structural_path=("computations", "answer"),
        role="computation",
        source_protocol=SOURCE_PROTOCOL,
        file="example.qvr",
        line=1,
        column=1,
    )


def _request_provenance(
    name: str,
    *,
    module: str = "example",
    source_protocol: str = SOURCE_PROTOCOL,
) -> SiteProvenance:
    return SiteProvenance(
        SourceOrigin(
            module,
            ("computations", name, "request"),
            "perform",
            source_protocol,
        )
    )


def test_module_is_serializable_and_recheckable() -> None:
    computation = NamedComputation(
        ComputationId.derive("tests", "answer"),
        "answer",
        (),
        (),
        Return(LiteralValue(42, INT)),
        ComputationType(EMPTY_ROW, INT),
        _origin(),
    )
    module = QiecModule("example", SOURCE_PROTOCOL, computations=(computation,))

    assert loads(dumps(module)) == module
    assert validate_module(module).families == {}


def test_module_rejects_abi_guessing() -> None:
    with pytest.raises(ValueError, match="module ABI"):
        QiecModule("example", SOURCE_PROTOCOL, abi="qiec-core/v99")

    assert QIEC_ABI == "qiec-core/v1alpha1"


def test_module_rejects_origin_from_another_route() -> None:
    computation = NamedComputation(
        ComputationId.derive("tests", "answer"),
        "answer",
        (),
        (),
        Return(LiteralValue(42, INT)),
        ComputationType(EMPTY_ROW, INT),
        SourceOrigin(
            "example",
            ("computations", "answer"),
            "computation",
            "qvr-source/v0.18",
        ),
    )

    with pytest.raises(ValueError, match="another source protocol"):
        QiecModule("example", SOURCE_PROTOCOL, computations=(computation,))


def test_module_checks_computation_static_scope() -> None:
    a = TypeVariable("a")
    computation = NamedComputation(
        ComputationId.derive("tests", "identity"),
        "identity",
        (TypeBinder("a"),),
        (),
        Return(LiteralValue(42, INT)),
        ComputationType(EMPTY_ROW, a),
        _origin(),
    )

    with pytest.raises(Exception, match="not declared type"):
        validate_module(
            QiecModule("example", SOURCE_PROTOCOL, computations=(computation,))
        )


def test_module_rejects_undeclared_static_variable() -> None:
    """A result type mentioning a variable no telescope binds is rejected.

    The rejection now happens while the computation's signature is
    registered, before any body is rechecked, and names the offending
    variable. Registering signatures first is what lets a call resolve
    to a computation declared later, so this check moved earlier with
    them.
    """
    a = TypeVariable("a")
    computation = NamedComputation(
        ComputationId.derive("tests", "bad"),
        "bad",
        (),
        (),
        Return(LiteralValue(42, INT)),
        ComputationType(EMPTY_ROW, a),
        _origin(),
    )

    with pytest.raises(Exception, match="unbound type variable 'a'"):
        validate_module(
            QiecModule("example", SOURCE_PROTOCOL, computations=(computation,))
        )


def _effect_fixture() -> tuple[EffectDef, NamedEffectInstance, OperationDef]:
    ref = EffectRef(EffectId.derive("example", "Reader"), "Reader")
    operation = OperationDef(
        OperationId.derive(ref.id, "read"),
        "read",
        (),
        (),
        INT,
    )
    effect = EffectDef(ref, (), (operation,))
    entry = RowEntry(EffectInstanceId.derive("example", "reader"), ref)
    instance = NamedEffectInstance("reader", entry, _origin())
    return effect, instance, operation


def test_module_rejects_request_for_undeclared_lexical_instance() -> None:
    effect, _instance, operation = _effect_fixture()
    request = EffectRequest(
        EffectInstanceId.derive("example", "forged"),
        effect.ref,
        operation.id,
        (),
        (),
        INT,
        _request_provenance("read"),
    )
    computation = NamedComputation(
        ComputationId.derive("tests", "read"),
        "read",
        (),
        (),
        Perform(request),
        ComputationType(EffectRow((RowEntry(request.instance, effect.ref),)), INT),
        _origin(),
    )

    with pytest.raises(Exception, match="undeclared lexical effect instance"):
        validate_module(
            QiecModule(
                "example",
                SOURCE_PROTOCOL,
                effects=(effect,),
                computations=(computation,),
            )
        )


def test_module_rejects_open_tail_that_does_not_prove_declared_lacks() -> None:
    effect, instance, operation = _effect_fixture()
    actual_tail = RowVariable(
        "sigma",
        RowVariableId.derive("example", "actual"),
    )
    declared_tail = RowVariable(
        "rho",
        RowVariableId.derive("example", "declared"),
        (instance.entry.instance,),
    )
    handler = HandlerDef(
        HandlerId.derive("example", "open-handler"),
        "open-handler",
        effect.ref,
        (HandlerClauseDef(operation.id, ResumptionGrade.LINEAR),),
        INT,
        INT,
        EffectRow((), actual_tail),
    )
    request = EffectRequest(
        instance.entry.instance,
        effect.ref,
        operation.id,
        (),
        (),
        INT,
        _request_provenance("open"),
    )
    computation = NamedComputation(
        ComputationId.derive("tests", "open"),
        "open",
        (),
        (),
        Handle(instance.entry.instance, handler.id, Perform(request)),
        ComputationType(EffectRow((), declared_tail), INT),
        _origin(),
    )

    with pytest.raises(Exception, match="not declared type"):
        validate_module(
            QiecModule(
                "example",
                SOURCE_PROTOCOL,
                effects=(effect,),
                instances=(instance,),
                handlers=(handler,),
                computations=(computation,),
            )
        )


def test_module_validates_unused_entries_in_declared_effect_row() -> None:
    effect, instance, _operation = _effect_fixture()
    external = EffectRef(EffectId.derive("external", "Other"), "Other")
    bad_entry = RowEntry(instance.entry.instance, external)
    computation = NamedComputation(
        ComputationId.derive("tests", "pure"),
        "pure",
        (),
        (),
        Return(LiteralValue(1, INT)),
        ComputationType(
            EffectRow(
                (bad_entry,),
                RowVariable(
                    "rho",
                    RowVariableId.derive("example", "declared-row"),
                    (bad_entry.instance,),
                ),
            ),
            INT,
        ),
        _origin(),
    )

    with pytest.raises(Exception, match="wrong interface"):
        validate_module(
            QiecModule(
                "example",
                SOURCE_PROTOCOL,
                effects=(effect,),
                instances=(instance,),
                computations=(computation,),
            )
        )


def test_module_rejects_embedded_index_sort_that_disagrees_with_declaration() -> None:
    from quivers.qiec import FamilyDecl, FamilyId, IndexBinder

    declared = UserIndexSort("Nat", ("Z", "S"), (0, 1))
    conflicting = UserIndexSort("Nat", ("Only",), (0,))
    family = FamilyDecl(
        FamilyId.derive("example", "Vector"),
        "Vector",
        (),
        (IndexBinder("n", conflicting, refinable=True),),
        (),
    )

    with pytest.raises(Exception, match="disagrees with its declaration"):
        validate_module(
            QiecModule(
                "example",
                SOURCE_PROTOCOL,
                index_sorts=(declared,),
                families=(family,),
            )
        )


@pytest.mark.parametrize(
    ("request_origin", "message"),
    [
        (_request_provenance("read", module="elsewhere"), "another QIEC module"),
        (
            _request_provenance("read", source_protocol="qvr-source/v0.20"),
            "another source protocol",
        ),
    ],
)
def test_module_rejects_request_provenance_from_another_route(
    request_origin: SiteProvenance,
    message: str,
) -> None:
    effect, instance, operation = _effect_fixture()
    request = EffectRequest(
        instance.entry.instance,
        effect.ref,
        operation.id,
        (),
        (),
        INT,
        request_origin,
    )
    computation = NamedComputation(
        ComputationId.derive("tests", "read"),
        "read",
        (),
        (),
        Perform(request),
        ComputationType(EffectRow((instance.entry,)), INT),
        _origin(),
    )

    with pytest.raises(Exception, match=message):
        validate_module(
            QiecModule(
                "example",
                SOURCE_PROTOCOL,
                effects=(effect,),
                instances=(instance,),
                computations=(computation,),
            )
        )


def test_module_rejects_non_string_identity_fields() -> None:
    with pytest.raises(ValueError, match="module name"):
        QiecModule(1, "qvr-source/v0.19")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source protocol"):
        QiecModule("example", 2)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "argument",
    [
        TypeVariable("state"),
        TypeVariable(
            "state",
            identity=StaticVariableId.derive("example", "instance", "state"),
        ),
    ],
)
def test_module_rejects_static_variables_in_module_effect_instances(
    argument: TypeVariable,
) -> None:
    ref = EffectRef(EffectId.derive("example", "State"), "State")
    operation = OperationDef(
        OperationId.derive(ref.id, "get"),
        "get",
        (),
        (),
        TypeVariable("state"),
    )
    effect = EffectDef(ref, (TypeBinder("state"),), (operation,))
    concrete = effect.apply((argument,))
    instance = NamedEffectInstance(
        "state",
        RowEntry(EffectInstanceId.derive("example", "state"), concrete),
        _origin(),
    )

    with pytest.raises(
        Exception, match="module-level effect instance.*static variable"
    ):
        validate_module(
            QiecModule(
                "example",
                SOURCE_PROTOCOL,
                effects=(effect,),
                instances=(instance,),
            )
        )
