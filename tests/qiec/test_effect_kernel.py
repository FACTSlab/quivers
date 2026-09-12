import pytest

from quivers.qiec import (
    INT,
    NAT,
    STRING,
    ArgumentDef,
    AttachmentId,
    AttachmentRef,
    ComputationType,
    EffectBinder,
    EffectDef,
    EffectId,
    EffectRef,
    EffectRequest,
    EffectRow,
    Handle,
    HandlerClauseDef,
    HandlerDef,
    HandlerId,
    InterfaceEvolution,
    IndexBinder,
    KernelError,
    KernelRegistry,
    LiteralValue,
    OperationDef,
    OperationId,
    Perform,
    ResumptionGrade,
    RowEntry,
    SiteProvenance,
    SourceOrigin,
    TypeBinder,
    TypeApplication,
    TypeConstructorRef,
    TypeId,
    TypeVariable,
    infer_computation,
    infer_value,
    instantiate_effect,
)


def _reader() -> tuple[EffectDef, OperationDef]:
    effect_id = EffectId.derive("tests", "Reader", 1)
    operation = OperationDef(
        OperationId.derive(str(effect_id), "ask"),
        "ask",
        (TypeBinder("a"),),
        (ArgumentDef("key", STRING),),
        TypeVariable("a"),
    )
    effect = EffectDef(
        EffectRef(effect_id, "Reader", 1),
        (),
        (operation,),
    )
    return effect, operation


def _origin() -> SiteProvenance:
    return SiteProvenance(
        SourceOrigin(
            "tests",
            ("body", 0),
            "reader.ask",
            "qvr-source/v0.19",
        )
    )


def test_typed_request_and_total_handler_remove_only_the_matched_instance() -> None:
    effect, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(effect)
    entry = instantiate_effect(effect.ref, module="tests", lexical_path=("reader",))
    request = EffectRequest(
        entry.instance,
        effect.ref,
        operation.id,
        (INT,),
        (LiteralValue("answer", STRING),),
        INT,
        _origin(),
    )
    performed = infer_computation(Perform(request), registry)
    assert performed.result == INT
    assert performed.effects == EffectRow((entry,))

    handler = HandlerDef(
        HandlerId.derive("tests", "run_reader"),
        "run_reader",
        effect.ref,
        (HandlerClauseDef(operation.id, ResumptionGrade.LINEAR),),
        INT,
        INT,
    )
    registry.register_handler(handler)
    handled = infer_computation(
        Handle(entry.instance, handler.id, Perform(request)), registry
    )
    assert handled.result == INT
    assert handled.effects.entries == ()


def test_request_rejects_an_operation_from_another_interface_version() -> None:
    effect, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(effect)
    wrong = EffectRef(EffectId.derive("tests", "Reader", 2), "Reader", 2)
    request = EffectRequest(
        instantiate_effect(wrong, module="tests", lexical_path=(0,)).instance,
        wrong,
        operation.id,
        (INT,),
        (LiteralValue("answer", STRING),),
        INT,
        _origin(),
    )

    with pytest.raises(KernelError, match="interface/version"):
        infer_computation(Perform(request), registry)


def test_total_and_forwarding_handler_contracts_are_checked() -> None:
    effect, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(effect)
    missing = HandlerDef(
        HandlerId.derive("tests", "missing"),
        "missing",
        effect.ref,
        (),
        INT,
        INT,
    )
    with pytest.raises(KernelError, match="missing operations"):
        registry.register_handler(missing)

    forwarding = HandlerDef(
        HandlerId.derive("tests", "forward"),
        "forward",
        effect.ref,
        (HandlerClauseDef(operation.id, ResumptionGrade.AFFINE),),
        INT,
        INT,
        total=False,
        forwards_unknown=True,
    )
    with pytest.raises(KernelError, match="forwarding interface"):
        registry.register_handler(forwarding)

    open_effect = EffectDef(
        EffectRef(EffectId.derive("tests", "OpenReader", 1), "OpenReader", 1),
        (),
        (),
        InterfaceEvolution.FORWARDING,
    )
    registry.register_effect(open_effect)
    open_handler = HandlerDef(
        HandlerId.derive("tests", "open_forward"),
        "open_forward",
        open_effect.ref,
        (),
        INT,
        INT,
        total=False,
        forwards_unknown=True,
    )
    registry.register_handler(open_handler)


def test_partial_handler_retains_the_instance_row() -> None:
    effect, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(effect)
    entry = instantiate_effect(effect.ref, module="tests", lexical_path=("partial",))
    request = EffectRequest(
        entry.instance,
        effect.ref,
        operation.id,
        (INT,),
        (LiteralValue("answer", STRING),),
        INT,
        _origin(),
    )
    handler = HandlerDef(
        HandlerId.derive("tests", "partial"),
        "partial",
        effect.ref,
        (HandlerClauseDef(operation.id, ResumptionGrade.ZERO),),
        INT,
        INT,
        total=False,
    )
    registry.register_handler(handler)

    inferred = infer_computation(
        Handle(entry.instance, handler.id, Perform(request)),
        registry,
    )
    assert inferred.effects == EffectRow((RowEntry(entry.instance, effect.ref),))


def test_parameterized_effect_applications_are_distinct_and_checked() -> None:
    effect_id = EffectId.derive("tests", "State", 1)
    get = OperationDef(
        OperationId.derive(str(effect_id), "get"),
        "get",
        (),
        (),
        TypeVariable("state"),
    )
    state = EffectDef(
        EffectRef(effect_id, "State", 1),
        (TypeBinder("state"),),
        (get,),
    )
    int_state = state.apply((INT,))
    string_state = state.apply((STRING,))
    int_entry = instantiate_effect(
        int_state,
        module="tests",
        lexical_path=("state",),
    )
    string_entry = instantiate_effect(
        string_state,
        module="tests",
        lexical_path=("state",),
    )
    assert int_state != string_state
    assert int_entry.instance != string_entry.instance
    assert len(EffectRow((int_entry, string_entry)).entries) == 2

    registry = KernelRegistry()
    registry.register_effect(state)
    mismatched = EffectRequest(
        string_entry.instance,
        string_state,
        get.id,
        (),
        (),
        INT,
        _origin(),
    )
    with pytest.raises(KernelError, match="request result"):
        infer_computation(Perform(mismatched), registry)

    valid = EffectRequest(
        string_entry.instance,
        string_state,
        get.id,
        (),
        (),
        STRING,
        _origin(),
    )
    int_handler = HandlerDef(
        HandlerId.derive("tests", "int_state"),
        "int_state",
        int_state,
        (HandlerClauseDef(get.id, ResumptionGrade.LINEAR),),
        STRING,
        STRING,
    )
    registry.register_handler(int_handler)
    with pytest.raises(KernelError, match="does not match"):
        infer_computation(
            Handle(string_entry.instance, int_handler.id, Perform(valid)),
            registry,
        )


def test_effect_registration_checks_interface_and_operation_scope() -> None:
    effect_id = EffectId.derive("tests", "ScopedEffect", 1)
    valid_operation = OperationDef(
        OperationId.derive(str(effect_id), "valid"),
        "valid",
        (TypeBinder("result"),),
        (ArgumentDef("state", TypeVariable("state")),),
        TypeVariable("result"),
    )
    registry = KernelRegistry()
    registry.register_effect(
        EffectDef(
            EffectRef(effect_id, "ScopedEffect", 1),
            (TypeBinder("state"),),
            (valid_operation,),
        )
    )

    bad_id = EffectId.derive("tests", "BadScope", 1)
    unbound_operation = OperationDef(
        OperationId.derive(str(bad_id), "bad"),
        "bad",
        (),
        (),
        TypeVariable("missing"),
    )
    with pytest.raises(KernelError, match="unbound type variable"):
        KernelRegistry().register_effect(
            EffectDef(
                EffectRef(bad_id, "BadScope", 1),
                (TypeBinder("state"),),
                (unbound_operation,),
            )
        )

    with pytest.raises(ValueError, match="shadows interface binders"):
        EffectDef(
            EffectRef(EffectId.derive("tests", "Shadow", 1), "Shadow", 1),
            (TypeBinder("a"),),
            (
                OperationDef(
                    OperationId.derive("tests", "shadow"),
                    "shadow",
                    (TypeBinder("a"),),
                    (),
                    INT,
                ),
            ),
        )


def test_handler_telescope_is_scoped_and_instantiated_at_handle_site() -> None:
    effect, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(effect)
    entry = instantiate_effect(effect.ref, module="tests", lexical_path=("generic",))
    request = EffectRequest(
        entry.instance,
        effect.ref,
        operation.id,
        (INT,),
        (LiteralValue("answer", STRING),),
        INT,
        _origin(),
    )
    answer = TypeVariable("answer")
    generic = HandlerDef(
        HandlerId.derive("tests", "generic-reader"),
        "generic-reader",
        effect.ref,
        (HandlerClauseDef(operation.id, ResumptionGrade.LINEAR),),
        answer,
        answer,
        telescope=(TypeBinder("answer"),),
    )
    registry.register_handler(generic)
    inferred = infer_computation(
        Handle(entry.instance, generic.id, Perform(request), (INT,)),
        registry,
    )
    assert inferred.result == INT

    unsaturated = HandlerDef(
        HandlerId.derive("tests", "unsaturated-reader"),
        "unsaturated-reader",
        effect.ref,
        (HandlerClauseDef(operation.id, ResumptionGrade.LINEAR),),
        TypeVariable("free"),
        TypeVariable("free"),
    )
    with pytest.raises(KernelError, match="unbound type variable"):
        registry.register_handler(unsaturated)


def test_effect_application_deeply_validates_nested_static_arguments() -> None:
    malformed_inner = TypeApplication(
        TypeConstructorRef(
            TypeId.derive("tests", "Indexed"),
            "Indexed",
            (IndexBinder("index", NAT),),
        ),
        (INT,),
    )
    effect = EffectDef(
        EffectRef(EffectId.derive("tests", "Deep", 1), "Deep", 1),
        (TypeBinder("value"),),
        (),
    )

    with pytest.raises(TypeError, match="index argument"):
        effect.apply((malformed_inner,))

    reader, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(reader)
    entry = instantiate_effect(reader.ref, module="tests", lexical_path=("deep",))
    malformed_request = EffectRequest(
        entry.instance,
        reader.ref,
        operation.id,
        (malformed_inner,),
        (LiteralValue("key", STRING),),
        INT,
        _origin(),
    )
    with pytest.raises(TypeError, match="index argument"):
        infer_computation(Perform(malformed_request), registry)


def test_type_constructor_metadata_is_canonical_per_stable_id() -> None:
    type_id = TypeId.derive("tests", "Canonical")

    def effect_with_argument(
        label: str,
        constructor: TypeConstructorRef,
        arguments: tuple[object, ...] = (),
    ) -> EffectDef:
        effect_id = EffectId.derive("tests", "metadata", label)
        operation = OperationDef(
            OperationId.derive(effect_id, "use"),
            "use",
            (),
            (
                ArgumentDef(
                    "value",
                    TypeApplication(constructor, arguments),  # type: ignore[arg-type]
                ),
            ),
            INT,
        )
        return EffectDef(EffectRef(effect_id, label, 1), (), (operation,))

    registry = KernelRegistry()
    registry.register_effect(
        effect_with_argument("first", TypeConstructorRef(type_id, "Before"))
    )
    registry.register_effect(
        effect_with_argument("renamed", TypeConstructorRef(type_id, "After"))
    )

    conflicting = TypeConstructorRef(
        type_id,
        "After",
        (TypeBinder("argument"),),
    )
    with pytest.raises(KernelError, match="conflicting telescope metadata"):
        registry.register_effect(effect_with_argument("conflict", conflicting, (INT,)))


def test_handle_rejects_an_unsaturated_known_effect_static_argument() -> None:
    reader, operation = _reader()
    unary_id = EffectId.derive("tests", "UnaryEffect", 1)
    unary = EffectDef(
        EffectRef(unary_id, "UnaryEffect", 1),
        (TypeBinder("value"),),
        (),
    )
    registry = KernelRegistry()
    registry.register_effect(reader)
    registry.register_effect(unary)
    entry = instantiate_effect(reader.ref, module="tests", lexical_path=("effect-arg",))
    request = EffectRequest(
        entry.instance,
        reader.ref,
        operation.id,
        (INT,),
        (LiteralValue("key", STRING),),
        INT,
        _origin(),
    )
    handler = HandlerDef(
        HandlerId.derive("tests", "effect-polymorphic"),
        "effect-polymorphic",
        reader.ref,
        (HandlerClauseDef(operation.id, ResumptionGrade.LINEAR),),
        INT,
        INT,
        telescope=(EffectBinder("nested"),),
    )
    registry.register_handler(handler)

    with pytest.raises(KernelError, match="invalid application or version"):
        infer_computation(
            Handle(entry.instance, handler.id, Perform(request), (unary.ref,)),
            registry,
        )


def test_registry_validates_effect_references_on_every_static_surface() -> None:
    reader, reader_operation = _reader()
    state_id = EffectId.derive("tests", "SurfaceState", 1)
    state = EffectDef(
        EffectRef(state_id, "SurfaceState", 1),
        (TypeBinder("state"),),
        (),
    )
    registry = KernelRegistry()
    registry.register_effect(reader)
    registry.register_effect(state)
    carrier = TypeConstructorRef(
        TypeId.derive("tests", "EffectCarrierSurface"),
        "EffectCarrierSurface",
        (EffectBinder("effect"),),
    )

    invalid_references = (
        state.ref,
        EffectRef(state_id, "SurfaceState", 1, (reader.ref,)),
        EffectRef(state_id, "SurfaceState", 2, (INT,)),
    )
    for position, invalid in enumerate(invalid_references):
        invalid_type = TypeApplication(carrier, (invalid,))
        with pytest.raises(KernelError, match="invalid application or version"):
            infer_value(
                AttachmentRef(
                    AttachmentId.derive("tests", "invalid-effect", position),
                    invalid_type,
                ),
                registry,
            )

    unsaturated_type = TypeApplication(carrier, (state.ref,))

    def malformed_declaration(label: str, *, in_result: bool) -> EffectDef:
        effect_id = EffectId.derive("tests", "surface-declaration", label)
        operation = OperationDef(
            OperationId.derive(effect_id, "operation"),
            "operation",
            (),
            () if in_result else (ArgumentDef("bad", unsaturated_type),),
            unsaturated_type if in_result else INT,
        )
        return EffectDef(EffectRef(effect_id, label, 1), (), (operation,))

    for label, in_result in (("argument", False), ("result", True)):
        with pytest.raises(KernelError, match="invalid application or version"):
            registry.register_effect(malformed_declaration(label, in_result=in_result))

    entry = instantiate_effect(
        reader.ref,
        module="tests",
        lexical_path=("surface-request",),
    )
    bad_result_request = EffectRequest(
        entry.instance,
        reader.ref,
        reader_operation.id,
        (INT,),
        (LiteralValue("key", STRING),),
        unsaturated_type,
        _origin(),
    )
    with pytest.raises(KernelError, match="invalid application or version"):
        infer_computation(Perform(bad_result_request), registry)

    static_operation_id = OperationId.derive("tests", "surface-static-operation")
    static_operation = OperationDef(
        static_operation_id,
        "static",
        (EffectBinder("nested"),),
        (),
        INT,
    )
    static_effect = EffectDef(
        EffectRef(EffectId.derive("tests", "StaticSurface", 1), "StaticSurface", 1),
        (),
        (static_operation,),
    )
    registry.register_effect(static_effect)
    static_entry = instantiate_effect(
        static_effect.ref,
        module="tests",
        lexical_path=("surface-static-request",),
    )
    bad_static_request = EffectRequest(
        static_entry.instance,
        static_effect.ref,
        static_operation.id,
        (state.ref,),
        (),
        INT,
        _origin(),
    )
    with pytest.raises(KernelError, match="invalid application or version"):
        infer_computation(Perform(bad_static_request), registry)

    clause = (HandlerClauseDef(reader_operation.id, ResumptionGrade.LINEAR),)
    handler_surfaces = (
        HandlerDef(
            HandlerId.derive("tests", "bad-handler-input"),
            "bad-handler-input",
            reader.ref,
            clause,
            unsaturated_type,
            INT,
        ),
        HandlerDef(
            HandlerId.derive("tests", "bad-handler-output"),
            "bad-handler-output",
            reader.ref,
            clause,
            INT,
            unsaturated_type,
        ),
        HandlerDef(
            HandlerId.derive("tests", "bad-handler-row"),
            "bad-handler-row",
            reader.ref,
            clause,
            INT,
            INT,
            EffectRow(
                (
                    RowEntry(
                        instantiate_effect(
                            state.ref,
                            module="tests",
                            lexical_path=("bad-handler-row",),
                        ).instance,
                        state.ref,
                    ),
                )
            ),
        ),
    )
    for handler in handler_surfaces:
        with pytest.raises(KernelError, match="invalid application or version"):
            registry.register_handler(handler)

    invalid_row = handler_surfaces[-1].introduced
    with pytest.raises(KernelError, match="invalid application or version"):
        registry.validate_effect_row(invalid_row)
    with pytest.raises(KernelError, match="invalid application or version"):
        registry.validate_computation_type(
            ComputationType(EffectRow(), unsaturated_type)
        )

    unknown = EffectRef(EffectId.derive("tests", "External", 1), "External", 1)
    external_type = TypeApplication(carrier, (unknown,))
    attachment = AttachmentRef(AttachmentId.derive("tests", "external"), external_type)
    assert infer_value(attachment, registry) == external_type
