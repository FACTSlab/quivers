import pytest

from quivers.qiec import (
    BOOL,
    INT,
    NAT,
    SHAPE,
    STRING,
    TYPE,
    UNIT,
    BranchPattern,
    Case,
    CaseBranch,
    CaseMotive,
    CheckContext,
    ConstructorDecl,
    ConstructorId,
    ConstructorValue,
    CoverageStatus,
    EffectDef,
    EffectBinder,
    EffectId,
    EffectRef,
    EffectRequest,
    EffectVariable,
    EvidenceValue,
    FamilyDecl,
    FamilyId,
    FieldDef,
    FunctionType,
    EqualityType,
    IndexBinder,
    IndexConstructor,
    IndexLiteral,
    IndexVariable,
    KernelError,
    KernelRegistry,
    LiteralValue,
    Local,
    OperationDef,
    OperationId,
    Perform,
    Reachability,
    Return,
    SiteProvenance,
    SourceOrigin,
    StaticScopeId,
    TypeApplication,
    TypeBinder,
    TypeConstructorRef,
    TypeId,
    TypeVariable,
    UserIndexSort,
    Var,
    check_indexed_coverage,
    constructor_skolems,
    infer_computation,
    infer_value,
    instantiate_effect,
    refine_branch,
)


def _expr_family() -> tuple[
    FamilyDecl,
    ConstructorDecl,
    ConstructorDecl,
    ConstructorDecl,
]:
    family_id = FamilyId.derive("tests", "Expr")
    bool_id = ConstructorId.derive(str(family_id), "Bool")
    int_id = ConstructorId.derive(str(family_id), "Int")
    any_id = ConstructorId.derive(str(family_id), "Any")
    family = FamilyDecl(
        family_id,
        "Expr",
        (),
        (TypeBinder("a", TYPE, refinable=True),),
        (bool_id, int_id, any_id),
    )
    bool_constructor = ConstructorDecl(
        bool_id,
        family_id,
        "Bool",
        (),
        (),
        (BOOL,),
    )
    int_constructor = ConstructorDecl(
        int_id,
        family_id,
        "Int",
        (),
        (),
        (INT,),
    )
    any_constructor = ConstructorDecl(
        any_id,
        family_id,
        "Any",
        (TypeBinder("x"),),
        (),
        (TypeVariable("x"),),
    )
    return family, bool_constructor, int_constructor, any_constructor


def _registry_with_expr() -> tuple[
    KernelRegistry,
    FamilyDecl,
    ConstructorDecl,
    ConstructorDecl,
    ConstructorDecl,
]:
    family, bool_constructor, int_constructor, any_constructor = _expr_family()
    registry = KernelRegistry()
    registry.register_family(family)
    registry.register_constructor(bool_constructor)
    registry.register_constructor(int_constructor)
    registry.register_constructor(any_constructor)
    return registry, family, bool_constructor, int_constructor, any_constructor


def _expr(family: FamilyDecl, index: object) -> TypeApplication:
    return TypeApplication(family.type_constructor, (index,))  # type: ignore[arg-type]


def _origin() -> SiteProvenance:
    return SiteProvenance(SourceOrigin("tests", ("case", 1), "ask", "qvr-source/v0.19"))


def _scope(label: str) -> StaticScopeId:
    return StaticScopeId.derive("tests", "case-branch", label)


def test_constructor_result_indices_are_checked() -> None:
    registry, family, bool_constructor, _, _ = _registry_with_expr()
    value = ConstructorValue(bool_constructor.id, (), (), _expr(family, BOOL))
    assert infer_value(value, registry) == _expr(family, BOOL)

    malformed = ConstructorValue(bool_constructor.id, (), (), _expr(family, INT))
    with pytest.raises(KernelError, match="constructor result"):
        infer_value(malformed, registry)


def test_indexed_coverage_omits_only_certified_impossible_constructors() -> None:
    registry, family, bool_constructor, _, any_constructor = _registry_with_expr()
    constructors = tuple(
        registry.constructor(constructor) for constructor in family.constructors
    )
    scrutinee = _expr(family, BOOL)

    missing_polymorphic = check_indexed_coverage(
        family,
        constructors,
        scrutinee,
        (BranchPattern(bool_constructor.id),),
    )
    assert missing_polymorphic.status is CoverageStatus.INCOMPLETE
    assert missing_polymorphic.missing == (any_constructor.id,)

    complete = check_indexed_coverage(
        family,
        constructors,
        scrutinee,
        (BranchPattern(bool_constructor.id), BranchPattern(any_constructor.id)),
    )
    assert complete.status is CoverageStatus.COMPLETE


def test_indexed_coverage_distinguishes_nullary_and_applied_index_constructors() -> (
    None
):
    natural = UserIndexSort("Nat", ("Z", "S"), (0, 1))
    family_id = FamilyId.derive("tests", "Vec")
    nil_id = ConstructorId.derive(family_id, "Nil")
    cons_id = ConstructorId.derive(family_id, "Cons")
    family = FamilyDecl(
        family_id,
        "Vec",
        (),
        (IndexBinder("n", natural, refinable=True),),
        (nil_id, cons_id),
    )
    nil = ConstructorDecl(
        nil_id, family_id, "Nil", (), (), (IndexLiteral("Z", natural),)
    )
    cons = ConstructorDecl(
        cons_id,
        family_id,
        "Cons",
        (IndexBinder("m", natural),),
        (),
        (IndexConstructor("S", (IndexVariable("m", natural),), natural),),
    )
    scrutinee = TypeApplication(
        family.type_constructor,
        (IndexConstructor("S", (IndexVariable("n", natural),), natural),),
    )

    coverage = check_indexed_coverage(
        family,
        (nil, cons),
        scrutinee,
        (BranchPattern(cons_id),),
    )

    assert coverage.status is CoverageStatus.COMPLETE


def test_unknown_branch_refinement_retains_its_effect_row() -> None:
    registry, family, bool_constructor, _, any_constructor = _registry_with_expr()
    effect_id = EffectId.derive("tests", "Ask")
    operation = OperationDef(
        OperationId.derive(str(effect_id), "ask"),
        "ask",
        (),
        (),
        STRING,
    )
    effect = EffectDef(EffectRef(effect_id, "Ask"), (), (operation,))
    registry.register_effect(effect)
    instance = instantiate_effect(effect.ref, module="tests", lexical_path=("case",))
    request = EffectRequest(
        instance.instance,
        effect.ref,
        operation.id,
        (),
        (),
        STRING,
        _origin(),
    )
    scrutinee_type = _expr(family, BOOL)
    scrutinee = ConstructorValue(bool_constructor.id, (), (), scrutinee_type)
    bool_scope = _scope("effect-bool")
    any_scope = _scope("effect-any")
    unknown_type = constructor_skolems(any_constructor, any_scope)[0]
    case = Case(
        scrutinee,
        CaseMotive(
            (TypeBinder("result_index", TYPE, refinable=True),),
            STRING,
        ),
        (
            CaseBranch(
                bool_constructor.id,
                (),
                (),
                Return(LiteralValue("bool", STRING)),
                bool_scope,
            ),
            CaseBranch(
                any_constructor.id,
                (unknown_type,),
                (),
                Perform(request),
                any_scope,
            ),
        ),
    )

    inferred = infer_computation(case, registry)
    assert inferred.result == STRING
    assert inferred.effects.lookup(instance.instance) == effect.ref
    refinement = refine_branch(
        family,
        any_constructor,
        scrutinee_type,
        (unknown_type,),
    )
    assert refinement.reachability is Reachability.UNKNOWN


def test_branch_equality_evidence_cannot_escape_its_context() -> None:
    registry, family, _, _, any_constructor = _registry_with_expr()
    refinement = refine_branch(
        family,
        any_constructor,
        _expr(family, BOOL),
        (TypeVariable("x"),),
    )
    evidence = EvidenceValue(refinement.givens[0])

    with pytest.raises(KernelError, match="escaped or was forged"):
        infer_value(evidence, registry)
    assert (
        infer_value(
            evidence,
            registry,
            CheckContext().with_givens(refinement.givens),
        )
        == refinement.givens[0].equality
    )


@pytest.mark.parametrize(
    "forbidden_type",
    (
        FunctionType(INT, INT),
        EqualityType(TYPE, INT, INT),
        TypeApplication(TypeConstructorRef(TypeId.derive("tests", "Box"), "Box")),
    ),
)
def test_literals_are_restricted_to_primitive_types(forbidden_type: object) -> None:
    with pytest.raises(KernelError, match="restricted to"):
        infer_value(
            LiteralValue(None, forbidden_type),  # type: ignore[arg-type]
            KernelRegistry(),
        )


def test_case_branch_cannot_choose_a_constructor_existential() -> None:
    family_id = FamilyId.derive("tests", "Packed")
    constructor_id = ConstructorId.derive(str(family_id), "Pack")
    family = FamilyDecl(family_id, "Packed", (), (), (constructor_id,))
    constructor = ConstructorDecl(
        constructor_id,
        family_id,
        "Pack",
        (TypeBinder("packed"),),
        (FieldDef("value", TypeVariable("packed")),),
        (),
    )
    registry = KernelRegistry()
    registry.register_family(family)
    registry.register_constructor(constructor)
    packed_type = TypeApplication(family.type_constructor)
    scrutinee = ConstructorValue(
        constructor_id,
        (STRING,),
        (LiteralValue("not an integer", STRING),),
        packed_type,
    )
    field = Local("value", INT)
    computation = Case(
        scrutinee,
        CaseMotive((), INT),
        (
            CaseBranch(
                constructor_id,
                (INT,),
                (field,),
                Return(Var(field)),
                _scope("forged-pack"),
            ),
        ),
    )

    with pytest.raises(KernelError, match="canonical rigid skolems"):
        infer_computation(computation, registry)


def test_constructor_skolem_cannot_escape_through_a_case_motive() -> None:
    family_id = FamilyId.derive("tests", "EscapingPack")
    constructor_id = ConstructorId.derive(family_id, "Pack")
    family = FamilyDecl(family_id, "EscapingPack", (), (), (constructor_id,))
    constructor = ConstructorDecl(
        constructor_id,
        family_id,
        "Pack",
        (TypeBinder("packed"),),
        (FieldDef("value", TypeVariable("packed")),),
        (),
    )
    registry = KernelRegistry()
    registry.register_family(family)
    registry.register_constructor(constructor)
    packed_type = TypeApplication(family.type_constructor)
    scrutinee = ConstructorValue(
        constructor_id,
        (STRING,),
        (LiteralValue("hidden", STRING),),
        packed_type,
    )
    scope = _scope("escape")
    skolem = constructor_skolems(constructor, scope)[0]
    assert isinstance(skolem, TypeVariable)
    field = Local("hidden", skolem)
    computation = Case(
        scrutinee,
        CaseMotive((), skolem),
        (
            CaseBranch(
                constructor_id,
                (skolem,),
                (field,),
                Return(Var(field)),
                scope,
            ),
        ),
    )

    with pytest.raises(KernelError, match="escapes its case branch"):
        infer_computation(computation, registry)

    effect_id = EffectId.derive("tests", "SkolemEffect")
    operation = OperationDef(
        OperationId.derive(effect_id, "emit"),
        "emit",
        (),
        (),
        UNIT,
    )
    effect = EffectDef(
        EffectRef(effect_id, "SkolemEffect"),
        (TypeBinder("payload"),),
        (operation,),
    )
    registry.register_effect(effect)
    effect_scope = _scope("effect-row-escape")
    effect_skolem = constructor_skolems(constructor, effect_scope)[0]
    assert isinstance(effect_skolem, TypeVariable)
    applied = effect.apply((effect_skolem,))
    instance = instantiate_effect(
        applied,
        module="tests",
        lexical_path=("effect-row-escape",),
    )
    request = EffectRequest(
        instance.instance,
        applied,
        operation.id,
        (),
        (),
        UNIT,
        _origin(),
    )
    effect_case = Case(
        scrutinee,
        CaseMotive((), UNIT),
        (
            CaseBranch(
                constructor_id,
                (effect_skolem,),
                (Local("ignored", effect_skolem),),
                Perform(request),
                effect_scope,
            ),
        ),
    )

    with pytest.raises(KernelError, match="case branch effect row"):
        infer_computation(effect_case, registry)


def test_constructor_registration_checks_scope_shadowing_and_index_sort() -> None:
    family_id = FamilyId.derive("tests", "Scoped")
    constructor_id = ConstructorId.derive(str(family_id), "Scoped")
    family = FamilyDecl(
        family_id,
        "Scoped",
        (TypeBinder("parameter"),),
        (IndexBinder("index", NAT, refinable=True),),
        (constructor_id,),
    )
    registry = KernelRegistry()
    registry.register_family(family)

    shadowing = ConstructorDecl(
        constructor_id,
        family_id,
        "Scoped",
        (TypeBinder("parameter"),),
        (),
        (IndexVariable("missing", NAT),),
    )
    with pytest.raises(KernelError, match="shadows family binders"):
        registry.register_constructor(shadowing)

    unbound = ConstructorDecl(
        constructor_id,
        family_id,
        "Scoped",
        (),
        (FieldDef("bad", TypeVariable("missing")),),
        (IndexVariable("missing", NAT),),
    )
    with pytest.raises(KernelError, match="unbound type variable"):
        registry.register_constructor(unbound)

    unbound_index = ConstructorDecl(
        constructor_id,
        family_id,
        "Scoped",
        (),
        (),
        (IndexVariable("missing", NAT),),
    )
    with pytest.raises(KernelError, match="unbound index variable"):
        registry.register_constructor(unbound_index)

    effect_carrier = TypeConstructorRef(
        TypeId.derive("tests", "EffectCarrier"),
        "EffectCarrier",
        (EffectBinder("effect"),),
    )
    unbound_effect = ConstructorDecl(
        constructor_id,
        family_id,
        "Scoped",
        (),
        (
            FieldDef(
                "bad",
                TypeApplication(effect_carrier, (EffectVariable("missing"),)),
            ),
        ),
        (IndexVariable("also_missing", NAT),),
    )
    with pytest.raises(KernelError, match="unbound effect variable"):
        registry.register_constructor(unbound_effect)

    wrong_sort = ConstructorDecl(
        constructor_id,
        family_id,
        "Scoped",
        (IndexBinder("shape", SHAPE),),
        (),
        (IndexVariable("shape", SHAPE),),
    )
    with pytest.raises(KernelError, match="wrong index sort"):
        registry.register_constructor(wrong_sort)


def test_constructor_skolems_are_alpha_stable_and_scope_fresh() -> None:
    _, _, _, _, constructor = _registry_with_expr()
    left_scope = _scope("left")
    right_scope = _scope("right")
    left = constructor_skolems(constructor, left_scope)[0]
    right = constructor_skolems(constructor, right_scope)[0]

    assert isinstance(left, TypeVariable)
    assert isinstance(right, TypeVariable)
    assert left != right
    assert left.identity != right.identity
    assert TypeVariable("renamed", left.kind, left.identity) == left

    with pytest.raises(KernelError, match="reused an enclosing static scope"):
        CheckContext().with_static_scope(left_scope).with_static_scope(left_scope)


def test_case_branch_scopes_must_be_globally_fresh() -> None:
    registry, family, bool_constructor, _, any_constructor = _registry_with_expr()
    repeated = _scope("repeated")
    scrutinee = ConstructorValue(
        bool_constructor.id,
        (),
        (),
        _expr(family, BOOL),
    )
    computation = Case(
        scrutinee,
        CaseMotive((TypeBinder("index", TYPE, refinable=True),), STRING),
        (
            CaseBranch(
                bool_constructor.id,
                (),
                (),
                Return(LiteralValue("bool", STRING)),
                repeated,
            ),
            CaseBranch(
                any_constructor.id,
                constructor_skolems(any_constructor, repeated),
                (),
                Return(LiteralValue("any", STRING)),
                repeated,
            ),
        ),
    )

    with pytest.raises(KernelError, match="globally fresh"):
        infer_computation(computation, registry)


def test_constructor_instantiation_deeply_validates_static_arguments() -> None:
    family_id = FamilyId.derive("tests", "DeepArguments")
    constructor_id = ConstructorId.derive(family_id, "Deep")
    family = FamilyDecl(
        family_id,
        "DeepArguments",
        (TypeBinder("value"),),
        (),
        (constructor_id,),
    )
    constructor = ConstructorDecl(
        constructor_id,
        family_id,
        "Deep",
        (),
        (),
        (),
    )
    registry = KernelRegistry()
    registry.register_family(family)
    registry.register_constructor(constructor)
    malformed = TypeApplication(
        TypeConstructorRef(
            TypeId.derive("tests", "NeedsIndex"),
            "NeedsIndex",
            (IndexBinder("index", NAT),),
        ),
        (INT,),
    )
    result = TypeApplication(family.type_constructor, (malformed,))

    with pytest.raises(TypeError, match="index argument"):
        infer_value(
            ConstructorValue(constructor_id, (malformed,), (), result),
            registry,
        )


def test_family_lookup_enforces_canonical_type_constructor_metadata() -> None:
    registry, family, _, _, _ = _registry_with_expr()
    renamed = TypeConstructorRef(
        family.type_constructor.id,
        "RenamedExpr",
        family.type_constructor.telescope,
    )
    assert registry.family_for_type(TypeApplication(renamed, (BOOL,))) == family

    conflicting = TypeConstructorRef(
        family.type_constructor.id,
        "RenamedExpr",
        (),
    )
    with pytest.raises(KernelError, match="conflicting telescope metadata"):
        registry.family_for_type(TypeApplication(conflicting))
