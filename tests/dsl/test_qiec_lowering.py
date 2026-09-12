"""End-to-end checks for the exact QVR v0.19 to QIEC route."""

from __future__ import annotations

import pytest
from didactic.extensions import UnsupportedLoweringRouteError
from didactic.gadt import GADT

from quivers.dsl import Compiler, loads as compile_source, parse
from quivers.dsl.ast_nodes import (
    QiecEffectBinder,
    QiecFamilyDecl,
    QiecIndexName,
    QiecTypeApplication,
)
from quivers.dsl.qiec_lowering import (
    QVR_SOURCE_VERSION,
    QiecDiagnosticError,
    has_qiec_surface,
    lower_qvr_to_qiec,
)
from quivers.qiec import (
    INT,
    QIEC_ABI,
    Case,
    EffectRef,
    LiteralValue,
    QiecModule,
    Return,
    TypeApplication,
    TypeVariable,
    dumps,
    loads,
)


DECLARATIONS = """\
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

effect State[S : Type] [version=1, evolution=sealed]
    get : Unit -> S
    put : S -> Unit

instance cell : State[Int]

handler run_state[S : Type, A : Type] for State[S] : A -> A [coverage=total]
    get resumes 1
    put resumes 0
"""


COMPUTATIONS = """\
define swap(next : Int) : Int !{cell | rho lacks cell} =
    let prior <- perform cell.get()
    perform cell.put(next)
    return prior

define singleton[A : Type](value : A) : Vec[A](S(Z)) !{} =
    return construct Cons[A, Z](value, construct Nil[A]() as Vec[A](Z)) as Vec[A](S(Z))

define head[A : Type, n : Nat](xs : Vec[A](S(n))) : A !{} =
    case xs motive (m : Nat) => A
        Cons[n](head, tail) =>
            return head

define handled() : Int !{} =
    handle cell with run_state[Int, Int] in
        let current <- perform cell.get()
        return current
"""


def test_complete_surface_lowers_to_serializable_checked_module() -> None:
    parsed = parse(DECLARATIONS + "\n" + COMPUTATIONS, "models/state.qvr")

    module = lower_qvr_to_qiec(
        parsed,
        module_name="models.state",
        file_path="models/state.qvr",
    )

    assert isinstance(module, QiecModule)
    assert module.source_protocol == QVR_SOURCE_VERSION
    assert module.abi == QIEC_ABI
    assert [family.name for family in module.families] == ["Vec"]
    assert [effect.ref.name for effect in module.effects] == ["State"]
    assert [item.name for item in module.computations] == [
        "swap",
        "singleton",
        "head",
        "handled",
    ]
    assert module.computations[0].type.effects.tail is not None
    assert loads(dumps(module)) == module


def test_indexed_signature_is_checked_through_didactics_public_gadt_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[GADT] = []
    original_compile = GADT.compile

    def record_compile(language: GADT):
        compiled.append(language)
        return original_compile(language)

    monkeypatch.setattr(GADT, "compile", record_compile)

    lower_qvr_to_qiec(parse(DECLARATIONS), module_name="models.state")

    assert len(compiled) == 1
    indexed_family = next(
        family
        for family in compiled[0].families
        if family.closed and len(family.parameters) == 2
    )
    assert len(indexed_family.constructors) == 2
    assert all(
        operation.output.name == indexed_family.name
        for operation in compiled[0].operations
        if operation.name in indexed_family.constructors
    )


def test_didactic_constructor_static_arguments_are_explicit_even_when_phantom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[GADT] = []
    original_compile = GADT.compile

    def record_compile(language: GADT):
        compiled.append(language)
        return original_compile(language)

    monkeypatch.setattr(GADT, "compile", record_compile)
    lower_qvr_to_qiec(
        parse(
            """family Phantom[A : Type] : Type
    constructor Pack[X : Type] : Unit -> Phantom[A]
"""
        )
    )

    constructor = next(
        operation
        for operation in compiled[0].operations
        if operation.role == "constructor"
    )
    assert len(constructor.inputs) == 3
    assert constructor.explicit_inputs == constructor.inputs


def test_constructor_cannot_capture_a_refinable_family_index() -> None:
    parsed = parse(
        """index Nat = Z | S(Nat)
family Bad[A : Type](n : Nat) : Type
    constructor Capture : Bad[A](n) -> Bad[A](Z)
""",
        "capture.qvr",
    )
    declaration = parsed.statements[1]
    assert isinstance(declaration, QiecFamilyDecl)
    field_type = declaration.constructors[0].arguments[0]
    assert isinstance(field_type, QiecTypeApplication)
    authored_use = field_type.indices[0]
    assert isinstance(authored_use, QiecIndexName)

    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parsed, file_path="capture.qvr")

    assert captured.value.code == "qiec-index"
    assert captured.value.line == authored_use.line
    assert captured.value.column == authored_use.col
    assert "unknown index 'n'" in captured.value.message


def test_constructor_local_telescope_cannot_shadow_a_family_binder() -> None:
    parsed = parse(
        """index Nat = Z | S(Nat)
family Bad[A : Type](n : Nat) : Type
    constructor Shadow[n : Nat] : Bad[A](Z)
""",
        "shadow.qvr",
    )
    declaration = parsed.statements[1]
    assert isinstance(declaration, QiecFamilyDecl)
    authored_binder = declaration.constructors[0].binders[0]

    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parsed, file_path="shadow.qvr")

    assert captured.value.code == "qiec-index"
    assert captured.value.line == authored_binder.line
    assert captured.value.column == authored_binder.col
    assert "shadows family binder 'n'" in captured.value.message


def test_concrete_effect_cannot_refine_a_uniform_family_parameter() -> None:
    parsed = parse(
        """family Token[E : Effect] : Type
    constructor StateToken : Token[State[Int]]

effect State[S : Type]
    get : Unit -> S
""",
        "effect-parameter.qvr",
    )
    declaration = parsed.statements[0]
    assert isinstance(declaration, QiecFamilyDecl)
    assert isinstance(declaration.parameters[0], QiecEffectBinder)
    authored_result = declaration.constructors[0].result

    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parsed, file_path="effect-parameter.qvr")

    assert captured.value.code == "qiec-index"
    assert captured.value.line == authored_result.line
    assert captured.value.column == authored_result.col
    assert "must preserve family parameters" in captured.value.message


def test_effect_operation_type_can_reference_its_own_interface_header() -> None:
    module = lower_qvr_to_qiec(
        parse(
            """family Token[E : Effect] : Type
    constructor Keep : Token[E]

effect Self
    inspect : Token[Self] -> Unit
"""
        )
    )

    self_effect = module.effects[0]
    argument_type = self_effect.operations[0].arguments[0].type
    assert isinstance(argument_type, TypeApplication)
    assert argument_type.arguments == (self_effect.ref,)


def test_effect_operation_type_can_reference_a_later_interface_header() -> None:
    module = lower_qvr_to_qiec(
        parse(
            """family Token[E : Effect] : Type
    constructor Keep : Token[E]

effect First
    inspect : Token[Second] -> Unit
effect Second
    inspect : Unit -> Unit
"""
        )
    )

    first, second = module.effects
    argument_type = first.operations[0].arguments[0].type
    assert isinstance(argument_type, TypeApplication)
    assert argument_type.arguments == (second.ref,)


def test_effect_operation_signatures_do_not_depend_on_declaration_order() -> None:
    family = """family Token[E : Effect] : Type
    constructor Keep : Token[E]
"""
    first = """effect First
    inspect : Token[Second] -> Unit
"""
    second = """effect Second
    inspect : Token[First] -> Unit
"""

    forward = lower_qvr_to_qiec(parse(f"{family}\n{first}{second}"))
    reverse = lower_qvr_to_qiec(parse(f"{family}\n{second}{first}"))

    def signatures(module: QiecModule):
        return {effect.ref.name: effect.operations[0] for effect in module.effects}

    assert signatures(forward) == signatures(reverse)


def test_lowering_route_is_exact_and_negotiated_before_elaboration() -> None:
    parsed = parse(DECLARATIONS)

    with pytest.raises(
        UnsupportedLoweringRouteError, match="unsupported lowering route"
    ):
        lower_qvr_to_qiec(parsed, target_version="qiec-core/v2")


def test_unknown_effect_instance_is_a_source_located_diagnostic() -> None:
    parsed = parse(
        """\
define invalid() : Unit !{} =
    let ignored <- perform missing.fire()
    return unit
""",
        "invalid.qvr",
    )

    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parsed, file_path="invalid.qvr")

    assert captured.value.code == "qiec-unhandled-effect"
    assert captured.value.file == "invalid.qvr"
    assert captured.value.line == 2


def test_projection_detects_qiec_without_reclassifying_categorical_declarations() -> (
    None
):
    qiec = parse(DECLARATIONS)
    categorical = parse("object X : Real 1\n")

    assert has_qiec_surface(qiec)
    assert not has_qiec_surface(categorical)


def test_case_branch_static_binders_scope_annotations_and_lower_to_skolems() -> None:
    parsed = parse(
        DECLARATIONS
        + """
effect Inspect
    probe[n : Nat] : Unit -> Int
instance inspector : Inspect
define annotated_probe[n : Nat](xs : Vec[Int](S(n))) : Int !{inspector} =
    case xs motive (k : Nat) => Int
        Cons[tail_index](head, tail : Vec[Int](tail_index)) =>
            perform inspector.probe[tail_index]()
""",
        "annotated-probe.qvr",
    )

    module = lower_qvr_to_qiec(parsed, file_path="annotated-probe.qvr")
    body = module.computations[0].body

    assert isinstance(body, Case)
    assert body.branches[0].static_arguments[0].identity is not None
    assert body.branches[0].static_arguments[0].name == "m"


def test_case_branch_static_binder_typos_are_source_located() -> None:
    parsed = parse(
        DECLARATIONS
        + """
define typo[A : Type, n : Nat](xs : Vec[A](S(n))) : A !{} =
    case xs motive (k : Nat) => A
        Cons[tail_index](head, tail : Vec[A](tail_indxe)) =>
            return head
""",
        "branch-typo.qvr",
    )

    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parsed, file_path="branch-typo.qvr")

    assert captured.value.code == "qiec-index"
    assert captured.value.file == "branch-typo.qvr"
    assert captured.value.line > 0


def test_case_motive_binds_indices_in_dependent_result_types() -> None:
    parsed = parse(
        DECLARATIONS
        + """
define rebuild[A : Type, n : Nat](xs : Vec[A](S(n))) : Vec[A](S(n)) !{} =
    case xs motive (k : Nat) => Vec[A](k)
        Cons[tail_index](head, tail : Vec[A](tail_index)) =>
            return construct Cons[A, tail_index](head, tail) as Vec[A](S(tail_index))
""",
        "dependent-motive.qvr",
    )

    module = lower_qvr_to_qiec(parsed, file_path="dependent-motive.qvr")

    assert isinstance(module.computations[0].body, Case)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            """effect E [version=1, version=2]
    op : Unit -> Unit
""",
            "duplicate effect option(s): version",
        ),
        (
            """effect E
    op : Unit -> Unit
handler h for E : Unit -> Unit [coverage=total, coverage=partial]
    op resumes 0
""",
            "duplicate handler option(s): coverage",
        ),
    ],
)
def test_duplicate_qiec_options_are_source_located(source: str, message: str) -> None:
    parsed = parse(source, "duplicates.qvr")

    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parsed, file_path="duplicates.qvr")

    assert message in captured.value.message
    assert captured.value.file == "duplicates.qvr"
    assert captured.value.line > 0


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("index I = Z | Z\n", "qiec-index"),
        (
            """family F[A : Type, A : Type] : Type
    constructor Mk : F[A, A]
""",
            "qiec-kind",
        ),
        (
            """effect E
    op : Unit -> Unit
handler h for E : Unit -> Unit [coverage=partial]
    op resumes 0
    op resumes 1
""",
            "qiec-handler",
        ),
        (
            """effect E
    op : Unit -> Unit
instance x : E
define f() : Unit !{| rho lacks x, x} =
    return unit
""",
            "qiec-row",
        ),
        (
            """index Nat = Z | S(Nat)
family Vec[A : Type](n : Nat) : Type
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))
define duplicate_fields[n : Nat](xs : Vec[Int](S(n))) : Int !{} =
    case xs motive (k : Nat) => Int
        Cons[m](item, item) =>
            return item
""",
            "qiec-kind",
        ),
    ],
)
def test_malformed_qiec_declarations_keep_source_locations(
    source: str, code: str
) -> None:
    parsed = parse(source, "malformed.qvr")

    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parsed, file_path="malformed.qvr")

    assert captured.value.code == code
    assert captured.value.file == "malformed.qvr"
    assert captured.value.line > 0


def test_signed_numeric_literals_lower_with_their_intrinsic_types() -> None:
    parsed = parse(
        """define negative_int() : Int !{} =
    return -1
define negative_real() : Real !{} =
    return -1.5
"""
    )

    module = lower_qvr_to_qiec(parsed)
    values = []
    for computation in module.computations:
        assert isinstance(computation.body, Return)
        assert isinstance(computation.body.value, LiteralValue)
        values.append(computation.body.value.value)
    assert values == [-1, -1.5]


def test_open_row_lacks_must_be_proved_by_the_inferred_tail() -> None:
    parsed = parse(
        """effect State[S : Type] [version=1, evolution=sealed]
    get : Unit -> S
instance cell : State[Int]
handler h for State[Int] : Int -> Int [coverage=total, introduces=!{| sigma}]
    get resumes 1
define f() : Int !{| rho lacks cell} =
    handle cell with h in
        perform cell.get()
""",
        "rows.qvr",
    )

    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parsed, file_path="rows.qvr")

    assert captured.value.code == "qiec-row"
    assert captured.value.file == "rows.qvr"
    assert captured.value.line == 6


def test_type_telescope_binders_shadow_builtin_type_names() -> None:
    parsed = parse(
        """family Shadow[Int : Type] : Type
    constructor Box : Int -> Shadow[Int]
""",
        "shadow.qvr",
    )

    module = lower_qvr_to_qiec(parsed, file_path="shadow.qvr")

    field_type = module.constructors[0].fields[0].type
    result_argument = module.constructors[0].result_indices
    assert isinstance(field_type, TypeVariable)
    assert field_type.name == "Int"
    assert result_argument == ()


def test_effect_binder_accepts_a_fully_applied_concrete_effect() -> None:
    parsed = parse(
        """effect Inner[T : Type]
    get : Unit -> T
effect Outer[F : Effect]
    lift : Unit -> Unit
instance concrete : Outer[Inner[Int]]
""",
        "nested-effects.qvr",
    )

    module = lower_qvr_to_qiec(parsed, file_path="nested-effects.qvr")

    nested = module.instances[0].entry.effect.arguments[0]
    assert isinstance(nested, EffectRef)
    assert nested.name == "Inner"
    assert nested.arguments == (INT,)


def test_applied_effect_arguments_check_the_inner_telescope_exactly() -> None:
    parsed = parse(
        """effect Inner[T : Type]
    get : Unit -> T
effect Outer[F : Effect]
    lift : Unit -> Unit
instance bad : Outer[Inner[Int, String]]
""",
        "bad-nested-effects.qvr",
    )

    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parsed, file_path="bad-nested-effects.qvr")

    assert captured.value.code == "qiec-kind"
    assert captured.value.line == 5
    assert "effect 'Inner' expects 1 arguments" in captured.value.message


def test_compiler_preserves_checked_qiec_projection() -> None:
    parsed = parse(DECLARATIONS + "\n" + COMPUTATIONS)
    compiler = Compiler(parsed, module_name="compiled")

    assert compiler.qiec_module is not None
    assert compiler.compile_env()["__qiec__"] is compiler.qiec_module

    program = compile_source(DECLARATIONS + "\n" + COMPUTATIONS)
    assert program.qiec is not None
    assert [item.name for item in program.qiec.computations] == [
        "swap",
        "singleton",
        "head",
        "handled",
    ]
