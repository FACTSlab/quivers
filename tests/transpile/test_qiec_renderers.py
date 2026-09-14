"""Executable and refusal conformance for QIEC-capable renderers."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess

import pytest

from quivers.dsl import parse
from quivers.transpile import (
    _RENDERERS,
    UnsupportedConstruct,
    available_targets,
    transpile,
)
from quivers.transpile._pipeline import parser_registry
from quivers.transpile.lower import Lower
from quivers.transpile.qiec_ir import (
    IRQiecAttachmentRef,
    IRQiecBind,
    IRQiecComputationType,
    IRQiecEqualityType,
    IRQiecEvidenceValue,
    IRQiecId,
    IRQiecLocal,
    IRQiecReflexivity,
    IRQiecReturn,
    IRQiecTransportValue,
    IRQiecTypeKind,
    IRQiecVar,
)
from quivers.transpile.renderers._qiec import _ir_data, _julia_data, _scheme_data


DYNAMIC_TARGETS = (
    "pyro",
    "numpyro",
    "pymc",
    "edward2",
    "turing",
    "gen",
    "webppl",
    "church",
)
STATIC_TARGETS = ("stan", "bugs", "jags")
SCHEME_EXECUTABLE = next(
    (
        executable
        for name in ("scheme", "chez", "petite", "chezscheme")
        if (executable := shutil.which(name)) is not None
    ),
    None,
)

PURE = """\
define answer() : Int !{} =
    return 42
"""

PARAMETER = """\
define identity(value : Int) : Int !{} =
    return value
"""

KEYWORD_PARAMETERS = """\
define collision(class : Int, qiec_handlers : Int, end : Int) : Int !{} =
    return class
"""

STRING_RESULT = """\
define label() : String !{} =
    return "ok"
"""

DECLARATION_ONLY = "index Nat = Z | S(Nat)\n"

EFFECTFUL = """\
effect Ask
    get : Unit -> Int
instance ask : Ask
define read() : Int !{ask} =
    let value <- perform ask.get()
    return value
"""

HANDLED = """\
effect Ask
    get : Unit -> Int
instance ask : Ask
handler answer for Ask : Int -> Int [coverage=total]
    get resumes 1
define read() : Int !{} =
    handle ask with answer in
        let value <- perform ask.get()
        return value
"""

MULTISHOT = """\
effect Choose
    pick : Unit -> Int
effect Trace
    record : Int -> Unit
instance choice : Choose
instance trace : Trace
handler branch for Choose : Int -> Int [coverage=total]
    pick resumes omega
define explored() : Int !{trace} =
    handle choice with branch in
        let selected <- perform choice.pick()
        perform trace.record(selected)
        return selected
"""

MULTISHOT_CAPTURE = MULTISHOT.replace(
    "define explored() : Int !{trace} =",
    "define explored(seed : Int) : Int !{trace} =",
).replace("perform trace.record(selected)", "perform trace.record(seed)")

NESTED_STATE_OMEGA = """\
effect State
    get : Unit -> Int
    put : Int -> Unit
effect Choose
    pick : Unit -> Int
effect Trace
    record : Int -> Unit
instance cell : State
instance choice : Choose
instance trace : Trace
handler run_state for State : Int -> Int [coverage=total]
    get resumes 1
    put resumes 1
handler branch for Choose : Int -> Int [coverage=total]
    pick resumes omega
define explored() : Int !{trace} =
    handle cell with run_state in
        handle choice with branch in
            let selected <- perform choice.pick()
            let before <- perform cell.get()
            perform trace.record(before)
            perform cell.put(selected)
            return selected
"""

PARTIAL_FORWARDING = """\
effect Forwarded
    a : Unit -> Int
    b : Unit -> Int
instance forwarded : Forwarded
handler inner for Forwarded : Int -> Int [coverage=partial]
    a resumes 1
handler outer for Forwarded : Int -> Int [coverage=total]
    a resumes 1
    b resumes 1
define forwarded_result() : Int !{} =
    handle forwarded with outer in
        handle forwarded with inner in
            let value <- perform forwarded.b()
            return value
"""

GENERIC_REQUEST = """\
effect Reflect
    echo[A : Type] : A -> A
instance reflect : Reflect
define reflect_int(value : Int) : Int !{reflect} =
    let result <- perform reflect.echo[Int](value)
    return result
"""

PARAMETERIZED_HANDLER = """\
effect Read[A : Type]
    get : Unit -> A
instance read_int : Read[Int]
handler run_read[A : Type] for Read[A] : A -> A [coverage=total]
    get resumes 1
define read_specialized() : Int !{} =
    handle read_int with run_read[Int] in
        let value <- perform read_int.get()
        return value
"""

INDEXED = """\
index Nat = Z | S(Nat)
family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))
define singleton[A : Type](value : A) : Vec[A](S(Z)) !{} =
    return construct Cons[A, Z](value, construct Nil[A]() as Vec[A](Z)) as Vec[A](S(Z))
define head[A : Type, n : Nat](xs : Vec[A](S(n))) : A !{} =
    case xs motive (m : Nat) => A
        Cons[n](head, tail) =>
            return head
"""

ZERO_INDEX_BRANCH = """\
index Nat = Z | S(Nat)
family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
define make_empty[A : Type]() : Vec[A](Z) !{} =
    return construct Nil[A]() as Vec[A](Z)
define empty[A : Type](xs : Vec[A](Z)) : Int !{} =
    case xs motive (m : Nat) => Int
        Nil() =>
            return 1
"""

EXISTENTIAL_REQUEST = """\
family Box : Type
    constructor Pack[X : Type] : X -> Box
effect Tag
    tag[X : Type] : Unit -> String
instance tagger : Tag
define pack[A : Type](value : A) : Box !{} =
    return construct Pack[A](value) as Box
define tag_box(box : Box) : String !{tagger} =
    case box motive => String
        Pack[X](value) =>
            let result <- perform tagger.tag[X]()
            return result
"""

MIXED = (
    PURE
    + """\
object Resp : FinSet 4
program model_program : Resp -> Resp
    sample coin <- Bernoulli(0.5)
    return coin
export model_program
"""
)


def _grammar(target: str) -> str:
    return {
        "pyro": "python",
        "numpyro": "python",
        "pymc": "python",
        "edward2": "python",
        "turing": "julia",
        "gen": "julia",
        "webppl": "javascript",
        "church": "scheme",
        "stan": "stan",
        "bugs": "bugs",
        "jags": "jags",
    }[target]


def _render_ir(ir, target: str) -> bytes:
    renderer, grammar, _ = _RENDERERS[target]
    return bytes(parser_registry().emit_pretty(grammar, renderer().render(ir)))


def _derived_id(base: IRQiecId, suffix: str) -> IRQiecId:
    """A distinct identifier for a computation derived from another.

    These tests build extra computations out of one the pipeline
    produced. Each is a separate declaration, so each needs its own
    identity: two declarations sharing one would make the module's
    signature table lossy and a call to either ambiguous.

    Parameters
    ----------
    base : IRQiecId
        The identifier to derive from.
    suffix : str
        What distinguishes the new declaration from the base.

    Returns
    -------
    IRQiecId
        An identifier in the same namespace with a digest derived from
        the base digest and the suffix, so it is stable across runs and
        distinct from every other derivation.
    """
    digest = hashlib.sha256(f"{base.digest}:{suffix}".encode()).hexdigest()
    return IRQiecId(namespace=base.namespace, digest=digest)


@pytest.mark.parametrize("target", available_targets())
def test_pure_named_computation_emits_and_reparses(target: str) -> None:
    output = transpile(parse(PURE), target=target)

    assert b"qiec_answer" in output
    reparsed = parser_registry().parse_with_protocol(
        _grammar(target), output, f"answer.{_grammar(target)}"
    )
    assert reparsed.vertices
    assert bytes(parser_registry().emit_pretty(_grammar(target), reparsed)) == output
    if _grammar(target) == "python":
        ast.parse(output)


@pytest.mark.parametrize("target", DYNAMIC_TARGETS)
def test_dynamic_targets_emit_full_qiec_runtime(target: str) -> None:
    output = transpile(parse(HANDLED + "\n" + INDEXED), target=target)

    assert b"qiec_read" in output
    assert b"qiec_singleton" in output
    assert b"qiec_head" in output
    assert b"qiec:handler:" in output
    assert b"qiec:constructor:" in output


@pytest.mark.parametrize("target", available_targets())
def test_mixed_module_preserves_model_abi_and_named_qiec_entrypoint(
    target: str,
) -> None:
    output = transpile(parse(MIXED), target=target)

    assert b"qiec_answer" in output
    assert b"model" in output


@pytest.mark.parametrize("target", ("bugs", "jags"))
def test_dataflow_targets_emit_valid_declaration_only_noop(target: str) -> None:
    output = transpile(parse(DECLARATION_ONLY), target=target)

    assert b"qiec_declarations <- 0" in output


def test_stan_emits_closed_monomorphic_scalar_parameters() -> None:
    output = transpile(parse(PARAMETER), target="stan")

    assert b"int qiec_identity(int qv_value)" in output
    assert b"return qv_value;" in output


@pytest.mark.parametrize("target", ("bugs", "jags"))
def test_dataflow_targets_refuse_named_computation_parameters(target: str) -> None:
    with pytest.raises(UnsupportedConstruct) as captured:
        transpile(parse(PARAMETER), target=target)

    assert captured.value.kinds == ["qiec:capability:named-parameter:identity"]


@pytest.mark.parametrize("target", ("stan",))
def test_static_targets_lower_pure_bind_ir(target: str) -> None:
    ir = Lower().forward(parse(PARAMETER))
    assert ir.qiec is not None
    computation = ir.qiec.computations[0]
    parameter = computation.parameters[0]
    binder = IRQiecLocal(name="bound", type=parameter.type)
    body = IRQiecBind(
        binder=binder,
        first=IRQiecReturn(value=IRQiecVar(local=parameter)),
        then=IRQiecReturn(value=IRQiecVar(local=binder)),
    )
    bound_computation = type(computation)(
        id=computation.id,
        name=computation.name,
        telescope=computation.telescope,
        parameters=computation.parameters,
        body=body,
        type=computation.type,
        origin=computation.origin,
    )
    qiec = type(ir.qiec)(
        module=ir.qiec.module,
        source_protocol=ir.qiec.source_protocol,
        index_sorts=ir.qiec.index_sorts,
        families=ir.qiec.families,
        constructors=ir.qiec.constructors,
        effects=ir.qiec.effects,
        instances=ir.qiec.instances,
        handlers=ir.qiec.handlers,
        computations=(bound_computation,),
        abi=ir.qiec.abi,
    )
    ir = type(ir)(
        name=ir.name,
        inputs=ir.inputs,
        body=ir.body,
        cards=ir.cards,
        qiec=qiec,
    )
    renderer, grammar, _ = _RENDERERS[target]
    output = bytes(parser_registry().emit_pretty(grammar, renderer().render(ir)))

    assert b"int qv_bound = qv_value;" in output
    assert b"return qv_bound;" in output


@pytest.mark.parametrize("target", (*DYNAMIC_TARGETS, "stan"))
def test_authored_target_keywords_and_abi_names_are_hygienic(target: str) -> None:
    output = transpile(parse(KEYWORD_PARAMETERS), target=target)

    assert b"qv_qiec_handlers" in output
    if _grammar(target) == "python":
        compile(output, f"collision.{target}.py", "exec")


@pytest.mark.parametrize("target", STATIC_TARGETS)
def test_static_targets_refuse_effects_with_precise_capability_kinds(
    target: str,
) -> None:
    with pytest.raises(UnsupportedConstruct) as captured:
        transpile(parse(EFFECTFUL), target=target)

    assert captured.value.target == f"qvr-{target}"
    assert set(captured.value.kinds) == {
        "qiec:capability:effectful-row:read",
        "qiec:capability:perform:read",
    }


@pytest.mark.parametrize("target", STATIC_TARGETS)
@pytest.mark.parametrize(
    ("grade", "feature"),
    (
        ("0", "zero-resumption"),
        ("aff", "affine-resumption"),
        ("1", "linear-resumption"),
        ("omega", "unrestricted-resumption"),
    ),
)
def test_static_targets_refuse_handlers_and_each_resumption_grade_precisely(
    target: str,
    grade: str,
    feature: str,
) -> None:
    source = HANDLED.replace("resumes 1", f"resumes {grade}")

    with pytest.raises(UnsupportedConstruct) as captured:
        transpile(parse(source), target=target)

    assert captured.value.target == f"qvr-{target}"
    assert set(captured.value.kinds) == {
        "qiec:capability:handle:read",
        "qiec:capability:perform:read",
        f"qiec:capability:{feature}:read",
    }


@pytest.mark.parametrize("target", STATIC_TARGETS)
def test_static_targets_refuse_indexed_values_without_erasure(target: str) -> None:
    with pytest.raises(UnsupportedConstruct) as captured:
        transpile(parse(INDEXED), target=target)

    assert "qiec:capability:structured-value:singleton" in captured.value.kinds
    assert "qiec:capability:case:head" in captured.value.kinds
    assert "qiec:capability:index-polymorphism:head" in captured.value.kinds


@pytest.mark.parametrize("target", STATIC_TARGETS)
def test_static_targets_refuse_string_results_as_non_scalar(target: str) -> None:
    with pytest.raises(UnsupportedConstruct) as captured:
        transpile(parse(STRING_RESULT), target=target)

    assert captured.value.kinds == ["qiec:capability:non-scalar-result:label"]


@pytest.mark.parametrize("target", ("pyro", "pymc", "edward2"))
def test_generated_python_qiec_functions_execute(target: str) -> None:
    namespace: dict[str, object] = {}
    exec(transpile(parse(PURE + "\n" + PARAMETER), target=target), namespace)

    assert namespace["qiec_answer"]() == 42  # type: ignore[operator]
    assert namespace["qiec_identity"](17) == 17  # type: ignore[operator]


def test_generated_python_operation_and_handler_abis_use_stable_ids() -> None:
    effect_ir = Lower().forward(parse(EFFECTFUL)).qiec
    assert effect_ir is not None
    request = effect_ir.computations[0].body.first.request  # type: ignore[union-attr]
    namespace: dict[str, object] = {}
    exec(transpile(parse(EFFECTFUL), target="pyro"), namespace)

    seen_request: dict[str, object] = {}

    def operation(runtime_request):
        seen_request.update(runtime_request)
        return 29

    operations = {(request.instance.text, request.operation.text): operation}
    assert (
        namespace["qiec_read"](  # type: ignore[operator]
            qiec_operations=operations
        )
        == 29
    )
    assert seen_request["effect"]["id"] == request.effect.id.text  # type: ignore[index]
    assert seen_request["result_type"]["kind"] == "type-application"  # type: ignore[index]
    assert "origin" in seen_request

    handled_ir = Lower().forward(parse(HANDLED)).qiec
    assert handled_ir is not None
    handler = handled_ir.handlers[0]
    operation = handler.clauses[0].operation.text
    namespace = {}
    exec(transpile(parse(HANDLED), target="pyro"), namespace)
    seen_context: dict[str, object] = {}

    def clause(runtime_request, resume, context):
        seen_context.update(context)
        assert runtime_request["operation"] == operation
        return resume(31)

    handlers = {handler.id.text: {"operations": {operation: clause}}}
    assert namespace["qiec_read"](qiec_handlers=handlers) == 31  # type: ignore[operator]
    assert seen_context["definition"]["input_type"]["kind"] == "type-application"  # type: ignore[index]
    assert seen_context["static_arguments"] == []

    with pytest.raises(RuntimeError, match="linear clause must resume exactly once"):
        namespace["qiec_read"](  # type: ignore[operator]
            qiec_handlers={
                handler.id.text: {
                    "operations": {
                        operation: lambda runtime_request, resume, context: 31
                    }
                }
            }
        )

    def resume_twice(runtime_request, resume, context):
        resume(1)
        return resume(2)

    with pytest.raises(RuntimeError, match="resumption exceeds grade 1"):
        namespace["qiec_read"](  # type: ignore[operator]
            qiec_handlers={handler.id.text: {"operations": {operation: resume_twice}}}
        )


def test_generated_python_request_preserves_operation_static_arguments() -> None:
    qiec = Lower().forward(parse(GENERIC_REQUEST)).qiec
    assert qiec is not None
    request = qiec.computations[0].body.first.request  # type: ignore[union-attr]
    seen: dict[str, object] = {}
    namespace: dict[str, object] = {}
    exec(transpile(parse(GENERIC_REQUEST), target="pyro"), namespace)

    def echo(runtime_request):
        seen.update(runtime_request)
        return runtime_request["arguments"][0]

    assert (
        namespace["qiec_reflect_int"](  # type: ignore[operator]
            37,
            qiec_operations={(request.instance.text, request.operation.text): echo},
        )
        == 37
    )
    assert seen["static_arguments"][0]["kind"] == "type-application"  # type: ignore[index]


def test_parameterized_handler_manifest_is_specialized_before_attachment() -> None:
    qiec = Lower().forward(parse(PARAMETERIZED_HANDLER)).qiec
    assert qiec is not None
    handler = qiec.handlers[0]
    operation = handler.clauses[0].operation.text
    seen: dict[str, object] = {}
    namespace: dict[str, object] = {}
    exec(transpile(parse(PARAMETERIZED_HANDLER), target="pyro"), namespace)

    def clause(request, resume, context):
        seen.update(context)
        return resume(61)

    assert (
        namespace["qiec_read_specialized"](  # type: ignore[operator]
            qiec_handlers={handler.id.text: {"operations": {operation: clause}}}
        )
        == 61
    )
    definition = seen["definition"]
    assert definition["telescope"] == []  # type: ignore[index]
    assert definition["effect"]["arguments"][0]["constructor"]["name"] == "Int"  # type: ignore[index]


def test_handler_validators_apply_at_resume_and_final_answer_boundaries() -> None:
    qiec = Lower().forward(parse(HANDLED)).qiec
    assert qiec is not None
    handler = qiec.handlers[0]
    operation = handler.clauses[0].operation.text
    namespace: dict[str, object] = {}
    exec(transpile(parse(HANDLED), target="pyro"), namespace)

    with pytest.raises(TypeError, match="handler operation result type"):
        namespace["qiec_read"](  # type: ignore[operator]
            qiec_handlers={
                handler.id.text: {
                    "operations": {
                        operation: {
                            "invoke": (lambda request, resume, context: resume("bad")),
                            "result_validator": lambda value: isinstance(value, int),
                        }
                    }
                }
            }
        )

    with pytest.raises(TypeError, match="handler output type"):
        namespace["qiec_read"](  # type: ignore[operator]
            qiec_handlers={
                handler.id.text: {
                    "operations": {
                        operation: lambda request, resume, context: (
                            resume(1),
                            "bad",
                        )[1]
                    },
                    "output_validator": lambda value: isinstance(value, int),
                }
            }
        )


def test_handler_lifecycle_is_exact_once_on_exit_and_drop() -> None:
    qiec = Lower().forward(parse(HANDLED)).qiec
    assert qiec is not None
    handler = qiec.handlers[0]
    operation = handler.clauses[0].operation.text
    namespace: dict[str, object] = {}
    exec(transpile(parse(HANDLED), target="pyro"), namespace)

    events: list[str] = []

    def installed(*, fail: bool = False):
        def clause(request, resume, context):
            if fail:
                raise RuntimeError("clause failed")
            return resume(67)

        return {
            "operations": {operation: clause},
            "mutable_context": True,
            "on_enter": lambda: events.append("enter"),
            "on_exit": lambda: events.append("exit"),
            "on_drop": lambda: events.append("drop"),
        }

    prototype = {
        "operations": {operation: lambda request, resume, context: resume(0)},
        "mutable_context": True,
        "context_factory": installed,
    }
    assert (
        namespace["qiec_read"](  # type: ignore[operator]
            qiec_handlers={handler.id.text: prototype}
        )
        == 67
    )
    assert events == ["enter", "exit"]

    events.clear()
    prototype["context_factory"] = lambda: installed(fail=True)
    with pytest.raises(RuntimeError, match="clause failed"):
        namespace["qiec_read"](  # type: ignore[operator]
            qiec_handlers={handler.id.text: prototype}
        )
    assert events == ["enter", "drop"]


def test_partial_handler_forwards_known_missing_clause_to_outer_handler() -> None:
    qiec = Lower().forward(parse(PARTIAL_FORWARDING)).qiec
    assert qiec is not None
    inner, outer = qiec.handlers
    a, b = (clause.operation.text for clause in outer.clauses)
    namespace: dict[str, object] = {}
    exec(transpile(parse(PARTIAL_FORWARDING), target="pyro"), namespace)

    handlers = {
        inner.id.text: {
            "operations": {a: lambda runtime_request, resume, context: resume(1)}
        },
        outer.id.text: {
            "operations": {
                a: lambda runtime_request, resume, context: resume(1),
                b: lambda runtime_request, resume, context: resume(47),
            }
        },
    }

    assert namespace["qiec_forwarded_result"](qiec_handlers=handlers) == 47  # type: ignore[operator]


def test_nested_forwarded_handler_lifecycles_drop_once_on_outer_failure() -> None:
    qiec = Lower().forward(parse(PARTIAL_FORWARDING)).qiec
    assert qiec is not None
    inner, outer = qiec.handlers
    a, b = (clause.operation.text for clause in outer.clauses)
    namespace: dict[str, object] = {}
    exec(transpile(parse(PARTIAL_FORWARDING), target="pyro"), namespace)
    events: list[str] = []

    def installed(label: str, operations):
        return {
            "operations": operations,
            "mutable_context": True,
            "on_enter": lambda: events.append("enter:" + label),
            "on_exit": lambda: events.append("exit:" + label),
            "on_drop": lambda: events.append("drop:" + label),
        }

    def fail(request, resume, context):
        raise RuntimeError("outer failed")

    inner_operations = {a: lambda request, resume, context: resume(1)}
    outer_operations = {
        a: lambda request, resume, context: resume(1),
        b: fail,
    }
    handlers = {
        inner.id.text: {
            "operations": inner_operations,
            "mutable_context": True,
            "context_factory": lambda: installed("inner", inner_operations),
        },
        outer.id.text: {
            "operations": outer_operations,
            "mutable_context": True,
            "context_factory": lambda: installed("outer", outer_operations),
        },
    }

    with pytest.raises(RuntimeError, match="outer failed"):
        namespace["qiec_forwarded_result"](qiec_handlers=handlers)  # type: ignore[operator]
    assert events.count("drop:inner") == 1
    assert events.count("drop:outer") == 1
    assert not any(event.startswith("exit:") for event in events)


def test_generated_python_gadt_constructor_and_case_execute() -> None:
    parameter_ir = Lower().forward(parse(PARAMETER)).qiec
    assert parameter_ir is not None
    int_type = parameter_ir.computations[0].parameters[0].type.model_dump()
    namespace: dict[str, object] = {}
    exec(transpile(parse(INDEXED), target="pyro"), namespace)

    with pytest.raises(TypeError, match="static argument arity"):
        namespace["qiec_singleton"](37)  # type: ignore[operator]
    singleton = namespace["qiec_singleton"](  # type: ignore[operator]
        37, qiec_static_arguments=[int_type]
    )
    assert singleton["static_arguments"]
    assert singleton["result_type"]["arguments"]
    zero = singleton["static_arguments"][1]
    assert (
        namespace["qiec_head"](  # type: ignore[operator]
            singleton, qiec_static_arguments=[int_type, zero]
        )
        == 37
    )


def test_generated_python_zero_index_branch_ignores_uniform_statics() -> None:
    parameter_ir = Lower().forward(parse(PARAMETER)).qiec
    assert parameter_ir is not None
    int_type = _ir_data(parameter_ir.computations[0].parameters[0].type)
    namespace: dict[str, object] = {}
    exec(transpile(parse(ZERO_INDEX_BRANCH), target="pyro"), namespace)

    empty = namespace["qiec_make_empty"](  # type: ignore[operator]
        qiec_static_arguments=[int_type]
    )
    assert (
        namespace["qiec_empty"](  # type: ignore[operator]
            empty, qiec_static_arguments=[int_type]
        )
        == 1
    )


def test_existential_case_specializes_branch_effect_request() -> None:
    parameter_ir = Lower().forward(parse(PARAMETER)).qiec
    assert parameter_ir is not None
    int_type = _ir_data(parameter_ir.computations[0].parameters[0].type)
    qiec = Lower().forward(parse(EXISTENTIAL_REQUEST)).qiec
    assert qiec is not None
    request = qiec.computations[1].body.branches[0].body.first.request  # type: ignore[union-attr]
    namespace: dict[str, object] = {}
    exec(transpile(parse(EXISTENTIAL_REQUEST), target="pyro"), namespace)
    observed: list[object] = []

    box = namespace["qiec_pack"](  # type: ignore[operator]
        7, qiec_static_arguments=[int_type]
    )
    result = namespace["qiec_tag_box"](  # type: ignore[operator]
        box,
        qiec_operations={
            (request.instance.text, request.operation.text): (
                lambda operation: (
                    observed.extend(operation["static_arguments"]) or "tagged"
                )
            )
        },
    )

    assert result == "tagged"
    assert observed == [int_type]


def _attachment_program_ir():
    ir = Lower().forward(parse(PURE))
    assert ir.qiec is not None
    computation = ir.qiec.computations[0]
    attachment = IRQiecId(namespace="attachment", digest="a" * 64)
    body = IRQiecReturn(
        value=IRQiecAttachmentRef(
            attachment=attachment,
            type=computation.type.result,
        )
    )
    bound_computation = type(computation)(
        id=computation.id,
        name=computation.name,
        telescope=computation.telescope,
        parameters=computation.parameters,
        body=body,
        type=computation.type,
        origin=computation.origin,
    )
    qiec = type(ir.qiec)(
        module=ir.qiec.module,
        source_protocol=ir.qiec.source_protocol,
        index_sorts=ir.qiec.index_sorts,
        families=ir.qiec.families,
        constructors=ir.qiec.constructors,
        effects=ir.qiec.effects,
        instances=ir.qiec.instances,
        handlers=ir.qiec.handlers,
        computations=(bound_computation,),
        abi=ir.qiec.abi,
    )
    ir = type(ir)(
        name=ir.name,
        inputs=ir.inputs,
        body=ir.body,
        cards=ir.cards,
        qiec=qiec,
    )

    return ir, attachment, _ir_data(computation.type.result)


def _evidence_transport_program_ir():
    ir = Lower().forward(parse(PURE))
    assert ir.qiec is not None
    computation = ir.qiec.computations[0]
    assert isinstance(computation.body, IRQiecReturn)
    result_type = computation.type.result
    equality = IRQiecEqualityType(
        classifier=IRQiecTypeKind(),
        left=result_type,
        right=result_type,
    )
    evidence = IRQiecReflexivity(equality=equality)
    evidence_computation = type(computation)(
        id=_derived_id(computation.id, "evidence"),
        name="evidence",
        telescope=computation.telescope,
        parameters=computation.parameters,
        body=IRQiecReturn(value=IRQiecEvidenceValue(evidence=evidence)),
        type=IRQiecComputationType(
            effects=computation.type.effects,
            result=equality,
        ),
        origin=computation.origin,
    )
    transport_computation = type(computation)(
        id=_derived_id(computation.id, "transport"),
        name="transport",
        telescope=computation.telescope,
        parameters=computation.parameters,
        body=IRQiecReturn(
            value=IRQiecTransportValue(
                evidence=evidence,
                value=computation.body.value,
                target_type=result_type,
            )
        ),
        type=computation.type,
        origin=computation.origin,
    )
    qiec = type(ir.qiec)(
        module=ir.qiec.module,
        source_protocol=ir.qiec.source_protocol,
        index_sorts=ir.qiec.index_sorts,
        families=ir.qiec.families,
        constructors=ir.qiec.constructors,
        effects=ir.qiec.effects,
        instances=ir.qiec.instances,
        handlers=ir.qiec.handlers,
        computations=(evidence_computation, transport_computation),
        abi=ir.qiec.abi,
    )
    return type(ir)(
        name=ir.name,
        inputs=ir.inputs,
        body=ir.body,
        cards=ir.cards,
        qiec=qiec,
    )


@pytest.mark.parametrize("target", DYNAMIC_TARGETS)
def test_dynamic_targets_emit_evidence_and_transport_nodes(target: str) -> None:
    output = _render_ir(_evidence_transport_program_ir(), target)

    assert b"qiec_evidence" in output
    assert b"qiec_transport" in output
    assert b"reflexivity" in output


@pytest.mark.parametrize("target", STATIC_TARGETS)
def test_static_targets_refuse_evidence_and_transport_without_erasure(
    target: str,
) -> None:
    with pytest.raises(UnsupportedConstruct) as captured:
        _render_ir(_evidence_transport_program_ir(), target)

    assert "qiec:capability:evidence:evidence" in captured.value.kinds
    assert "qiec:capability:evidence:transport" in captured.value.kinds
    assert "qiec:capability:transport:transport" in captured.value.kinds


def test_generated_python_evidence_and_transport_execute() -> None:
    namespace: dict[str, object] = {}
    exec(_render_ir(_evidence_transport_program_ir(), "pyro"), namespace)

    evidence = namespace["qiec_evidence"]()  # type: ignore[operator]
    assert evidence["qiec"] == "evidence"  # type: ignore[index]
    assert evidence["evidence"]["kind"] == "reflexivity"  # type: ignore[index]
    assert namespace["qiec_transport"]() == 42  # type: ignore[operator]


def test_generated_python_attachment_lookup_uses_stable_id() -> None:
    ir, attachment, expected_type = _attachment_program_ir()
    output = _render_ir(ir, "pyro")
    namespace: dict[str, object] = {}
    exec(output, namespace)

    with pytest.raises(TypeError, match="typed binding descriptor"):
        namespace["qiec_answer"](  # type: ignore[operator]
            qiec_attachments={attachment.text: 53}
        )

    validated: list[int] = []
    binding = {
        "qiec": "binding",
        "value": 53,
        "type": expected_type,
        "validator": lambda value: validated.append(value) or True,
    }
    assert (
        namespace["qiec_answer"](  # type: ignore[operator]
            qiec_attachments={attachment.text: binding}
        )
        == 53
    )
    assert validated == [53]

    binding["type"] = {"kind": "wrong"}
    with pytest.raises(TypeError, match="type disagrees"):
        namespace["qiec_answer"](  # type: ignore[operator]
            qiec_attachments={attachment.text: binding}
        )


def test_generated_python_multishot_resumptions_extend_addresses() -> None:
    qiec = Lower().forward(parse(MULTISHOT)).qiec
    assert qiec is not None
    handler = qiec.handlers[0]
    choose_operation = handler.clauses[0].operation.text
    body = qiec.computations[0].body
    trace_request = body.computation.then.first.request  # type: ignore[union-attr]
    addresses: list[object] = []
    namespace: dict[str, object] = {}
    exec(transpile(parse(MULTISHOT), target="pyro"), namespace)

    def branch(request, resume, context):
        first = resume(1)
        second = resume(2)
        pure = namespace["_qvr_qiec_pure"]
        bind = namespace["_qvr_qiec_bind"]
        return bind(  # type: ignore[operator]
            first,
            lambda left: bind(  # type: ignore[operator]
                second,
                lambda right: pure(left + right),  # type: ignore[operator]
            ),
        )

    result = namespace["qiec_explored"](  # type: ignore[operator]
        qiec_handlers={
            handler.id.text: {
                "operations": {choose_operation: branch},
                "duplicable_context": True,
            }
        },
        qiec_operations={
            (trace_request.instance.text, trace_request.operation.text): (
                lambda request: addresses.append(request["address"])
            )
        },
    )

    assert result == 3
    assert [address[2][-1] for address in addresses] == [0, 1]  # type: ignore[index]


def test_unrestricted_resumption_requires_and_forks_captured_values() -> None:
    qiec = Lower().forward(parse(MULTISHOT_CAPTURE)).qiec
    assert qiec is not None
    handler = qiec.handlers[0]
    choose_operation = handler.clauses[0].operation.text
    body = qiec.computations[0].body
    trace_request = body.computation.then.first.request  # type: ignore[union-attr]
    namespace: dict[str, object] = {}
    exec(transpile(parse(MULTISHOT_CAPTURE), target="pyro"), namespace)

    def branch(request, resume, context):
        first = resume(1)
        second = resume(2)
        return namespace["_qvr_qiec_bind"](  # type: ignore[operator]
            first,
            lambda left: namespace["_qvr_qiec_bind"](  # type: ignore[operator]
                second,
                lambda right: namespace["_qvr_qiec_pure"](left + right),  # type: ignore[operator]
            ),
        )

    handlers = {
        handler.id.text: {
            "operations": {choose_operation: branch},
            "duplicable_context": True,
        }
    }
    operation = (trace_request.instance.text, trace_request.operation.text)
    with pytest.raises(RuntimeError, match="nonduplicable local seed"):
        namespace["qiec_explored"](  # type: ignore[operator]
            object(),
            qiec_handlers=handlers,
            qiec_operations={operation: lambda request: None},
        )

    recorded: list[int] = []
    seed = {
        "qiec": "binding",
        "value": 5,
        "duplicable": True,
        "mutable": True,
        "fork": lambda value: value + 1,
    }
    assert (
        namespace["qiec_explored"](  # type: ignore[operator]
            seed,
            qiec_handlers=handlers,
            qiec_operations={
                operation: lambda request: recorded.append(request["arguments"][0])
            },
        )
        == 3
    )
    assert recorded == [6, 6]


def test_unrestricted_handler_forks_finalize_each_branch_exactly_once() -> None:
    qiec = Lower().forward(parse(MULTISHOT)).qiec
    assert qiec is not None
    handler = qiec.handlers[0]
    choose_operation = handler.clauses[0].operation.text
    body = qiec.computations[0].body
    trace_request = body.computation.then.first.request  # type: ignore[union-attr]
    namespace: dict[str, object] = {}
    exec(transpile(parse(MULTISHOT), target="pyro"), namespace)
    events: list[str] = []

    def branch(request, resume, context):
        first = resume(1)
        second = resume(2)
        return namespace["_qvr_qiec_bind"](  # type: ignore[operator]
            first,
            lambda left: namespace["_qvr_qiec_bind"](  # type: ignore[operator]
                second,
                lambda right: namespace["_qvr_qiec_pure"](left + right),  # type: ignore[operator]
            ),
        )

    def installed(label: str):
        return {
            "operations": {choose_operation: branch},
            "duplicable_context": True,
            "mutable_context": True,
            "fork_context": lambda seed: installed("shot"),
            "on_enter": lambda: events.append("enter:" + label),
            "on_exit": lambda: events.append("exit:" + label),
            "on_drop": lambda: events.append("drop:" + label),
        }

    prototype = installed("prototype")
    prototype["context_factory"] = lambda: installed("root")
    result = namespace["qiec_explored"](  # type: ignore[operator]
        qiec_handlers={handler.id.text: prototype},
        qiec_operations={
            (trace_request.instance.text, trace_request.operation.text): (
                lambda request: None
            )
        },
    )

    assert result == 3
    assert events.count("enter:root") == 1
    assert events.count("exit:root") == 1
    assert events.count("enter:shot") == 2
    assert events.count("exit:shot") == 2
    assert not any(event.startswith("drop:") for event in events)


def test_nested_state_handler_is_isolated_across_unrestricted_shots() -> None:
    qiec = Lower().forward(parse(NESTED_STATE_OMEGA)).qiec
    assert qiec is not None
    state, choose = qiec.handlers
    choose_operation = choose.clauses[0].operation.text
    body = qiec.computations[0].body.computation.computation  # type: ignore[union-attr]
    trace_request = body.then.then.first.request  # type: ignore[union-attr]
    namespace: dict[str, object] = {}
    exec(transpile(parse(NESTED_STATE_OMEGA), target="pyro"), namespace)
    trace: list[int] = []
    events: list[str] = []

    def state_handler(value: int, label: str):
        cell = {"value": value}

        def get(request, resume, context):
            return resume(cell["value"])

        def put(request, resume, context):
            cell["value"] = request["arguments"][0]
            return resume(None)

        return {
            "cell": cell,
            "operations": {
                state.clauses[0].operation.text: get,
                state.clauses[1].operation.text: put,
            },
            "duplicable_context": True,
            "mutable_context": True,
            "fork_context": lambda seed: state_handler(seed["cell"]["value"], "shot"),
            "on_enter": lambda: events.append("enter:" + label),
            "on_exit": lambda: events.append("exit:" + label),
            "on_drop": lambda: events.append("drop:" + label),
        }

    def choose_clause(request, resume, context):
        first = resume(1)
        second = resume(2)
        return namespace["_qvr_qiec_bind"](  # type: ignore[operator]
            first,
            lambda left: namespace["_qvr_qiec_bind"](  # type: ignore[operator]
                second,
                lambda right: namespace["_qvr_qiec_pure"](  # type: ignore[operator]
                    left + right
                ),
            ),
        )

    state_prototype = state_handler(0, "prototype")
    state_prototype["context_factory"] = lambda: state_handler(0, "root")
    handlers = {
        state.id.text: state_prototype,
        choose.id.text: {
            "operations": {choose_operation: choose_clause},
            "duplicable_context": True,
        },
    }
    result = namespace["qiec_explored"](  # type: ignore[operator]
        qiec_handlers=handlers,
        qiec_operations={
            (trace_request.instance.text, trace_request.operation.text): (
                lambda request: trace.append(request["arguments"][0])
            )
        },
    )

    assert result == 3
    assert trace == [0, 0]
    assert events.count("enter:shot") == 2
    assert events.count("exit:shot") == 2
    assert not any(event.startswith("drop:") for event in events)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_generated_webppl_qiec_function_executes_in_javascript(
    tmp_path,
) -> None:
    script = tmp_path / "qiec.js"
    script.write_bytes(
        transpile(parse(PURE), target="webppl") + b"\nconsole.log(qiec_answer());\n"
    )

    completed = subprocess.run(
        ["node", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "42"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_generated_webppl_effect_dispatch_uses_stable_ids(tmp_path) -> None:
    qiec = Lower().forward(parse(EFFECTFUL)).qiec
    assert qiec is not None
    request = qiec.computations[0].body.first.request  # type: ignore[union-attr]
    key = request.instance.text + "|" + request.operation.text
    script = tmp_path / "qiec-effect.js"
    script.write_bytes(
        transpile(parse(EFFECTFUL), target="webppl")
        + (
            "\nvar operations = {};\n"
            f"operations[{json.dumps(key)}] = function(request) {{ return 41; }};\n"
            "console.log(qiec_read(null, {}, {}, operations));\n"
        ).encode()
    )

    completed = subprocess.run(
        ["node", str(script)], check=True, capture_output=True, text=True
    )
    assert completed.stdout.strip() == "41"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_generated_webppl_attachment_is_typed_and_validated(tmp_path) -> None:
    ir, attachment, expected_type = _attachment_program_ir()
    script = tmp_path / "qiec-attachment.js"
    script.write_bytes(
        _render_ir(ir, "webppl")
        + (
            "\nvar validated = []; var attachments = {};\n"
            f"attachments[{json.dumps(attachment.text)}] = {{ qiec: 'binding', value: 53, type: {json.dumps(expected_type)}, validator: function(value) {{ validated.push(value); return true; }} }};\n"
            "console.log(JSON.stringify([qiec_answer(null, attachments, {}, {}), validated]));\n"
        ).encode()
    )

    completed = subprocess.run(
        ["node", str(script)], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == [53, [53]]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_generated_webppl_evidence_and_transport_execute(tmp_path) -> None:
    script = tmp_path / "qiec-evidence.js"
    script.write_bytes(
        _render_ir(_evidence_transport_program_ir(), "webppl")
        + b"\nconsole.log(JSON.stringify([qiec_evidence().qiec, qiec_evidence().evidence.kind, qiec_transport()]));\n"
    )

    completed = subprocess.run(
        ["node", str(script)], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == ["evidence", "reflexivity", 42]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_generated_webppl_linear_grade_and_handler_specialization(tmp_path) -> None:
    linear_qiec = Lower().forward(parse(HANDLED)).qiec
    parameterized_qiec = Lower().forward(parse(PARAMETERIZED_HANDLER)).qiec
    assert linear_qiec is not None and parameterized_qiec is not None
    linear_handler = linear_qiec.handlers[0]
    parameterized_handler = parameterized_qiec.handlers[0]
    script = tmp_path / "qiec-handler-contracts.js"
    script.write_bytes(
        transpile(parse(HANDLED + "\n" + PARAMETERIZED_HANDLER), target="webppl")
        + (
            "\nvar linearOperations = {}; var linearHandlers = {};\n"
            f"linearOperations[{json.dumps(linear_handler.clauses[0].operation.text)}] = function(request, resume, context) {{ resume(1); return resume(2); }};\n"
            f"linearHandlers[{json.dumps(linear_handler.id.text)}] = {{ operations: linearOperations }};\n"
            "var grade = null; try { qiec_read(null, {}, linearHandlers, {}); } catch (failure) { grade = failure.message; }\n"
            "var seen = null; var specializedOperations = {}; var specializedHandlers = {};\n"
            f"specializedOperations[{json.dumps(parameterized_handler.clauses[0].operation.text)}] = function(request, resume, context) {{ seen = context.definition; return resume(61); }};\n"
            f"specializedHandlers[{json.dumps(parameterized_handler.id.text)}] = {{ operations: specializedOperations }};\n"
            "var result = qiec_read_specialized(null, {}, specializedHandlers, {});\n"
            "console.log(JSON.stringify([grade, result, seen.telescope, seen.effect.arguments[0].constructor.name]));\n"
        ).encode()
    )

    completed = subprocess.run(
        ["node", str(script)], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == [
        "QIEC resumption exceeds grade 1",
        61,
        [],
        "Int",
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_generated_webppl_handler_forwarding_validation_and_lifecycle(tmp_path) -> None:
    qiec = Lower().forward(parse(PARTIAL_FORWARDING)).qiec
    assert qiec is not None
    inner, outer = qiec.handlers
    operations = {
        handler.name: [clause.operation.text for clause in handler.clauses]
        for handler in qiec.handlers
    }
    script = tmp_path / "qiec-handlers.js"
    script.write_bytes(
        transpile(parse(PARTIAL_FORWARDING), target="webppl")
        + (
            "\nvar events = [];\n"
            "var handlers = {};\n"
            "var innerOperations = {};\n"
            f"innerOperations[{json.dumps(operations['inner'][0])}] = function(request, resume, context) {{ return resume(11); }};\n"
            "var outerOperations = {};\n"
            f"outerOperations[{json.dumps(operations['outer'][0])}] = function(request, resume, context) {{ return resume(13); }};\n"
            f"outerOperations[{json.dumps(operations['outer'][1])}] = {{ invoke: function(request, resume, context) {{ return resume(19); }}, result_validator: function(value) {{ return typeof value === 'number'; }} }};\n"
            "var installed = function(label, operations) { return { operations: operations, mutable_context: true, output_validator: function(value) { return typeof value === 'number'; }, on_enter: function() { events.push('enter:' + label); }, on_exit: function() { events.push('exit:' + label); }, on_drop: function() { events.push('drop:' + label); } }; };\n"
            "var innerPrototype = installed('prototype-inner', innerOperations);\n"
            "innerPrototype.context_factory = function() { return installed('inner', innerOperations); };\n"
            "var outerPrototype = installed('prototype-outer', outerOperations);\n"
            "outerPrototype.context_factory = function() { return installed('outer', outerOperations); };\n"
            f"handlers[{json.dumps(inner.id.text)}] = innerPrototype;\n"
            f"handlers[{json.dumps(outer.id.text)}] = outerPrototype;\n"
            "console.log(JSON.stringify([qiec_forwarded_result(null, {}, handlers, {}), events]));\n"
        ).encode()
    )

    completed = subprocess.run(
        ["node", str(script)], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == [
        19,
        ["enter:outer", "enter:inner", "exit:inner", "exit:outer"],
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_generated_webppl_existential_case_specializes_request(tmp_path) -> None:
    parameter_ir = Lower().forward(parse(PARAMETER)).qiec
    qiec = Lower().forward(parse(EXISTENTIAL_REQUEST)).qiec
    assert parameter_ir is not None and qiec is not None
    int_type = _ir_data(parameter_ir.computations[0].parameters[0].type)
    request = qiec.computations[1].body.branches[0].body.first.request  # type: ignore[union-attr]
    key = request.instance.text + "|" + request.operation.text
    script = tmp_path / "qiec-existential.js"
    script.write_bytes(
        transpile(parse(EXISTENTIAL_REQUEST), target="webppl")
        + (
            "\nvar seen = null; var operations = {};\n"
            f"operations[{json.dumps(key)}] = function(request) {{ seen = request.static_arguments; return 'tagged'; }};\n"
            f"var box = qiec_pack(7, [{json.dumps(int_type)}], {{}}, {{}}, {{}});\n"
            "console.log(JSON.stringify([qiec_tag_box(box, null, {}, {}, operations), seen]));\n"
        ).encode()
    )

    completed = subprocess.run(
        ["node", str(script)], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == ["tagged", [int_type]]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_generated_webppl_unrestricted_resumption_isolates_shots(tmp_path) -> None:
    qiec = Lower().forward(parse(MULTISHOT)).qiec
    assert qiec is not None
    handler = qiec.handlers[0]
    operation = handler.clauses[0].operation.text
    trace = qiec.computations[0].body.computation.then.first.request  # type: ignore[union-attr]
    key = trace.instance.text + "|" + trace.operation.text
    script = tmp_path / "qiec-omega.js"
    script.write_bytes(
        transpile(parse(MULTISHOT), target="webppl")
        + (
            "\nvar addresses = []; var handlers = {}; var clauses = {}; var operations = {};\n"
            f"clauses[{json.dumps(operation)}] = function(request, resume, context) {{ var first = resume(1); var second = resume(2); return _qvr_qiec_bind(first, function(left) {{ return _qvr_qiec_bind(second, function(right) {{ return _qvr_qiec_pure(left + right); }}); }}); }};\n"
            f"handlers[{json.dumps(handler.id.text)}] = {{ operations: clauses, duplicable_context: true }};\n"
            f"operations[{json.dumps(key)}] = function(request) {{ addresses.push(request.address); }};\n"
            "var result = qiec_explored(null, {}, handlers, operations);\n"
            "console.log(JSON.stringify([result, addresses.map(function(address) { return address[2][address[2].length - 1]; })]));\n"
        ).encode()
    )

    completed = subprocess.run(
        ["node", str(script)], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == [3, [0, 1]]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_generated_webppl_nested_mutable_state_is_branch_local(tmp_path) -> None:
    qiec = Lower().forward(parse(NESTED_STATE_OMEGA)).qiec
    assert qiec is not None
    state, choose = qiec.handlers
    trace_request = qiec.computations[
        0
    ].body.computation.computation.then.then.first.request  # type: ignore[union-attr]
    key = trace_request.instance.text + "|" + trace_request.operation.text
    script = tmp_path / "qiec-nested-state.js"
    script.write_bytes(
        transpile(parse(NESTED_STATE_OMEGA), target="webppl")
        + (
            "\nvar trace = []; var events = []; var handlers = {}; var hostOperations = {};\n"
            "var stateHandler = function(value, label) { var cell = { value: value }; var operations = {}; var handler = { cell: cell, operations: operations, duplicable_context: true, mutable_context: true, on_enter: function() { events.push('enter:' + label); }, on_exit: function() { events.push('exit:' + label); }, on_drop: function() { events.push('drop:' + label); } };\n"
            f"operations[{json.dumps(state.clauses[0].operation.text)}] = function(request, resume, context) {{ return resume(cell.value); }};\n"
            f"operations[{json.dumps(state.clauses[1].operation.text)}] = function(request, resume, context) {{ cell.value = request.arguments[0]; return resume(null); }};\n"
            "handler.fork_context = function(seed) { return stateHandler(seed.cell.value, 'shot'); }; return handler; };\n"
            "var statePrototype = stateHandler(0, 'prototype'); statePrototype.context_factory = function() { return stateHandler(0, 'root'); };\n"
            f"handlers[{json.dumps(state.id.text)}] = statePrototype;\n"
            "var chooseOperations = {};\n"
            f"chooseOperations[{json.dumps(choose.clauses[0].operation.text)}] = function(request, resume, context) {{ var first = resume(1); var second = resume(2); return _qvr_qiec_bind(first, function(left) {{ return _qvr_qiec_bind(second, function(right) {{ return _qvr_qiec_pure(left + right); }}); }}); }};\n"
            f"handlers[{json.dumps(choose.id.text)}] = {{ operations: chooseOperations, duplicable_context: true }};\n"
            f"hostOperations[{json.dumps(key)}] = function(request) {{ trace.push(request.arguments[0]); }};\n"
            "console.log(JSON.stringify([qiec_explored(null, {}, handlers, hostOperations), trace, events.filter(function(event) { return event.indexOf(':shot') >= 0; })]));\n"
        ).encode()
    )

    completed = subprocess.run(
        ["node", str(script)], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == [
        3,
        [0, 0],
        ["enter:shot", "enter:shot", "exit:shot", "exit:shot"],
    ]


@pytest.mark.skipif(SCHEME_EXECUTABLE is None, reason="Chez Scheme is unavailable")
def test_generated_church_qiec_function_executes_in_chez(tmp_path) -> None:
    script = tmp_path / "qiec.scm"
    script.write_bytes(
        transpile(parse(PURE), target="church")
        + b"\n(display (qiec_answer))\n(newline)\n"
    )

    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "42"


@pytest.mark.skipif(SCHEME_EXECUTABLE is None, reason="Chez Scheme is unavailable")
def test_generated_church_effect_dispatch_uses_structural_request(tmp_path) -> None:
    qiec = Lower().forward(parse(EFFECTFUL)).qiec
    assert qiec is not None
    request = qiec.computations[0].body.first.request  # type: ignore[union-attr]
    script = tmp_path / "qiec-effect.scm"
    operation_key = (
        f"(cons {json.dumps(request.instance.text)} "
        f"{json.dumps(request.operation.text)})"
    )
    script.write_bytes(
        transpile(parse(EFFECTFUL), target="church")
        + (
            "\n(define operations "
            f"(list (cons {operation_key} (lambda (request) 45))))\n"
            "(display (qiec_read #f '() '() operations))\n(newline)\n"
        ).encode()
    )

    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "45"


@pytest.mark.skipif(SCHEME_EXECUTABLE is None, reason="Chez Scheme is unavailable")
def test_generated_church_attachment_is_typed_and_validated(tmp_path) -> None:
    ir, attachment, expected_type = _attachment_program_ir()
    script = tmp_path / "qiec-attachment.scm"
    script.write_bytes(
        _render_ir(ir, "church")
        + (
            "\n(define validated '())\n"
            '(define binding (list (cons "qiec" "binding") (cons "cell" (vector 53)) '
            f'(cons "type" {_scheme_data(expected_type)}) '
            '(cons "validator" (lambda (value) (set! validated (append validated (list value))) #t))))\n'
            f"(define attachments (list (cons {json.dumps(attachment.text)} binding)))\n"
            "(write (list (qiec_answer #f attachments '() '()) validated))\n"
        ).encode()
    )

    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "(53 (53))"


@pytest.mark.skipif(SCHEME_EXECUTABLE is None, reason="Chez Scheme is unavailable")
def test_generated_church_evidence_and_transport_execute(tmp_path) -> None:
    script = tmp_path / "qiec-evidence.scm"
    script.write_bytes(
        _render_ir(_evidence_transport_program_ir(), "church")
        + b'\n(write (list (_qvr-qiec-get (qiec_evidence) "qiec") (_qvr-qiec-get (_qvr-qiec-get (qiec_evidence) "evidence") "kind") (qiec_transport)))\n'
    )

    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == '("evidence" "reflexivity" 42)'


@pytest.mark.skipif(SCHEME_EXECUTABLE is None, reason="Chez Scheme is unavailable")
def test_generated_church_linear_grade_and_handler_specialization(tmp_path) -> None:
    linear_qiec = Lower().forward(parse(HANDLED)).qiec
    parameterized_qiec = Lower().forward(parse(PARAMETERIZED_HANDLER)).qiec
    assert linear_qiec is not None and parameterized_qiec is not None
    linear_handler = linear_qiec.handlers[0]
    parameterized_handler = parameterized_qiec.handlers[0]
    script = tmp_path / "qiec-handler-contracts.scm"
    script.write_bytes(
        transpile(parse(HANDLED + "\n" + PARAMETERIZED_HANDLER), target="church")
        + (
            "\n(define linear-operations (list "
            f"(cons {json.dumps(linear_handler.clauses[0].operation.text)} (lambda (request resume context) (resume 1) (resume 2)))))\n"
            "(define linear-handlers (list "
            f'(cons {json.dumps(linear_handler.id.text)} (list (cons "operations" linear-operations)))))\n'
            "(define grade (guard (condition (else (condition-message condition))) (qiec_read #f '() linear-handlers '())))\n"
            "(define seen #f)\n"
            "(define specialized-operations (list "
            f'(cons {json.dumps(parameterized_handler.clauses[0].operation.text)} (lambda (request resume context) (set! seen (_qvr-qiec-get context "definition")) (resume 61)))))\n'
            "(define specialized-handlers (list "
            f'(cons {json.dumps(parameterized_handler.id.text)} (list (cons "operations" specialized-operations)))))\n'
            "(define result (qiec_read_specialized #f '() specialized-handlers '()))\n"
            '(define effect-argument (car (_qvr-qiec-get (_qvr-qiec-get seen "effect") "arguments")))\n'
            '(write (list grade result (_qvr-qiec-get seen "telescope") (_qvr-qiec-get (_qvr-qiec-get effect-argument "constructor") "name")))\n'
        ).encode()
    )

    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == (
        '("QIEC resumption exceeds grade 1" 61 () "Int")'
    )


@pytest.mark.skipif(SCHEME_EXECUTABLE is None, reason="Chez Scheme is unavailable")
def test_generated_church_existential_case_specializes_request(tmp_path) -> None:
    parameter_ir = Lower().forward(parse(PARAMETER)).qiec
    qiec = Lower().forward(parse(EXISTENTIAL_REQUEST)).qiec
    assert parameter_ir is not None and qiec is not None
    int_type = _ir_data(parameter_ir.computations[0].parameters[0].type)
    request = qiec.computations[1].body.branches[0].body.first.request  # type: ignore[union-attr]
    operation_key = (
        f"(cons {json.dumps(request.instance.text)} "
        f"{json.dumps(request.operation.text)})"
    )
    script = tmp_path / "qiec-existential.scm"
    script.write_bytes(
        transpile(parse(EXISTENTIAL_REQUEST), target="church")
        + (
            "\n(define seen '())\n"
            "(define operations (list (cons "
            + operation_key
            + ' (lambda (request) (set! seen (_qvr-qiec-get request "static_arguments")) "tagged"))))\n'
            f"(define box (qiec_pack 7 (list {_scheme_data(int_type)}) '() '() '()))\n"
            f"(write (list (qiec_tag_box box #f '() '() operations) (equal? seen (list {_scheme_data(int_type)}))))\n"
        ).encode()
    )

    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == '("tagged" #t)'


@pytest.mark.skipif(SCHEME_EXECUTABLE is None, reason="Chez Scheme is unavailable")
def test_generated_church_partial_handler_forwards_to_outer_handler(tmp_path) -> None:
    qiec = Lower().forward(parse(PARTIAL_FORWARDING)).qiec
    assert qiec is not None
    inner, outer = qiec.handlers
    script = tmp_path / "qiec-partial.scm"
    script.write_bytes(
        transpile(parse(PARTIAL_FORWARDING), target="church")
        + (
            "\n(define inner-operations (list "
            f"(cons {json.dumps(inner.clauses[0].operation.text)} (lambda (request resume context) (resume 11)))))\n"
            "(define outer-operations (list "
            + " ".join(
                f"(cons {json.dumps(clause.operation.text)} (lambda (request resume context) (resume 19)))"
                for clause in outer.clauses
            )
            + "))\n"
            "(define handlers (list "
            f'(cons {json.dumps(inner.id.text)} (list (cons "operations" inner-operations))) '
            f'(cons {json.dumps(outer.id.text)} (list (cons "operations" outer-operations)))))\n'
            "(write (qiec_forwarded_result #f '() handlers '()))\n"
        ).encode()
    )

    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "19"


@pytest.mark.skipif(SCHEME_EXECUTABLE is None, reason="Chez Scheme is unavailable")
def test_generated_church_nested_mutable_state_is_branch_local(tmp_path) -> None:
    qiec = Lower().forward(parse(NESTED_STATE_OMEGA)).qiec
    assert qiec is not None
    state, choose = qiec.handlers
    trace_request = qiec.computations[
        0
    ].body.computation.computation.then.then.first.request  # type: ignore[union-attr]
    operation_key = (
        f"(cons {json.dumps(trace_request.instance.text)} "
        f"{json.dumps(trace_request.operation.text)})"
    )
    script = tmp_path / "qiec-nested-state.scm"
    script.write_bytes(
        transpile(parse(NESTED_STATE_OMEGA), target="church")
        + (
            "\n(define trace '()) (define events '()) (define context-counts '())\n"
            "(define (state-handler value label)\n"
            "  (let* ((cell (vector value))\n"
            "         (operations (list\n"
            f"           (cons {json.dumps(state.clauses[0].operation.text)} (lambda (request resume context) (resume (vector-ref cell 0))))\n"
            f'           (cons {json.dumps(state.clauses[1].operation.text)} (lambda (request resume context) (vector-set! cell 0 (car (_qvr-qiec-get request "arguments"))) (resume \'())))))\n'
            '         (handler (list (cons "cell" cell) (cons "operations" operations) (cons "duplicable_context" #t) (cons "mutable_context" #t) (cons "on_enter" (lambda () (set! events (append events (list (string-append "enter:" label)))))) (cons "on_exit" (lambda () (set! events (append events (list (string-append "exit:" label)))))) (cons "on_drop" (lambda () (set! events (append events (list (string-append "drop:" label)))))))))\n'
            '    (_qvr-qiec-set handler "fork_context" (lambda (seed) (state-handler (vector-ref (_qvr-qiec-get seed "cell") 0) "shot")))))\n'
            '(define state-prototype (_qvr-qiec-set (state-handler 0 "prototype") "context_factory" (lambda () (state-handler 0 "root"))))\n'
            "(define choose-operations (list\n"
            f"  (cons {json.dumps(choose.clauses[0].operation.text)}\n"
            "    (lambda (request resume context)\n"
            '      (set! context-counts (append context-counts (list (_qvr-qiec-get context "resumption_uses"))))\n'
            "      (let ((first (resume 1)))\n"
            '        (set! context-counts (append context-counts (list (_qvr-qiec-get context "resumption_uses"))))\n'
            "        (let ((second (resume 2)))\n"
            '          (set! context-counts (append context-counts (list (_qvr-qiec-get context "resumption_uses"))))\n'
            "          (_qvr-qiec-bind first (lambda (left) (_qvr-qiec-bind second (lambda (right) (_qvr-qiec-pure (+ left right))))))))))))\n"
            "(define handlers (list\n"
            f"  (cons {json.dumps(state.id.text)} state-prototype)\n"
            f'  (cons {json.dumps(choose.id.text)} (list (cons "operations" choose-operations) (cons "duplicable_context" #t)))))\n'
            "(define host-operations (list (cons "
            + operation_key
            + ' (lambda (request) (set! trace (append trace (list (car (_qvr-qiec-get request "arguments"))))) \'()))))\n'
            "(define result (qiec_explored #f '() handlers host-operations))\n"
            '(write (list result trace (filter (lambda (event) (or (equal? event "enter:shot") (equal? event "exit:shot") (equal? event "drop:shot"))) events) context-counts))\n'
        ).encode()
    )

    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == (
        '(3 (0 0) ("enter:shot" "enter:shot" "exit:shot" "exit:shot") (0 1 2))'
    )


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
@pytest.mark.parametrize(("target", "macro"), (("turing", "model"), ("gen", "gen")))
def test_generated_julia_qiec_function_executes(
    target: str,
    macro: str,
    tmp_path,
) -> None:
    script = tmp_path / f"qiec-{target}.jl"
    script.write_text(
        f"macro {macro}(expression)\n    esc(expression)\nend\n"
        + transpile(parse(PURE), target=target).decode()
        + "\nprintln(qiec_answer())\n"
    )

    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "42"


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
@pytest.mark.parametrize(("target", "macro"), (("turing", "model"), ("gen", "gen")))
def test_generated_julia_effect_dispatch_uses_stable_ids(
    target: str,
    macro: str,
    tmp_path,
) -> None:
    qiec = Lower().forward(parse(EFFECTFUL)).qiec
    assert qiec is not None
    request = qiec.computations[0].body.first.request  # type: ignore[union-attr]
    script = tmp_path / f"qiec-effect-{target}.jl"
    script.write_text(
        f"macro {macro}(expression)\n    esc(expression)\nend\n"
        + transpile(parse(EFFECTFUL), target=target).decode()
        + "\noperations = Dict("
        + f"({json.dumps(request.instance.text)}, {json.dumps(request.operation.text)})"
        + " => (request -> 43))\n"
        + "println(qiec_read(qiec_operations=operations))\n"
    )

    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "43"


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
@pytest.mark.parametrize(("target", "macro"), (("turing", "model"), ("gen", "gen")))
def test_generated_julia_attachment_is_typed_and_validated(
    target: str,
    macro: str,
    tmp_path,
) -> None:
    ir, attachment, expected_type = _attachment_program_ir()
    script = tmp_path / f"qiec-attachment-{target}.jl"
    script.write_text(
        f"macro {macro}(expression)\n    esc(expression)\nend\n"
        + _render_ir(ir, target).decode()
        + "\nvalidated = Any[]; attachments = Dict("
        + f'{_julia_data(attachment.text)} => Dict("qiec" => "binding", "value" => 53, "type" => {_julia_data(expected_type)}, "validator" => (value -> begin push!(validated, value); true end)));\n'
        + "println((qiec_answer(qiec_attachments=attachments), validated));\n"
    )

    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "(53, Any[53])"


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
@pytest.mark.parametrize(("target", "macro"), (("turing", "model"), ("gen", "gen")))
def test_generated_julia_evidence_and_transport_execute(
    target: str,
    macro: str,
    tmp_path,
) -> None:
    script = tmp_path / f"qiec-evidence-{target}.jl"
    script.write_text(
        f"macro {macro}(expression)\n    esc(expression)\nend\n"
        + _render_ir(_evidence_transport_program_ir(), target).decode()
        + '\nevidence = qiec_evidence(); println((evidence["qiec"], evidence["evidence"]["kind"], qiec_transport()))\n'
    )

    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == '("evidence", "reflexivity", 42)'


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
@pytest.mark.parametrize(("target", "macro"), (("turing", "model"), ("gen", "gen")))
def test_generated_julia_indexed_constructor_and_case_execute(
    target: str,
    macro: str,
    tmp_path,
) -> None:
    parameter_ir = Lower().forward(parse(PARAMETER)).qiec
    assert parameter_ir is not None
    int_type = _ir_data(parameter_ir.computations[0].parameters[0].type)
    script = tmp_path / f"qiec-indexed-{target}.jl"
    script.write_text(
        f"macro {macro}(expression)\n    esc(expression)\nend\n"
        + transpile(parse(INDEXED), target=target).decode()
        + f"\nint_type = {_julia_data(int_type)}\n"
        + "singleton = qiec_singleton(37; qiec_static_arguments=Any[int_type])\n"
        + 'zero = singleton["static_arguments"][2]\n'
        + "println(qiec_head(singleton; qiec_static_arguments=Any[int_type, zero]))\n"
    )

    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "37"


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
@pytest.mark.parametrize(("target", "macro"), (("turing", "model"), ("gen", "gen")))
def test_generated_julia_existential_case_specializes_request(
    target: str,
    macro: str,
    tmp_path,
) -> None:
    parameter_ir = Lower().forward(parse(PARAMETER)).qiec
    qiec = Lower().forward(parse(EXISTENTIAL_REQUEST)).qiec
    assert parameter_ir is not None and qiec is not None
    int_type = _ir_data(parameter_ir.computations[0].parameters[0].type)
    request = qiec.computations[1].body.branches[0].body.first.request  # type: ignore[union-attr]
    script = tmp_path / f"qiec-existential-{target}.jl"
    script.write_text(
        f"macro {macro}(expression)\n    esc(expression)\nend\n"
        + transpile(parse(EXISTENTIAL_REQUEST), target=target).decode()
        + f"\nint_type = {_julia_data(int_type)}; seen = Ref{{Any}}(nothing); operations = Dict();\n"
        + f'operations[({_julia_data(request.instance.text)}, {_julia_data(request.operation.text)})] = request -> begin seen[] = request["static_arguments"]; "tagged" end;\n'
        + "box = qiec_pack(7; qiec_static_arguments=Any[int_type]);\n"
        + "println(qiec_tag_box(box; qiec_operations=operations)); println(seen[] == Any[int_type]);\n"
    )

    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip().splitlines() == ["tagged", "true"]


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
@pytest.mark.parametrize(("target", "macro"), (("turing", "model"), ("gen", "gen")))
def test_generated_julia_linear_grade_and_handler_specialization(
    target: str,
    macro: str,
    tmp_path,
) -> None:
    linear_qiec = Lower().forward(parse(HANDLED)).qiec
    parameterized_qiec = Lower().forward(parse(PARAMETERIZED_HANDLER)).qiec
    assert linear_qiec is not None and parameterized_qiec is not None
    linear_handler = linear_qiec.handlers[0]
    parameterized_handler = parameterized_qiec.handlers[0]
    script = tmp_path / f"qiec-handler-contracts-{target}.jl"
    script.write_text(
        f"macro {macro}(expression)\n    esc(expression)\nend\n"
        + transpile(
            parse(HANDLED + "\n" + PARAMETERIZED_HANDLER), target=target
        ).decode()
        + "\nlinear_operations = Dict(); linear_handlers = Dict();\n"
        + f"linear_operations[{_julia_data(linear_handler.clauses[0].operation.text)}] = (request, resume, context) -> begin resume(1); resume(2) end;\n"
        + f'linear_handlers[{_julia_data(linear_handler.id.text)}] = Dict("operations" => linear_operations);\n'
        + "grade = try qiec_read(qiec_handlers=linear_handlers); nothing catch failure sprint(showerror, failure) end;\n"
        + "seen = Ref{Any}(nothing); specialized_operations = Dict(); specialized_handlers = Dict();\n"
        + f'specialized_operations[{_julia_data(parameterized_handler.clauses[0].operation.text)}] = (request, resume, context) -> begin seen[] = context["definition"]; resume(61) end;\n'
        + f'specialized_handlers[{_julia_data(parameterized_handler.id.text)}] = Dict("operations" => specialized_operations);\n'
        + 'result = qiec_read_specialized(qiec_handlers=specialized_handlers); effect_argument = seen[]["effect"]["arguments"][1];\n'
        + 'println(grade); println(result); println(length(seen[]["telescope"])); println(effect_argument["constructor"]["name"]);\n'
    )

    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip().splitlines() == [
        "QIEC resumption exceeds grade 1",
        "61",
        "0",
        "Int",
    ]


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
@pytest.mark.parametrize(("target", "macro"), (("turing", "model"), ("gen", "gen")))
def test_generated_julia_partial_handler_forwards_to_outer_handler(
    target: str,
    macro: str,
    tmp_path,
) -> None:
    qiec = Lower().forward(parse(PARTIAL_FORWARDING)).qiec
    assert qiec is not None
    inner, outer = qiec.handlers
    script = tmp_path / f"qiec-partial-{target}.jl"
    script.write_text(
        f"macro {macro}(expression)\n    esc(expression)\nend\n"
        + transpile(parse(PARTIAL_FORWARDING), target=target).decode()
        + "\ninner_operations = Dict("
        + f"{_julia_data(inner.clauses[0].operation.text)} => ((request, resume, context) -> resume(11)));\n"
        + "outer_operations = Dict("
        + ", ".join(
            f"{_julia_data(clause.operation.text)} => ((request, resume, context) -> resume(19))"
            for clause in outer.clauses
        )
        + ");\nhandlers = Dict("
        + f'{_julia_data(inner.id.text)} => Dict("operations" => inner_operations), '
        + f'{_julia_data(outer.id.text)} => Dict("operations" => outer_operations));\n'
        + "println(qiec_forwarded_result(qiec_handlers=handlers));\n"
    )

    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "19"


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
@pytest.mark.parametrize(("target", "macro"), (("turing", "model"), ("gen", "gen")))
def test_generated_julia_nested_mutable_state_is_branch_local(
    target: str,
    macro: str,
    tmp_path,
) -> None:
    qiec = Lower().forward(parse(NESTED_STATE_OMEGA)).qiec
    assert qiec is not None
    state, choose = qiec.handlers
    trace_request = qiec.computations[
        0
    ].body.computation.computation.then.then.first.request  # type: ignore[union-attr]
    script = tmp_path / f"qiec-nested-state-{target}.jl"
    script.write_text(
        f"macro {macro}(expression)\n    esc(expression)\nend\n"
        + transpile(parse(NESTED_STATE_OMEGA), target=target).decode()
        + "\ntrace = Int[]; events = String[]; handlers = Dict(); host_operations = Dict();\n"
        + "function state_handler(value, label)\n"
        + '    cell = Dict("value" => value); operations = Dict();\n'
        + f'    operations[{_julia_data(state.clauses[0].operation.text)}] = (request, resume, context) -> resume(cell["value"]);\n'
        + f'    operations[{_julia_data(state.clauses[1].operation.text)}] = (request, resume, context) -> begin cell["value"] = request["arguments"][1]; resume(nothing) end;\n'
        + '    handler = Dict("cell" => cell, "operations" => operations, "duplicable_context" => true, "mutable_context" => true, "on_enter" => (() -> push!(events, "enter:" * label)), "on_exit" => (() -> push!(events, "exit:" * label)), "on_drop" => (() -> push!(events, "drop:" * label)));\n'
        + '    handler["fork_context"] = seed -> state_handler(seed["cell"]["value"], "shot"); handler\n'
        + "end\n"
        + 'state_prototype = state_handler(0, "prototype"); state_prototype["context_factory"] = () -> state_handler(0, "root");\n'
        + f"handlers[{_julia_data(state.id.text)}] = state_prototype;\n"
        + "choose_operations = Dict();\n"
        + f"choose_operations[{_julia_data(choose.clauses[0].operation.text)}] = (request, resume, context) -> begin first = resume(1); second = resume(2); _qvr_qiec_bind(first, left -> _qvr_qiec_bind(second, right -> _qvr_qiec_pure(left + right))) end;\n"
        + f'handlers[{_julia_data(choose.id.text)}] = Dict("operations" => choose_operations, "duplicable_context" => true);\n'
        + f'host_operations[({_julia_data(trace_request.instance.text)}, {_julia_data(trace_request.operation.text)})] = request -> push!(trace, request["arguments"][1]);\n'
        + "println(qiec_explored(qiec_handlers=handlers, qiec_operations=host_operations));\n"
        + 'println(join(trace, ","));\n'
        + 'println(join(filter(event -> occursin(":shot", event), events), ","));\n'
    )

    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip().splitlines() == [
        "3",
        "0,0",
        "enter:shot,enter:shot,exit:shot,exit:shot",
    ]
