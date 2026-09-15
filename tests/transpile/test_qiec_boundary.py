"""Conformance tests for the shared QIEC-to-transpiler boundary."""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from quivers.transpile.lower import Lower
from quivers.transpile.qiec_ir import (
    IRQiecModule,
    IRQiecReturn,
    analyze_qiec_capabilities,
)


_OLD_PROGRAM = """\
object Resp : FinSet 4
program prog : Resp -> Resp
    sample x <- Bernoulli(0.5)
    return x
export prog
"""

_OLD_PROGRAM_WITH_QIEC_METADATA = "index Nat = Z | S(Nat)\n\n" + _OLD_PROGRAM

_QIEC_COMPUTATION = """\
define answer() : Int !{} =
    return 42

object Resp : FinSet 4
program prog : Resp -> Resp
    sample x <- Bernoulli(0.5)
    return x
export prog
"""

_ILL_TYPED_QIEC_METADATA = """\
effect Broken
    ask : Unit -> Missing

object Resp : FinSet 4
program prog : Resp -> Resp
    sample x <- Bernoulli(0.5)
    return x
export prog
"""

_EFFECTFUL_QIEC = """\
effect State[S : Type]
    get : Unit -> S

instance cell : State[Int]

define read() : Int !{cell} =
    perform cell.get()
"""

_PARAMETERIZED_QIEC = """\
define identity(value : Int) : Int !{} =
    return value
"""


@pytest.mark.parametrize("target", available_targets())
def test_checked_qiec_metadata_can_accompany_old_program(target: str) -> None:
    """Static QIEC declarations preserve the accompanying program."""
    output = transpile(parse(_OLD_PROGRAM_WITH_QIEC_METADATA), target=target)

    assert output
    assert b"x" in output
    if target in {"bugs", "jags"}:
        # These graph languages reject an empty declaration-only model. The
        # neutral sentinel also keeps a mixed model syntactically uniform.
        assert b"qiec_declarations" in output


def test_qiec_computation_enters_typed_ir_without_blanket_refusal() -> None:
    ir = Lower().forward(parse(_QIEC_COMPUTATION))

    assert ir.qiec is not None
    assert [item.name for item in ir.qiec.computations] == ["prog", "answer"]
    computation = ir.qiec.computations[1]
    assert computation.name == "answer"
    assert isinstance(computation.body, IRQiecReturn)
    assert analyze_qiec_capabilities(ir.qiec, "pyro") == ()
    assert analyze_qiec_capabilities(ir.qiec, "stan") == ()


@pytest.mark.parametrize("target", available_targets())
def test_invalid_qiec_metadata_fails_exact_check_before_render(target: str) -> None:
    """A mixed module cannot use metadata treatment to bypass QIEC checking."""
    with pytest.raises(QiecDiagnosticError) as exc_info:
        transpile(parse(_ILL_TYPED_QIEC_METADATA), target=target)

    assert exc_info.value.code == "qiec-kind"
    assert "unknown or unsaturated type 'Missing'" in exc_info.value.message


def test_qiec_declarations_lower_without_a_probabilistic_program() -> None:
    ir = Lower().forward(parse("index Nat = Z | S(Nat)\n"))

    assert ir.body == ()
    assert ir.inputs == ()
    assert ir.qiec is not None
    assert [sort.name for sort in ir.qiec.index_sorts] == ["Nat"]


def test_checked_qiec_metadata_is_preserved_in_structural_ir() -> None:
    ir = Lower().forward(parse(_OLD_PROGRAM_WITH_QIEC_METADATA))

    assert ir.qiec is not None
    assert isinstance(ir.qiec, IRQiecModule)
    assert [sort.name for sort in ir.qiec.index_sorts] == ["Nat"]


def test_typed_qiec_ir_round_trips_through_didactic_json() -> None:
    ir = Lower().forward(parse(_EFFECTFUL_QIEC, "state.qvr"))

    assert type(ir).model_validate_json(ir.model_dump_json()) == ir


def test_capability_analysis_is_target_specific_and_source_located() -> None:
    qiec = Lower().forward(parse(_EFFECTFUL_QIEC, "state.qvr")).qiec

    assert qiec is not None
    assert analyze_qiec_capabilities(qiec, "pyro") == ()
    diagnostics = analyze_qiec_capabilities(qiec, "stan")
    assert {diagnostic.feature for diagnostic in diagnostics} == {
        "effectful-row",
        "perform",
    }
    assert all(diagnostic.code == "qiec-capability" for diagnostic in diagnostics)
    assert all(diagnostic.computation == "read" for diagnostic in diagnostics)
    assert all(diagnostic.origin is not None for diagnostic in diagnostics)
    assert all(
        diagnostic.origin.line == 6 for diagnostic in diagnostics if diagnostic.origin
    )
    message = str(
        UnsupportedConstruct(
            "qvr-stan", [diagnostic.kind for diagnostic in diagnostics]
        )
    )
    assert "QIEC computation `read`" in message
    assert "`perform` QIEC capability" in message
    assert "silently erasing it would change the program" in message


def test_graphical_targets_refuse_named_qiec_parameters() -> None:
    qiec = Lower().forward(parse(_PARAMETERIZED_QIEC)).qiec

    assert qiec is not None
    assert analyze_qiec_capabilities(qiec, "stan") == ()
    for target in ("bugs", "jags"):
        diagnostics = analyze_qiec_capabilities(qiec, target)
        assert [diagnostic.kind for diagnostic in diagnostics] == [
            "qiec:capability:named-parameter:identity"
        ]
