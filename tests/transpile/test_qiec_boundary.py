"""Conformance tests for the shared QIEC-to-transpiler boundary."""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from quivers.transpile.lower import Lower
from quivers.qiec import QiecModule, loads as load_qiec


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
effect Broken [version=1, evolution=sealed]
    ask : Unit -> Missing

object Resp : FinSet 4
program prog : Resp -> Resp
    sample x <- Bernoulli(0.5)
    return x
export prog
"""


@pytest.mark.parametrize("target", available_targets())
def test_checked_qiec_metadata_can_accompany_old_program(target: str) -> None:
    """Static QIEC declarations reach the checker, then the old program emits."""
    output = transpile(parse(_OLD_PROGRAM_WITH_QIEC_METADATA), target=target)

    assert output == transpile(parse(_OLD_PROGRAM), target=target)


@pytest.mark.parametrize("target", available_targets())
def test_qiec_computation_is_refused_before_old_ir(target: str) -> None:
    """Every backend reports one common refusal instead of erasing the body."""
    with pytest.raises(UnsupportedConstruct) as exc_info:
        transpile(parse(_QIEC_COMPUTATION), target=target)

    assert exc_info.value.target == f"qvr-{target}"
    assert exc_info.value.kinds == ["qiec:computation-body:answer"]
    message = str(exc_info.value)
    assert "has passed QIEC type and effect checking" in message
    assert "existing probabilistic IR" in message
    assert "removing the body would change the program's meaning" in message


@pytest.mark.parametrize("target", available_targets())
def test_invalid_qiec_metadata_fails_exact_check_before_render(target: str) -> None:
    """A mixed module cannot use metadata treatment to bypass QIEC checking."""
    with pytest.raises(QiecDiagnosticError) as exc_info:
        transpile(parse(_ILL_TYPED_QIEC_METADATA), target=target)

    assert exc_info.value.code == "qiec-kind"
    assert "unknown or unsaturated type 'Missing'" in exc_info.value.message


def test_qiec_declarations_alone_are_not_a_probabilistic_program() -> None:
    """Metadata treatment applies only when an old ``program`` is present."""
    with pytest.raises(UnsupportedConstruct) as exc_info:
        transpile(parse("index Nat = Z | S(Nat)\n"), target="stan")

    assert exc_info.value.kinds == ["index_decl"]


def test_checked_qiec_metadata_is_preserved_in_structural_ir() -> None:
    ir = Lower().forward(parse(_OLD_PROGRAM_WITH_QIEC_METADATA))

    assert ir.qiec is not None
    qiec = load_qiec(ir.qiec)
    assert isinstance(qiec, QiecModule)
    assert [sort.name for sort in qiec.index_sorts] == ["Nat"]
