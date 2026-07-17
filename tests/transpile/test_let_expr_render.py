"""Rendering of `let`-expression arithmetic and builtin calls.

Two properties are pinned per backend, built from programmatic
[`Module`][quivers.dsl.ast_nodes.module.Module] values (no parser
round-trip):

- A nested arithmetic operand keeps its grouping. ``(a + b) * c`` must
  not print as ``a + b * c`` (which reassociates under the target's
  precedence). Every backend that has a grouping construct emits the
  parentheses; the Stan and Julia renderers already did, and the
  Python / BUGS / JavaScript renderers are checked here too.
- A [`LetExprCall`][quivers.dsl.ast_nodes.let_expressions.LetExprCall]
  to a math builtin lowers to the target's own symbol (``erf`` ->
  ``torch.erf`` / ``jax.scipy.special.erf`` / ``pymc.math.erf`` /
  ``tf.math.erf``), with NumPyro emitting the import its symbol needs.
  A builtin with no mapping for a target raises rather than emit an
  undefined name.
"""

from __future__ import annotations

import quivers.transpile as _transpile
from quivers.dsl.compiler.programs import _LET_EXPR_BUILTINS
from quivers.dsl.ast_nodes.declarations import (
    ExportDecl,
    ObjectDecl,
    ProgramDecl,
    TypeFromExpr,
)
from quivers.dsl.ast_nodes.expressions import ExprIdent
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.dsl.ast_nodes.module import Module
from quivers.dsl.ast_nodes.objects import DiscreteConstructor, TypeName
from quivers.dsl.ast_nodes.program_steps import (
    DrawArgName,
    DrawArgScalar,
    LetStep,
    ObserveStep,
    SampleStep,
)
from quivers.transpile.renderers._python_helpers import _MATH_BUILTIN_NAMES

import pytest

_ALL_TARGETS = (
    "stan", "gen", "turing", "pyro", "numpyro", "pymc", "edward2",
    "jags", "bugs", "webppl",
)
_PYTHON_TARGETS = ("pyro", "numpyro", "pymc", "edward2")

_RESP = TypeName(name="Resp")


def _scalar(x: float) -> DrawArgScalar:
    return DrawArgScalar(value=float(x))


def _module(let_value: LetExprNode, *, extra_samples: tuple[str, ...] = ()) -> Module:
    """A one-program module binding ``m = <let_value>`` and observing it."""
    samples = ("a", "b", "c", "u", *extra_samples)
    draws = tuple(
        SampleStep(vars=(name,), morphism="Normal", args=(_scalar(0.0), _scalar(1.0)))
        for name in samples
    )
    draws += (
        SampleStep(vars=("sigma",), morphism="Uniform", args=(_scalar(0.0), _scalar(1.0))),
        LetStep(name="m", value=let_value),
        ObserveStep(
            var="y",
            morphism="Normal",
            args=(DrawArgName(text="m"), DrawArgName(text="sigma")),
            index=_RESP,
        ),
    )
    return Module(statements=(
        ObjectDecl(
            name="Resp",
            init=TypeFromExpr(
                expr=DiscreteConstructor(constructor="FinSet", args=("100",))
            ),
        ),
        ProgramDecl(
            name="model", domain=_RESP, codomain=_RESP, draws=draws, return_vars=("y",)
        ),
        ExportDecl(expr=ExprIdent(name="model")),
    ))


def _m_line(target: str, module: Module) -> str:
    """The rendered ``m = ...`` binding line for ``target``."""
    src = _transpile.transpile(module, target=target).decode()
    for line in src.splitlines():
        stripped = line.strip().replace(" ", "")
        if stripped.startswith(("m=", "m[", "varm=", "m<-")):
            return line.strip()
    msg = f"no `m` binding line in {target} output:\n{src}"
    raise AssertionError(msg)


# -- #52: nested arithmetic keeps its parentheses ----------------------


@pytest.mark.parametrize("target", _ALL_TARGETS)
def test_nested_binop_operand_keeps_parens(target: str) -> None:
    """``(a + b) * c`` renders with the grouping, not ``a + b * c``."""
    a, b, c = LetExprVar(name="a"), LetExprVar(name="b"), LetExprVar(name="c")
    value = LetExprBinOp(op="*", left=LetExprBinOp(op="+", left=a, right=b), right=c)
    line = _m_line(target, _module(value))
    assert "(a + b)" in line or "(a+b)" in line, line


@pytest.mark.parametrize("target", _ALL_TARGETS)
def test_unary_operand_binop_keeps_parens(target: str) -> None:
    """``-(a + b)`` keeps the parentheses around the summed operand."""
    a, b = LetExprVar(name="a"), LetExprVar(name="b")
    value = LetExprUnaryOp(operand=LetExprBinOp(op="+", left=a, right=b))
    line = _m_line(target, _module(value))
    assert "(a + b)" in line or "(a+b)" in line, line


@pytest.mark.parametrize("target", _ALL_TARGETS)
def test_flat_operands_are_not_parenthesized(target: str) -> None:
    """A non-nested operand is left bare: ``a + b`` gains no grouping."""
    a, b = LetExprVar(name="a"), LetExprVar(name="b")
    line = _m_line(target, _module(LetExprBinOp(op="+", left=a, right=b)))
    assert "(a" not in line.replace("(a + b)", "").replace("(a+b)", ""), line


# -- #53: builtin calls lower to the target's symbol -------------------


_ERF_SYMBOL = {
    "pyro": "torch.erf(",
    "numpyro": "jsp.erf(",
    "pymc": "pymc.math.erf(",
    "edward2": "tf.math.erf(",
}


@pytest.mark.parametrize("target", _PYTHON_TARGETS)
def test_builtin_call_maps_to_target_symbol(target: str) -> None:
    """``erf(u)`` renders as the target's namespaced symbol."""
    value = LetExprCall(func="erf", args=(LetExprVar(name="u"),))
    line = _m_line(target, _module(value))
    assert _ERF_SYMBOL[target] in line.replace(" ", ""), line


def test_numpyro_emits_special_import() -> None:
    """NumPyro emits ``import jax.scipy.special`` for its ``erf`` symbol."""
    value = LetExprCall(func="erf", args=(LetExprVar(name="u"),))
    src = _transpile.transpile(_module(value), target="numpyro").decode()
    assert "import jax.scipy.special as jsp" in src, src
    # the alias is emitted once even when several special funcs are used
    value2 = LetExprBinOp(
        op="+",
        left=LetExprCall(func="erf", args=(LetExprVar(name="u"),)),
        right=LetExprCall(func="erfc", args=(LetExprVar(name="u"),)),
    )
    src2 = _transpile.transpile(_module(value2), target="numpyro").decode()
    assert src2.count("import jax.scipy.special as jsp") == 1, src2


@pytest.mark.parametrize("target", _PYTHON_TARGETS)
def test_unmapped_builtin_raises(target: str) -> None:
    """A builtin with no per-target mapping raises, not emits a bare name.

    ``softmax`` needs a ``dim`` argument the single-call surface cannot
    supply, so no target maps it.
    """
    value = LetExprCall(func="softmax", args=(LetExprVar(name="u"),))
    with pytest.raises(_transpile.UnsupportedConstruct):
        _transpile.transpile(_module(value), target=target)


@pytest.mark.parametrize("target", _PYTHON_TARGETS)
def test_non_builtin_call_passes_through(target: str) -> None:
    """A callee that is not a math builtin (a domain / user function) is
    emitted verbatim, not mapped and not raised on."""
    value = LetExprCall(func="my_helper", args=(LetExprVar(name="u"),))
    line = _m_line(target, _module(value))
    assert "my_helper(u)" in line.replace(" ", ""), line


def test_math_builtin_names_match_compiler() -> None:
    """The helper's builtin-name set mirrors the native dispatch table.

    Guards against drift between the transpile renderers' notion of "a
    math builtin that must map or raise" and the compiler's actual
    let-expression primitive table.
    """
    assert _MATH_BUILTIN_NAMES == frozenset(_LET_EXPR_BUILTINS)
