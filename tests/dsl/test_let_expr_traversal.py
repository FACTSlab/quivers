"""Binder-aware invariants for the shared let-expression traversal."""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprBool,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLambda,
    LetExprList,
    LetExprLiteral,
    LetExprMethodCall,
    LetExprNode,
    LetExprString,
    LetExprTuple,
    LetExprUnaryOp,
    LetExprUnit,
    LetExprVar,
    LetFactorBinder,
    QiecConstructorValue,
    QiecTypeName,
    TypeName,
)
from quivers.dsl.let_expr_traversal import (
    free_let_names,
    let_expr_children,
    substitute_let_expr,
)


def _var(name: str) -> LetExprVar:
    return LetExprVar(name=name)


def test_call_callee_respects_the_current_binder_scope() -> None:
    expression = LetExprLambda(
        param="f",
        body=LetExprCall(func="f", args=(_var("x"),)),
    )
    assert free_let_names(expression, include_callees=True) == ("x",)


def test_lambda_substitution_renames_a_capturing_binder() -> None:
    expression = LetExprLambda(param="y", body=_var("x"))
    replaced = substitute_let_expr(expression, {"x": _var("y")})

    assert isinstance(replaced, LetExprLambda)
    assert replaced.param == "y_1"
    assert replaced.body == _var("y")
    assert free_let_names(replaced) == ("y",)


def test_factor_substitution_renames_a_capturing_binder() -> None:
    expression = LetExprFactor(
        binders=(LetFactorBinder(var="i", index=TypeName(name="Items")),),
        body=_var("x"),
    )
    replaced = substitute_let_expr(expression, {"x": _var("i")})

    assert isinstance(replaced, LetExprFactor)
    assert replaced.binders[0].var == "i_1"
    assert replaced.body == _var("i")
    assert free_let_names(replaced) == ("i",)


def test_every_surface_variant_has_an_explicit_traversal_case() -> None:
    leaf = _var("x")
    samples: tuple[LetExprNode, ...] = (
        LetExprLiteral(value=1.0, integral=True),
        LetExprBool(value=True),
        LetExprUnit(),
        leaf,
        LetExprString(value="x"),
        LetExprBinOp(op="+", left=leaf, right=leaf),
        LetExprUnaryOp(op="-", operand=leaf),
        LetExprCall(func="f", args=(leaf,)),
        LetExprIndex(array=leaf, indices=(leaf,)),
        LetExprList(items=(leaf,)),
        LetExprTuple(items=(leaf, leaf)),
        LetExprLambda(param="x", body=leaf),
        LetExprFactor(
            binders=(LetFactorBinder(var="i", index=TypeName(name="Items")),),
            body=leaf,
        ),
        LetExprMethodCall(receiver=leaf, method="value", args=(leaf,)),
        QiecConstructorValue(
            constructor="Box",
            fields=(leaf,),
            result_type=QiecTypeName(name="Box"),
        ),
    )
    surface_variants = {
        variant
        for variant in LetExprNode.__subclasses__()
        if variant.__module__.startswith("quivers.dsl.ast_nodes")
    }

    assert {type(sample) for sample in samples} == surface_variants
    for sample in samples:
        let_expr_children(sample)
        substitute_let_expr(sample, {"unused": leaf})
