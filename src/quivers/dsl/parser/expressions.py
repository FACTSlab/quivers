"""Walkers for type expressions, morphism expressions, and let-arith
expressions ().

The ``_walk_type`` handles every type expression, including
discrete_constructor (``FinSet(N)``) and continuous_constructor
(``Real(64)``, ``Simplex(K)``, ...).
"""

from __future__ import annotations

from typing import Literal

from quivers.dsl.ast_nodes import (
    ContinuousConstructor,
    DiscreteConstructor,
    Expr,
    ExprCap,
    ExprChangeBase,
    ExprChartFold,
    ExprCompose,
    ExprCup,
    ExprCurry,
    ExprDagger,
    ExprFan,
    ExprFreeze,
    ExprFromData,
    ExprIdent,
    ExprIdentity,
    ExprMarginalize,
    ExprMorphismCall,
    ExprParser,
    ExprRepeat,
    ExprScan,
    ExprStack,
    ExprTensorProduct,
    ExprTrace,
    ExprTransCompose,
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLambda,
    LetExprList,
    LetExprLiteral,
    LetExprMethodCall,
    LetExprNode,
    LetExprString,
    LetExprUnaryOp,
    LetExprVar,
    LetFactorBinder,
    LetFactorCase,
    TypeCoproduct,
    TypeEffectApply,
    TypeExpr,
    TypeName,
    TypeProduct,
    TypeSlash,
)
from quivers.dsl.parser._helpers import _required_text
from quivers.dsl.parser._registry import ParseError, _Tree

# ---------------------------------------------------------------------------
# type expressions
# ---------------------------------------------------------------------------

_CONTINUOUS_CTORS = frozenset({
    "Real",
    "Simplex",
    "Sphere",
    "Ball",
    "CholeskyFactor",
    "Covariance",
    "Correlation",
    "Orthogonal",
    "Stiefel",
    "LowerTriangular",
    "Diagonal",
})

def _walk_type(t: _Tree, vid: str) -> TypeExpr:
    """Walk a tree-sitter type-expr vertex into the unified TypeExpr AST."""
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "type_atom":
        kids = t.positional(vid)
        if not kids:
            raise ParseError(f"type_atom {vid} has no child")
        return TypeName(name=t.text(kids[0]), line=line, col=col)
    if k == "type_paren":
        return _walk_type(t, t.positional(vid)[0])
    if k == "type_product":
        return TypeProduct(
            components=tuple(_flatten_type(t, vid, "type_product")),
            line=line,
            col=col,
        )
    if k == "type_coproduct":
        return TypeCoproduct(
            components=tuple(_flatten_type(t, vid, "type_coproduct")),
            line=line,
            col=col,
        )
    if k == "type_slash":
        result_vid = t.field(vid, "result")
        argument_vid = t.field(vid, "argument")
        if result_vid is None or argument_vid is None:
            raise ParseError(f"type_slash missing result/argument at {vid}")
        rcs = t.consts(result_vid)
        acs = t.consts(argument_vid)
        direction: Literal["/", "\\"] = "/"
        if rcs.get("end-byte") is not None and acs.get("start-byte") is not None:
            mid = t.source[int(rcs["end-byte"]) : int(acs["start-byte"])].decode(
                "utf-8"
            )
            direction = "\\" if "\\" in mid else "/"
        return TypeSlash(
            result=_walk_type(t, result_vid),
            argument=_walk_type(t, argument_vid),
            direction=direction,
            line=line,
            col=col,
        )
    if k == "type_effect_apply":
        effect_vid = t.field(vid, "effect")
        if effect_vid is None:
            raise ParseError(f"type_effect_apply missing effect at {vid}")
        arg_vids = t.fields(vid, "args")
        return TypeEffectApply(
            effect=t.text(effect_vid),
            args=tuple(_walk_type(t, av) for av in arg_vids),
            line=line,
            col=col,
        )
    if k == "discrete_constructor":
        ctor = _required_text(t, t.field(vid, "constructor"), vid, "constructor")
        args, kwargs = _walk_constructor_args(t, vid)
        return DiscreteConstructor(
            constructor=ctor,  # type: ignore[arg-type]
            args=tuple(args),
            kwargs=kwargs,
            line=line,
            col=col,
        )
    if k == "continuous_constructor":
        ctor = _required_text(t, t.field(vid, "constructor"), vid, "constructor")
        if ctor not in _CONTINUOUS_CTORS:
            raise ParseError(f"unknown continuous constructor {ctor!r} at {vid}")
        args, kwargs = _walk_constructor_args(t, vid)
        return ContinuousConstructor(
            constructor=ctor,  # type: ignore[arg-type]
            args=tuple(args),
            kwargs=kwargs,
            line=line,
            col=col,
        )
    raise ParseError(f"unexpected type-expression kind: {k}")

def _walk_constructor_args(
    t: _Tree, vid: str
) -> tuple[list[str], dict[str, str]]:
    """Split a constructor's children into positional args and kwargs."""
    args: list[str] = []
    kwargs: dict[str, str] = {}
    for arg_vid in t.fields(vid, "args"):
        ak = t.kind(arg_vid)
        if ak == "type_constructor_kwarg":
            key_vid = t.field(arg_vid, "key")
            val_vid = t.field(arg_vid, "value")
            if key_vid is not None and val_vid is not None:
                kwargs[t.text(key_vid)] = t.text(val_vid)
        elif ak in ("integer", "float"):
            args.append(t.text(arg_vid))
    return args, kwargs

def _flatten_type(t: _Tree, vid: str, op_kind: str) -> list[TypeExpr]:
    """Flatten a left-associative binary type operator into a tuple."""
    out: list[TypeExpr] = []
    left_vid = t.field(vid, "left")
    right_vid = t.field(vid, "right")
    if left_vid is None or right_vid is None:
        raise ParseError(f"{op_kind} missing left/right at {vid}")
    if t.kind(left_vid) == op_kind:
        out.extend(_flatten_type(t, left_vid, op_kind))
    else:
        out.append(_walk_type(t, left_vid))
    if t.kind(right_vid) == op_kind:
        out.extend(_flatten_type(t, right_vid, op_kind))
    else:
        out.append(_walk_type(t, right_vid))
    return out

# ---------------------------------------------------------------------------
# morphism expressions
# ---------------------------------------------------------------------------

def _walk_expr(t: _Tree, vid: str) -> Expr:
    """Walk a tree-sitter expr vertex into the Expr AST family."""
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "expr_paren":
        return _walk_expr(t, t.positional(vid)[0])
    if k == "expr_ident":
        kids = t.positional(vid)
        if not kids:
            raise ParseError(f"expr_ident has no child at {vid}")
        return ExprIdent(name=t.text(kids[0]), line=line, col=col)
    if k == "identity_expr":
        obj_vid = t.field(vid, "object")
        return ExprIdentity(
            object_name=_required_text(t, obj_vid, vid, "object"),
            line=line,
            col=col,
        )
    if k == "cup_expr":
        return ExprCup(
            object_name=_required_text(t, t.field(vid, "object"), vid, "object"),
            line=line,
            col=col,
        )
    if k == "cap_expr":
        return ExprCap(
            object_name=_required_text(t, t.field(vid, "object"), vid, "object"),
            line=line,
            col=col,
        )
    if k == "from_data_expr":
        key_vid = t.field(vid, "key")
        if key_vid is None:
            raise ParseError(f"from_data missing key at {vid}")
        key_text = t.text(key_vid)
        # Strip surrounding quotes if present.
        if key_text.startswith('"') and key_text.endswith('"'):
            key_text = key_text[1:-1]
        return ExprFromData(key=key_text, line=line, col=col)
    if k == "fan_expr":
        arg_vids = t.fields(vid, "args")
        return ExprFan(
            exprs=tuple(_walk_expr(t, av) for av in arg_vids),
            line=line,
            col=col,
        )
    if k == "repeat_expr":
        inner_vid = t.field(vid, "inner")
        count_vid = t.field(vid, "count")
        return ExprRepeat(
            expr=_walk_expr(t, inner_vid) if inner_vid else _err_expr(t, vid, "inner"),
            count=int(t.text(count_vid)) if count_vid else None,
            line=line,
            col=col,
        )
    if k == "stack_expr":
        inner_vid = t.field(vid, "inner")
        count_vid = t.field(vid, "count")
        if inner_vid is None or count_vid is None:
            raise ParseError(f"stack_expr missing inner/count at {vid}")
        return ExprStack(
            expr=_walk_expr(t, inner_vid),
            count=int(t.text(count_vid)),
            line=line,
            col=col,
        )
    if k == "scan_expr":
        inner_vid = t.field(vid, "inner")
        init_vid = t.field(vid, "init")
        if inner_vid is None:
            raise ParseError(f"scan_expr missing inner at {vid}")
        return ExprScan(
            expr=_walk_expr(t, inner_vid),
            init=t.text(init_vid) if init_vid else "zeros",
            line=line,
            col=col,
        )
    if k == "parser_expr":
        return _walk_parser_expr(t, vid, line, col)
    if k == "chart_fold_expr":
        return _walk_chart_fold_expr(t, vid, line, col)
    if k == "morphism_call":
        callee_vid = t.field(vid, "callee")
        if callee_vid is None:
            raise ParseError(f"morphism_call missing callee at {vid}")
        return ExprMorphismCall(
            callee=t.text(callee_vid),
            args=tuple(t.text(av) for av in t.fields(vid, "args")),
            line=line,
            col=col,
        )
    if k == "trans_compose":
        left_vid = t.field(vid, "left")
        right_vid = t.field(vid, "right")
        if left_vid is None or right_vid is None:
            raise ParseError(f"trans_compose missing operands at {vid}")
        return ExprTransCompose(
            left=_walk_expr(t, left_vid),
            right=_walk_expr(t, right_vid),
            line=line,
            col=col,
        )
    if k == "compose_expr":
        left_vid = t.field(vid, "left")
        right_vid = t.field(vid, "right")
        op_vid = t.field(vid, "op")
        if left_vid is None or right_vid is None:
            raise ParseError(f"compose_expr missing operands at {vid}")
        op_text = t.text(op_vid) if op_vid else _op_between(t, left_vid, right_vid)
        return ExprCompose(
            left=_walk_expr(t, left_vid),
            right=_walk_expr(t, right_vid),
            op=op_text,
            line=line,
            col=col,
        )
    if k == "tensor_expr":
        left_vid = t.field(vid, "left")
        right_vid = t.field(vid, "right")
        if left_vid is None or right_vid is None:
            raise ParseError(f"tensor_expr missing operands at {vid}")
        return ExprTensorProduct(
            left=_walk_expr(t, left_vid),
            right=_walk_expr(t, right_vid),
            line=line,
            col=col,
        )
    if k == "postfix_expr":
        inner_vid = t.field(vid, "inner")
        method_vid = t.field(vid, "method")
        if inner_vid is None or method_vid is None:
            raise ParseError(f"postfix_expr missing inner/method at {vid}")
        inner = _walk_expr(t, inner_vid)
        method_name_vid = t.field(method_vid, "name")
        method_name = t.text(method_name_vid) if method_name_vid else ""
        if method_name == "marginalize":
            names = tuple(
                t.text(av) for av in t.fields(method_vid, "args")
            )
            return ExprMarginalize(inner=inner, names=names, line=line, col=col)
        if method_name == "freeze":
            return ExprFreeze(inner=inner, line=line, col=col)
        if method_name == "dagger":
            return ExprDagger(inner=inner, line=line, col=col)
        if method_name == "trace":
            args_vid = t.field(method_vid, "args")
            if args_vid is None:
                raise ParseError(f"trace() missing arg at {method_vid}")
            return ExprTrace(
                inner=inner,
                object_name=t.text(args_vid),
                line=line,
                col=col,
            )
        if method_name == "change_base":
            phi_vid = t.field(method_vid, "arg")
            if phi_vid is None:
                raise ParseError(f"change_base() missing arg at {method_vid}")
            return ExprChangeBase(
                inner=inner,
                phi=_walk_expr(t, phi_vid),
                line=line,
                col=col,
            )
        if method_name in ("curry_right", "curry_left"):
            return ExprCurry(
                inner=inner,
                direction="right" if method_name == "curry_right" else "left",
                line=line,
                col=col,
            )
        raise ParseError(f"unknown postfix method {method_name!r} at {vid}")
    raise ParseError(f"unexpected expression kind: {k}")

def _err_expr(t: _Tree, vid: str, field: str) -> Expr:
    raise ParseError(f"expression node {vid} missing {field}")

def _op_between(t: _Tree, left_vid: str | None, right_vid: str | None) -> str:
    """Recover the compose operator string between two operand spans."""
    if left_vid is None or right_vid is None:
        return ">>"
    le = t.consts(left_vid).get("end-byte")
    rs = t.consts(right_vid).get("start-byte")
    if le is None or rs is None:
        return ">>"
    return t.source[int(le) : int(rs)].decode("utf-8").strip()

def _walk_parser_expr(t: _Tree, vid: str, line: int, col: int) -> ExprParser:
    keyword_vid = t.field(vid, "keyword")
    args = t.fields(vid, "args")
    rules: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    terminal: str | None = None
    start: str | int = "S"
    depth = 1
    constructors: tuple[str, ...] | None = None
    for arg_vid in args:
        key_vid = t.field(arg_vid, "key")
        val_vid = t.field(arg_vid, "value")
        if key_vid is None or val_vid is None:
            continue
        key = t.text(key_vid)
        if key == "rules":
            rules = _ident_list_to_tuple(t, val_vid)
        elif key == "categories":
            categories = _ident_list_to_tuple(t, val_vid)
        elif key == "constructors":
            constructors = _ident_list_to_tuple(t, val_vid)
        elif key == "terminal":
            terminal = t.text(val_vid)
        elif key == "start":
            start_text = t.text(val_vid)
            try:
                start = int(start_text)
            except ValueError:
                start = start_text
        elif key == "depth":
            depth = int(t.text(val_vid))
    del keyword_vid
    return ExprParser(
        rules=rules,
        categories=categories,
        terminal=terminal,
        start=start,
        depth=depth,
        constructors=constructors,
        line=line,
        col=col,
    )

def _walk_chart_fold_expr(
    t: _Tree, vid: str, line: int, col: int
) -> ExprChartFold:
    args = t.fields(vid, "args")
    lex: Expr | None = None
    binary: Expr | None = None
    unary: Expr | None = None
    start: str | int = "S"
    depth = 1
    effect_depth = 0
    for arg_vid in args:
        key_vid = t.field(arg_vid, "key")
        val_vid = t.field(arg_vid, "value")
        if key_vid is None or val_vid is None:
            continue
        key = t.text(key_vid)
        if key == "lex":
            lex = _walk_expr(t, val_vid)
        elif key == "binary":
            binary = _walk_expr(t, val_vid)
        elif key == "unary":
            unary = _walk_expr(t, val_vid)
        elif key == "start":
            start_text = t.text(val_vid)
            try:
                start = int(start_text)
            except ValueError:
                start = start_text
        elif key == "depth":
            depth = int(t.text(val_vid))
        elif key == "effect_depth":
            effect_depth = int(t.text(val_vid))
    if lex is None:
        raise ParseError(f"chart_fold missing lex= at {vid}")
    return ExprChartFold(
        lex=lex,
        binary=binary,
        unary=unary,
        start=start,
        depth=depth,
        effect_depth=effect_depth,
        line=line,
        col=col,
    )

def _ident_list_to_tuple(t: _Tree, vid: str) -> tuple[str, ...]:
    if t.kind(vid) == "ident_list":
        return tuple(t.text(c) for c in t.positional(vid))
    return (t.text(vid),)

# ---------------------------------------------------------------------------
# let-arithmetic
# ---------------------------------------------------------------------------

def _walk_let_arith(t: _Tree, vid: str) -> LetExprNode:
    """Walk a let-arithmetic vertex into the LetExprNode AST family."""
    k = t.kind(vid)
    if k == "let_paren":
        return _walk_let_arith(t, t.positional(vid)[0])
    if k == "let_var":
        return LetExprVar(name=t.text(vid))
    if k in ("integer", "float", "signed_number", "let_literal"):
        text = t.text(vid)
        return LetExprLiteral(value=float(text))
    if k == "let_string":
        text = t.text(vid)
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return LetExprString(value=text)
    if k == "string":
        text = t.text(vid)
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return LetExprString(value=text)
    if k == "let_unary":
        operand_vid = t.field(vid, "operand")
        if operand_vid is None:
            raise ParseError(f"let_unary missing operand at {vid}")
        return LetExprUnaryOp(operand=_walk_let_arith(t, operand_vid))
    if k == "let_binop":
        left_vid = t.field(vid, "left")
        right_vid = t.field(vid, "right")
        op_vid = t.field(vid, "op")
        if left_vid is None or right_vid is None or op_vid is None:
            raise ParseError(f"let_binop missing field at {vid}")
        op = t.text(op_vid)
        if op not in ("+", "-", "*", "/"):
            raise ParseError(f"unexpected let_binop op {op!r} at {vid}")
        return LetExprBinOp(
            op=op,  # type: ignore[arg-type]
            left=_walk_let_arith(t, left_vid),
            right=_walk_let_arith(t, right_vid),
        )
    if k == "let_call":
        func_vid = t.field(vid, "func")
        if func_vid is None:
            raise ParseError(f"let_call missing func at {vid}")
        return LetExprCall(
            func=t.text(func_vid),
            args=tuple(_walk_let_arith(t, av) for av in t.fields(vid, "args")),
        )
    if k == "let_index":
        array_vid = t.field(vid, "array")
        if array_vid is None:
            raise ParseError(f"let_index missing array at {vid}")
        return LetExprIndex(
            array=_walk_let_arith(t, array_vid),
            indices=tuple(
                _walk_let_arith(t, iv) for iv in t.fields(vid, "indices")
            ),
        )
    if k == "let_list":
        kids = t.positional(vid)
        return LetExprList(items=tuple(_walk_let_arith(t, c) for c in kids))
    if k == "let_lambda":
        param_vid = t.field(vid, "param")
        body_vid = t.field(vid, "body")
        if param_vid is None or body_vid is None:
            raise ParseError(f"let_lambda missing field at {vid}")
        return LetExprLambda(
            param=t.text(param_vid),
            body=_walk_let_arith(t, body_vid),
        )
    if k == "let_method_call":
        receiver_vid = t.field(vid, "receiver")
        method_vid = t.field(vid, "method")
        if receiver_vid is None or method_vid is None:
            raise ParseError(f"let_method_call missing field at {vid}")
        return LetExprMethodCall(
            receiver=_walk_let_arith(t, receiver_vid),
            method=t.text(method_vid),
            args=tuple(_walk_let_arith(t, av) for av in t.fields(vid, "args")),
        )
    if k == "let_factor":
        binders = tuple(
            LetFactorBinder(
                var=_required_text(t, t.field(bv, "var"), bv, "var"),
                index=_walk_type(t, t.field(bv, "index"))
                if t.field(bv, "index") is not None
                else _err_type(t, bv, "index"),
            )
            for bv in t.fields(vid, "binders")
        )
        cases = tuple(
            LetFactorCase(
                label=int(_required_text(t, t.field(cv, "label"), cv, "label")),
                value=_walk_let_arith(t, t.field(cv, "value"))
                if t.field(cv, "value") is not None
                else _err_let(t, cv, "value"),
            )
            for cv in t.fields(vid, "cases")
        )
        body_vid = t.field(vid, "body")
        body = _walk_let_arith(t, body_vid) if body_vid else None
        return LetExprFactor(binders=binders, body=body, cases=cases)
    raise ParseError(f"unexpected let-expression kind: {k}")

def _err_type(t: _Tree, vid: str, field: str) -> TypeExpr:
    raise ParseError(f"let-factor binder at {vid} missing {field}")

def _err_let(t: _Tree, vid: str, field: str) -> LetExprNode:
    raise ParseError(f"let-factor case at {vid} missing {field}")
