"""Walkers for type expressions, space expressions, morphism expressions,
and let-arithmetic expressions."""

from __future__ import annotations

from typing import Literal

from quivers.dsl.ast_nodes import (
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
    SpaceConstructor,
    SpaceExpr,
    SpaceName,
    SpaceProduct,
    TypeCoproduct,
    TypeEffectApply,
    TypeExpr,
    TypeName,
    TypeProduct,
    TypeSlash,
)
from quivers.dsl.parser._helpers import _required_text
from quivers.dsl.parser._registry import ParseError, _Tree


def _walk_type(t: _Tree, vid: str) -> TypeExpr:
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
    raise ParseError(f"unexpected type-expression kind: {k}")


def _flatten_type(t: _Tree, vid: str, op_kind: str) -> list[TypeExpr]:
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


def _walk_space(t: _Tree, vid: str) -> SpaceExpr:
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "space_atom":
        return SpaceName(name=t.text(t.positional(vid)[0]), line=line, col=col)
    if k == "space_constructor":
        ctor_vid = t.field(vid, "constructor")
        ctor = _required_text(t, ctor_vid, vid, "constructor")
        args: list[str] = []
        kwargs: dict[str, str] = {}
        for arg_vid in t.fields(vid, "args"):
            ak = t.kind(arg_vid)
            if ak == "space_kwarg":
                key_vid = t.field(arg_vid, "key")
                val_vid = t.field(arg_vid, "value")
                if key_vid and val_vid:
                    kwargs[t.text(key_vid)] = t.text(val_vid)
            elif ak in ("integer", "float"):
                args.append(t.text(arg_vid))
        return SpaceConstructor(
            constructor=ctor,
            args=tuple(args),
            kwargs=kwargs,
            line=line,
            col=col,
        )
    if k == "space_constructor_bare":
        ctor_vid = t.field(vid, "constructor")
        arg_vid = t.field(vid, "arg")
        return SpaceConstructor(
            constructor=_required_text(t, ctor_vid, vid, "constructor"),
            args=(_required_text(t, arg_vid, vid, "arg"),),
            kwargs={},
            line=line,
            col=col,
        )
    if k == "space_product":
        return SpaceProduct(
            components=tuple(_flatten_space(t, vid)),
            line=line,
            col=col,
        )
    raise ParseError(f"unexpected space-expression kind: {k}")


def _flatten_space(t: _Tree, vid: str) -> list[SpaceExpr]:
    out: list[SpaceExpr] = []
    left = t.field(vid, "left")
    right = t.field(vid, "right")
    if left is None or right is None:
        raise ParseError(f"space_product missing left/right at {vid}")
    if t.kind(left) == "space_product":
        out.extend(_flatten_space(t, left))
    else:
        out.append(_walk_space(t, left))
    if t.kind(right) == "space_product":
        out.extend(_flatten_space(t, right))
    else:
        out.append(_walk_space(t, right))
    return out


def _walk_expr(t: _Tree, vid: str) -> Expr:
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "expr_paren":
        return _walk_expr(t, t.positional(vid)[0])
    if k == "expr_ident":
        return ExprIdent(name=t.text(t.positional(vid)[0]), line=line, col=col)
    if k == "morphism_call":
        callee_vid = t.field(vid, "callee")
        if callee_vid is None:
            raise ParseError(f"morphism_call missing callee at {vid}")
        arg_vids = t.fields(vid, "args")
        return ExprMorphismCall(
            callee=t.text(callee_vid),
            args=tuple(t.text(a) for a in arg_vids),
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
        op_text = _op_between(t, left_vid, right_vid)
        if op_text == "<<":
            l, r = right_vid, left_vid
        else:
            l, r = left_vid, right_vid
        if l is None or r is None:
            raise ParseError(f"compose_expr missing operands at {vid}")
        # Normalize the operator text. The reverse composition
        # operator ``<<`` has already been handled by swapping the
        # operands, so the stored op is the forward form ``>>``.
        op_normalized = ">>" if op_text == "<<" else op_text
        return ExprCompose(
            left=_walk_expr(t, l),
            right=_walk_expr(t, r),
            op=op_normalized,
            line=line,
            col=col,
        )
    if k == "tensor_expr":
        left = t.field(vid, "left")
        right = t.field(vid, "right")
        if left is None or right is None:
            raise ParseError(f"tensor_expr missing operands at {vid}")
        return ExprTensorProduct(
            left=_walk_expr(t, left),
            right=_walk_expr(t, right),
            line=line,
            col=col,
        )
    if k == "postfix_expr":
        method_vid = t.field(vid, "method")
        if method_vid and t.kind(method_vid) == "method_call":
            inner_vid = t.field(vid, "inner")
            if inner_vid is None:
                raise ParseError(f"postfix_expr missing inner at {vid}")
            # method_call's `name` field is an anonymous-keyword token
            # (panproto/panproto#86) so its t.text() may be empty.
            # Recover the method name from the leading bytes of the
            # method_call node itself: the keyword always appears at
            # the start, optionally followed by `(args)`.
            mtext = t.text(method_vid)
            paren = mtext.find("(")
            method_name = (mtext[:paren] if paren >= 0 else mtext).strip()
            if method_name == "marginalize":
                names = tuple(t.text(av) for av in t.fields(method_vid, "args"))
                return ExprMarginalize(
                    inner=_walk_expr(t, inner_vid),
                    names=names,
                    line=line,
                    col=col,
                )
            if method_name in ("curry_right", "curry_left"):
                direction: Literal["right", "left"] = (
                    "right" if method_name == "curry_right" else "left"
                )
                return ExprCurry(
                    inner=_walk_expr(t, inner_vid),
                    direction=direction,
                    line=line,
                    col=col,
                )
            if method_name == "dagger":
                return ExprDagger(inner=_walk_expr(t, inner_vid), line=line, col=col)
            if method_name == "trace":
                args = t.fields(method_vid, "args")
                if len(args) != 1:
                    raise ParseError(
                        f"trace() takes exactly one object argument at {vid}"
                    )
                return ExprTrace(
                    inner=_walk_expr(t, inner_vid),
                    object_name=t.text(args[0]),
                    line=line,
                    col=col,
                )
            if method_name == "freeze":
                return ExprFreeze(inner=_walk_expr(t, inner_vid), line=line, col=col)
            if method_name == "change_base":
                # The argument is any expression that evaluates to
                # a transformation (a AlgebraHomomorphism or
                # MorphismTransformation): a bare identifier
                # (registered singleton or let-bound trans), a
                # constructor call ``softmax(B)`` /
                # ``bayes_invert(prior)`` (parsed as
                # :class:`ExprMorphismCall` and dispatched in the
                # compiler), or a composition ``t1 >>> t2`` (parsed
                # as :class:`ExprTransCompose`).
                arg_vid = t.field(method_vid, "arg")
                if arg_vid is None:
                    raise ParseError(f"change_base() missing argument at {vid}")
                return ExprChangeBase(
                    inner=_walk_expr(t, inner_vid),
                    phi=_walk_expr(t, arg_vid),
                    line=line,
                    col=col,
                )
            raise ParseError(f"unknown postfix method {method_name!r} at {vid}")
        raise ParseError(f"unexpected postfix method at {vid}")
    if k == "identity_expr":
        obj_vid = t.field(vid, "object")
        if obj_vid is None:
            raise ParseError(f"identity_expr missing object at {vid}")
        return ExprIdentity(object_name=t.text(obj_vid), line=line, col=col)
    if k == "cup_expr":
        obj_vid = t.field(vid, "object")
        if obj_vid is None:
            raise ParseError(f"cup_expr missing object at {vid}")
        return ExprCup(object_name=t.text(obj_vid), line=line, col=col)
    if k == "cap_expr":
        obj_vid = t.field(vid, "object")
        if obj_vid is None:
            raise ParseError(f"cap_expr missing object at {vid}")
        return ExprCap(object_name=t.text(obj_vid), line=line, col=col)
    if k == "from_data_expr":
        key_vid = t.field(vid, "key")
        if key_vid is None:
            raise ParseError(f"from_data_expr missing key at {vid}")
        raw_key = t.text(key_vid)
        # Strip surrounding quotes from the string literal.
        key = raw_key.strip()
        if len(key) >= 2 and key[0] == key[-1] and key[0] in ('"', "'"):
            key = key[1:-1]
        return ExprFromData(key=key, line=line, col=col)
    if k == "fan_expr":
        return ExprFan(
            exprs=tuple(_walk_expr(t, av) for av in t.fields(vid, "args")),
            line=line,
            col=col,
        )
    if k == "repeat_expr":
        inner = t.field(vid, "inner")
        cv = t.field(vid, "count")
        if inner is None:
            raise ParseError(f"repeat_expr missing inner at {vid}")
        return ExprRepeat(
            expr=_walk_expr(t, inner),
            count=int(t.text(cv)) if cv else None,
            line=line,
            col=col,
        )
    if k == "stack_expr":
        inner = t.field(vid, "inner")
        cv = t.field(vid, "count")
        if inner is None or cv is None:
            raise ParseError(f"stack_expr missing operands at {vid}")
        return ExprStack(
            expr=_walk_expr(t, inner),
            count=int(t.text(cv)),
            line=line,
            col=col,
        )
    if k == "scan_expr":
        inner = t.field(vid, "inner")
        if inner is None:
            raise ParseError(f"scan_expr missing inner at {vid}")
        init_vid = t.field(vid, "init")
        return ExprScan(
            expr=_walk_expr(t, inner),
            init=t.text(init_vid) if init_vid else "zeros",
            line=line,
            col=col,
        )
    if k == "parser_expr":
        return _walk_parser_expr(t, vid, line, col)
    if k == "chart_fold_expr":
        return _walk_chart_fold_expr(t, vid, line, col)
    raise ParseError(f"unexpected expr kind: {k}")


def _op_between(t: _Tree, left_vid: str | None, right_vid: str | None) -> str:
    if left_vid is None or right_vid is None:
        return ""
    lcs = t.consts(left_vid)
    rcs = t.consts(right_vid)
    sb, eb = lcs.get("end-byte"), rcs.get("start-byte")
    if sb is None or eb is None:
        return ""
    return t.source[int(sb) : int(eb)].decode("utf-8").strip()


def _walk_parser_expr(t: _Tree, vid: str, line: int, col: int) -> ExprParser:
    rules: tuple[str, ...] | None = None
    categories: tuple[str, ...] = ()
    terminal: str | None = None
    start: str | int = "S"
    depth = 1
    constructors: tuple[str, ...] | None = None

    # `keyword` is an anonymous-token field (`choice('parser', 'ccg',
    # 'lambek')`) — panproto/panproto#86 means it doesn't surface as an
    # edge target. Recover from source: the keyword occupies the bytes
    # between the parser_expr's start and the first '('.
    cs = t.consts(vid)
    sb = int(cs["start-byte"])
    paren_at = t.source.find(b"(", sb)
    if paren_at < 0:
        raise ParseError(f"parser_expr at {vid} has no '(': source malformed")
    keyword = t.source[sb:paren_at].decode("utf-8").strip()

    for arg_vid in t.fields(vid, "args"):
        val_vid = t.field(arg_vid, "value")
        cs = t.consts(arg_vid)
        sb = int(cs.get("start-byte", 0))
        eb = int(cs.get("end-byte", 0))
        arg_src = t.source[sb:eb].decode("utf-8")
        eq_pos = arg_src.find("=")
        key = arg_src[:eq_pos].strip() if eq_pos >= 0 else ""

        if val_vid is None:
            continue
        vk = t.kind(val_vid)

        if key == "rules":
            if vk == "ident_list":
                rules = tuple(t.text(c) for c in t.positional(val_vid))
            elif vk == "identifier":
                rules = (t.text(val_vid),)
        elif key == "categories" and vk == "ident_list":
            categories = tuple(t.text(c) for c in t.positional(val_vid))
        elif key == "terminal" and vk == "identifier":
            terminal = t.text(val_vid)
        elif key == "start":
            txt = t.text(val_vid)
            start = int(txt) if vk == "integer" else txt
        elif key == "depth" and vk == "integer":
            depth = int(t.text(val_vid))
        elif key == "constructors" and vk == "ident_list":
            constructors = tuple(t.text(c) for c in t.positional(val_vid))

    if rules is None:
        if keyword == "ccg":
            rules = ("evaluation", "harmonic_composition", "crossed_composition")
        elif keyword == "lambek":
            rules = (
                "evaluation",
                "adjunction_units",
                "tensor_introduction",
                "tensor_projection",
            )
        else:
            raise ParseError("parser() requires rules=[...]")

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


def _walk_chart_fold_expr(t: _Tree, vid: str, line: int, col: int) -> ExprChartFold:
    """Walk a chart_fold_expr into ExprChartFold.

    Each chart_fold_arg has a `key` field (one of lex, binary, unary,
    start, depth, effect_depth) and a `value` field that is either an
    expression or an integer literal.
    """
    lex: Expr | None = None
    binary: Expr | None = None
    unary: Expr | None = None
    start: str | int = "S"
    depth = 1
    effect_depth = 0

    for arg_vid in t.fields(vid, "args"):
        if t.kind(arg_vid) != "chart_fold_arg":
            continue
        val_vid = t.field(arg_vid, "value")
        if val_vid is None:
            raise ParseError(f"chart_fold_arg missing value at {arg_vid}")
        # The `key` is an anonymous-keyword token (panproto/panproto#86)
        # whose t.field()/t.text() returns nothing useful; recover it
        # from the leading bytes of the chart_fold_arg node, taking
        # everything before the first '='.
        arg_text = t.text(arg_vid)
        eq_idx = arg_text.find("=")
        if eq_idx < 0:
            raise ParseError(f"chart_fold_arg missing '=' at {arg_vid}")
        key_text = arg_text[:eq_idx].strip()
        if key_text == "lex":
            lex = _walk_expr(t, val_vid)
        elif key_text == "binary":
            binary = _walk_expr(t, val_vid)
        elif key_text == "unary":
            unary = _walk_expr(t, val_vid)
        elif key_text == "start":
            v_text = t.text(val_vid)
            try:
                start = int(v_text)
            except ValueError:
                start = v_text
        elif key_text == "depth":
            depth = int(t.text(val_vid))
        elif key_text == "effect_depth":
            effect_depth = int(t.text(val_vid))
        else:
            raise ParseError(
                f"unknown chart_fold argument key {key_text!r} at {arg_vid}"
            )

    if lex is None:
        raise ParseError(f"chart_fold(...) requires lex= argument at {vid}")

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


def _walk_let_arith(t: _Tree, vid: str) -> LetExprNode:
    k = t.kind(vid)
    if k == "let_paren":
        return _walk_let_arith(t, t.positional(vid)[0])
    if k == "let_literal":
        return LetExprLiteral(value=float(t.text(t.positional(vid)[0])))
    if k == "let_var":
        return LetExprVar(name=t.text(t.positional(vid)[0]))
    if k == "let_index":
        # arr[i, j, ...] — the Kleisli pullback of arr along the
        # finite-fibration index expression. The `array` field is a
        # nested let_arith (typically a let_var) and the `indices`
        # fields are the index expressions in order.
        array_vid = t.field(vid, "array")
        if array_vid is None:
            raise ParseError(f"let_index missing array at {vid}")
        index_vids = t.fields(vid, "indices")
        if not index_vids:
            raise ParseError(f"let_index requires at least one index at {vid}")
        return LetExprIndex(
            array=_walk_let_arith(t, array_vid),
            indices=tuple(_walk_let_arith(t, iv) for iv in index_vids),
        )
    if k == "let_call":
        # The `func` field is now an `$.identifier` non-terminal. The
        # walker reads it as a plain text node.
        func_vid = t.field(vid, "func")
        if func_vid is None:
            raise ParseError(f"let_call missing func at {vid}")
        func = t.text(func_vid)
        return LetExprCall(
            func=func,
            args=tuple(_walk_let_arith(t, av) for av in t.fields(vid, "args")),
        )
    if k == "let_list":
        # `[a, b, c]` — list literal in let-expressions.
        return LetExprList(
            items=tuple(
                _walk_let_arith(t, av)
                for av in t.positional(vid)
                if t.kind(av) not in ("[", "]", ",")
            ),
        )
    if k == "let_string":
        # `"..."` — string literal. Strip the surrounding quotes and
        # interpret backslash escapes.
        raw = t.text(t.positional(vid)[0])
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
        return LetExprString(value=raw.encode("utf-8").decode("unicode_escape"))
    if k == "let_lambda":
        # `param -> body` — lambda in the let-sublanguage.
        param_vid = t.field(vid, "param")
        body_vid = t.field(vid, "body")
        if param_vid is None or body_vid is None:
            raise ParseError(f"let_lambda missing param/body at {vid}")
        return LetExprLambda(
            param=t.text(param_vid),
            body=_walk_let_arith(t, body_vid),
        )
    if k == "let_factor":
        # ``factor v1 : I1, ..., vn : In in <body>`` or ``factor v : I
        # in { 0 -> e0, ... }`` — multi-axis indexed-tensor builder.
        binder_vids = t.fields(vid, "binders")
        if not binder_vids:
            raise ParseError(f"let_factor missing binders at {vid}")
        binders = []
        for bv in binder_vids:
            var_vid = t.field(bv, "var")
            idx_vid = t.field(bv, "index")
            if var_vid is None or idx_vid is None:
                raise ParseError(f"let_factor_binder missing var or index at {bv}")
            bline, bcol = t.line_col(bv)
            binders.append(
                LetFactorBinder(
                    var=t.text(var_vid),
                    index=_walk_type(t, idx_vid),
                    line=bline,
                    col=bcol,
                )
            )
        case_vids = t.fields(vid, "cases")
        body_vid = t.field(vid, "body")
        if case_vids and body_vid is not None:
            raise ParseError(
                f"let_factor accepts either a pattern-match body or a "
                f"uniform body, not both, at {vid}"
            )
        if case_vids:
            cases = []
            for cv in case_vids:
                label_vid = t.field(cv, "label")
                value_vid = t.field(cv, "value")
                if label_vid is None or value_vid is None:
                    raise ParseError(f"let_factor_case missing label or value at {cv}")
                cline, ccol = t.line_col(cv)
                cases.append(
                    LetFactorCase(
                        label=int(t.text(label_vid)),
                        value=_walk_let_arith(t, value_vid),
                        line=cline,
                        col=ccol,
                    )
                )
            return LetExprFactor(
                binders=tuple(binders),
                body=None,
                cases=tuple(cases),
            )
        if body_vid is None:
            raise ParseError(
                f"let_factor requires either a uniform body or a "
                f"pattern-match block at {vid}"
            )
        return LetExprFactor(
            binders=tuple(binders),
            body=_walk_let_arith(t, body_vid),
            cases=(),
        )
    if k == "let_method_call":
        # `receiver.method(args)` — dispatched at runtime.
        recv_vid = t.field(vid, "receiver")
        method_vid = t.field(vid, "method")
        if recv_vid is None or method_vid is None:
            raise ParseError(f"let_method_call missing receiver/method at {vid}")
        args = tuple(_walk_let_arith(t, av) for av in t.fields(vid, "args"))
        return LetExprMethodCall(
            receiver=_walk_let_arith(t, recv_vid),
            method=t.text(method_vid),
            args=args,
        )
    if k == "let_unary":
        op_vid = t.field(vid, "operand")
        if op_vid is None:
            raise ParseError(f"let_unary missing operand at {vid}")
        return LetExprUnaryOp(operand=_walk_let_arith(t, op_vid))
    if k == "let_binop":
        left = t.field(vid, "left")
        right = t.field(vid, "right")
        op_vid = t.field(vid, "op")
        op_text = t.text(op_vid).strip() if op_vid else _op_between(t, left, right)
        if left is None or right is None:
            raise ParseError(f"let_binop missing operands at {vid}")
        return LetExprBinOp(
            op=op_text,  # type: ignore[arg-type]
            left=_walk_let_arith(t, left),
            right=_walk_let_arith(t, right),
        )
    raise ParseError(f"unexpected let_arith kind: {k}")
