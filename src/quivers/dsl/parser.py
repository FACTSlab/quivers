"""Parser for the quivers DSL.

The lexer/parser pipeline is delegated to panproto via the `qvr`
tree-sitter grammar registered in `panproto-grammars-all`. The public
:func:`parse` entry point consumes `.qvr` source bytes and returns a
:class:`Module` of dataclass AST nodes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import panproto

from quivers.dsl.ast_nodes import (
    AliasDecl,
    BindStep,
    BundleDecl,
    CategoryDecl,
    BinderArg,
    BinderDecl,
    BinderVar,
    EncoderDecl,
    EncoderInitRule,
    EncoderMessageRule,
    EncoderRule,
    EncoderUpdateRule,
    EncoderVarInit,
    ConstructorDecl,
    DecoderDecl,
    DeductionDecl,
    EdgeKindDecl,
    LexiconEntry,
    LossAttachment,
    LossDecl,
    SequentRule,
    SignatureDecl,
    SortDecl,
    SortDim,
    SortVocabLiteral,
    VertexKindDecl,
    ContinuousMorphismDecl,
    DiscretizeDecl,
    EmbedDecl,
    EnumSetLiteral,
    ExportDecl,
    Expr,
    ExprChartFold,
    ExprCompose,
    ExprCurry,
    ExprFan,
    ExprIdent,
    ExprIdentity,
    ExprCap,
    ExprChangeBase,
    ExprCup,
    ExprDagger,
    ExprMarginalize,
    ExprTrace,
    ExprParser,
    ExprRepeat,
    ExprScan,
    ExprStack,
    ExprTensorProduct,
    FreeMonoidExpr,
    FreeResiduatedExpr,
    LetDecl,
    LetExprBinOp,
    LetExprCall,
    LetExprIndex,
    LetExprLambda,
    LetExprList,
    LetExprLiteral,
    LetExprMethodCall,
    LetExprNode,
    LetExprString,
    LetExprUnaryOp,
    LetExprVar,
    LetStep,
    Module,
    MorphismDecl,
    MorphismParam,
    ObjectDecl,
    ObjectParam,
    ProgramDecl,
    ProgramParam,
    ProgramStep,
    ScalarParam,
    QuantaleDecl,
    RuleDecl,
    SchemaDecl,
    SpaceConstructor,
    SpaceDecl,
    SpaceExpr,
    SpaceName,
    SpaceProduct,
    Statement,
    StochasticMorphismDecl,
    TypeCoproduct,
    TypeEffectApply,
    TypeExpr,
    TypeName,
    TypeProduct,
    TypeSlash,
)


class ParseError(Exception):
    """Raised when the .qvr source fails to parse or wrap into AST nodes."""


# ---------------------------------------------------------------------------
# panproto registry singleton
# ---------------------------------------------------------------------------

_REGISTRY: panproto.AstParserRegistry | None = None


def _registry() -> panproto.AstParserRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        from quivers.dsl import _dev_grammar

        if _dev_grammar.is_active():
            _REGISTRY = _dev_grammar.registry()  # type: ignore[assignment]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _REGISTRY = panproto.AstParserRegistry()
        if "qvr" not in _REGISTRY.protocol_names():
            raise ParseError(
                "panproto registry has no `qvr` protocol; install "
                "`panproto-grammars-all` (or a pack containing qvr)"
            )
    return _REGISTRY


# ---------------------------------------------------------------------------
# tree-walking helpers
# ---------------------------------------------------------------------------


class _Tree:
    """Indexed view of a parsed panproto schema."""

    def __init__(self, schema, source: bytes) -> None:
        self.schema = schema
        self.source = source
        self.vertices = {v.id: v for v in schema.vertices}
        self.children: dict[str, list] = {}
        for e in schema.edges:
            self.children.setdefault(e.src, []).append(e)
        self._consts: dict[str, dict[str, str]] = {}

    def consts(self, vid: str) -> dict[str, str]:
        c = self._consts.get(vid)
        if c is None:
            c = {item.sort: item.value for item in self.schema.constraints_for(vid)}
            self._consts[vid] = c
        return c

    def kind(self, vid: str) -> str:
        return self.vertices[vid].kind

    def text(self, vid: str) -> str:
        c = self.consts(vid)
        lit = c.get("literal-value")
        if lit is not None:
            return lit
        sb = c.get("start-byte")
        eb = c.get("end-byte")
        if sb is not None and eb is not None:
            return self.source[int(sb) : int(eb)].decode("utf-8")
        return ""

    def line_col(self, vid: str) -> tuple[int, int]:
        c = self.consts(vid)
        sb = c.get("start-byte")
        if sb is None:
            return 0, 0
        prefix = self.source[: int(sb)]
        line = prefix.count(b"\n") + 1
        last_nl = prefix.rfind(b"\n")
        col = (int(sb) - last_nl - 1) if last_nl >= 0 else int(sb)
        return line, col

    def _sort_key(self, vid: str) -> int:
        sb = self.consts(vid).get("start-byte")
        return int(sb) if sb is not None else 0

    def positional(self, parent_id: str) -> list[str]:
        kids = [e.tgt for e in self.children.get(parent_id, []) if e.kind == "child_of"]
        kids.sort(key=self._sort_key)
        return kids

    def field(self, parent_id: str, name: str) -> str | None:
        for e in self.children.get(parent_id, []):
            if e.kind == name:
                return e.tgt
        return None

    def fields(self, parent_id: str, name: str) -> list[str]:
        kids = [e.tgt for e in self.children.get(parent_id, []) if e.kind == name]
        kids.sort(key=self._sort_key)
        return kids


# ---------------------------------------------------------------------------
# kind-dispatched conversion
# ---------------------------------------------------------------------------


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
        # Normalise the operator text. The reverse composition
        # operator ``<<`` has already been handled by swapping the
        # operands, so the stored op is the forward form ``>>``.
        op_normalised = ">>" if op_text == "<<" else op_text
        return ExprCompose(
            left=_walk_expr(t, l),
            right=_walk_expr(t, r),
            op=op_normalised,
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
                return ExprDagger(
                    inner=_walk_expr(t, inner_vid), line=line, col=col
                )
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
            if method_name == "change_base":
                args = t.fields(method_vid, "args")
                if len(args) != 1:
                    raise ParseError(
                        f"change_base() takes exactly one "
                        f"homomorphism argument at {vid}"
                    )
                return ExprChangeBase(
                    inner=_walk_expr(t, inner_vid),
                    homomorphism=t.text(args[0]),
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


def _walk_program_step(t: _Tree, vid: str) -> ProgramStep:
    """Walk a program-body step into its AST node.

    The v0.5 surface admits four step kinds: ``bind_step`` (sample),
    ``observe_step`` (score), ``marginalize_step`` (scoped
    marginalize), and ``let_step``. The first three all denote
    Kleisli binds and walk into :class:`BindStep` with a populated
    ``mode`` field.
    """
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "bind_step":
        return _walk_bind_step(t, vid, mode="sample", line=line, col=col)
    if k == "observe_step":
        return _walk_bind_step(t, vid, mode="score", line=line, col=col)
    if k == "marginalize_step":
        return _walk_bind_step(t, vid, mode="marginal", line=line, col=col)
    if k == "let_step":
        name_vid = t.field(vid, "name")
        val_vid = t.field(vid, "value")
        if val_vid is None:
            raise ParseError(f"let_step missing value at {vid}")
        return LetStep(
            name=_required_text(t, name_vid, vid, "name"),
            value=_walk_let_arith(t, val_vid),
            line=line,
            col=col,
        )
    raise ParseError(f"unexpected program-step kind: {k}")


def _walk_bind_step(
    t: _Tree,
    vid: str,
    mode: Literal["sample", "score", "marginal"],
    line: int,
    col: int,
) -> BindStep:
    """Walk a bind / observe / marginalize step into a unified BindStep.

    All three surface shapes share the same underlying grammar
    fields (optional index annotation, morphism, optional args);
    marginalize additionally carries a `scope` block. The mode
    parameter picks between sample / score / marginal Kleisli-bind
    semantics, all of which are captured by the BindStep's
    ``mode`` field.
    """
    # var-name(s): sample uses _var_pattern (single or tuple);
    # score / marginal use a single 'var' identifier field.
    if mode == "sample":
        var_vid = t.field(vid, "vars")
        if var_vid is None:
            raise ParseError(f"bind_step missing vars at {vid}")
        if t.kind(var_vid) == "var_tuple":
            vars_t = tuple(t.text(c) for c in t.positional(var_vid))
        else:
            vars_t = (t.text(var_vid),)
    else:
        var_vid = t.field(vid, "var")
        if var_vid is None:
            raise ParseError(f"{mode}_step missing var at {vid}")
        vars_t = (_required_text(t, var_vid, vid, "var"),)

    morph_vid = t.field(vid, "morphism")
    if morph_vid is None:
        raise ParseError(f"{mode}_step missing morphism at {vid}")
    morphism = _required_text(t, morph_vid, vid, "morphism")

    args_list: list[str | float] = []
    for av in t.fields(vid, "args"):
        args_list.append(_walk_draw_arg(t, av))
    args_t: tuple[str | float, ...] | None = tuple(args_list) if args_list else None

    idx_vid = t.field(vid, "index")
    index_expr = _walk_type(t, idx_vid) if idx_vid is not None else None

    scope_t: tuple[ProgramStep, ...] | None = None
    over_t: str | None = None
    over_objs_t: tuple[str, ...] | None = None
    via_t: str | None = None
    via_axes_t: tuple[str, ...] | None = None
    reduction_t: str | None = None
    if mode == "marginal":
        scope_t = tuple(_walk_program_step(t, sv) for sv in t.fields(vid, "scope"))
        over_vid = t.field(vid, "over")
        if over_vid is not None:
            # `over` is now a type expression; for a single plate
            # it's a TypeName, for a product `G * H` it's a TypeProduct.
            over_expr = _walk_type(t, over_vid)
            if isinstance(over_expr, TypeName):
                over_t = over_expr.name
            elif isinstance(over_expr, TypeProduct):
                names: list[str] = []
                for comp in over_expr.components:
                    if not isinstance(comp, TypeName):
                        raise ParseError(
                            f"marginalize: `over` product components must "
                            f"be plate names; got {type(comp).__name__} "
                            f"at {over_vid}"
                        )
                    names.append(comp.name)
                over_objs_t = tuple(names)
            else:
                raise ParseError(
                    f"marginalize: `over` must be a plate name or "
                    f"product of plate names; got "
                    f"{type(over_expr).__name__} at {over_vid}"
                )
        via_vid = t.field(vid, "via")
        if via_vid is not None:
            # `via` is either a bare identifier or a `via_product(...)`.
            if t.kind(via_vid) == "via_product":
                axis_ids = t.fields(via_vid, "axis")
                via_axes_t = tuple(
                    _required_text(t, av, via_vid, "axis") for av in axis_ids
                )
            else:
                via_t = _required_text(t, via_vid, vid, "via")
        red_vid = t.field(vid, "reduction")
        if red_vid is not None:
            reduction_t = _required_text(t, red_vid, vid, "reduction")

    return BindStep(
        vars=vars_t,
        morphism=morphism,
        args=args_t,
        index=index_expr,
        mode=mode,
        scope=scope_t,
        over=over_t,
        over_objs=over_objs_t,
        via=via_t,
        via_axes=via_axes_t,
        reduction=reduction_t,
        line=line,
        col=col,
    )


def _walk_program_param(t: _Tree, vid: str) -> ProgramParam:
    """Walk a ``typed_program_param`` node into a typed ProgramParam.

    Recognises the three universes (object / scalar / morphism) and
    builds the matching AST variant. The parametric-program denotation
    treats each parameter as a dependent quantification over the
    corresponding category of its kind.
    """
    nv = t.field(vid, "name")
    kv = t.field(vid, "kind")
    if nv is None or kv is None:
        raise ParseError(f"typed_program_param missing name/kind at {vid}")
    name = t.text(nv)
    line, col = t.line_col(vid)
    kk = t.kind(kv)
    if kk == "object_kind":
        universe = t.text(kv)
        if universe not in ("FinSet", "Space", "Object"):
            raise ParseError(f"unknown object universe {universe!r} at {vid}")
        return ObjectParam(name=name, universe=universe, line=line, col=col)  # type: ignore[arg-type]
    if kk == "scalar_kind":
        sk = t.text(kv)
        if sk not in ("Real", "Nat"):
            raise ParseError(f"unknown scalar kind {sk!r} at {vid}")
        return ScalarParam(name=name, scalar_kind=sk, line=line, col=col)  # type: ignore[arg-type]
    if kk == "morphism_kind":
        dv = t.field(kv, "domain")
        cv = t.field(kv, "codomain")
        if dv is None or cv is None:
            raise ParseError(f"morphism_kind missing domain/codomain at {kv}")
        return MorphismParam(
            name=name,
            domain=_walk_type(t, dv),
            codomain=_walk_type(t, cv),
            line=line,
            col=col,
        )
    raise ParseError(f"unexpected program-param kind: {kk}")


def _walk_draw_arg(t: _Tree, vid: str) -> str | float:
    """Walk a family-argument into its compiler representation.

    Identifiers and numeric literals are walked into their natural
    Python values. A ``bracket_index_arg`` (e.g., ``theta[N]``) is
    encoded as the string ``"theta[N]"`` — the compiler detects
    the bracket and unpacks the section's name and index set when
    resolving the argument at draw / observe time.
    """
    k = t.kind(vid)
    if k == "identifier":
        return t.text(vid)
    if k == "signed_number":
        return float(t.text(vid))
    if k in ("integer", "float"):
        return float(t.text(vid))
    if k == "bracket_index_arg":
        nv = t.field(vid, "name")
        iv = t.field(vid, "index")
        if nv is None or iv is None:
            raise ParseError(f"bracket_index_arg malformed at {vid}")
        return f"{t.text(nv)}[{t.text(iv)}]"
    raise ParseError(f"unexpected draw arg kind: {k}")


# -- top-level statements --


def _walk_statement(t: _Tree, vid: str) -> Statement | list[Statement]:
    k = t.kind(vid)
    line, col = t.line_col(vid)

    if k == "quantale_decl":
        nv = t.field(vid, "name")
        return QuantaleDecl(name=_required_text(t, nv, vid, "name"), line=line, col=col)
    if k == "category_decl":
        out: list[Statement] = []
        for nv in t.fields(vid, "names"):
            ln, cl = t.line_col(nv)
            out.append(CategoryDecl(name=t.text(nv), line=ln, col=cl))
        if not out:
            return CategoryDecl(name="", line=line, col=col)
        return out if len(out) > 1 else out[0]
    if k == "rule_decl":
        return _walk_rule_decl(t, vid, line, col)
    if k == "schema_decl":
        return _walk_schema_decl(t, vid, line, col)
    if k == "object_decl":
        nv = t.field(vid, "name")
        tv = t.field(vid, "type")
        iv = t.field(vid, "init")
        if tv is not None:
            return ObjectDecl(
                name=_required_text(t, nv, vid, "name"),
                type_expr=_walk_type(t, tv),
                init=None,
                line=line,
                col=col,
            )
        if iv is not None:
            return ObjectDecl(
                name=_required_text(t, nv, vid, "name"),
                type_expr=None,
                init=_walk_object_initializer(t, iv),
                line=line,
                col=col,
            )
        raise ParseError(f"object_decl missing type/init at {vid}")
    if k == "morphism_decl":
        cs = t.consts(vid)
        prefix = t.source[int(cs["start-byte"]) : int(cs["start-byte"]) + 8].decode(
            "utf-8"
        )
        morph_kind = "observed" if prefix.startswith("observed") else "latent"
        opt_vid = t.field(vid, "options")
        options = _walk_options(t, opt_vid) if opt_vid else {}
        init_vid = t.field(vid, "init")
        init_expr = _walk_expr(t, init_vid) if init_vid else None
        nv = t.field(vid, "name")
        dv = t.field(vid, "domain")
        cv = t.field(vid, "codomain")
        if dv is None or cv is None:
            raise ParseError(f"morphism_decl missing domain/codomain at {vid}")
        return MorphismDecl(
            morphism_kind=morph_kind,  # type: ignore[arg-type]
            name=_required_text(t, nv, vid, "name"),
            domain=_walk_type(t, dv),
            codomain=_walk_type(t, cv),
            init_expr=init_expr,
            options=options,
            line=line,
            col=col,
        )
    if k == "ERROR":
        raise ParseError(
            f"syntax error at line {line}, col {col}: "
            f"{t.source[int(t.consts(vid)['start-byte']) : int(t.consts(vid)['end-byte'])].decode('utf-8')!r}"
        )
    if k in ("space_decl", "type_alias_decl"):
        nv = t.field(vid, "name")
        vv = t.field(vid, "value")
        if vv is None:
            raise ParseError(f"{k} missing value at {vid}")
        return SpaceDecl(
            name=_required_text(t, nv, vid, "name"),
            space_expr=_walk_space(t, vv),
            line=line,
            col=col,
        )
    if k == "alias_decl":
        nv = t.field(vid, "name")
        vv = t.field(vid, "value")
        if vv is None:
            raise ParseError(f"alias_decl missing value at {vid}")
        return AliasDecl(
            name=_required_text(t, nv, vid, "name"),
            type_expr=_walk_type(t, vv),
            line=line,
            col=col,
        )
    if k == "bundle_decl":
        nv = t.field(vid, "name")
        rule_vids = t.fields(vid, "rules")
        return BundleDecl(
            name=_required_text(t, nv, vid, "name"),
            rules=tuple(t.text(r) for r in rule_vids),
            line=line,
            col=col,
        )
    if k == "continuous_decl":
        rep_vid = t.field(vid, "replicate")
        replicate = int(t.text(t.positional(rep_vid)[0])) if rep_vid else None
        opt_vid = t.field(vid, "options")
        options = _walk_options(t, opt_vid) if opt_vid else {}
        nv = t.field(vid, "name")
        dv = t.field(vid, "domain")
        cv = t.field(vid, "codomain")
        fv = t.field(vid, "family")
        if dv is None or cv is None:
            raise ParseError(f"continuous_decl missing domain/codomain at {vid}")
        return ContinuousMorphismDecl(
            name=_required_text(t, nv, vid, "name"),
            domain=_walk_type(t, dv),
            codomain=_walk_type(t, cv),
            family=_required_text(t, fv, vid, "family"),
            options=options,
            replicate=replicate,
            line=line,
            col=col,
        )
    if k == "stochastic_decl":
        rep_vid = t.field(vid, "replicate")
        replicate = int(t.text(t.positional(rep_vid)[0])) if rep_vid else None
        nv = t.field(vid, "name")
        dv = t.field(vid, "domain")
        cv = t.field(vid, "codomain")
        if dv is None or cv is None:
            raise ParseError(f"stochastic_decl missing domain/codomain at {vid}")
        return StochasticMorphismDecl(
            name=_required_text(t, nv, vid, "name"),
            domain=_walk_type(t, dv),
            codomain=_walk_type(t, cv),
            replicate=replicate,
            line=line,
            col=col,
        )
    if k == "discretize_decl":
        opt_vid = t.field(vid, "options")
        options = _walk_options(t, opt_vid) if opt_vid else {}
        nv = t.field(vid, "name")
        sv = t.field(vid, "space")
        bv = t.field(vid, "bins")
        if bv is None:
            raise ParseError(f"discretize_decl missing bins at {vid}")
        return DiscretizeDecl(
            name=_required_text(t, nv, vid, "name"),
            space_name=_required_text(t, sv, vid, "space"),
            n_bins=int(t.text(bv)),
            options=options,
            line=line,
            col=col,
        )
    if k == "embed_decl":
        rep_vid = t.field(vid, "replicate")
        replicate = int(t.text(t.positional(rep_vid)[0])) if rep_vid else None
        nv = t.field(vid, "name")
        dv = t.field(vid, "domain")
        cv = t.field(vid, "codomain")
        return EmbedDecl(
            name=_required_text(t, nv, vid, "name"),
            domain_name=_required_text(t, dv, vid, "domain"),
            codomain_name=_required_text(t, cv, vid, "codomain"),
            replicate=replicate,
            line=line,
            col=col,
        )
    if k == "program_decl":
        params_vids = t.fields(vid, "params")
        data_params: list[str] = []
        type_params_list: list[ProgramParam] = []
        for pv in params_vids:
            if t.kind(pv) == "typed_program_param":
                type_params_list.append(_walk_program_param(t, pv))
            else:
                data_params.append(t.text(pv))
        if data_params and type_params_list:
            raise ParseError(
                f"program {_required_text(t, t.field(vid, 'name'), vid, 'name')!r} "
                f"mixes bare data parameters and typed template parameters"
            )
        params: tuple[str, ...] | None = tuple(data_params) if data_params else None
        type_params: tuple[ProgramParam, ...] | None = (
            tuple(type_params_list) if type_params_list else None
        )
        effects_vids = t.fields(vid, "effects")
        effects_set: frozenset[str] | None = (
            frozenset(t.text(ev) for ev in effects_vids) if effects_vids else None
        )
        over_vid = t.field(vid, "over_model")
        over_model: str | None = t.text(over_vid) if over_vid is not None else None
        steps = tuple(_walk_program_step(t, sv) for sv in t.fields(vid, "steps"))
        ret_vid = t.field(vid, "return")
        if ret_vid is None:
            raise ParseError(f"program_decl missing return at {vid}")
        return_vars, return_labels = _walk_return_pattern(t, ret_vid)
        nv = t.field(vid, "name")
        dv = t.field(vid, "domain")
        cv = t.field(vid, "codomain")
        if dv is None or cv is None:
            raise ParseError(f"program_decl missing domain/codomain at {vid}")
        # Posterior-block constraint: when `over M` is set, the body
        # must be deterministic — no `sample` / `score` binds.
        if over_model is not None:
            for s in steps:
                if isinstance(s, BindStep) and s.mode in ("sample", "score"):
                    raise ParseError(
                        f"program with 'over {over_model}' is a posterior "
                        "block and may not contain sample / observe binds "
                        "(posterior runs after conditioning); use 'let' "
                        "and 'marginalize' only"
                    )
        return ProgramDecl(
            name=_required_text(t, nv, vid, "name"),
            params=params,
            domain=_walk_type(t, dv),
            codomain=_walk_type(t, cv),
            draws=steps,
            return_vars=return_vars,
            return_labels=return_labels,
            effects=effects_set,
            over_model=over_model,
            type_params=type_params,
            line=line,
            col=col,
        )
    if k == "let_decl":
        where_vids = t.fields(vid, "where")
        where: tuple[Statement, ...] | None = None
        if where_vids:
            wd: list[Statement] = []
            for wv in where_vids:
                result = _walk_statement(t, wv)
                if isinstance(result, list):
                    wd.extend(result)
                else:
                    wd.append(result)
            where = tuple(wd) if wd else None
        nv = t.field(vid, "name")
        vv = t.field(vid, "value")
        if vv is None:
            raise ParseError(f"let_decl missing value at {vid}")
        return LetDecl(
            name=_required_text(t, nv, vid, "name"),
            expr=_walk_expr(t, vv),
            where=where,
            line=line,
            col=col,
        )
    if k == "export_decl":
        vv = t.field(vid, "value")
        if vv is None:
            raise ParseError(f"export_decl missing value at {vid}")
        return ExportDecl(
            expr=_walk_expr(t, vv),
            line=line,
            col=col,
        )
    if k == "deduction_decl":
        return _walk_deduction_decl(t, vid, line, col)
    if k == "signature_decl":
        return _walk_signature_decl(t, vid, line, col)
    if k == "encoder_decl":
        return _walk_encoder_decl(t, vid, line, col)
    if k == "decoder_decl":
        return _walk_decoder_decl(t, vid, line, col)
    if k == "loss_decl":
        return _walk_loss_decl(t, vid, line, col)
    raise ParseError(f"unexpected statement kind: {k}")


def _walk_deduction_decl(t: _Tree, vid: str, line: int, col: int) -> DeductionDecl:
    """Walk a ``deduction NAME : dom -> cod { … }`` block.

    Collects the brace-delimited body's atoms, sequent rules,
    and field assignments (semiring, start, depth) into a
    :class:`DeductionDecl`.
    """
    nv = t.field(vid, "name")
    dv = t.field(vid, "domain")
    cv = t.field(vid, "codomain")
    if nv is None or dv is None or cv is None:
        raise ParseError(f"deduction_decl missing name/domain/codomain at {vid}")
    atoms: list[str] = []
    rules: list[SequentRule] = []
    semiring: str | None = None
    start: str | None = None
    depth: int | None = None
    lex_entries: list[LexiconEntry] = []
    lex_from_file: str | None = None
    lex_from_file_learnable: bool = False
    axioms_source: str | None = None
    item_signature: str | None = None
    item_encoder: str | None = None
    # Walk children in order.
    for child in t.positional(vid):
        kk = t.kind(child)
        if kk == "deduction_atoms":
            for av in t.fields(child, "atoms"):
                atoms.append(t.text(av))
        elif kk == "deduction_rule":
            r_name = _required_text(t, t.field(child, "name"), child, "name")
            premises = tuple(_walk_type(t, pv) for pv in t.fields(child, "premises"))
            conc_vid = t.field(child, "conclusion")
            if conc_vid is None:
                raise ParseError(f"deduction_rule missing conclusion at {child}")
            conclusion = _walk_type(t, conc_vid)
            rl, rc = t.line_col(child)
            rules.append(
                SequentRule(
                    name=r_name,
                    premises=premises,
                    conclusion=conclusion,
                    line=rl,
                    col=rc,
                )
            )
        elif kk == "deduction_semiring":
            semiring = _required_text(t, t.field(child, "semiring"), child, "semiring")
        elif kk == "deduction_start":
            start = _required_text(t, t.field(child, "start"), child, "start")
        elif kk == "deduction_depth":
            depth = int(_required_text(t, t.field(child, "depth"), child, "depth"))
        elif kk == "deduction_lexicon":
            for entry_vid in t.positional(child):
                if t.kind(entry_vid) != "lexicon_entry":
                    continue
                lex_entries.append(_walk_lexicon_entry(t, entry_vid))
        elif kk == "deduction_lexicon_from_file":
            path_raw = _required_text(t, t.field(child, "path"), child, "path")
            # Strip surrounding quotes from the string literal.
            if path_raw.startswith('"') and path_raw.endswith('"'):
                path_raw = path_raw[1:-1]
            lex_from_file = path_raw
            lex_from_file_learnable = t.field(child, "learnable") is not None
        elif kk == "deduction_axioms":
            axioms_source = _required_text(t, t.field(child, "source"), child, "source")
        elif kk == "deduction_signature":
            item_signature = _required_text(
                t, t.field(child, "signature"), child, "signature"
            )
        elif kk == "deduction_encoder_attach":
            item_encoder = _required_text(
                t, t.field(child, "encoder"), child, "encoder"
            )
    return DeductionDecl(
        name=_required_text(t, nv, vid, "name"),
        domain=_walk_type(t, dv),
        codomain=_walk_type(t, cv),
        atoms=tuple(atoms),
        rules=tuple(rules),
        semiring=semiring,
        start=start,
        depth=depth,
        lexicon=tuple(lex_entries),
        lexicon_from_file=lex_from_file,
        lexicon_from_file_learnable=lex_from_file_learnable,
        axioms_source=axioms_source,
        item_signature=item_signature,
        item_encoder=item_encoder,
        line=line,
        col=col,
    )


def _walk_lexicon_entry(t: _Tree, vid: str) -> LexiconEntry:
    """Walk a single `lexicon_entry` into a :class:`LexiconEntry`.

    Surface form: ``"word" : Category = lf_template @ learnable``.
    """
    wv = t.field(vid, "word")
    cv = t.field(vid, "category")
    lv = t.field(vid, "lf")
    if wv is None or cv is None or lv is None:
        raise ParseError(f"lexicon_entry malformed at {vid}")
    word_raw = t.text(wv)
    if word_raw.startswith('"') and word_raw.endswith('"'):
        word_raw = word_raw[1:-1]
    learnable_vid = t.field(vid, "learnable")
    learnable = learnable_vid is not None
    line, col = t.line_col(vid)
    return LexiconEntry(
        word=word_raw,
        category=_walk_type(t, cv),
        lf=_walk_let_arith(t, lv),
        learnable=learnable,
        line=line,
        col=col,
    )


def _walk_rule_decl(t: _Tree, vid: str, line: int, col: int) -> RuleDecl:
    nv = t.field(vid, "name")
    var_vids = t.fields(vid, "variables")
    prem_vids = t.fields(vid, "premises")
    concl_vid = t.field(vid, "conclusion")
    if concl_vid is None:
        raise ParseError(f"rule_decl missing conclusion at {vid}")
    return RuleDecl(
        name=_required_text(t, nv, vid, "name"),
        variables=tuple(t.text(v) for v in var_vids),
        premises=tuple(_walk_type(t, p) for p in prem_vids),
        conclusion=_walk_type(t, concl_vid),
        line=line,
        col=col,
    )


def _walk_object_initializer(t: _Tree, vid: str) -> EnumSetLiteral | FreeResiduatedExpr:
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "enum_set_literal":
        elem_vids = t.fields(vid, "elements")
        return EnumSetLiteral(
            elements=tuple(t.text(e) for e in elem_vids),
            line=line,
            col=col,
        )
    if k == "free_monoid_expr":
        gen_vid = t.field(vid, "generators")
        ml_vid = t.field(vid, "max_length")
        if gen_vid is None or ml_vid is None:
            raise ParseError(f"free_monoid_expr missing generators/max_length at {vid}")
        return FreeMonoidExpr(
            generators=t.text(gen_vid),
            max_length=int(t.text(ml_vid)),
            line=line,
            col=col,
        )
    if k == "free_residuated_expr":
        gen_vid = t.field(vid, "generators")
        if gen_vid is None:
            raise ParseError(f"free_residuated_expr missing generators at {vid}")
        depth = 1
        ops: list[str] = []
        # The grammar's free_residuated_arg variants carry one of two
        # field-tagged children: a depth integer or per-op identifier(s).
        for arg_vid in t.positional(vid):
            if t.kind(arg_vid) != "free_residuated_arg":
                continue
            d = t.field(arg_vid, "depth")
            if d is not None:
                depth = int(t.text(d))
                continue
            for op_vid in t.fields(arg_vid, "op"):
                ops.append(t.text(op_vid))
        if not ops:
            ops = ["slash"]
        return FreeResiduatedExpr(
            generators=t.text(gen_vid),
            depth=depth,
            ops=tuple(ops),
            line=line,
            col=col,
        )
    raise ParseError(f"unexpected object_initializer kind: {k}")


def _walk_schema_decl(t: _Tree, vid: str, line: int, col: int) -> SchemaDecl:
    nv = t.field(vid, "name")
    param_vids = t.fields(vid, "parameters")
    dom_vid = t.field(vid, "domain")
    cod_vid = t.field(vid, "codomain")
    if dom_vid is None or cod_vid is None:
        raise ParseError(f"schema_decl missing domain/codomain at {vid}")
    param_names: list[tuple[str, ...]] = []
    param_types: list[TypeExpr] = []
    for pv in param_vids:
        name_vids = t.fields(pv, "names")
        type_vid = t.field(pv, "type")
        if type_vid is None:
            raise ParseError(f"schema_parameter missing type at {pv}")
        param_names.append(tuple(t.text(n) for n in name_vids))
        param_types.append(_walk_type(t, type_vid))
    return SchemaDecl(
        name=_required_text(t, nv, vid, "name"),
        parameter_names=tuple(param_names),
        parameter_types=tuple(param_types),
        domain=_walk_type(t, dom_vid),
        codomain=_walk_type(t, cod_vid),
        line=line,
        col=col,
    )


def _required_text(
    t: _Tree, child_vid: str | None, parent_vid: str, field_name: str
) -> str:
    """Return the text of a required-by-grammar field, raising if missing.

    Several Statement variants — quantale, object, morphism, space, etc. —
    declare an identifier ``name`` field. Tree-sitter guarantees the
    field exists on a successful parse, so a ``None`` here means the
    parse was corrupted (an ``ERROR`` node leaked through, or the
    grammar was edited without updating the walker).
    """
    if child_vid is None:
        raise ParseError(
            f"missing required {field_name!r} field at {parent_vid} (malformed parse)"
        )
    return t.text(child_vid)


def _walk_options(t: _Tree, vid: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in t.positional(vid):
        if t.kind(entry) != "option_entry":
            continue
        kvid = t.field(entry, "key")
        vvid = t.field(entry, "value")
        out[_required_text(t, kvid, entry, "key")] = _required_text(
            t, vvid, entry, "value"
        )
    return out


def _walk_return_pattern(
    t: _Tree, vid: str
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Walk a return clause into (vars, labels).

    Three forms:

    * ``return x`` — single variable; ``vars=(x,)``, ``labels=None``.
    * ``return (x, y, z)`` — positional tuple; ``vars=(x, y, z)``,
      ``labels=None``.
    * ``return (a: x, b: y)`` — labelled tuple; ``vars=(x, y)``,
      ``labels=(a, b)``. The labels rename the coordinates of the
      output product space at the schema level.
    """
    k = t.kind(vid)
    if k == "identifier":
        return (t.text(vid),), None
    if k == "return_tuple":
        return tuple(t.text(c) for c in t.positional(vid)), None
    if k == "return_labeled_tuple":
        labels: list[str] = []
        vars_l: list[str] = []
        for entry in t.positional(vid):
            if t.kind(entry) != "return_label_entry":
                continue
            lvid = t.field(entry, "label")
            vvid = t.field(entry, "var")
            labels.append(_required_text(t, lvid, entry, "label"))
            vars_l.append(_required_text(t, vvid, entry, "var"))
        return tuple(vars_l), tuple(labels)
    raise ParseError(f"unexpected return pattern kind: {k}")


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def parse(source: str | bytes, file_path: str = "<source>") -> Module:
    """Parse `.qvr` source bytes into a :class:`Module`."""
    if isinstance(source, str):
        source_bytes = source.encode("utf-8")
    else:
        source_bytes = source

    schema = _registry().parse_with_protocol("qvr", source_bytes, file_path)
    tree = _Tree(schema, source_bytes)

    root_id = next(
        (v.id for v in schema.vertices if v.kind == "source_file"),
        None,
    )
    if root_id is None:
        raise ParseError(f"panproto schema has no source_file vertex for {file_path}")

    statements: list[Statement] = []
    pending_docs: list[str] = []
    for child in tree.positional(root_id):
        ckind = tree.kind(child)
        if ckind == "line_comment":
            # plain `# ...` comments are dropped at parse time
            continue
        if ckind == "doc_comment":
            # `## ...` doc comments are accumulated; attached to the
            # next statement that carries a docs field.
            text = tree.text(child)
            stripped = text[2:].lstrip() if text.startswith("##") else text
            pending_docs.append(stripped.rstrip())
            continue
        result = _walk_statement(tree, child)
        results = result if isinstance(result, list) else [result]
        if pending_docs:
            docs = tuple(pending_docs)
            results = [_attach_docs(s, docs) for s in results]
            pending_docs = []
        statements.extend(results)
    return Module(statements=tuple(statements))


def _attach_docs(stmt: Statement, docs: tuple[str, ...]) -> Statement:
    """Attach accumulated ``##`` doc-comment lines to a Statement.

    Returns a copy of ``stmt`` with its ``docs`` field extended;
    Statement variants that lack a ``docs`` field are returned
    unchanged. didactic Models are immutable; :meth:`Model.with_` is
    the field-replacement constructor.
    """
    # `docs` is a declared field on a fixed subset of Statement
    # variants (ObjectDecl, MorphismDecl, SchemaDecl, ProgramDecl,
    # AliasDecl, BundleDecl). Probe via the class's field-spec
    # registry rather than instance __getattr__, since dx.Model's
    # attribute fall-through raises AttributeError on undeclared
    # field accesses.
    fields = getattr(type(stmt), "__field_specs__", None)
    if fields is None or "docs" not in fields:
        return stmt
    existing = stmt.docs  # type: ignore[attr-defined]
    return stmt.with_(docs=tuple(existing) + docs)  # type: ignore[attr-defined]


def parse_file(path: str | Path) -> Module:
    """Parse a `.qvr` file at `path`."""
    p = Path(path)
    return parse(p.read_bytes(), str(p))


# ---------------------------------------------------------------------------
# structural-compression walkers
# ---------------------------------------------------------------------------


def _walk_signature_decl(t: _Tree, vid: str, line: int, col: int) -> SignatureDecl:
    name_vid = t.field(vid, "name")
    if name_vid is None:
        raise ParseError(f"signature_decl missing name at {vid}")
    name = t.text(name_vid)
    params = tuple(t.text(c) for c in t.fields(vid, "params"))

    sorts: list[SortDecl] = []
    constructors: list[ConstructorDecl] = []
    binders: list[BinderDecl] = []
    vertex_kinds: list[VertexKindDecl] = []
    edge_kinds: list[EdgeKindDecl] = []

    for child in t.positional(vid):
        ck = t.kind(child)
        if ck == "signature_sorts":
            for s in t.fields(child, "sorts"):
                sorts.append(_walk_sort_decl(t, s))
        elif ck == "signature_constructors":
            for c in t.fields(child, "constructors"):
                constructors.append(_walk_constructor_decl(t, c))
        elif ck == "signature_binders":
            for b in t.fields(child, "binders"):
                binders.append(_walk_binder_decl(t, b))
        elif ck == "signature_vertex_kinds":
            for v in t.fields(child, "vertex_kinds"):
                vertex_kinds.append(_walk_vertex_kind_decl(t, v))
        elif ck == "signature_edge_kinds":
            for e in t.fields(child, "edge_kinds"):
                edge_kinds.append(_walk_edge_kind_decl(t, e))

    return SignatureDecl(
        name=name,
        params=params,
        sorts=tuple(sorts),
        constructors=tuple(constructors),
        binders=tuple(binders),
        vertex_kinds=tuple(vertex_kinds),
        edge_kinds=tuple(edge_kinds),
        line=line,
        col=col,
    )


def _walk_sort_decl(t: _Tree, vid: str) -> SortDecl:
    name = t.text(t.field(vid, "name"))
    kind_vid = t.field(vid, "kind")
    if kind_vid is None:
        raise ParseError(f"sort_decl missing kind at {vid}")
    kind_txt = t.text(kind_vid)
    dim_vid = t.field(vid, "dim")
    dim = int(t.text(dim_vid)) if dim_vid is not None else None
    vocab: list[SortVocabLiteral] = []
    for vlit_vid in t.fields(vid, "vocab"):
        # `vocab_literal` is a wrapper choice over string/integer/
        # float; descend to the inner concrete literal node.
        inner_vids = [
            c
            for c in t.positional(vlit_vid)
            if t.kind(c) in ("string", "integer", "float")
        ]
        if len(inner_vids) != 1:
            raise ParseError(
                f"sort_decl {name!r}: vocabulary literal at {vlit_vid} is "
                f"malformed (expected one of string/integer/float, got "
                f"{[t.kind(c) for c in inner_vids]!r})"
            )
        inner = inner_vids[0]
        lit_kind = t.kind(inner)
        vocab.append(SortVocabLiteral(kind=lit_kind, text=t.text(inner)))
    ln, cl = t.line_col(vid)
    return SortDecl(
        name=name,
        kind=kind_txt,
        dim=dim,
        vocab=tuple(vocab),
        line=ln,
        col=cl,
    )


def _walk_constructor_decl(t: _Tree, vid: str) -> ConstructorDecl:
    name = t.text(t.field(vid, "name"))
    domain_vids = t.fields(vid, "domain")
    domain = tuple(t.text(d) for d in domain_vids)
    codomain = t.text(t.field(vid, "codomain"))
    ln, cl = t.line_col(vid)
    return ConstructorDecl(
        name=name,
        domain=domain,
        codomain=codomain,
        line=ln,
        col=cl,
    )


def _walk_binder_decl(t: _Tree, vid: str) -> BinderDecl:
    name = t.text(t.field(vid, "name"))
    binds_list: list[BinderVar] = []
    for b in t.fields(vid, "binds"):
        annot_vid = t.field(b, "annot")
        annot_sort_vid = t.field(b, "annot_sort")
        binds_list.append(
            BinderVar(
                var=t.text(t.field(b, "var")),
                sort=t.text(t.field(b, "sort")),
                annot=t.text(annot_vid) if annot_vid is not None else None,
                annot_sort=(
                    t.text(annot_sort_vid) if annot_sort_vid is not None else None
                ),
            )
        )
    binds = tuple(binds_list)
    scoped = tuple(
        BinderArg(
            arg=t.text(t.field(a, "arg")),
            sort=t.text(t.field(a, "sort")),
        )
        for a in t.fields(vid, "scoped")
    )
    codomain = t.text(t.field(vid, "codomain"))
    ln, cl = t.line_col(vid)
    return BinderDecl(
        name=name,
        binds=binds,
        scoped=scoped,
        codomain=codomain,
        line=ln,
        col=cl,
    )


def _walk_vertex_kind_decl(t: _Tree, vid: str) -> VertexKindDecl:
    name = t.text(t.field(vid, "name"))
    kind_vid = t.field(vid, "kind")
    if kind_vid is None:
        raise ParseError(f"vertex_kind_decl missing kind at {vid}")
    kind_txt = t.text(kind_vid)
    dim_vid = t.field(vid, "dim")
    dim = int(t.text(dim_vid)) if dim_vid is not None else None
    ln, cl = t.line_col(vid)
    return VertexKindDecl(name=name, kind=kind_txt, dim=dim, line=ln, col=cl)


def _walk_edge_kind_decl(t: _Tree, vid: str) -> EdgeKindDecl:
    name = t.text(t.field(vid, "name"))
    src = t.text(t.field(vid, "src"))
    tgt = t.text(t.field(vid, "tgt"))
    arrow_vid = t.field(vid, "arrow")
    if arrow_vid is None:
        raise ParseError(f"edge_kind_decl missing arrow at {vid}")
    arrow_txt = t.text(arrow_vid)
    if arrow_txt == "->":
        directed = True
    elif arrow_txt == "--":
        directed = False
    else:
        raise ParseError(f"edge_kind_decl at {vid}: unknown arrow {arrow_txt!r}")
    ln, cl = t.line_col(vid)
    return EdgeKindDecl(
        name=name,
        src=src,
        tgt=tgt,
        directed=directed,
        line=ln,
        col=cl,
    )


def _walk_encoder_decl(t: _Tree, vid: str, line: int, col: int) -> EncoderDecl:
    name = t.text(t.field(vid, "name"))
    signature = t.text(t.field(vid, "signature"))
    sig_args = tuple(t.text(c) for c in t.fields(vid, "sig_args"))

    dims: list[SortDim] = []
    op_rules: list[EncoderRule] = []
    init_rules: list[EncoderInitRule] = []
    message_rules: list[EncoderMessageRule] = []
    update_rules: list[EncoderUpdateRule] = []
    iterations: int | None = None
    readout: LetExprNode | None = None
    var_inits: list[EncoderVarInit] = []

    for child in t.positional(vid):
        ck = t.kind(child)
        if ck == "encoder_dim":
            sort = t.text(t.field(child, "sort"))
            dim = int(t.text(t.field(child, "dim")))
            dims.append(SortDim(sort=sort, dim=dim))
        elif ck == "encoder_iterations":
            iterations = int(t.text(t.field(child, "iterations")))
        elif ck == "encoder_readout":
            readout = _walk_let_arith(t, t.field(child, "body"))
        elif ck == "encoder_op_rule":
            op = t.text(t.field(child, "op"))
            args = tuple(t.text(a) for a in t.fields(child, "args"))
            body = _walk_let_arith(t, t.field(child, "body"))
            state_v = t.field(child, "state")
            prefix_v = t.field(child, "prefix")
            mode = "plain"
            state_var = None
            prefix_var = None
            if state_v is not None:
                mode = "recurrent"
                state_var = t.text(state_v)
            elif prefix_v is not None:
                mode = "attention"
                prefix_var = t.text(prefix_v)
            cln, ccl = t.line_col(child)
            op_rules.append(
                EncoderRule(
                    op=op,
                    args=args,
                    body=body,
                    mode=mode,
                    state_var=state_var,
                    prefix_var=prefix_var,
                    line=cln,
                    col=ccl,
                )
            )
        elif ck == "encoder_init_rule":
            init_rules.append(
                EncoderInitRule(
                    kind=t.text(t.field(child, "kind")),
                    arg=t.text(t.field(child, "arg")),
                    body=_walk_let_arith(t, t.field(child, "body")),
                )
            )
        elif ck == "encoder_message_rule":
            message_rules.append(
                EncoderMessageRule(
                    edge_kind=t.text(t.field(child, "edge_kind")),
                    src=t.text(t.field(child, "src")),
                    tgt=t.text(t.field(child, "tgt")),
                    body=_walk_let_arith(t, t.field(child, "body")),
                )
            )
        elif ck == "encoder_update_rule":
            update_rules.append(
                EncoderUpdateRule(
                    vertex_kind=t.text(t.field(child, "vertex_kind")),
                    self_var=t.text(t.field(child, "self")),
                    msgs_var=t.text(t.field(child, "msgs")),
                    body=_walk_let_arith(t, t.field(child, "body")),
                )
            )
        elif ck == "encoder_var_init":
            vs_vid = t.field(child, "var_sort")
            if vs_vid is None:
                raise ParseError(f"encoder_var_init at {child} missing var_sort")
            annot_vid = t.field(child, "annot_sort")
            ty_vid = t.field(child, "ty")
            ln, cl = t.line_col(child)
            var_inits.append(
                EncoderVarInit(
                    var_sort=t.text(vs_vid),
                    annot_sort=(t.text(annot_vid) if annot_vid is not None else None),
                    ty=t.text(ty_vid) if ty_vid is not None else None,
                    body=_walk_let_arith(t, t.field(child, "body")),
                    line=ln,
                    col=cl,
                )
            )

    return EncoderDecl(
        name=name,
        signature=signature,
        sig_args=sig_args,
        dims=tuple(dims),
        op_rules=tuple(op_rules),
        init_rules=tuple(init_rules),
        message_rules=tuple(message_rules),
        update_rules=tuple(update_rules),
        iterations=iterations,
        readout=readout,
        var_inits=tuple(var_inits),
        line=line,
        col=col,
    )


def _walk_decoder_decl(t: _Tree, vid: str, line: int, col: int) -> DecoderDecl:
    name = t.text(t.field(vid, "name"))
    signature = t.text(t.field(vid, "signature"))
    sig_args = tuple(t.text(c) for c in t.fields(vid, "sig_args"))
    depth_vid = t.field(vid, "depth")
    depth = int(t.text(depth_vid)) if depth_vid is not None else 8

    dims: list[SortDim] = []
    structure_body: LetExprNode | None = None
    structure_arg: str | None = None
    primitive_body: LetExprNode | None = None
    primitive_arg: str | None = None
    factor_body: LetExprNode | None = None
    factor_arg: str | None = None
    binder_body: LetExprNode | None = None
    binder_arg: str | None = None
    recursive_default = False

    for child in t.positional(vid):
        ck = t.kind(child)
        if ck == "decoder_dim":
            sort = t.text(t.field(child, "sort"))
            dim = int(t.text(t.field(child, "dim")))
            dims.append(SortDim(sort=sort, dim=dim))
        elif ck == "decoder_structure":
            structure_arg = t.text(t.field(child, "arg"))
            structure_body = _walk_let_arith(t, t.field(child, "body"))
        elif ck == "decoder_primitive":
            primitive_arg = t.text(t.field(child, "arg"))
            primitive_body = _walk_let_arith(t, t.field(child, "body"))
        elif ck == "decoder_factor":
            factor_arg = t.text(t.field(child, "arg"))
            factor_body = _walk_let_arith(t, t.field(child, "body"))
        elif ck == "decoder_binder_select":
            binder_arg = t.text(t.field(child, "arg"))
            binder_body = _walk_let_arith(t, t.field(child, "body"))
        elif ck == "decoder_body_default":
            recursive_default = True

    return DecoderDecl(
        name=name,
        signature=signature,
        sig_args=sig_args,
        depth=depth,
        dims=tuple(dims),
        structure=structure_body,
        structure_arg=structure_arg,
        primitive=primitive_body,
        primitive_arg=primitive_arg,
        factor=factor_body,
        factor_arg=factor_arg,
        binder_select=binder_body,
        binder_select_arg=binder_arg,
        recursive_default=recursive_default,
        line=line,
        col=col,
    )


def _walk_loss_decl(t: _Tree, vid: str, line: int, col: int) -> LossDecl:
    name = t.text(t.field(vid, "name"))
    weight_vid = t.field(vid, "weight")
    weight = _walk_let_arith(t, weight_vid) if weight_vid is not None else None
    body = _walk_let_arith(t, t.field(vid, "body"))

    attachment = LossAttachment(attachment_kind="global")
    att_vid = t.field(vid, "attachment")
    if att_vid is not None:
        kind_vid = t.field(att_vid, "kind")
        tgt_vid = t.field(att_vid, "target")
        rule_vid = t.field(att_vid, "rule_name")
        ded_vid = t.field(att_vid, "deduction")
        chart_vid = t.field(att_vid, "chart_of")
        if rule_vid is not None and ded_vid is not None:
            attachment = LossAttachment(
                attachment_kind="rule",
                target=t.text(rule_vid),
                rule_deduction=t.text(ded_vid),
            )
        elif chart_vid is not None:
            attachment = LossAttachment(
                attachment_kind="chart",
                target=t.text(chart_vid),
            )
        elif tgt_vid is not None:
            if kind_vid is None:
                raise ParseError(f"loss_attachment at {att_vid}: missing kind")
            k = t.text(kind_vid)
            if k not in ("program", "deduction", "encoder", "decoder"):
                raise ParseError(f"loss_attachment at {att_vid}: unknown kind {k!r}")
            attachment = LossAttachment(
                attachment_kind=k,  # type: ignore[arg-type]
                target=t.text(tgt_vid),
            )

    return LossDecl(
        name=name,
        weight=weight,
        attachment=attachment,
        body=body,
        line=line,
        col=col,
    )
