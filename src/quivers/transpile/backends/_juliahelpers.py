"""Shared SchemaBuilder helpers for the Julia tree-sitter grammar."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import panproto


class JlCtx:
    def __init__(self, sb: panproto.SchemaBuilder) -> None:
        self._sb = sb
        self._n = 0

    def fresh(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def v(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)


def ident(ctx: JlCtx, text: str) -> str:
    vid = ctx.v(ctx.fresh("id"), "identifier")
    ctx.lit(vid, text)
    return vid


def integer(ctx: JlCtx, value: int) -> str:
    vid = ctx.v(ctx.fresh("int"), "integer_literal")
    ctx.lit(vid, str(value))
    return vid


def float_lit(ctx: JlCtx, value: float) -> str:
    vid = ctx.v(ctx.fresh("flt"), "float_literal")
    ctx.lit(vid, repr(value))
    return vid


def number(ctx: JlCtx, value: float) -> str:
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return integer(ctx, int(value))
    return float_lit(ctx, float(value))


def arg(ctx: JlCtx, raw: str | float) -> str:
    if isinstance(raw, str):
        return ident(ctx, raw)
    return number(ctx, raw)


def operator(ctx: JlCtx, text: str) -> str:
    vid = ctx.v(ctx.fresh("op"), "operator")
    ctx.lit(vid, text)
    return vid


def call(ctx: JlCtx, callee: str, positional: tuple[str, ...]) -> str:
    """Build ``callee(arg1, arg2, ...)``."""
    c = ctx.v(ctx.fresh("call"), "call_expression")
    args = ctx.v(ctx.fresh("args"), "argument_list")
    ctx.e(c, callee)
    ctx.e(c, args)
    for pid in positional:
        ctx.e(args, pid)
    return c


def tilde_assignment(ctx: JlCtx, lhs: str, rhs: str) -> str:
    """Build ``<lhs> ~ <rhs>`` (Turing.jl-style sampling)."""
    asn = ctx.v(ctx.fresh("ca"), "compound_assignment_expression")
    ctx.e(asn, lhs)
    ctx.e(asn, operator(ctx, "~"))
    ctx.e(asn, rhs)
    return asn


def macro_call(ctx: JlCtx, macro_name: str, body_vid: str) -> str:
    """Build ``@macro_name body`` (no parentheses; long-form).

    Used for the ``@model function ... end`` / ``@gen function ... end``
    style invocation where the macro body is a single expression
    (typically a function_definition).
    """
    mc = ctx.v(ctx.fresh("mc"), "macrocall_expression")
    macro_id = ctx.v(ctx.fresh("mid"), "macro_identifier")
    ctx.e(macro_id, ident(ctx, macro_name))
    margs = ctx.v(ctx.fresh("mal"), "macro_argument_list")
    ctx.e(margs, body_vid)
    ctx.e(mc, macro_id)
    ctx.e(mc, margs)
    return mc


def macro_call_paren(
    ctx: JlCtx, macro_name: str, args: tuple[str, ...]
) -> str:
    """Build ``@macro_name(arg1, arg2, ...)`` (parenthesised short-form).

    Used by Gen.jl's ``@trace(dist, :address)``. The parenthesised
    form uses ``argument_list`` instead of ``macro_argument_list``.
    """
    mc = ctx.v(ctx.fresh("mc"), "macrocall_expression")
    macro_id = ctx.v(ctx.fresh("mid"), "macro_identifier")
    ctx.e(macro_id, ident(ctx, macro_name))
    al = ctx.v(ctx.fresh("al"), "argument_list")
    for a in args:
        ctx.e(al, a)
    ctx.e(mc, macro_id)
    ctx.e(mc, al)
    return mc


def function_def(
    ctx: JlCtx,
    *,
    name: str,
    params: tuple[str, ...],
    body_vid: str,
) -> str:
    """Build ``function <name>(<p1>, <p2>, ...) <body> end``."""
    fn = ctx.v(ctx.fresh("fn"), "function_definition")
    sig = ctx.v(ctx.fresh("sig"), "signature")
    callsig = ctx.v(ctx.fresh("scall"), "call_expression")
    args = ctx.v(ctx.fresh("sargs"), "argument_list")
    for p in params:
        ctx.e(args, ident(ctx, p))
    ctx.e(callsig, ident(ctx, name))
    ctx.e(callsig, args)
    ctx.e(sig, callsig)
    ctx.e(fn, sig)
    ctx.e(fn, body_vid)
    return fn
