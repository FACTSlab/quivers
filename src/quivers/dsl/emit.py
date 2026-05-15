"""AST → ``.qvr`` source emit.

Walks a :class:`~quivers.dsl.ast_nodes.Module` and produces canonical
``.qvr`` source text. The emit covers the subset of AST variants the
formula-frontend builds (object / morphism / let / program / export
declarations, plus the let-arithmetic and program-step nodes those
declarations contain); it raises :class:`NotImplementedError` for
variants outside that subset rather than guessing a serialisation.

The emit is a one-way printer, not a panproto :class:`ParseEmitLens`:
quivers' AST is already the resolved form, and reconstructing the
exact source from it (preserving original whitespace, comments,
token positions) is a separate concern.  This printer's contract is
*semantic*: the emitted source, re-parsed by :func:`quivers.dsl.loads`,
produces a :class:`Module` that compiles to the same program as the
original AST.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    AliasDecl,
    BindStep,
    ExportDecl,
    Expr,
    ExprChangeBase,
    ExprCompose,
    ExprFromData,
    ExprIdent,
    ExprTensorProduct,
    LetDecl,
    LetExprBinOp,
    LetExprCall,
    LetExprIndex,
    LetExprLiteral,
    LetExprNode,
    LetExprString,
    LetExprUnaryOp,
    LetExprVar,
    LetStep,
    Module,
    MorphismDecl,
    ObjectDecl,
    ProgramDecl,
    SpaceConstructor,
    SpaceDecl,
    SpaceExpr,
    SpaceName,
    SpaceProduct,
    Statement,
    TypeCoproduct,
    TypeExpr,
    TypeName,
    TypeProduct,
)


def module_to_source(module: Module) -> str:
    """Serialize a :class:`Module` AST to canonical ``.qvr`` source."""
    parts: list[str] = []
    last_was_program = False
    for stmt in module.statements:
        line = _emit_statement(stmt)
        if last_was_program:
            parts.append("")
        parts.append(line)
        last_was_program = isinstance(stmt, ProgramDecl)
    return "\n".join(parts) + "\n"


def _emit_statement(stmt: Statement) -> str:
    if isinstance(stmt, ObjectDecl):
        return _emit_object(stmt)
    if isinstance(stmt, SpaceDecl):
        return _emit_space(stmt)
    if isinstance(stmt, AliasDecl):
        return f"alias {stmt.name} = {_emit_type(stmt.type_expr)}"
    if isinstance(stmt, MorphismDecl):
        return _emit_morphism(stmt)
    if isinstance(stmt, LetDecl):
        return f"let {stmt.name} = {_emit_expr(stmt.expr)}"
    if isinstance(stmt, ProgramDecl):
        return _emit_program(stmt)
    if isinstance(stmt, ExportDecl):
        return f"export {_emit_expr(stmt.expr)}"
    raise NotImplementedError(
        f"emit: statement variant {type(stmt).__name__} not supported"
    )


def _emit_object(decl: ObjectDecl) -> str:
    if decl.type_expr is None:
        raise NotImplementedError(
            f"emit: object {decl.name!r} without type_expr "
            f"(EnumSet / FreeResiduated init) not supported"
        )
    return f"object {decl.name} : {_emit_type(decl.type_expr)}"


def _emit_space(decl: SpaceDecl) -> str:
    return f"space {decl.name} : {_emit_space_expr(decl.space_expr)}"


def _emit_morphism(decl: MorphismDecl) -> str:
    head = (
        f"{decl.morphism_kind} {decl.name} : "
        f"{_emit_type(decl.domain)} -> {_emit_type(decl.codomain)}"
    )
    if decl.init_expr is not None:
        head = f"{head} = {_emit_expr(decl.init_expr)}"
    if decl.prior is not None:
        args = ", ".join(_emit_arg(a) for a in decl.prior.args)
        head = f"{head} ~ {decl.prior.family}({args})"
        if decl.prior.axes is not None:
            over = ", ".join(decl.prior.axes.over)
            head = f"{head} over ({over})"
    return head


def _emit_program(decl: ProgramDecl) -> str:
    domain = _emit_type(decl.domain)
    codomain = _emit_type(decl.codomain)
    head = f"program {decl.name} : {domain} -> {codomain}"
    if decl.effects:
        head = f"{head} ! {', '.join(sorted(decl.effects))}"
    lines: list[str] = [head]
    for step in decl.draws:
        lines.append("    " + _emit_program_step(step))
    lines.append(f"    return {', '.join(decl.return_vars)}")
    return "\n".join(lines)


def _emit_program_step(step) -> str:
    if isinstance(step, BindStep):
        return _emit_bind(step)
    if isinstance(step, LetStep):
        return f"let {step.name} = {_emit_let_expr(step.value)}"
    raise NotImplementedError(
        f"emit: program step variant {type(step).__name__} not supported"
    )


def _emit_bind(step: BindStep) -> str:
    binder = step.vars[0] if len(step.vars) == 1 else ("(" + ", ".join(step.vars) + ")")
    if step.index is not None:
        binder = f"{binder} : {_emit_type(step.index)}"
    family_call = step.morphism
    if step.args is not None:
        family_call = f"{step.morphism}({', '.join(_emit_arg(a) for a in step.args)})"
    prefix = {"sample": "", "score": "observe ", "marginal": "marginalize "}[step.mode]
    return f"{prefix}{binder} <- {family_call}"


def _emit_arg(arg: str | float) -> str:
    if isinstance(arg, (int, float)) and not isinstance(arg, bool):
        return _emit_number(float(arg))
    return str(arg)


def _emit_number(value: float) -> str:
    if value == int(value):
        return f"{value:.1f}"
    return repr(value)


def _emit_type(t: TypeExpr) -> str:
    if isinstance(t, TypeName):
        return t.name
    if isinstance(t, TypeProduct):
        return " * ".join(_emit_type(c) for c in t.components)
    if isinstance(t, TypeCoproduct):
        return " + ".join(_emit_type(c) for c in t.components)
    raise NotImplementedError(
        f"emit: type expression variant {type(t).__name__} not supported"
    )


def _emit_space_expr(s: SpaceExpr) -> str:
    if isinstance(s, SpaceName):
        return s.name
    if isinstance(s, SpaceConstructor):
        args = ", ".join(_emit_arg(a) for a in s.args)
        return f"{s.constructor}({args})"
    if isinstance(s, SpaceProduct):
        return " * ".join(_emit_space_expr(c) for c in s.components)
    raise NotImplementedError(
        f"emit: space expression variant {type(s).__name__} not supported"
    )


def _emit_expr(e: Expr) -> str:
    if isinstance(e, ExprIdent):
        return e.name
    if isinstance(e, ExprFromData):
        return f'from_data("{e.key}")'
    if isinstance(e, ExprCompose):
        return f"{_emit_expr(e.left)} {e.op} {_emit_expr(e.right)}"
    if isinstance(e, ExprTensorProduct):
        return f"{_emit_expr(e.left)} @ {_emit_expr(e.right)}"
    if isinstance(e, ExprChangeBase):
        return f"{_emit_expr(e.inner)}.change_base({_emit_expr(e.transform)})"
    raise NotImplementedError(
        f"emit: expression variant {type(e).__name__} not supported"
    )


def _emit_let_expr(e: LetExprNode) -> str:
    if isinstance(e, LetExprVar):
        return e.name
    if isinstance(e, LetExprLiteral):
        return _emit_number(float(e.value))
    if isinstance(e, LetExprString):
        return f'"{e.value}"'
    if isinstance(e, LetExprBinOp):
        return f"({_emit_let_expr(e.left)} {e.op} {_emit_let_expr(e.right)})"
    if isinstance(e, LetExprUnaryOp):
        return f"-{_emit_let_expr(e.operand)}"
    if isinstance(e, LetExprCall):
        args = ", ".join(_emit_let_expr(a) for a in e.args)
        return f"{e.func}({args})"
    if isinstance(e, LetExprIndex):
        idx = ", ".join(_emit_let_expr(i) for i in e.indices)
        return f"{_emit_let_expr(e.array)}[{idx}]"
    raise NotImplementedError(
        f"emit: let-expression variant {type(e).__name__} not supported"
    )
