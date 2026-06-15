"""AST -> ``.qvr`` source emit.

Walks a [`quivers.dsl.ast_nodes.Module`][quivers.dsl.ast_nodes.Module] and produces canonical
``.qvr`` source text under the homogenized surface (type / morphism
/ program / sample / observe / return / option-block / etc.).

The emit is a one-way printer, not a panproto `ParseEmitLens`:
quivers' AST is already the resolved form. The printer's contract is
*semantic*: the emitted source, re-parsed by `quivers.dsl.loads`,
produces a `Module` that compiles to the same program as the
original AST.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    CompositionDecl,
    ContinuousConstructor,
    DiscreteConstructor,
    DrawArgList,
    DrawArgMatrix,
    DrawArgName,
    DrawArgScalar,
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
    MarginalizeStep,
    Module,
    MorphismDecl,
    ObserveStep,
    OptionCall,
    OptionEntry,
    OptionFlag,
    OptionList,
    OptionName,
    OptionNumber,
    OptionString,
    OptionValue,
    ProgramDecl,
    ProgramStep,
    ReturnStep,
    SampleStep,
    Statement,
    ObjectCoproduct,
    ObjectDecl,
    ObjectEffectApply,
    TypeEnumSet,
    ObjectExpr,
    TypeFreeMonoid,
    TypeFreeResiduated,
    TypeFromExpr,
    TypeInitializer,
    TypeName,
    ObjectProduct,
    ObjectSlash,
)


def module_to_source(module: Module) -> str:
    """Serialize a `Module` AST to canonical ``.qvr`` source."""
    parts: list[str] = []
    last_was_block = False
    for stmt in module.statements:
        emitted = _emit_statement(stmt)
        if last_was_block:
            parts.append("")
        parts.append(emitted)
        last_was_block = isinstance(stmt, ProgramDecl)
    return "\n".join(parts) + "\n"


def _emit_statement(stmt: Statement) -> str:
    if isinstance(stmt, CompositionDecl):
        head = f"composition {stmt.name}"
        if stmt.level is not None:
            head = f"{head} at {stmt.level}"
        return head
    if isinstance(stmt, ObjectDecl):
        return f"object {stmt.name} : {_emit_type_init(stmt.init)}"
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


# ---------------------------------------------------------------------------
# type initializers and expressions
# ---------------------------------------------------------------------------


def _emit_type_init(init: TypeInitializer) -> str:
    if isinstance(init, TypeEnumSet):
        return "{" + ", ".join(init.elements) + "}"
    if isinstance(init, TypeFreeMonoid):
        return f"FreeMonoid({init.generators}, max_length={init.max_length})"
    if isinstance(init, TypeFreeResiduated):
        body = init.generators
        if init.depth != 1:
            body = f"{body}, depth={init.depth}"
        if init.ops:
            body = body + ", ops=[" + ", ".join(init.ops) + "]"
        return f"FreeResiduated({body})"
    if isinstance(init, TypeFromExpr):
        return _emit_type(init.expr)
    raise NotImplementedError(
        f"emit: type initializer {type(init).__name__} not supported"
    )


def _emit_type(t: ObjectExpr) -> str:
    if isinstance(t, TypeName):
        return t.name
    if isinstance(t, ObjectProduct):
        return " * ".join(_emit_type(c) for c in t.components)
    if isinstance(t, ObjectCoproduct):
        return " + ".join(_emit_type(c) for c in t.components)
    if isinstance(t, ObjectSlash):
        return f"{_emit_type(t.result)} {t.direction} {_emit_type(t.argument)}"
    if isinstance(t, ObjectEffectApply):
        args = ", ".join(_emit_type(a) for a in t.args)
        return f"{t.effect}({args})"
    if isinstance(t, DiscreteConstructor):
        # ``FinSet N``: Haskell-style space-separated positional args,
        # with any keyword options demoted to the trailing option block.
        head = " ".join((t.constructor, *t.args))
        if t.kwargs:
            opts = ", ".join(f"{k}={v}" for k, v in t.kwargs.items())
            return f"{head} [{opts}]"
        return head
    if isinstance(t, ContinuousConstructor):
        head = " ".join((t.constructor, *t.args))
        if t.kwargs:
            opts = ", ".join(f"{k}={v}" for k, v in t.kwargs.items())
            return f"{head} [{opts}]"
        return head
    raise NotImplementedError(f"emit: type expression {type(t).__name__} not supported")


# ---------------------------------------------------------------------------
# option block
# ---------------------------------------------------------------------------


def _emit_options(options: tuple[OptionEntry, ...]) -> str:
    if not options:
        return ""
    return " [" + ", ".join(_emit_option_entry(e) for e in options) + "]"


def _emit_option_entry(entry: OptionEntry) -> str:
    if isinstance(entry.value, OptionFlag):
        return entry.key
    return f"{entry.key}={_emit_option_value(entry.value)}"


def _emit_option_value(value: OptionValue) -> str:
    if isinstance(value, OptionFlag):
        return ""
    if isinstance(value, OptionName):
        return value.value
    if isinstance(value, OptionNumber):
        return _emit_number(value.value)
    if isinstance(value, OptionString):
        return f'"{value.value}"'
    if isinstance(value, OptionList):
        return "[" + ", ".join(_emit_option_value(v) for v in value.items) + "]"
    if isinstance(value, OptionCall):
        args = ", ".join(_emit_option_value(v) for v in value.args)
        return f"{value.func}({args})"
    raise NotImplementedError(
        f"emit: option value {type(value).__name__} not supported"
    )


# ---------------------------------------------------------------------------
# morphism / program
# ---------------------------------------------------------------------------


def _emit_morphism(decl: MorphismDecl) -> str:
    head = (
        f"morphism {decl.name} : "
        f"{_emit_type(decl.domain)} -> {_emit_type(decl.codomain)}"
        f"{_emit_options(decl.options)}"
    )
    if decl.init_family is not None:
        args = ", ".join(_emit_arg(a) for a in decl.init_family.args)
        return f"{head} ~ {decl.init_family.family}({args})"
    if decl.init_expr is not None:
        return f"{head} ~ {_emit_expr(decl.init_expr)}"
    return head


def _emit_program(decl: ProgramDecl) -> str:
    domain = _emit_type(decl.domain)
    codomain = _emit_type(decl.codomain)
    params = ""
    if decl.params:
        params = "(" + ", ".join(decl.params) + ")"
    head = (
        f"program {decl.name}{params} : {domain} -> {codomain}"
        f"{_emit_options(decl.options)}:"
    )
    lines: list[str] = [head]
    for step in decl.draws:
        lines.extend(_emit_program_step(step, indent=1))
    lines.append(f"    return {_emit_return(decl)}")
    return "\n".join(lines)


def _emit_return(decl: ProgramDecl) -> str:
    if decl.return_labels is None:
        if len(decl.return_vars) == 1:
            return decl.return_vars[0]
        return "(" + ", ".join(decl.return_vars) + ")"
    pairs = ", ".join(
        f"{lab}: {var}" for lab, var in zip(decl.return_labels, decl.return_vars)
    )
    return f"({pairs})"


def _emit_program_step(step: ProgramStep, *, indent: int = 1) -> list[str]:
    pad = "    " * indent
    if isinstance(step, SampleStep):
        binder = _emit_var_pattern(step.vars)
        if step.index is not None:
            binder = f"{binder} : {_emit_type(step.index)}"
        morphism = step.morphism
        if step.args is not None:
            morphism = (
                f"{step.morphism}(" + ", ".join(_emit_arg(a) for a in step.args) + ")"
            )
        return [f"{pad}sample {binder} <- {morphism}{_emit_options(step.options)}"]
    if isinstance(step, ObserveStep):
        binder = step.var
        if step.index is not None:
            binder = f"{binder} : {_emit_type(step.index)}"
        morphism = step.morphism
        if step.args is not None:
            morphism = (
                f"{step.morphism}(" + ", ".join(_emit_arg(a) for a in step.args) + ")"
            )
        return [f"{pad}observe {binder} <- {morphism}{_emit_options(step.options)}"]
    if isinstance(step, MarginalizeStep):
        binder = step.var
        if step.index is not None:
            binder = f"{binder} : {_emit_type(step.index)}"
        morphism = step.morphism
        if step.args is not None:
            morphism = (
                f"{step.morphism}(" + ", ".join(_emit_arg(a) for a in step.args) + ")"
            )
        head = f"{pad}marginalize {binder} <- {morphism}{_emit_options(step.options)}:"
        nested = [head]
        for inner in step.scope:
            nested.extend(_emit_program_step(inner, indent=indent + 1))
        return nested
    if isinstance(step, LetStep):
        return [f"{pad}let {step.name} = {_emit_let_expr(step.value)}"]
    if isinstance(step, ReturnStep):
        return [f"{pad}return {_emit_var_pattern(step.vars)}"]
    raise NotImplementedError(f"emit: program step {type(step).__name__} not supported")


def _emit_var_pattern(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    return "(" + ", ".join(names) + ")"


# ---------------------------------------------------------------------------
# expressions / let-arith
# ---------------------------------------------------------------------------


def _emit_arg(arg: object) -> str:
    if isinstance(arg, DrawArgScalar):
        return _emit_number(arg.value)
    if isinstance(arg, DrawArgName):
        return arg.text
    if isinstance(arg, DrawArgList):
        return "[" + ", ".join(_emit_arg_atom(e) for e in arg.elements) + "]"
    if isinstance(arg, DrawArgMatrix):
        rows = ", ".join(
            "[" + ", ".join(_emit_arg_atom(e) for e in row.elements) + "]"
            for row in arg.rows
        )
        return f"[{rows}]"
    if isinstance(arg, (int, float)) and not isinstance(arg, bool):
        return _emit_number(float(arg))
    if isinstance(arg, str):
        return arg
    raise TypeError(f"_emit_arg: unsupported argument {arg!r}")


def _emit_arg_atom(value: str | float) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _emit_number(float(value))
    return str(value)


def _emit_number(value: float) -> str:
    if value == int(value):
        return f"{value:.1f}"
    return repr(value)


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
    raise NotImplementedError(f"emit: expression {type(e).__name__} not supported")


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
    raise NotImplementedError(f"emit: let-expression {type(e).__name__} not supported")


__all__ = ["module_to_source"]
