"""AST -> ``.qvr`` source emit.

Walks a [`quivers.dsl.ast_nodes.Module`][quivers.dsl.ast_nodes.Module]
and produces canonical ``.qvr`` source text.

The emit is a one-way printer, not a panproto `ParseEmitLens`:
quivers' AST is already the resolved form. The printer's contract is
*semantic*: the emitted source, re-parsed by `quivers.dsl.loads`,
produces a `Module` that compiles to the same program as the
original AST. The printer is additionally a *canonical fixed point*:
``emit(parse(emit(ast))) == emit(ast)`` byte for byte.

Canonical formatting conventions:

* 4-space indentation per nesting level.
* One blank line between top-level statements; a single trailing
  newline at end of file.
* Option blocks, constructor brace options, lists, and tuples are
  always emitted inline.
* Doc comments are emitted as ``#! text`` lines directly above the
  declaration they document.
* Numeric option values and draw-arg scalars render integral floats
  as ``X.0``; constructor keyword arguments render ``int`` values
  without a decimal point and ``float`` values in ``repr`` form.
* Convenience fields that mirror option-block entries (``axes``,
  ``via``, ``over``, ``reduction`` on program steps; ``level`` on
  ``composition`` rides its own option block) are not re-emitted
  from the mirror: the option block is the single surface source.
"""

from __future__ import annotations

import math

from quivers.dsl.ast_nodes.declarations import (
    BundleDecl,
    CategoryDecl,
    CompositionDecl,
    CompositionRuleEntry,
    ContractionDecl,
    DecoderDecl,
    DeductionDecl,
    DefineDecl,
    EncoderDecl,
    ExportDecl,
    LexiconCategory,
    LexiconCategoryFixed,
    LexiconCategoryRestricted,
    LexiconCategoryWildcard,
    LexiconEntry,
    LossDecl,
    MorphismDecl,
    MorphismParam,
    ObjectDecl,
    ObjectParam,
    OptionCall,
    OptionEntry,
    OptionFlag,
    OptionList,
    OptionName,
    OptionNumber,
    OptionString,
    OptionValue,
    ProgramDecl,
    ProgramParam,
    RuleDecl,
    ScalarParam,
    SchemaDecl,
    SequentRule,
    SignatureDecl,
    Statement,
    TypeEnumSet,
    TypeFreeMonoid,
    TypeFreeResiduated,
    TypeFromExpr,
    TypeInitializer,
)
from quivers.dsl.ast_nodes.expressions import (
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
)
from quivers.dsl.ast_nodes.let_expressions import (
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
)
from quivers.dsl.ast_nodes.module import Module
from quivers.dsl.ast_nodes.objects import (
    ContinuousConstructor,
    DiscreteConstructor,
    ObjectCoproduct,
    ObjectEffectApply,
    ObjectExpr,
    ObjectProduct,
    ObjectSlash,
    TypeName,
)
from quivers.dsl.ast_nodes.program_steps import (
    BindStep,
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgScalar,
    DrawStep,
    GroupedBodyObserveStep,
    GroupedLatentInitStep,
    GroupedMarginalizeStep,
    LetStep,
    MarginalizeStep,
    ObserveStep,
    PlateDrawStep,
    ProgramStep,
    ReturnStep,
    SampleStep,
    ScoreStep,
    VectorisedObserveStep,
)
from quivers.dsl.ast_nodes.structural import (
    BinderDecl,
    BinderVar,
    ConstructorDecl,
    EdgeKindDecl,
    EncoderInitRule,
    EncoderMessageRule,
    EncoderRule,
    EncoderUpdateRule,
    EncoderVarInit,
    SortDecl,
    VertexKindDecl,
)


class EmitError(Exception):
    """Raised when an AST value has no valid ``.qvr`` surface form."""


_INDENT = "    "


def module_to_source(module: Module) -> str:
    """Serialize a `Module` AST to canonical ``.qvr`` source."""
    parts = [_emit_statement(stmt, 0) for stmt in module.statements]
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _pad(indent: int) -> str:
    return _INDENT * indent


def _doc_lines(docs: tuple[str, ...], indent: int) -> list[str]:
    return [f"{_pad(indent)}#! {doc.strip()}" for doc in docs]


def _emit_number(value: float) -> str:
    if not math.isfinite(value):
        raise EmitError(f"emit: non-finite numeric literal {value!r}")
    if value == int(value) and abs(value) < 1e16:
        return f"{value:.1f}"
    return repr(value)


def _emit_string(value: str) -> str:
    # The parser stores a string's inner text verbatim: it strips the
    # surrounding quotes but leaves every escape sequence exactly as it
    # appeared in source. Re-wrapping in quotes therefore reproduces the
    # source string, and re-escaping here would double every backslash
    # and quote on each round-trip.
    return f'"{value}"'


def _emit_names(names: tuple[str, ...]) -> str:
    if not names:
        raise EmitError("emit: declaration with an empty name list")
    return ", ".join(names)


# ---------------------------------------------------------------------------
# option block
# ---------------------------------------------------------------------------


def _emit_options(options: tuple[OptionEntry, ...]) -> str:
    if not options:
        return ""
    return " [" + ", ".join(_emit_option_entry(e) for e in options) + "]"


def _emit_pragma(options: tuple[OptionEntry, ...]) -> str:
    if not options:
        return ""
    return " #[" + ", ".join(_emit_option_entry(e) for e in options) + "]"


def _emit_option_entry(entry: OptionEntry) -> str:
    if isinstance(entry.value, OptionFlag):
        return entry.key
    return f"{entry.key}={_emit_option_value(entry.value)}"


def _emit_option_value(value: OptionValue) -> str:
    if isinstance(value, OptionName):
        return value.value
    if isinstance(value, OptionNumber):
        return _emit_number(value.value)
    if isinstance(value, OptionString):
        return _emit_string(value.value)
    if isinstance(value, OptionList):
        return "[" + ", ".join(_emit_option_value(v) for v in value.items) + "]"
    if isinstance(value, OptionCall):
        args = ", ".join(_emit_option_value(v) for v in value.args)
        return f"{value.func}({args})"
    if isinstance(value, OptionFlag):
        raise EmitError("emit: OptionFlag is only valid as a bare option key")
    raise EmitError(f"emit: unknown OptionValue kind {type(value).__name__!r}")


# ---------------------------------------------------------------------------
# type expressions
# ---------------------------------------------------------------------------

_OBJECT_PREC_COPRODUCT = 1
_OBJECT_PREC_SLASH = 2
_OBJECT_PREC_PRODUCT = 3
_OBJECT_PREC_ATOM = 4


def _object_prec(t: ObjectExpr) -> int:
    if isinstance(t, ObjectCoproduct):
        return _OBJECT_PREC_COPRODUCT
    if isinstance(t, ObjectSlash):
        return _OBJECT_PREC_SLASH
    if isinstance(t, ObjectProduct):
        return _OBJECT_PREC_PRODUCT
    return _OBJECT_PREC_ATOM


def _emit_object_child(t: ObjectExpr, min_prec: int) -> str:
    text = _emit_object_expr(t)
    if _object_prec(t) < min_prec:
        return f"({text})"
    return text


def _emit_object_expr(t: ObjectExpr) -> str:
    if isinstance(t, TypeName):
        return t.name
    if isinstance(t, ObjectProduct):
        return " * ".join(
            _emit_object_child(c, _OBJECT_PREC_PRODUCT) for c in t.components
        )
    if isinstance(t, ObjectCoproduct):
        return " + ".join(
            _emit_object_child(c, _OBJECT_PREC_COPRODUCT) for c in t.components
        )
    if isinstance(t, ObjectSlash):
        left = _emit_object_child(t.result, _OBJECT_PREC_SLASH)
        right = _emit_object_child(t.argument, _OBJECT_PREC_SLASH + 1)
        return f"{left} {t.direction} {right}"
    if isinstance(t, ObjectEffectApply):
        if not t.args:
            raise EmitError(f"emit: effect apply {t.effect!r} with no arguments")
        args = ", ".join(_emit_object_expr(a) for a in t.args)
        return f"{t.effect}({args})"
    if isinstance(t, DiscreteConstructor):
        if t.kwargs:
            raise EmitError(
                "emit: discrete constructor "
                f"{t.constructor!r} does not admit keyword options"
            )
        return " ".join((t.constructor, *t.args))
    if isinstance(t, ContinuousConstructor):
        head = " ".join((t.constructor, *t.args))
        if t.kwargs:
            opts = ", ".join(
                f"{k}={_emit_constructor_kwarg(v)}" for k, v in t.kwargs.items()
            )
            return f"{head} {{{opts}}}"
        return head
    raise EmitError(f"emit: unknown ObjectExpr kind {type(t).__name__!r}")


def _emit_constructor_kwarg(value: float | int | str) -> str:
    if isinstance(value, bool):
        raise EmitError("emit: boolean constructor keyword values have no surface")
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise EmitError(f"emit: non-finite constructor keyword value {value!r}")
    return repr(value)


def _emit_type_init(init: TypeInitializer) -> str:
    if isinstance(init, TypeEnumSet):
        return "{" + ", ".join(init.elements) + "}"
    if isinstance(init, TypeFreeMonoid):
        return f"FreeMonoid({init.generators}, max_length={init.max_length})"
    if isinstance(init, TypeFreeResiduated):
        body = init.generators
        if init.depth != 1:
            body = f"{body}, depth={init.depth}"
        if init.ops and init.ops != ("slash",):
            body = body + ", ops=[" + ", ".join(init.ops) + "]"
        return f"FreeResiduated({body})"
    if isinstance(init, TypeFromExpr):
        return _emit_object_expr(init.expr)
    raise EmitError(f"emit: unknown TypeInitializer kind {type(init).__name__!r}")


# ---------------------------------------------------------------------------
# value (morphism) expressions
# ---------------------------------------------------------------------------

_EXPR_PREC_COMPOSE = 1
_EXPR_PREC_TENSOR = 2
_EXPR_PREC_POSTFIX = 3
_EXPR_PREC_ATOM = 4

_COMPOSE_OPS = (">>", "<<")

_POSTFIX_KINDS = (
    ExprChangeBase,
    ExprCurry,
    ExprDagger,
    ExprFreeze,
    ExprMarginalize,
    ExprTrace,
)


def _expr_prec(e: Expr) -> int:
    if isinstance(e, (ExprCompose, ExprTransCompose)):
        return _EXPR_PREC_COMPOSE
    if isinstance(e, ExprTensorProduct):
        return _EXPR_PREC_TENSOR
    if isinstance(e, _POSTFIX_KINDS):
        return _EXPR_PREC_POSTFIX
    return _EXPR_PREC_ATOM


def _emit_expr_child(e: Expr, min_prec: int) -> str:
    text = _emit_expr(e)
    if _expr_prec(e) < min_prec:
        return f"({text})"
    return text


def _emit_expr(e: Expr) -> str:
    if isinstance(e, ExprIdent):
        return e.name
    if isinstance(e, ExprIdentity):
        return f"identity({e.object_name})"
    if isinstance(e, ExprFromData):
        return f"from_data({_emit_string(e.key)})"
    if isinstance(e, ExprCup):
        return f"cup({e.object_name})"
    if isinstance(e, ExprCap):
        return f"cap({e.object_name})"
    if isinstance(e, ExprCompose):
        if e.op not in _COMPOSE_OPS:
            raise EmitError(f"emit: unknown compose operator {e.op!r}")
        left = _emit_expr_child(e.left, _EXPR_PREC_COMPOSE)
        right = _emit_expr_child(e.right, _EXPR_PREC_COMPOSE + 1)
        return f"{left} {e.op} {right}"
    if isinstance(e, ExprTransCompose):
        left = _emit_expr_child(e.left, _EXPR_PREC_COMPOSE)
        right = _emit_expr_child(e.right, _EXPR_PREC_COMPOSE + 1)
        return f"{left} >>> {right}"
    if isinstance(e, ExprTensorProduct):
        left = _emit_expr_child(e.left, _EXPR_PREC_TENSOR)
        right = _emit_expr_child(e.right, _EXPR_PREC_TENSOR + 1)
        return f"{left} @ {right}"
    if isinstance(e, ExprFreeze):
        return f"{_emit_expr_child(e.inner, _EXPR_PREC_POSTFIX)}.freeze"
    if isinstance(e, ExprDagger):
        return f"{_emit_expr_child(e.inner, _EXPR_PREC_POSTFIX)}.dagger"
    if isinstance(e, ExprTrace):
        inner = _emit_expr_child(e.inner, _EXPR_PREC_POSTFIX)
        return f"{inner}.trace({e.object_name})"
    if isinstance(e, ExprChangeBase):
        inner = _emit_expr_child(e.inner, _EXPR_PREC_POSTFIX)
        return f"{inner}.change_base({_emit_expr(e.phi)})"
    if isinstance(e, ExprMarginalize):
        if not e.names:
            raise EmitError("emit: .marginalize(...) requires at least one name")
        inner = _emit_expr_child(e.inner, _EXPR_PREC_POSTFIX)
        return f"{inner}.marginalize({', '.join(e.names)})"
    if isinstance(e, ExprCurry):
        inner = _emit_expr_child(e.inner, _EXPR_PREC_POSTFIX)
        return f"{inner}.curry_{e.direction}"
    if isinstance(e, ExprFan):
        if not e.exprs:
            raise EmitError("emit: fan(...) requires at least one expression")
        return "fan(" + ", ".join(_emit_expr(x) for x in e.exprs) + ")"
    if isinstance(e, ExprRepeat):
        if e.count is None:
            return f"repeat({_emit_expr(e.expr)})"
        return f"repeat({_emit_expr(e.expr)}, {e.count})"
    if isinstance(e, ExprStack):
        return f"stack({_emit_expr(e.expr)}, {e.count})"
    if isinstance(e, ExprScan):
        if e.init == "zeros":
            return f"scan({_emit_expr(e.expr)})"
        return f"scan({_emit_expr(e.expr)}, init={e.init})"
    if isinstance(e, ExprParser):
        return _emit_parser_expr(e)
    if isinstance(e, ExprChartFold):
        return _emit_chart_fold_expr(e)
    if isinstance(e, ExprMorphismCall):
        if not e.args:
            raise EmitError(
                f"emit: morphism call {e.callee!r} requires at least one argument"
            )
        return f"{e.callee}({', '.join(e.args)})"
    raise EmitError(f"emit: unknown Expr kind {type(e).__name__!r}")


def _emit_parser_expr(e: ExprParser) -> str:
    args: list[str] = []
    if e.rules:
        args.append("rules=[" + ", ".join(e.rules) + "]")
    if e.categories:
        args.append("categories=[" + ", ".join(e.categories) + "]")
    if e.terminal is not None:
        args.append(f"terminal={e.terminal}")
    if e.start != "S":
        args.append(f"start={e.start}")
    if e.depth != 1:
        args.append(f"depth={e.depth}")
    if e.constructors is not None:
        args.append("constructors=[" + ", ".join(e.constructors) + "]")
    return "parser(" + ", ".join(args) + ")"


def _emit_chart_fold_expr(e: ExprChartFold) -> str:
    args: list[str] = [f"lex={_emit_expr(e.lex)}"]
    if e.binary is not None:
        args.append(f"binary={_emit_expr(e.binary)}")
    if e.unary is not None:
        args.append(f"unary={_emit_expr(e.unary)}")
    if e.start != "S":
        args.append(f"start={e.start}")
    if e.depth != 1:
        args.append(f"depth={e.depth}")
    if e.effect_depth != 0:
        args.append(f"effect_depth={e.effect_depth}")
    return "chart_fold(" + ", ".join(args) + ")"


# ---------------------------------------------------------------------------
# let-arithmetic
# ---------------------------------------------------------------------------


def _emit_let_expr(e: LetExprNode) -> str:
    if isinstance(e, LetExprVar):
        return e.name
    if isinstance(e, LetExprLiteral):
        return _emit_number(e.value)
    if isinstance(e, LetExprString):
        return _emit_string(e.value)
    if isinstance(e, LetExprList):
        return "[" + ", ".join(_emit_let_expr(item) for item in e.items) + "]"
    if isinstance(e, LetExprBinOp):
        left = _emit_let_operand(e.left)
        right = _emit_let_operand(e.right)
        return f"({left} {e.op} {right})"
    if isinstance(e, LetExprUnaryOp):
        return f"-{_emit_let_atom(e.operand)}"
    if isinstance(e, LetExprCall):
        return f"{e.func}(" + ", ".join(_emit_let_expr(a) for a in e.args) + ")"
    if isinstance(e, LetExprIndex):
        base = _emit_let_receiver(e.array, context="indexed access")
        return f"{base}[" + ", ".join(_emit_let_expr(i) for i in e.indices) + "]"
    if isinstance(e, LetExprMethodCall):
        base = _emit_let_receiver(e.receiver, context="method call")
        args = ", ".join(_emit_let_expr(a) for a in e.args)
        return f"{base}.{e.method}({args})"
    if isinstance(e, LetExprLambda):
        return f"{e.param} -> {_emit_let_expr(e.body)}"
    if isinstance(e, LetExprFactor):
        return _emit_let_factor(e)
    raise EmitError(f"emit: unknown LetExprNode kind {type(e).__name__!r}")


def _emit_let_operand(e: LetExprNode) -> str:
    """Emit a binary-operator operand, parenthesizing the low-binding
    prefix forms (lambda, factor) whose bodies would otherwise absorb
    the operator."""
    text = _emit_let_expr(e)
    if isinstance(e, (LetExprLambda, LetExprFactor)):
        return f"({text})"
    return text


def _emit_let_atom(e: LetExprNode) -> str:
    """Emit in an atom-only position (the operand of unary minus)."""
    text = _emit_let_expr(e)
    if isinstance(e, LetExprUnaryOp):
        return f"({text})"
    if isinstance(e, LetExprLiteral) and text.startswith("-"):
        return f"({text})"
    return text


def _emit_let_receiver(e: LetExprNode, *, context: str) -> str:
    if isinstance(e, LetExprVar):
        return e.name
    raise EmitError(
        f"emit: {context} requires a variable receiver, got {type(e).__name__!r}"
    )


def _emit_let_factor(e: LetExprFactor) -> str:
    if not e.binders:
        raise EmitError("emit: factor expression with no binders")
    binders = ", ".join(
        f"{b.var} : {_emit_object_expr(b.index)}" for b in e.binders
    )
    if e.cases:
        cases = ", ".join(
            f"{c.label} -> {_emit_let_expr(c.value)}" for c in e.cases
        )
        return f"factor {binders} in {{{cases}}}"
    if e.body is None:
        raise EmitError("emit: factor expression with neither body nor cases")
    return f"factor {binders} in {_emit_let_expr(e.body)}"


# ---------------------------------------------------------------------------
# draw arguments
# ---------------------------------------------------------------------------


def _emit_draw_arg(arg: DrawArg) -> str:
    if isinstance(arg, DrawArgName):
        return arg.text
    if isinstance(arg, DrawArgIndex):
        if not arg.indices:
            raise EmitError(
                f"emit: bracket index {arg.name!r} has no indices; "
                f"a DrawArgIndex must carry at least one index name",
            )
        return f"{arg.name}[" + ", ".join(arg.indices) + "]"
    if isinstance(arg, DrawArgScalar):
        return _emit_number(arg.value)
    if isinstance(arg, DrawArgDist):
        return f"{arg.family}(" + ", ".join(_emit_draw_arg(a) for a in arg.args) + ")"
    if isinstance(arg, DrawArgList):
        return "[" + ", ".join(_emit_draw_arg(a) for a in arg.items) + "]"
    raise EmitError(f"emit: unknown DrawArg kind {type(arg).__name__!r}")


def _emit_draw_args(args: tuple[DrawArg, ...] | None) -> str:
    if not args:
        return ""
    return "(" + ", ".join(_emit_draw_arg(a) for a in args) + ")"


def _emit_init_family_arg(arg: str | float) -> str:
    if isinstance(arg, str):
        return arg
    return _emit_number(float(arg))


# ---------------------------------------------------------------------------
# program steps
# ---------------------------------------------------------------------------


def _emit_var_pattern(names: tuple[str, ...]) -> str:
    if not names:
        raise EmitError("emit: variable pattern with no names")
    if len(names) == 1:
        return names[0]
    return "(" + ", ".join(names) + ")"


def _emit_return_pattern(
    names: tuple[str, ...],
    labels: tuple[str, ...] | None,
) -> str:
    if labels is None:
        return _emit_var_pattern(names)
    if len(labels) != len(names):
        raise EmitError("emit: labelled return with mismatched label/var arity")
    pairs = ", ".join(f"{lab}: {var}" for lab, var in zip(labels, names))
    return f"({pairs})"


def _emit_draw_head(
    keyword: str,
    binder: str,
    index: ObjectExpr | None,
    morphism: str,
    args: tuple[DrawArg, ...] | None,
    options: tuple[OptionEntry, ...],
    indent: int,
) -> str:
    if index is not None:
        binder = f"{binder} : {_emit_object_expr(index)}"
    return (
        f"{_pad(indent)}{keyword} {binder} <- {morphism}"
        f"{_emit_draw_args(args)}{_emit_options(options)}"
    )


def _emit_program_step(step: ProgramStep, indent: int) -> list[str]:
    if isinstance(step, SampleStep):
        binder = _emit_var_pattern(step.vars)
        return [
            _emit_draw_head(
                "sample", binder, step.index, step.morphism,
                step.args, step.options, indent,
            )
        ]
    if isinstance(step, ObserveStep):
        binder = _emit_var_pattern(step.vars)
        return [
            _emit_draw_head(
                "observe", binder, step.index, step.morphism,
                step.args, step.options, indent,
            )
        ]
    if isinstance(step, MarginalizeStep):
        if not step.scope:
            raise EmitError("emit: marginalize step with an empty scope")
        lines = [
            _emit_draw_head(
                "marginalize", step.var, step.index, step.morphism,
                step.args, step.options, indent,
            )
        ]
        for inner in step.scope:
            lines.extend(_emit_program_step(inner, indent + 1))
        return lines
    if isinstance(step, LetStep):
        return [f"{_pad(indent)}let {step.name} = {_emit_let_expr(step.value)}"]
    if isinstance(step, ScoreStep):
        return [f"{_pad(indent)}score {step.name} = {_emit_let_expr(step.value)}"]
    if isinstance(step, ReturnStep):
        pattern = _emit_return_pattern(step.vars, step.labels)
        return [f"{_pad(indent)}return {pattern}"]
    if isinstance(step, BindStep):
        return _emit_bind_step(step, indent)
    if isinstance(step, DrawStep):
        keyword = "observe" if step.is_observed else "sample"
        binder = _emit_var_pattern(step.vars)
        return [
            _emit_draw_head(
                keyword, binder, None, step.morphism, step.args, (), indent,
            )
        ]
    if isinstance(step, PlateDrawStep):
        return [
            _emit_draw_head(
                "sample", step.name, step.index, step.morphism,
                step.args, (), indent,
            )
        ]
    if isinstance(step, VectorisedObserveStep):
        return [
            _emit_draw_head(
                "observe", step.response_var, step.index_set, step.morphism,
                step.args, (), indent,
            )
        ]
    if isinstance(step, GroupedBodyObserveStep):
        return [
            _emit_draw_head(
                "observe", step.response_var, step.index_set, step.morphism,
                step.args, (), indent,
            )
        ]
    if isinstance(step, (GroupedMarginalizeStep, GroupedLatentInitStep)):
        raise EmitError(
            f"emit: {type(step).__name__} is compiler-internal IR with no "
            "surface form; emit the surface MarginalizeStep it was lowered from"
        )
    raise EmitError(f"emit: unknown ProgramStep kind {type(step).__name__!r}")


def _emit_bind_step(step: BindStep, indent: int) -> list[str]:
    binder = _emit_var_pattern(step.vars)
    if step.mode == "sample":
        return [
            _emit_draw_head(
                "sample", binder, step.index, step.morphism, step.args, (), indent,
            )
        ]
    if step.mode == "score":
        return [
            _emit_draw_head(
                "observe", binder, step.index, step.morphism, step.args, (), indent,
            )
        ]
    if not step.scope:
        raise EmitError("emit: marginal bind step with an empty scope")
    lines = [
        _emit_draw_head(
            "marginalize", step.vars[0], step.index, step.morphism,
            step.args, (), indent,
        )
    ]
    for inner in step.scope:
        lines.extend(_emit_program_step(inner, indent + 1))
    return lines


# ---------------------------------------------------------------------------
# top-level statements
# ---------------------------------------------------------------------------


def _emit_statement(stmt: Statement, indent: int) -> str:
    if isinstance(stmt, CompositionDecl):
        return _emit_composition(stmt, indent)
    if isinstance(stmt, CategoryDecl):
        return _with_docs(stmt.docs, indent, f"category {_emit_names(stmt.names)}")
    if isinstance(stmt, RuleDecl):
        return _emit_rule(stmt, indent)
    if isinstance(stmt, SchemaDecl):
        return _emit_schema(stmt, indent)
    if isinstance(stmt, ObjectDecl):
        text = f"object {_emit_names(stmt.names)} : {_emit_type_init(stmt.init)}"
        return _with_docs(stmt.docs, indent, text)
    if isinstance(stmt, MorphismDecl):
        return _emit_morphism(stmt, indent)
    if isinstance(stmt, BundleDecl):
        text = f"bundle {stmt.name} : [" + ", ".join(stmt.rules) + "]"
        return _with_docs(stmt.docs, indent, text)
    if isinstance(stmt, ContractionDecl):
        return _emit_contraction(stmt, indent)
    if isinstance(stmt, DefineDecl):
        return _emit_define(stmt, indent)
    if isinstance(stmt, ExportDecl):
        return _with_docs(stmt.docs, indent, f"export {_emit_expr(stmt.expr)}")
    if isinstance(stmt, DeductionDecl):
        return _emit_deduction(stmt, indent)
    if isinstance(stmt, SignatureDecl):
        return _emit_signature(stmt, indent)
    if isinstance(stmt, EncoderDecl):
        return _emit_encoder(stmt, indent)
    if isinstance(stmt, DecoderDecl):
        return _emit_decoder(stmt, indent)
    if isinstance(stmt, LossDecl):
        return _emit_loss(stmt, indent)
    if isinstance(stmt, ProgramDecl):
        return _emit_program(stmt, indent)
    raise EmitError(f"emit: unknown Statement kind {type(stmt).__name__!r}")


def _with_docs(docs: tuple[str, ...], indent: int, text: str) -> str:
    lines = _doc_lines(docs, indent)
    lines.append(f"{_pad(indent)}{text}")
    return "\n".join(lines)


def _emit_composition(decl: CompositionDecl, indent: int) -> str:
    head = f"composition {decl.name}"
    if decl.level is not None:
        head = f"{head} [level={decl.level}]"
    lines = _doc_lines(decl.docs, indent)
    lines.append(f"{_pad(indent)}{head}")
    for entry in decl.body:
        lines.append(_emit_composition_entry(entry, indent + 1))
    return "\n".join(lines)


def _emit_composition_entry(entry: CompositionRuleEntry, indent: int) -> str:
    key = entry.key
    if entry.params:
        key = f"{key}(" + ", ".join(entry.params) + ")"
    return f"{_pad(indent)}{key} = {_emit_let_expr(entry.body)}"


def _emit_rule(decl: RuleDecl, indent: int) -> str:
    if not decl.variables:
        raise EmitError(f"emit: rule {decl.name!r} requires at least one variable")
    premises = ", ".join(_emit_object_expr(p) for p in decl.premises)
    text = (
        f"rule {decl.name}(" + ", ".join(decl.variables) + ") : "
        f"{premises} |- {_emit_object_expr(decl.conclusion)}"
    )
    return _with_docs(decl.docs, indent, text)


def _emit_schema(decl: SchemaDecl, indent: int) -> str:
    if not decl.parameters:
        raise EmitError(f"emit: schema {decl.name!r} requires at least one parameter")
    params = ", ".join(
        f"{_emit_names(p.names)} : {_emit_object_expr(p.type_expr)}"
        for p in decl.parameters
    )
    text = (
        f"schema {decl.name}({params}) : "
        f"{_emit_object_expr(decl.domain)} -> {_emit_object_expr(decl.codomain)}"
    )
    return _with_docs(decl.docs, indent, text)


def _emit_morphism(decl: MorphismDecl, indent: int) -> str:
    head = (
        f"morphism {_emit_names(decl.names)} : "
        f"{_emit_object_expr(decl.domain)} -> {_emit_object_expr(decl.codomain)}"
        f"{_emit_options(decl.options)}"
    )
    if decl.init_family is not None and decl.init_expr is not None:
        raise EmitError(
            f"emit: morphism {decl.names!r} carries both init_family and init_expr"
        )
    if decl.init_family is not None:
        args = ", ".join(_emit_init_family_arg(a) for a in decl.init_family.args)
        head = f"{head} ~ {decl.init_family.family}({args})"
    elif decl.init_expr is not None:
        head = f"{head} ~ {_emit_expr(decl.init_expr)}"
    return _with_docs(decl.docs, indent, head)


def _emit_contraction(decl: ContractionDecl, indent: int) -> str:
    if not decl.inputs:
        raise EmitError(
            f"emit: contraction {decl.name!r} requires at least one input"
        )
    inputs = ", ".join(
        f"{i.name} : {_emit_object_expr(i.input_domain)} -> "
        f"{_emit_object_expr(i.input_codomain)}"
        for i in decl.inputs
    )
    text = (
        f"contraction {decl.name}({inputs}) : "
        f"{_emit_object_expr(decl.domain)} -> {_emit_object_expr(decl.codomain)}"
        f"{_emit_options(decl.options)}"
    )
    return _with_docs(decl.docs, indent, text)


def _emit_define(decl: DefineDecl, indent: int) -> str:
    lines = _doc_lines(decl.docs, indent)
    head = f"{_pad(indent)}define {decl.name} = {_emit_expr(decl.expr)}"
    if decl.where:
        lines.append(f"{head} where")
        for nested in decl.where:
            if not isinstance(nested, DefineDecl):
                raise EmitError(
                    "emit: a define where-block admits only nested defines, "
                    f"got {type(nested).__name__!r}"
                )
            lines.append(_emit_define(nested, indent + 1))
    else:
        lines.append(head)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# deduction
# ---------------------------------------------------------------------------


def _emit_deduction(decl: DeductionDecl, indent: int) -> str:
    lines = _doc_lines(decl.docs, indent)
    lines.append(
        f"{_pad(indent)}deduction {decl.name} : "
        f"{_emit_object_expr(decl.domain)} -> {_emit_object_expr(decl.codomain)}"
        f"{_emit_options(decl.options)}"
    )
    body_start = len(lines)
    pad1 = _pad(indent + 1)
    if decl.atoms:
        lines.append(f"{pad1}atoms " + ", ".join(decl.atoms))
    if decl.binders:
        lines.append(f"{pad1}binders " + ", ".join(decl.binders))
    for rule in decl.rules:
        lines.append(_emit_sequent_rule(rule, indent + 1))
    if decl.lexicon:
        lines.append(f"{pad1}lexicon")
        for entry in decl.lexicon:
            lines.append(_emit_lexicon_entry(entry, indent + 2))
    if decl.lexicon_from_file is not None:
        lines.append(
            f"{pad1}lexicon from {_emit_string(decl.lexicon_from_file)}"
            f"{_emit_options(decl.lexicon_from_file_options)}"
        )
    if len(lines) == body_start:
        raise EmitError(f"emit: deduction {decl.name!r} has an empty body")
    return "\n".join(lines)


def _emit_sequent_rule(rule: SequentRule, indent: int) -> str:
    premises = ", ".join(_emit_object_expr(p) for p in rule.premises)
    return (
        f"{_pad(indent)}rule {rule.name} : {premises} |- "
        f"{_emit_object_expr(rule.conclusion)}{_emit_pragma(rule.options)}"
    )


def _emit_lexicon_entry(entry: LexiconEntry, indent: int) -> str:
    if not entry.words:
        raise EmitError("emit: lexicon entry with no words")
    words = ", ".join(_emit_string(w) for w in entry.words)
    category = _emit_lexicon_category(entry.category)
    return (
        f"{_pad(indent)}{words} : {category} = "
        f"{_emit_let_expr(entry.lf)}{_emit_pragma(entry.options)}"
    )


def _emit_lexicon_category(category: LexiconCategory) -> str:
    if isinstance(category, LexiconCategoryWildcard):
        return "*"
    if isinstance(category, LexiconCategoryRestricted):
        return "{" + ", ".join(category.atoms) + "}"
    if isinstance(category, LexiconCategoryFixed):
        return _emit_object_expr(category.category)
    raise EmitError(
        f"emit: unknown LexiconCategory kind {type(category).__name__!r}"
    )


# ---------------------------------------------------------------------------
# signature
# ---------------------------------------------------------------------------


def _emit_signature(decl: SignatureDecl, indent: int) -> str:
    lines = _doc_lines(decl.docs, indent)
    head = f"signature {decl.name}"
    if decl.params:
        head = f"{head}(" + ", ".join(decl.params) + ")"
    lines.append(f"{_pad(indent)}{head}")
    body_start = len(lines)
    pad1 = _pad(indent + 1)
    if decl.sorts:
        lines.append(f"{pad1}sorts")
        for sort in decl.sorts:
            lines.append(_emit_sort_decl(sort, indent + 2))
    if decl.constructors:
        lines.append(f"{pad1}constructors")
        for ctor in decl.constructors:
            lines.append(_emit_constructor_decl(ctor, indent + 2))
    if decl.binders:
        lines.append(f"{pad1}binders")
        for binder in decl.binders:
            lines.append(_emit_binder_decl(binder, indent + 2))
    if decl.vertex_kinds:
        lines.append(f"{pad1}vertex_kinds")
        for vk in decl.vertex_kinds:
            lines.append(_emit_vertex_kind_decl(vk, indent + 2))
    if decl.edge_kinds:
        lines.append(f"{pad1}edge_kinds")
        for ek in decl.edge_kinds:
            lines.append(_emit_edge_kind_decl(ek, indent + 2))
    if len(lines) == body_start:
        raise EmitError(f"emit: signature {decl.name!r} has an empty body")
    return "\n".join(lines)


def _emit_sort_decl(sort: SortDecl, indent: int) -> str:
    return f"{_pad(indent)}{sort.name} : {sort.kind}{_emit_options(sort.options)}"


def _emit_constructor_decl(ctor: ConstructorDecl, indent: int) -> str:
    if ctor.domain:
        domain = ", ".join(ctor.domain)
        return f"{_pad(indent)}{ctor.name} : {domain} -> {ctor.codomain}"
    return f"{_pad(indent)}{ctor.name} : -> {ctor.codomain}"


def _emit_binder_decl(binder: BinderDecl, indent: int) -> str:
    if not binder.binds or not binder.scoped:
        raise EmitError(
            f"emit: binder {binder.name!r} requires bound and scoped arguments"
        )
    binds = ", ".join(_emit_binder_var(v) for v in binder.binds)
    scoped = ", ".join(f"{a.arg} : {a.sort}" for a in binder.scoped)
    return (
        f"{_pad(indent)}{binder.name} : binds ({binds}) in ({scoped}) "
        f"-> {binder.codomain}"
    )


def _emit_binder_var(var: BinderVar) -> str:
    head = f"{var.var} : {var.sort}"
    if var.annot_sort is not None:
        if var.annot is None:
            raise EmitError(
                f"emit: binder variable {var.var!r} has an annotation sort "
                "but no annotation name"
            )
        head = f"{head} : {var.annot} : {var.annot_sort}"
    elif var.annot is not None:
        raise EmitError(
            f"emit: binder variable {var.var!r} has an annotation name "
            "but no annotation sort"
        )
    return head


def _emit_vertex_kind_decl(vk: VertexKindDecl, indent: int) -> str:
    return f"{_pad(indent)}{vk.name} : {vk.kind}{_emit_options(vk.options)}"


def _emit_edge_kind_decl(ek: EdgeKindDecl, indent: int) -> str:
    arrow = "->" if ek.directed else "--"
    return f"{_pad(indent)}{ek.name} : {ek.src} {arrow} {ek.tgt}"


# ---------------------------------------------------------------------------
# encoder / decoder / loss
# ---------------------------------------------------------------------------


def _emit_encoder(decl: EncoderDecl, indent: int) -> str:
    lines = _doc_lines(decl.docs, indent)
    head = f"encoder {decl.name} : {decl.signature}"
    if decl.sig_args:
        head = f"{head}(" + ", ".join(decl.sig_args) + ")"
    lines.append(f"{_pad(indent)}{head}{_emit_options(decl.options)}")
    pad1 = _pad(indent + 1)
    for dim in decl.dims:
        lines.append(f"{pad1}dim {dim.sort} = {dim.dim}")
    if decl.iterations is not None:
        lines.append(f"{pad1}iterations {decl.iterations}")
    for op_rule in decl.op_rules:
        lines.append(_emit_encoder_op_rule(op_rule, indent + 1))
    for init_rule in decl.init_rules:
        lines.append(_emit_encoder_init_rule(init_rule, indent + 1))
    for message_rule in decl.message_rules:
        lines.append(_emit_encoder_message_rule(message_rule, indent + 1))
    for update_rule in decl.update_rules:
        lines.append(_emit_encoder_update_rule(update_rule, indent + 1))
    for var_init in decl.var_inits:
        lines.append(_emit_encoder_var_init(var_init, indent + 1))
    if decl.readout is not None:
        lines.append(f"{pad1}readout |-> {_emit_let_expr(decl.readout)}")
    return "\n".join(lines)


def _emit_encoder_op_rule(rule: EncoderRule, indent: int) -> str:
    head = f"op {rule.op}"
    if rule.args:
        head = f"{head}(" + ", ".join(rule.args) + ")"
    if rule.mode == "recurrent":
        if rule.state_var is None:
            raise EmitError(f"emit: recurrent op {rule.op!r} missing its state var")
        head = f"{head} recurrent {rule.state_var}"
    elif rule.mode == "attention":
        if rule.prefix_var is None:
            raise EmitError(f"emit: attention op {rule.op!r} missing its prefix var")
        head = f"{head} attention {rule.prefix_var}"
    return f"{_pad(indent)}{head} |-> {_emit_let_expr(rule.body)}"


def _emit_encoder_init_rule(rule: EncoderInitRule, indent: int) -> str:
    return (
        f"{_pad(indent)}init {rule.kind}({rule.arg}) |-> "
        f"{_emit_let_expr(rule.body)}"
    )


def _emit_encoder_message_rule(rule: EncoderMessageRule, indent: int) -> str:
    return (
        f"{_pad(indent)}message [{rule.edge_kind}]({rule.src}, {rule.tgt}) |-> "
        f"{_emit_let_expr(rule.body)}"
    )


def _emit_encoder_update_rule(rule: EncoderUpdateRule, indent: int) -> str:
    return (
        f"{_pad(indent)}update [{rule.vertex_kind}]"
        f"({rule.self_var}, {rule.msgs_var}) |-> {_emit_let_expr(rule.body)}"
    )


def _emit_encoder_var_init(rule: EncoderVarInit, indent: int) -> str:
    head = f"var_init {rule.var_sort}"
    if rule.annot_sort is not None:
        head = f"{head} from {rule.annot_sort}"
        if rule.ty is not None:
            head = f"{head} as {rule.ty}"
    elif rule.ty is not None:
        raise EmitError(
            f"emit: var_init for {rule.var_sort!r} names a ty binding "
            "without an annotation sort"
        )
    return f"{_pad(indent)}{head} |-> {_emit_let_expr(rule.body)}"


def _emit_decoder(decl: DecoderDecl, indent: int) -> str:
    lines = _doc_lines(decl.docs, indent)
    head = f"decoder {decl.name} : {decl.signature}"
    if decl.sig_args:
        head = f"{head}(" + ", ".join(decl.sig_args) + ")"
    lines.append(f"{_pad(indent)}{head}{_emit_options(decl.options)}")
    body_start = len(lines)
    pad1 = _pad(indent + 1)
    for dim in decl.dims:
        lines.append(f"{pad1}dim {dim.sort} = {dim.dim}")
    for keyword, arg, body in (
        ("structure", decl.structure_arg, decl.structure),
        ("primitive", decl.primitive_arg, decl.primitive),
        ("factor", decl.factor_arg, decl.factor),
        ("binder_select", decl.binder_select_arg, decl.binder_select),
    ):
        if body is None:
            continue
        if arg is None:
            raise EmitError(
                f"emit: decoder {decl.name!r} {keyword} rule missing its argument"
            )
        lines.append(f"{pad1}{keyword} ({arg}) |-> {_emit_let_expr(body)}")
    if decl.recursive_default:
        lines.append(f"{pad1}body |-> recursive")
    if len(lines) == body_start:
        raise EmitError(f"emit: decoder {decl.name!r} has an empty body")
    return "\n".join(lines)


def _emit_loss(decl: LossDecl, indent: int) -> str:
    lines = _doc_lines(decl.docs, indent)
    lines.append(f"{_pad(indent)}loss {decl.name}{_emit_options(decl.options)}")
    lines.append(f"{_pad(indent + 1)}{_emit_let_expr(decl.body)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# program
# ---------------------------------------------------------------------------


def _emit_program(decl: ProgramDecl, indent: int) -> str:
    lines = _doc_lines(decl.docs, indent)
    if decl.params and decl.type_params:
        raise EmitError(
            f"emit: program {decl.name!r} mixes bare and typed parameters"
        )
    params = ""
    if decl.params:
        params = "(" + ", ".join(decl.params) + ")"
    elif decl.type_params:
        params = (
            "(" + ", ".join(_emit_program_param(p) for p in decl.type_params) + ")"
        )
    lines.append(
        f"{_pad(indent)}program {decl.name}{params} : "
        f"{_emit_object_expr(decl.domain)} -> {_emit_object_expr(decl.codomain)}"
        f"{_emit_options(decl.options)}"
    )
    for step in decl.draws:
        lines.extend(_emit_program_step(step, indent + 1))
    if not decl.return_vars:
        raise EmitError(f"emit: program {decl.name!r} has no return step")
    pattern = _emit_return_pattern(decl.return_vars, decl.return_labels)
    lines.append(f"{_pad(indent + 1)}return {pattern}")
    return "\n".join(lines)


def _emit_program_param(param: ProgramParam) -> str:
    if isinstance(param, ObjectParam):
        return f"{param.name} : {param.universe}"
    if isinstance(param, ScalarParam):
        return f"{param.name} : {param.scalar_kind}"
    if isinstance(param, MorphismParam):
        return (
            f"{param.name} : Mor[{_emit_object_expr(param.domain)}, "
            f"{_emit_object_expr(param.codomain)}]"
        )
    raise EmitError(f"emit: unknown ProgramParam kind {type(param).__name__!r}")


__all__ = ["EmitError", "module_to_source"]
