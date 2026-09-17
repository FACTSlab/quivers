"""Instantiating a program at a draw from it.

A draw from a program, ``sample theta <- sub(args)`` or
``sample (u, v) <- sub``, denotes the program's body run in place: the
callee's parameters take the draw's arguments, and its steps join the
caller's under names of the caller's own. The instantiation is a
substitution on the callee's source steps followed by an alpha-renaming
of every name the callee binds, which realises the categorical
substitution lemma (substituting actuals for formals commutes with
denotation up to renaming) and gives every draw of a program its own
latents.

The renaming follows one convention on every runtime: a local ``z`` of a
program drawn under the binder ``theta`` becomes ``theta$z``, and the
name the program returns becomes the binder itself, so ``sample theta <-
school_effects(0.6, School)`` binds ``theta`` to the returned value and
contributes the site ``theta$z``. A parenthesized pattern maps the
program's returned names onto its own positionally, and a single binder
taking a pair binds the pair of the renamed components. ``$`` is no
character of a source identifier, so a renamed name never collides with
one the source declares.
"""

from __future__ import annotations

from collections.abc import Mapping

from quivers.dsl.ast_nodes import (
    AxisSpec,
    CallStep,
    DrawArg,
    DrawArgIndex,
    DrawArgName,
    DrawArgScalar,
    LetExprFactor,
    LetExprLambda,
    LetExprLiteral,
    LetExprNode,
    LetExprTuple,
    LetExprVar,
    LetFactorBinder,
    LetStep,
    MarginalizeStep,
    ObjectExpr,
    ObjectParam,
    ObjectProduct,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    ReturnStep,
    SampleStep,
    ScalarParam,
    ScoreStep,
    TypeName,
)


class TemplateError(Exception):
    """A draw's arguments or pattern do not fit the program drawn from.

    Parameters
    ----------
    message : str
        What does not fit.
    gap : bool
        Whether the draw is well formed but names a construct the
        elaboration has no form for, a morphism parameter, rather
        than an error of the source.
    """

    def __init__(self, message: str, gap: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.gap = gap


def template_bindings(
    template: ProgramDecl,
    arguments: tuple[DrawArg, ...],
    bound: Mapping[str, LetExprNode],
) -> tuple[dict[str, str], dict[str, LetExprNode]]:
    """The substitutions a draw's arguments fix for a program's parameters.

    Parameters
    ----------
    template : ProgramDecl
        The program drawn from.
    arguments : tuple[DrawArg, ...]
        The draw's arguments, in the program's parameter order.
    bound : Mapping[str, LetExprNode]
        Every name bound in the caller, each as the expression a
        reference to it denotes.

    Returns
    -------
    tuple[dict[str, str], dict[str, LetExprNode]]
        The object each object parameter takes, and the expression each
        scalar parameter takes.

    Raises
    ------
    TemplateError
        If the argument count differs from the parameter count, an
        argument does not fit its parameter, or a parameter is a
        morphism, which no draw supplies.
    """
    parameters = template.type_params or ()
    if len(arguments) != len(parameters):
        raise TemplateError(
            f"program {template.name!r} takes {len(parameters)} template "
            f"parameter(s); the draw supplies {len(arguments)}"
        )
    objects: dict[str, str] = {}
    values: dict[str, LetExprNode] = {}
    for argument, parameter in zip(arguments, parameters, strict=True):
        if isinstance(parameter, ObjectParam):
            if not isinstance(argument, DrawArgName):
                raise TemplateError(
                    f"parameter {parameter.name!r} of program {template.name!r} "
                    f"takes an object"
                )
            objects[parameter.name] = argument.text
        elif isinstance(parameter, ScalarParam):
            if isinstance(argument, DrawArgScalar):
                values[parameter.name] = LetExprLiteral(value=argument.value)
            elif isinstance(argument, DrawArgName) and argument.text in bound:
                values[parameter.name] = bound[argument.text]
            else:
                raise TemplateError(
                    f"parameter {parameter.name!r} of program {template.name!r} "
                    f"takes a number or a bound name"
                )
        else:
            raise TemplateError(
                f"parameter {parameter.name!r} of program {template.name!r} is a "
                f"morphism; a template over morphisms has no elaboration",
                gap=True,
            )
    return objects, values


def instantiate_program(
    template: ProgramDecl,
    binders: tuple[str, ...],
    objects: Mapping[str, str],
    values: Mapping[str, LetExprNode],
) -> tuple[ProgramStep, ...]:
    """The steps a draw from a program runs in the caller.

    Parameters
    ----------
    template : ProgramDecl
        The program drawn from.
    binders : tuple[str, ...]
        The draw's pattern: one name, or one per returned name.
    objects : Mapping[str, str]
        The object each object parameter takes.
    values : Mapping[str, LetExprNode]
        The expression each scalar parameter takes.

    Returns
    -------
    tuple[ProgramStep, ...]
        The program's steps under the caller's names, without its
        return; a single binder taking a pair is bound to the pair of
        the renamed components as a final let.

    Raises
    ------
    TemplateError
        If the pattern's arity is neither one nor the number of names
        the program returns.
    """
    returned = template.return_vars
    prefix = "_".join(binders)
    rename: dict[str, str] = {}
    if len(binders) == len(returned):
        for name, binder in zip(returned, binders, strict=True):
            rename[name] = binder
    elif len(binders) != 1:
        raise TemplateError(
            f"pattern ({', '.join(binders)}) destructures {len(binders)} names, "
            f"but program {template.name!r} returns {len(returned)}"
        )
    for name in _bound_names(template.draws):
        rename.setdefault(name, f"{prefix}${name}")
    steps = tuple(
        _rename_step(step, objects, values, rename)
        for step in template.draws
        if not isinstance(step, ReturnStep)
    )
    if len(binders) == 1 and len(returned) != 1:
        steps = (
            *steps,
            LetStep(
                name=binders[0],
                value=LetExprTuple(
                    items=tuple(LetExprVar(name=rename[name]) for name in returned)
                ),
            ),
        )
    return steps


def _bound_names(steps: tuple[ProgramStep, ...]) -> list[str]:
    """Every name the steps bind, marginalization scopes included.

    Parameters
    ----------
    steps : tuple[ProgramStep, ...]
        The steps.

    Returns
    -------
    list[str]
        The names in binding order, each once.
    """
    out: list[str] = []
    for step in steps:
        if isinstance(step, (SampleStep, ObserveStep)):
            out.extend(step.vars)
        elif isinstance(step, MarginalizeStep):
            out.append(step.var)
            out.extend(_bound_names(step.scope))
        elif isinstance(step, (LetStep, ScoreStep, CallStep)):
            out.append(step.name)
    seen: set[str] = set()
    unique: list[str] = []
    for name in out:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _rename_step(
    step: ProgramStep,
    objects: Mapping[str, str],
    values: Mapping[str, LetExprNode],
    rename: Mapping[str, str],
) -> ProgramStep:
    """One step under the substitution and the renaming.

    Parameters
    ----------
    step : ProgramStep
        The step.
    objects : Mapping[str, str]
        The object each object parameter takes.
    values : Mapping[str, LetExprNode]
        The expression each scalar parameter takes.
    rename : Mapping[str, str]
        The caller's name of each name the program binds.

    Returns
    -------
    ProgramStep
        The rewritten step.

    Raises
    ------
    TemplateError
        If a scalar parameter is drawn through where only a number or
        a name fits.
    """
    if isinstance(step, (SampleStep, ObserveStep)):
        return step.with_(
            vars=tuple(rename.get(name, name) for name in step.vars),
            morphism=rename.get(step.morphism, step.morphism),
            args=_rename_args(step.args, values, rename),
            index=None if step.index is None else _rename_object(step.index, objects),
            axes=_rename_axes(step.axes, objects),
        )
    if isinstance(step, MarginalizeStep):
        return step.with_(
            var=rename.get(step.var, step.var),
            morphism=rename.get(step.morphism, step.morphism),
            args=_rename_args(step.args, values, rename),
            index=None if step.index is None else _rename_object(step.index, objects),
            over=None if step.over is None else objects.get(step.over, step.over),
            over_objs=(
                None
                if step.over_objs is None
                else tuple(objects.get(name, name) for name in step.over_objs)
            ),
            scope=tuple(
                _rename_step(inner, objects, values, rename) for inner in step.scope
            ),
        )
    if isinstance(step, (LetStep, ScoreStep)):
        return step.with_(
            name=rename.get(step.name, step.name),
            value=_rename_expr(step.value, objects, values, rename, frozenset()),
        )
    if isinstance(step, CallStep):
        return step.with_(
            name=rename.get(step.name, step.name),
            call=step.call.with_(
                arguments=tuple(
                    _rename_expr(argument, objects, values, rename, frozenset())
                    for argument in step.call.arguments
                )
            ),
        )
    return step


def _rename_args(
    args: tuple[DrawArg, ...] | None,
    values: Mapping[str, LetExprNode],
    rename: Mapping[str, str],
) -> tuple[DrawArg, ...] | None:
    """Draw arguments under the substitution and the renaming.

    Parameters
    ----------
    args : tuple[DrawArg, ...] | None
        The arguments.
    values : Mapping[str, LetExprNode]
        The expression each scalar parameter takes.
    rename : Mapping[str, str]
        The caller's name of each name the program binds.

    Returns
    -------
    tuple[DrawArg, ...] | None
        The rewritten arguments.

    Raises
    ------
    TemplateError
        If a scalar parameter used as a draw argument takes an
        expression that is neither a number nor a name.
    """
    if args is None:
        return None
    out: list[DrawArg] = []
    for argument in args:
        if isinstance(argument, DrawArgName):
            replacement = values.get(argument.text)
            if replacement is not None:
                out.append(_draw_argument(replacement, argument))
            else:
                out.append(
                    argument.with_(text=rename.get(argument.text, argument.text))
                )
        elif isinstance(argument, DrawArgIndex):
            out.append(
                argument.with_(
                    name=rename.get(argument.name, argument.name),
                    indices=tuple(
                        rename.get(index, index) for index in argument.indices
                    ),
                )
            )
        else:
            out.append(argument)
    return tuple(out)


def _draw_argument(value: LetExprNode, argument: DrawArgName) -> DrawArg:
    """A scalar parameter's expression as a draw argument.

    Parameters
    ----------
    value : LetExprNode
        The expression the parameter takes.
    argument : DrawArgName
        The argument naming the parameter, for its position.

    Returns
    -------
    DrawArg
        A number or a name.

    Raises
    ------
    TemplateError
        If the expression is neither.
    """
    if isinstance(value, LetExprLiteral):
        return DrawArgScalar(
            value=float(value.value), line=argument.line, col=argument.col
        )
    if isinstance(value, LetExprVar):
        return DrawArgName(text=value.name, line=argument.line, col=argument.col)
    raise TemplateError(
        f"parameter {argument.text!r} is drawn through, so it takes a number or a name"
    )


def _rename_object(expr: ObjectExpr, objects: Mapping[str, str]) -> ObjectExpr:
    """An object expression under the object substitution.

    Parameters
    ----------
    expr : ObjectExpr
        The expression.
    objects : Mapping[str, str]
        The object each object parameter takes.

    Returns
    -------
    ObjectExpr
        The rewritten expression.
    """
    if isinstance(expr, TypeName):
        replacement = objects.get(expr.name)
        return expr if replacement is None else expr.with_(name=replacement)
    if isinstance(expr, ObjectProduct):
        return expr.with_(
            components=tuple(_rename_object(item, objects) for item in expr.components)
        )
    return expr


def _rename_axes(axes: AxisSpec | None, objects: Mapping[str, str]) -> AxisSpec | None:
    """An axis annotation under the object substitution.

    Parameters
    ----------
    axes : AxisSpec | None
        The annotation.
    objects : Mapping[str, str]
        The object each object parameter takes.

    Returns
    -------
    AxisSpec | None
        The rewritten annotation.
    """
    if axes is None:
        return None
    return axes.with_(
        over=tuple(objects.get(name, name) for name in axes.over),
        iid_over=tuple(objects.get(name, name) for name in axes.iid_over),
    )


def _rename_expr(
    expr: LetExprNode,
    objects: Mapping[str, str],
    values: Mapping[str, LetExprNode],
    rename: Mapping[str, str],
    shadowed: frozenset[str],
) -> LetExprNode:
    """A let expression under the substitution and the renaming.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    objects : Mapping[str, str]
        The object each object parameter takes.
    values : Mapping[str, LetExprNode]
        The expression each scalar parameter takes.
    rename : Mapping[str, str]
        The caller's name of each name the program binds.
    shadowed : frozenset[str]
        The names a lambda or a factor binds inside the expression,
        which refer to those binders rather than to the program's.

    Returns
    -------
    LetExprNode
        The rewritten expression.
    """
    if isinstance(expr, LetExprVar):
        if expr.name in shadowed:
            return expr
        replacement = values.get(expr.name)
        if replacement is not None:
            return replacement
        return expr.with_(name=rename.get(expr.name, expr.name))
    if isinstance(expr, LetExprLambda):
        return expr.with_(
            body=_rename_expr(
                expr.body, objects, values, rename, shadowed | {expr.param}
            )
        )
    if isinstance(expr, LetExprFactor):
        inner = shadowed | {binder.var for binder in expr.binders}
        return expr.with_(
            binders=tuple(
                LetFactorBinder(
                    var=binder.var,
                    index=_rename_object(binder.index, objects),
                    line=binder.line,
                    col=binder.col,
                )
                for binder in expr.binders
            ),
            body=(
                None
                if expr.body is None
                else _rename_expr(expr.body, objects, values, rename, inner)
            ),
            cases=tuple(
                case.with_(
                    value=_rename_expr(case.value, objects, values, rename, inner)
                )
                for case in expr.cases
            ),
        )
    changes: dict[str, LetExprNode | tuple[LetExprNode, ...]] = {}
    for field in type(expr).__field_specs__:
        current = getattr(expr, field)
        if isinstance(current, LetExprNode):
            changes[field] = _rename_expr(current, objects, values, rename, shadowed)
        elif (
            isinstance(current, tuple)
            and current
            and all(isinstance(item, LetExprNode) for item in current)
        ):
            changes[field] = tuple(
                _rename_expr(item, objects, values, rename, shadowed)
                for item in current
            )
    if not changes:
        return expr
    return expr.with_(**changes)


__all__ = [
    "TemplateError",
    "instantiate_program",
    "template_bindings",
]
