"""Exhaustive, binder-aware operations over the shared let-expression AST.

The native compiler, QIEC elaborator, and transpilation preflight all consume
the same expression nodes.  Keeping their recursive walks here makes adding a
new node an explicit cross-surface change: an unrecognised variant raises
instead of being silently treated as a leaf.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from quivers.dsl import ast_nodes as surface


def let_expr_children(expr: surface.LetExprNode) -> tuple[surface.LetExprNode, ...]:
    """Return the immediate expression children of ``expr`` in source order."""
    if isinstance(
        expr,
        surface.LetExprLiteral
        | surface.LetExprBool
        | surface.LetExprUnit
        | surface.LetExprVar
        | surface.LetExprString,
    ):
        return ()
    if isinstance(expr, surface.LetExprBinOp):
        return (expr.left, expr.right)
    if isinstance(expr, surface.LetExprUnaryOp):
        return (expr.operand,)
    if isinstance(expr, surface.LetExprCall):
        return expr.args
    if isinstance(expr, surface.LetExprIndex):
        return (expr.array, *expr.indices)
    if isinstance(expr, surface.LetExprList | surface.LetExprTuple):
        return expr.items
    if isinstance(expr, surface.LetExprLambda):
        return (expr.body,)
    if isinstance(expr, surface.LetExprFactor):
        body = () if expr.body is None else (expr.body,)
        return (*body, *(case.value for case in expr.cases))
    if isinstance(expr, surface.LetExprMethodCall):
        return (expr.receiver, *expr.args)
    if isinstance(expr, surface.QiecConstructorValue):
        return expr.fields
    raise TypeError(
        f"let_expr_children: unsupported let-expression variant {type(expr).__name__}"
    )


def walk_let_expr(expr: surface.LetExprNode) -> Iterator[surface.LetExprNode]:
    """Yield ``expr`` and all descendants in preorder."""
    yield expr
    for child in let_expr_children(expr):
        yield from walk_let_expr(child)


def free_let_names(
    expr: surface.LetExprNode,
    *,
    bound: frozenset[str] = frozenset(),
    include_callees: bool = False,
) -> tuple[str, ...]:
    """Return free value names in first-use order.

    Lambda parameters and factor indices bind in their bodies.  Call names are
    declarations by default; callers that model a callee as a host value can
    include them explicitly.
    """
    found: list[str] = []

    def visit(node: surface.LetExprNode, local_bound: frozenset[str]) -> None:
        if isinstance(node, surface.LetExprVar):
            if node.name not in local_bound and node.name not in found:
                found.append(node.name)
            return
        if isinstance(node, surface.LetExprCall):
            if (
                include_callees
                and node.func not in local_bound
                and node.func not in found
            ):
                found.append(node.func)
            for argument in node.args:
                visit(argument, local_bound)
            return
        if isinstance(node, surface.LetExprLambda):
            visit(node.body, local_bound | {node.param})
            return
        if isinstance(node, surface.LetExprFactor):
            inner = local_bound | {binder.var for binder in node.binders}
            if node.body is not None:
                visit(node.body, inner)
            for case in node.cases:
                visit(case.value, inner)
            return
        for child in let_expr_children(node):
            visit(child, local_bound)

    visit(expr, bound)
    return tuple(found)


def substitute_let_expr(
    expr: surface.LetExprNode,
    substitution: Mapping[str, surface.LetExprNode],
) -> surface.LetExprNode:
    """Capture-avoiding substitution of free variables in ``expr``."""
    occupied = set(free_let_names(expr, include_callees=True))
    for replacement in substitution.values():
        occupied.update(free_let_names(replacement, include_callees=True))
    for node in walk_let_expr(expr):
        if isinstance(node, surface.LetExprLambda):
            occupied.add(node.param)
        elif isinstance(node, surface.LetExprFactor):
            occupied.update(binder.var for binder in node.binders)

    def fresh(base: str) -> str:
        suffix = 1
        candidate = f"{base}_{suffix}"
        while candidate in occupied:
            suffix += 1
            candidate = f"{base}_{suffix}"
        occupied.add(candidate)
        return candidate

    def replacement_names(
        active: Mapping[str, surface.LetExprNode],
        body: surface.LetExprNode,
    ) -> set[str]:
        used_keys = set(free_let_names(body)) & active.keys()
        return {name for key in used_keys for name in free_let_names(active[key])}

    def replace(
        node: surface.LetExprNode,
        active: Mapping[str, surface.LetExprNode],
    ) -> surface.LetExprNode:
        if isinstance(node, surface.LetExprVar):
            return active.get(node.name, node)
        if isinstance(
            node,
            surface.LetExprLiteral
            | surface.LetExprBool
            | surface.LetExprUnit
            | surface.LetExprString,
        ):
            return node
        if isinstance(node, surface.LetExprBinOp):
            return node.with_(
                left=replace(node.left, active),
                right=replace(node.right, active),
            )
        if isinstance(node, surface.LetExprUnaryOp):
            return node.with_(operand=replace(node.operand, active))
        if isinstance(node, surface.LetExprCall):
            return node.with_(args=tuple(replace(item, active) for item in node.args))
        if isinstance(node, surface.LetExprIndex):
            return node.with_(
                array=replace(node.array, active),
                indices=tuple(replace(item, active) for item in node.indices),
            )
        if isinstance(node, surface.LetExprList | surface.LetExprTuple):
            return node.with_(items=tuple(replace(item, active) for item in node.items))
        if isinstance(node, surface.LetExprLambda):
            inner = {key: value for key, value in active.items() if key != node.param}
            param = node.param
            body = node.body
            if param in replacement_names(inner, body):
                renamed = fresh(param)
                body = replace(
                    body,
                    {
                        param: surface.LetExprVar(
                            name=renamed,
                            line=node.line,
                            col=node.col,
                        )
                    },
                )
                param = renamed
            return node.with_(param=param, body=replace(body, inner))
        if isinstance(node, surface.LetExprFactor):
            bound = {binder.var for binder in node.binders}
            inner = {key: value for key, value in active.items() if key not in bound}
            bodies = (() if node.body is None else (node.body,)) + tuple(
                case.value for case in node.cases
            )
            captures = {
                name for body in bodies for name in replacement_names(inner, body)
            }
            binders = list(node.binders)
            body = node.body
            cases = node.cases
            for position, binder in enumerate(binders):
                if binder.var not in captures:
                    continue
                renamed = fresh(binder.var)
                alpha = {
                    binder.var: surface.LetExprVar(
                        name=renamed,
                        line=binder.line,
                        col=binder.col,
                    )
                }
                body = None if body is None else replace(body, alpha)
                cases = tuple(
                    case.with_(value=replace(case.value, alpha)) for case in cases
                )
                binders[position] = binder.with_(var=renamed)
            return node.with_(
                binders=tuple(binders),
                body=None if body is None else replace(body, inner),
                cases=tuple(
                    case.with_(value=replace(case.value, inner)) for case in cases
                ),
            )
        if isinstance(node, surface.LetExprMethodCall):
            return node.with_(
                receiver=replace(node.receiver, active),
                args=tuple(replace(item, active) for item in node.args),
            )
        if isinstance(node, surface.QiecConstructorValue):
            return node.with_(
                fields=tuple(replace(field, active) for field in node.fields)
            )
        raise TypeError(
            f"substitute_let_expr: unsupported let-expression variant "
            f"{type(node).__name__}"
        )

    return replace(expr, substitution)


__all__ = [
    "free_let_names",
    "let_expr_children",
    "substitute_let_expr",
    "walk_let_expr",
]
