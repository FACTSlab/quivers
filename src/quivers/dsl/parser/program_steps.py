"""Walkers for program-block steps and program parameters."""

from __future__ import annotations

from typing import Literal

from quivers.dsl.ast_nodes import (
    BindStep,
    LetStep,
    MorphismParam,
    ObjectParam,
    ProgramParam,
    ProgramStep,
    ScalarParam,
    TypeExpr,
    TypeName,
    TypeProduct,
)
from quivers.dsl.parser._helpers import _required_text, _walk_draw_arg, _walk_options
from quivers.dsl.parser._registry import ParseError, _Tree
from quivers.dsl.parser.axes import _walk_axis_role_clause
from quivers.dsl.parser.expressions import _walk_let_arith, _walk_type


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
    # The `via <idx>` clause appears on both observe steps (inside
    # a grouped marginalize body, declaring the per-observe
    # fibration into the shared grouping plate) and on marginalize
    # steps themselves (the legacy header form).  For observe
    # steps the grammar restricts `via` to a bare identifier; for
    # marginalize steps it accepts either an identifier or a
    # `via_product(...)`.
    via_vid = t.field(vid, "via")
    if via_vid is not None:
        if t.kind(via_vid) == "via_product":
            axis_ids = t.fields(via_vid, "axis")
            via_axes_t = tuple(
                _required_text(t, av, via_vid, "axis") for av in axis_ids
            )
        else:
            via_t = _required_text(t, via_vid, vid, "via")
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
        red_vid = t.field(vid, "reduction")
        if red_vid is not None:
            reduction_t = _required_text(t, red_vid, vid, "reduction")

    axes_vid = t.field(vid, "axes")
    axes_t = _walk_axis_role_clause(t, axes_vid) if axes_vid else None
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
        axes=axes_t,
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



# -- top-level statements --
