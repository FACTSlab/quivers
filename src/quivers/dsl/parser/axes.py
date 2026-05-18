\
"""Walkers for axis-role clauses and morphism priors."""

from __future__ import annotations

from quivers.dsl.ast_nodes import AxisSpec, MorphismPrior
from quivers.dsl.parser._registry import _Tree
from quivers.dsl.parser._helpers import _walk_options


def _walk_axis_role_clause(t: _Tree, vid: str) -> AxisSpec:
    """Walk an ``axis_role_clause`` grammar node into an :class:`AxisSpec`.

    Surface form: ``over <axes> [iid over <axes>]`` where each
    ``<axes>`` is either a single identifier (single-axis event) or
    a parenthesised tuple ``(a, b, ...)`` (multi-axis event).
    """
    line, col = t.line_col(vid)
    over_vid = t.field(vid, "over")
    if over_vid is None:
        raise ParseError(f"axis_role_clause missing 'over' at {vid}")
    over_t = _walk_axis_list(t, over_vid)
    iid_vid = t.field(vid, "iid_over")
    iid_t: tuple[str, ...] = _walk_axis_list(t, iid_vid) if iid_vid else ()
    return AxisSpec(over=over_t, iid_over=iid_t, line=line, col=col)


def _walk_axis_list(t: _Tree, vid: str) -> tuple[str, ...]:
    """Walk an ``_axis_list`` node into a tuple of axis names.

    Accepts either a bare identifier (single-axis event) or an
    ``axis_tuple`` node listing multiple identifiers (multi-axis
    event, e.g. ``(dom, cod)`` for a MatrixNormal prior).
    """
    if t.kind(vid) == "axis_tuple":
        return tuple(t.text(av) for av in t.fields(vid, "axis"))
    return (t.text(vid),)


def _walk_morphism_prior(t: _Tree, vid: str) -> MorphismPrior:
    """Walk a ``morphism_prior`` node into a :class:`MorphismPrior`.

    Surface form: ``~ Family(args) [options] [axis_role_clause]``.
    Required on the latent-decl side; the literal ``(args)`` carry
    the prior's hyperparameters at declaration time.
    """
    line, col = t.line_col(vid)
    fv = t.field(vid, "family")
    family = _required_text(t, fv, vid, "family")
    args_list: list[str | float] = []
    for av in t.fields(vid, "args"):
        args_list.append(_walk_draw_arg(t, av))
    opt_vid = t.field(vid, "options")
    options = _walk_options(t, opt_vid) if opt_vid else {}
    axes_vid = t.field(vid, "axes")
    axes = _walk_axis_role_clause(t, axes_vid) if axes_vid else None
    return MorphismPrior(
        family=family,
        args=tuple(args_list),
        options=options,
        axes=axes,
        line=line,
        col=col,
    )
