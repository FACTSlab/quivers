"""Morphism / let-binding resolution shared by every transpile
backend.

A ``sample x <- morphism_or_let_name`` step's ``morphism`` slot may
refer to one of three things:

1. A distribution family name (e.g. ``Beta``) — the existing
   ``_FAMILIES`` map carries the target name.
2. A declared ``morphism`` whose ``~ Family(args)`` init clause
   names the underlying distribution.
3. A ``let`` binding whose RHS is itself a morphism reference (the
   common shape is a pure alias or a Kleisli composition).

The resolver here turns case 2 into the equivalent of case 1 by
unfolding the declared morphism's `init_family`. Case 3 is unfolded
recursively: a let binding to a bare identifier resolves to whatever
that identifier resolves to; composite expressions raise
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
with a clear message naming the composition operator.

The output is a [`ResolvedDist`][quivers.transpile._resolve.ResolvedDist]
record carrying ``family`` (the canonical QVR family name) and
``args`` (the tuple of literal-or-variable arguments). Callers feed
``family`` into their backend-specific ``_FAMILIES`` table and emit
the call with ``args``.
"""

from __future__ import annotations

import dataclasses

import torch.distributions as _td

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgList,
    DrawArgMatrix,
    DrawArgName,
    DrawArgScalar,
    Expr,
    ExprIdent,
    LetDecl,
    Module,
    MorphismDecl,
    MorphismInitFamily,
)
from quivers.dsl.ast_nodes._shared import (
    OptionEntry,
    OptionNumber,
    OptionString,
)
from quivers.transpile._api import UnsupportedConstruct


@dataclasses.dataclass(frozen=True)
class ResolvedDist:
    """The (family, args) pair a sample / observe step resolves to.

    ``family`` is the canonical QVR family name, exactly as it would
    appear if the user had written ``sample x <- family(args)``.
    ``args`` is the tuple of positional arguments after resolution,
    held in the legacy wire form ``str | float`` for backends that
    have not migrated to the structural `DrawArg` representation.

    The ``original_morphism_name`` and ``via`` fields are diagnostic:
    when a sample resolves through a morphism declaration, the
    transpiled output can reference both the underlying family and the
    original morphism name (some backends prefer one or the other).
    """

    family: str
    args: tuple[str | float, ...]
    original_morphism_name: str
    via: tuple[str, ...] = ()


def _draw_arg_to_wire(arg: DrawArg) -> str | float:
    """Lower a `DrawArg` atomic variant to its wire form. Compound
    variants encode positionally so the legacy backend pipeline
    receives the structural literal as a parseable string surrogate."""
    if isinstance(arg, DrawArgScalar):
        return arg.value
    if isinstance(arg, DrawArgName):
        return arg.text
    if isinstance(arg, DrawArgList):
        return (
            "[" + ", ".join(_atom_to_text(e) for e in arg.elements) + "]"
        )
    if isinstance(arg, DrawArgMatrix):
        rows = ", ".join(
            "[" + ", ".join(_atom_to_text(e) for e in row.elements) + "]"
            for row in arg.rows
        )
        return f"[{rows}]"
    raise TypeError(
        f"_draw_arg_to_wire: unsupported arg variant {type(arg).__name__}"
    )


def _atom_to_text(value: str | float) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _format_number(float(value))
    return str(value)


def _format_number(value: float) -> str:
    if value == int(value):
        return f"{int(value)}"
    return repr(value)


def build_morphism_table(module: Module) -> dict[str, MorphismDecl]:
    """Return name → MorphismDecl for every morphism declaration in
    ``module``. Duplicate names are an error (the QVR compiler also
    rejects them, but the resolver catches it locally with a clearer
    transpile-time message)."""
    out: dict[str, MorphismDecl] = {}
    for stmt in module.statements:
        if isinstance(stmt, MorphismDecl):
            if stmt.name in out:
                msg = (
                    f"duplicate morphism declaration {stmt.name!r}: "
                    f"first at line {out[stmt.name].line}, again at "
                    f"line {stmt.line}"
                )
                raise UnsupportedConstruct("qvr-transpile", [msg])
            out[stmt.name] = stmt
    return out


def build_let_table(module: Module) -> dict[str, Expr]:
    """Return name → expr for every top-level ``let_decl``."""
    out: dict[str, Expr] = {}
    for stmt in module.statements:
        if isinstance(stmt, LetDecl):
            out[stmt.name] = stmt.expr
    return out


def resolve_step_dist(
    morphism_name: str,
    raw_args: tuple[DrawArg, ...] | None,
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    family_registry: frozenset[str],
    target: str,
    _seen: tuple[str, ...] = (),
) -> ResolvedDist:
    """Resolve a sample / observe step's morphism slot to a
    ``(family, args)`` pair.

    Parameters
    ----------
    morphism_name
        The string in ``SampleStep.morphism`` /
        ``ObserveStep.morphism``. Either a family name, a declared
        morphism's name, or a let-binding name.
    raw_args
        Positional arguments on the step (``None`` is treated as
        empty). When the resolver unfolds a morphism with its own
        ``~ Family(args)`` init clause, the step-supplied ``raw_args``
        take precedence over the declaration's defaults; in current
        practice the step does not supply args when referring to a
        morphism (the morphism's args carry the parameter set).
    morphisms
        Name → MorphismDecl table from
        [`build_morphism_table`][quivers.transpile._resolve.build_morphism_table].
    lets
        Name → Expr table from
        [`build_let_table`][quivers.transpile._resolve.build_let_table].
    family_registry
        Frozen set of canonical QVR family names. When
        ``morphism_name`` is in this set, the resolver returns
        immediately with the step's own args.
    target
        Backend name (for error messages).
    _seen
        Internal: the chain of names visited during resolution. The
        resolver detects cycles (``let a = b; let b = a``) by
        membership in this tuple.
    """
    if morphism_name in family_registry:
        wire: tuple[str | float, ...]
        if raw_args:
            wire = tuple(_draw_arg_to_wire(a) for a in raw_args)
        else:
            wire = _FAMILY_DEFAULT_ARGS.get(morphism_name, ())
        return ResolvedDist(
            family=morphism_name,
            args=wire,
            original_morphism_name=morphism_name,
        )

    if morphism_name in _seen:
        msg = (
            f"morphism / let cycle while resolving {morphism_name!r}; "
            f"chain: {' -> '.join((*_seen, morphism_name))}"
        )
        raise UnsupportedConstruct(target, [msg])

    chain = (*_seen, morphism_name)

    if morphism_name in morphisms:
        decl = morphisms[morphism_name]
        if decl.init_family is not None:
            return _from_init_family(
                morphism_name=morphism_name,
                init=decl.init_family,
                step_args=raw_args,
                chain=chain,
                morphism_options=decl.options,
            )
        if decl.init_expr is not None:
            # When the init expression is a bare family identifier
            # (`morphism foo : T -> T [scale=0.1] ~ Normal`) the
            # parser models it as an `init_expr=ExprIdent("Normal")`
            # rather than `init_family=Normal`. Route this case
            # through the same option-aware family-merge that
            # `_from_init_family` uses so the morphism's option
            # block (`[scale=0.1]`) populates the matching family
            # parameter slot.
            if (
                isinstance(decl.init_expr, ExprIdent)
                and decl.init_expr.name in family_registry
            ):
                return _from_init_family(
                    morphism_name=morphism_name,
                    init=MorphismInitFamily(
                        family=decl.init_expr.name,
                        args=(),
                    ),
                    step_args=raw_args,
                    chain=chain,
                    morphism_options=decl.options,
                )
            return _resolve_expr(
                morphism_name=morphism_name,
                expr=decl.init_expr,
                raw_args=raw_args,
                morphisms=morphisms,
                lets=lets,
                family_registry=family_registry,
                target=target,
                chain=chain,
            )
        msg = (
            f"morphism {morphism_name!r} has neither `~ Family(...)` "
            f"nor `~ <expr>` init clause; transpile cannot derive a "
            f"distribution"
        )
        raise UnsupportedConstruct(target, [msg])

    if morphism_name in lets:
        return _resolve_expr(
            morphism_name=morphism_name,
            expr=lets[morphism_name],
            raw_args=raw_args,
            morphisms=morphisms,
            lets=lets,
            family_registry=family_registry,
            target=target,
            chain=chain,
        )

    msg = (
        f"sample / observe step references {morphism_name!r} which "
        f"is neither a family in the registry, a declared morphism, "
        f"nor a let-bound name"
    )
    raise UnsupportedConstruct(target, [f"family:{morphism_name}", msg])


def _from_init_family(
    *,
    morphism_name: str,
    init: MorphismInitFamily,
    step_args: tuple[DrawArg, ...] | None,
    chain: tuple[str, ...],
    morphism_options: tuple[OptionEntry, ...] = (),
) -> ResolvedDist:
    """Unfold a ``~ Family(args)`` init clause.

    When the morphism's declaration carries explicit init args
    (``~ Normal(0, 1)``), those override any step-supplied args
    completely.

    When the init clause is the bare ``~ Family`` form (no
    parentheses), arg slots are filled in canonical order from
    three sources:

    1. Step args (``emission(s_new)``) populate the leading
       positional slots.
    2. Remaining slots whose canonical name matches a morphism
       option (`[scale=0.1]` -> the `scale` slot of `Normal`) take
       the option value.
    3. Anything still empty falls back to the family's canonical
       default (`_FAMILY_DEFAULT_ARGS`).
    """
    if init.args:
        source: tuple[DrawArg, ...] = step_args if step_args else init.args
        wire = tuple(_draw_arg_to_wire(a) for a in source)
        return ResolvedDist(
            family=init.family,
            args=wire,
            original_morphism_name=morphism_name,
            via=chain[:-1] if chain else (),
        )
    defaults = _FAMILY_DEFAULT_ARGS.get(init.family, ())
    arg_names = _FAMILY_ARG_NAMES.get(init.family, ())
    option_map = _options_to_map(morphism_options)
    step_wire = (
        tuple(_draw_arg_to_wire(a) for a in step_args)
        if step_args
        else ()
    )
    wire_list: list[str | float] = list(step_wire)
    for i in range(len(step_wire), len(defaults)):
        name = arg_names[i] if i < len(arg_names) else None
        if name is not None and name in option_map:
            wire_list.append(option_map[name])
        else:
            wire_list.append(defaults[i])
    return ResolvedDist(
        family=init.family,
        args=tuple(wire_list),
        original_morphism_name=morphism_name,
        via=chain[:-1] if chain else (),
    )


def _options_to_map(
    options: tuple[OptionEntry, ...],
) -> dict[str, str | float]:
    """Reduce a tuple of `OptionEntry` to a `key -> wire-value` map
    of numeric / string-valued options. Identifier / list / call
    option shapes are ignored (they describe declaration metadata
    like `role=kernel` rather than family arguments)."""
    out: dict[str, str | float] = {}
    for entry in options:
        value = entry.value
        if isinstance(value, OptionNumber):
            out[entry.key] = value.value
        elif isinstance(value, OptionString):
            out[entry.key] = value.value
    return out


# Per-family ordered parameter names. Mirrors the positional layout
# used by `_FAMILY_DEFAULT_ARGS` so an option-block entry
# (`scale=0.1`) can be routed to the right slot by name.
#
# IMPORTANT: the names AND their order must match
# `torch.distributions.<Family>.arg_constraints.keys()` for every
# entry where torch has a Distribution of that name. The
# `_assert_family_arg_names_match_torch` startup check below
# enforces this so a typo here cannot silently transpose the loc
# and scale slots (which would route `[scale=0.1]` to `loc=0.1`).
_FAMILY_ARG_NAMES: dict[str, tuple[str, ...]] = {
    "Normal": ("loc", "scale"),
    "HalfNormal": ("scale",),
    "Cauchy": ("loc", "scale"),
    "HalfCauchy": ("scale",),
    "Laplace": ("loc", "scale"),
    "LogNormal": ("loc", "scale"),
    "Beta": ("concentration1", "concentration0"),
    "Bernoulli": ("probs",),
    "Gamma": ("concentration", "rate"),
    "InverseGamma": ("concentration", "rate"),
    "Exponential": ("rate",),
    "Uniform": ("low", "high"),
    "StudentT": ("df", "loc", "scale"),
    "Pareto": ("alpha", "scale"),
    "Weibull": ("scale", "concentration"),
}


def _assert_family_arg_names_match_torch() -> None:
    """Startup invariant: every entry in `_FAMILY_ARG_NAMES` for a
    family that torch ships must match
    `torch.distributions.<Family>.arg_constraints.keys()` in name
    AND order.

    Raises `AssertionError` at module-import time if the table
    drifts from torch's ground truth. The cost of running this once
    per process is negligible; the cost of letting a transposition
    bug ship is a silent wrong-density that no syntactic test
    catches.

    Families whose `arg_constraints` is a Python `property` (Uniform,
    Wishart) or that are not present in torch (the custom shim
    families defined in `family_meta`) are skipped: the table's
    entry is the source of truth for those.
    """
    for family, declared in _FAMILY_ARG_NAMES.items():
        cls = getattr(_td, family, None)
        if cls is None:
            continue
        ac = getattr(cls, "arg_constraints", None)
        if not isinstance(ac, dict):
            continue
        # `Bernoulli.arg_constraints` lists both ``probs`` and
        # ``logits`` (alternative parameterisations). Accept any
        # `declared` whose names are a prefix-ordered subset of the
        # torch keys; transposition is still caught.
        torch_keys = tuple(ac.keys())
        prefix = torch_keys[: len(declared)]
        if declared != prefix:
            raise AssertionError(
                f"_FAMILY_ARG_NAMES[{family!r}] = {declared!r} does "
                f"not match the leading prefix of "
                f"torch.distributions.{family}.arg_constraints "
                f"keys ({torch_keys!r}). The option-block routing "
                "depends on this ordering matching torch's positional "
                "parameter contract; fix the table to agree with "
                "torch's `arg_constraints` order."
            )


_assert_family_arg_names_match_torch()


# Canonical default args for kernel morphisms declared `~ Family` with
# no explicit init args. These let the resolver produce a target call
# with the right arity for backends whose family signature is
# positional (Stan's `normal(0, 1)`, WebPPL's `Gaussian({mu, sigma})`).
# Mirror of `_expand_composites._FAMILY_DEFAULT_ARGS` for
# resolver-side use (the expansion pass does in-chain substitution,
# this fallback is for standalone kernels).
_FAMILY_DEFAULT_ARGS: dict[str, tuple[str | float, ...]] = {
    "Normal":       (0.0, 1.0),
    "HalfNormal":   (1.0,),
    "Cauchy":       (0.0, 1.0),
    "HalfCauchy":   (1.0,),
    "Laplace":      (0.0, 1.0),
    "LogNormal":    (0.0, 1.0),
    "Beta":         (1.0, 1.0),
    "Bernoulli":    (0.5,),
    "Gamma":        (1.0, 1.0),
    "InverseGamma": (1.0, 1.0),
    "Exponential":  (1.0,),
    "Uniform":      (0.0, 1.0),
    "StudentT":     (1.0, 0.0, 1.0),
    "Pareto":       (1.0, 1.0),
    "Weibull":      (1.0, 1.0),
    # Multivariate-shape families: placeholder mu / cov literals so
    # the resolved call has the right arity for backends that
    # distinguish positional signatures. Real GP / MatrixNormal
    # emission would derive these from declared axis types.
    # MatrixNormal carries three placeholders (loc, row_cov, col_cov)
    # so backends with a strict matrix-normal signature (Stan's
    # user-defined `matrix_normal_lpdf`, Edward2's
    # `MatrixNormalLinearOperator`) receive the full argument triple.
    "MultivariateNormal": ("[0.0]", "[[1.0]]"),
    "GP":                 ("[0.0]", "[[1.0]]"),
    "MatrixNormal":       ("[0.0]", "[[1.0]]", "[[1.0]]"),
}


def _resolve_expr(
    *,
    morphism_name: str,
    expr: Expr,
    raw_args: tuple[DrawArg, ...] | None,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    family_registry: frozenset[str],
    target: str,
    chain: tuple[str, ...],
) -> ResolvedDist:
    """Unfold a let / morphism init expression. Pure aliases
    (``let a = b``) recurse into the aliased name; composite
    expressions (``a >> b``, ``a @ b``, ``scan(cell)``, ``fan(a, b)``)
    are normally pre-expanded into atomic sample / let chains by
    [`expand_composite_lets`][quivers.transpile._expand_composites.expand_composite_lets]
    before lowering reaches the resolver. Reaching this branch means
    the pre-expansion pass bailed (an unrecognised leaf shape inside
    the composition) and the surface composite has flowed through to
    the resolver intact; raise a precise unsupported-construct error
    naming the offending kind."""
    if isinstance(expr, ExprIdent):
        return resolve_step_dist(
            expr.name,
            raw_args,
            morphisms=morphisms,
            lets=lets,
            family_registry=family_registry,
            target=target,
            _seen=chain,
        )
    expr_kind = str(getattr(expr, "kind", type(expr).__name__))
    raise UnsupportedConstruct(
        target,
        [
            f"let:composite_expression:{expr_kind}",
            (
                f"morphism / let {morphism_name!r} resolves to a "
                f"composite expression of kind {expr_kind!r} that the "
                "pre-lower expansion pass could not flatten into "
                "atomic sample / let steps. Extend "
                "`_flatten_compose` in "
                "`quivers.transpile._expand_composites` to recognise "
                "the leaf shape, or rewrite the source with a direct "
                "`~ Family(args)` declaration / a per-step `sample`."
            ),
        ],
    )


__all__ = [
    "ResolvedDist",
    "build_let_table",
    "build_morphism_table",
    "resolve_step_dist",
]
