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
import inspect as _inspect

import torch.distributions as _td

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgScalar,
    Expr,
    ExprIdent,
    DefineDecl,
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
from quivers.transpile._draw_args import (
    encode_index,
    is_matrix,
    list_atoms,
    matrix_rows,
)


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
    if isinstance(arg, DrawArgIndex):
        return encode_index(arg)
    if isinstance(arg, DrawArgDist):
        raise UnsupportedConstruct(
            "qvr-transpile",
            [
                f"nested-distribution-arg:{arg.family}: a "
                "distribution-valued argument has no wire form"
            ],
        )
    if isinstance(arg, DrawArgList):
        if is_matrix(arg):
            rows = ", ".join(
                "[" + ", ".join(_atom_to_text(e) for e in row) + "]"
                for row in matrix_rows(arg)
            )
            return f"[{rows}]"
        return (
            "[" + ", ".join(_atom_to_text(e) for e in list_atoms(arg)) + "]"
        )
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
    ``module``. A plural-name declaration contributes one entry per
    name (each name is an independent morphism with the same
    signature and init). Duplicate names are an error (the QVR
    compiler also rejects them, but the resolver catches it locally
    with a clearer transpile-time message)."""
    out: dict[str, MorphismDecl] = {}
    for stmt in module.statements:
        if isinstance(stmt, MorphismDecl):
            for name in stmt.names:
                if name in out:
                    msg = (
                        f"duplicate morphism declaration {name!r}: "
                        f"first at line {out[name].line}, again at "
                        f"line {stmt.line}"
                    )
                    raise UnsupportedConstruct("qvr-transpile", [msg])
                out[name] = stmt
    return out


def build_let_table(module: Module) -> dict[str, Expr]:
    """Return name → expr for every top-level ``let_decl``."""
    out: dict[str, Expr] = {}
    for stmt in module.statements:
        if isinstance(stmt, DefineDecl):
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
        param_source = next(
            (e.value.value for e in decl.options if e.key == "param_source"),
            None,
        )
        if param_source is not None:
            msg = (
                f"morphism {morphism_name!r} draws its parameters from a "
                f"{param_source!r} network. The network's weights are "
                f"model-internal and appear in neither the wire form nor the "
                f"sample sites, so no backend can reconstruct the mean the "
                f"morphism computes. Express the network as explicit sampled "
                f"weights and a deterministic forward pass, or observe against "
                f"a closed-form family."
            )
            raise UnsupportedConstruct(
                target, [f"param-source:{param_source}", msg]
            )
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

    ``init.args`` are already in wire form (``str`` identifiers or
    ``float`` literals); the step's own args arrive as `DrawArg`
    variants and are lowered to wire form here. When the declaration
    carries explicit init args and the step also supplies args, the
    step args take precedence; otherwise the declaration's init args
    are used.

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
        wire: tuple[str | float, ...]
        if step_args:
            wire = tuple(_draw_arg_to_wire(a) for a in step_args)
        else:
            wire = init.args
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
    "Pareto": ("scale", "alpha"),
    "Weibull": ("scale", "concentration"),
}


def _assert_family_arg_names_match_torch() -> None:
    """Startup invariant: every entry in `_FAMILY_ARG_NAMES` for a
    family that torch ships must match that distribution's positional
    constructor parameters in name AND order.

    The constructor signature, not ``arg_constraints``, is the
    positional contract a QVR call site writes against: ``~ Pareto(a,
    b)`` binds ``a`` and ``b`` the way ``torch.distributions.Pareto``
    binds them, and the conditional family classes in
    [quivers.continuous.families][] declare their parameters in the
    same order. For most families the two agree, but they can diverge
    (``Pareto.arg_constraints`` is keyed ``alpha, scale`` while the
    constructor takes ``scale, alpha``), and following the wrong one
    transposes the slots and silently changes the density.

    Raises `AssertionError` at module-import time if the table drifts
    from torch's ground truth. The cost of running this once per
    process is negligible; the cost of letting a transposition bug
    ship is a silent wrong-density that no syntactic test catches.

    Families that are not present in torch (the custom shim families
    defined in `family_meta`) are skipped: the table's entry is the
    source of truth for those.
    """
    for family, declared in _FAMILY_ARG_NAMES.items():
        cls = getattr(_td, family, None)
        if cls is None:
            continue
        try:
            params = _inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            continue
        # Alternative parameterisations trail the canonical ones
        # (``Bernoulli(probs, logits)``), so compare against the
        # leading prefix; transposition is still caught.
        ctor_names = tuple(
            name
            for name in params
            if name not in ("self", "validate_args")
        )
        prefix = ctor_names[: len(declared)]
        if declared != prefix:
            raise AssertionError(
                f"_FAMILY_ARG_NAMES[{family!r}] = {declared!r} does "
                f"not match the leading positional parameters of "
                f"torch.distributions.{family}.__init__ "
                f"({ctor_names!r}). The option-block routing depends "
                "on this ordering matching torch's positional "
                "parameter contract; fix the table to agree with the "
                "constructor signature."
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
    # Multivariate-shape families (`MultivariateNormal`,
    # `MatrixNormal`, `GP`) have no entry here. They opt in to
    # structured lowering via the `structured_lowering` field on their
    # [`FamilyMeta`][quivers.transpile.family_meta.FamilyMeta]; the
    # [`Lower._lower_sample_from_meta`][quivers.transpile.lower.Lower._lower_sample_from_meta]
    # dispatch then intercepts the bare ``~ Family`` form and
    # synthesises the data-input names with the right matrix / vector
    # / cov-matrix shape from the morphism's `[over=...]` axes. A
    # placeholder string here would survive into the data block as a
    # free identifier and produce invalid syntax in every backend.
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
