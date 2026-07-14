"""Morphism / define-binding resolution shared by every transpile
backend.

A ``sample x <- name`` step's ``morphism`` slot may refer to one of
three things:

1. A distribution family name (e.g. ``Beta``): the existing
   ``_FAMILIES`` map carries the target name.
2. A declared ``morphism`` whose ``~ Family(args)`` init clause
   names the underlying distribution.
3. A ``define`` binding whose RHS is itself a morphism reference (the
   common shape is a pure alias or a Kleisli composition).

The resolver here turns case 2 into the equivalent of case 1 by
unfolding the declared morphism's `init_family`. Case 3 is unfolded
recursively: a define binding to a bare identifier resolves to
whatever that identifier resolves to; composite expressions raise
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
with a clear message naming the composition operator.

The output is a [`ResolvedDist`][quivers.transpile.backends._resolve.ResolvedDist]
record carrying ``family`` (the canonical QVR family name) and
``args`` (the tuple of literal-or-variable arguments). Callers feed
``family`` into their backend-specific ``_FAMILIES`` table and emit
the call with ``args``.
"""

from __future__ import annotations

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    DefineDecl,
    DrawArg,
    DrawArgIndex,
    DrawArgName,
    DrawArgScalar,
    Expr,
    ExprIdent,
    Module,
    MorphismDecl,
    MorphismInitFamily,
)
from quivers.transpile._api import UnsupportedConstruct


class ResolvedDist(dx.Model):
    """The (family, args) pair a sample / observe step resolves to.

    ``family`` is the canonical QVR family name, exactly as it would
    appear if the user had written ``sample x <- family(args)``.
    ``args`` is the tuple of positional arguments after resolution.

    The ``original_morphism_name`` and ``via`` fields are diagnostic:
    when a sample resolves through a morphism declaration, the
    transpiled output can reference both the underlying family and the
    original morphism name (some backends prefer one or the other).
    """

    family: str
    args: tuple[str | float, ...]
    original_morphism_name: str
    via: tuple[str, ...] = ()
    """The chain of intermediate define / morphism names the resolver
    walked through to reach ``family``. Empty when the morphism
    slot was already a family name."""


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


def build_define_table(module: Module) -> dict[str, Expr]:
    """Return name → expr for every top-level ``define_decl``."""
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
    defines: dict[str, Expr],
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
        morphism's name, or a define-binding name.
    raw_args
        Positional [`DrawArg`][quivers.dsl.ast_nodes.DrawArg]
        arguments on the step (``None`` is treated as empty); the
        resolver flattens them to atomic ``str | float`` forms
        before resolution. When the resolver unfolds a morphism with
        its own ``~ Family(args)`` init clause, the step-supplied
        ``raw_args`` take precedence over the declaration's defaults;
        in current practice the step does not supply args when
        referring to a morphism (the morphism's args carry the
        parameter set).
    morphisms
        Name → MorphismDecl table from
        [`build_morphism_table`][quivers.transpile.backends._resolve.build_morphism_table].
    defines
        Name → Expr table from
        [`build_define_table`][quivers.transpile.backends._resolve.build_define_table].
    family_registry
        Frozen set of canonical QVR family names. When
        ``morphism_name`` is in this set, the resolver returns
        immediately with the step's own args.
    target
        Backend name (for error messages).
    _seen
        Internal: the chain of names visited during resolution. The
        resolver detects cycles (``define a = b; define b = a``) by
        membership in this tuple.
    """
    atoms = _step_args_to_atoms(raw_args, target=target)
    return _resolve_atoms(
        morphism_name,
        atoms,
        morphisms=morphisms,
        defines=defines,
        family_registry=family_registry,
        target=target,
        _seen=_seen,
    )


def _step_args_to_atoms(
    args: tuple[DrawArg, ...] | None,
    *,
    target: str,
) -> tuple[str | float, ...] | None:
    """Flatten a step's tagged `DrawArg` tuple into the atomic
    ``str | float`` forms the resolver and renderers consume.

    `DrawArgName` contributes its identifier text, `DrawArgScalar`
    its value, and `DrawArgIndex` the bracket-indexed reference
    string (``name[i0][i1]``) that the lowering pass parses back
    into an [`IRArgRef`][quivers.transpile.ir.IRArgRef].
    Compositional args (`DrawArgDist`, `DrawArgList`) have no atomic
    form and raise
    [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct].
    """
    if args is None:
        return None
    out: list[str | float] = []
    for arg in args:
        if isinstance(arg, DrawArgName):
            out.append(arg.text)
        elif isinstance(arg, DrawArgScalar):
            out.append(arg.value)
        elif isinstance(arg, DrawArgIndex):
            out.append(arg.name + "".join(f"[{i}]" for i in arg.indices))
        else:
            raise UnsupportedConstruct(
                target,
                [
                    f"draw-arg:{arg.kind}: compositional draw args "
                    f"are not resolvable to a (family, args) pair"
                ],
            )
    return tuple(out)


def _resolve_atoms(
    morphism_name: str,
    raw_args: tuple[str | float, ...] | None,
    *,
    morphisms: dict[str, MorphismDecl],
    defines: dict[str, Expr],
    family_registry: frozenset[str],
    target: str,
    _seen: tuple[str, ...] = (),
) -> ResolvedDist:
    """Resolution core over atomic ``str | float`` args; the public
    [`resolve_step_dist`][quivers.transpile.backends._resolve.resolve_step_dist]
    flattens the step's `DrawArg` tuple and delegates here."""
    if morphism_name in family_registry:
        args = raw_args or ()
        if not args:
            args = _FAMILY_DEFAULT_ARGS.get(morphism_name, ())
        return ResolvedDist(
            family=morphism_name,
            args=args,
            original_morphism_name=morphism_name,
        )

    if morphism_name in _seen:
        msg = (
            f"morphism / define cycle while resolving "
            f"{morphism_name!r}; "
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
            )
        if decl.init_expr is not None:
            return _resolve_expr(
                morphism_name=morphism_name,
                expr=decl.init_expr,
                raw_args=raw_args,
                morphisms=morphisms,
                defines=defines,
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

    if morphism_name in defines:
        return _resolve_expr(
            morphism_name=morphism_name,
            expr=defines[morphism_name],
            raw_args=raw_args,
            morphisms=morphisms,
            defines=defines,
            family_registry=family_registry,
            target=target,
            chain=chain,
        )

    msg = (
        f"sample / observe step references {morphism_name!r} which "
        f"is neither a family in the registry, a declared morphism, "
        f"nor a define-bound name"
    )
    raise UnsupportedConstruct(target, [f"family:{morphism_name}", msg])


def _from_init_family(
    *,
    morphism_name: str,
    init: MorphismInitFamily,
    step_args: tuple[str | float, ...] | None,
    chain: tuple[str, ...],
) -> ResolvedDist:
    """Unfold a ``~ Family(args)`` init clause. Step-supplied args
    override the declaration's defaults when both are present. When
    neither the step nor the init clause supplies args (the common
    `morphism foo : T -> T [role=kernel] ~ Family` form), fall back
    to the family's canonical default parameters so the resulting
    call has the arity the target backend expects."""
    args = step_args if step_args else init.args
    if not args:
        args = _FAMILY_DEFAULT_ARGS.get(init.family, ())
    return ResolvedDist(
        family=init.family,
        args=args,
        original_morphism_name=morphism_name,
        via=chain[:-1] if chain else (),
    )


# Canonical default args for kernel morphisms declared `~ Family` with
# no explicit init args. These let the resolver produce a target call
# with the right arity for backends whose family signature is
# positional (Stan's `normal(0, 1)`, WebPPL's `Gaussian({mu, sigma})`).
# Mirror of `_expand_composites._FAMILY_DEFAULT_ARGS` for
# resolver-side use (the expansion pass does in-chain substitution,
# this fallback is for standalone kernels).
_FAMILY_DEFAULT_ARGS: dict[str, tuple[str | float, ...]] = {
    "Normal": (0.0, 1.0),
    "HalfNormal": (1.0,),
    "Cauchy": (0.0, 1.0),
    "HalfCauchy": (1.0,),
    "Laplace": (0.0, 1.0),
    "LogNormal": (0.0, 1.0),
    "Beta": (1.0, 1.0),
    "Bernoulli": (0.5,),
    "Gamma": (1.0, 1.0),
    "InverseGamma": (1.0, 1.0),
    "Exponential": (1.0,),
    "Uniform": (0.0, 1.0),
    "StudentT": (1.0, 0.0, 1.0),
    "Pareto": (1.0, 1.0),
    "Weibull": (1.0, 1.0),
    # Multivariate-shape families: placeholder mu / cov vector
    # literals so the resolved call has the standard 2-arg signature
    # WebPPL / NumPyro / Stan / PyMC expect. Real GP / MatrixNormal
    # emission would derive these from declared axis types.
    "MultivariateNormal": ("[0.0]", "[[1.0]]"),
    "GP": ("[0.0]", "[[1.0]]"),
    "MatrixNormal": ("[0.0]", "[[1.0]]"),
}


def _resolve_expr(
    *,
    morphism_name: str,
    expr: Expr,
    raw_args: tuple[str | float, ...] | None,
    morphisms: dict[str, MorphismDecl],
    defines: dict[str, Expr],
    family_registry: frozenset[str],
    target: str,
    chain: tuple[str, ...],
) -> ResolvedDist:
    """Unfold a define / morphism init expression. Pure aliases
    (``define a = b``) recurse; composite expressions (``a >> b``,
    ``a @ b``, ``identity(X)``) are not yet supported and raise."""
    if isinstance(expr, ExprIdent):
        return _resolve_atoms(
            expr.name,
            raw_args,
            morphisms=morphisms,
            defines=defines,
            family_registry=family_registry,
            target=target,
            _seen=chain,
        )
    expr_kind = str(getattr(expr, "kind", type(expr).__name__))
    raise UnsupportedConstruct(
        target,
        [
            f"define:composite_expression:{expr_kind}",
            (
                f"morphism / define {morphism_name!r} resolves to a "
                f"composite expression of kind {expr_kind!r}; "
                f"transpile backends only unfold pure-alias "
                f"bindings today. Replace the composition with a "
                f"direct ``~ Family(args)`` declaration or with a "
                f"separate `sample` per stochastic step."
            ),
        ],
    )


__all__ = [
    "ResolvedDist",
    "build_define_table",
    "build_morphism_table",
    "resolve_step_dist",
]
