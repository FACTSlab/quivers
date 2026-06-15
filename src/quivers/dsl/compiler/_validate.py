"""Static validation passes that emit `Violation` diagnostics.

The pass walks the parsed `Module`, locates every
SampleStep / ObserveStep / MarginalizeStep / `MorphismInitFamily`
that resolves to a registered distribution family, and emits the
following structured diagnostics:

* ``code="implicit-family-defaults"`` -- ``severity="warning"``:
  the step (or init clause) supplied no positional args and the
  resolver substituted the family's canonical default parameter
  set. The current pipeline still accepts this form, but the
  defaults will be removed once every shipped example moves to an
  explicit ``~ Family(args)`` declaration.

* ``code="family-arg-shape"`` -- ``severity="error"``: the
  positional arity of the user-supplied args does not match the
  family's `arg_constraints` dict (the
  authoritative parameter set held on the underlying
  `torch.distributions.Distribution`).

* ``code="family-arg-shape"`` -- ``severity="warning"``: a
  literal vector argument has the wrong length for the family's
  reported event size, or a literal simplex argument has elements
  that do not sum to 1.
"""

from __future__ import annotations

import torch.distributions.constraints as _c
from torch.distributions.constraints import Constraint

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgList,
    DrawArgMatrix,
    DrawArgScalar,
    LetDecl,
    MarginalizeStep,
    Module,
    MorphismDecl,
    MorphismInitFamily,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    SampleStep,
)
from quivers.dsl.constraints import Violation
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.backends._resolve import (
    ResolvedDist,
    build_let_table,
    build_morphism_table,
    resolve_step_dist,
)
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta


def validate_family_arg_shapes(module: Module) -> list[Violation]:
    """Walk every draw step in ``module`` and emit `Violation`
    diagnostics for argument-shape mismatches and implicit-default
    fallbacks.

    Steps and init clauses whose morphism slot does not resolve to a
    registered family are silently skipped: the resolver-level
    diagnostic surfaces those cases when the user actually invokes a
    backend transpile or compile.
    """
    out: list[Violation] = []
    morphisms = build_morphism_table(module)
    lets = build_let_table(module)
    family_set = frozenset(FAMILY_META)

    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            _walk_program(stmt, morphisms, lets, family_set, out)
        elif isinstance(stmt, MorphismDecl):
            _walk_morphism(stmt, morphisms, family_set, out)

    return out


def _walk_program(
    program: ProgramDecl,
    morphisms: dict[str, MorphismDecl],
    lets: dict,
    family_set: frozenset[str],
    out: list[Violation],
) -> None:
    _walk_steps(program.draws, morphisms, lets, family_set, out)


def _walk_steps(
    steps: tuple[ProgramStep, ...],
    morphisms: dict[str, MorphismDecl],
    lets: dict,
    family_set: frozenset[str],
    out: list[Violation],
) -> None:
    for step in steps:
        if isinstance(step, (SampleStep, ObserveStep, MarginalizeStep)):
            _check_step(step, morphisms, lets, family_set, out)
        if isinstance(step, MarginalizeStep):
            _walk_steps(step.scope, morphisms, lets, family_set, out)


def _walk_morphism(
    decl: MorphismDecl,
    morphisms: dict[str, MorphismDecl],
    family_set: frozenset[str],
    out: list[Violation],
) -> None:
    init = decl.init_family
    if init is None:
        return
    if init.family not in family_set:
        return
    meta = FAMILY_META[init.family]
    if not init.args:
        out.append(
            Violation(
                code="implicit-family-defaults",
                severity="warning",
                message=(
                    f"morphism {decl.name!r}: `~ {init.family}` carries no "
                    f"explicit arguments; the resolver substitutes the "
                    f"family's canonical default parameters. Declare "
                    f"`~ {init.family}(args)` explicitly to silence this "
                    f"warning."
                ),
                line=decl.line,
                col=decl.col,
            )
        )
    else:
        _check_args_shape(
            family=init.family,
            args=init.args,
            meta=meta,
            line=init.line or decl.line,
            col=init.col or decl.col,
            origin=f"morphism {decl.name!r} init clause",
            out=out,
        )


def _check_step(
    step: SampleStep | ObserveStep | MarginalizeStep,
    morphisms: dict[str, MorphismDecl],
    lets: dict,
    family_set: frozenset[str],
    out: list[Violation],
) -> None:
    try:
        resolved: ResolvedDist = resolve_step_dist(
            step.morphism,
            step.args,
            morphisms=morphisms,
            lets=lets,
            family_registry=family_set,
            target="qvr-validate",
        )
    except UnsupportedConstruct:
        return
    meta = FAMILY_META.get(resolved.family)
    if meta is None:
        return
    # Implicit-defaults check: the step (or its referenced init
    # clause) carried no args, the resolver filled defaults.
    if not step.args:
        # When the morphism slot is itself a morphism declaration with
        # explicit init args, the warning emits at the morphism site
        # (handled by `_walk_morphism`) rather than here.
        decl = morphisms.get(step.morphism)
        decl_has_explicit = (
            decl is not None
            and decl.init_family is not None
            and bool(decl.init_family.args)
        )
        if not decl_has_explicit and resolved.args:
            out.append(
                Violation(
                    code="implicit-family-defaults",
                    severity="warning",
                    message=(
                        f"step `<- {step.morphism}`: no positional "
                        f"arguments; the resolver substitutes the "
                        f"family's canonical default parameters. "
                        f"Declare `<- {step.morphism}(args)` explicitly "
                        f"to silence this warning."
                    ),
                    line=step.line,
                    col=step.col,
                )
            )
        return
    _check_args_shape(
        family=resolved.family,
        args=step.args,
        meta=meta,
        line=step.line,
        col=step.col,
        origin=f"step `<- {step.morphism}`",
        out=out,
    )


def _check_args_shape(
    *,
    family: str,
    args: tuple[DrawArg, ...],
    meta: FamilyMeta,
    line: int,
    col: int,
    origin: str,
    out: list[Violation],
) -> None:
    """Check positional arity and elementwise shape compatibility
    against the family's `arg_constraints`."""
    arg_constraints = _read_arg_constraints(meta)
    if arg_constraints is None:
        # Property-form arg_constraints: skip the shape check rather
        # than raise; the transpile-time Lower handles the sentinel
        # path for these families.
        return
    arg_names = tuple(arg_constraints.keys())
    if len(args) > len(arg_names):
        out.append(
            Violation(
                code="family-arg-shape",
                severity="error",
                message=(
                    f"family {family!r} expects {len(arg_names)} positional "
                    f"argument(s) {list(arg_names)!r}; {origin} supplied "
                    f"{len(args)}"
                ),
                line=line,
                col=col,
            )
        )
        return
    for arg, (arg_name, constraint) in zip(args, arg_constraints.items()):
        _check_arg_against_constraint(
            family=family,
            arg=arg,
            arg_name=arg_name,
            constraint=constraint,
            line=line,
            col=col,
            origin=origin,
            out=out,
        )


def _check_arg_against_constraint(
    *,
    family: str,
    arg: DrawArg,
    arg_name: str,
    constraint: Constraint,
    line: int,
    col: int,
    origin: str,
    out: list[Violation],
) -> None:
    if isinstance(constraint, _c._IndependentConstraint) and constraint.event_dim >= 1:
        if isinstance(arg, DrawArgScalar):
            # Scalar broadcasting is the Lower-path's responsibility;
            # this is not a shape error.
            return
        if isinstance(arg, DrawArgList):
            literal = _list_literal_length(arg)
            if literal is None:
                return
            # Without an instance event_shape we can't determine the
            # required length; the per-call validation in Lower fills
            # this in. Skip silently here.
            return
        if isinstance(arg, DrawArgMatrix):
            # Same: matrix-shape validation requires a sentinel.
            return
    if isinstance(constraint, _c._Simplex) and isinstance(arg, DrawArgList):
        literal_values = _list_literal_floats(arg)
        if literal_values is None:
            return
        total = sum(literal_values)
        if not _approx_equal(total, 1.0):
            out.append(
                Violation(
                    code="family-arg-shape",
                    severity="warning",
                    message=(
                        f"family {family!r}: argument {arg_name!r} is "
                        f"declared as a simplex but the literal "
                        f"{literal_values!r} sums to {total!r} (expected 1.0)"
                    ),
                    line=line,
                    col=col,
                )
            )


def _read_arg_constraints(meta: FamilyMeta) -> dict[str, Constraint] | None:
    cls_attr = meta.distribution_class.arg_constraints
    if isinstance(cls_attr, dict):
        return cls_attr
    return None


def _list_literal_length(arg: DrawArgList) -> int | None:
    """Return the literal length when every element is a numeric
    literal; ``None`` otherwise."""
    for e in arg.elements:
        if not isinstance(e, (int, float)) or isinstance(e, bool):
            return None
    return len(arg.elements)


def _list_literal_floats(arg: DrawArgList) -> list[float] | None:
    """Return the float values when every element is a numeric
    literal; ``None`` otherwise."""
    out: list[float] = []
    for e in arg.elements:
        if isinstance(e, bool):
            return None
        if isinstance(e, (int, float)):
            out.append(float(e))
        else:
            return None
    return out


def _approx_equal(a: float, b: float, *, atol: float = 1e-6) -> bool:
    return abs(a - b) <= atol


__all__ = ["validate_family_arg_shapes"]
