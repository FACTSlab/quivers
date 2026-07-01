"""Sugar table for the compositional measure algebra.

Brms- and Stan-style named families that desugar to canonical
operator-form expressions at compile time when all arguments are
constants. The user-facing surface keeps the ergonomic names:

    observe y <- TruncatedNormal(0.0, 1.0, 0.0, 1.0)
    observe y <- HalfNormal(2.0)
    observe y <- HalfCauchy(2.5)
    observe y <- HalfStudentT(4.0, 1.5)

while the compiler internally walks the canonical desugared form:

    observe y <- Restrict(Normal(0.0, 1.0), 0.0, 1.0)
    observe y <- Restrict(Normal(0.0, 2.0), 0.0)
    observe y <- Restrict(Cauchy(0.0, 2.5), 0.0)
    observe y <- Restrict(StudentT(4.0, 0.0, 1.5), 0.0)

The sugar desugaring runs only when every argument is a literal
(`DrawArgScalar`); sugar calls with free-variable arguments continue
to route through their dedicated inline family entries
(`ZeroInflatedPoisson`, `HurdlePoisson`, `MixtureNormal` etc.),
which internally compose the same `Mixture` / `Restrict` / `PointMass`
operators. The dual surface is the "two-way recognition" the design
note specifies: source can be either form, the compiler canonicalises
to the operator form internally when it can, and the pretty-printer
re-sugars on the way out.
"""

from __future__ import annotations

from collections.abc import Callable

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgDist,
    DrawArgList,
    DrawArgScalar,
    ObserveStep,
    SampleStep,
)


def _dist(family: str, *args: DrawArg) -> DrawArgDist:
    return DrawArgDist(family=family, args=tuple(args))


def _scalar(value: float) -> DrawArgScalar:
    return DrawArgScalar(value=float(value))


def _all_scalar(args: tuple[DrawArg, ...]) -> bool:
    return all(isinstance(a, DrawArgScalar) for a in args)


def _desugar_truncated_normal(
    args: tuple[DrawArg, ...],
) -> tuple[str, tuple[DrawArg, ...]]:
    if len(args) != 4:
        raise ValueError(
            f"TruncatedNormal: expected (mu, sigma, low, high), got {len(args)} args"
        )
    mu, sigma, low, high = args
    return "Restrict", (_dist("Normal", mu, sigma), low, high)


def _desugar_half(base_family: str, default_loc: float = 0.0):
    """Build a `Half{base_family}(scale)` -> `Restrict(base(0, scale), 0)`
    rewriter. Used for Normal, Cauchy, Laplace, ...
    """

    def _impl(args: tuple[DrawArg, ...]) -> tuple[str, tuple[DrawArg, ...]]:
        if len(args) != 1:
            raise ValueError(
                f"Half{base_family}: expected (scale,), got {len(args)} args"
            )
        (scale,) = args
        return "Restrict", (
            _dist(base_family, _scalar(default_loc), scale),
            _scalar(0.0),
        )

    return _impl


def _desugar_half_student_t(
    args: tuple[DrawArg, ...],
) -> tuple[str, tuple[DrawArg, ...]]:
    if len(args) != 2:
        raise ValueError(f"HalfStudentT: expected (nu, scale), got {len(args)} args")
    nu, scale = args
    return "Restrict", (
        _dist("StudentT", nu, _scalar(0.0), scale),
        _scalar(0.0),
    )


# Sugar entries that produce constant-arg operator-form expressions.
# Each rewriter consumes the original draw args (already validated
# all-scalar by `_all_scalar`) and emits a (operator_family, new_args)
# pair the compiler dispatches on.
SUGAR_TABLE: dict[
    str, Callable[[tuple[DrawArg, ...]], tuple[str, tuple[DrawArg, ...]]]
] = {
    "TruncatedNormal": _desugar_truncated_normal,
    "HalfNormal": _desugar_half("Normal"),
    "HalfCauchy": _desugar_half("Cauchy"),
    "HalfLaplace": _desugar_half("Laplace"),
    "HalfStudentT": _desugar_half_student_t,
}


def desugar_step(step: SampleStep | ObserveStep) -> SampleStep | ObserveStep:
    """Rewrite a step whose morphism is a sugar family into the
    canonical operator-algebra form. Steps whose morphism is not in
    the sugar table, or whose args contain free variables, pass
    through unchanged.

    Recurses on the args so nested sugar (e.g. `Mixture([0.3, 0.7],
    [PointMass(0), HalfNormal(1.0)])` with all literals) is fully
    desugared in one pass.
    """
    args = step.args
    if args is not None:
        args = tuple(_desugar_arg(a) for a in args)
    morphism = step.morphism
    if morphism in SUGAR_TABLE and args is not None and _all_scalar(args):
        morphism, args = SUGAR_TABLE[morphism](args)
    if morphism == step.morphism and args == step.args:
        return step
    return step.with_(morphism=morphism, args=args)


def _desugar_arg(arg: DrawArg) -> DrawArg:
    """Recursively desugar a draw arg: `DrawArgDist`s whose family
    name is in the sugar table and whose args are all scalar are
    rewritten; lists are mapped element-wise; everything else passes
    through.
    """
    if isinstance(arg, DrawArgDist):
        inner_args = tuple(_desugar_arg(a) for a in arg.args)
        if arg.family in SUGAR_TABLE and _all_scalar(inner_args):
            new_family, new_args = SUGAR_TABLE[arg.family](inner_args)
            return DrawArgDist(family=new_family, args=new_args)
        return DrawArgDist(family=arg.family, args=inner_args)
    if isinstance(arg, DrawArgList):
        return DrawArgList(items=tuple(_desugar_arg(item) for item in arg.items))
    return arg


__all__ = ["SUGAR_TABLE", "desugar_step"]
