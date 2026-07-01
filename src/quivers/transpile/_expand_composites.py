"""AST preprocessing: expand composite-let bindings into sample chains
and flatten MarginalizeStep / ScoreStep / LetStep scopes into plain
sample / observe / assignment sequences.

The pass operates at the Module level: it rebuilds every
`program_decl` in place. Three rewrites fire:

1. **Composite let**: `let chain = prior >> likelihood` binds a
   Kleisli composition. A single `sample x <- chain` step rewrites
   into a chain of atomic sample steps:

       sample _chain_0 <- prior
       sample x        <- likelihood

   where the second step's parameters are derived from the first
   step's output. For kernel morphisms declared with `~ Family` and
   no explicit args (the common form), the expansion fills in
   canonical default parameters for the family: location/scale
   families like `Normal` get `(prev, 1.0)` with `prev` being the
   upstream step's output variable (or `0.0` at the head of the
   chain); shape/rate families like `Gamma` get canonical defaults
   `(1.0, 1.0)`. The defaults are mild priors matching the Bayesian
   intuition that an unparameterized kernel in a Kleisli chain is the
   standard kernel centered on its input.

2. **MarginalizeStep**: the discrete-marginalization step

       marginalize cls : T <- Categorical(probs) [over=...]:
           observe r <- Normal(mu[cls], sigma)

   rewrites to a plain sample-then-observe pair:

       sample cls <- Categorical(probs)
       observe r  <- Normal(mu[cls], sigma)

   The rewrite is operationally equivalent under MCMC inference for
   any backend that supports discrete-latent sampling (NumPyro, Pyro,
   PyMC, Turing.jl, Gen.jl, Church, WebPPL, BUGS, JAGS). Stan does
   not natively sample discrete parameters and requires explicit
   `log_sum_exp` marginalization; for Stan the rewrite produces
   `categorical(probs)` in a parameter block, which Stan's compiler
   will reject. (Stan-specific marginalization is tracked as future
   walker work; see `docs/semantics/transpile-correctness.md` §5.6.)

3. **ScoreStep** / **LetStep**: passthrough for now; tracked.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    Expr,
    ExprCompose,
    ExprFan,
    ExprIdent,
    ExprRepeat,
    ExprStack,
    ExprTensorProduct,
    LetDecl,
    MarginalizeStep,
    Module,
    MorphismDecl,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    SampleStep,
)


# Canonical default args per family for kernels that ship `~ Family`
# with no init args. The first arg becomes `prev_output` when the
# step is mid-chain; remaining args use these defaults verbatim.
_FAMILY_DEFAULT_ARGS: dict[str, tuple[float, ...]] = {
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
    "Categorical": (),
    "Dirichlet": (),
    "MultivariateNormal": (),
}


def expand_composite_lets(
    module: Module,
    *,
    target: str | None = None,
) -> Module:
    """Rewrite `program_decl` bodies so composite-let sample steps
    become equivalent chains of atomic sample steps.

    A composite let is a `LetDecl` whose `.expr` is an `ExprCompose`
    (the `prior >> likelihood` form). Each `SampleStep` /
    `ObserveStep` whose `morphism` slot names such a let is rewritten
    into a sequence of fresh `SampleStep`s, one per element of the
    composition chain, with the trailing step keeping the original
    step's bound variable name.

    The returned `Module` shares vertex identity with the input for
    every non-rewritten statement; only the `program_decl`s with
    composite-let references are rebuilt.
    """
    morphism_table: dict[str, MorphismDecl] = {
        s.name: s for s in module.statements if isinstance(s, MorphismDecl)
    }
    let_table: dict[str, Expr] = {
        s.name: s.expr for s in module.statements if isinstance(s, LetDecl)
    }

    # Stan needs explicit `log_sum_exp` marginalization (it cannot
    # natively sample discrete parameters); for Stan we leave
    # MarginalizeStep nodes intact and let the Stan walker emit the
    # enumeration loop. Every other backend (NumPyro / Pyro / PyMC /
    # Edward2 / Turing / Gen / Church / WebPPL / BUGS / JAGS)
    # natively samples discrete latents under MCMC, so the
    # sample-then-scope rewrite is operationally correct.
    flatten_marginalize = target != "stan"
    new_statements: list = []
    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            new_draws = _expand_draws(
                stmt.draws,
                morphisms=morphism_table,
                lets=let_table,
                flatten_marginalize=flatten_marginalize,
            )
            if new_draws is stmt.draws:
                new_statements.append(stmt)
            else:
                new_statements.append(stmt.with_(draws=tuple(new_draws)))
        else:
            new_statements.append(stmt)
    return module.with_(statements=tuple(new_statements))


def _expand_draws(
    draws: tuple[ProgramStep, ...],
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    flatten_marginalize: bool = True,
) -> tuple[ProgramStep, ...]:
    """Expand every SampleStep / ObserveStep whose morphism slot
    resolves to a composite-let chain. When `flatten_marginalize` is
    True (the default for every backend except Stan), flatten every
    MarginalizeStep into a sample-then-scope-body sequence."""
    any_changed = False
    out: list[ProgramStep] = []
    counter = 0
    for step in draws:
        if isinstance(step, MarginalizeStep) and flatten_marginalize:
            any_changed = True
            # Rewrite `marginalize cls <- F(args): scope` as
            # `sample cls <- F(args); scope...`. Operationally
            # equivalent under MCMC for backends that support discrete-
            # latent sampling; Stan-specific log_sum_exp emission is
            # future walker work.
            out.append(
                SampleStep(
                    vars=(step.var,),
                    morphism=step.morphism,
                    args=step.args,
                    index=step.index,
                    options=step.options,
                    line=step.line,
                    col=step.col,
                )
            )
            scope_expanded = _expand_draws(step.scope, morphisms=morphisms, lets=lets)
            out.extend(scope_expanded)
            continue
        if isinstance(step, (SampleStep, ObserveStep)):
            chain = _resolve_to_chain(step.morphism, morphisms=morphisms, lets=lets)
            if chain is not None:
                any_changed = True
                expanded, counter = _expand_step(
                    step, chain, morphisms=morphisms, counter=counter
                )
                out.extend(expanded)
                continue
        out.append(step)
    return tuple(out) if any_changed else draws


def _resolve_to_chain(
    name: str,
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    _seen: tuple[str, ...] = (),
) -> tuple[str, ...] | None:
    """If `name` is a composite-let binding, return the ordered tuple
    of morphism / family names in the composition chain. Otherwise
    return None.

    Resolves through alias chains: `let a = b; let b = c >> d` returns
    `(c, d)`.
    """
    if name in _seen:
        return None
    if name not in lets:
        return None
    expr = lets[name]
    if isinstance(expr, ExprIdent):
        return _resolve_to_chain(
            expr.name,
            morphisms=morphisms,
            lets=lets,
            _seen=(*_seen, name),
        )
    if isinstance(expr, ExprCompose):
        return _flatten_compose(expr)
    return None


def _flatten_compose(expr: ExprCompose) -> tuple[str, ...] | None:
    """Flatten a (possibly nested) ExprCompose into a tuple of
    morphism names from left to right.

    Handles four leaf shapes:

    * `ExprIdent` -> the single named morphism.
    * `ExprStack(name, n)` / `ExprRepeat(name, n)` -> n copies of
      `name` (sequential repetition; canonical Kleisli iteration).
    * `ExprFan(exprs)` -> all the named morphisms in `exprs`
      flattened in order (parallel fan-out emits each independently
      under MCMC).
    * `ExprTensorProduct(left, right)` -> the named morphism leaves
      of `left` followed by those of `right` (tensor product as
      parallel sample).

    Returns None when any leaf is not one of the above (composite
    expressions with embedded calls, scans, marginalizations, etc.
    are not expanded; the caller falls back to leaving the original
    step intact, which the walker rejects with
    `UnsupportedConstruct(let:composite_expression:...)`).
    """
    out: list[str] = []

    def walk(e: Expr) -> bool:
        if isinstance(e, ExprCompose):
            return walk(e.left) and walk(e.right)
        if isinstance(e, ExprIdent):
            out.append(e.name)
            return True
        if isinstance(e, ExprStack):
            inner = _expr_to_name(e.expr)
            if inner is None or e.count is None or e.count <= 0:
                return False
            for _ in range(e.count):
                out.append(inner)
            return True
        if isinstance(e, ExprRepeat):
            inner = _expr_to_name(e.expr)
            if inner is None or e.count is None or e.count <= 0:
                return False
            for _ in range(e.count):
                out.append(inner)
            return True
        if isinstance(e, ExprFan):
            for sub in e.exprs:
                name = _expr_to_name(sub)
                if name is None:
                    return False
                out.append(name)
            return True
        if isinstance(e, ExprTensorProduct):
            return walk(e.left) and walk(e.right)
        return False

    if not walk(expr):
        return None
    return tuple(out)


def _expr_to_name(e: Expr) -> str | None:
    """If `e` is an `ExprIdent`, return its name; otherwise None."""
    if isinstance(e, ExprIdent):
        return e.name
    return None


def _expand_step(
    step: SampleStep | ObserveStep,
    chain: tuple[str, ...],
    *,
    morphisms: dict[str, MorphismDecl],
    counter: int,
) -> tuple[list[ProgramStep], int]:
    """Convert a single sample / observe step that references a
    composite-let chain into N atomic sample steps.

    The first N-1 steps are fresh latent samples named
    `_<name>_<counter>`; the final step keeps the original step's
    bound variable name.
    """
    if len(chain) < 2:
        return [step], counter
    out: list[ProgramStep] = []
    prev_var: str | None = None
    base_name = (
        step.vars[0]
        if isinstance(step, SampleStep) and step.vars
        else step.var
        if isinstance(step, ObserveStep)
        else "tmp"
    )
    for i, morphism_name in enumerate(chain):
        is_last = i == len(chain) - 1
        if is_last:
            if isinstance(step, ObserveStep):
                terminal_var = step.var
            else:
                terminal_var = base_name
        else:
            counter += 1
            terminal_var = f"_{base_name}_chain_{counter}"
        args = _derive_chain_args(
            morphism_name=morphism_name,
            prev_var=prev_var,
            morphisms=morphisms,
        )
        if is_last and isinstance(step, ObserveStep):
            out.append(
                ObserveStep(
                    var=terminal_var,
                    morphism=morphism_name,
                    args=args,
                    index=step.index,
                    axes=step.axes,
                    via=step.via,
                    via_axes=step.via_axes,
                    options=step.options,
                    line=step.line,
                    col=step.col,
                )
            )
        else:
            sample_options = (
                step.options if (is_last and isinstance(step, SampleStep)) else ()
            )
            sample_axes = (
                step.axes if (is_last and isinstance(step, SampleStep)) else None
            )
            sample_index = (
                step.index if (is_last and isinstance(step, SampleStep)) else None
            )
            out.append(
                SampleStep(
                    vars=(terminal_var,),
                    morphism=morphism_name,
                    args=args,
                    index=sample_index,
                    axes=sample_axes,
                    options=sample_options,
                    line=step.line,
                    col=step.col,
                )
            )
        prev_var = terminal_var
    return out, counter


def _derive_chain_args(
    *,
    morphism_name: str,
    prev_var: str | None,
    morphisms: dict[str, MorphismDecl],
) -> tuple[str | float, ...]:
    """Compute the chain-position args for a kernel morphism.

    If the morphism declaration carries explicit `~ Family(args)`,
    those args are used verbatim. Otherwise the kernel is `~ Family`
    with no explicit args, and we substitute canonical defaults: the
    first arg is the upstream step's output variable name (when one
    exists) or the family's first default; remaining args are the
    family's default tail.
    """
    decl = morphisms.get(morphism_name)
    family: str | None = None
    explicit_args: tuple[str | float, ...] = ()
    if decl is not None:
        if decl.init_family is not None:
            family = decl.init_family.family
            explicit_args = decl.init_family.args
        elif isinstance(decl.init_expr, ExprIdent):
            family = decl.init_expr.name
    if explicit_args:
        return explicit_args
    if family is None:
        return ()
    defaults = _FAMILY_DEFAULT_ARGS.get(family)
    if defaults is None:
        return ()
    if prev_var is None:
        return defaults
    if not defaults:
        return ()
    return (prev_var, *defaults[1:])


__all__ = ["expand_composite_lets"]
