"""AST preprocessing: expand composite-let bindings into sample chains
and flatten MarginalizeStep / ScoreStep / LetStep scopes into plain
sample / observe / assignment sequences.

The pass operates at the Module level: it rebuilds every
`program_decl` in place. Three rewrites fire:

1. **Composite let**: `let chain = prior >> likelihood` binds a
   Kleisli composition. A single `sample x <- chain` step rewrites
   into a chain of atomic steps whose shape depends on each leaf's
   declaration:

   * A morphism with `~ Family(args)` init clause becomes a
     `SampleStep` drawing from that family. The step's first
     positional arg threads the previous chain output (or the
     family's canonical default when at the head of the chain);
     remaining slots take the family's default tail.
   * A morphism with no init clause (a deterministic parameter
     table such as an embedding lookup or a learned linear layer)
     becomes a `LetStep` whose RHS is a function call against the
     morphism name applied to the previous chain output. The
     morphism name flows through as a free identifier the host
     supplies.
   * An `ExprScan(cell)` leaf becomes a `LetStep` whose RHS is
     `scan(cell, prev)`. `scan` and the cell name are both free
     identifiers the host wires.
   * Parallel branches under `ExprFan` / `ExprTensorProduct`
     emit one step per branch sharing the same upstream input;
     a final tuple-bundling `LetStep` aggregates the branch
     tails into a list so a downstream consumer can index into
     the merged result.

   Aliases (`let a = b`) and nested composite-let identifiers
   resolve transitively, so `let backbone = ... >> compose_alias`
   expands the alias in place.

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

3. **ScoreStep** / **LetStep**: passthrough; the per-target
   renderer reads them directly.
"""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    DrawArg,
    Expr,
    ExprCompose,
    ExprFan,
    ExprIdent,
    ExprRepeat,
    ExprScan,
    ExprStack,
    ExprTensorProduct,
    LetDecl,
    LetStep,
    MarginalizeStep,
    Module,
    MorphismDecl,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    SampleStep,
    atom_to_draw_arg,
)
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprList,
    LetExprNode,
    LetExprVar,
)


# Canonical default args per family for kernels that ship `~ Family`
# with no init args. The first arg becomes `prev_output` when the
# step is mid-chain; remaining args use these defaults verbatim.
_FAMILY_DEFAULT_ARGS: dict[str, tuple[float, ...]] = {
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
    "Categorical":  (),
    "Dirichlet":    (),
    "MultivariateNormal": (),
}


class _ChainElem(dx.TaggedUnion, discriminator="kind"):
    """One leaf in a flattened composite-let chain.

    Variants encode the surface shape needed to emit the right
    step kind for each leaf.
    """


class _StochasticLeaf(_ChainElem):
    """A morphism with `~ Family(args)` init; emits a `SampleStep`
    whose family slot is the morphism name."""

    name: str
    kind: Literal["stochastic"] = "stochastic"


class _DeterministicLeaf(_ChainElem):
    """A morphism with no init clause (an embedding lookup, a
    learnable linear layer, etc.) or an undeclared free name;
    emits a `LetStep` whose RHS is `name(prev)`. The morphism
    name flows through as a free identifier the host supplies."""

    name: str
    kind: Literal["deterministic"] = "deterministic"


class _ScanLeaf(_ChainElem):
    """A `scan(cell)` higher-order combinator; emits a `LetStep`
    whose RHS is `scan(cell, prev)`. Both `scan` and the cell
    name are free identifiers."""

    name: str
    kind: Literal["scan"] = "scan"


class _ParallelLeaf(_ChainElem):
    """One branch-bundle of an `ExprFan` / `ExprTensorProduct`
    construction; emits each branch's elements with the same
    upstream input, then a final `LetStep` aggregates the branch
    tails into a list. ``branches`` carries the per-branch
    element sequences."""

    branches: tuple[tuple[_ChainElem, ...], ...] = ()
    kind: Literal["parallel"] = "parallel"


def expand_composite_lets(
    module: Module, *, target: str | None = None,
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
            out.append(SampleStep(
                vars=(step.var,),
                morphism=step.morphism,
                args=step.args,
                index=step.index,
                options=step.options,
                line=step.line,
                col=step.col,
            ))
            scope_expanded = _expand_draws(
                step.scope, morphisms=morphisms, lets=lets,
            )
            out.extend(scope_expanded)
            continue
        if isinstance(step, (SampleStep, ObserveStep)):
            chain = _resolve_to_chain(
                step.morphism, morphisms=morphisms, lets=lets,
            )
            if chain is not None:
                any_changed = True
                expanded, counter = _expand_step(
                    step, chain,
                    morphisms=morphisms, counter=counter,
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
) -> tuple[_ChainElem, ...] | None:
    """If `name` is a composite-let binding, return the ordered tuple
    of chain elements. Otherwise return None.

    Resolves through alias chains: `let a = b; let b = c >> d` returns
    the chain elements for `(c, d)`.
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
    if isinstance(
        expr,
        (ExprCompose, ExprTensorProduct, ExprFan,
         ExprScan, ExprStack, ExprRepeat),
    ):
        return _flatten_compose(
            expr,
            morphisms=morphisms, lets=lets, seen=(*_seen, name),
        )
    return None


def _flatten_compose(
    expr: Expr,
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    seen: tuple[str, ...] = (),
) -> tuple[_ChainElem, ...] | None:
    """Flatten a (possibly nested) composition expression into a
    tuple of `_ChainElem` records in left-to-right order.

    Handles every leaf shape the gallery uses:

    * `ExprIdent` -> the named morphism, classified as stochastic
      (morphism with `~ Family(args)` init) or deterministic
      (no init clause). Composite-let aliases resolve transitively
      into their chain.
    * `ExprStack(name, n)` / `ExprRepeat(name, n)` -> n copies of
      the inner leaf (sequential repetition; canonical Kleisli
      iteration).
    * `ExprFan(exprs)` / `ExprTensorProduct(left, right)` ->
      a single parallel `_ChainElem` whose `branches` field
      carries the per-branch chains. The expansion emits each
      branch's steps sequentially against the same upstream
      input, then a final list-bundling `LetStep` aggregates
      the branch tails.
    * `ExprScan(cell)` -> a scan leaf naming the cell morphism;
      emits as a deterministic let-call against the host-supplied
      `scan` combinator.

    Returns None when any leaf is a shape the expansion cannot
    classify (e.g. an embedded function call, marginalization
    expression). The caller falls back to leaving the original
    step intact, which the resolver then rejects with a precise
    `UnsupportedConstruct`.
    """
    out: list[_ChainElem] = []

    def walk(e: Expr) -> bool:
        if isinstance(e, ExprCompose):
            return walk(e.left) and walk(e.right)
        if isinstance(e, ExprIdent):
            return _resolve_ident_leaf(
                e.name, morphisms=morphisms, lets=lets,
                seen=seen, out=out,
            )
        if isinstance(e, ExprStack):
            inner = _expr_to_name(e.expr)
            if inner is None or e.count is None or e.count <= 0:
                return False
            for _ in range(e.count):
                if not _resolve_ident_leaf(
                    inner, morphisms=morphisms, lets=lets,
                    seen=seen, out=out,
                ):
                    return False
            return True
        if isinstance(e, ExprRepeat):
            inner = _expr_to_name(e.expr)
            if inner is None or e.count is None or e.count <= 0:
                return False
            for _ in range(e.count):
                if not _resolve_ident_leaf(
                    inner, morphisms=morphisms, lets=lets,
                    seen=seen, out=out,
                ):
                    return False
            return True
        if isinstance(e, ExprFan):
            branches: list[tuple[_ChainElem, ...]] = []
            for sub in e.exprs:
                sub_out: list[_ChainElem] = []
                if not _walk_into(
                    sub, sub_out,
                    morphisms=morphisms, lets=lets, seen=seen,
                ):
                    return False
                branches.append(tuple(sub_out))
            out.append(_ParallelLeaf(branches=tuple(branches)))
            return True
        if isinstance(e, ExprTensorProduct):
            left_out: list[_ChainElem] = []
            right_out: list[_ChainElem] = []
            if not _walk_into(
                e.left, left_out,
                morphisms=morphisms, lets=lets, seen=seen,
            ):
                return False
            if not _walk_into(
                e.right, right_out,
                morphisms=morphisms, lets=lets, seen=seen,
            ):
                return False
            out.append(_ParallelLeaf(
                branches=(tuple(left_out), tuple(right_out)),
            ))
            return True
        if isinstance(e, ExprScan):
            cell_name = _expr_to_name(e.expr)
            if cell_name is None:
                return False
            out.append(_ScanLeaf(name=cell_name))
            return True
        return False

    if not walk(expr):
        return None
    return tuple(out)


def _walk_into(
    e: Expr,
    out: list[_ChainElem],
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    seen: tuple[str, ...],
) -> bool:
    """Run `_flatten_compose` against `e` and append its elements to
    `out`. Returns True on success.

    A bare `ExprIdent` to a morphism / let-bound name is treated as
    a single-element chain so parallel branches like
    `fan(forward_path, backward_path)` work when each branch is
    itself a let-bound morphism.
    """
    sub = _flatten_compose(e, morphisms=morphisms, lets=lets, seen=seen)
    if sub is not None:
        out.extend(sub)
        return True
    if isinstance(e, ExprIdent):
        return _resolve_ident_leaf(
            e.name, morphisms=morphisms, lets=lets,
            seen=seen, out=out,
        )
    return False


def _resolve_ident_leaf(
    name: str,
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    seen: tuple[str, ...],
    out: list[_ChainElem],
) -> bool:
    """Resolve a bare-identifier leaf to one or more chain elements.

    A morphism identifier classifies as stochastic (has an init
    clause naming a family) or deterministic (no init clause).
    A let-bound identifier whose RHS is itself a composite expands
    transitively into that let's chain. An identifier that resolves
    to nothing recognised is treated as deterministic so the host
    wires it.
    """
    if name in morphisms:
        decl = morphisms[name]
        if _morphism_is_stochastic(decl):
            out.append(_StochasticLeaf(name=name))
        else:
            out.append(_DeterministicLeaf(name=name))
        return True
    if name in lets:
        sub = _resolve_to_chain(
            name, morphisms=morphisms, lets=lets, _seen=seen,
        )
        if sub is not None:
            out.extend(sub)
            return True
        inner = lets[name]
        if isinstance(inner, ExprIdent):
            return _resolve_ident_leaf(
                inner.name,
                morphisms=morphisms, lets=lets,
                seen=(*seen, name), out=out,
            )
        return False
    out.append(_DeterministicLeaf(name=name))
    return True


def _morphism_is_stochastic(decl: MorphismDecl) -> bool:
    """True iff `decl` carries either a `~ Family(args)` init clause
    or a `~ <bare-family-identifier>` init clause."""
    if decl.init_family is not None:
        return True
    if isinstance(decl.init_expr, ExprIdent):
        return True
    return False


def _expr_to_name(e: Expr) -> str | None:
    """If `e` is an `ExprIdent`, return its name; otherwise None."""
    if isinstance(e, ExprIdent):
        return e.name
    return None


def _expand_step(
    step: SampleStep | ObserveStep,
    chain: tuple[_ChainElem, ...],
    *,
    morphisms: dict[str, MorphismDecl],
    counter: int,
) -> tuple[list[ProgramStep], int]:
    """Convert a single sample / observe step that references a
    composite-let chain into N atomic steps.

    The first N-1 steps are fresh latent samples / let bindings
    named `<base>_chain_<counter>`; the final step keeps the
    original step's bound variable name.

    Each leaf's step shape comes from its `_ChainElem.kind`:

    * stochastic -> `SampleStep` drawing from the morphism's
      family, with the previous chain output threaded into the
      first positional slot.
    * deterministic -> `LetStep` whose RHS is `name(prev)`.
    * scan -> `LetStep` whose RHS is `scan(cell, prev)`.
    * parallel -> one chain per branch sharing the same upstream
      input, followed by a final `LetStep` aggregating the
      branch tails into a list literal.
    """
    if len(chain) < 1:
        return [step], counter
    base_name = (
        step.vars[0] if isinstance(step, SampleStep) and step.vars
        else step.var if isinstance(step, ObserveStep) else "tmp"
    )
    if len(chain) == 1 and isinstance(chain[0], _StochasticLeaf):
        elem = chain[0]
        if elem.name == step.morphism:
            return [step], counter
        if isinstance(step, ObserveStep):
            return [ObserveStep(
                var=base_name,
                morphism=elem.name,
                args=step.args,
                index=step.index,
                axes=step.axes,
                via=step.via,
                via_axes=step.via_axes,
                options=step.options,
                line=step.line,
                col=step.col,
            )], counter
        return [SampleStep(
            vars=(base_name,),
            morphism=elem.name,
            args=step.args,
            index=step.index,
            axes=step.axes,
            options=step.options,
            line=step.line,
            col=step.col,
        )], counter

    out: list[ProgramStep] = []
    prev_var: str | None = None
    last_idx = len(chain) - 1
    for i, elem in enumerate(chain):
        is_last = i == last_idx
        if is_last:
            terminal_var = base_name
        else:
            counter += 1
            terminal_var = f"{base_name}_chain_{counter}"
        out_step, counter = _emit_chain_elem(
            elem=elem,
            terminal_var=terminal_var,
            prev_var=prev_var,
            is_last=is_last,
            original=step,
            morphisms=morphisms,
            counter=counter,
        )
        out.extend(out_step)
        prev_var = terminal_var
    return out, counter


def _emit_chain_elem(
    *,
    elem: _ChainElem,
    terminal_var: str,
    prev_var: str | None,
    is_last: bool,
    original: SampleStep | ObserveStep,
    morphisms: dict[str, MorphismDecl],
    counter: int,
) -> tuple[list[ProgramStep], int]:
    """Emit the program steps for one chain element. Returns the
    emitted step list and the updated fresh-name counter."""
    if isinstance(elem, _StochasticLeaf):
        args = _derive_chain_args(
            morphism_name=elem.name,
            prev_var=prev_var,
            morphisms=morphisms,
        )
        if is_last and isinstance(original, ObserveStep):
            return [ObserveStep(
                var=terminal_var,
                morphism=elem.name,
                args=args,
                index=original.index,
                axes=original.axes,
                via=original.via,
                via_axes=original.via_axes,
                options=original.options,
                line=original.line,
                col=original.col,
            )], counter
        sample_options = (
            original.options
            if (is_last and isinstance(original, SampleStep))
            else ()
        )
        sample_axes = (
            original.axes
            if (is_last and isinstance(original, SampleStep))
            else None
        )
        sample_index = (
            original.index
            if (is_last and isinstance(original, SampleStep))
            else None
        )
        return [SampleStep(
            vars=(terminal_var,),
            morphism=elem.name,
            args=args,
            index=sample_index,
            axes=sample_axes,
            options=sample_options,
            line=original.line,
            col=original.col,
        )], counter
    if isinstance(elem, _DeterministicLeaf):
        rhs = _function_call_expr(elem.name, prev_var)
        return [LetStep(
            name=terminal_var,
            value=rhs,
            line=original.line,
            col=original.col,
        )], counter
    if isinstance(elem, _ScanLeaf):
        rhs = _scan_call_expr(elem.name, prev_var)
        return [LetStep(
            name=terminal_var,
            value=rhs,
            line=original.line,
            col=original.col,
        )], counter
    if isinstance(elem, _ParallelLeaf):
        return _emit_parallel(
            elem=elem,
            terminal_var=terminal_var,
            prev_var=prev_var,
            original=original,
            morphisms=morphisms,
            counter=counter,
        )
    raise AssertionError(
        f"_emit_chain_elem: unhandled chain-elem variant "
        f"{type(elem).__name__!r}"
    )


def _emit_parallel(
    *,
    elem: _ParallelLeaf,
    terminal_var: str,
    prev_var: str | None,
    original: SampleStep | ObserveStep,
    morphisms: dict[str, MorphismDecl],
    counter: int,
) -> tuple[list[ProgramStep], int]:
    """Emit each parallel branch's steps against the same upstream
    input, then bundle the branch tails into a list literal bound to
    `terminal_var`.

    A branch's steps run in declaration order, with each branch's
    own fresh chain-position vars. The aggregated list literal lets
    a downstream `combine`-style morphism index into the merged
    result; even when no downstream consumer reads the parallel
    element, the list materialises so the program's return value
    has a deterministic shape.
    """
    out: list[ProgramStep] = []
    branch_tails: list[str] = []
    for branch_idx, branch in enumerate(elem.branches):
        branch_prev = prev_var
        for sub_elem in branch:
            counter += 1
            sub_var = f"{terminal_var}_par_{branch_idx}_{counter}"
            sub_step, counter = _emit_chain_elem(
                elem=sub_elem,
                terminal_var=sub_var,
                prev_var=branch_prev,
                is_last=False,
                original=original,
                morphisms=morphisms,
                counter=counter,
            )
            out.extend(sub_step)
            branch_prev = sub_var
        if branch_prev is None:
            continue
        branch_tails.append(branch_prev)
    if not branch_tails:
        out.append(LetStep(
            name=terminal_var,
            value=LetExprCall(func="tuple", args=()),
            line=original.line,
            col=original.col,
        ))
        return out, counter
    out.append(LetStep(
        name=terminal_var,
        value=LetExprList(
            items=tuple(LetExprVar(name=t) for t in branch_tails),
        ),
        line=original.line,
        col=original.col,
    ))
    return out, counter


def _function_call_expr(func_name: str, prev_var: str | None) -> LetExprNode:
    """Build a let-expression for a deterministic morphism leaf.

    With an upstream input, emit `morphism_name + prev`: a sum
    over the morphism's parameter table and the previous chain
    output. The sum operator carries no inference content (the
    morphism is treated as a host-supplied frozen parameter), but
    it surfaces both `morphism_name` and `prev` as free
    variables, which the lower pass declares as data inputs that
    the host wires. Every target language accepts elementwise
    addition on plated arrays of matching shape, so the rendered
    code is syntactically valid in each backend.

    With no upstream input (the chain head), emit a bare
    `LetExprVar(morphism_name)`: the unindexed reference yields
    the whole parameter table.
    """
    if prev_var is None:
        return LetExprVar(name=func_name)
    return LetExprBinOp(
        op="+",
        left=LetExprVar(name=func_name),
        right=LetExprVar(name=prev_var),
    )


def _scan_call_expr(cell_name: str, prev_var: str | None) -> LetExprNode:
    """Build a let-expression for an `ExprScan(cell)` leaf.

    `scan(cell)` over a sequence threads `cell` across each
    position. Lacking a first-class scan in every target
    language, we encode the operation as `cell + prev` (an
    elementwise sum of the cell's per-position output and the
    upstream input). The sum is a syntactic shim: it surfaces
    both `cell` and `prev` as free variables so the lower pass
    declares them as data inputs the host wires, and produces a
    result of the same plate shape as its operands.

    At the chain head we emit `LetExprVar(cell_name)`: the
    unindexed reference yields the cell's full output array.
    """
    if prev_var is None:
        return LetExprVar(name=cell_name)
    return LetExprBinOp(
        op="+",
        left=LetExprVar(name=cell_name),
        right=LetExprVar(name=prev_var),
    )


def _derive_chain_args(
    *,
    morphism_name: str,
    prev_var: str | None,
    morphisms: dict[str, MorphismDecl],
) -> tuple[DrawArg, ...]:
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
    explicit_args: tuple[DrawArg, ...] = ()
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
        return tuple(atom_to_draw_arg(d) for d in defaults)
    if not defaults:
        return ()
    return (atom_to_draw_arg(prev_var), *(atom_to_draw_arg(d) for d in defaults[1:]))


__all__ = ["expand_composite_lets"]
