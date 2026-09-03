"""AST preprocessing: expand composite-let bindings into sample chains
and flatten MarginalizeStep / ScoreStep / LetStep scopes into plain
sample / observe / assignment sequences.

The pass operates at the Module level: it rebuilds every
`program_decl` in place. Three rewrites fire:

1. **Composite let**: `let chain = prior >> likelihood` binds a
   Kleisli composition. A single `sample x <- chain` step rewrites
   into a chain of atomic steps whose shape depends on each leaf's
   declaration:

   * A morphism with a family init clause becomes a `SampleStep`
     drawing from that family. A bare `~ Family` declaration reads
     its parameters off the morphism's own parameter map, so the
     step's argument list is the conditioning row that map is
     applied to: the previous chain output, or nothing at all at
     the head of the chain, where the row is the chain's own input.
     A declaration that writes its parameters (`~ Family(args)`)
     denotes a constant kernel and keeps those arguments.
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

2. **MarginalizeStep** / **ScoreStep** / **LetStep**: passthrough
   apart from expanding composite-let references inside a
   marginalize scope. A marginalize step's axis roles are not
   expressible as a `SampleStep`: its `[over=...]` names the
   grouping (batch) axes while its `: T` index names either the
   enumerated support or, for a non-enumerable latent, a
   replication axis. Lowering resolves those roles against the
   family metadata, so the step reaches each renderer intact as an
   [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]. Backends
   that sample discrete latents natively rewrite it to a sample
   plus inline scope through `RendererBase.explicit_latent_scope`,
   which reuses the lowered plate.
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
    DefineDecl,
    LetStep,
    MarginalizeStep,
    Module,
    MorphismDecl,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    SampleStep,
)
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprList,
    LetExprNode,
    LetExprVar,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._draw_args import atom_to_draw_arg


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

    A composite let is a `DefineDecl` whose `.expr` is an `ExprCompose`
    (the `prior >> likelihood` form). Each `SampleStep` /
    `ObserveStep` whose `morphism` slot names such a let is rewritten
    into a sequence of fresh `SampleStep`s, one per element of the
    composition chain, with the trailing step keeping the original
    step's bound variable name.

    The returned `Module` shares vertex identity with the input for
    every non-rewritten statement; only the `program_decl`s with
    composite-let references are rebuilt.

    The rewrite is target-independent: every backend consumes the same
    expanded AST, and the choice of whether to enumerate or to sample a
    marginalize latent is each renderer's, taken against the lowered
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]. `target` is
    accepted so callers can name the backend they are compiling for
    without the pass branching on it.
    """
    morphism_table: dict[str, MorphismDecl] = {
        name: s
        for s in module.statements
        if isinstance(s, MorphismDecl)
        for name in s.names
    }
    let_table: dict[str, Expr] = {
        s.name: s.expr for s in module.statements if isinstance(s, DefineDecl)
    }

    del target
    new_statements: list = []
    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            new_draws = _expand_draws(
                stmt.draws,
                morphisms=morphism_table,
                lets=let_table,
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
) -> tuple[ProgramStep, ...]:
    """Expand every SampleStep / ObserveStep whose morphism slot
    resolves to a composite-let chain.

    A `MarginalizeStep` passes through with its axis roles intact:
    its `[over=...]` names the grouping (batch) axes while its `: T`
    index names either the enumerated support or a replication axis,
    a distinction `SampleStep` has no slot for. Only the step's scope
    is rewritten, so composite-let references nested under a
    marginalize still expand.
    """
    any_changed = False
    out: list[ProgramStep] = []
    counter = 0
    for step in draws:
        if isinstance(step, MarginalizeStep):
            scope_expanded = _expand_draws(
                step.scope, morphisms=morphisms, lets=lets,
            )
            if scope_expanded is step.scope:
                out.append(step)
            else:
                any_changed = True
                out.append(step.with_(scope=scope_expanded))
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
        else step.vars[0] if isinstance(step, ObserveStep) and step.vars
        else "tmp"
    )
    if len(chain) == 1 and isinstance(chain[0], _StochasticLeaf):
        elem = chain[0]
        if elem.name == step.morphism:
            return [step], counter
        if isinstance(step, ObserveStep):
            return [ObserveStep(
                vars=(base_name,),
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
                vars=(terminal_var,),
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
    """Refuse an `ExprScan(cell)` leaf.

    `scan(cell)` threads `cell` across the positions of a sequence,
    so the measure it denotes is the product of one cell density per
    position, over intermediate states that are drawn rather than
    given. Writing that out needs a loop whose bound is the sequence
    length and one sample site per position, and the sequence axis is
    not an object the module declares: it arrives with the data, so
    there is no extent to size the loop from and no name to bind the
    per-position states to.

    There is no lowering that keeps the measure, and a target given a
    program that reads `cell` as a free input would score a different
    one: the recurrent density would be absent from the joint
    entirely, leaving a program whose log density does not depend on
    the parameters the chain declares. This refuses instead.
    """
    del prev_var
    raise UnsupportedConstruct(
        "qvr-expand", [f"scan:no-lowering:{cell_name}"]
    )


def _derive_chain_args(
    *,
    morphism_name: str,
    prev_var: str | None,
    morphisms: dict[str, MorphismDecl],
) -> tuple[DrawArg, ...] | None:
    """Compute the chain-position args for a kernel morphism.

    The argument list of a draw whose head names a declared morphism
    is a *conditioning row*, not a list of family parameters: the
    morphism's own parameter map computes the family's parameters
    from that row (``docs/semantics/morphisms.md`` §2.1), and the row
    is what the runtime assembles from the list before applying the
    morphism.
    A chained kernel therefore conditions on the upstream step's
    output, and an absent list conditions on the chain's own input,
    which is exactly the pair of draws
    [`SampledComposition`][quivers.continuous.morphisms.SampledComposition]
    makes: ``y ~ f(x, .)`` then ``z ~ g(y, .)``.

    A declaration that writes its own parameters (``~ Normal(0, 1)``)
    means them: it denotes a constant kernel, so the chain position
    keeps the declared arguments and reads nothing upstream.
    """
    decl = morphisms.get(morphism_name)
    if decl is not None and decl.init_family is not None:
        explicit_args = decl.init_family.args
        if explicit_args:
            return tuple(atom_to_draw_arg(a) for a in explicit_args)
    if prev_var is None:
        return None
    return (atom_to_draw_arg(prev_var),)


__all__ = ["expand_composite_lets"]
