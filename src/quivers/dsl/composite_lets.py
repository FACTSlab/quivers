"""Normalize program ASTs before IR lowering.

The pass expands sampled composite bindings into atomic sample and let
steps. It preserves marginalize blocks for the lowering stage while
expanding composite references inside their scopes.
"""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgName,
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
    ObjectExpr,
    ObjectProduct,
    ObserveStep,
    OptionName,
    OptionNumber,
    ProgramDecl,
    TypeName,
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
from quivers.dsl.step_resolution import StepResolutionError
from quivers.dsl.draw_args import atom_to_draw_arg


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
    product: bool = False
    kind: Literal["parallel"] = "parallel"


def expand_composite_lets(
    module: Module,
    *,
    target: str | None = None,
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
    declared = set(morphism_table)
    new_statements: list = []
    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            new_draws = _expand_draws(
                stmt.draws,
                morphisms=morphism_table,
                lets=let_table,
                domain_names=_domain_names(stmt),
            )
            if new_draws is stmt.draws:
                new_statements.append(stmt)
            else:
                new_statements.append(stmt.with_(draws=tuple(new_draws)))
        else:
            new_statements.append(stmt)
    # A replica of a `[replicate=k]` morphism and a copy a `stack`
    # makes are morphisms of their own, with their own parameters;
    # the expanded module declares them so every consumer sees them.
    synthesized = [
        decl for name, decl in morphism_table.items() if name not in declared
    ]
    if synthesized:
        last = max(
            index
            for index, stmt in enumerate(new_statements)
            if isinstance(stmt, MorphismDecl)
        )
        new_statements[last + 1 : last + 1] = synthesized
    return module.with_(statements=tuple(new_statements))


def _domain_names(program: ProgramDecl) -> tuple[str, ...]:
    """The names a program's domain factors are read through.

    Parameters
    ----------
    program : ProgramDecl
        The program.

    Returns
    -------
    tuple[str, ...]
        The declared parameter names when the program names them,
        else each factor's object name in lowercase.
    """
    if program.params:
        return tuple(program.params)
    return tuple(
        factor.name.lower() if isinstance(factor, TypeName) else ""
        for factor in _domain_factors(program.domain)
    )


def _domain_factors(domain: ObjectExpr) -> tuple[ObjectExpr, ...]:
    """A domain's product factors, left to right.

    Parameters
    ----------
    domain : ObjectExpr
        The domain expression.

    Returns
    -------
    tuple[ObjectExpr, ...]
        The factors of a product, or the expression itself.
    """
    if isinstance(domain, ObjectProduct):
        return tuple(
            factor
            for component in domain.components
            for factor in _domain_factors(component)
        )
    return (domain,)


def _replicas(name: str, morphisms: dict[str, MorphismDecl]) -> tuple[str, ...] | None:
    """The replica names of a ``[replicate=k]`` morphism.

    Each replica is declared in ``morphisms`` as a single-name copy of
    the declaration without the ``replicate`` option.

    Parameters
    ----------
    name : str
        The morphism.
    morphisms : dict[str, MorphismDecl]
        The morphism table, extended with the replicas.

    Returns
    -------
    tuple[str, ...] | None
        ``name_0`` through ``name_{k-1}``, or ``None`` when the
        morphism is not replicated.
    """
    decl = morphisms.get(name)
    if decl is None:
        return None
    count = next(
        (
            int(entry.value.value)
            for entry in decl.options
            if entry.key == "replicate" and isinstance(entry.value, OptionNumber)
        ),
        None,
    )
    if count is None:
        return None
    names = tuple(f"{name}_{index}" for index in range(count))
    for member in names:
        if member not in morphisms:
            morphisms[member] = decl.with_(
                names=(member,),
                options=tuple(
                    entry for entry in decl.options if entry.key != "replicate"
                ),
            )
    return names


def _copied(
    chain: tuple[_ChainElem, ...],
    suffix: str,
    morphisms: dict[str, MorphismDecl],
) -> tuple[_ChainElem, ...]:
    """A chain with every declared morphism replaced by a fresh copy.

    A ``stack`` composes copies of its morphism that share nothing: the
    copy's parameters are its own, so each copy is a morphism of its
    own, declared in ``morphisms`` under the suffixed name.

    Parameters
    ----------
    chain : tuple[_ChainElem, ...]
        The chain to copy.
    suffix : str
        The suffix the copies' names carry.
    morphisms : dict[str, MorphismDecl]
        The morphism table, extended with the copies.

    Returns
    -------
    tuple[_ChainElem, ...]
        The chain over the copies; names that are not declared
        morphisms are left as they are.
    """
    out: list[_ChainElem] = []
    for elem in chain:
        if isinstance(elem, _StochasticLeaf | _DeterministicLeaf):
            decl = morphisms.get(elem.name)
            if decl is None:
                out.append(elem)
                continue
            copy_name = f"{elem.name}{suffix}"
            if copy_name not in morphisms:
                morphisms[copy_name] = decl.with_(names=(copy_name,))
            out.append(elem.with_(name=copy_name))
        elif isinstance(elem, _ParallelLeaf):
            out.append(
                elem.with_(
                    branches=tuple(
                        _copied(branch, suffix, morphisms) for branch in elem.branches
                    )
                )
            )
        else:
            out.append(elem)
    return tuple(out)


def _expand_draws(
    draws: tuple[ProgramStep, ...],
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    domain_names: tuple[str, ...],
) -> tuple[ProgramStep, ...]:
    """Expand every SampleStep / ObserveStep whose morphism slot
    resolves to a composite-let chain.

    A `MarginalizeStep` passes through with its axis roles intact:
    its `[over=...]` names the grouping (batch) axes while its `: T`
    index names either the enumerated support or a replication axis,
    a distinction `SampleStep` has no slot for. Only the step's scope
    is rewritten, so composite-let references nested under a
    marginalize still expand.

    Parameters
    ----------
    draws
        The steps.
    morphisms
        The morphism table.
    lets
        The define table.
    domain_names
        The names the program's domain factors are read through, which
        a tensor product at the head of a chain conditions its branches
        on.

    Returns
    -------
    tuple[ProgramStep, ...]
        The steps with every composite draw expanded; the input itself
        when nothing expanded.
    """
    any_changed = False
    out: list[ProgramStep] = []
    counter = 0
    for step in draws:
        if isinstance(step, MarginalizeStep):
            scope_expanded = _expand_draws(
                step.scope,
                morphisms=morphisms,
                lets=lets,
                domain_names=domain_names,
            )
            if scope_expanded is step.scope:
                out.append(step)
            else:
                any_changed = True
                out.append(step.with_(scope=scope_expanded))
            continue
        if isinstance(step, (SampleStep, ObserveStep)):
            chain = _resolve_to_chain(
                step.morphism,
                morphisms=morphisms,
                lets=lets,
            )
            if chain is not None:
                any_changed = True
                expanded, counter = _expand_step(
                    step,
                    chain,
                    morphisms=morphisms,
                    counter=counter,
                    domain_names=domain_names,
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
        (ExprCompose, ExprTensorProduct, ExprFan, ExprScan, ExprStack, ExprRepeat),
    ):
        return _flatten_compose(
            expr,
            morphisms=morphisms,
            lets=lets,
            seen=(*_seen, name),
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
                e.name,
                morphisms=morphisms,
                lets=lets,
                seen=seen,
                out=out,
            )
        if isinstance(e, ExprStack):
            inner = _expr_to_name(e.expr)
            if inner is None or e.count is None or e.count <= 0:
                return False
            first: list[_ChainElem] = []
            if not _resolve_ident_leaf(
                inner,
                morphisms=morphisms,
                lets=lets,
                seen=seen,
                out=first,
            ):
                return False
            out.extend(first)
            for index in range(1, e.count):
                out.extend(_copied(tuple(first), f"_copy{index}", morphisms))
            return True
        if isinstance(e, ExprRepeat):
            inner = _expr_to_name(e.expr)
            if inner is None or e.count is None or e.count <= 0:
                return False
            for _ in range(e.count):
                if not _resolve_ident_leaf(
                    inner,
                    morphisms=morphisms,
                    lets=lets,
                    seen=seen,
                    out=out,
                ):
                    return False
            return True
        if isinstance(e, ExprFan):
            branches: list[tuple[_ChainElem, ...]] = []
            for sub in e.exprs:
                replicas = (
                    _replicas(sub.name, morphisms)
                    if isinstance(sub, ExprIdent)
                    else None
                )
                if replicas is not None:
                    for member in replicas:
                        member_out: list[_ChainElem] = []
                        _resolve_ident_leaf(
                            member,
                            morphisms=morphisms,
                            lets=lets,
                            seen=seen,
                            out=member_out,
                        )
                        branches.append(tuple(member_out))
                    continue
                sub_out: list[_ChainElem] = []
                if not _walk_into(
                    sub,
                    sub_out,
                    morphisms=morphisms,
                    lets=lets,
                    seen=seen,
                ):
                    return False
                branches.append(tuple(sub_out))
            out.append(_ParallelLeaf(branches=tuple(branches)))
            return True
        if isinstance(e, ExprTensorProduct):
            left_out: list[_ChainElem] = []
            right_out: list[_ChainElem] = []
            if not _walk_into(
                e.left,
                left_out,
                morphisms=morphisms,
                lets=lets,
                seen=seen,
            ):
                return False
            if not _walk_into(
                e.right,
                right_out,
                morphisms=morphisms,
                lets=lets,
                seen=seen,
            ):
                return False
            out.append(
                _ParallelLeaf(
                    branches=(tuple(left_out), tuple(right_out)),
                    product=True,
                )
            )
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
            e.name,
            morphisms=morphisms,
            lets=lets,
            seen=seen,
            out=out,
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
            name,
            morphisms=morphisms,
            lets=lets,
            _seen=seen,
        )
        if sub is not None:
            out.extend(sub)
            return True
        inner = lets[name]
        if isinstance(inner, ExprIdent):
            return _resolve_ident_leaf(
                inner.name,
                morphisms=morphisms,
                lets=lets,
                seen=(*seen, name),
                out=out,
            )
        return False
    out.append(_DeterministicLeaf(name=name))
    return True


def _morphism_is_stochastic(decl: MorphismDecl) -> bool:
    """True iff `decl` carries either a `~ Family(args)` init clause
    or a `~ <bare-family-identifier>` init clause, or is an embedding,
    which places a Gaussian kernel at each element's centre."""
    if decl.init_family is not None:
        return True
    if isinstance(decl.init_expr, ExprIdent):
        return True
    return any(
        entry.key == "role"
        and isinstance(entry.value, OptionName)
        and entry.value.value == "embed"
        for entry in decl.options
    )


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
    domain_names: tuple[str, ...],
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
      input, or, for a tensor product, each branch on its own
      factor of the input, followed by a final `LetStep`
      aggregating the branch tails into a list literal.

    Parameters
    ----------
    step
        The step.
    chain
        The chain its morphism slot resolves to.
    morphisms
        The morphism table.
    counter
        The fresh-name counter.
    domain_names
        The names the program's domain factors are read through.

    Returns
    -------
    tuple[list[ProgramStep], int]
        The atomic steps and the updated counter.
    """
    if len(chain) < 1:
        return [step], counter
    base_name = (
        step.vars[0]
        if isinstance(step, SampleStep) and step.vars
        else step.vars[0]
        if isinstance(step, ObserveStep) and step.vars
        else "tmp"
    )
    if len(chain) == 1 and isinstance(chain[0], _StochasticLeaf):
        elem = chain[0]
        if elem.name == step.morphism:
            return [step], counter
        if isinstance(step, ObserveStep):
            return [
                ObserveStep(
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
                )
            ], counter
        return [
            SampleStep(
                vars=(base_name,),
                morphism=elem.name,
                args=step.args,
                index=step.index,
                axes=step.axes,
                options=step.options,
                line=step.line,
                col=step.col,
            )
        ], counter

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
            head_args=step.args,
            domain_names=domain_names,
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
    head_args: tuple[DrawArg, ...] | None,
    domain_names: tuple[str, ...],
) -> tuple[list[ProgramStep], int]:
    """Emit the program steps for one chain element.

    Parameters
    ----------
    elem
        The element.
    terminal_var
        The name its output binds.
    prev_var
        The upstream chain output, or ``None`` at the head.
    is_last
        Whether the element is the chain's last.
    original
        The step being expanded.
    morphisms
        The morphism table.
    counter
        The fresh-name counter.
    head_args
        The row the chain's head conditions on: the original step's
        arguments, or a branch's factor of them.
    domain_names
        The names the program's domain factors are read through.

    Returns
    -------
    tuple[list[ProgramStep], int]
        The emitted steps and the updated counter.

    Raises
    ------
    AssertionError
        If the element is of no known variant.
    """
    if isinstance(elem, _StochasticLeaf):
        args = _derive_chain_args(
            morphism_name=elem.name,
            prev_var=prev_var,
            head_args=head_args,
            morphisms=morphisms,
        )
        if is_last and isinstance(original, ObserveStep):
            return [
                ObserveStep(
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
                )
            ], counter
        sample_options = (
            original.options if (is_last and isinstance(original, SampleStep)) else ()
        )
        sample_axes = (
            original.axes if (is_last and isinstance(original, SampleStep)) else None
        )
        sample_index = (
            original.index if (is_last and isinstance(original, SampleStep)) else None
        )
        return [
            SampleStep(
                vars=(terminal_var,),
                morphism=elem.name,
                args=args,
                index=sample_index,
                axes=sample_axes,
                options=sample_options,
                line=original.line,
                col=original.col,
            )
        ], counter
    if isinstance(elem, _DeterministicLeaf):
        rhs = _function_call_expr(elem.name, prev_var)
        return [
            LetStep(
                name=terminal_var,
                value=rhs,
                line=original.line,
                col=original.col,
            )
        ], counter
    if isinstance(elem, _ScanLeaf):
        rhs = _scan_call_expr(elem.name, prev_var)
        return [
            LetStep(
                name=terminal_var,
                value=rhs,
                line=original.line,
                col=original.col,
            )
        ], counter
    if isinstance(elem, _ParallelLeaf):
        return _emit_parallel(
            elem=elem,
            terminal_var=terminal_var,
            prev_var=prev_var,
            original=original,
            morphisms=morphisms,
            counter=counter,
            head_args=head_args,
            domain_names=domain_names,
        )
    raise AssertionError(
        f"_emit_chain_elem: unhandled chain-elem variant {type(elem).__name__!r}"
    )


def _emit_parallel(
    *,
    elem: _ParallelLeaf,
    terminal_var: str,
    prev_var: str | None,
    original: SampleStep | ObserveStep,
    morphisms: dict[str, MorphismDecl],
    counter: int,
    head_args: tuple[DrawArg, ...] | None,
    domain_names: tuple[str, ...],
) -> tuple[list[ProgramStep], int]:
    """Emit each parallel branch's steps against the same upstream
    input, then bundle the branch tails into a list literal bound to
    `terminal_var`.

    A fan's branches all read the upstream input; a tensor product's
    each read their own factor of it, which at the head of a chain is
    the original step's argument at their position or, absent
    arguments, the program's domain factor at their position.

    A branch's steps run in declaration order, with each branch's
    own fresh chain-position vars. The aggregated list literal lets
    a downstream `combine`-style morphism index into the merged
    result; even when no downstream consumer reads the parallel
    element, the list materialises so the program's return value
    has a deterministic shape.

    Parameters
    ----------
    elem
        The parallel element.
    terminal_var
        The name the bundle binds.
    prev_var
        The upstream chain output, or ``None`` at the head.
    original
        The step being expanded.
    morphisms
        The morphism table.
    counter
        The fresh-name counter.
    head_args
        The row the chain's head conditions on.
    domain_names
        The names the program's domain factors are read through.

    Returns
    -------
    tuple[list[ProgramStep], int]
        The emitted steps and the updated counter.
    """
    out: list[ProgramStep] = []
    branch_tails: list[str] = []
    for branch_idx, branch in enumerate(elem.branches):
        branch_prev = prev_var
        branch_head = head_args
        if elem.product and prev_var is None:
            if head_args and len(head_args) == len(elem.branches):
                branch_head = (head_args[branch_idx],)
            elif (
                not head_args
                and len(domain_names) == len(elem.branches)
                and domain_names[branch_idx]
            ):
                branch_head = (DrawArgName(text=domain_names[branch_idx]),)
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
                head_args=branch_head,
                domain_names=domain_names,
            )
            out.extend(sub_step)
            branch_prev = sub_var
        if branch_prev is None:
            continue
        branch_tails.append(branch_prev)
    if not branch_tails:
        out.append(
            LetStep(
                name=terminal_var,
                value=LetExprCall(func="tuple", args=()),
                line=original.line,
                col=original.col,
            )
        )
        return out, counter
    out.append(
        LetStep(
            name=terminal_var,
            value=LetExprList(
                items=tuple(LetExprVar(name=t) for t in branch_tails),
            ),
            line=original.line,
            col=original.col,
        )
    )
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
    raise StepResolutionError("qvr-expand", [f"scan:no-lowering:{cell_name}"])


def _derive_chain_args(
    *,
    morphism_name: str,
    prev_var: str | None,
    head_args: tuple[DrawArg, ...] | None,
    morphisms: dict[str, MorphismDecl],
) -> tuple[DrawArg, ...] | None:
    """Compute the chain-position args for a kernel morphism.

    The argument list of a draw whose head names a declared morphism
    is a *conditioning row*, not a list of family parameters: the
    morphism's own parameter map computes the family's parameters
    from that row (``docs/semantics/morphisms.md`` §2.1), and the row
    is what the runtime assembles from the list before applying the
    morphism.
    A chained kernel thus conditions on the upstream step's
    output; the chain's head conditions on the row the original step
    wrote, and an absent list conditions on the chain's own input,
    which is exactly the pair of draws
    [`SampledComposition`][quivers.continuous.morphisms.SampledComposition]
    makes: ``y ~ f(x, .)`` then ``z ~ g(y, .)``.

    A declaration that writes its own parameters (``~ Normal(0, 1)``)
    means them: it denotes a constant kernel, so the chain position
    keeps the declared arguments and reads nothing upstream.

    Parameters
    ----------
    morphism_name
        The kernel at this chain position.
    prev_var
        The upstream chain output, or ``None`` at the head.
    head_args
        The original step's argument list, which the head conditions on.
    morphisms
        The module's morphism table.

    Returns
    -------
    tuple[DrawArg, ...] | None
        The position's argument list, or ``None`` when the position
        conditions on the chain's own input.
    """
    decl = morphisms.get(morphism_name)
    if decl is not None and decl.init_family is not None:
        explicit_args = decl.init_family.args
        if explicit_args:
            return tuple(atom_to_draw_arg(a) for a in explicit_args)
    if prev_var is None:
        return head_args if head_args else None
    return (atom_to_draw_arg(prev_var),)


__all__ = ["expand_composite_lets"]
