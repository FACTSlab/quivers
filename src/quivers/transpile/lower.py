"""`Lower`: structural mapping from a QVR program to the transpile IR.

`Lower` is a [`didactic.api.Mapping`][didactic.api.Mapping] from a
parsed [`Module`][quivers.dsl.ast_nodes.Module] to an
[`IRProgram`][quivers.transpile.ir.IRProgram]. It is target-
independent: no backend is imported here. Every renderer downstream
consumes the same IR and the same
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] registry.

The forward pass:

1. Runs the existing
   [`expand_composite_lets`][quivers.transpile._expand_composites.expand_composite_lets]
   preprocessor.
2. Builds the morphism / let / object-cardinality tables.
3. Picks the active
   [`ProgramDecl`][quivers.dsl.ast_nodes.declarations.ProgramDecl]
   (the export target if any, else the last one).
4. For each program step resolves the morphism slot via
   [`resolve_step_dist`][quivers.transpile.backends._resolve.resolve_step_dist],
   looks up the family in
   [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META], builds
   a sentinel parameter set, reads `arg_constraints` and `support`,
   matches the user's args against the constraints, and constructs
   the appropriate IR node.
5. Discovers exogenous identifiers (free names in let / score
   bodies and bracket-indexed args; `via=` fibrations; scalar
   program parameters) and emits
   [`IRDataInput`][quivers.transpile.ir.IRDataInput] entries.

Sentinel construction and property-form `arg_constraints` resolution
are handled by `_make_sentinel` and `_resolve_arg_constraints`. The
sentinel is the only place in the transpile layer that materialises
torch tensors; renderers never do.
"""

from __future__ import annotations

import re

import didactic.api as dx
import torch
import torch.distributions.constraints as c
from torch.distributions.distribution import Distribution

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgList,
    DrawArgMatrix,
    DrawArgName,
    DrawArgScalar,
    Expr,
    ExprIdent,
    LetStep,
    MarginalizeStep,
    Module,
    MorphismDecl,
    ObjectDecl,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    ReturnStep,
    SampleStep,
    ScoreStep,
    TypeFromExpr,
)
from quivers.dsl.ast_nodes.declarations import (
    ExportDecl,
    ScalarParam,
)
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLambda,
    LetExprList,
    LetExprLiteral,
    LetExprMethodCall,
    LetExprNode,
    LetExprString,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.dsl.ast_nodes.objects import (
    ContinuousConstructor,
    DiscreteConstructor,
    ObjectExpr,
    TypeName,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile.backends._resolve import (
    ResolvedDist,
    build_let_table,
    build_morphism_table,
    resolve_step_dist,
)
from quivers.transpile.family_meta import (
    FAMILY_META,
    FamilyMeta,
)
from quivers.transpile.ir import (
    CSIntegerInterval,
    CSNonnegativeInteger,
    CSReal,
    Constraint,
    ConstraintSpec,
    Dim,
    DimDynamic,
    DimStatic,
    IRArg,
    IRArgBroadcast,
    IRArgFamilyRef,
    IRArgList,
    IRArgMatrix,
    IRArgNumber,
    IRArgRef,
    IRDataInput,
    IRDeterministic,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRReturn,
    IRSample,
    IRScore,
    Plate,
    event_shape_of,
    from_constraint,
)


# A parsed bracket-indexed argument string: `name[i0][i1]...`.
_BRACKET_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[[^\]]+\])+)$")
_BRACKET_INDICES_RE = re.compile(r"\[([^\]]+)\]")


class Lower(dx.Mapping[Module, IRProgram]):
    """Map a parsed [`Module`][quivers.dsl.ast_nodes.Module] to an
    [`IRProgram`][quivers.transpile.ir.IRProgram].

    Stateless: the registry and resolver tables are derived from the
    input module on every call. Sentinel parameter sets are cached
    per-call by `(family_name, IR-arg-tuple-key)`.
    """

    def forward(self, module: Module) -> IRProgram:  # type: ignore[override]
        expanded = expand_composite_lets(module, target="stan")
        morphisms = build_morphism_table(expanded)
        lets = build_let_table(expanded)
        cards = object_cardinalities(expanded)
        program = self._pick_program(expanded)
        family_set = frozenset(FAMILY_META)

        # Build a sentinel cache shared across the call so repeated
        # call sites pay one instantiation cost.
        sentinel_cache: dict[tuple[str, tuple[str, ...]], Distribution] = {}

        ctx = _LowerCtx(
            morphisms=morphisms,
            lets=lets,
            cards=cards,
            family_set=family_set,
            sentinel_cache=sentinel_cache,
            program=program,
        )

        body = self._lower_steps(program.draws, ctx)
        if program.return_vars:
            body = (*body, IRReturn(names=tuple(program.return_vars)))
        inputs = self._build_inputs(program, body, ctx)
        return IRProgram(name=program.name, inputs=inputs, body=body)

    def _pick_program(self, module: Module) -> ProgramDecl:
        """Pick the `ProgramDecl` to lower.

        When the module declares multiple programs, prefer one
        referenced by an `export` declaration; otherwise pick the
        last declared program.
        """
        programs: list[ProgramDecl] = []
        exported_names: set[str] = set()
        for stmt in module.statements:
            if isinstance(stmt, ProgramDecl):
                programs.append(stmt)
            elif isinstance(stmt, ExportDecl) and isinstance(
                stmt.expr, ExprIdent
            ):
                exported_names.add(stmt.expr.name)
        if not programs:
            raise UnsupportedConstruct(
                "qvr-lower",
                ["no program_decl: nothing to lower"],
            )
        return next(
            (p for p in programs if p.name in exported_names),
            programs[-1],
        )

    def _lower_steps(
        self,
        steps: tuple[ProgramStep, ...],
        ctx: _LowerCtx,
        enclosing_latents: frozenset[str] = frozenset(),
    ) -> tuple[IRNode, ...]:
        """Lower a tuple of program steps into IR nodes.

        ``enclosing_latents`` carries the set of latent variable names
        bound by surrounding marginalize scopes. Inside an observe
        whose ``via=`` clause is present, references to these latents
        are rewritten so the IR carries the per-position fibration
        threading (see `_thread_via`).
        """
        out: list[IRNode] = []
        for step in steps:
            out.append(self._lower_step(step, ctx, enclosing_latents))
        return tuple(out)

    def _lower_step(
        self,
        step: ProgramStep,
        ctx: _LowerCtx,
        enclosing_latents: frozenset[str] = frozenset(),
    ) -> IRNode:
        if isinstance(step, SampleStep):
            return self._lower_sample(step, ctx)
        if isinstance(step, ObserveStep):
            return self._lower_observe(step, ctx, enclosing_latents)
        if isinstance(step, MarginalizeStep):
            return self._lower_marginalize(step, ctx, enclosing_latents)
        if isinstance(step, LetStep):
            return self._lower_let(step, ctx)
        if isinstance(step, ScoreStep):
            return self._lower_score(step)
        if isinstance(step, ReturnStep):
            return IRReturn(names=step.vars)
        raise UnsupportedConstruct(
            "qvr-lower", [f"step:{step.kind}"]
        )

    def _lower_sample(self, step: SampleStep, ctx: _LowerCtx) -> IRSample:
        resolved = resolve_step_dist(
            step.morphism,
            step.args,
            morphisms=ctx.morphisms,
            lets=ctx.lets,
            family_registry=ctx.family_set,
            target="qvr-lower",
        )
        meta = FAMILY_META[resolved.family]
        ir_args, arg_names = self._lower_args(
            meta, resolved, ctx,
            event_axes=_event_axis_names(step),
            axes_index=step.index,
            structural_args=step.args,
        )
        plate = self._build_plate(step, ctx, meta, ir_args)
        constraint = from_constraint(_resolve_support(meta, ir_args, ctx))
        if len(step.vars) != 1:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    "sample:destructuring-tuple: lower expects one "
                    "bound name per SampleStep after composite "
                    "expansion"
                ],
            )
        return IRSample(
            name=step.vars[0],
            family=resolved.family,
            args=ir_args,
            arg_names=arg_names,
            constraint=constraint,
            plate=plate,
        )

    def _lower_observe(
        self,
        step: ObserveStep,
        ctx: _LowerCtx,
        enclosing_latents: frozenset[str] = frozenset(),
    ) -> IRObserve:
        resolved = resolve_step_dist(
            step.morphism,
            step.args,
            morphisms=ctx.morphisms,
            lets=ctx.lets,
            family_registry=ctx.family_set,
            target="qvr-lower",
        )
        meta = FAMILY_META[resolved.family]
        ir_args, arg_names = self._lower_args(
            meta, resolved, ctx,
            event_axes=_event_axis_names(step),
            axes_index=step.index,
            structural_args=step.args,
        )
        if step.via is not None and enclosing_latents:
            ir_args = tuple(
                _thread_via(a, step.via, enclosing_latents) for a in ir_args
            )
        plate = self._build_plate(step, ctx, meta, ir_args)
        constraint = from_constraint(_resolve_support(meta, ir_args, ctx))
        return IRObserve(
            name=step.var,
            family=resolved.family,
            args=ir_args,
            arg_names=arg_names,
            constraint=constraint,
            plate=plate,
            via=step.via,
        )

    def _lower_marginalize(
        self,
        step: MarginalizeStep,
        ctx: _LowerCtx,
        enclosing_latents: frozenset[str] = frozenset(),
    ) -> IRMarginalize:
        resolved = resolve_step_dist(
            step.morphism,
            step.args,
            morphisms=ctx.morphisms,
            lets=ctx.lets,
            family_registry=ctx.family_set,
            target="qvr-lower",
        )
        meta = FAMILY_META[resolved.family]
        ir_args, arg_names = self._lower_args(
            meta, resolved, ctx,
            event_axes=_marginalize_event_axis_names(step),
            axes_index=step.index,
            structural_args=step.args,
        )
        plate = self._build_marginalize_plate(step, ctx, meta, ir_args)
        constraint = from_constraint(_resolve_support(meta, ir_args, ctx))
        if step.reduction not in (None, "logsumexp"):
            raise UnsupportedConstruct(
                "qvr-lower",
                [f"marginalize:reduction:{step.reduction}"],
            )
        scope = self._lower_steps(
            step.scope, ctx, enclosing_latents | frozenset({step.var})
        )
        return IRMarginalize(
            latent=step.var,
            family=resolved.family,
            args=ir_args,
            arg_names=arg_names,
            constraint=constraint,
            plate=plate,
            reduction="logsumexp",
            scope=scope,
        )

    def _lower_let(self, step: LetStep, ctx: _LowerCtx) -> IRDeterministic:
        # Deterministic let-step. The constraint is `Real()` by
        # default; a richer derivation would inspect the expression
        # tree for its output type. Lower keeps the expression tree
        # unchanged (rendered per-backend downstream).
        del ctx
        plate = Plate(event_dims=(), batch_dims=())
        return IRDeterministic(
            name=step.name,
            expr=step.value,
            constraint=CSReal(),
            plate=plate,
        )

    def _lower_score(self, step: ScoreStep) -> IRScore:
        return IRScore(name=step.name, expr=step.value)

    def _lower_args(
        self,
        meta: FamilyMeta,
        resolved: ResolvedDist,
        ctx: _LowerCtx,
        *,
        event_axes: tuple[str, ...],
        axes_index: ObjectExpr | None,
        structural_args: tuple[DrawArg, ...] | None,
    ) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
        """Match the user-supplied args against `arg_constraints` and
        build the IR-level arg tuple plus the parallel arg-name tuple.

        ``structural_args`` is the original step's tagged-union arg
        list (preserves compound `DrawArgList` / `DrawArgMatrix`
        forms). When it is ``None`` (e.g. the step had no explicit
        args and the resolver supplied family defaults) the wire-form
        ``resolved.args`` is decoded into IR; that decoding handles
        atomic literals and bracket-indexed references but cannot
        recover compound structure (none arises through this default
        path today).

        Bracket-indexed string args (`"phi[z]"`) are parsed into
        `IRArgRef(name=..., indices=...)` trees. Scalar args that
        need to satisfy an `IndependentConstraint(base, n)` are
        wrapped in `IRArgBroadcast`. Morphism-name args referenced
        by wrapper families become `IRArgFamilyRef`.
        """
        # First-pass IR conversion (without knowing arg_constraints
        # yet so we can compute the sentinel for property-form
        # arg_constraints).
        if structural_args is not None:
            pre_args = tuple(
                self._raw_arg_to_ir(a, ctx) for a in structural_args
            )
        else:
            raw_args = resolved.args or ()
            pre_args = tuple(
                self._raw_arg_to_ir(a, ctx) for a in raw_args
            )
        arg_names = self._arg_names_for(meta, pre_args, ctx)
        # Re-walk with arg_constraints in hand to apply
        # IRArgBroadcast wrapping where needed.
        constraints_map = _resolve_arg_constraints(meta, pre_args, ctx)
        out: list[IRArg] = []
        for arg, name in zip(pre_args, arg_names, strict=False):
            expected = constraints_map.get(name)
            out.append(
                self._wrap_for_constraint(
                    arg, expected, event_axes, axes_index, ctx
                )
            )
        return tuple(out), arg_names

    def _arg_names_for(
        self,
        meta: FamilyMeta,
        args: tuple[IRArg, ...],
        ctx: _LowerCtx,
    ) -> tuple[str, ...]:
        """Return the parallel arg-name tuple for `args`.

        Reads from `meta.distribution_class.arg_constraints` when
        possible. When `arg_constraints` is a property (Wishart,
        Uniform, etc.), instantiates a sentinel to read the
        instance-level dict. The returned tuple is positional: the
        i'th entry names the i'th user-supplied arg.
        """
        cls_attr = meta.distribution_class.arg_constraints
        if isinstance(cls_attr, dict):
            names = tuple(cls_attr.keys())
        else:
            sentinel = _make_sentinel(meta, args, ctx)
            names = tuple(sentinel.arg_constraints.keys())
        if len(args) > len(names):
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"family:{meta.qvr_name}:arity-mismatch: user "
                    f"supplied {len(args)} args; "
                    f"arg_constraints has {len(names)} positions "
                    f"({list(names)})"
                ],
            )
        return names[: len(args)]

    def _raw_arg_to_ir(
        self,
        raw: DrawArg | str | float,
        ctx: _LowerCtx,
    ) -> IRArg:
        """First-pass conversion: parser-form arg to IR form, without
        broadcast wrapping.

        Accepts either the wire-form ``str | float`` (used for nested
        index expressions and resolver-internal calls) or a tagged
        `DrawArg` variant (used at top-level for step args).
        """
        if isinstance(raw, DrawArgScalar):
            return IRArgNumber(value=raw.value)
        if isinstance(raw, DrawArgName):
            return self._atom_text_to_ir(raw.text, ctx)
        if isinstance(raw, DrawArgList):
            return IRArgList(
                elements=tuple(
                    self._raw_arg_to_ir(e, ctx) for e in raw.elements
                )
            )
        if isinstance(raw, DrawArgMatrix):
            ir_rows: list[IRArgList] = []
            for row in raw.rows:
                ir_rows.append(
                    IRArgList(
                        elements=tuple(
                            self._raw_arg_to_ir(e, ctx) for e in row.elements
                        )
                    )
                )
            return IRArgMatrix(rows=tuple(ir_rows))
        if isinstance(raw, (int, float)):
            return IRArgNumber(value=float(raw))
        return self._atom_text_to_ir(raw, ctx)

    def _atom_text_to_ir(self, text: str, ctx: _LowerCtx) -> IRArg:
        """Convert an atomic wire-form string (identifier or encoded
        bracket form) into the corresponding IR arg."""
        # Wrapper family ref: a bare name pointing at a morphism with
        # an `~ Family(...)` init clause.
        if text in ctx.morphisms:
            decl = ctx.morphisms[text]
            if decl.init_family is not None:
                return IRArgFamilyRef(name=text)
        m = _BRACKET_RE.match(text)
        if m is not None:
            name = m.group(1)
            indices_text = m.group(2)
            indices = tuple(
                self._raw_arg_to_ir(idx, ctx)
                for idx in _BRACKET_INDICES_RE.findall(indices_text)
            )
            return IRArgRef(name=name, indices=indices)
        # A bare identifier reference. Could be a scalar program
        # param, a previously-bound var, a free name (exogenous data
        # input), or an explicit numeric like "0.5" wrapped as str.
        if _is_number_text(text):
            return IRArgNumber(value=float(text))
        return IRArgRef(name=text, indices=())

    def _wrap_for_constraint(
        self,
        arg: IRArg,
        expected: Constraint | None,
        event_axes: tuple[str, ...],
        axes_index: ObjectExpr | None,
        ctx: _LowerCtx,
    ) -> IRArg:
        """When the user supplied a scalar but the constraint is
        `IndependentConstraint(base, n>=1)`, wrap as
        `IRArgBroadcast` whose `target_shape` is derived from the
        step's event axes (the `over=` clause when present, falling
        back to the step's `index`).

        Scalar literals (`IRArgNumber`) always qualify. An unindexed
        `IRArgRef` qualifies when it names a scalar binding (a
        `ScalarParam` of the active program); a renderer must then
        broadcast the scalar to the vector arg position rather than
        passing the scalar through as if it were already a tensor of
        the expected shape.
        """
        if expected is None:
            return arg
        if not isinstance(expected, c._IndependentConstraint):
            return arg
        if not isinstance(arg, (IRArgNumber, IRArgRef)):
            return arg
        if isinstance(arg, IRArgRef):
            if arg.indices:
                return arg
            if arg.name not in _scalar_binding_names(ctx):
                return arg
        target = self._broadcast_target(
            expected, event_axes, axes_index, ctx
        )
        if target is None:
            return arg
        return IRArgBroadcast(value=arg, target_shape=target)

    def _broadcast_target(
        self,
        expected: c._IndependentConstraint,
        event_axes: tuple[str, ...],
        axes_index: ObjectExpr | None,
        ctx: _LowerCtx,
    ) -> tuple[int, ...] | None:
        """Derive the broadcast `target_shape` from the step's event
        axes when the expected constraint is `IndependentConstraint`.

        Prefers the axis names supplied by `event_axes` (the `over=`
        clause of the surrounding step). Falls back to a single-axis
        shape derived from `axes_index` when no event axes are
        declared (the bare scalar-family form).
        """
        if event_axes:
            base: list[int] = []
            for axis_name in event_axes:
                size = ctx.cards.get(axis_name)
                if size is None:
                    return None
                base.append(size)
            base_shape = tuple(base)
        elif axes_index is not None:
            size = axis_shape(axes_index, ctx.cards)
            if size is None:
                return None
            base_shape = (size,) * expected.event_dim
        else:
            return None
        return event_shape_of(expected, base_shape)

    def _build_plate(
        self,
        step: SampleStep | ObserveStep,
        ctx: _LowerCtx,
        meta: FamilyMeta,
        ir_args: tuple[IRArg, ...],
    ) -> Plate:
        """Build the [`Plate`][quivers.transpile.ir.Plate] for a
        sample / observe step.

        `over` axes become `event_dims`; `iid_over` axes become
        `batch_dims` in source declaration order. When no
        AxisSpec is present, the step's `index` is treated as the
        single batch axis (the common scalar-family form).
        """
        axes = step.axes
        if axes is not None:
            event_dims = tuple(self._axis_dim(a, ctx) for a in axes.over)
            batch_dims = tuple(
                self._axis_dim(a, ctx) for a in axes.iid_over
            )
            return Plate(event_dims=event_dims, batch_dims=batch_dims)
        # No AxisSpec on this step. If the step has an index, treat
        # it as the batch axis (scalar family) or as the event axis
        # (vector family per FAMILY_META). Distinguishing dispatches
        # on the family's class-level event_dim.
        event_dim = _event_dim_of(meta, ir_args, ctx)
        if step.index is None:
            return Plate(event_dims=(), batch_dims=())
        dim = self._object_expr_dim(step.index, ctx)
        if event_dim == 0:
            return Plate(event_dims=(), batch_dims=(dim,))
        return Plate(event_dims=(dim,), batch_dims=())

    def _build_marginalize_plate(
        self,
        step: MarginalizeStep,
        ctx: _LowerCtx,
        meta: FamilyMeta,
        ir_args: tuple[IRArg, ...],
    ) -> Plate:
        """The marginalize step carries an `over=` list (the grouping
        axes) which become `batch_dims`; the latent itself contributes
        an event_dim only when the family's event_dim > 0.
        """
        event_dim = _event_dim_of(meta, ir_args, ctx)
        batch_dims: tuple[Dim, ...] = ()
        if step.over is not None:
            batch_dims = (
                self._axis_dim(step.over, ctx),
            )
        elif step.over_objs is not None:
            batch_dims = tuple(self._axis_dim(a, ctx) for a in step.over_objs)
        event_dims: tuple[Dim, ...] = ()
        if step.index is not None and event_dim == 0:
            # The latent's index (e.g. `Topic`) is its support
            # cardinality; it doesn't become an event_dim of the
            # Plate (the latent is integrated out), but the
            # constraint still records the support range.
            pass
        if step.index is not None and event_dim > 0:
            event_dims = (self._object_expr_dim(step.index, ctx),)
        return Plate(event_dims=event_dims, batch_dims=batch_dims)

    def _axis_dim(self, axis_name: str, ctx: _LowerCtx) -> Dim:
        """Convert an axis name into a `DimStatic` from the cardinality
        table or a `DimDynamic` when the axis is unknown."""
        size = ctx.cards.get(axis_name)
        if size is None:
            return DimDynamic(size_name=f"N_{axis_name}", name=axis_name)
        return DimStatic(size=size, name=axis_name)

    def _object_expr_dim(
        self, expr: ObjectExpr, ctx: _LowerCtx
    ) -> Dim:
        """Convert an `ObjectExpr` into a `Dim`."""
        if isinstance(expr, TypeName):
            return self._axis_dim(expr.name, ctx)
        if isinstance(expr, DiscreteConstructor) and expr.args:
            return DimStatic(size=int(expr.args[0]), name="anon")
        raise UnsupportedConstruct(
            "qvr-lower",
            [f"object-expr:{expr.kind}"],
        )

    def _build_inputs(
        self,
        program: ProgramDecl,
        body: tuple[IRNode, ...],
        ctx: _LowerCtx,
    ) -> tuple[IRDataInput, ...]:
        """Discover exogenous identifiers and emit `IRDataInput`
        entries.

        Sources of exogenous identifiers:
        * Scalar `ProgramDecl.type_params` (e.g. `alpha : Real`).
        * Free names in let / score expressions.
        * Free names referenced in bracket-indexed arg expressions
          (`mu[cls]` references `cls` and `mu`).
        * `observe ... [via=fibration]` fibrations.
        * Observed variables themselves (the rhs of `observe`).
        """
        bound = self._bound_names(body)
        used = self._used_names(body)
        # Scalar program parameters: typed `Real` / `Nat` from
        # `type_params`, plus the bare-name shorthand `params`.
        param_inputs: list[IRDataInput] = []
        seen_param: set[str] = set()
        if program.type_params is not None:
            for p in program.type_params:
                if isinstance(p, ScalarParam):
                    if p.name in seen_param:
                        continue
                    seen_param.add(p.name)
                    spec: ConstraintSpec = (
                        CSReal()
                        if p.scalar_kind == "Real"
                        else CSNonnegativeInteger()
                    )
                    param_inputs.append(
                        IRDataInput(
                            name=p.name,
                            constraint=spec,
                            plate=Plate(event_dims=(), batch_dims=()),
                        )
                    )
        if program.params is not None:
            for name in program.params:
                if name in seen_param:
                    continue
                seen_param.add(name)
                param_inputs.append(
                    IRDataInput(
                        name=name,
                        constraint=CSReal(),
                        plate=Plate(event_dims=(), batch_dims=()),
                    )
                )

        # Exogenous via / fibration inputs.
        via_inputs: list[IRDataInput] = []
        seen_via: set[str] = set()
        for node in _walk_nodes(body):
            if isinstance(node, IRObserve) and node.via is not None:
                vname = node.via
                if vname in seen_via or vname in bound:
                    continue
                seen_via.add(vname)
                # via_fibrations are integer indexers; the size is the
                # batch dim of the surrounding observe step.
                plate = node.plate
                upper = self._first_batch_card(plate, ctx)
                via_constraint: ConstraintSpec
                if upper is None:
                    via_constraint = CSNonnegativeInteger()
                else:
                    via_constraint = CSIntegerInterval(
                        lower=0, upper=max(upper - 1, 0)
                    )
                via_inputs.append(
                    IRDataInput(
                        name=vname,
                        constraint=via_constraint,
                        plate=Plate(
                            event_dims=(),
                            batch_dims=plate.batch_dims,
                        ),
                    )
                )

        # Observed-var inputs (the `var` named in each `observe`).
        obs_inputs: list[IRDataInput] = []
        seen_obs: set[str] = set()
        for node in _walk_nodes(body):
            if isinstance(node, IRObserve):
                if node.name in seen_obs:
                    continue
                seen_obs.add(node.name)
                obs_inputs.append(
                    IRDataInput(
                        name=node.name,
                        constraint=node.constraint,
                        plate=node.plate,
                    )
                )

        # Free names referenced in let / score expressions and in
        # bracket-index args. Any name used but not bound (and not
        # already in param/via/obs) is an exogenous data input.
        seen_param_names = seen_param | seen_via | seen_obs
        free_inputs: list[IRDataInput] = []
        seen_free: set[str] = set()
        for name in used:
            if name in bound:
                continue
            if name in seen_param_names:
                continue
            if name in seen_free:
                continue
            seen_free.add(name)
            free_inputs.append(
                IRDataInput(
                    name=name,
                    constraint=CSReal(),
                    plate=Plate(event_dims=(), batch_dims=()),
                )
            )

        return tuple(param_inputs + via_inputs + obs_inputs + free_inputs)

    def _bound_names(self, body: tuple[IRNode, ...]) -> set[str]:
        """Return the set of names bound anywhere in the body
        (recursive)."""
        out: set[str] = set()
        for node in _walk_nodes(body):
            if isinstance(node, (IRSample, IRObserve, IRDeterministic)):
                out.add(node.name)
            elif isinstance(node, IRScore):
                out.add(node.name)
            elif isinstance(node, IRMarginalize):
                out.add(node.latent)
            elif isinstance(node, IRDataInput):
                out.add(node.name)
        return out

    def _used_names(self, body: tuple[IRNode, ...]) -> list[str]:
        """Return the ordered list of names referenced in the body,
        from arg references, bracket indices, and let / score
        expression trees. Order preserved for deterministic input
        emission.

        The via-threading sentinel `VIA_ROW_VAR_SENTINEL` is filtered
        out: it stands in for whichever per-position loop variable a
        renderer chooses and is never an exogenous data input.
        """
        out: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name == VIA_ROW_VAR_SENTINEL:
                return
            if name in seen:
                return
            seen.add(name)
            out.append(name)

        for node in _walk_nodes(body):
            if isinstance(node, (IRSample, IRObserve, IRMarginalize)):
                for a in node.args:
                    for n in free_names_in_arg(a):
                        add(n)
            if isinstance(node, IRDeterministic):
                for n in free_vars_in_let(node.expr):
                    add(n)
            if isinstance(node, IRScore):
                for n in free_vars_in_let(node.expr):
                    add(n)
        return out

    def _first_batch_card(
        self, plate: Plate, ctx: _LowerCtx
    ) -> int | None:
        """Return the cardinality of the first batch dim, when known."""
        del ctx  # cards already resolved into the Dim
        if not plate.batch_dims:
            return None
        first = plate.batch_dims[0]
        if isinstance(first, DimStatic):
            return first.size
        return None


# ---------------------------------------------------------------------------
# Internal context carried through the lowering pass.
# ---------------------------------------------------------------------------


class _LowerCtx(dx.Model):
    """Internal carrier for the resolver / cardinality tables and the
    sentinel cache. Threaded through the lowering recursion."""

    morphisms: dict[str, MorphismDecl] = dx.field(opaque=True)
    lets: dict[str, Expr] = dx.field(opaque=True)
    cards: dict[str, int]
    family_set: frozenset[str]
    sentinel_cache: dict[tuple[str, tuple[str, ...]], Distribution] = (
        dx.field(opaque=True)
    )
    program: ProgramDecl = dx.field(opaque=True)


# ---------------------------------------------------------------------------
# Lower-internal helpers: object cardinalities, axis shape, free names.
# ---------------------------------------------------------------------------


def object_cardinalities(module: Module) -> dict[str, int]:
    """Return name -> cardinality for every `FinSet N` object decl."""
    out: dict[str, int] = {}
    for stmt in module.statements:
        if not isinstance(stmt, ObjectDecl):
            continue
        init = stmt.init
        if isinstance(init, TypeFromExpr):
            expr = init.expr
            if isinstance(expr, DiscreteConstructor) and expr.args:
                out[stmt.name] = int(expr.args[0])
            elif isinstance(expr, ContinuousConstructor) and expr.args:
                # `Real D` etc.: take the first arg as the size.
                try:
                    out[stmt.name] = int(expr.args[0])
                except ValueError:
                    pass
    return out


def axis_shape(
    expr: ObjectExpr, cards: dict[str, int]
) -> int | None:
    """Return the cardinality of an axis object expression."""
    if isinstance(expr, TypeName):
        return cards.get(expr.name)
    if isinstance(expr, DiscreteConstructor) and expr.args:
        try:
            return int(expr.args[0])
        except ValueError:
            return None
    return None


def build_shape_table(
    program: ProgramDecl, cards: dict[str, int]
) -> dict[str, tuple[int, ...]]:
    """Return name -> shape for every sample / observe / let binding
    in `program`."""
    out: dict[str, tuple[int, ...]] = {}
    for step in program.draws:
        if isinstance(step, SampleStep):
            shape = _step_shape(step.index, cards)
            for v in step.vars:
                out[v] = shape
        elif isinstance(step, ObserveStep):
            out[step.var] = _step_shape(step.index, cards)
        elif isinstance(step, LetStep):
            out[step.name] = ()
        elif isinstance(step, MarginalizeStep):
            out[step.var] = _step_shape(step.index, cards)
            inner = build_shape_table(
                ProgramDecl(
                    name=program.name,
                    domain=program.domain,
                    codomain=program.codomain,
                    draws=step.scope,
                ),
                cards,
            )
            out.update(inner)
    return out


def _step_shape(
    index: ObjectExpr | None, cards: dict[str, int]
) -> tuple[int, ...]:
    if index is None:
        return ()
    n = axis_shape(index, cards)
    return (n,) if n is not None else ()


def exogenous_data_inputs(
    program: ProgramDecl, bound: set[str]
) -> list[str]:
    """Return the ordered list of exogenous identifier names
    referenced in `program` but not bound by any step.

    Sources: free names in let / score expressions; bracket-indexed
    arg references (`mu[cls]` -> `cls`, `mu`); `via=` fibrations.
    Order is deterministic by traversal order.
    """
    out: list[str] = []
    seen: set[str] = set()

    def consider(name: str) -> None:
        if name in bound or name in seen:
            return
        seen.add(name)
        out.append(name)

    for step in program.draws:
        for n in _names_in_step(step):
            consider(n)
    return out


def _names_in_step(step: ProgramStep) -> list[str]:
    out: list[str] = []
    if isinstance(step, SampleStep):
        for a in step.args or ():
            out.extend(_names_in_raw_arg(a))
    elif isinstance(step, ObserveStep):
        for a in step.args or ():
            out.extend(_names_in_raw_arg(a))
        if step.via is not None:
            out.append(step.via)
    elif isinstance(step, MarginalizeStep):
        for a in step.args or ():
            out.extend(_names_in_raw_arg(a))
        for inner in step.scope:
            out.extend(_names_in_step(inner))
    elif isinstance(step, LetStep):
        out.extend(free_vars_in_let(step.value))
    elif isinstance(step, ScoreStep):
        out.extend(free_vars_in_let(step.value))
    return out


def _names_in_raw_arg(arg: DrawArg | str | float) -> list[str]:
    if isinstance(arg, DrawArgScalar):
        return []
    if isinstance(arg, DrawArgName):
        return _names_in_atom_text(arg.text)
    if isinstance(arg, DrawArgList):
        out: list[str] = []
        for element in arg.elements:
            out.extend(_names_in_atom_text(element) if isinstance(element, str) else [])
        return out
    if isinstance(arg, DrawArgMatrix):
        out = []
        for row in arg.rows:
            for element in row.elements:
                if isinstance(element, str):
                    out.extend(_names_in_atom_text(element))
        return out
    if not isinstance(arg, str):
        return []
    return _names_in_atom_text(arg)


def _names_in_atom_text(text: str) -> list[str]:
    """Walk a wire-form atom text and return the names it references."""
    m = _BRACKET_RE.match(text)
    if m is None:
        if _is_number_text(text):
            return []
        return [text]
    out: list[str] = [m.group(1)]
    for idx in _BRACKET_INDICES_RE.findall(m.group(2)):
        out.extend(_names_in_atom_text(idx))
    return out


def free_vars_in_let(expr: LetExprNode) -> list[str]:
    """Return the ordered list of free variable names in a let
    expression tree."""
    out: list[str] = []
    seen: set[str] = set()

    def visit(n: LetExprNode, bound: frozenset[str]) -> None:
        if isinstance(n, LetExprVar):
            if n.name in bound or n.name in seen:
                return
            seen.add(n.name)
            out.append(n.name)
            return
        if isinstance(n, LetExprLiteral):
            return
        if isinstance(n, LetExprString):
            return
        if isinstance(n, LetExprBinOp):
            visit(n.left, bound)
            visit(n.right, bound)
            return
        if isinstance(n, LetExprUnaryOp):
            visit(n.operand, bound)
            return
        if isinstance(n, LetExprCall):
            for a in n.args:
                visit(a, bound)
            return
        if isinstance(n, LetExprIndex):
            visit(n.array, bound)
            for i in n.indices:
                visit(i, bound)
            return
        if isinstance(n, LetExprList):
            for it in n.items:
                visit(it, bound)
            return
        if isinstance(n, LetExprLambda):
            visit(n.body, bound | frozenset({n.param}))
            return
        if isinstance(n, LetExprFactor):
            inner_bound = bound | frozenset(b.var for b in n.binders)
            if n.body is not None:
                visit(n.body, inner_bound)
            for case in n.cases:
                visit(case.value, inner_bound)
            return
        if isinstance(n, LetExprMethodCall):
            visit(n.receiver, bound)
            for a in n.args:
                visit(a, bound)
            return

    visit(expr, frozenset())
    return out


def free_names_in_arg(arg: IRArg) -> list[str]:
    """Return the ordered list of free names in an IR arg tree."""
    out: list[str] = []
    seen: set[str] = set()

    def visit(a: IRArg) -> None:
        if isinstance(a, IRArgRef):
            if a.name not in seen:
                seen.add(a.name)
                out.append(a.name)
            for i in a.indices:
                visit(i)
            return
        if isinstance(a, IRArgBroadcast):
            visit(a.value)
            return
        if isinstance(a, IRArgList):
            for e in a.elements:
                visit(e)
            return
        if isinstance(a, IRArgMatrix):
            for row in a.rows:
                visit(row)
            return
        if isinstance(a, IRArgFamilyRef):
            if a.name not in seen:
                seen.add(a.name)
                out.append(a.name)
            return
        if isinstance(a, IRArgNumber):
            return

    visit(arg)
    return out


def arg_ref_shape(
    arg: IRArg, shape_table: dict[str, tuple[int, ...]]
) -> tuple[int, ...]:
    """Return the broadcast-evaluated shape of an IR arg referenced
    through `shape_table`."""
    if isinstance(arg, IRArgRef):
        base = shape_table.get(arg.name, ())
        # Indexing peels off one dim per index expression.
        peeled = max(0, len(base) - len(arg.indices))
        return base[:peeled]
    if isinstance(arg, IRArgList):
        return (len(arg.elements),)
    if isinstance(arg, IRArgMatrix):
        return (len(arg.rows), len(arg.rows[0].elements) if arg.rows else 0)
    if isinstance(arg, IRArgBroadcast):
        return arg.target_shape
    return ()


def lower_factors(
    program: ProgramDecl, cards: dict[str, int]
) -> dict[str, tuple[int, ...]]:
    """Return name -> factor shape for every plated binding in
    `program`. Mirror of [`build_shape_table`][quivers.transpile.lower.build_shape_table]
    intended for renderers that need the factor shape (event_dims
    plus batch_dims) of every name."""
    return build_shape_table(program, cards)


def inline_list_lets(
    program: ProgramDecl,
) -> dict[str, tuple[str, ...]]:
    """Return name -> tuple of element names for every let-step
    whose RHS is a [`LetExprList`][quivers.dsl.ast_nodes.let_expressions.LetExprList]
    of bare-variable references."""
    out: dict[str, tuple[str, ...]] = {}
    for step in program.draws:
        if not isinstance(step, LetStep):
            continue
        if isinstance(step.value, LetExprList):
            names: list[str] = []
            ok = True
            for it in step.value.items:
                if not isinstance(it, LetExprVar):
                    ok = False
                    break
                names.append(it.name)
            if ok:
                out[step.name] = tuple(names)
    return out


# ---------------------------------------------------------------------------
# Sentinel parameter construction and `arg_constraints` resolution.
# ---------------------------------------------------------------------------


def _make_sentinel(
    meta: FamilyMeta,
    args: tuple[IRArg, ...],
    ctx: _LowerCtx,
) -> Distribution:
    """Build a sentinel `Distribution` instance for `meta` with
    placeholder tensors derived from `args`.

    The instance carries the right `event_shape`, `batch_shape`,
    `support`, and (when `arg_constraints` is a property) the right
    per-arg constraints. Cached by `(family_name, IR-arg-tuple-key)`
    so repeated call sites pay one instantiation cost.

    Reference args are dimensioned using the class-level
    `arg_constraints` (when available) so distributions like
    `Categorical(probs)` get a vector placeholder rather than a
    scalar.
    """
    key = (meta.qvr_name, tuple(_arg_key(a) for a in args))
    if key in ctx.sentinel_cache:
        return ctx.sentinel_cache[key]
    expected_shapes = _expected_arg_shapes(meta, len(args))
    sentinel_args = tuple(
        _arg_to_tensor(a, ctx, expected_shapes[i])
        for i, a in enumerate(args)
    )
    try:
        instance = meta.distribution_class(*sentinel_args)
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"family:{meta.qvr_name}:sentinel-failed: cannot "
                f"instantiate {meta.distribution_class.__name__} "
                f"with placeholder args derived from "
                f"{args!r}: {exc!r}"
            ],
        ) from exc
    ctx.sentinel_cache[key] = instance
    return instance


def _expected_arg_shapes(
    meta: FamilyMeta, n_args: int
) -> tuple[tuple[int, ...], ...]:
    """Per-arg expected shape derived from class-level
    `arg_constraints`. Used to size placeholder tensors for the
    sentinel."""
    cls_attr = meta.distribution_class.arg_constraints
    if not isinstance(cls_attr, dict):
        return tuple(() for _ in range(n_args))
    out: list[tuple[int, ...]] = []
    for _, constraint in list(cls_attr.items())[:n_args]:
        out.append(_constraint_default_shape(constraint))
    while len(out) < n_args:
        out.append(())
    return tuple(out)


def _constraint_default_shape(constraint: Constraint) -> tuple[int, ...]:
    """Pick a small default shape for a parameter whose constraint
    requires nonzero event_dim."""
    if isinstance(constraint, c._IndependentConstraint):
        return (2,) * constraint.event_dim
    if isinstance(constraint, c._Simplex):
        return (2,)
    if isinstance(constraint, c._PositiveDefinite):
        return (2, 2)
    if isinstance(constraint, c._PositiveSemidefinite):
        return (2, 2)
    if isinstance(constraint, c._CorrCholesky):
        return (2, 2)
    if isinstance(constraint, c._LowerCholesky):
        return (2, 2)
    if isinstance(constraint, c._OneHot):
        return (2,)
    return ()


def _arg_to_tensor(
    arg: IRArg, ctx: _LowerCtx, expected_shape: tuple[int, ...]
) -> torch.Tensor:
    """Materialise a placeholder torch tensor for one IR arg."""
    if isinstance(arg, IRArgNumber):
        if expected_shape:
            return torch.full(expected_shape, arg.value, dtype=torch.float32)
        return torch.tensor(arg.value, dtype=torch.float32)
    if isinstance(arg, IRArgRef):
        if expected_shape:
            # For simplex-shaped expectations pad with a uniform mass
            # so the constraint check succeeds.
            return _shape_default_tensor(expected_shape)
        return torch.zeros((), dtype=torch.float32)
    if isinstance(arg, IRArgBroadcast):
        return torch.zeros(arg.target_shape, dtype=torch.float32)
    if isinstance(arg, IRArgList):
        return torch.tensor(
            [
                e.value if isinstance(e, IRArgNumber) else 0.0
                for e in arg.elements
            ],
            dtype=torch.float32,
        )
    if isinstance(arg, IRArgMatrix):
        return torch.tensor(
            [
                [
                    e.value if isinstance(e, IRArgNumber) else 0.0
                    for e in row.elements
                ]
                for row in arg.rows
            ],
            dtype=torch.float32,
        )
    if isinstance(arg, IRArgFamilyRef):
        decl = ctx.morphisms.get(arg.name)
        if decl is None or decl.init_family is None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"arg:family-ref:{arg.name}: morphism not "
                    f"declared with `~ Family(...)` init"
                ],
            )
        inner_meta = FAMILY_META[decl.init_family.family]
        inner_args = tuple(
            _raw_to_ir_for_sentinel(a) for a in decl.init_family.args
        )
        return _make_sentinel(inner_meta, inner_args, ctx)  # type: ignore[return-value]
    raise UnsupportedConstruct(
        "qvr-lower", [f"arg:unknown:{type(arg).__name__}"]
    )


def _shape_default_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    """A placeholder tensor of `shape` valid as a simplex / PD / etc.

    Uniform 1/n entries satisfy Simplex; an identity matrix satisfies
    PositiveDefinite / LowerCholesky.
    """
    if len(shape) == 1:
        n = shape[0]
        return torch.full(shape, 1.0 / n, dtype=torch.float32)
    if len(shape) == 2 and shape[0] == shape[1]:
        return torch.eye(shape[0], dtype=torch.float32)
    return torch.zeros(shape, dtype=torch.float32)


def _raw_to_ir_for_sentinel(raw: DrawArg | str | float) -> IRArg:
    """Cheap arg-to-IR conversion used only for inner sentinel
    construction (no morphism table required)."""
    if isinstance(raw, DrawArgScalar):
        return IRArgNumber(value=raw.value)
    if isinstance(raw, DrawArgName):
        return _atom_text_for_sentinel(raw.text)
    if isinstance(raw, DrawArgList):
        return IRArgList(
            elements=tuple(_raw_to_ir_for_sentinel(e) for e in raw.elements)
        )
    if isinstance(raw, DrawArgMatrix):
        return IRArgMatrix(
            rows=tuple(
                IRArgList(
                    elements=tuple(
                        _raw_to_ir_for_sentinel(e) for e in row.elements
                    )
                )
                for row in raw.rows
            )
        )
    if isinstance(raw, (int, float)):
        return IRArgNumber(value=float(raw))
    return _atom_text_for_sentinel(raw)


def _atom_text_for_sentinel(text: str) -> IRArg:
    if _is_number_text(text):
        return IRArgNumber(value=float(text))
    return IRArgRef(name=text, indices=())


def _arg_key(a: IRArg) -> str:
    """Stable string key for caching purposes."""
    if isinstance(a, IRArgNumber):
        return f"n:{a.value}"
    if isinstance(a, IRArgRef):
        return f"r:{a.name}:{len(a.indices)}"
    if isinstance(a, IRArgBroadcast):
        return f"b:{a.target_shape}:{_arg_key(a.value)}"
    if isinstance(a, IRArgList):
        return "l:" + ",".join(_arg_key(e) for e in a.elements)
    if isinstance(a, IRArgMatrix):
        return "m:" + ";".join(
            ",".join(_arg_key(e) for e in row.elements) for row in a.rows
        )
    if isinstance(a, IRArgFamilyRef):
        return f"f:{a.name}"
    return type(a).__name__


def _resolve_arg_constraints(
    meta: FamilyMeta,
    args: tuple[IRArg, ...],
    ctx: _LowerCtx,
) -> dict[str, Constraint]:
    """Return the `arg_constraints` dict for `meta`, evaluating the
    sentinel instance when `arg_constraints` is a property rather
    than a class-level dict."""
    cls_attr = meta.distribution_class.arg_constraints
    if isinstance(cls_attr, dict):
        return cls_attr
    instance = _make_sentinel(meta, args, ctx)
    return dict(instance.arg_constraints)


def _resolve_support(
    meta: FamilyMeta,
    args: tuple[IRArg, ...],
    ctx: _LowerCtx,
) -> Constraint:
    """Return the support of a call site, evaluating the sentinel
    when the family's support is a `dependent_property`."""
    cls_support = meta.distribution_class.support
    if isinstance(cls_support, c.Constraint) and not isinstance(
        cls_support, c._DependentProperty
    ):
        return cls_support
    instance = _make_sentinel(meta, args, ctx)
    return instance.support


def _event_dim_of(
    meta: FamilyMeta, args: tuple[IRArg, ...], ctx: _LowerCtx
) -> int:
    """Return the family's event_dim, evaluating the sentinel when
    the class-level support is not a concrete Constraint."""
    cls_support = meta.distribution_class.support
    if isinstance(cls_support, c.Constraint) and not isinstance(
        cls_support, c._DependentProperty
    ):
        return int(getattr(cls_support, "event_dim", 0))
    instance = _make_sentinel(meta, args, ctx)
    return int(getattr(instance.support, "event_dim", 0))


def _is_number_text(text: str) -> bool:
    """True when `text` parses as a Python float / int literal."""
    try:
        float(text)
    except ValueError:
        return False
    return True


def _event_axis_names(
    step: SampleStep | ObserveStep,
) -> tuple[str, ...]:
    """Return the event-axis names for a sample / observe step.

    These are the axes the family's event-dim ranges over; in surface
    syntax they appear as the `over=` clause of the step's
    [`AxisSpec`][quivers.dsl.ast_nodes._shared.AxisSpec]. Returns an
    empty tuple when no `AxisSpec` is present (the bare scalar-family
    form whose event-shape derivation falls back to `step.index`).
    """
    if step.axes is None:
        return ()
    return step.axes.over


#: Sentinel name carried inside an observe arg's via-threaded
#: index. Each renderer substitutes its own per-position loop
#: variable (e.g. ``n`` for the observe row).
VIA_ROW_VAR_SENTINEL = "__row_var__"


def _thread_via(
    arg: IRArg,
    via_name: str,
    enclosing_latents: frozenset[str],
) -> IRArg:
    """Thread a `[via=fibration]` per-position fibration through an
    observe-arg tree.

    For every `IRArgRef` whose ``name`` is one of the latents bound by
    a surrounding marginalize scope, append the per-position lookup
    `IRArgRef(via_name, indices=(IRArgRef(VIA_ROW_VAR_SENTINEL),))`
    to the reference's index list. The transform is structural:
    nested `IRArgRef.indices` are rewritten in place; broadcast and
    list / matrix wrappers descend into their payload.

    Without the rewrite, a renderer staring at
    `IRArgRef("phi", indices=(IRArgRef("z"),))` cannot tell from the
    IR alone that the `z` index must be looked up per-position via
    the `word_idx` fibration: `phi[z[word_idx[n]]]` rather than
    `phi[z]`. Threading the lookup at lowering time keeps every
    renderer free of the same indexing logic.
    """

    def rewrite(node: IRArg) -> IRArg:
        if isinstance(node, IRArgRef):
            new_indices = tuple(rewrite(i) for i in node.indices)
            if node.name in enclosing_latents:
                via_lookup = IRArgRef(
                    name=via_name,
                    indices=(
                        IRArgRef(name=VIA_ROW_VAR_SENTINEL, indices=()),
                    ),
                )
                new_indices = (*new_indices, via_lookup)
            return IRArgRef(name=node.name, indices=new_indices)
        if isinstance(node, IRArgBroadcast):
            return IRArgBroadcast(
                value=rewrite(node.value),
                target_shape=node.target_shape,
            )
        if isinstance(node, IRArgList):
            return IRArgList(
                elements=tuple(rewrite(e) for e in node.elements)
            )
        if isinstance(node, IRArgMatrix):
            return IRArgMatrix(
                rows=tuple(
                    IRArgList(
                        elements=tuple(rewrite(e) for e in row.elements)
                    )
                    for row in node.rows
                )
            )
        return node

    return rewrite(arg)


def _marginalize_event_axis_names(
    step: MarginalizeStep,
) -> tuple[str, ...]:
    """Event-axis names for a marginalize step.

    Marginalize carries its grouping axes on `over` / `over_objs`; the
    latent's support cardinality is taken from `step.index`. The
    grouping axes are not the event axes for the conditioning family
    (they replicate the family across groups), so this returns the
    empty tuple and lets `_broadcast_target` fall back to the latent
    index.
    """
    del step
    return ()


def _scalar_binding_names(ctx: _LowerCtx) -> frozenset[str]:
    """Return the set of identifier names bound to a scalar value in
    the active program.

    Scalar bindings are program parameters whose ``type_params`` entry
    is a [`ScalarParam`][quivers.dsl.ast_nodes.declarations.ScalarParam]
    (`Real` / `Nat`). Used by `_wrap_for_constraint` to decide when an
    unindexed reference must be broadcast to satisfy an
    `IndependentConstraint` arg position.
    """
    program = ctx.program
    if program.type_params is None:
        return frozenset()
    return frozenset(
        p.name for p in program.type_params if isinstance(p, ScalarParam)
    )


def _walk_nodes(body: tuple[IRNode, ...]):
    """Yield every IRNode in `body`, descending into `IRMarginalize`
    scopes."""
    for node in body:
        yield node
        if isinstance(node, IRMarginalize):
            yield from _walk_nodes(node.scope)


__all__ = [
    "VIA_ROW_VAR_SENTINEL",
    "Lower",
    "arg_ref_shape",
    "axis_shape",
    "build_shape_table",
    "exogenous_data_inputs",
    "free_names_in_arg",
    "free_vars_in_let",
    "inline_list_lets",
    "lower_factors",
    "object_cardinalities",
]
