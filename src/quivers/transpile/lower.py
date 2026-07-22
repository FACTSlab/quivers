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

import inspect
import re
from collections.abc import Callable

import didactic.api as dx
import torch
import torch.distributions.constraints as c
from torch.distributions.distribution import Distribution

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
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
    OptionEntry,
    OptionList,
    OptionName,
    OptionNumber,
    OptionValue,
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
    ObjectProduct,
    TypeName,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._draw_args import (
    encode_index,
    is_matrix,
    list_items,
    matrix_rows,
)
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile._resolve import (
    ResolvedDist,
    build_let_table,
    build_morphism_table,
    resolve_step_dist,
)
from quivers.transpile.family_meta import (
    FAMILY_META,
    FamilyMeta,
    finite_enumerable_at_call_site,
)
from quivers.transpile.ir import (
    CSIntegerInterval,
    CSNonnegativeInteger,
    CSPositiveDefinite,
    CSReal,
    CSRealMatrix,
    CSRealVector,
    Constraint,
    ConstraintSpec,
    Dim,
    DimDynamic,
    DimStatic,
    DomainGridAxis,
    IRArg,
    IRArgBroadcast,
    IRArgFamilyRef,
    IRArgKernel,
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
    OverOrCodomainAxes,
    Plate,
    StructuredArgSpec,
    StructuredDataArg,
    StructuredKernelArg,
    StructuredZeroVectorArg,
    event_shape_of,
    from_constraint,
)


def _family_meta_or_raise(family: str) -> FamilyMeta:
    """Return ``FAMILY_META[family]`` or raise
    [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
    with a precise kind. Replaces bare-dict-access ``KeyError`` so
    callers up the stack see the documented exception type instead
    of an opaque dict failure."""
    meta = FAMILY_META.get(family)
    if meta is None:
        raise UnsupportedConstruct(
            "qvr-lower",
            [f"family:{family}: not in FAMILY_META registry"],
        )
    return meta


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

    def forward(self, module: Module) -> IRProgram:
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
        inputs, body = _propagate_let_plates(inputs, body)
        return IRProgram(
            name=program.name,
            inputs=inputs,
            body=body,
            cards=dict(cards),
        )

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
    ) -> tuple[IRNode, ...]:
        """Lower a tuple of program steps into IR nodes."""
        out: list[IRNode] = []
        for step in steps:
            out.append(self._lower_step(step, ctx))
        return tuple(out)

    def _lower_step(self, step: ProgramStep, ctx: _LowerCtx) -> IRNode:
        if isinstance(step, SampleStep):
            return self._lower_sample(step, ctx)
        if isinstance(step, ObserveStep):
            return self._lower_observe(step, ctx)
        if isinstance(step, MarginalizeStep):
            return self._lower_marginalize(step, ctx)
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
        meta = _family_meta_or_raise(resolved.family)
        if meta.structured_lowering is not None and (
            not step.args or meta.structured_lowering.always_apply
        ):
            return self._lower_sample_from_meta(meta, step, ctx)
        ir_args, arg_names = self._lower_args(
            meta, resolved, ctx,
            event_axes=_event_axis_names(step, ctx),
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

    def _lower_sample_from_meta(
        self, meta: FamilyMeta, step: SampleStep, ctx: _LowerCtx,
    ) -> IRSample:
        """Lower a no-args `~ Family` SampleStep using the family's
        declarative `structured_lowering` metadata.

        Walks `meta.structured_lowering.args` in order, synthesising
        the per-position :class:`IRArg` and binding the sample's
        event-axis plate via the family's
        :class:`EventAxisSource`. Replaces the per-family bespoke
        methods with one uniform path: every family that needs no-args
        lowering opts in by declaring metadata, never by adding a
        branch here.
        """
        assert meta.structured_lowering is not None
        if len(step.vars) != 1:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"sample:{meta.qvr_name}:destructuring-tuple: "
                    f"lower expects one bound name per "
                    f"{meta.qvr_name} SampleStep"
                ],
            )
        sample_name = step.vars[0]
        event_dims = _derive_event_dims(meta, step, ctx)
        ir_args = tuple(
            _build_structured_ir_arg(
                spec, sample_name, event_dims, step, ctx, meta,
            )
            for spec in meta.structured_lowering.args
        )
        arg_names = tuple(
            spec.arg_name for spec in meta.structured_lowering.args
        )
        plate = Plate(event_dims=event_dims, batch_dims=())
        constraint = _SAMPLE_CONSTRAINT_FACTORY[
            meta.structured_lowering.sample_constraint_kind
        ]()
        return IRSample(
            name=sample_name,
            family=meta.qvr_name,
            args=ir_args,
            arg_names=arg_names,
            constraint=constraint,
            plate=plate,
        )

    def _lower_observe(
        self, step: ObserveStep, ctx: _LowerCtx
    ) -> IRObserve:
        resolved = resolve_step_dist(
            step.morphism,
            step.args,
            morphisms=ctx.morphisms,
            lets=ctx.lets,
            family_registry=ctx.family_set,
            target="qvr-lower",
        )
        meta = _family_meta_or_raise(resolved.family)
        ir_args, arg_names = self._lower_args(
            meta, resolved, ctx,
            event_axes=_event_axis_names(step, ctx),
            axes_index=step.index,
            structural_args=step.args,
        )
        plate = self._build_plate(step, ctx, meta, ir_args)
        constraint = from_constraint(_resolve_support(meta, ir_args, ctx))
        return IRObserve(
            name=_observe_var(step),
            family=resolved.family,
            args=ir_args,
            arg_names=arg_names,
            constraint=constraint,
            plate=plate,
            via=step.via,
        )

    def _lower_marginalize(
        self, step: MarginalizeStep, ctx: _LowerCtx
    ) -> IRMarginalize:
        resolved = resolve_step_dist(
            step.morphism,
            step.args,
            morphisms=ctx.morphisms,
            lets=ctx.lets,
            family_registry=ctx.family_set,
            target="qvr-lower",
        )
        meta = _family_meta_or_raise(resolved.family)
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
        scope = self._lower_steps(step.scope, ctx)
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
        """Deterministic let-step.

        When the bound expression is a
        [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor] the
        result is rank-`n` over its binders' axes, so the IR plate
        carries one `DimStatic` per binder. Other expression shapes
        denote a scalar (the default `Real()` constraint, no plate
        dims). Renderers use the plate to choose the declared type
        (`array[...] real` vs `real`).
        """
        plate = _let_step_plate(step.value, ctx)
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
        list (preserves compound `DrawArgList` forms, including the
        nested `DrawArgList` that models a matrix literal). When it
        is ``None`` (e.g. the step had no explicit
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
        #
        # `structural_args` is the original step's call-site arg
        # tuple (preserves compound `DrawArgList` forms, including
        # the nested `DrawArgList` that models a matrix literal).
        # `resolved.args` is the resolver's output: when the
        # step references a morphism whose option block fills extra
        # family slots (`[scale=0.1]` -> Normal's `scale`), the
        # resolved tuple is longer than the structural tuple. We
        # keep the leading positions from the structural args (which
        # preserve compound shapes) and extend with the resolver's
        # tail so the morphism's option-derived args reach the IR.
        if structural_args is not None:
            structural_ir = tuple(
                self._raw_arg_to_ir(a, ctx) for a in structural_args
            )
            resolved_tail = (
                tuple(
                    self._raw_arg_to_ir(a, ctx)
                    for a in resolved.args[len(structural_ir):]
                )
                if resolved.args
                and len(resolved.args) > len(structural_ir)
                else ()
            )
            pre_args = structural_ir + resolved_tail
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

        Names come from the distribution's positional constructor
        signature, because that is the contract a QVR call site writes
        against: ``~ Pareto(a, b)`` binds its arguments the way
        ``torch.distributions.Pareto`` binds them. The constructor can
        differ from ``arg_constraints``, which is keyed by the
        *constrained* parameters and so both reorders (``Pareto`` is
        keyed ``alpha, scale`` but constructed ``scale, alpha``) and
        omits leading parameters that carry no constraint
        (``RelaxedBernoulli``'s ``temperature``). Reading the wrong one
        transposes or drops a slot and silently changes the density.

        `_STRUCTURAL_ARG_NAMES` overrides the families whose leading
        constructor parameter is supplied structurally by the renderer
        rather than written at the call site. When the constructor
        cannot be introspected the constrained parameters are used; if
        those are a property (Wishart, Uniform), a sentinel supplies
        the instance-level dict. The returned tuple is positional: the
        i'th entry names the i'th user-supplied arg.
        """
        names = _STRUCTURAL_ARG_NAMES.get(meta.qvr_name, ())
        if not names:
            names = _ctor_param_names(meta.distribution_class)
        if not names:
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
        if isinstance(raw, DrawArgIndex):
            return self._atom_text_to_ir(encode_index(raw), ctx)
        if isinstance(raw, DrawArgDist):
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"nested-distribution-arg:{raw.family}: a "
                    "distribution-valued draw argument is not "
                    "representable in this backend's IR"
                ],
            )
        if isinstance(raw, DrawArgList):
            if is_matrix(raw):
                return IRArgMatrix(
                    rows=tuple(
                        IRArgList(
                            elements=tuple(
                                self._raw_arg_to_ir(e, ctx) for e in row
                            )
                        )
                        for row in matrix_rows(raw)
                    )
                )
            return IRArgList(
                elements=tuple(
                    self._raw_arg_to_ir(e, ctx) for e in list_items(raw)
                )
            )
        if isinstance(raw, (int, float)):
            return IRArgNumber(value=float(raw))
        assert isinstance(raw, str), (
            f"unexpected raw arg form {type(raw).__name__!r}: {raw!r}"
        )
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

        Event axes come from (in priority order):

        1. The step's `[over=...]` `AxisSpec`.
        2. The referenced morphism's `[over=...]` option block.
        3. The morphism's codomain components (one event axis per
           named factor) when the family is multivariate and the
           codomain factor count matches the family's event rank.

        Batch axes come from the step's `[iid_over=...]` clause when
        present; otherwise from `step.index` for scalar families.
        """
        axes = step.axes
        if axes is not None:
            event_dims = tuple(self._axis_dim(a, ctx) for a in axes.over)
            batch_dims = tuple(
                self._axis_dim(a, ctx) for a in axes.iid_over
            )
            return Plate(event_dims=event_dims, batch_dims=batch_dims)
        # No AxisSpec on this step. Try the morphism-level `[over=...]`
        # option next: `morphism k : A * B -> A * B [over=[A, B]] ~ F`
        # carries the event axes on the morphism declaration when no
        # per-call axes are supplied.
        morphism_over = _morphism_over_axes(step.morphism, ctx)
        event_dim = _event_dim_of(meta, ir_args, ctx)
        if morphism_over:
            event_dims = tuple(self._axis_dim(a, ctx) for a in morphism_over)
            return Plate(event_dims=event_dims, batch_dims=())
        if event_dim > 0:
            # Multivariate family without any user-declared axes: read
            # the morphism's codomain to recover the event axis names.
            decl = ctx.morphisms.get(step.morphism)
            if decl is not None:
                codomain_axes = _codomain_axes(decl.codomain, ctx.cards)
                if len(codomain_axes) == event_dim:
                    event_dims = tuple(
                        self._axis_dim(a, ctx) for a in codomain_axes
                    )
                    return Plate(event_dims=event_dims, batch_dims=())
                if codomain_axes and event_dim == 1:
                    # Vector family whose codomain is a product of
                    # named axes: use the first axis as the event
                    # dimension (Stan vector / MultivariateNormal /
                    # LowRankMVN / GP take one event axis).
                    return Plate(
                        event_dims=(self._axis_dim(codomain_axes[0], ctx),),
                        batch_dims=(),
                    )
                # Last resort: the morphism's codomain is a single
                # `Real N` cardinality with no named axis; size the
                # event dim from the codomain's known cardinality.
                sentinel_dims = _sentinel_event_dims_from_meta(
                    meta, ir_args, ctx, decl.codomain,
                )
                if sentinel_dims is not None:
                    return Plate(
                        event_dims=sentinel_dims, batch_dims=(),
                    )
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
        """Build the [`Plate`][quivers.transpile.ir.Plate] for a
        marginalize step.

        The `over=` / `over_objs` axes are the grouping axes: one
        latent per group, so they are always `batch_dims`.

        The role of `step.index` depends on the latent's support:

        * Multivariate family (`event_dim > 0`): the index names the
          family's event axis, so it becomes an `event_dim`.
        * Scalar family with finite enumerable support at the call
          site (Categorical over `Topic`, Bernoulli, ...): the index
          names the *support* cardinality, which the integration
          sums over rather than replicates. It contributes no plate
          dim; the constraint records the support range.
        * Scalar family whose support is not finite-enumerable
          (ContinuousBernoulli, Beta, ...): nothing is summed over,
          so the index is a replication axis, one latent value per
          index. It becomes a `batch_dim`, and renderers declare the
          latent as a per-index vector rather than a scalar.
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
        if step.index is not None:
            index_dim = self._object_expr_dim(step.index, ctx)
            if event_dim > 0:
                event_dims = (index_dim,)
            elif not finite_enumerable_at_call_site(meta, ir_args):
                batch_dims = (*batch_dims, index_dim)
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
        # Inputs that flow into a known-integer position (an array
        # index in a let expression, or an integer-typed family arg
        # such as `BetaBinomial(total_count, ...)`) carry an integer
        # constraint so renderers declare them as `int` rather than
        # `real`. The rest default to `CSReal()`.
        # GP kernel-input names ride on `IRArgKernel.x_name` and need
        # a vector plate sized by the grid axis so renderers declare
        # them as `vector[N]` / `array[N] real`.
        seen_param_names = seen_param | seen_via | seen_obs
        integer_names = self._integer_typed_free_names(body)
        kernel_input_plates = self._kernel_input_plates(body)
        structured_input_specs = self._structured_input_specs(body)
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
            spec = structured_input_specs.get(name)
            if spec is not None:
                struct_constraint, struct_plate = spec
                free_inputs.append(
                    IRDataInput(
                        name=name,
                        constraint=struct_constraint,
                        plate=struct_plate,
                    )
                )
                continue
            plate = kernel_input_plates.get(
                name, Plate(event_dims=(), batch_dims=())
            )
            constraint: ConstraintSpec = (
                CSNonnegativeInteger()
                if name in integer_names
                else CSReal()
            )
            free_inputs.append(
                IRDataInput(
                    name=name,
                    constraint=constraint,
                    plate=plate,
                )
            )

        return tuple(param_inputs + via_inputs + obs_inputs + free_inputs)

    def _structured_input_specs(
        self, body: tuple[IRNode, ...],
    ) -> dict[str, tuple[ConstraintSpec, Plate]]:
        """Return ``name`` → ``(ConstraintSpec, Plate)`` for every
        per-sample data input synthesised by
        :meth:`_lower_sample_from_meta`.

        For each :class:`IRSample` whose family declares
        `structured_lowering`, walk the spec tuple in lockstep with
        the IR's ``args`` / ``arg_names``: every
        :class:`StructuredDataArg` contributes one entry whose plate
        is built by indexing the sample's event-axis tuple at the
        spec's ``axis_indices``. The shape of the data declaration
        thus flows from declarative metadata; no family-name branch
        appears here.
        """
        out: dict[str, tuple[ConstraintSpec, Plate]] = {}
        for node in _walk_nodes(body):
            if not isinstance(node, IRSample):
                continue
            meta = FAMILY_META.get(node.family)
            if meta is None or meta.structured_lowering is None:
                continue
            event_dims = node.plate.event_dims
            for spec, arg in zip(
                meta.structured_lowering.args, node.args, strict=True,
            ):
                if not isinstance(spec, StructuredDataArg):
                    continue
                if not isinstance(arg, IRArgRef):
                    raise UnsupportedConstruct(
                        "qvr-lower",
                        [
                            f"family:{node.family}:structured_arg:"
                            f"{spec.arg_name}: expected IRArgRef from "
                            f"structured lowering, got {type(arg).__name__}"
                        ],
                    )
                out_of_range = [
                    i for i in spec.axis_indices if i >= len(event_dims)
                ]
                if out_of_range:
                    raise UnsupportedConstruct(
                        "qvr-lower",
                        [
                            f"family:{node.family}:structured_arg:"
                            f"{spec.arg_name}: axis_indices "
                            f"{spec.axis_indices} reference event-dim "
                            f"positions {out_of_range} but sample plate "
                            f"has only {len(event_dims)} event dims"
                        ],
                    )
                plate_dims = tuple(
                    event_dims[i] for i in spec.axis_indices
                )
                plate = Plate(event_dims=plate_dims, batch_dims=())
                constraint = _DATA_CONSTRAINT_FACTORY[
                    spec.constraint_kind
                ]()
                out[arg.name] = (constraint, plate)
        return out

    def _kernel_input_plates(
        self, body: tuple[IRNode, ...]
    ) -> dict[str, Plate]:
        """Return ``x_name`` → `Plate` for every
        [`IRArgKernel`][quivers.transpile.ir.IRArgKernel] that appears
        in the body.

        The kernel input is a one-dimensional vector of grid
        locations; the renderer declares it with a single batch
        dim sized by the kernel's `grid_size`. Used by
        `_build_inputs` to give the GP kernel-input data declaration
        the right shape rather than defaulting to scalar.
        """
        out: dict[str, Plate] = {}
        for node in _walk_nodes(body):
            if not isinstance(node, (IRSample, IRObserve, IRMarginalize)):
                continue
            for arg in node.args:
                if not isinstance(arg, IRArgKernel):
                    continue
                if arg.x_name in out:
                    continue
                grid_dim = DimStatic(size=arg.grid_size, name="grid")
                out[arg.x_name] = Plate(
                    event_dims=(),
                    batch_dims=(grid_dim,),
                )
        return out

    def _integer_typed_free_names(
        self, body: tuple[IRNode, ...]
    ) -> set[str]:
        """Return the set of free names whose usage proves they
        must be declared as integers.

        Two evidence kinds promote a name to integer typing:

        * The name appears as an index in any `LetExprIndex.indices`
          inside an `IRDeterministic.expr`; let-expression indices
          must be integers in every target language.
        * The name appears as an `IRArgRef` argument in an
          `IRSample` or `IRObserve` whose corresponding
          `arg_name` is an integer-typed parameter in the family's
          `arg_constraints` (e.g. `BetaBinomial(total_count, ...)`).
        """
        out: set[str] = set()
        for node in _walk_nodes(body):
            if isinstance(node, IRDeterministic):
                _collect_integer_index_names(node.expr, out)
            elif isinstance(node, IRScore):
                _collect_integer_index_names(node.expr, out)
            elif isinstance(node, (IRSample, IRObserve)):
                meta = FAMILY_META.get(node.family)
                if meta is None:
                    continue
                arg_constraints = getattr(
                    meta.distribution_class, "arg_constraints", {}
                )
                for arg, arg_name in zip(
                    node.args, node.arg_names, strict=True
                ):
                    if not isinstance(arg, IRArgRef):
                        continue
                    constraint = arg_constraints.get(arg_name)
                    if constraint is None:
                        continue
                    if _is_integer_constraint(constraint):
                        out.add(arg.name)
        return out

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
        emission."""
        out: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
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


def _observe_var(step: ObserveStep) -> str:
    """The single observed variable name of an observe step.

    The transpile models one observed variable per step; a
    multi-variable observe has no single-name IR form.
    """
    if len(step.vars) != 1:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"observe:multiple-vars:{step.vars}: the transpile "
                "emits one observed variable per step"
            ],
        )
    return step.vars[0]


def object_cardinalities(module: Module) -> dict[str, int]:
    """Return name -> cardinality for every `FinSet N` object decl."""
    out: dict[str, int] = {}
    for stmt in module.statements:
        if not isinstance(stmt, ObjectDecl):
            continue
        init = stmt.init
        if isinstance(init, TypeFromExpr):
            expr = init.expr
            card: int | None = None
            if isinstance(expr, DiscreteConstructor) and expr.args:
                card = int(expr.args[0])
            elif isinstance(expr, ContinuousConstructor) and expr.args:
                # `Real D` etc.: take the first arg as the size.
                try:
                    card = int(expr.args[0])
                except ValueError:
                    card = None
            if card is not None:
                for name in stmt.names:
                    out[name] = card
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
            out[_observe_var(step)] = _step_shape(step.index, cards)
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
    if isinstance(arg, DrawArgIndex):
        return _names_in_atom_text(encode_index(arg))
    if isinstance(arg, DrawArgDist):
        dist_names: list[str] = []
        for a in arg.args:
            dist_names.extend(_names_in_raw_arg(a))
        return dist_names
    if isinstance(arg, DrawArgList):
        list_names: list[str] = []
        for item in list_items(arg):
            list_names.extend(_names_in_raw_arg(item))
        return list_names
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
        if isinstance(a, IRArgKernel):
            if a.x_name not in seen:
                seen.add(a.x_name)
                out.append(a.x_name)
            return
        if isinstance(a, IRArgNumber):
            return

    visit(arg)
    return out


_DATA_CONSTRAINT_FACTORY: dict[str, Callable[[], ConstraintSpec]] = {
    "real_matrix": CSRealMatrix,
    "real_vector": CSRealVector,
    "positive_definite": CSPositiveDefinite,
}

_SAMPLE_CONSTRAINT_FACTORY: dict[str, Callable[[], ConstraintSpec]] = {
    "real_matrix": CSRealMatrix,
    "real_vector": CSRealVector,
}


def _derive_event_dims(
    meta: FamilyMeta, step: SampleStep, ctx: "_LowerCtx",
) -> tuple[Dim, ...]:
    """Recover a family's event-axis tuple from the source declared
    on its :class:`StructuredSampleLowering`.

    :class:`OverOrCodomainAxes`: the n axes come from the step's
    `[over=...]` AxisSpec, the morphism declaration's `[over=...]`
    option block, or (last resort) the morphism's codomain factors.
    :class:`DomainGridAxis`: the single axis comes from the
    morphism's domain (a `FinSet N` object), GP-style.
    """
    assert meta.structured_lowering is not None
    src = meta.structured_lowering.event_axis_source
    if isinstance(src, OverOrCodomainAxes):
        return _n_event_dims(step, ctx, src.axis_count, meta.qvr_name)
    if isinstance(src, DomainGridAxis):
        decl = ctx.morphisms.get(step.morphism)
        if decl is None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"family:{meta.qvr_name}:morphism-unknown:"
                    f"{step.morphism}"
                ],
            )
        grid_axis_name, grid_size = _gp_grid_axis(decl.domain, ctx.cards)
        return (DimStatic(size=grid_size, name=grid_axis_name),)
    raise UnsupportedConstruct(
        "qvr-lower",
        [
            f"family:{meta.qvr_name}:event_axis_source:unknown-kind:"
            f"{src.kind!r}"
        ],
    )


def _n_event_dims(
    step: SampleStep, ctx: "_LowerCtx", n: int, family_name: str,
) -> tuple[Dim, ...]:
    """Return `n` event dims for a sample step, derived from the
    step's `[over=...]`, the morphism's `[over=...]`, or the
    morphism's codomain factors. Single-axis families also accept
    an anonymous-cardinality (`Real N`) codomain as a last resort.
    """
    axes = step.axes
    if axes is not None and len(axes.over) == n:
        return tuple(_axis_dim_at(a, ctx) for a in axes.over)
    morphism_over = _morphism_over_axes(step.morphism, ctx)
    if len(morphism_over) == n:
        return tuple(_axis_dim_at(a, ctx) for a in morphism_over)
    decl = ctx.morphisms.get(step.morphism)
    if decl is not None:
        codomain_axes = _codomain_axes(decl.codomain, ctx.cards)
        if len(codomain_axes) == n:
            return tuple(_axis_dim_at(a, ctx) for a in codomain_axes)
        if n == 1 and codomain_axes:
            return (_axis_dim_at(codomain_axes[0], ctx),)
        if n == 1:
            anon = _anon_codomain_size(decl.codomain)
            if anon is not None:
                size, base_name = anon
                return (DimStatic(size=size, name=base_name),)
    raise UnsupportedConstruct(
        "qvr-lower",
        [
            f"family:{family_name}:morphism:{step.morphism}: cannot "
            f"derive {n} event axes. Declare them via `[over=...]` on "
            f"the morphism or its call site, or give the morphism a "
            f"codomain whose factor count is {n}."
        ],
    )


def _build_structured_ir_arg(
    spec: "StructuredArgSpec",
    sample_name: str,
    event_dims: tuple[Dim, ...],
    step: SampleStep,
    ctx: "_LowerCtx",
    meta: FamilyMeta,
) -> IRArg:
    """Synthesise the :class:`IRArg` for one
    :class:`StructuredArgSpec` variant.

    :class:`StructuredDataArg` -> :class:`IRArgRef` whose name is
    ``<sample_name>_<arg_name>``; the data declaration is materialised
    later by :meth:`Lower._build_inputs` via
    :meth:`Lower._structured_input_specs`.
    :class:`StructuredZeroVectorArg` -> :class:`IRArgNumber(0.0)`.
    :class:`StructuredKernelArg` -> :class:`IRArgKernel` whose kernel
    name and length_scale come from the morphism's `[kernel=...,
    length_scale=...]` option block and whose grid_size comes from
    the sample's first event dim.
    """
    if isinstance(spec, StructuredDataArg):
        return IRArgRef(
            name=f"{sample_name}_{spec.arg_name}", indices=(),
        )
    if isinstance(spec, StructuredZeroVectorArg):
        return IRArgNumber(value=0.0)
    if isinstance(spec, StructuredKernelArg):
        decl = ctx.morphisms.get(step.morphism)
        if decl is None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"family:{meta.qvr_name}:morphism-unknown:"
                    f"{step.morphism}"
                ],
            )
        kernel_name, length_scale = _gp_kernel_options(decl.options)
        return IRArgKernel(
            kernel=kernel_name,
            length_scale=length_scale,
            x_name=spec.x_input_name,
            grid_size=event_dims[0].size,
        )
    raise UnsupportedConstruct(
        "qvr-lower",
        [
            f"family:{meta.qvr_name}:structured_arg:unknown-kind:"
            f"{spec.kind!r}"
        ],
    )


def _gp_grid_axis(
    domain: ObjectExpr, cards: dict[str, int]
) -> tuple[str, int]:
    """Return the (axis-name, cardinality) for a GP morphism's
    domain.

    A GP is realised at a finite collection of input locations
    indexed by a `FinSet N` axis (the morphism's domain). The
    helper accepts a `TypeName` domain whose cardinality is recorded
    in `cards`; raises `UnsupportedConstruct` otherwise.
    """
    if isinstance(domain, TypeName):
        size = cards.get(domain.name)
        if size is None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"family:GP:grid-axis: domain `{domain.name}` "
                    f"has no static cardinality in object table"
                ],
            )
        return domain.name, size
    raise UnsupportedConstruct(
        "qvr-lower",
        [
            f"family:GP:grid-axis: unsupported domain shape "
            f"{type(domain).__name__}; the GP grid axis must be a "
            f"single FinSet object"
        ],
    )


def _gp_kernel_options(
    options: tuple[OptionEntry, ...],
) -> tuple[str, float]:
    """Extract ``(kernel, length_scale)`` from a GP morphism's
    option block.

    Accepts ``[kernel=rbf, length_scale=1.0]``; missing entries
    raise `UnsupportedConstruct`. The kernel name is normalised to
    a lower-case identifier and must be in the registry of
    supported kernels (``"rbf"`` today).
    """
    kernel: str | None = None
    length_scale: float | None = None
    for opt in options:
        if opt.key == "kernel" and isinstance(opt.value, OptionName):
            kernel = opt.value.value.lower()
        elif opt.key == "length_scale" and isinstance(
            opt.value, OptionNumber
        ):
            length_scale = float(opt.value.value)
    if kernel is None:
        raise UnsupportedConstruct(
            "qvr-lower",
            ["family:GP:missing-option: kernel"],
        )
    if length_scale is None:
        raise UnsupportedConstruct(
            "qvr-lower",
            ["family:GP:missing-option: length_scale"],
        )
    if kernel != "rbf":
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"family:GP:unsupported-kernel:{kernel}: only `rbf` is "
                f"implemented"
            ],
        )
    if length_scale <= 0.0:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"family:GP:length_scale:{length_scale}: must be > 0"
            ],
        )
    return kernel, length_scale


def _axis_dim_at(axis_name: str, ctx: _LowerCtx) -> Dim:
    """Module-level version of
    [`Lower._axis_dim`][quivers.transpile.lower.Lower._axis_dim] so
    free-function helpers can convert an axis name into a
    [`Dim`][quivers.transpile.ir.Dim] without instantiating
    [`Lower`][quivers.transpile.lower.Lower]."""
    size = ctx.cards.get(axis_name)
    if size is None:
        return DimDynamic(size_name=f"N_{axis_name}", name=axis_name)
    return DimStatic(size=size, name=axis_name)


def _anon_codomain_size(codomain: ObjectExpr) -> tuple[int, str] | None:
    """Return ``(size, base_name)`` for an anonymous-cardinality
    codomain (`Real N`), or `None` when the codomain has no usable
    anonymous size.
    """
    if (
        isinstance(codomain, DiscreteConstructor)
        and codomain.args
        and isinstance(codomain.args[0], int)
    ):
        return int(codomain.args[0]), "event"
    if isinstance(codomain, TypeName):
        return None
    if isinstance(codomain, ObjectProduct):
        for comp in codomain.components:
            inner = _anon_codomain_size(comp)
            if inner is not None:
                return inner
    return None


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


#: Families whose leading constructor parameter is supplied by the
#: renderer from the sample's event axis rather than written at the QVR
#: call site, so it must not consume a user-supplied argument slot.
#: ``LKJCholesky(dim, concentration)`` takes its matrix dimension from
#: the codomain axis; the call site writes only the concentration.
_STRUCTURAL_ARG_NAMES: dict[str, tuple[str, ...]] = {
    "LKJCholesky": ("concentration",),
}


def _ctor_param_names(cls: type) -> tuple[str, ...]:
    """Return the positional parameter names of ``cls.__init__``.

    Drops ``self``, ``validate_args``, and any variadic parameter, so
    the result is the positional contract a call site binds against.
    Returns an empty tuple when the signature cannot be read (a C
    extension type or a shim without an introspectable ``__init__``),
    which the caller treats as "fall back to the constrained
    parameters".
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return ()
    return tuple(
        name
        for name, param in params.items()
        if name not in ("self", "validate_args")
        and param.kind
        not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    )


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
    except TypeError as exc:
        # The family requires args the user did not supply (e.g.
        # `~ Wishart` with no call-site arguments). Fall through to
        # the signature-inspection path: build a sentinel from every
        # required constructor parameter, sized by the parameter's
        # `arg_constraints` (or the class's
        # [`Distribution.support`][torch.distributions.constraints]
        # when `arg_constraints` is a property at the class level).
        instance = _construct_sentinel_from_signature(meta, sentinel_args)
        if instance is None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"family:{meta.qvr_name}:sentinel-failed: cannot "
                    f"instantiate {meta.distribution_class.__name__} "
                    f"with placeholder args derived from "
                    f"{args!r}: {exc!r}"
                ],
            ) from exc
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


def _construct_sentinel_from_signature(
    meta: FamilyMeta,
    user_args: tuple[
        torch.Tensor | torch.distributions.Distribution, ...
    ],
) -> torch.distributions.Distribution | None:
    """Construct a placeholder
    [`Distribution`][torch.distributions.Distribution] instance for
    `meta.distribution_class` by inspecting its constructor signature
    and supplying a sensible default for every required parameter not
    covered by `user_args`.

    Used as a fallback when
    [`_lookup_or_build_sentinel`][quivers.transpile.lower._lookup_or_build_sentinel]'s
    primary path (calling the constructor with `user_args` directly)
    raises `TypeError` because the user didn't supply enough
    arguments. The sentinel is only used to derive shape /
    event-dimension information downstream; its numerical values
    are arbitrary.

    Returns `None` if the signature inspection cannot construct a
    valid instance (e.g. the class has an `__init__` that requires
    backend-specific kwargs the inspector doesn't know about); the
    caller falls back to `UnsupportedConstruct`.

    Heuristics per parameter:

    * `df` / `degree_of_freedom` / `nu`: a scalar float at least
      `event_dim + 1` so any positive-definite scale-matrix
      constraint is satisfied.
    * `covariance_matrix` / `scale_matrix` / `precision_matrix` /
      `scale_tril`: a `(dim, dim)` identity matrix where `dim` is
      derived from a matrix-valued `user_args[0]` shape when
      present, or 2 by default.
    * other named parameters: skipped (the constructor's own
      defaults are used).
    """
    sig = inspect.signature(meta.distribution_class.__init__)
    bound_positional = list(user_args)
    kwargs: dict[str, torch.Tensor] = {}
    dim = _infer_sentinel_dim(user_args)
    sigparams = list(sig.parameters.values())[1:]  # drop `self`
    # Group A: every parameter the user did not supply positionally AND
    # that has no default. Pick a sensible sentinel based on the name.
    for i, param in enumerate(sigparams):
        if i < len(bound_positional):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        sentinel = _sentinel_value_for_param(param.name, dim)
        if sentinel is None:
            return None
        kwargs[param.name] = sentinel
    # Group B: parameter families where the constructor enforces
    # "exactly one of {a, b, c} must be non-None" at runtime (Wishart,
    # MultivariateNormal, InverseWishart on covariance/precision/scale).
    # Supply the canonical Cholesky-factor variant when none of the
    # trio is already bound.
    trio_group = (
        "covariance_matrix", "precision_matrix", "scale_tril",
    )
    param_names = {p.name for p in sigparams}
    bound_names = {
        p.name for p in sigparams[: len(bound_positional)]
    } | set(kwargs)
    present_in_sig = [n for n in trio_group if n in param_names]
    if (
        present_in_sig
        and not any(n in bound_names for n in present_in_sig)
    ):
        kwargs[present_in_sig[-1]] = torch.eye(dim)
    try:
        return meta.distribution_class(*bound_positional, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def _sentinel_value_for_param(
    name: str, dim: int,
) -> torch.Tensor | None:
    """Pick a sensible sentinel default for a constructor parameter
    by name. Returns `None` for parameters whose name is not
    recognised; the caller falls back to UnsupportedConstruct.
    """
    if name in ("df", "degree_of_freedom", "nu", "concentration"):
        return torch.tensor(float(dim + 1))
    if name in (
        "covariance_matrix", "scale_matrix",
        "precision_matrix", "scale_tril",
    ):
        return torch.eye(dim)
    if name in ("loc", "mean"):
        return torch.zeros(dim)
    if name in ("scale", "sigma"):
        return torch.tensor(1.0)
    if name in ("rate", "lambda"):
        return torch.tensor(1.0)
    if name in ("low", "lower"):
        return torch.tensor(0.0)
    if name in ("high", "upper"):
        return torch.tensor(1.0)
    if name in ("probs", "p"):
        return torch.full((dim,), 1.0 / max(dim, 1))
    if name in ("total_count",):
        return torch.tensor(1)
    if name == "validate_args":
        return torch.tensor(False)  # not really used; signature ask
    return None


def _infer_sentinel_dim(
    user_args: tuple[
        torch.Tensor | torch.distributions.Distribution, ...
    ],
) -> int:
    """Best-effort guess at the event dimension for a sentinel
    matrix-valued parameter. Inspects the first matrix-shaped tensor
    in `user_args`; defaults to 2 if nothing useful surfaces."""
    for arg in user_args:
        if isinstance(arg, torch.Tensor) and arg.dim() >= 2:
            return int(arg.shape[-1])
    return 2


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
) -> torch.Tensor | torch.distributions.Distribution:
    """Materialise a placeholder argument for one IR arg.

    Numeric / ref / list / matrix args produce a `torch.Tensor` of
    `expected_shape`. `IRArgFamilyRef` produces a fully-instantiated
    `torch.distributions.Distribution` so wrapper families that
    accept a base distribution (e.g. `Truncated(base, ...)`) get a
    valid first positional.
    """
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
        inner_meta = _family_meta_or_raise(decl.init_family.family)
        inner_args = tuple(
            _raw_to_ir_for_sentinel(a) for a in decl.init_family.args
        )
        return _make_sentinel(inner_meta, inner_args, ctx)
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
    if isinstance(raw, DrawArgIndex):
        return _atom_text_for_sentinel(encode_index(raw))
    if isinstance(raw, DrawArgDist):
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"nested-distribution-arg:{raw.family}: a "
                "distribution-valued draw argument is not "
                "representable in this backend's IR"
            ],
        )
    if isinstance(raw, DrawArgList):
        if is_matrix(raw):
            return IRArgMatrix(
                rows=tuple(
                    IRArgList(
                        elements=tuple(
                            _raw_to_ir_for_sentinel(e) for e in row
                        )
                    )
                    for row in matrix_rows(raw)
                )
            )
        return IRArgList(
            elements=tuple(
                _raw_to_ir_for_sentinel(e) for e in list_items(raw)
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
    ctx: _LowerCtx | None = None,
) -> tuple[str, ...]:
    """Return the event-axis names for a sample / observe step.

    Priority order:

    1. The step's own [`AxisSpec`][quivers.dsl.ast_nodes._shared.AxisSpec]
       `over` clause (`sample x <- F(args) [over=[A, B]]`).
    2. The referenced morphism's `[over=[A, B]]` option block
       (`morphism k : A * B -> A * B [over=[A, B]] ~ F`).
    3. Empty tuple (no event axes declared at any layer).
    """
    if step.axes is not None:
        return step.axes.over
    if ctx is not None:
        return _morphism_over_axes(step.morphism, ctx)
    return ()


def _morphism_over_axes(
    morphism_name: str, ctx: _LowerCtx,
) -> tuple[str, ...]:
    """Return the morphism declaration's `[over=[A, B]]` axis names.

    Reads the morphism's option block via the
    [`build_morphism_table`][quivers.transpile._resolve.build_morphism_table]
    map carried on the lowering context; returns an empty tuple when
    the morphism is absent, has no `over=` option, or the `over=`
    value is not a list-of-identifiers.
    """
    decl = ctx.morphisms.get(morphism_name)
    if decl is None:
        return ()
    for opt in decl.options:
        if opt.key != "over":
            continue
        return _option_axes_to_tuple(opt.value)
    return ()


def _option_axes_to_tuple(value: OptionValue) -> tuple[str, ...]:
    """Extract identifier names from an `[over=...]` option value.

    Accepts a single name (`over=A`) or a list of names
    (`over=[A, B]`); ignores other option-value shapes.
    """
    if isinstance(value, OptionName):
        return (value.value,)
    if isinstance(value, OptionList):
        out: list[str] = []
        for item in value.items:
            if isinstance(item, OptionName):
                out.append(item.value)
        return tuple(out)
    return ()


def _sentinel_event_dims_from_meta(
    meta: FamilyMeta,
    ir_args: tuple[IRArg, ...],
    ctx: _LowerCtx,
    codomain: ObjectExpr,
) -> tuple[Dim, ...] | None:
    """Derive a multivariate family's event dims from its sentinel
    `event_shape` plus the morphism codomain's anonymous cardinality.

    Used when the codomain has no named axes a user would write into
    `[over=...]` (e.g. `Obs : Real 9` for a `Wishart` morphism whose
    9 elements flatten a 3x3 matrix). The sentinel's `event_shape`
    is the canonical size tuple; the helper rebuilds matching
    `DimStatic` entries named after the codomain axis when present,
    or `event_axis_{i}` otherwise.

    Returns `None` when the sentinel cannot be constructed or the
    codomain provides no usable cardinality.
    """
    try:
        sentinel = _make_sentinel(meta, ir_args, ctx)
    except UnsupportedConstruct:
        return None
    event_shape = tuple(int(s) for s in sentinel.event_shape)
    if not event_shape:
        return None
    base_name = _codomain_base_name(codomain)
    dims: list[Dim] = []
    for i, size in enumerate(event_shape):
        axis_name = base_name if len(event_shape) == 1 else f"{base_name}_{i}"
        dims.append(DimStatic(size=size, name=axis_name))
    return tuple(dims)


def _codomain_base_name(codomain: ObjectExpr) -> str:
    """Return a printable name for the codomain's leading factor.

    Falls back to `"event"` when the codomain shape has no surface
    name (e.g. a `Real N` constructor with no preceding identifier).
    """
    if isinstance(codomain, TypeName):
        return codomain.name
    if isinstance(codomain, ObjectProduct):
        for comp in codomain.components:
            if isinstance(comp, TypeName):
                return comp.name
    return "event"


def _codomain_axes(
    codomain: ObjectExpr, cards: dict[str, int],
) -> tuple[str, ...]:
    """Return the named axes of a morphism codomain.

    A `TypeName(name=A)` codomain contributes a single axis `A`; a
    `ObjectProduct(components=...)` codomain contributes one axis
    per `TypeName` component, in declaration order. Returns an empty
    tuple when no component is a named axis carrying a known
    cardinality (`Real` / `Sphere` / anonymous types).
    """
    if isinstance(codomain, TypeName):
        return (codomain.name,) if codomain.name in cards else ()
    if isinstance(codomain, ObjectProduct):
        out: list[str] = []
        for comp in codomain.components:
            if isinstance(comp, TypeName) and comp.name in cards:
                out.append(comp.name)
        return tuple(out)
    return ()


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


def _let_step_plate(
    expr: LetExprNode, ctx: _LowerCtx
) -> Plate:
    """Derive the IR plate for an `IRDeterministic` whose bound
    expression is `expr`.

    A `LetExprFactor` of `n` binders produces a rank-`n` result
    whose event dimensions are the binders' static axis sizes (in
    declaration order); every other expression denotes a scalar.
    Resolving a binder's axis size walks the ctx's `cards` map
    for the bound object name.
    """
    if isinstance(expr, LetExprFactor):
        dims: list[Dim] = []
        for binder in expr.binders:
            idx = binder.index
            if isinstance(idx, TypeName):
                size = ctx.cards.get(idx.name)
                if size is None:
                    return Plate(event_dims=(), batch_dims=())
                dims.append(
                    DimStatic(size=size, name=idx.name)
                )
            else:
                return Plate(event_dims=(), batch_dims=())
        # Per-binder axes are the result's batch dimensions (the
        # tensor's prepended shape), mirroring how `IRSample` /
        # `IRObserve` plates carry per-plate axes; renderers consume
        # batch_dims to size the `array [...]` declaration prefix.
        return Plate(
            event_dims=(),
            batch_dims=tuple(dims),
        )
    return Plate(event_dims=(), batch_dims=())


def _walk_nodes(body: tuple[IRNode, ...]):
    """Yield every IRNode in `body`, descending into `IRMarginalize`
    scopes."""
    for node in body:
        yield node
        if isinstance(node, IRMarginalize):
            yield from _walk_nodes(node.scope)


def _collect_let_expr_var_names(
    expr: LetExprNode, out: set[str]
) -> None:
    """Walk `expr` collecting every leaf
    [`LetExprVar.name`][quivers.dsl.ast_nodes.LetExprVar.name]. Used
    by the plate-propagation pass to find which exogenous /
    previously-bound names a `let`-expression reads."""
    if isinstance(expr, LetExprVar):
        out.add(expr.name)
        return
    if isinstance(expr, LetExprBinOp):
        _collect_let_expr_var_names(expr.left, out)
        _collect_let_expr_var_names(expr.right, out)
        return
    if isinstance(expr, LetExprUnaryOp):
        _collect_let_expr_var_names(expr.operand, out)
        return
    if isinstance(expr, LetExprCall):
        for a in expr.args:
            _collect_let_expr_var_names(a, out)
        return
    if isinstance(expr, LetExprIndex):
        _collect_let_expr_var_names(expr.array, out)
        for i in expr.indices:
            _collect_let_expr_var_names(i, out)
        return
    if isinstance(expr, LetExprList):
        for e in expr.items:
            _collect_let_expr_var_names(e, out)
        return
    if isinstance(expr, LetExprLambda):
        _collect_let_expr_var_names(expr.body, out)
        return
    if isinstance(expr, LetExprFactor):
        for case in expr.cases:
            _collect_let_expr_var_names(case.value, out)
        if expr.body is not None:
            _collect_let_expr_var_names(expr.body, out)
        return
    if isinstance(expr, LetExprMethodCall):
        _collect_let_expr_var_names(expr.receiver, out)
        for a in expr.args:
            _collect_let_expr_var_names(a, out)
        return


def _let_expr_needs_plate(expr: LetExprNode) -> bool:
    """True iff `expr` denotes a value whose shape inherits from its
    free variables (arithmetic, index, call, method-call), false iff
    it is shape-self-determining (literal, string, list, factor,
    lambda).

    The plate-propagation pass uses this to decide whether a
    `let v = expr` may pick up the surrounding observe/sample's
    batch_dims. A `let sigma = 0.5` (literal) never inherits a
    plate; a `let mu = a + b * x_design` (arithmetic that reads a
    free name) may, when one of its free names is itself plated.
    """
    if isinstance(
        expr, (LetExprLiteral, LetExprString, LetExprList, LetExprFactor,
               LetExprLambda)
    ):
        return False
    return True


def _propagate_let_plates(
    inputs: tuple[IRDataInput, ...],
    body: tuple[IRNode, ...],
) -> tuple[tuple[IRDataInput, ...], tuple[IRNode, ...]]:
    """Shape-inference fixpoint: promote scalar
    [`IRDataInput`][quivers.transpile.ir.IRDataInput] /
    [`IRDeterministic`][quivers.transpile.ir.IRDeterministic] plates
    that are inferable from a plated consumer.

    A free-name input declared as scalar but referenced by a let-bound
    `mu` that flows into ``observe y : Obs <- Normal(mu, ...)`` must
    actually be a vector over ``Obs``; otherwise the emitted target
    code declares the input as a scalar and the runtime trips on the
    shape mismatch at data binding. Same logic for an
    `IRDeterministic` (a `let`) whose expression reads such an input:
    the let result inherits the input's plate, and any downstream
    `let` reading it propagates further.

    Promotion rules (the discriminator that keeps the pass from
    over-promoting):

    * [`IRDataInput`][quivers.transpile.ir.IRDataInput]: scalar
      plate is *always* a guess (no `[over=...]` axes attach to a
      free input), so the consumer's `batch_dims` is the best
      inferred shape. Promote unconditionally.
    * [`IRDeterministic`][quivers.transpile.ir.IRDeterministic]
      with an expression that
      [`_let_expr_needs_plate`][quivers.transpile.lower._let_expr_needs_plate]
      classifies as shape-inheriting (arithmetic, index, call):
      promote, then recurse into the expression's free names so
      transitively-referenced inputs / lets also pick up the plate.
    * [`IRDeterministic`][quivers.transpile.ir.IRDeterministic]
      with a literal / string / list / factor / lambda RHS:
      shape-self-determining; *never* promote. ``let sigma = 0.5``
      stays scalar even when it flows into an
      ``Obs``-plated `Normal(mu, sigma)`; Stan's `normal_lpdf`
      broadcasts the scalar across the observation loop natively.
    * [`IRSample`][quivers.transpile.ir.IRSample]: plate is declared
      by `<- Family(...) [over=...]`; never inferred. Never promote.

    Plated consumers are `IRSample`, `IRObserve`, and
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]: a
    marginalize latent replicated over a batch axis conditions on
    args that must carry the same axis, so its `batch_dims` drive
    promotion exactly as a sample's do.

    Iterate to fixpoint so a chain like
    ``observe y : Obs <- f(g) ; let g = h * x_design`` promotes
    ``g`` (arithmetic, free) → ``x_design`` (data input, free) in
    the same pass.

    Lets bound inside an
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] scope
    participate on the same footing as top-level lets: the scope body
    runs once per latent value, so a let reading the plated latent
    (``let gated_rate = z * rate``) carries the latent's batch axis
    and propagates it back out to every name it reads.

    Idempotent: returns inputs/body unchanged on the second call.
    """
    lets_by_name: dict[str, IRDeterministic] = {}
    sample_event_dims: dict[str, tuple[Dim, ...]] = {}
    for node in _walk_nodes(body):
        if isinstance(node, IRDeterministic):
            lets_by_name[node.name] = node
        if isinstance(node, IRSample):
            sample_event_dims[node.name] = node.plate.event_dims
    input_map: dict[str, int] = {
        inp.name: i for i, inp in enumerate(inputs)
    }
    inputs_list = list(inputs)

    def _promote_input(name: str, batch_dims: tuple[Dim, ...]) -> bool:
        idx = input_map.get(name)
        if idx is None:
            return False
        cur = inputs_list[idx]
        if cur.plate.batch_dims:
            return False
        inputs_list[idx] = IRDataInput(
            name=cur.name,
            constraint=cur.constraint,
            plate=Plate(
                event_dims=cur.plate.event_dims, batch_dims=batch_dims,
            ),
        )
        return True

    def _promote_let(name: str, batch_dims: tuple[Dim, ...]) -> bool:
        cur = lets_by_name.get(name)
        if cur is None:
            return False
        if cur.plate.batch_dims:
            return False
        if not _let_expr_needs_plate(cur.expr):
            return False
        # Inherit event_dims from a gathered source: `let g = A[i]`
        # where `A` is an IRSample with event_dims (e.g. `array[User]
        # vector[LatentDim] U_mat` gathered by an integer index `u_idx`
        # of batch shape `Rating`) produces a `array[Rating]
        # vector[LatentDim] g`. The renderer otherwise emits `array
        # [Rating] real g` and stanc rejects the vector-to-real
        # assignment.
        inherited_event_dims = cur.plate.event_dims
        if not inherited_event_dims and isinstance(cur.expr, LetExprIndex):
            arr = cur.expr.array
            if isinstance(arr, LetExprVar):
                src_event = sample_event_dims.get(arr.name, ())
                if src_event:
                    inherited_event_dims = src_event
        lets_by_name[name] = IRDeterministic(
            name=cur.name,
            expr=cur.expr,
            constraint=cur.constraint,
            plate=Plate(
                event_dims=inherited_event_dims, batch_dims=batch_dims,
            ),
        )
        # Recurse into the let-expression's free names so any
        # transitively-referenced data input / let also picks up
        # the plate. IRSample names are left alone (their plate is
        # declared, never inferred).
        leaves: set[str] = set()
        _collect_let_expr_var_names(cur.expr, leaves)
        for leaf in leaves:
            _promote_input(leaf, batch_dims)
            _promote_let(leaf, batch_dims)
        return True

    changed = True
    while changed:
        changed = False
        for node in _walk_nodes(body):
            if not isinstance(node, (IRSample, IRObserve, IRMarginalize)):
                continue
            if not node.plate.batch_dims:
                continue
            for arg in node.args:
                if not isinstance(arg, IRArgRef):
                    continue
                if _promote_let(arg.name, node.plate.batch_dims):
                    changed = True

    return tuple(inputs_list), _rebuild_with_lets(body, lets_by_name)


def _rebuild_with_lets(
    body: tuple[IRNode, ...],
    lets_by_name: dict[str, IRDeterministic],
) -> tuple[IRNode, ...]:
    """Rebuild `body`, substituting each
    [`IRDeterministic`][quivers.transpile.ir.IRDeterministic] with the
    promoted node of the same name and descending into
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] scopes."""
    out: list[IRNode] = []
    for node in body:
        if isinstance(node, IRDeterministic):
            out.append(lets_by_name.get(node.name, node))
            continue
        if isinstance(node, IRMarginalize):
            out.append(
                node.with_(
                    scope=_rebuild_with_lets(node.scope, lets_by_name),
                )
            )
            continue
        out.append(node)
    return tuple(out)


def _collect_integer_index_names(
    expr: LetExprNode, out: set[str]
) -> None:
    """Walk `expr` and add to `out` every `LetExprVar.name` that
    appears as an index in a `LetExprIndex`. Used during data-input
    typing inference to promote integer-indexed names from `real`
    to a nonnegative-integer constraint.
    """
    if isinstance(expr, LetExprIndex):
        _collect_integer_index_names(expr.array, out)
        for idx in expr.indices:
            if isinstance(idx, LetExprVar):
                out.add(idx.name)
            _collect_integer_index_names(idx, out)
        return
    if isinstance(expr, LetExprBinOp):
        _collect_integer_index_names(expr.left, out)
        _collect_integer_index_names(expr.right, out)
        return
    if isinstance(expr, LetExprUnaryOp):
        _collect_integer_index_names(expr.operand, out)
        return
    if isinstance(expr, LetExprCall):
        for a in expr.args:
            _collect_integer_index_names(a, out)
        return
    if isinstance(expr, LetExprList):
        for item in expr.items:
            _collect_integer_index_names(item, out)
        return
    if isinstance(expr, LetExprLambda):
        _collect_integer_index_names(expr.body, out)
        return
    if isinstance(expr, LetExprFactor):
        if expr.body is not None:
            _collect_integer_index_names(expr.body, out)
        for c in expr.cases:
            _collect_integer_index_names(c.value, out)
        return
    if isinstance(expr, LetExprMethodCall):
        _collect_integer_index_names(expr.receiver, out)
        for a in expr.args:
            _collect_integer_index_names(a, out)
        return


def _is_integer_constraint(constraint: object) -> bool:
    """Return True iff `constraint` is one of torch's integer-typed
    parameter constraints. Used to decide whether a `IRArgRef`
    target should be declared as an integer data input.
    """
    return constraint in (
        c.nonnegative_integer,
        c.positive_integer,
        c.integer_interval,
    )


__all__ = [
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
