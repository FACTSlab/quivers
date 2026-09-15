"""The target plan of a program, derived from its checked kernel module.

A program is one named computation of the checked
[`QiecModule`][quivers.qiec.QiecModule], performing ``Random.sample`` and
``Score.add`` on the module's canonical instances, binding pure values,
and calling its marginalization helpers. A probabilistic-programming
target has statements for exactly those things: a draw, an observation,
a deterministic binding, a log-density increment, and an integration
scope. The plan pass reads the computation and recognizes each of them,
producing the [`IRProgram`][quivers.transpile.ir.IRProgram] body the
renderers consume: one [`IRSample`][quivers.transpile.ir.IRSample] per
``Random.sample`` request, one [`IRObserve`][quivers.transpile.ir.IRObserve]
per scored density at an observation, one
[`IRDeterministic`][quivers.transpile.ir.IRDeterministic] per pure binding,
one [`IRScore`][quivers.transpile.ir.IRScore] per other weight, and one
[`IRMarginalize`][quivers.transpile.ir.IRMarginalize] per helper call whose
answer is scored. The plan's inputs are the entry point's parameters,
typed by the module and constrained by how the body reads them.

The pass reads the source declarations only for what the module does
not carry: object cardinalities and bounds, the morphism a site draws
through (which fixes a class-index draw's alphabet and a structured
family's wire form), and the option blocks of structured families. A
construct the target vocabulary has no statement for, a call of a
deduction or a recurrence, is reported with the kind of the request or
call the plan cannot place, so a renderer never receives a body with a
hole in it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Literal

from torch.distributions import Distribution

from quivers.dsl.ast_nodes import (
    Expr,
    MarginalizeStep,
    Module,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    SampleStep,
    TypeName,
)
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprList,
    LetExprLiteral,
    LetExprNode,
    LetExprString,
    LetExprUnaryOp,
    LetExprVar,
    LetFactorBinder,
    LetFactorCase,
)
from quivers.transpile._resolve import build_let_table, build_morphism_table
from quivers.qiec import QiecModule, validate_module
from quivers.qiec.module import program_computations
from quivers.qiec.effects import EffectRequest
from quivers.qiec.module import NamedComputation
from quivers.qiec.programs import ProgramEntry, ProgramParameter
from quivers.qiec.terms import (
    AffineMap,
    Bind,
    Call,
    Comprehension,
    Computation,
    DistributionValue,
    Gather,
    Handle,
    KernelMatrix,
    LiteralValue,
    Local,
    LogDensity,
    NewInstance,
    Perform,
    PlateAxis,
    PlateShape,
    PrimitiveApplication,
    Reduction,
    Return,
    Rowwise,
    SegmentSum,
    SiteValue,
    TableMap,
    TensorValue,
    TupleValue,
    Value,
    Var,
    WeightSum,
)
from quivers.dsl.program_elaboration import GAP_CODE
from quivers.dsl.pure_builtins import PRIMITIVE_BUILTINS, PRIMITIVE_OPERATORS
from quivers.dsl.qiec_lowering import (
    QiecDiagnosticError,
    has_qiec_surface,
    lower_qvr_to_qiec,
)
from quivers.qiec.canonical import tensor_shape
from quivers.qiec.types import (
    INT,
    IndexLiteral,
    IndexTerm,
    IndexVariable,
    TypeExpr,
    render_static,
)
import didactic.api as dx

from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta, class_index_outcome
from quivers.transpile.ir import (
    CSIntegerInterval,
    CSNonnegativeInteger,
    CSPositive,
    CSReal,
    ConstraintSpec,
    Dim,
    DimDynamic,
    DimStatic,
    IRArg,
    IRArgBroadcast,
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
    LetAffineSource,
    LetExprAffineMap,
    Plate,
    StructuredDataArg,
    StructuredZeroVectorArg,
    from_constraint,
)
from quivers.transpile.lower import (
    _DATA_CONSTRAINT_FACTORY,
    _LowerCtx,
    _ParamHead,
    _apply_declared_bounds,
    _arg_names_for,
    _codomain_alphabet,
    _wrap_for_constraint,
    _floor_expr,
    _head_binding_name,
    _head_raw_binding_name,
    _resolve_arg_constraints,
    _resolve_support,
    _retarget_alphabet_arg,
    object_shapes,
    pick_program,
)
from quivers.transpile.qiec_ir import lower_qiec_ir

#: The registry's name for a structured family's constructor argument
#: where the two differ.
_REGISTRY_NAMES: dict[str, dict[str, str]] = {"GP": {"covariance_matrix": "kernel"}}

#: Primitives that are conversions the target's arithmetic performs on
#: its own, so they read back as their operand.
_TRANSPARENT: frozenset[str] = frozenset({"int_to_real", "as_weight", "weight_value"})

#: Unary primitives the expression language writes as operators.
_UNARY_OPERATORS: dict[str, Literal["-", "not"]] = {
    "neg_int": "-",
    "neg_real": "-",
    "not": "not",
}


class _Planner:
    """Derive one program's plan from the checked module.

    Parameters
    ----------
    module : QiecModule
        The checked module.
    source : Module
        The expanded source, read for object cardinalities and bounds,
        morphism declarations, and the morphism each site draws
        through.
    target : str
        The transpile target the plan serves, named in diagnostics.
    """

    def __init__(self, module: QiecModule, source: Module, target: str) -> None:
        self.module = module
        self.source = source
        self.target = target
        self.computations = {item.name: item for item in module.computations}
        self.entries = {entry.name: entry for entry in module.entries}
        shapes = object_shapes(source)
        self.cards = {
            name: shape.extent
            for name, shape in shapes.items()
            if shape.extent is not None
        }
        self.morphisms = build_morphism_table(source)
        self.lets: dict[str, Expr] = build_let_table(source)
        self.shapes = shapes
        self.sentinel_cache: dict[tuple[str, tuple[str, ...]], Distribution] = {}
        self.marginal_names = {
            name
            for name in self.computations
            if name.endswith("_marginal") and "__" in name
        }

    # ------------------------------------------------------------------
    # entry

    def plan(self, program: ProgramDecl) -> IRProgram:
        """The plan of one program.

        Parameters
        ----------
        program : ProgramDecl
            The program's source declaration, which the expanded
            module's steps and the plan's name are read from.

        Returns
        -------
        IRProgram
            The plan: the entry point's inputs, the body derived from
            its computation, the object cardinalities, and the module.

        Raises
        ------
        UnsupportedConstruct
            If the module records no entry for the program, or the
            body performs a request or call the target vocabulary has
            no statement for.
        """
        entry = self.entries.get(program.name)
        computation = self.computations.get(program.name)
        if entry is None or computation is None:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"program:unelaborated:{program.name}"],
            )
        ctx = _LowerCtx(
            morphisms=self.morphisms,
            lets=self.lets,
            cards=dict(self.cards),
            family_set=frozenset(FAMILY_META),
            sentinel_cache=self.sentinel_cache,
            program=program,
            real_widths={
                name: shape.real_width
                for name, shape in self.shapes.items()
                if shape.real_width is not None
            },
            bounds={
                name: shape.bounds
                for name, shape in self.shapes.items()
                if shape.bounds.is_bounded
            },
            shapes=self.shapes,
        )
        walk = _ProgramWalk(self, entry, computation, program, ctx)
        body = walk.body()
        inputs = walk.inputs(body)
        return IRProgram(
            name=program.name,
            inputs=inputs,
            body=body,
            module=lower_qiec_ir(self.module),
            cards=dict(self.cards),
        )


class _ProgramWalk:
    """The derivation of one program's body and inputs.

    Parameters
    ----------
    planner : _Planner
        The planner, holding the module and the declaration tables.
    entry : ProgramEntry
        The program's entry point.
    computation : NamedComputation
        The program's computation.
    program : ProgramDecl
        The program's expanded source declaration.
    ctx : _LowerCtx
        The support tables the constraints are resolved against.
    """

    def __init__(
        self,
        planner: _Planner,
        entry: ProgramEntry,
        computation: NamedComputation,
        program: ProgramDecl,
        ctx: _LowerCtx,
    ) -> None:
        self.planner = planner
        self.entry = entry
        self.computation = computation
        self.program = program
        self.ctx = ctx
        self.site_steps: dict[str, SampleStep | ObserveStep | MarginalizeStep] = {}
        _collect_site_steps(program.draws, self.site_steps)
        self.observed: dict[str, IRObserve] = {}
        self.structured_inputs: dict[str, tuple[ConstraintSpec, Plate]] = {}
        self.param_map_inputs: dict[str, Plate] = {}
        self.bound_plates = ctx.bound_plates
        self.head_nodes: list[IRNode] = []
        self.axes: dict[str, tuple[str, ...]] = {
            parameter.name: parameter.axes for parameter in entry.parameters
        }
        self.local_types: dict[str, TypeExpr] = {}
        self.via: str | None = None

    # ------------------------------------------------------------------
    # body

    def body(self) -> tuple[IRNode, ...]:
        """The plan's body.

        Returns
        -------
        tuple[IRNode, ...]
            One node per statement the computation performs, in order.
        """
        return tuple(self._nodes(self.computation.body))

    def _nodes(self, computation: Computation) -> Iterator[IRNode]:
        """The nodes of a computation, one per bound statement.

        Parameters
        ----------
        computation : Computation
            A chain of binds ending in a return.

        Yields
        ------
        IRNode
            The statements in order.

        Raises
        ------
        UnsupportedConstruct
            If a computation form is not a statement of the target
            vocabulary.
        """
        node: Computation = computation
        pending_marginal: dict[str, IRMarginalize] = {}
        emitted: list[IRNode] = []
        while isinstance(node, Bind):
            first = node.first
            binder = node.binder
            self.local_types[binder.name] = binder.type
            if isinstance(first, Perform):
                request = first.request
                if request.effect.name == "Random":
                    emitted.extend(self._sample(request, binder))
                elif request.effect.name == "Score":
                    emitted.extend(
                        self._score(request, binder, pending_marginal, emitted)
                    )
                else:
                    raise UnsupportedConstruct(
                        f"qvr-{self.planner.target}",
                        [f"qiec:effect:{request.effect.name}:{self.program.name}"],
                    )
            elif isinstance(first, Return):
                emitted.extend(self._deterministic(binder, first.value))
            elif isinstance(first, Call):
                if first.name in self.planner.marginal_names:
                    pending_marginal[binder.name] = self._marginalize(first)
                    emitted.extend(self._flush_heads())
                else:
                    raise UnsupportedConstruct(
                        f"qvr-{self.planner.target}",
                        [f"qiec:call:{first.name}:{self.program.name}"],
                    )
            else:
                raise UnsupportedConstruct(
                    f"qvr-{self.planner.target}",
                    [f"qiec:computation:{type(first).__name__}:{self.program.name}"],
                )
            node = node.then
        if isinstance(node, Return):
            names = _returned_names(node.value)
            if names is not None:
                emitted.append(IRReturn(names=names))
                yield from emitted
                return
        raise UnsupportedConstruct(
            f"qvr-{self.planner.target}",
            [f"qiec:computation:{type(node).__name__}:{self.program.name}"],
        )

    # ------------------------------------------------------------------
    # draws

    def _sample(self, request: EffectRequest, binder: Local) -> Iterator[IRNode]:
        """The node of a ``Random.sample`` request.

        Parameters
        ----------
        request : EffectRequest
            The request, carrying the site and its sampleable.
        binder : Local
            The local the draw binds.

        Yields
        ------
        IRNode
            The head bindings of a parameter map, then the sample.
        """
        site, distribution = _site_and_distribution(request, self.planner.target)
        name = site.label
        family, args, arg_names, plate, constraint = self._distribution(
            distribution, name
        )
        yield from self._flush_heads()
        node = IRSample(
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            constraint=constraint,
            plate=plate,
        )
        self.bound_plates[name] = plate
        self.ctx.bound_kinds[name] = "sample"
        self.axes[name] = _dim_names(plate)
        if binder.name != name:
            self.bound_plates[binder.name] = plate
            self.axes[binder.name] = self.axes[name]
        yield node

    def _score(
        self,
        request: EffectRequest,
        binder: Local,
        pending_marginal: dict[str, IRMarginalize],
        emitted: list[IRNode],
    ) -> Iterator[IRNode]:
        """The node of a ``Score.add`` request.

        Parameters
        ----------
        request : EffectRequest
            The request, carrying the weight added.
        binder : Local
            The local the unit answer binds.
        pending_marginal : dict[str, IRMarginalize]
            The marginalization helpers called so far, by the local
            their answer is bound to.
        emitted : list[IRNode]
            The nodes emitted so far, whose last binding a score of
            that binding's value absorbs: ``score p = e`` is one
            statement binding ``p`` and adding it.

        Yields
        ------
        IRNode
            An observation, a marginalization scope, or a score.
        """
        (weight,) = request.arguments
        scored = weight
        if isinstance(scored, PrimitiveApplication) and scored.name in _TRANSPARENT:
            scored = scored.arguments[0]
        if (
            isinstance(scored, Var)
            and emitted
            and isinstance(emitted[-1], IRDeterministic)
            and emitted[-1].name == scored.local.name
        ):
            bound = emitted.pop()
            assert isinstance(bound, IRDeterministic)
            yield IRScore(name=bound.name, expr=bound.expr)
            return
        via: str | None = None
        density = weight
        if isinstance(density, WeightSum):
            density = density.value
        if isinstance(density, SegmentSum):
            index = density.index
            if not isinstance(index, Var):
                raise UnsupportedConstruct(
                    f"qvr-{self.planner.target}",
                    [f"qiec:fibration:{self.program.name}"],
                )
            via = index.local.name
            density = density.value
        if isinstance(density, LogDensity) and isinstance(density.value, Var):
            observation = density.value.local.name
            if isinstance(density.sampleable, DistributionValue):
                family, args, arg_names, plate, constraint = self._distribution(
                    density.sampleable, observation, via=via
                )
                yield from self._flush_heads()
                node = IRObserve(
                    name=observation,
                    family=family,
                    args=args,
                    arg_names=arg_names,
                    constraint=constraint,
                    plate=plate,
                    via=via,
                )
                self.observed[observation] = node
                self.bound_plates[observation] = plate
                yield node
                return
        if isinstance(weight, Var) and weight.local.name in pending_marginal:
            yield pending_marginal.pop(weight.local.name)
            return
        yield IRScore(name=binder.name, expr=self._expr(weight))

    def _deterministic(self, binder: Local, value: Value) -> Iterator[IRNode]:
        """The node of a pure binding.

        Parameters
        ----------
        binder : Local
            The bound local.
        value : Value
            The value bound.

        Yields
        ------
        IRNode
            The deterministic binding.
        """
        axes = self._inferred_axes(binder.type, value)
        self.axes[binder.name] = axes
        plate = self._inferred_plate(binder.type, axes, value)
        self.bound_plates[binder.name] = plate
        self.ctx.bound_kinds[binder.name] = "deterministic"
        yield IRDeterministic(
            name=binder.name,
            expr=self._expr(value),
            constraint=CSReal(),
            plate=plate,
        )

    def _inferred_axes(self, type_: TypeExpr, value: Value) -> tuple[str, ...]:
        """The axis names of a bound value.

        Parameters
        ----------
        type_ : TypeExpr
            The value's type.
        value : Value
            The value, whose bindings' axes name its own when their
            extents agree.

        Returns
        -------
        tuple[str, ...]
            Per dimension: the name a referenced binding of the same
            extents gives it, else the one declared object of that
            cardinality, else the position.
        """
        shape = tensor_shape(type_)
        if shape is None:
            return ()
        wanted = tuple(shape[1])
        for name in _referenced(value):
            known = self.axes.get(name)
            if known is None or len(known) < len(wanted):
                continue
            local_type = self._type_of(name)
            local_shape = tensor_shape(local_type) if local_type is not None else None
            if local_shape is None:
                continue
            offset = len(local_shape[1]) - len(wanted)
            if offset >= 0 and tuple(local_shape[1][offset:]) == wanted:
                return tuple(known[offset:])
        names: list[str] = []
        for index, size in enumerate(wanted):
            if isinstance(size, IndexVariable):
                names.append(size.name)
                continue
            matching = [
                object_name
                for object_name, card in self.ctx.cards.items()
                if isinstance(size, IndexLiteral) and card == size.value
            ]
            names.append(matching[0] if len(matching) == 1 else f"axis{index}")
        return tuple(names)

    def _inferred_plate(
        self, type_: TypeExpr, axes: tuple[str, ...], value: Value
    ) -> Plate:
        """The plate of a bound value.

        A value read off a binding with event dimensions keeps them
        as its own trailing event, ``Z[i]`` being one row of the
        event-shaped ``Z``; every other dimension is a batch.

        Parameters
        ----------
        type_ : TypeExpr
            The value's type.
        axes : tuple[str, ...]
            The dimensions' names.
        value : Value
            The value.

        Returns
        -------
        Plate
            The dimensions split into batch and event.
        """
        dims = _typed_dims(type_, axes)
        for name in _referenced(value):
            known = self.bound_plates.get(name)
            if known is None or not known.event_dims:
                continue
            count = len(known.event_dims)
            if len(dims) >= count and tuple(dims[len(dims) - count :]) == tuple(
                known.event_dims
            ):
                return Plate(
                    event_dims=tuple(dims[len(dims) - count :]),
                    batch_dims=tuple(dims[: len(dims) - count]),
                )
        return Plate(event_dims=(), batch_dims=dims)

    def _type_of(self, name: str) -> TypeExpr | None:
        """The type of a bound name.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        TypeExpr | None
            The entry parameter's or the computation local's type.
        """
        for parameter in self.entry.parameters:
            if parameter.name == name:
                return parameter.type
        return self.local_types.get(name)

    def _marginalize(self, call: Call) -> IRMarginalize:
        """The scope a marginalization helper's call denotes.

        Parameters
        ----------
        call : Call
            The call of the helper.

        Returns
        -------
        IRMarginalize
            The latent's draw and the scope's statements.

        Raises
        ------
        UnsupportedConstruct
            If the helper's body is not one enumerated draw over a
            collected scope, or reduces other than by log-sum-exp.
        """
        helper = self.planner.computations[call.name]
        handler_names = {
            handler.id: handler.name for handler in self.planner.module.handlers
        }
        body = helper.body
        # The helper allocates a Random instance for the latent, handles
        # it with the enumeration handler, draws the latent, allocates
        # a Weight instance, handles it with the collecting handler, and
        # scores the scope inside.
        if not isinstance(body, NewInstance):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}", [f"marginalize:shape:{call.name}"]
            )
        handled = body.body
        if not isinstance(handled, Handle):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}", [f"marginalize:shape:{call.name}"]
            )
        enumeration = handler_names.get(handled.handler, "")
        reduction = _marginal_reduction(enumeration)
        if reduction != "logsumexp":
            raise UnsupportedConstruct(
                "qvr-lower",
                [f"marginalize:reduction:{reduction}"],
            )
        draw = handled.computation
        if not (
            isinstance(draw, Bind)
            and isinstance(draw.first, Perform)
            and draw.first.request.effect.name == "Random"
        ):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}", [f"marginalize:shape:{call.name}"]
            )
        site, distribution = _site_and_distribution(
            draw.first.request, self.planner.target
        )
        latent = site.label
        family, args, arg_names, plate, constraint = self._distribution(
            distribution, latent, marginalized=True
        )
        self.bound_plates[latent] = plate
        self.ctx.bound_kinds[latent] = "marginalize"
        self.axes[latent] = _dim_names(plate)
        self.local_types[latent] = draw.binder.type
        collected = draw.then
        if not isinstance(collected, NewInstance) or not isinstance(
            collected.body, Handle
        ):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}", [f"marginalize:shape:{call.name}"]
            )
        scope = tuple(self._scope_nodes(collected.body.computation))
        node = IRMarginalize(
            latent=latent,
            family=family,
            args=args,
            arg_names=arg_names,
            constraint=constraint,
            plate=plate,
            reduction="logsumexp",
            scope=scope,
        )
        return node

    def _scope_nodes(self, computation: Computation) -> Iterator[IRNode]:
        """The statements of a marginalization scope.

        Parameters
        ----------
        computation : Computation
            The scope's computation, whose weights are added on the
            scope's ``Weight`` instance.

        Yields
        ------
        IRNode
            The scope's statements, without its unit return.
        """
        node: Computation = computation
        pending: dict[str, IRMarginalize] = {}
        emitted: list[IRNode] = []
        while isinstance(node, Bind):
            first = node.first
            self.local_types[node.binder.name] = node.binder.type
            if isinstance(first, Perform):
                if first.request.effect.name == "Random":
                    emitted.extend(self._sample(first.request, node.binder))
                elif first.request.effect.name in ("Weight", "Score"):
                    emitted.extend(
                        self._score(first.request, node.binder, pending, emitted)
                    )
                else:
                    raise UnsupportedConstruct(
                        f"qvr-{self.planner.target}",
                        [
                            f"qiec:effect:{first.request.effect.name}:{self.program.name}"
                        ],
                    )
            elif isinstance(first, Return):
                emitted.extend(self._deterministic(node.binder, first.value))
            elif isinstance(first, Call) and first.name in self.planner.marginal_names:
                pending[node.binder.name] = self._marginalize(first)
                emitted.extend(self._flush_heads())
            else:
                raise UnsupportedConstruct(
                    f"qvr-{self.planner.target}",
                    [f"qiec:computation:{type(first).__name__}:{self.program.name}"],
                )
            node = node.then
        if not isinstance(node, Return):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [f"qiec:computation:{type(node).__name__}:{self.program.name}"],
            )
        yield from emitted

    def _flush_heads(self) -> Iterator[IRNode]:
        """The head bindings queued by the last distribution read.

        Yields
        ------
        IRNode
            The deterministic bindings, in order.
        """
        nodes = tuple(self.head_nodes)
        self.head_nodes.clear()
        yield from nodes

    # ------------------------------------------------------------------
    # distributions

    def _distribution(
        self,
        distribution: DistributionValue,
        site: str,
        *,
        marginalized: bool = False,
        via: str | None = None,
    ) -> tuple[str, tuple[IRArg, ...], tuple[str, ...], Plate, ConstraintSpec]:
        """Read a construction as a family call.

        Parameters
        ----------
        distribution : DistributionValue
            The construction.
        site : str
            The site drawn or scored, which names the head bindings of
            a mapped kernel.
        marginalized : bool
            Whether the site is a marginalized latent, whose plate
            keeps only the grouping axes.
        via : str | None
            The fibration an observation reads its arguments through;
            the plan states it on the observation rather than on the
            arguments, so a gather by it is stripped.

        Returns
        -------
        tuple[str, tuple[IRArg, ...], tuple[str, ...], Plate, ConstraintSpec]
            The family, its arguments and their names, the plate, and
            the support.

        Raises
        ------
        UnsupportedConstruct
            If the family is outside the target vocabulary or an
            argument has no wire form.
        """
        family = distribution.name
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [f"family:{family}: not in FAMILY_META registry"],
            )
        step = self.site_steps.get(site)
        morphism = step.morphism if step is not None else ""
        plate = _plate_of(distribution.plate, marginalized, meta, distribution, self)
        named = {name: value for name, value in distribution.arguments}
        self.via = via
        if meta.structured_lowering is not None and _structured(distribution):
            args, arg_names = self._structured_args(distribution, site, plate)
        else:
            args, arg_names = self._arguments(distribution, site, plate, meta, morphism)
        self.via = None
        constraint = _apply_declared_bounds(
            from_constraint(_resolve_support(meta, args, self.ctx)),
            plate,
            self.ctx,
            site,
        )
        args, constraint = self._class_index(
            morphism, meta, args, arg_names, constraint, site
        )
        del named
        return family, args, arg_names, plate, constraint

    def _arguments(
        self,
        distribution: DistributionValue,
        site: str,
        plate: Plate,
        meta: FamilyMeta,
        morphism: str,
    ) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
        """The wire arguments of a family call.

        Parameters
        ----------
        distribution : DistributionValue
            The construction.
        site : str
            The site the call is at.
        plate : Plate
            The site's plate.
        meta : FamilyMeta
            The family's transpile metadata.
        morphism : str
            The morphism the site draws through, if any.

        Returns
        -------
        tuple[tuple[IRArg, ...], tuple[str, ...]]
            The arguments in the family's constructor order and their
            names.
        """
        pre: list[IRArg] = []
        names: list[str] = []
        for name, value in distribution.arguments:
            pre.append(self._arg(value, site, name, plate, morphism))
            names.append(name)
        ordered = _arg_names_for(meta, tuple(pre), self.ctx)
        if set(ordered) >= set(names) and len(ordered) >= len(names):
            order = {name: index for index, name in enumerate(ordered)}
            pairs = sorted(
                zip(names, pre, strict=True), key=lambda item: order[item[0]]
            )
            names = [name for name, _ in pairs]
            pre = [arg for _, arg in pairs]
        constraints = _resolve_arg_constraints(meta, tuple(pre), self.ctx)
        event_axes = tuple(
            dim.name for dim in plate.event_dims if isinstance(dim, DimStatic)
        )
        step = self.site_steps.get(site)
        axes_index = step.index if step is not None else None
        out = tuple(
            _wrap_for_constraint(
                arg, constraints.get(name), event_axes, axes_index, self.ctx
            )
            for arg, name in zip(pre, names, strict=True)
        )
        return out, tuple(names)

    def _arg(
        self, value: Value, site: str, head: str, plate: Plate, morphism: str
    ) -> IRArg:
        """The wire form of one family argument.

        Parameters
        ----------
        value : Value
            The argument.
        site : str
            The site the call is at.
        head : str
            The family parameter the argument fills.
        plate : Plate
            The site's plate.
        morphism : str
            The morphism the site draws through, if any.

        Returns
        -------
        IRArg
            A literal, a reference, a list, a matrix, a broadcast, a
            kernel, or a reference to a head binding the argument is
            computed by.

        Raises
        ------
        UnsupportedConstruct
            If the argument has no wire form.
        """
        if isinstance(value, PrimitiveApplication) and value.name in _TRANSPARENT:
            return self._arg(value.arguments[0], site, head, plate, morphism)
        if isinstance(value, Var):
            return IRArgRef(name=value.local.name, indices=())
        if isinstance(value, LiteralValue) and isinstance(value.value, int | float):
            return IRArgNumber(value=float(value.value))
        if isinstance(value, Comprehension):
            spread = _spread_literal(value)
            if spread is not None:
                literal, shape = spread
                return IRArgBroadcast(
                    value=IRArgNumber(value=float(literal)), target_shape=shape
                )
        if isinstance(value, TensorValue):
            if value.items and all(
                isinstance(item, TensorValue) for item in value.items
            ):
                return IRArgMatrix(
                    rows=tuple(
                        IRArgList(
                            elements=tuple(
                                self._arg(entry, site, head, plate, morphism)
                                for entry in row.items
                            )
                        )
                        for row in value.items
                        if isinstance(row, TensorValue)
                    )
                )
            return IRArgList(
                elements=tuple(
                    self._arg(item, site, head, plate, morphism) for item in value.items
                )
            )
        if isinstance(value, Gather):
            if (
                self.via is not None
                and isinstance(value.index, Var)
                and value.index.local.name == self.via
            ):
                return self._arg(value.value, site, head, plate, morphism)
            base = self._arg(value.value, site, head, plate, morphism)
            index = self._arg(value.index, site, head, plate, morphism)
            if isinstance(base, IRArgRef):
                return IRArgRef(name=base.name, indices=(*base.indices, index))
        if isinstance(value, KernelMatrix):
            inputs = value.inputs
            if isinstance(inputs, Var):
                shape = tensor_shape(inputs.local.type)
                size = (
                    shape[1][0].value
                    if shape is not None
                    and shape[1]
                    and isinstance(shape[1][0], IndexLiteral)
                    and isinstance(shape[1][0].value, int)
                    else 0
                )
                return IRArgKernel(
                    kernel=value.kernel,
                    length_scale=value.length_scale,
                    x_name=inputs.local.name,
                    grid_size=size,
                    jitter=value.jitter,
                )
        if isinstance(value, AffineMap):
            return self._affine_head(value, site, head, plate, morphism)
        if isinstance(value, TableMap):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [f"param-source:table:{morphism}"],
            )
        if isinstance(value, DistributionValue):
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"nested-distribution-arg:{value.name}: a "
                    "distribution-valued draw argument is not "
                    "representable in this backend's IR"
                ],
            )
        raise UnsupportedConstruct(
            f"qvr-{self.planner.target}",
            [f"qiec:argument:{type(value).__name__}:{site}:{head}"],
        )

    def _affine_head(
        self, value: AffineMap, site: str, head: str, plate: Plate, morphism: str
    ) -> IRArg:
        """Bind one head of a parameter map and refer to it.

        Parameters
        ----------
        value : AffineMap
            The head.
        site : str
            The site whose argument the head fills.
        head : str
            The family parameter the head fills.
        plate : Plate
            The site's plate.
        morphism : str
            The morphism carrying the map.

        Returns
        -------
        IRArg
            A reference to the head's binding.

        Raises
        ------
        UnsupportedConstruct
            If the map's weight or bias is not a program input, or a
            source is not a binding.
        """
        if not isinstance(value.weight, Var) or not isinstance(value.bias, Var):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [f"param-source:linear:map:{morphism}"],
            )
        sources: list[LetAffineSource] = []
        for source in value.sources:
            if not isinstance(source, Var):
                raise UnsupportedConstruct(
                    f"qvr-{self.planner.target}",
                    [f"param-source:linear:argument:{morphism}"],
                )
            shape = tensor_shape(source.local.type)
            width = (
                int(shape[1][0].value)
                if shape is not None
                and len(shape[1]) == 1
                and isinstance(shape[1][0], IndexLiteral)
                else 1
            )
            sources.append(
                LetAffineSource(value=LetExprVar(name=source.local.name), width=width)
            )
        weight_shape = tensor_shape(value.weight.local.type)
        bias_shape = tensor_shape(value.bias.local.type)
        if weight_shape is not None and bias_shape is not None:
            self.param_map_inputs[value.weight.local.name] = Plate(
                event_dims=(),
                batch_dims=tuple(
                    _dim(size, f"{morphism}_param_{axis}")
                    for size, axis in zip(weight_shape[1], ("row", "col"), strict=False)
                ),
            )
            self.param_map_inputs[value.bias.local.name] = Plate(
                event_dims=(),
                batch_dims=(_dim(bias_shape[1][0], f"{morphism}_param_row"),),
            )
        transform = value.transform
        param_head = _ParamHead(
            arg_name=head,
            transform="exp_floor" if transform == "exp_floor" else "identity",
        )
        name = _head_binding_name(site, param_head)
        if transform == "identity":
            self.head_nodes.append(
                IRDeterministic(
                    name=name,
                    expr=LetExprAffineMap(
                        weight=LetExprVar(name=value.weight.local.name),
                        bias=LetExprVar(name=value.bias.local.name),
                        sources=tuple(sources),
                        row_offset=value.row_offset,
                        rows=value.rows,
                        transform="identity",
                    ),
                    constraint=CSReal(),
                    plate=plate,
                )
            )
            return IRArgRef(name=name, indices=())
        if transform != "exp_floor":
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [f"param-source:linear:transform:{transform}:{morphism}"],
            )
        raw = _head_raw_binding_name(site, param_head)
        axis = plate.batch_dims[-1] if plate.batch_dims else None
        axis_name = axis.name if isinstance(axis, DimStatic) else morphism
        self.head_nodes.append(
            IRDeterministic(
                name=raw,
                expr=LetExprAffineMap(
                    weight=LetExprVar(name=value.weight.local.name),
                    bias=LetExprVar(name=value.bias.local.name),
                    sources=tuple(sources),
                    row_offset=value.row_offset,
                    rows=value.rows,
                    transform="exp",
                ),
                constraint=CSReal(),
                plate=plate,
            )
        )
        self.head_nodes.append(
            IRDeterministic(
                name=name,
                expr=LetExprFactor(
                    binders=(
                        LetFactorBinder(
                            var=f"_{axis_name.lower()}_i",
                            index=TypeName(name=axis_name),
                        ),
                    ),
                    cases=tuple(
                        LetFactorCase(label=index, value=_floor_expr(raw, index))
                        for index in range(value.rows)
                    ),
                ),
                constraint=CSPositive(),
                plate=plate,
            )
        )
        return IRArgRef(name=name, indices=())

    def _structured_args(
        self, distribution: DistributionValue, site: str, plate: Plate
    ) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
        """The wire arguments of a structured family.

        A Gaussian process reads its mean as one literal and its
        covariance as a kernel argument; a multivariate or matrix
        normal reads its location and its scale factors as references
        to the inputs the construction names.

        Parameters
        ----------
        distribution : DistributionValue
            The construction.
        site : str
            The site the call is at.
        plate : Plate
            The site's plate.

        Returns
        -------
        tuple[tuple[IRArg, ...], tuple[str, ...]]
            The arguments and the constructor names they fill.
        """
        meta = FAMILY_META[distribution.name]
        assert meta.structured_lowering is not None
        named = dict(distribution.arguments)
        registry = _REGISTRY_NAMES.get(distribution.name, {})
        args: list[IRArg] = []
        names: list[str] = []
        for spec in meta.structured_lowering.args:
            arg_name = str(spec.arg_name)
            value = named.get(registry.get(arg_name, arg_name))
            if value is None:
                raise UnsupportedConstruct(
                    f"qvr-{self.planner.target}",
                    [f"family:{distribution.name}:structured_arg:{arg_name}"],
                )
            if isinstance(spec, StructuredZeroVectorArg):
                args.append(IRArgNumber(value=0.0))
                names.append(arg_name)
                continue
            arg = self._arg(value, site, arg_name, plate, "")
            if isinstance(spec, StructuredDataArg) and isinstance(arg, IRArgRef):
                dims = tuple(plate.event_dims[i] for i in spec.axis_indices)
                self.structured_inputs[arg.name] = (
                    _DATA_CONSTRAINT_FACTORY[spec.constraint_kind](),
                    Plate(event_dims=dims, batch_dims=()),
                )
            args.append(arg)
            names.append(arg_name)
        return tuple(args), tuple(names)

    def _class_index(
        self,
        morphism: str,
        meta: FamilyMeta,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        constraint: ConstraintSpec,
        site: str,
    ) -> tuple[tuple[IRArg, ...], ConstraintSpec]:
        """Restate a class-index draw's alphabet from its codomain.

        Parameters
        ----------
        morphism : str
            The morphism the site draws through, if any.
        meta : FamilyMeta
            The family's metadata.
        args : tuple[IRArg, ...]
            The arguments.
        arg_names : tuple[str, ...]
            Their names.
        constraint : ConstraintSpec
            The support read off the family.
        site : str
            The site.

        Returns
        -------
        tuple[tuple[IRArg, ...], ConstraintSpec]
            The arguments with the alphabet widened and the support
            over the codomain's elements.
        """
        outcome = class_index_outcome(meta)
        if outcome is None or not morphism:
            return args, constraint
        alphabet = _codomain_alphabet(morphism, self.ctx)
        if alphabet is None:
            return args, constraint
        width = alphabet.size
        if width < 2:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"class-index:{site}:codomain-width:{width}: a "
                    f"{meta.qvr_name} draw through {morphism!r} "
                    f"needs a codomain naming at least two classes"
                ],
            )
        arg_dim = DimStatic(size=width - outcome.extent_offset, name=alphabet.name)
        out = list(args)
        for index, name in enumerate(arg_names):
            if name in outcome.alphabet_args:
                out[index] = _retarget_alphabet_arg(out[index], arg_dim, site, self.ctx)
        return tuple(out), CSIntegerInterval(lower=0, upper=width - 1)

    # ------------------------------------------------------------------
    # pure expressions

    def _expr(self, value: Value) -> LetExprNode:
        """A pure value as a let expression.

        Parameters
        ----------
        value : Value
            The value.

        Returns
        -------
        LetExprNode
            The expression the renderers translate.

        Raises
        ------
        UnsupportedConstruct
            If the value has no expression form.
        """
        if isinstance(value, Var):
            return LetExprVar(name=value.local.name)
        if isinstance(value, LiteralValue):
            if isinstance(value.value, bool):
                return LetExprLiteral(value=float(value.value), integral=True)
            if isinstance(value.value, int | float):
                return LetExprLiteral(
                    value=float(value.value), integral=isinstance(value.value, int)
                )
            if isinstance(value.value, str):
                return LetExprString(value=value.value)
        if isinstance(value, PrimitiveApplication):
            if value.name in _TRANSPARENT:
                return self._expr(value.arguments[0])
            operator = PRIMITIVE_OPERATORS.get(value.name)
            if operator is not None and len(value.arguments) == 2:
                left, right = value.arguments
                return LetExprBinOp(
                    op=operator, left=self._expr(left), right=self._expr(right)
                )
            unary = _UNARY_OPERATORS.get(value.name)
            if unary is not None and len(value.arguments) == 1:
                return LetExprUnaryOp(op=unary, operand=self._expr(value.arguments[0]))
            builtin = PRIMITIVE_BUILTINS.get(value.name)
            if builtin is not None:
                return LetExprCall(
                    func=builtin,
                    args=tuple(self._expr(argument) for argument in value.arguments),
                )
        if isinstance(value, Gather):
            array = self._expr(value.value)
            index = self._expr(value.index)
            if isinstance(array, LetExprIndex):
                # Nested gathers read back as one multi-index subscript.
                return LetExprIndex(array=array.array, indices=(*array.indices, index))
            return LetExprIndex(array=array, indices=(index,))
        if isinstance(value, TensorValue):
            return LetExprList(items=tuple(self._expr(item) for item in value.items))
        if isinstance(value, Reduction):
            return LetExprCall(func=value.operator, args=(self._expr(value.value),))
        rowwise = _row_reduction(value)
        if rowwise is not None:
            operator, reduced = rowwise
            return LetExprCall(func=operator, args=(self._expr(reduced),))
        if isinstance(value, Rowwise):
            return LetExprCall(func=value.operator, args=(self._expr(value.value),))
        if isinstance(value, Comprehension):
            axis = _axis_name_of(value.extent, self.ctx.cards)
            binder = LetFactorBinder(var=value.binder.name, index=TypeName(name=axis))
            body = self._expr(value.body)
            if isinstance(body, LetExprFactor) and body.body is not None:
                # Nested comprehensions read back as one factor over
                # every binder.
                return LetExprFactor(binders=(binder, *body.binders), body=body.body)
            return LetExprFactor(binders=(binder,), body=body)
        if isinstance(value, WeightSum):
            return LetExprCall(func="sum", args=(self._expr(value.value),))
        if isinstance(value, LogDensity):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [f"qiec:expression:log_density:{self.program.name}"],
            )
        raise UnsupportedConstruct(
            f"qvr-{self.planner.target}",
            [f"qiec:expression:{type(value).__name__}:{self.program.name}"],
        )

    # ------------------------------------------------------------------
    # inputs

    def inputs(self, body: tuple[IRNode, ...]) -> tuple[IRDataInput, ...]:
        """The plan's inputs.

        Parameters
        ----------
        body : tuple[IRNode, ...]
            The derived body, whose observations fix their inputs'
            supports and plates.

        Returns
        -------
        tuple[IRDataInput, ...]
            One input per entry-point parameter, grouped by role as
            the targets declare them: the program's domain, its
            scalars, fibrations, observations, parameter maps, and
            data.
        """
        groups: dict[str, list[IRDataInput]] = {
            "domain": [],
            "scalar": [],
            "fibration": [],
            "observation": [],
            "map": [],
            "data": [],
        }
        for parameter in self.entry.parameters:
            node = self._input(parameter, body)
            if parameter.role in ("weight", "bias", "table"):
                groups["map"].append(node)
            elif parameter.role in ("kernel-input",):
                groups["data"].append(node)
            else:
                groups[parameter.role].append(node)
        return tuple(
            node
            for role in ("domain", "scalar", "fibration", "observation", "map", "data")
            for node in groups[role]
        )

    def _input(
        self, parameter: ProgramParameter, body: tuple[IRNode, ...]
    ) -> IRDataInput:
        """One input.

        Parameters
        ----------
        parameter : ProgramParameter
            The entry-point parameter.
        body : tuple[IRNode, ...]
            The derived body.

        Returns
        -------
        IRDataInput
            The input, typed by the parameter and constrained by its
            role and use.
        """
        name = parameter.name
        if parameter.role == "observation" and name in self.observed:
            node = self.observed[name]
            return IRDataInput(name=name, constraint=node.constraint, plate=node.plate)
        if parameter.role == "fibration":
            upper = None
            batch: tuple[Dim, ...] = ()
            for node in _walk(body):
                if isinstance(node, IRObserve) and node.via == name:
                    batch = node.plate.batch_dims
                    first = batch[0] if batch else None
                    upper = first.size if isinstance(first, DimStatic) else None
                    break
            constraint: ConstraintSpec = (
                CSNonnegativeInteger()
                if upper is None
                else CSIntegerInterval(lower=0, upper=max(upper - 1, 0))
            )
            return IRDataInput(
                name=name,
                constraint=constraint,
                plate=Plate(event_dims=(), batch_dims=batch),
            )
        if name in self.structured_inputs:
            constraint, plate = self.structured_inputs[name]
            return IRDataInput(name=name, constraint=constraint, plate=plate)
        if name in self.param_map_inputs:
            return IRDataInput(
                name=name, constraint=CSReal(), plate=self.param_map_inputs[name]
            )
        dims = _typed_dims(parameter.type, parameter.axes)
        if parameter.role == "domain":
            # The program's input is one value of its domain: the
            # factor's width is the input's event.
            plate = Plate(event_dims=dims, batch_dims=())
            return IRDataInput(
                name=name,
                constraint=_apply_declared_bounds(CSReal(), plate, self.ctx, name),
                plate=plate,
            )
        plate = Plate(event_dims=(), batch_dims=dims)
        element = _element_type(parameter.type)
        constraint = CSNonnegativeInteger() if element == INT else CSReal()
        return IRDataInput(name=name, constraint=constraint, plate=plate)


# ---------------------------------------------------------------------------
# module-level helpers


def _collect_site_steps(
    steps: Sequence[ProgramStep],
    out: dict[str, SampleStep | ObserveStep | MarginalizeStep],
) -> None:
    """Index a program's draws by the site each binds.

    Parameters
    ----------
    steps : Sequence[ProgramStep]
        The steps, searched through marginalization scopes.
    out : dict[str, SampleStep | ObserveStep | MarginalizeStep]
        The index, extended in place.
    """
    for step in steps:
        if isinstance(step, SampleStep):
            for name in step.vars:
                out[name] = step
        elif isinstance(step, ObserveStep):
            for name in step.vars:
                out[name] = step
        elif isinstance(step, MarginalizeStep):
            out[step.var] = step
            _collect_site_steps(step.scope, out)


def _site_and_distribution(
    request: EffectRequest, target: str
) -> tuple[SiteValue, DistributionValue]:
    """The site and construction of a sample request.

    Parameters
    ----------
    request : EffectRequest
        The request.
    target : str
        The transpile target, for the diagnostic.

    Returns
    -------
    tuple[SiteValue, DistributionValue]
        The site and the family construction drawn from.

    Raises
    ------
    UnsupportedConstruct
        If the request's arguments are not a site and a construction.
    """
    if (
        len(request.arguments) == 2
        and isinstance(request.arguments[0], SiteValue)
        and isinstance(request.arguments[1], DistributionValue)
    ):
        return request.arguments[0], request.arguments[1]
    raise UnsupportedConstruct(f"qvr-{target}", ["qiec:sample:unstructured"])


def _returned_names(value: Value) -> tuple[str, ...] | None:
    """The names a return value lists.

    Parameters
    ----------
    value : Value
        The returned value.

    Returns
    -------
    tuple[str, ...] | None
        The bound names, or ``None`` when the value is not a binding
        or a tuple of bindings.
    """
    if isinstance(value, Var):
        return (value.local.name,)
    if isinstance(value, TupleValue) and all(
        isinstance(item, Var) for item in value.items
    ):
        return tuple(item.local.name for item in value.items if isinstance(item, Var))
    return None


def _marginal_reduction(handler: str) -> str:
    """The reduction an enumeration handler's name records.

    Parameters
    ----------
    handler : str
        The handler's name.

    Returns
    -------
    str
        ``logsumexp``, ``sum``, or ``mean``.
    """
    if handler.endswith("_sum"):
        return "sum"
    if handler.endswith("_mean"):
        return "mean"
    return "logsumexp"


def _dim_names(plate: Plate) -> tuple[str, ...]:
    """The axis names of a plate, batch dimensions first.

    Parameters
    ----------
    plate : Plate
        The plate.

    Returns
    -------
    tuple[str, ...]
        One name per dimension.
    """
    return tuple(str(dim.name) for dim in (*plate.batch_dims, *plate.event_dims))


def _dim(size: IndexTerm, name: str) -> Dim:
    """A plate dimension from an index term.

    Parameters
    ----------
    size : IndexTerm
        The extent.
    name : str
        The axis's name.

    Returns
    -------
    Dim
        A static dimension for a literal extent, a dynamic one for a
        variable.
    """
    if isinstance(size, IndexLiteral) and isinstance(size.value, int):
        return DimStatic(size=size.value, name=name)
    if isinstance(size, IndexVariable):
        return DimDynamic(size_name=size.name, name=name)
    raise UnsupportedConstruct("qvr-lower", [f"axis:unknown-cardinality:{name}"])


def _axis_dim(axis: PlateAxis) -> Dim:
    """A plate dimension from a plate axis.

    Parameters
    ----------
    axis : PlateAxis
        The axis.

    Returns
    -------
    Dim
        The dimension, named after the axis.
    """
    return _dim(axis.size, axis.name)


def _plate_of(
    plate: PlateShape,
    marginalized: bool,
    meta: FamilyMeta,
    distribution: DistributionValue,
    walk: _ProgramWalk,
) -> Plate:
    """The IR plate of a construction.

    Parameters
    ----------
    plate : PlateShape
        The construction's plate.
    marginalized : bool
        Whether the construction is a marginalized latent's.
    meta : FamilyMeta
        The family's metadata.
    distribution : DistributionValue
        The construction.
    walk : _ProgramWalk
        The walk, for the support tables.

    Returns
    -------
    Plate
        Batch and event dimensions named after the axes.
    """
    del marginalized, distribution, walk
    event = tuple(_axis_dim(axis) for axis in plate.event)
    if meta.qvr_name == "LKJCholesky" and len(event) == 2 and event[0] == event[1]:
        # A correlation factor is square over one axis, which the
        # plan states once.
        event = event[:1]
    return Plate(
        event_dims=event,
        batch_dims=tuple(_axis_dim(axis) for axis in plate.batch),
    )


def _typed_dims(type_: TypeExpr, axes: tuple[str, ...]) -> tuple[Dim, ...]:
    """The dimensions a tensor type states.

    Parameters
    ----------
    type_ : TypeExpr
        The type.
    axes : tuple[str, ...]
        The dimensions' names, when known.

    Returns
    -------
    tuple[Dim, ...]
        One dimension per tensor axis, named by ``axes`` or by
        position; none for a scalar.
    """
    shape = tensor_shape(type_)
    if shape is None:
        return ()
    return tuple(
        _dim(size, axes[index] if index < len(axes) else f"axis{index}")
        for index, size in enumerate(shape[1])
    )


def _value_plate(type_: TypeExpr, axes: tuple[str, ...] = ()) -> Plate:
    """The plate a value's type states.

    Parameters
    ----------
    type_ : TypeExpr
        The type.
    axes : tuple[str, ...]
        The dimensions' names, when known.

    Returns
    -------
    Plate
        The tensor's dimensions as batch dimensions; no dimensions for
        a scalar.
    """
    return Plate(event_dims=(), batch_dims=_typed_dims(type_, axes))


def _element_type(type_: TypeExpr) -> TypeExpr:
    """The element of a tensor type, or the type itself.

    Parameters
    ----------
    type_ : TypeExpr
        The type.

    Returns
    -------
    TypeExpr
        The scalar type.
    """
    shape = tensor_shape(type_)
    return shape[0] if shape is not None else type_


def _structured(distribution: DistributionValue) -> bool:
    """Whether a construction is a structured family's.

    Parameters
    ----------
    distribution : DistributionValue
        The construction.

    Returns
    -------
    bool
        ``True`` when an argument is a kernel matrix or the family
        reads its factors structurally.
    """
    return any(
        isinstance(value, KernelMatrix) for _, value in distribution.arguments
    ) or distribution.name in ("GP", "MultivariateNormal", "MatrixNormal")


def _axis_name_of(extent: IndexTerm, cards: dict[str, int]) -> str:
    """The object an extent names.

    Parameters
    ----------
    extent : IndexTerm
        The extent.
    cards : dict[str, int]
        The object cardinalities.

    Returns
    -------
    str
        The first object with that cardinality, or the extent rendered.
    """
    if isinstance(extent, IndexLiteral):
        for name, size in cards.items():
            if size == extent.value:
                return name
    if isinstance(extent, IndexVariable):
        return extent.name
    return render_static(extent)


def _row_reduction(value: Value) -> tuple[str, Value] | None:
    """A reduction along a tensor's last axis, as the operator and tensor.

    The elaboration writes ``sum(x)`` over a tensor of rank two or
    more as a comprehension over the leading axes of the reduction of
    each row; the wire form is the call itself.

    Parameters
    ----------
    value : Value
        The value.

    Returns
    -------
    tuple[str, Value] | None
        The operator and the tensor reduced, or ``None`` when the
        value is not such a comprehension.
    """
    if not isinstance(value, Comprehension):
        return None
    body = value.body
    inner = _row_reduction(body)
    if inner is not None:
        operator, reduced = inner
    elif isinstance(body, Reduction):
        operator, reduced = body.operator, body.value
    else:
        return None
    if (
        isinstance(reduced, Gather)
        and isinstance(reduced.index, Var)
        and reduced.index.local == value.binder
    ):
        return operator, reduced.value
    return None


def _spread_literal(value: Comprehension) -> tuple[float, tuple[int, ...]] | None:
    """A literal spread over a vector, as the literal and the shape.

    A single literal a construction spreads over a vector parameter,
    ``Dirichlet(1.0)`` over an axis, is a comprehension of the literal,
    which is a broadcast on the wire.

    Parameters
    ----------
    value : Comprehension
        The comprehension.

    Returns
    -------
    tuple[float, tuple[int, ...]] | None
        The literal and the shape, or ``None`` when the body is not a
        literal or a spread itself, or an extent is open.
    """
    if not isinstance(value.extent, IndexLiteral) or not isinstance(
        value.extent.value, int
    ):
        return None
    body = value.body
    if isinstance(body, LiteralValue) and isinstance(body.value, int | float):
        return float(body.value), (value.extent.value,)
    if isinstance(body, Comprehension):
        inner = _spread_literal(body)
        if inner is not None:
            literal, shape = inner
            return literal, (value.extent.value, *shape)
    return None


def _referenced(value: Value) -> Iterator[str]:
    """The bindings a value reads, in order of appearance.

    Parameters
    ----------
    value : Value
        The value.

    Yields
    ------
    str
        Each referenced local's name.
    """
    if isinstance(value, Var):
        yield value.local.name
        return
    for name in getattr(value, "__dataclass_fields__", {}):
        field = getattr(value, name)
        if isinstance(field, tuple):
            for item in field:
                if hasattr(item, "__dataclass_fields__"):
                    yield from _referenced(item)  # type: ignore[arg-type]
        elif hasattr(field, "__dataclass_fields__"):
            yield from _referenced(field)  # type: ignore[arg-type]


def _walk(body: tuple[IRNode, ...]) -> Iterator[IRNode]:
    """Every node of a body, descending into scopes.

    Parameters
    ----------
    body : tuple[IRNode, ...]
        The body.

    Yields
    ------
    IRNode
        The nodes in order.
    """
    for node in body:
        yield node
        if isinstance(node, IRMarginalize):
            yield from _walk(node.scope)


def checked_module(module: Module, *, target: str) -> QiecModule | None:
    """Elaborate and check a module at the lowering boundary.

    A source with nothing to elaborate takes no checking path. Otherwise
    the elaboration constructs the complete
    :class:`~quivers.qiec.QiecModule` and ``validate_module`` independently
    checks its nominal declarations, static scopes, effect rows, and every
    computation type. Capability analysis is deliberately separate: the
    lowering always retains the complete typed module, and a selected
    renderer uses
    :func:`quivers.transpile.qiec_ir.analyze_qiec_capabilities` to determine
    whether it can preserve each feature.

    Parameters
    ----------
    module : Module
        The parsed module.
    target : str
        The transpile target, for diagnostics.

    Returns
    -------
    QiecModule | None
        The checked module, or ``None`` when the source has nothing to
        elaborate.

    Raises
    ------
    UnsupportedConstruct
        If a program uses a form the elaboration does not admit; the
        construct is reported the way a renderer reports one it cannot
        emit.
    QiecDiagnosticError
        If a QIEC declaration is rejected.
    """
    if not has_qiec_surface(module):
        return None
    try:
        qiec_module: QiecModule = lower_qvr_to_qiec(module)
    except QiecDiagnosticError as error:
        if error.code == GAP_CODE:
            # A program using a chart or a network morphism has no
            # elaboration yet. The module's other declarations still
            # lower; the program itself reaches the renderer through its
            # plan alone, and the gap is recorded on the plan.
            qiec_module = lower_qvr_to_qiec(module, elaborate_programs=False)
            validate_module(qiec_module)
            return replace(qiec_module, gap=error.message)
        if error.code != "qiec-program" and error.program is None:
            raise
        raise UnsupportedConstruct(f"qvr-{target}", [error.message]) from error
    validate_module(qiec_module)
    _refuse_deduction_calls(qiec_module, target)
    return qiec_module


def _refuse_deduction_calls(qiec_module: QiecModule, target: str) -> None:
    """Refuse a program that calls a deduction.

    A deduction enumerates its derivations through a search handler that
    no target runtime carries, so a program calling one is refused at
    the boundary, before any target-specific lowering, under the
    capability tag the renderers use.

    Parameters
    ----------
    qiec_module : QiecModule
        The checked module.
    target : str
        The transpile target, for the diagnostic.

    Raises
    ------
    UnsupportedConstruct
        If a program's computation reaches a deduction's computation.
    """
    by_id = {computation.id: computation for computation in qiec_module.computations}
    kinds = [
        f"qiec:capability:search:{by_id[identity].name}"
        for identity in sorted(
            program_computations(qiec_module), key=lambda item: item.digest
        )
        if identity in by_id
        and by_id[identity].origin.structural_path[:1] == ("deductions",)
        and by_id[identity].name.endswith("__run")
    ]
    if kinds:
        raise UnsupportedConstruct(f"qvr-{target}", kinds)


class Lower(dx.Mapping[Module, IRProgram]):
    """Map a parsed [`Module`][quivers.dsl.ast_nodes.Module] to an
    [`IRProgram`][quivers.transpile.ir.IRProgram].

    The lowering elaborates the module through the QIEC route first;
    the program's plan is then derived from its checked computation.
    A module without a program lowers to its checked computations
    alone. Stateless: every table is derived from the input module on
    each call.
    """

    def forward(self, module: Module, *, target: str = "ir") -> IRProgram:
        """Lower a parsed module to the transpile IR.

        Parameters
        ----------
        module : Module
            The parsed module, with composite lets already expanded for
            the target.
        target : str
            The transpile target the lowering serves, named in the
            diagnostics the QIEC boundary raises.

        Returns
        -------
        IRProgram
            The lowered program with its checked kernel module.

        Raises
        ------
        UnsupportedConstruct
            If the module has no program and no QIEC surface, uses a
            construct the elaboration does not admit, or falls in a
            gap of the elaboration.
        QiecDiagnosticError
            If a QIEC declaration is rejected.
        """
        qiec_module = checked_module(module, target=target)
        if qiec_module is None:
            # A source with nothing to elaborate has no program either;
            # report the established precise ``program:absent`` error.
            pick_program(module)
            raise UnsupportedConstruct(f"qvr-{target}", ["program:absent"])
        if not any(
            isinstance(statement, ProgramDecl) for statement in module.statements
        ):
            qiec_ir = lower_qiec_ir(qiec_module)
            return IRProgram(
                name=qiec_ir.module, inputs=(), body=(), module=qiec_ir, cards={}
            )
        expanded = expand_composite_lets(module, target="stan")
        program = pick_program(expanded)
        if qiec_module.gap:
            # The program falls in a gap of the elaboration: it has no
            # computation to derive a plan from, and the gap names the
            # construct by a kind the diagnostics explain.
            kind = qiec_module.gap.partition("; ")[0]
            raise UnsupportedConstruct(f"qvr-{target}", [kind])
        return program_plan(qiec_module, expanded, program, target)


def program_plan(
    module: QiecModule, source: Module, program: ProgramDecl, target: str
) -> IRProgram:
    """Derive a program's plan from its checked module.

    Parameters
    ----------
    module : QiecModule
        The checked module.
    source : Module
        The expanded source module.
    program : ProgramDecl
        The program.
    target : str
        The transpile target.

    Returns
    -------
    IRProgram
        The plan.
    """
    return _Planner(module, source, target).plan(program)


__all__ = ["Lower", "checked_module", "program_plan"]
