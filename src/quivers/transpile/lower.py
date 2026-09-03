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
5. Discovers exogenous identifiers (the `Real N` factors of the
   program's declared domain; free names in let / score bodies and
   bracket-indexed args; `via=` fibrations; scalar program
   parameters) and emits
   [`IRDataInput`][quivers.transpile.ir.IRDataInput] entries.

Sentinel construction and property-form `arg_constraints` resolution
are handled by `_make_sentinel` and `_resolve_arg_constraints`. The
sentinel is the only place in the transpile layer that materialises
torch tensors; renderers never do.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable
from typing import Literal

import didactic.api as dx
import torch
import torch.distributions.constraints as c
from torch.distributions.distribution import Distribution

from quivers.core._util import EPS
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
    OptionString,
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
    TypeEnumSet,
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
    LetFactorBinder,
    LetFactorCase,
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
    class_index_outcome,
    finite_enumerable_at_call_site,
)
from quivers.transpile.ir import (
    CSIntegerInterval,
    CSInterval,
    CSNonnegativeInteger,
    CSPositive,
    CSPositiveDefinite,
    CSReal,
    CSRealMatrix,
    CSRealVector,
    CSUnitInterval,
    Constraint,
    ConstraintSpec,
    Dim,
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
    LetAffineSource,
    LetExprAffineMap,
    OverOrCodomainAxes,
    Plate,
    StructuredArgSpec,
    StructuredDataArg,
    StructuredKernelArg,
    StructuredZeroVectorArg,
    event_shape_of,
    from_constraint,
)


def pick_program(module: Module) -> ProgramDecl:
    """Pick the `ProgramDecl` the module's `export` designates.

    When the module declares multiple programs, prefer one
    referenced by an `export` declaration; otherwise pick the
    last declared program.

    An `export` naming a `define` binding rather than a program
    is not a lowering target: a `define` is a morphism-level
    composition (`scan(cell) >> decoder`), and the transpile
    boundary is the probabilistic program. A `define` a step
    references is unfolded by
    [`expand_composite_lets`][quivers.transpile._expand_composites.expand_composite_lets]
    at the call site instead, so the recurrence a `scan` denotes
    reaches the IR through the program that samples it.
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
            ["program:absent"],
        )
    return next(
        (p for p in programs if p.name in exported_names),
        programs[-1],
    )


def exported_return_names(module: Module) -> tuple[str, ...]:
    """The variable names the module's exported program returns.

    A QVR program declared `prog : A -> B` denotes a Markov kernel
    from `A` to `B`, and its `return` clause names the components of
    the value in `B` that the kernel carries. Those names are what a
    faithful emit has to expose through the target's own return
    surface, so they are the contract the export-equivalence tier
    checks a backend against. An empty tuple means the program
    declares no `return` clause and denotes only its joint.
    """
    return tuple(pick_program(module).return_vars)


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


# ---------------------------------------------------------------------------
# Declared-morphism parameter maps.
#
# A kernel morphism declared `morphism f : X -> Y ~ Family` with no
# family arguments of its own is a *conditional* family: the runtime
# builds `Conditional<Family>(X, Y)`, which owns a
# [`ParamSource`][quivers.continuous.param_source.ParamSource] from `X`
# to a row of `k * dim(Y)` numbers and reads the family's `k`
# per-coordinate arguments off that row. The declaration names neither
# the map nor its numbers, so a program emitted from the declaration
# alone scores the family at its defaults: a different measure, and on
# a different space whenever `dim(X) != dim(Y)`.
#
# Lowering closes that by putting the map *into* the IR: the map's
# weight and bias become two `IRDataInput` entries (the numbers are
# fixed model data, exactly as the runtime's compile-time draw makes
# them), and one `IRDeterministic` per family argument carries the row
# block the head reads and the head's transform. The site then scores
# against those arguments rather than against the family's defaults.
#
# The encoding is linear algebra, not arithmetic:
# [`LetExprAffineMap`][quivers.transpile.ir.LetExprAffineMap] names the
# weight, the bias, the row block, and the ordered sources whose
# concatenation is the conditioning row, and each renderer spells the
# contraction in its own language. One expression node per head,
# whatever the object's width, so a 16-wide state costs what a 2-wide
# state costs.
# ---------------------------------------------------------------------------


class _ParamHead(dx.Model):
    """One family argument read off a row of the parameter map.

    Head `k` of a family whose codomain is `d` wide reads columns
    ``k * d .. k * d + d`` of the map's output row. `transform` names
    what the runtime applies to the raw column before the family sees
    it:

    * ``identity``: the column is the argument
      (`ConditionalNormal`'s `loc`).
    * ``exp_floor``: the column is a log-parameter, exponentiated and
      floored at [`EPS`][quivers.core._util.EPS]
      (`ConditionalNormal`'s `scale`, whose runtime spelling is
      ``log_sigma.exp().clamp(min=EPS)``).
    """

    arg_name: str
    transform: Literal["identity", "exp_floor"]


#: Families whose conditional runtime class reads its arguments off a
#: parameter-map row, in the family's canonical argument order. A
#: family absent from this table keeps whatever arguments its
#: declaration and option block supply, which is what the emission has
#: always done; see the module note above for why the table is not
#: simply every conditional family.
_CONDITIONAL_HEADS: dict[str, tuple[_ParamHead, ...]] = {
    "Normal": (
        _ParamHead(arg_name="loc", transform="identity"),
        _ParamHead(arg_name="scale", transform="exp_floor"),
    ),
}


class _ParamMapSource(dx.Model):
    """One factor of the conditioning row a parameter map reads.

    A morphism whose domain is a product reads the concatenation of
    its factors, in declaration order, exactly as the runtime's
    `MonadicProgram` stacks a multi-argument step along the feature
    axis.
    """

    name: str
    width: int


class _ParamMap(dx.Model):
    """The affine parameter map one declared kernel morphism carries.

    `weight` and `bias` are the wire names of the two data inputs the
    map's numbers arrive on; `sources` is the conditioning row in
    column order; `axis` and `width` name the codomain object and its
    width; `heads` is the family's head table.
    """

    morphism: str
    family: str
    weight: str
    bias: str
    sources: tuple[_ParamMapSource, ...]
    axis: str
    width: int
    heads: tuple[_ParamHead, ...]


def _param_map_domain_width(pmap: _ParamMap) -> int:
    """Total width of the map's conditioning row."""
    return sum(source.width for source in pmap.sources)


def _param_map_rows(pmap: _ParamMap) -> int:
    """Number of rows in the map's weight: one per head coordinate."""
    return pmap.width * len(pmap.heads)


def _param_map_weight_name(morphism: str) -> str:
    """Wire name of a morphism's parameter-map weight."""
    return f"{morphism}_param_weight"


def _param_map_bias_name(morphism: str) -> str:
    """Wire name of a morphism's parameter-map bias."""
    return f"{morphism}_param_bias"


def _head_binding_name(site: str, head: _ParamHead) -> str:
    """Wire name of the deterministic binding carrying one head."""
    return f"{site}_{head.arg_name}"


def _head_raw_binding_name(site: str, head: _ParamHead) -> str:
    """Wire name of the pre-floor binding of an ``exp_floor`` head."""
    return f"{site}_{head.arg_name}_raw"


def _affine_map_expr(
    pmap: _ParamMap, head_index: int, transform: Literal["identity", "exp"]
) -> LetExprAffineMap:
    """One head's row block of the map, as a single contraction.

    Head `k` reads rows ``k * width .. k * width + width`` of the
    weight, which is the layout the runtime's `ParamSource` writes its
    output row in; the columns are the concatenated domain
    coordinates, in the declaration order `sources` already carries.
    """
    if _param_map_domain_width(pmap) == 0:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"param-source:linear:empty-domain:{pmap.morphism}: the "
                f"morphism's domain carries no coordinates, so its "
                f"parameter map has no input row"
            ],
        )
    return LetExprAffineMap(
        weight=LetExprVar(name=pmap.weight),
        bias=LetExprVar(name=pmap.bias),
        sources=tuple(
            LetAffineSource(
                value=LetExprVar(name=source.name), width=source.width
            )
            for source in pmap.sources
        ),
        row_offset=head_index * pmap.width,
        rows=pmap.width,
        transform=transform,
    )


def _option_identifier(
    options: tuple[OptionEntry, ...], key: str
) -> str | None:
    """The identifier an option is bound to, when it is bound to one."""
    for entry in options:
        if entry.key == key and isinstance(entry.value, OptionName):
            return entry.value.value
    return None


def _has_bare_family_init(
    decl: MorphismDecl, family: str, family_set: frozenset[str]
) -> bool:
    """Whether the declaration's init clause is the bare
    ``~ Family`` form for `family`.

    The parser models ``~ Normal`` (no parentheses) as an
    `init_expr` naming an identifier and ``~ Normal(0, 1)`` as an
    `init_family` carrying args, so both shapes are read here. A
    declaration with args wrote its parameters and keeps them.
    """
    init = decl.init_family
    if init is not None:
        return init.family == family and not init.args
    expr = decl.init_expr
    return (
        isinstance(expr, ExprIdent)
        and expr.name == family
        and expr.name in family_set
    )


def _declares_head_argument(
    options: tuple[OptionEntry, ...], heads: tuple[_ParamHead, ...]
) -> bool:
    """Whether the option block writes one of the family's arguments.

    ``[scale=0.5] ~ Normal`` routes the option into the family's
    `scale` slot, which is a declaration of that parameter; the
    emission keeps it rather than replacing it with a map the source
    never asked for.
    """
    names = {head.arg_name for head in heads}
    return any(
        entry.key in names
        and isinstance(entry.value, (OptionNumber, OptionString))
        for entry in options
    )


def _assert_param_map_plate(
    morphism: str, plate: Plate, axis: str, width: int
) -> None:
    """Assert a mapped site is plated over its codomain's width.

    The map produces one row per codomain coordinate, so the site it
    feeds carries exactly that one axis; any other plate is scoring
    something other than the row the map computes.
    """
    dims = (*plate.event_dims, *plate.batch_dims)
    if (
        len(dims) == 1
        and isinstance(dims[0], DimStatic)
        and dims[0].size == width
        and dims[0].name == axis
    ):
        return
    raise UnsupportedConstruct(
        "qvr-lower",
        [
            f"param-source:linear:plate:{morphism}: the parameter map "
            f"produces one row per coordinate of {axis!r} ({width} of "
            f"them) and the site is plated {plate!r}"
        ],
    )


def _domain_wire_name(factor: TypeName) -> str:
    """Wire name of one ``Real N`` factor of a program domain."""
    return factor.name.lower()


def _program_domain_sources(ctx: _LowerCtx) -> tuple[_ParamMapSource, ...]:
    """The program input, factor by factor, as a conditioning row.

    Mirrors [`Lower._domain_inputs`][quivers.transpile.lower.Lower._domain_inputs]:
    same factors, same wire names, same order, so a map conditioned on
    the program input reads the inputs that pass declares.
    """
    out: list[_ParamMapSource] = []
    for factor in object_factors(ctx.program.domain):
        if not isinstance(factor, TypeName):
            continue
        width = ctx.real_widths.get(factor.name)
        if width is None:
            continue
        out.append(
            _ParamMapSource(name=_domain_wire_name(factor), width=width)
        )
    return tuple(out)


def _declared_domain_width(morphism_name: str, ctx: _LowerCtx) -> int:
    """Coordinate count of a declared morphism's domain.

    The runtime sizes a parameter source by this number, so an
    emission that conditions the map on anything else is applying a
    different map.
    """
    decl = ctx.morphisms[morphism_name]
    total = 0
    for factor in object_factors(decl.domain):
        if not isinstance(factor, TypeName):
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"param-source:linear:domain:{morphism_name}: the "
                    f"morphism's domain has a factor that is not a "
                    f"named object, so the width of its parameter "
                    f"map's input row is not readable"
                ],
            )
        width = ctx.real_widths.get(factor.name)
        if width is None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"param-source:linear:domain:{morphism_name}: the "
                    f"domain factor {factor.name!r} declares no real "
                    f"width, so the width of the parameter map's "
                    f"input row is not readable"
                ],
            )
        total += width
    return total


def _assert_free_binding(name: str, ctx: _LowerCtx) -> None:
    """Assert a synthesised binding name is not already spoken for."""
    if name not in ctx.reserved_names:
        return
    raise UnsupportedConstruct(
        "qvr-lower",
        [
            f"param-source:linear:name-collision:{name}: the parameter "
            f"map's head binds {name!r}, and the program already binds "
            f"that name; rename the binding so the map's arguments "
            f"have an unambiguous wire name"
        ],
    )


def _reserved_names(
    program: ProgramDecl,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
) -> frozenset[str]:
    """Every name the module already speaks for.

    A synthesised head binding or map input may not shadow a program
    binding, a declared morphism, or a let; the check is by name
    because a name is what every target's emission collides on.
    """
    names: set[str] = set(morphisms) | set(lets)
    _collect_step_names(program.draws, names)
    for factor in object_factors(program.domain):
        if isinstance(factor, TypeName):
            names.add(_domain_wire_name(factor))
    return frozenset(names)


def _collect_step_names(
    steps: tuple[ProgramStep, ...], names: set[str]
) -> None:
    """Add every name the steps bind, marginalize scopes included."""
    for step in steps:
        if isinstance(step, (SampleStep, ObserveStep, ReturnStep)):
            names.update(step.vars)
        elif isinstance(step, MarginalizeStep):
            names.add(step.var)
            _collect_step_names(step.scope, names)
        elif isinstance(step, (LetStep, ScoreStep)):
            names.add(step.name)


def _floor_expr(name: str, coordinate: int) -> LetExprNode:
    """``max(name[coordinate], EPS)`` as ``(a + e + |a - e|) / 2``.

    The runtime floors an exponentiated scale head at
    [`EPS`][quivers.core._util.EPS]. No target-portable two-argument
    `max` exists in the let-expression surface (`max` is the
    axis-reducing aggregator on the array-shaped backends), so the
    floor is spelled with the arithmetic identity, whose only call is
    `abs`.
    """
    value = LetExprIndex(
        array=LetExprVar(name=name),
        indices=(LetExprLiteral(value=float(coordinate)),),
    )
    floor = LetExprLiteral(value=EPS)
    return LetExprBinOp(
        op="/",
        left=LetExprBinOp(
            op="+",
            left=LetExprBinOp(op="+", left=value, right=floor),
            right=LetExprCall(
                func="abs",
                args=(
                    LetExprBinOp(op="-", left=value, right=floor),
                ),
            ),
        ),
        right=LetExprLiteral(value=2.0),
    )


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
        shapes = object_shapes(expanded)
        cards = {
            name: shape.extent
            for name, shape in shapes.items()
            if shape.extent is not None
        }
        real_widths = {
            name: shape.real_width
            for name, shape in shapes.items()
            if shape.real_width is not None
        }
        bounds = {
            name: shape.bounds
            for name, shape in shapes.items()
            if shape.bounds.is_bounded
        }
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
            real_widths=real_widths,
            bounds=bounds,
            shapes=shapes,
            reserved_names=_reserved_names(program, morphisms, lets),
        )

        body = self._lower_steps(program.draws, ctx)
        if program.return_vars:
            body = (*body, IRReturn(names=tuple(program.return_vars)))
        inputs = self._build_inputs(program, body, ctx)
        inputs, body = _propagate_let_plates(inputs, body)
        inputs, body = _propagate_alphabet_event_dims(
            inputs, body, ctx.alphabet_event_dims,
        )
        return IRProgram(
            name=program.name,
            inputs=inputs,
            body=body,
            cards=dict(cards),
        )

    def _pick_program(self, module: Module) -> ProgramDecl:
        """Pick the `ProgramDecl` to lower; see
        [`pick_program`][quivers.transpile.lower.pick_program]."""
        return pick_program(module)

    def _lower_steps(
        self,
        steps: tuple[ProgramStep, ...],
        ctx: _LowerCtx,
    ) -> tuple[IRNode, ...]:
        """Lower a tuple of program steps into IR nodes.

        Each lowered node's plate is recorded on the context before
        the next step lowers, so a step whose argument references an
        earlier binding can read whether that binding already carries
        an event shape.

        A step lowers to *one or more* nodes: a draw through a
        declared morphism that carries a parameter map is preceded by
        the deterministic bindings that compute the family's
        arguments from it.
        """
        out: list[IRNode] = []
        for step in steps:
            for node in self._lower_step(step, ctx):
                _record_bound_plate(node, ctx)
                out.append(node)
        return tuple(out)

    def _lower_step(
        self, step: ProgramStep, ctx: _LowerCtx
    ) -> tuple[IRNode, ...]:
        if isinstance(step, SampleStep):
            return self._lower_sample(step, ctx)
        if isinstance(step, ObserveStep):
            return self._lower_observe(step, ctx)
        if isinstance(step, MarginalizeStep):
            return self._lower_marginalize(step, ctx)
        if isinstance(step, LetStep):
            return (self._lower_let(step, ctx),)
        if isinstance(step, ScoreStep):
            return (self._lower_score(step),)
        if isinstance(step, ReturnStep):
            return (IRReturn(names=step.vars),)
        raise UnsupportedConstruct(
            "qvr-lower", [f"step:{step.kind}"]
        )

    def _lower_sample(
        self, step: SampleStep, ctx: _LowerCtx
    ) -> tuple[IRNode, ...]:
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
            return (self._lower_sample_from_meta(meta, step, ctx),)
        ir_args, arg_names = self._lower_args(
            meta, resolved, ctx,
            event_axes=_event_axis_names(step, ctx),
            axes_index=step.index,
            structural_args=step.args,
        )
        plate = self._build_plate(step, ctx, meta, ir_args)
        constraint = _apply_declared_bounds(
            from_constraint(_resolve_support(meta, ir_args, ctx)),
            plate,
            ctx,
            step.vars[0] if step.vars else step.morphism,
        )
        ir_args, constraint = self._apply_class_index_codomain(
            step.morphism,
            meta,
            ir_args,
            arg_names,
            constraint,
            ctx,
            step.vars[0] if step.vars else step.morphism,
        )
        if len(step.vars) != 1:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    "sample:destructuring-tuple: lower expects one "
                    "bound name per SampleStep after composite "
                    "expansion"
                ],
            )
        head_nodes, ir_args = self._apply_param_map(
            step.morphism,
            step.args,
            resolved.family,
            step.vars[0],
            arg_names,
            ir_args,
            plate,
            ctx,
        )
        return (
            *head_nodes,
            IRSample(
                name=step.vars[0],
                family=resolved.family,
                args=ir_args,
                arg_names=arg_names,
                constraint=constraint,
                plate=plate,
            ),
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
    ) -> tuple[IRNode, ...]:
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
        constraint = _apply_declared_bounds(
            from_constraint(_resolve_support(meta, ir_args, ctx)),
            plate,
            ctx,
            _observe_var(step),
        )
        ir_args, constraint = self._apply_class_index_codomain(
            step.morphism,
            meta,
            ir_args,
            arg_names,
            constraint,
            ctx,
            _observe_var(step),
        )
        head_nodes, ir_args = self._apply_param_map(
            step.morphism,
            step.args,
            resolved.family,
            _observe_var(step),
            arg_names,
            ir_args,
            plate,
            ctx,
            via=step.via,
        )
        return (
            *head_nodes,
            IRObserve(
                name=_observe_var(step),
                family=resolved.family,
                args=ir_args,
                arg_names=arg_names,
                constraint=constraint,
                plate=plate,
                via=step.via,
            ),
        )

    def _lower_marginalize(
        self, step: MarginalizeStep, ctx: _LowerCtx
    ) -> tuple[IRNode, ...]:
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
        constraint = _apply_declared_bounds(
            from_constraint(_resolve_support(meta, ir_args, ctx)),
            plate,
            ctx,
            step.var,
        )
        ir_args, constraint = self._apply_class_index_codomain(
            step.morphism,
            meta,
            ir_args,
            arg_names,
            constraint,
            ctx,
            step.var,
        )
        if step.reduction not in (None, "logsumexp"):
            raise UnsupportedConstruct(
                "qvr-lower",
                [f"marginalize:reduction:{step.reduction}"],
            )
        head_nodes, ir_args = self._apply_param_map(
            step.morphism,
            step.args,
            resolved.family,
            step.var,
            arg_names,
            ir_args,
            plate,
            ctx,
        )
        scope = self._lower_steps(step.scope, ctx)
        return (
            *head_nodes,
            IRMarginalize(
                latent=step.var,
                family=resolved.family,
                args=ir_args,
                arg_names=arg_names,
                constraint=constraint,
                plate=plate,
                reduction="logsumexp",
                scope=scope,
            ),
        )

    def _apply_param_map(
        self,
        morphism_name: str,
        step_args: tuple[DrawArg, ...] | None,
        family: str,
        site: str,
        arg_names: tuple[str, ...],
        ir_args: tuple[IRArg, ...],
        plate: Plate,
        ctx: _LowerCtx,
        *,
        via: str | None = None,
    ) -> tuple[tuple[IRNode, ...], tuple[IRArg, ...]]:
        """Replace a site's family arguments with its morphism's
        parameter map, when the morphism carries one.

        Returns the deterministic bindings the site's arguments are
        computed by, and the arguments themselves. A step that draws
        from anything but a declared kernel morphism with a mapped
        head keeps the arguments it already had, and the pair is
        ``((), ir_args)``.
        """
        pmap = self._param_map_for(morphism_name, family, plate, ctx)
        if pmap is None:
            return (), ir_args
        if via is not None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"param-source:linear:via:{morphism_name}: the "
                    f"observation reads its site through the {via!r} "
                    f"fibration, which regroups the rows the "
                    f"morphism's parameter map is applied to; the map "
                    f"and the fibration have no combined wire form"
                ],
            )
        if arg_names != tuple(head.arg_name for head in pmap.heads):
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"param-source:linear:head-mismatch:{morphism_name}: "
                    f"the site takes arguments {arg_names} where the "
                    f"{family} parameter head supplies "
                    f"{tuple(h.arg_name for h in pmap.heads)}"
                ],
            )
        sources = self._param_map_sources(morphism_name, step_args, ctx)
        declared = _declared_domain_width(morphism_name, ctx)
        supplied = sum(source.width for source in sources)
        if declared != supplied:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"param-source:linear:domain-width:{morphism_name}: "
                    f"the morphism's parameter map reads a "
                    f"{declared}-wide row, and the step conditions it "
                    f"on {supplied} coordinates "
                    f"({', '.join(f'{s.name}:{s.width}' for s in sources)})"
                ],
            )
        pmap = _ParamMap(
            morphism=pmap.morphism,
            family=pmap.family,
            weight=pmap.weight,
            bias=pmap.bias,
            sources=sources,
            axis=pmap.axis,
            width=pmap.width,
            heads=pmap.heads,
        )
        self._record_param_map_inputs(pmap, ctx)
        nodes = self._param_map_head_nodes(pmap, site, plate, ctx)
        return nodes, tuple(
            IRArgRef(name=_head_binding_name(site, head))
            for head in pmap.heads
        )

    def _param_map_for(
        self,
        morphism_name: str,
        family: str,
        plate: Plate,
        ctx: _LowerCtx,
    ) -> _ParamMap | None:
        """The parameter map a step's morphism carries, or None.

        Four conditions have to hold together, and each says
        something the emission would otherwise get wrong:

        1. The step draws from a *declared kernel morphism*. A draw
           from a family (`sample x <- Normal(0, 1)`) names its own
           parameters and has no map.
        2. The declaration's init clause is the bare ``~ Family``
           form and the option block populates no argument slot of
           that family. A declaration that writes its parameters
           (``~ Cauchy(0, 1)``, ``[scale=0.5] ~ Normal``) means them,
           and the emission honours them.
        3. The family is in
           [`_CONDITIONAL_HEADS`][quivers.transpile.lower._CONDITIONAL_HEADS],
           so how the runtime reads its arguments off the map is
           known rather than guessed.
        4. The codomain is a named ``Real`` object and the site's
           plate is exactly that object's width. The map produces one
           row per coordinate of the codomain; a site plated any
           other way is not scoring that row.

        The first three return None (the step keeps the arguments it
        had); the fourth raises, because a mapped family whose head
        shape cannot be read is a gap in this pass rather than a step
        outside its scope.
        """
        decl = ctx.morphisms.get(morphism_name)
        if decl is None:
            return None
        role = _option_identifier(decl.options, "role")
        if role is not None and role != "kernel":
            return None
        heads = _CONDITIONAL_HEADS.get(family)
        if heads is None:
            return None
        if not _has_bare_family_init(decl, family, ctx.family_set):
            return None
        if _declares_head_argument(decl.options, heads):
            return None
        codomain = decl.codomain
        if not isinstance(codomain, TypeName):
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"param-source:linear:codomain:{morphism_name}: the "
                    f"morphism's {family} parameter map produces one "
                    f"row per codomain coordinate, and the codomain is "
                    f"not a named object"
                ],
            )
        width = ctx.real_widths.get(codomain.name)
        if width is None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"param-source:linear:codomain:{morphism_name}: the "
                    f"morphism's {family} parameter map produces one "
                    f"row per coordinate of {codomain.name!r}, which "
                    f"declares no real width"
                ],
            )
        _assert_param_map_plate(morphism_name, plate, codomain.name, width)
        return _ParamMap(
            morphism=morphism_name,
            family=family,
            weight=_param_map_weight_name(morphism_name),
            bias=_param_map_bias_name(morphism_name),
            sources=(),
            axis=codomain.name,
            width=width,
            heads=heads,
        )

    def _param_map_sources(
        self,
        morphism_name: str,
        step_args: tuple[DrawArg, ...] | None,
        ctx: _LowerCtx,
    ) -> tuple[_ParamMapSource, ...]:
        """The conditioning row the map is applied to, in column
        order.

        A step that names arguments (``emission(s_new)``) conditions
        on those bindings; a step that names none conditions on the
        program's own input, which is what the runtime hands a step
        whose argument list is absent.

        The row is bindings the whole way down. A draw whose head
        names a declared morphism supplies that morphism's input, not
        the family's parameters, so a literal in the list is not a
        form the language admits: the compiler rejects it outright
        (``literal argument not allowed for named morphism``), and it
        names no coordinate the map could read.
        """
        if not step_args:
            return _program_domain_sources(ctx)
        out: list[_ParamMapSource] = []
        for arg in step_args:
            if not isinstance(arg, DrawArgName):
                raise UnsupportedConstruct(
                    "qvr-lower",
                    [
                        f"param-source:linear:argument:{morphism_name}: "
                        f"the parameter map conditions on a value, and "
                        f"the step supplies a {arg.kind} argument "
                        f"instead of a binding to read it from"
                    ],
                )
            width = self._binding_width(arg.text, ctx)
            if width is None:
                raise UnsupportedConstruct(
                    "qvr-lower",
                    [
                        f"param-source:linear:argument-width:"
                        f"{morphism_name}: the conditioning binding "
                        f"{arg.text!r} carries no statically known "
                        f"width, so the map's input row cannot be "
                        f"laid out"
                    ],
                )
            out.append(_ParamMapSource(name=arg.text, width=width))
        return tuple(out)

    def _binding_width(self, name: str, ctx: _LowerCtx) -> int | None:
        """Coordinate count of a binding a step conditions on.

        A drawn site carries its width in the plate lowering already
        recorded for it; a program-input factor carries it in the
        domain wire table.
        """
        plate = ctx.bound_plates.get(name)
        if plate is not None:
            dims = (*plate.event_dims, *plate.batch_dims)
            if len(dims) == 1 and isinstance(dims[0], DimStatic):
                return dims[0].size
            return None
        for source in _program_domain_sources(ctx):
            if source.name == name:
                return source.width
        return None

    def _record_param_map_inputs(
        self, pmap: _ParamMap, ctx: _LowerCtx
    ) -> None:
        """Declare the map's weight and bias as data inputs.

        Two sites drawing the same morphism share one map, exactly as
        the runtime registers one module for the morphism and reads it
        from every step that names it, so a second site records the
        same two entries rather than a second pair.
        """
        rows = _param_map_rows(pmap)
        columns = _param_map_domain_width(pmap)
        weight_plate = Plate(
            event_dims=(),
            batch_dims=(
                DimStatic(size=rows, name=f"{pmap.morphism}_param_row"),
                DimStatic(size=columns, name=f"{pmap.morphism}_param_col"),
            ),
        )
        bias_plate = Plate(
            event_dims=(),
            batch_dims=(
                DimStatic(size=rows, name=f"{pmap.morphism}_param_row"),
            ),
        )
        for name, plate in (
            (pmap.weight, weight_plate),
            (pmap.bias, bias_plate),
        ):
            previous = ctx.param_map_inputs.get(name)
            if previous is not None and previous != plate:
                raise UnsupportedConstruct(
                    "qvr-lower",
                    [
                        f"param-source:linear:shape-conflict:{name}: the "
                        f"same parameter map is read at two shapes"
                    ],
                )
            ctx.param_map_inputs[name] = plate

    def _param_map_head_nodes(
        self, pmap: _ParamMap, site: str, plate: Plate, ctx: _LowerCtx
    ) -> tuple[IRNode, ...]:
        """The deterministic bindings one site's arguments are read
        from: one contraction per head, and the floor for a head that
        carries one.

        An ``identity`` head is the contraction itself. An
        ``exp_floor`` head is the exponentiated contraction bound to
        a raw name, then floored coordinatewise at
        [`EPS`][quivers.core._util.EPS], which is the runtime's
        ``log_sigma.exp().clamp(min=EPS)``. The floor is the only
        part still written per coordinate, and it costs one term per
        coordinate rather than one per coordinate pair.
        """
        out: list[IRNode] = []
        for index, head in enumerate(pmap.heads):
            name = _head_binding_name(site, head)
            _assert_free_binding(name, ctx)
            if head.transform == "identity":
                out.append(
                    IRDeterministic(
                        name=name,
                        expr=_affine_map_expr(pmap, index, "identity"),
                        constraint=CSReal(),
                        plate=plate,
                    )
                )
                continue
            raw = _head_raw_binding_name(site, head)
            _assert_free_binding(raw, ctx)
            out.append(
                IRDeterministic(
                    name=raw,
                    expr=_affine_map_expr(pmap, index, "exp"),
                    constraint=CSReal(),
                    plate=plate,
                )
            )
            out.append(
                IRDeterministic(
                    name=name,
                    expr=LetExprFactor(
                        binders=(
                            LetFactorBinder(
                                var=f"_{pmap.axis.lower()}_i",
                                index=TypeName(name=pmap.axis),
                            ),
                        ),
                        cases=tuple(
                            LetFactorCase(
                                label=i, value=_floor_expr(raw, i)
                            )
                            for i in range(pmap.width)
                        ),
                    ),
                    constraint=CSPositive(),
                    plate=plate,
                )
            )
        return tuple(out)

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
        present; otherwise, for a scalar family, from `step.index`
        and from the width of the morphism's `Real N` codomain, in
        that order. A step over a `Real N` codomain draws one value
        per coordinate, so the width replicates the family whether or
        not the step also names a sequence axis.
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
        # Scalar family reached through a morphism whose codomain is a
        # `Real N` object: the step draws one value per coordinate of
        # that object, so the codomain width is a replication axis.
        # `sample s_new <- transition_cell` on `transition_cell :
        # Driver * State -> State` with `object State : Real 4` is
        # four iid scalar draws, not one. When the step also carries a
        # `: Axis` index the two stack: the index replicates the whole
        # codomain vector, so it is the outer batch dim.
        codomain_width = (
            self._codomain_width(step.morphism, ctx)
            if event_dim == 0
            else None
        )
        if step.index is None:
            if codomain_width is None:
                return Plate(event_dims=(), batch_dims=())
            return Plate(event_dims=(), batch_dims=(codomain_width,))
        dim = self._object_expr_dim(step.index, ctx)
        if event_dim == 0:
            if codomain_width is None:
                return Plate(event_dims=(), batch_dims=(dim,))
            return Plate(
                event_dims=(), batch_dims=(dim, codomain_width),
            )
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

    def _codomain_alphabet(
        self, morphism_name: str, ctx: _LowerCtx
    ) -> DimStatic | None:
        """The alphabet the declared morphism's codomain names, as a
        [`DimStatic`][quivers.transpile.ir.DimStatic] carrying the
        class count and the codomain's own name, or `None` when the
        step names no declared morphism or the codomain is not a
        finite object.

        ``morphism lm_head : Hidden -> Token ~ Categorical`` says the
        per-row value of every draw through `lm_head` is a `Token`,
        so the draw's alphabet is `|Token|`. The plate the draw is
        replicated over (`observe next_token : Resp <- lm_head(h)`)
        says how many rows there are and nothing about how wide each
        row's alphabet is, so it is not consulted here.
        """
        decl = ctx.morphisms.get(morphism_name)
        if decl is None:
            return None
        shape = _object_expr_shape(
            decl.codomain, ctx.shapes, (morphism_name,)
        )
        if shape is None or not shape.finite or shape.extent is None:
            return None
        return DimStatic(
            size=shape.extent, name=_axis_expr_name(decl.codomain),
        )

    def _apply_class_index_codomain(
        self,
        morphism_name: str,
        meta: FamilyMeta,
        ir_args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        constraint: ConstraintSpec,
        ctx: _LowerCtx,
        name: str,
    ) -> tuple[tuple[IRArg, ...], ConstraintSpec]:
        """Restate a class-index draw's support and alphabet width
        from the declared morphism's codomain.

        A family whose value is a subscript into its own alphabet
        (see
        [`class_index_outcome`][quivers.transpile.family_meta.class_index_outcome])
        carries no width of its own: the sentinel used to read its
        support is built from placeholder probabilities, so the IR
        would otherwise state the placeholder's two classes. The
        width is a positional fact, `|B|` for a morphism `f : A -> B`,
        and this is where the IR picks it up.

        The alphabet argument is restated along with the support when
        it names a binding with no width of its own, because every
        backend has to widen it before the family will accept it. An
        argument that already states a width keeps it, and a width
        that disagrees with the codomain raises: the two describe
        different alphabets, and scoring the value against either one
        would contradict the other.
        """
        outcome = class_index_outcome(meta)
        if outcome is None:
            return ir_args, constraint
        alphabet = self._codomain_alphabet(morphism_name, ctx)
        if alphabet is None:
            return ir_args, constraint
        width = alphabet.size
        if width < 2:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"class-index:{name}:codomain-width:{width}: a "
                    f"{meta.qvr_name} draw through {morphism_name!r} "
                    f"needs a codomain naming at least two classes"
                ],
            )
        arg_dim = DimStatic(
            size=width - outcome.extent_offset, name=alphabet.name,
        )
        out = list(ir_args)
        for i, arg_name in enumerate(arg_names):
            if arg_name not in outcome.alphabet_args:
                continue
            out[i] = _retarget_alphabet_arg(
                out[i], arg_dim, name, ctx,
            )
        return tuple(out), CSIntegerInterval(lower=0, upper=width - 1)

    def _codomain_width(
        self, morphism_name: str, ctx: _LowerCtx
    ) -> Dim | None:
        """Return the `Real N` width of `morphism_name`'s codomain as
        a `DimStatic`, or `None` when the step names no declared
        morphism or the codomain is not a single `Real N` object."""
        decl = ctx.morphisms.get(morphism_name)
        if decl is None or not isinstance(decl.codomain, TypeName):
            return None
        width = ctx.real_widths.get(decl.codomain.name)
        if width is None:
            return None
        return DimStatic(size=width, name=decl.codomain.name)

    def _axis_dim(self, axis_name: str, ctx: _LowerCtx) -> Dim:
        """Convert an axis name into a `DimStatic` read off the
        cardinality table.

        See [`_axis_dim_at`][quivers.transpile.lower._axis_dim_at] for
        why a name the table does not record raises.
        """
        return _axis_dim_at(axis_name, ctx)

    def _object_expr_dim(
        self, expr: ObjectExpr, ctx: _LowerCtx
    ) -> Dim:
        """Convert an `ObjectExpr` into a `Dim`.

        A product axis (`observe y : A * B`) flattens to the product
        of its factors' cardinalities, which is what the runtime's
        `ProductSet` does with the same expression: `|A| = 4` and
        `|B| = 6` give one 24-row plate. The flattened axis is named
        by joining the factor names so the emitted loop variable
        still reads back to the source expression.
        """
        if isinstance(expr, TypeName):
            return self._axis_dim(expr.name, ctx)
        if isinstance(expr, DiscreteConstructor) and expr.args:
            size = axis_shape(expr, ctx.cards)
            if size is None:
                raise UnsupportedConstruct(
                    "qvr-lower",
                    [
                        f"object-expr:discrete_constructor:"
                        f"{expr.args[0]}: a FinSet axis size must be "
                        f"an integer literal or the name of an object "
                        f"with a known cardinality"
                    ],
                )
            return DimStatic(size=size, name="anon")
        if isinstance(expr, ObjectProduct):
            size = axis_shape(expr, ctx.cards)
            if size is None:
                raise UnsupportedConstruct(
                    "qvr-lower",
                    [
                        f"object-expr:object_product:"
                        f"{_axis_expr_name(expr)}: every factor of a "
                        f"product axis needs a statically known "
                        f"cardinality to flatten"
                    ],
                )
            return DimStatic(size=size, name=_axis_expr_name(expr))
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
        * The `Real N` factors of `ProgramDecl.domain` (the value the
          Kleisli program is applied to).
        * Scalar `ProgramDecl.type_params` (e.g. `alpha : Real`).
        * Free names in let / score expressions.
        * Free names referenced in bracket-indexed arg expressions
          (`mu[cls]` references `cls` and `mu`).
        * `observe ... [via=fibration]` fibrations.
        * Observed variables themselves (the rhs of `observe`).
        * The weight and bias of every declared morphism's parameter
          map, whose shapes lowering recorded on the context as it
          built the heads that read them.
        """
        bound = self._bound_names(body)
        used = self._used_names(body)
        domain_inputs = self._domain_inputs(program, bound, ctx)
        seen_domain = {inp.name for inp in domain_inputs}
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
        # Parameter-map inputs: the numbers a declared morphism's
        # affine map is made of. Their shapes are declared rather than
        # inferred (the map's row count is the family's head layout
        # over the codomain, its column count the morphism's domain),
        # so they are emitted from the recorded table rather than left
        # to the free-name pass, which would type them as scalars.
        map_inputs: list[IRDataInput] = [
            IRDataInput(
                name=name,
                constraint=CSReal(),
                plate=plate,
            )
            for name, plate in ctx.param_map_inputs.items()
        ]
        seen_map = {inp.name for inp in map_inputs}

        seen_param_names = (
            seen_domain | seen_param | seen_via | seen_obs | seen_map
        )
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

        return tuple(
            domain_inputs
            + param_inputs
            + via_inputs
            + obs_inputs
            + map_inputs
            + free_inputs
        )

    def _domain_inputs(
        self,
        program: ProgramDecl,
        bound: set[str],
        ctx: _LowerCtx,
    ) -> list[IRDataInput]:
        """Emit one `IRDataInput` per `Real N` factor of the
        program's domain.

        A Kleisli program `p : Driver * State -> State` is applied to
        a value of its domain, and the morphisms its body names read
        that value; without a wire name for it the emitted program has
        nothing to thread the per-call input through. Only the
        `Real N` factors carry a value: a `FinSet N` factor names the
        index axis the program is replicated over, which the steps
        read through their own `: Axis` annotations rather than as a
        datum.

        The factor's object name, lowercased, is the wire name, and
        the object's width is the input's event dim. A name a step
        already binds, or that a second factor already claimed, has no
        unambiguous wire form and raises.
        """
        out: list[IRDataInput] = []
        seen: set[str] = set()
        for factor in object_factors(program.domain):
            if not isinstance(factor, TypeName):
                continue
            width = ctx.real_widths.get(factor.name)
            if width is None:
                continue
            name = factor.name.lower()
            if name in seen or name in bound:
                clash = (
                    "an earlier factor of the same domain claimed it"
                    if name in seen
                    else "a step in the body already binds it"
                )
                raise UnsupportedConstruct(
                    "qvr-lower",
                    [
                        f"program-domain:name-collision:{name}: the "
                        f"domain factor {factor.name!r} of program "
                        f"{program.name!r} lowers to a data input "
                        f"named {name!r}, and {clash}; rename the "
                        f"object or the binding so the program input "
                        f"has an unambiguous wire name"
                    ],
                )
            seen.add(name)
            plate = Plate(
                event_dims=(DimStatic(size=width, name=factor.name),),
                batch_dims=(),
            )
            out.append(
                IRDataInput(
                    name=name,
                    constraint=_apply_declared_bounds(
                        CSReal(), plate, ctx, name,
                    ),
                    plate=plate,
                )
            )
        return out

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


class RealBounds(dx.Model):
    """The ``{low=..., high=...}`` bounds a continuous object declares.

    ``object Rate : Real 3 {low=0.0, high=1.0}`` names the box
    ``[0, 1]^3`` rather than ``R^3``, so a variable whose event axis
    or per-row value space is `Rate` lives in that box. Renderers
    turn the pair into the target's own bounded declaration
    (Stan's ``<lower=0, upper=1>``, JAGS' ``T(0, 1)``, ...); a
    variant with neither bound set is the unbounded default.
    """

    low: float | None = None
    high: float | None = None

    @property
    def is_bounded(self) -> bool:
        """True when the declaration constrains at least one side."""
        return self.low is not None or self.high is not None


class ObjectShape(dx.Model):
    """The transpile-visible shape of one ``object`` declaration.

    Three independent readings, one per position an object name can
    occupy (see the module docstring):

    * `extent` is the size the object contributes as an *axis*: the
      cardinality of a finite object (`FinSet N`, an enum set, a
      product of finite factors) and the total coordinate count of a
      continuous one (`Real 28 28` is 784 coordinates wide).
    * `real_width` is set only for `Real`, the one constructor whose
      value is a plain real vector, and is what a program-domain
      factor or a morphism codomain reads to size its wire.
    * `bounds` carries the constructor's ``{low=..., high=...}``.

    `finite` separates the two readings of `extent`: a finite object
    has `|A|` elements, so its extent is an alphabet a class index
    can range over, while a continuous one has `extent` coordinates
    and names no alphabet at all.
    """

    extent: int | None = None
    real_width: int | None = None
    finite: bool = False
    bounds: RealBounds = dx.field(default_factory=RealBounds)


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
    real_widths: dict[str, int] = dx.field(default_factory=dict)
    bounds: dict[str, RealBounds] = dx.field(default_factory=dict)
    shapes: dict[str, ObjectShape] = dx.field(default_factory=dict)
    bound_plates: dict[str, Plate] = dx.field(
        default_factory=dict, opaque=True
    )
    bound_kinds: dict[str, str] = dx.field(
        default_factory=dict, opaque=True
    )
    alphabet_event_dims: dict[str, DimStatic] = dx.field(
        default_factory=dict, opaque=True
    )
    param_map_inputs: dict[str, Plate] = dx.field(
        default_factory=dict, opaque=True
    )
    reserved_names: frozenset[str] = frozenset()


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


def _size_arg_value(
    arg: str,
    table: dict[str, ObjectShape],
    names: tuple[str, ...],
    constructor: str,
) -> int:
    """Resolve one constructor size argument to an integer.

    An integer literal is itself; a bare name is the extent of a
    previously-declared object, matching the runtime's
    ``_eval_size_arg``. Anything else raises rather than silently
    dropping the declaration, because an object whose size the
    transpile cannot read would otherwise reach a renderer as a
    free ``N_<name>`` the target never declares.
    """
    if arg.isdigit():
        return int(arg)
    prior = table.get(arg)
    if prior is not None and prior.extent is not None:
        return prior.extent
    raise UnsupportedConstruct(
        "qvr-lower",
        [
            f"object:{'/'.join(names)}:{constructor}-size:{arg}: a "
            f"size argument must be an integer literal or the name "
            f"of an object declared earlier in the module with a "
            f"known extent"
        ],
    )


def _continuous_bounds(expr: ContinuousConstructor) -> RealBounds:
    """Read ``{low=..., high=...}`` off a continuous constructor."""
    low = expr.kwargs.get("low")
    high = expr.kwargs.get("high")
    if isinstance(low, str) or isinstance(high, str):
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"object:{expr.constructor}-bound:{low!r}/{high!r}: "
                f"`low=` and `high=` must be numeric literals"
            ],
        )
    return RealBounds(
        low=None if low is None else float(low),
        high=None if high is None else float(high),
    )


def _object_expr_shape(
    expr: ObjectExpr,
    table: dict[str, ObjectShape],
    names: tuple[str, ...],
) -> ObjectShape | None:
    """Return the [`ObjectShape`][quivers.transpile.lower.ObjectShape]
    a type expression denotes, or `None` when the expression names no
    statically-sized object (a `FreeResiduated` universe, a slash, an
    effect-apply)."""
    if isinstance(expr, TypeName):
        if expr.name.isdigit():
            return ObjectShape(extent=int(expr.name), finite=True)
        return table.get(expr.name)
    if isinstance(expr, DiscreteConstructor):
        if not expr.args:
            return None
        return ObjectShape(
            extent=_size_arg_value(
                expr.args[0], table, names, expr.constructor
            ),
            finite=True,
        )
    if isinstance(expr, ContinuousConstructor):
        if not expr.args:
            return None
        sizes = [
            _size_arg_value(a, table, names, expr.constructor)
            for a in expr.args
        ]
        bounds = _continuous_bounds(expr)
        if expr.constructor == "Real":
            width = math.prod(sizes)
            return ObjectShape(
                extent=width, real_width=width, bounds=bounds,
            )
        # Every other continuous constructor takes its leading
        # argument as the space's dimension; the flattened width of
        # a matrix-valued one (`Covariance n` is n * n reals) is not
        # a plain real vector, so no `real_width` is recorded.
        return ObjectShape(extent=sizes[0], bounds=bounds)
    if isinstance(expr, ObjectProduct):
        factors = [
            _object_expr_shape(f, table, names)
            for f in object_factors(expr)
        ]
        if any(f is None or f.extent is None for f in factors):
            return None
        total = 1
        finite = True
        for f in factors:
            assert f is not None and f.extent is not None
            total *= f.extent
            finite = finite and f.finite
        return ObjectShape(extent=total, finite=finite)
    return None


def object_shapes(module: Module) -> dict[str, ObjectShape]:
    """Return name -> [`ObjectShape`][quivers.transpile.lower.ObjectShape]
    for every `object` declaration in `module`, in source order.

    Declarations are read in order so a later one can size itself
    from an earlier name (``object M : FinSet N5``,
    ``object Grid : Real Rows Cols``) exactly as the runtime's
    resolver does. A declaration whose value has no static size (a
    `FreeResiduated` universe, a free monoid, a residuated slash)
    contributes no entry, and every downstream caller treats an
    absent name the way it always has.
    """
    out: dict[str, ObjectShape] = {}
    for stmt in module.statements:
        if not isinstance(stmt, ObjectDecl):
            continue
        init = stmt.init
        shape: ObjectShape | None = None
        if isinstance(init, TypeEnumSet):
            shape = ObjectShape(
                extent=len(init.elements), finite=True,
            )
        elif isinstance(init, TypeFromExpr):
            shape = _object_expr_shape(init.expr, out, stmt.names)
        if shape is None:
            continue
        for name in stmt.names:
            out[name] = shape
    return out


def continuous_object_widths(module: Module) -> dict[str, int]:
    """Return name -> width for every ``Real ...`` object declaration.

    ``object State : Real 4`` names a value in R^4: a program with
    that object in its domain is applied to a 4-wide vector, and a
    morphism with it as codomain writes one. ``object Img : Real 28
    28`` names a value in R^784, the runtime's own reading of a
    multi-argument `Real`, so the width is the product of the
    arguments rather than the first of them. ``object Step : FinSet
    64`` names an index axis instead, so it is absent here even
    though
    [`object_cardinalities`][quivers.transpile.lower.object_cardinalities]
    records its 64. The other continuous constructors (``Simplex``,
    ``Sphere``, ``Covariance``, ...) carry a support a plain real
    vector does not describe, so they are absent as well and a caller
    that needs them has to read the constructor itself.
    """
    return {
        name: shape.real_width
        for name, shape in object_shapes(module).items()
        if shape.real_width is not None
    }


def object_bounds(module: Module) -> dict[str, RealBounds]:
    """Return name -> [`RealBounds`][quivers.transpile.lower.RealBounds]
    for every continuous object that declares ``{low=...}`` or
    ``{high=...}``. Objects with neither bound are absent."""
    return {
        name: shape.bounds
        for name, shape in object_shapes(module).items()
        if shape.bounds.is_bounded
    }


def object_factors(expr: ObjectExpr) -> tuple[ObjectExpr, ...]:
    """Flatten a product object expression into its factors, in
    source order. A non-product expression is its own single
    factor."""
    if isinstance(expr, ObjectProduct):
        out: list[ObjectExpr] = []
        for comp in expr.components:
            out.extend(object_factors(comp))
        return tuple(out)
    return (expr,)


def _spec_real_range(
    spec: ConstraintSpec,
) -> tuple[float, float] | None:
    """The closed real range a `ConstraintSpec` denotes, or `None`
    when the spec is not a real-valued support at all (an integer
    outcome, a simplex, a Cholesky factor, ...)."""
    if isinstance(spec, (CSReal, CSRealVector, CSRealMatrix)):
        return (float("-inf"), float("inf"))
    if isinstance(spec, CSPositive):
        return (0.0, float("inf"))
    if isinstance(spec, CSUnitInterval):
        return (0.0, 1.0)
    if isinstance(spec, CSInterval):
        return (spec.lower, spec.upper)
    return None


def _range_spec(
    low: float, high: float, name: str, source: str,
) -> ConstraintSpec:
    """The `ConstraintSpec` for a closed real range.

    A one-sided bound other than ``> 0`` has no IR form, so it
    raises rather than reaching a renderer as an unbounded real: the
    declared support would then be wider in the target than in the
    source program.
    """
    lo_open = low == float("-inf")
    hi_open = high == float("inf")
    if lo_open and hi_open:
        return CSReal()
    if hi_open and low == 0.0:
        return CSPositive(strict=True)
    if lo_open or hi_open:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"object-bounds:one-sided:{source}:{name}: the IR "
                f"carries a real support that is unbounded, "
                f"positive, or a closed interval; declare both "
                f"`low=` and `high=` so the bound reaches the target"
            ],
        )
    if low == 0.0 and high == 1.0:
        return CSUnitInterval()
    return CSInterval(lower=low, upper=high)


def _bounds_of_axes(
    axis_names: tuple[str, ...],
    ctx: _LowerCtx,
    name: str,
) -> RealBounds | None:
    """The declared bounds the axes of one binding agree on.

    An unbounded axis contributes nothing. Two axes declaring
    different boxes describe no single support, so they raise rather
    than one silently winning.
    """
    found: list[tuple[str, RealBounds]] = []
    for axis in axis_names:
        bounds = ctx.bounds.get(axis)
        if bounds is None:
            continue
        if any(prior == bounds for _, prior in found):
            continue
        found.append((axis, bounds))
    if not found:
        return None
    if len(found) > 1:
        clashing = ", ".join(axis for axis, _ in found)
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"object-bounds:conflicting-axes:{name}:{clashing}: "
                f"the axes of one binding declare different "
                f"`{{low=..., high=...}}` boxes, so the binding has "
                f"no single support"
            ],
        )
    return found[0][1]


def _apply_declared_bounds(
    spec: ConstraintSpec,
    plate: Plate,
    ctx: _LowerCtx,
    name: str,
) -> ConstraintSpec:
    """Narrow a family's support by the ``{low=..., high=...}`` box
    the binding's own axis objects declare.

    ``sample z : Batch <- Normal(0, 1) [over=Rate]`` with
    ``object Rate : Real 3 {low=0.0, high=1.0}`` draws a value of
    `Rate`, so the value lives in ``[0, 1]^3``: the family's real
    support intersected with the declared box. A box on a support
    that is not real-valued at all contradicts the family rather
    than narrowing it, and an empty intersection describes no value,
    so both raise.
    """
    axis_names = tuple(
        dim.name
        for dim in (*plate.event_dims, *plate.batch_dims)
    )
    bounds = _bounds_of_axes(axis_names, ctx, name)
    if bounds is None:
        return spec
    current = _spec_real_range(spec)
    if current is None:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"object-bounds:non-real-support:{name}:{spec.kind}: "
                f"a `{{low=..., high=...}}` box narrows a real "
                f"support, and this binding's support is not real"
            ],
        )
    low = max(
        current[0],
        bounds.low if bounds.low is not None else float("-inf"),
    )
    high = min(
        current[1],
        bounds.high if bounds.high is not None else float("inf"),
    )
    if low >= high:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"object-bounds:empty-support:{name}:[{low}, {high}]: "
                f"the declared box and the family's own support do "
                f"not overlap"
            ],
        )
    return _range_spec(low, high, name, "axis")


def _assert_alphabet_width(
    stated: int, dim: DimStatic, name: str, source: str,
) -> None:
    """Reject an alphabet argument whose own width contradicts the
    width the declared codomain names.

    The two would score the draw against different alphabets, and no
    target can hold both, so the disagreement is reported at its
    source rather than resolved in favour of either side.
    """
    if stated == dim.size:
        return
    raise UnsupportedConstruct(
        "qvr-lower",
        [
            f"class-index:{name}:alphabet-width:{source}: the "
            f"argument states {stated} classes and the declared "
            f"codomain {dim.name!r} names {dim.size}; the draw has "
            f"one alphabet, so make the two agree"
        ],
    )


def _retarget_alphabet_arg(
    arg: IRArg, dim: DimStatic, name: str, ctx: _LowerCtx,
) -> IRArg:
    """Widen a class-index family's alphabet argument to `dim`.

    Anything that carries its own event shape (an indexed reference,
    a literal vector, a matrix row, a reference to a binding that
    already has event dims) states its width itself and passes
    through. A broadcast already built from the step's axes is
    re-targeted, its earlier shape having been read off the plate
    rather than the codomain.

    A bare reference splits by what binds it. A `sample` states its
    own shape in the source (`<- Family(...) [over=...]`), so a
    scalar one is broadcast to the alphabet width at the call site
    and a plated one raises: a per-row scalar draw is not a per-row
    probability vector, and widening it would silently restate the
    step. A `let` or a free data input carries no declared shape at
    all, so its width is recorded for
    [`_propagate_alphabet_event_dims`][quivers.transpile.lower._propagate_alphabet_event_dims]
    to push through the shape-inference pass.
    """
    if isinstance(arg, IRArgBroadcast):
        if arg.target_shape == (dim.size,):
            return arg
        return IRArgBroadcast(
            value=arg.value, target_shape=(dim.size,)
        )
    if isinstance(arg, IRArgList):
        _assert_alphabet_width(len(arg.elements), dim, name, "list")
        return arg
    if isinstance(arg, IRArgMatrix):
        row = arg.rows[0] if arg.rows else None
        if row is not None:
            _assert_alphabet_width(
                len(row.elements), dim, name, "matrix-row",
            )
        return arg
    if not isinstance(arg, IRArgRef) or arg.indices:
        return arg
    bound = ctx.bound_plates.get(arg.name)
    if bound is not None and bound.event_dims:
        trailing = bound.event_dims[-1]
        if isinstance(trailing, DimStatic):
            _assert_alphabet_width(
                trailing.size, dim, name, f"binding:{arg.name}",
            )
        return arg
    if ctx.bound_kinds.get(arg.name) in ("sample", "marginalize"):
        if bound is not None and bound.batch_dims:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"class-index:{name}:alphabet-arg:{arg.name}: the "
                    f"alphabet argument is a plated scalar draw, so "
                    f"it carries one number per row where the family "
                    f"needs one probability per class; draw it "
                    f"`[over=...]` the codomain instead"
                ],
            )
        return IRArgBroadcast(value=arg, target_shape=(dim.size,))
    prior = ctx.alphabet_event_dims.get(arg.name)
    if prior is not None and prior.size != dim.size:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"class-index:{name}:alphabet-width:binding:"
                f"{arg.name}: the same binding feeds alphabets of "
                f"{prior.size} and {dim.size} classes; one binding "
                f"carries one width, so split it in two"
            ],
        )
    ctx.alphabet_event_dims[arg.name] = dim
    return arg


def _record_bound_plate(node: IRNode, ctx: _LowerCtx) -> None:
    """Record the plate a lowered node binds its name to, and what
    kind of step bound it, so a later step can ask whether that name
    already carries an event shape and whether its shape is declared
    in the source or inferred."""
    if isinstance(node, (IRSample, IRObserve, IRDeterministic)):
        ctx.bound_plates[node.name] = node.plate
        ctx.bound_kinds[node.name] = node.kind
    elif isinstance(node, IRMarginalize):
        ctx.bound_plates[node.latent] = node.plate
        ctx.bound_kinds[node.latent] = node.kind


def _axis_expr_name(expr: ObjectExpr) -> str:
    """A wire-safe identifier for an axis object expression.

    Names the plate a renderer emits a loop variable for, so it has
    to survive into every target's identifier syntax: a named object
    keeps its name, a product joins its factors with an underscore,
    and an anonymous `FinSet N` is `anon`.
    """
    if isinstance(expr, TypeName):
        return expr.name
    if isinstance(expr, ObjectProduct):
        return "_".join(
            _axis_expr_name(f) for f in object_factors(expr)
        )
    return "anon"


def object_cardinalities(module: Module) -> dict[str, int]:
    """Return name -> axis extent for every statically-sized object
    declaration.

    The extent is what the object contributes in an *axis* position:
    the cardinality of a finite object (`FinSet N`, an enum set, a
    product of finite factors) and the coordinate count of a
    continuous one (`Real 28 28` is 784 wide). An object whose value
    has no static size contributes no entry.
    """
    return {
        name: shape.extent
        for name, shape in object_shapes(module).items()
        if shape.extent is not None
    }


def axis_shape(
    expr: ObjectExpr, cards: dict[str, int]
) -> int | None:
    """Return the cardinality of an axis object expression.

    A product axis (`A * B`) is the flattened cardinality of its
    factors, matching the runtime's `ProductSet`; a factor whose
    size `cards` does not record makes the whole product unsized.
    """
    if isinstance(expr, TypeName):
        if expr.name.isdigit():
            return int(expr.name)
        return cards.get(expr.name)
    if isinstance(expr, DiscreteConstructor) and expr.args:
        try:
            return int(expr.args[0])
        except ValueError:
            return cards.get(expr.args[0])
    if isinstance(expr, ObjectProduct):
        total = 1
        for factor in object_factors(expr):
            size = axis_shape(factor, cards)
            if size is None:
                return None
            total *= size
        return total
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
        if isinstance(n, LetExprAffineMap):
            visit(n.weight, bound)
            for source in n.sources:
                visit(source.value, bound)
            visit(n.bias, bound)
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
    """Convert an axis name into a
    [`DimStatic`][quivers.transpile.ir.DimStatic] read off the
    cardinality table, without instantiating
    [`Lower`][quivers.transpile.lower.Lower].

    A name the table does not record names no statically-sized
    object, which the QVR compiler already refuses (``undefined
    object or space``). Sizing the axis by an invented ``N_<name>``
    instead would put a free identifier in every target's loop bound
    and array extent that no emitted block declares, so it raises.
    """
    size = ctx.cards.get(axis_name)
    if size is None:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"axis:unknown-cardinality:{axis_name}: an axis must "
                f"name an object declared with a statically known "
                f"cardinality; declare `object {axis_name} : FinSet "
                f"<n>` (or an enum set) so the emitted plate has a "
                f"size the target can read"
            ],
        )
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
    if isinstance(expr, LetExprAffineMap):
        _collect_let_expr_var_names(expr.weight, out)
        for source in expr.sources:
            _collect_let_expr_var_names(source.value, out)
        _collect_let_expr_var_names(expr.bias, out)
        return


def _let_expr_needs_plate(expr: LetExprNode) -> bool:
    """True iff `expr` denotes a value whose shape inherits from its
    free variables (arithmetic, index, call, method-call), false iff
    it is shape-self-determining (literal, string, list, factor,
    lambda, affine map).

    The plate-propagation pass uses this to decide whether a
    `let v = expr` may pick up the surrounding observe/sample's
    batch_dims. A `let sigma = 0.5` (literal) never inherits a
    plate; a `let mu = a + b * x_design` (arithmetic that reads a
    free name) may, when one of its free names is itself plated. A
    [`LetExprAffineMap`][quivers.transpile.ir.LetExprAffineMap] is
    the `rows`-wide vector its own row block names, whatever the
    shapes it reads, so it never inherits either.
    """
    if isinstance(
        expr, (LetExprLiteral, LetExprString, LetExprList, LetExprFactor,
               LetExprLambda, LetExprAffineMap)
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


def _propagate_alphabet_event_dims(
    inputs: tuple[IRDataInput, ...],
    body: tuple[IRNode, ...],
    widths: dict[str, DimStatic],
) -> tuple[tuple[IRDataInput, ...], tuple[IRNode, ...]]:
    """Widen every shape-inferred name that feeds a class-index
    family's alphabet slot to the alphabet's own width.

    ``morphism lm_head : Hidden -> Token ~ Categorical`` used as
    ``observe next_token : Resp <- lm_head(h)`` says each of the 32
    `Resp` rows is scored against a `Token`-wide probability vector,
    so `h` is one 256-wide row per response, not one number per
    response. `widths` names the bindings that carry the alphabet,
    keyed by name; each is widened to a single event dim, and the
    pass then recurses into the free names a widened `let` reads so
    the arithmetic that computes the row is vector-valued end to
    end.

    A binding that already carries event dims states its own width
    and is left alone, as is a `sample`, whose shape the source
    declares. A free name reaches here as an
    [`IRDataInput`][quivers.transpile.ir.IRDataInput] and is widened
    in place.
    """
    if not widths:
        return inputs, body
    lets_by_name: dict[str, IRDeterministic] = {
        node.name: node
        for node in _walk_nodes(body)
        if isinstance(node, IRDeterministic)
    }
    input_map: dict[str, int] = {
        inp.name: i for i, inp in enumerate(inputs)
    }
    inputs_list = list(inputs)

    def widen(name: str, dim: DimStatic, seen: set[str]) -> None:
        if name in seen:
            return
        seen.add(name)
        idx = input_map.get(name)
        if idx is not None:
            cur = inputs_list[idx]
            if cur.plate.event_dims:
                return
            inputs_list[idx] = IRDataInput(
                name=cur.name,
                constraint=cur.constraint,
                plate=Plate(
                    event_dims=(dim,),
                    batch_dims=cur.plate.batch_dims,
                ),
            )
            return
        let = lets_by_name.get(name)
        if let is None or let.plate.event_dims:
            return
        lets_by_name[name] = IRDeterministic(
            name=let.name,
            expr=let.expr,
            constraint=let.constraint,
            plate=Plate(
                event_dims=(dim,), batch_dims=let.plate.batch_dims,
            ),
        )
        leaves: set[str] = set()
        _collect_let_expr_var_names(let.expr, leaves)
        for leaf in leaves:
            widen(leaf, dim, seen)

    for name, dim in widths.items():
        widen(name, dim, set())
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
    if isinstance(expr, LetExprAffineMap):
        _collect_integer_index_names(expr.weight, out)
        for source in expr.sources:
            _collect_integer_index_names(source.value, out)
        _collect_integer_index_names(expr.bias, out)
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
    "exported_return_names",
    "free_names_in_arg",
    "free_vars_in_let",
    "inline_list_lets",
    "lower_factors",
    "object_cardinalities",
    "pick_program",
]
