"""`RendererBase`: shared machinery for every transpile renderer.

Each backend's renderer is a [`Renderer`][quivers.transpile.renderers._base.Renderer]:
a class with one public method `render(ir: IRProgram) -> panproto.Schema`
and four private dispatch points (`declare`, `sample`, `marginalize`,
`broadcast`). `RendererBase` provides:

* The IR-walk dispatch: a default `render` implementation that walks
  the [`IRProgram`][quivers.transpile.ir.IRProgram] body and routes
  each [`IRNode`][quivers.transpile.ir.IRNode] to the right dispatch
  point.
* Index-substitution helpers: rewrite an
  [`IRArgRef`][quivers.transpile.ir.IRArgRef] indexed against the
  surrounding plate's `batch_dims` so a renderer's sample / observe
  emission gets `name[m_0, m_1, ...]` form for the LHS and indexed
  args.
* The explicit-latent rewrite helper, shared by every backend whose
  `marginalize` lowers `IRMarginalize` to `IRSample(latent)` plus
  the scope body inline.
* `assert_no_dangling_refs` / `assert_no_lists`: structural
  invariants every renderer checks before emission.

The `_RenderCtx` dataclass is the renderer-internal carrier for
the panproto `SchemaBuilder`, fresh-id counter, and resolved
morphism / define tables; it's the only `@dataclasses.dataclass` in
the transpile layer (the IR uses `dx.Model` exclusively).
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Literal, Protocol, runtime_checkable

import panproto

from quivers.dsl.ast_nodes import Expr, MorphismDecl
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.ir import (
    ConstraintSpec,
    IRArg,
    IRArgBroadcast,
    IRArgFamilyRef,
    IRArgKernel,
    IRArgList,
    IRArgMatrix,
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
)


#: Where a declaration lands in the target's program structure. Each
#: backend interprets the kind per its own program layout (Stan has
#: actual blocks; NumPyro's "block" is the function body).
type BlockKind = Literal[
    "data",
    "parameters",
    "transformed_parameters",
    "model",
    "generated_quantities",
    "function_body",
]


#: One panproto schema fragment: either an opaque vertex id, or the
#: empty string when the dispatch point emits nothing of its own.
type SchemaFragment = str


@dataclasses.dataclass
class _RenderCtx:
    """Renderer-internal mutable carrier for the panproto
    `SchemaBuilder`, fresh-id counter, and resolved morphism / define
    tables.

    One per `render` call. Threaded through the IR-walk dispatch
    and helpers as the first positional argument. The IR uses
    `dx.Model` exclusively; this `@dataclasses.dataclass` is the
    documented exception in the spec because it carries
    renderer-internal mutable state that does not round-trip.
    """

    sb: panproto.SchemaBuilder
    morphisms: dict[str, MorphismDecl]
    defines: dict[str, Expr]
    fresh_counter: int = 0
    cards: dict[str, int] = dataclasses.field(default_factory=dict)


class IRArgTransform(IRArg):
    """A renderer-applied transform on an IR arg.

    The renderer constructs this during emission when an
    `arg_aliases` rename targets a parameterisation that requires
    arithmetic conversion (e.g. BUGS `tau = 1/(scale*scale)` for
    Normal). Per-renderer `_ALIAS_TRANSFORMS` tables key the
    transform on the renamed target name; `RendererBase` provides
    the constructor.

    `IRArgTransform` is a renderer-internal IR extension. `Lower`
    never constructs it.
    """

    inner: IRArg
    transform: Literal["inv_square", "inv", "neg", "log", "exp"]
    kind: Literal["transform"] = "transform"


@runtime_checkable
class Renderer(Protocol):
    """The protocol every backend renderer satisfies."""

    @abc.abstractmethod
    def render(self, ir: IRProgram) -> panproto.Schema:
        """Render an `IRProgram` to a per-backend panproto schema."""
        ...

    @abc.abstractmethod
    def declare(
        self,
        ctx: _RenderCtx,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """Emit the declaration of a named variable in `block`."""
        ...

    @abc.abstractmethod
    def sample(
        self,
        ctx: _RenderCtx,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        constraint: ConstraintSpec,
        plate: Plate,
        observed: bool,
    ) -> SchemaFragment:
        """Emit the `~` statement for a sample / observe step."""
        ...

    @abc.abstractmethod
    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Emit the discrete-latent integration scope."""
        ...

    @abc.abstractmethod
    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit the target's broadcast op for `value` to
        `target_shape`."""
        ...


class RendererBase(abc.ABC):
    """Shared IR-walk dispatch and helpers for backend renderers.

    Subclasses implement `target_protocol`, `declare`, `sample`,
    `marginalize`, `broadcast`, and may override `render` to wrap
    the inherited walk with per-backend pre / post processing
    (block emission, function header, etc.).
    """

    target: str

    # ----- abstract dispatch points -----

    @abc.abstractmethod
    def target_protocol(self) -> panproto.Protocol:
        """Return the panproto protocol for the renderer's target
        language. Each renderer instantiates its own
        `panproto.SchemaBuilder(target_protocol())`."""

    @abc.abstractmethod
    def declare(
        self,
        ctx: _RenderCtx,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        ...

    @abc.abstractmethod
    def sample(
        self,
        ctx: _RenderCtx,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        constraint: ConstraintSpec,
        plate: Plate,
        observed: bool,
    ) -> SchemaFragment:
        ...

    @abc.abstractmethod
    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
    ) -> SchemaFragment:
        ...

    @abc.abstractmethod
    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        ...

    # ----- IR walk -----

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and route each node to the right dispatch.

        The default walk dispatches on the IRNode discriminator:

        | node | dispatch |
        | --- | --- |
        | `IRDataInput` | `declare(name, ..., block="data")` |
        | `IRSample` (latent) | `declare + sample(observed=False)` |
        | `IRObserve` | `declare + sample(observed=True)` |
        | `IRDeterministic` | `declare + sample(observed=True)` (deterministic) |
        | `IRScore` | renderer-specific (`target +=`) |
        | `IRMarginalize` | `marginalize(node)` |
        | `IRReturn` | `_emit_return(names)` |

        Subclasses may override `render` to wrap the walk with
        their own block prologue / epilogue.
        """
        assert_no_lists(ir)
        proto = self.target_protocol()
        sb = proto.schema()
        ctx = _RenderCtx(sb=sb, morphisms={}, defines={})
        self._walk(ctx, ir)
        return sb.build()

    def _walk(self, ctx: _RenderCtx, ir: IRProgram) -> None:
        for inp in ir.inputs:
            self.declare(ctx, inp.name, inp.constraint, inp.plate, block="data")
        for node in ir.body:
            self._dispatch_node(ctx, node)

    def _dispatch_node(self, ctx: _RenderCtx, node: IRNode) -> None:
        if isinstance(node, IRDataInput):
            self.declare(
                ctx, node.name, node.constraint, node.plate, block="data"
            )
            return
        if isinstance(node, IRSample):
            self.declare(
                ctx,
                node.name,
                node.constraint,
                node.plate,
                block="parameters",
            )
            self.sample(
                ctx,
                node.name,
                node.family,
                node.args,
                node.arg_names,
                node.constraint,
                node.plate,
                observed=False,
            )
            return
        if isinstance(node, IRObserve):
            self.declare(
                ctx, node.name, node.constraint, node.plate, block="data"
            )
            self.sample(
                ctx,
                node.name,
                node.family,
                node.args,
                node.arg_names,
                node.constraint,
                node.plate,
                observed=True,
            )
            return
        if isinstance(node, IRDeterministic):
            self.declare(
                ctx,
                node.name,
                node.constraint,
                node.plate,
                block="transformed_parameters",
            )
            return
        if isinstance(node, IRScore):
            # Per-renderer: BUGS / Stan handle `target +=` natively;
            # NumPyro / Pyro use `factor(...)`. Subclasses override
            # `_emit_score` for the target idiom.
            self._emit_score(ctx, node)
            return
        if isinstance(node, IRMarginalize):
            self.marginalize(ctx, node)
            return
        if isinstance(node, IRReturn):
            self._emit_return(ctx, node.names)
            return
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"node:{type(node).__name__}"],
        )

    # ----- index-substitution helpers -----

    def index_for(
        self, ctx: _RenderCtx, plate: Plate
    ) -> tuple[str, ...]:
        """Return loop-index names for the plate's batch_dims.

        Returns one name per batch_dim, generated via
        `ctx.fresh_counter`. Renderers pair these with each batch
        dim's size to emit nested `for (m_i in 1:B_i)` loops.
        """
        del ctx
        return tuple(f"m_{dim.name}" for dim in plate.batch_dims)

    def substitute_indices(
        self, arg: IRArg, names: tuple[str, ...]
    ) -> IRArg:
        """Apply the loop-index names from `names` to the arg's
        bracket indices.

        Used by sample / observe rendering to thread the
        surrounding plate's loop variables through an
        [`IRArgRef`][quivers.transpile.ir.IRArgRef]'s index list so
        the emitted call form is `name[m_0, m_1, ...]`.
        """
        if isinstance(arg, IRArgRef) and arg.indices:
            new_indices = tuple(
                self.substitute_indices(idx, names) for idx in arg.indices
            )
            return IRArgRef(name=arg.name, indices=new_indices)
        return arg

    # ----- explicit-latent rewrite for marginalize -----

    def explicit_latent_scope(
        self, node: IRMarginalize
    ) -> tuple[IRNode, ...]:
        """Lower an [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        scope to `IRSample(latent)` plus the scope body inline.

        Used by every backend whose `marginalize` lowers the
        construct to explicit sampling. The Stan renderer (which
        emits `log_sum_exp` enumeration natively) does not call
        this helper.
        """
        latent_sample = IRSample(
            name=node.latent,
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            constraint=node.constraint,
            plate=node.plate,
        )
        return (latent_sample, *node.scope)

    # ----- score / return defaults -----

    def _emit_score(self, ctx: _RenderCtx, node: IRScore) -> None:
        """Default: subclasses override to emit `target +=` /
        `numpyro.factor` / similar idiom for the renderer's target.
        """
        del ctx, node
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            ["node:IRScore: renderer does not implement score"],
        )

    def _emit_return(
        self, ctx: _RenderCtx, names: tuple[str, ...]
    ) -> None:
        """Default: subclasses override to emit `return ...` / Stan
        generated-quantities aliasing / similar.
        """
        del ctx, names


# ---------------------------------------------------------------------------
# Structural invariants every renderer checks before emission.
# ---------------------------------------------------------------------------


def assert_no_dangling_refs(ir: IRProgram) -> None:
    """Raise if the IR contains an `IRArgRef` whose name is not bound
    by a declaration, a previous step, or an input."""
    declared: set[str] = {inp.name for inp in ir.inputs}
    # ``__row_var__`` is the sentinel Lower writes into observe arg
    # threads to mark "the renderer's per-row loop variable; name
    # bound at render time". Each renderer substitutes the sentinel
    # for its actual loop variable (Stan ``n``, BUGS ``n``, NumPyro's
    # implicit plate index, ...). It is a structural marker, not a
    # true free name, so it does not count as dangling.
    declared.add("__row_var__")
    _walk_for_refs(ir.body, declared, ir.name)


def _walk_for_refs(
    body: tuple[IRNode, ...], declared: set[str], program_name: str
) -> None:
    for node in body:
        if isinstance(node, (IRSample, IRObserve, IRMarginalize)):
            for arg in node.args:
                _check_arg_refs(arg, declared, program_name)
        if isinstance(node, (IRSample, IRObserve, IRDeterministic)):
            declared.add(node.name)
        elif isinstance(node, IRMarginalize):
            declared.add(node.latent)
            _walk_for_refs(node.scope, declared, program_name)
        elif isinstance(node, IRScore):
            declared.add(node.name)


def _check_arg_refs(
    arg: IRArg, declared: set[str], program_name: str
) -> None:
    if isinstance(arg, IRArgRef):
        if arg.name not in declared:
            raise UnsupportedConstruct(
                "qvr-renderer",
                [
                    f"program:{program_name}: arg references "
                    f"undeclared name {arg.name!r}"
                ],
            )
        for idx in arg.indices:
            _check_arg_refs(idx, declared, program_name)
        return
    if isinstance(arg, IRArgBroadcast):
        _check_arg_refs(arg.value, declared, program_name)
        return
    if isinstance(arg, IRArgList):
        for e in arg.elements:
            _check_arg_refs(e, declared, program_name)
        return
    if isinstance(arg, IRArgMatrix):
        for row in arg.rows:
            for e in row.elements:
                _check_arg_refs(e, declared, program_name)
        return
    if isinstance(arg, IRArgFamilyRef):
        if arg.name not in declared:
            # Family-ref names resolve via the morphism table, not
            # the bound-name set; renderers tolerate this and
            # consult ctx.morphisms during emission.
            return
    if isinstance(arg, IRArgKernel):
        if arg.x_name not in declared:
            raise UnsupportedConstruct(
                "qvr-renderer",
                [
                    f"program:{program_name}: GP kernel input "
                    f"references undeclared name {arg.x_name!r}"
                ],
            )
        return


def assert_no_lists(ir: IRProgram) -> None:
    """Raise if the IR contains an `IRArgList` /
    [`IRArgMatrix`][quivers.transpile.ir.IRArgMatrix] in a position a
    backend does not support.

    The default implementation is permissive: list / matrix args are
    legal in the IR; only renderers that lack a native list / matrix
    literal raise. Per-renderer `assert_no_lists` overrides plug in
    the per-target rejection rule.
    """
    del ir


__all__ = [
    "BlockKind",
    "IRArgTransform",
    "Renderer",
    "RendererBase",
    "SchemaFragment",
    "assert_no_dangling_refs",
    "assert_no_lists",
]
