"""`GenRenderer`: emit IR as a Gen.jl `@gen function` definition.

Gen.jl has no native plate construct. Each batch axis becomes an
explicit `for` loop in Julia that writes per-element draws into a
pre-allocated `Vector{T}(undef, B)` so the loop body can read prior
draws by index. The trace address attached to each per-element call
includes the loop index so Gen sees a distinct address per iteration:

```julia
@gen function model(alpha, beta, word_idx, w)
    theta = Vector{Vector{Float64}}(undef, 20)
    for m_Doc in 1:20
        theta[m_Doc] = @trace(dirichlet(fill(alpha, 3)), (:theta, m_Doc))
    end
    ...
end
```

The renderer follows the contract spelled out in `notes/transpile-redesign.md`
sections 5, 6, and 10.10. It inherits the IR-walk dispatch from
[`RendererBase`][quivers.transpile.renderers._base.RendererBase] and
implements `declare`, `sample`, `marginalize`, and `broadcast` per the
Gen.jl idiom. `marginalize` lowers
[`IRMarginalize`][quivers.transpile.ir.IRMarginalize] to an explicit
[`IRSample`][quivers.transpile.ir.IRSample] followed by the scope
inline (Gen samples discrete latents natively, unlike Stan's
`log_sum_exp` enumeration).
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import panproto
import torch.distributions.constraints as _constraints

from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprIndex,
    LetExprList,
    LetExprLiteral,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import target_protocol
from quivers.transpile.family_meta import FAMILY_META
from quivers.transpile.ir import (
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
    is_int_bit,
    is_int_category,
    is_int_count,
    is_real_corr_chol,
    is_real_cov_matrix,
    is_real_matrix,
    is_real_one_hot,
    is_real_positive,
    is_real_scalar,
    is_real_simplex,
    is_real_unit_interval,
    is_real_vector,
)
from quivers.transpile.renderers._base import (
    BlockKind,
    RendererBase,
    SchemaFragment,
    _RenderCtx,
    assert_no_dangling_refs,
)


# ---------------------------------------------------------------------------
# Julia tree-sitter SchemaBuilder helpers.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _GenCtx:
    """Per-render Julia schema-construction carrier.

    Wraps the panproto [`SchemaBuilder`][panproto.SchemaBuilder] with a
    fresh-id counter, the accumulators the IR walk fills in for the
    function signature (parameters) and body (statements), and the
    cross-step tables the renderer consults when threading batch
    indices through references (which prior-step samples live on
    which axes, which inputs are scalar vs vector / index, which
    morphism decls back wrapper families).
    """

    sb: panproto.SchemaBuilder
    n: int = 0
    params: list[str] = dataclasses.field(default_factory=list)
    body_stmts: list[str] = dataclasses.field(default_factory=list)
    return_names: tuple[str, ...] = ()
    # Each previously-declared sample / observe / data-input mapped to
    # the tuple of batch Dims it lives on, in declaration order. Used
    # to thread loop indices through `IRArgRef` instances.
    decl_axes: dict[str, tuple[Dim, ...]] = dataclasses.field(default_factory=dict)
    # Cards harvested from the IR for axis-name lookups.
    cards: dict[str, int] = dataclasses.field(default_factory=dict)
    # Morphism declarations needed by wrapper-family rendering.
    morphisms: dict[str, object] = dataclasses.field(default_factory=dict)
    # IRDataInput records by name, to recover constraint info on the
    # fly during arg rendering.
    inputs_by_name: dict[str, IRDataInput] = dataclasses.field(default_factory=dict)
    # Axis names a prior `for` loop has already iterated over in the
    # function body. The loop-variable namer consults this set to
    # decide whether to emit the bare `m_<Axis>` form or the
    # disambiguated `m_<Axis>_<step>` form.
    used_axes: set[str] = dataclasses.field(default_factory=set)

    def fresh(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}_{self.n}"

    def v(self, kind: str, prefix: str | None = None) -> str:
        vid = self.fresh(prefix or kind[:3])
        self.sb.vertex(vid, kind)
        return vid

    def vlit(self, kind: str, text: str, prefix: str | None = None) -> str:
        vid = self.v(kind, prefix)
        self.sb.constraint(vid, "literal-value", text)
        return vid

    def e(self, src: str, tgt: str) -> None:
        self.sb.edge(src, tgt, "child_of")


def _ident(gx: _GenCtx, name: str) -> str:
    return gx.vlit("identifier", name, "id")


def _integer(gx: _GenCtx, value: int) -> str:
    return gx.vlit("integer_literal", str(value), "int")


def _float_lit(gx: _GenCtx, value: float) -> str:
    return gx.vlit("float_literal", repr(value), "flt")


def _number(gx: _GenCtx, value: float) -> str:
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return _integer(gx, int(value))
    return _float_lit(gx, float(value))


def _operator(gx: _GenCtx, text: str) -> str:
    return gx.vlit("operator", text, "op")


def _quote_sym(gx: _GenCtx, name: str) -> str:
    """`:name` as a Julia quote_expression containing an identifier."""
    q = gx.v("quote_expression", "qe")
    gx.e(q, _ident(gx, name))
    return q


def _call(gx: _GenCtx, callee_vid: str, args: tuple[str, ...]) -> str:
    """`<callee>(<a1>, <a2>, ...)`."""
    c = gx.v("call_expression", "call")
    al = gx.v("argument_list", "al")
    gx.e(c, callee_vid)
    gx.e(c, al)
    for a in args:
        gx.e(al, a)
    return c


def _index_into(gx: _GenCtx, base: str, indices: tuple[str, ...]) -> str:
    """`base[i1, i2, ...]` as `index_expression(base, vector_expression(...))`.

    Julia tree-sitter encodes the bracket pair as a `vector_expression`
    child of `index_expression`, not as parallel child edges of the
    `index_expression` itself.
    """
    ix = gx.v("index_expression", "ix")
    gx.e(ix, base)
    ve = gx.v("vector_expression", "ve")
    for i in indices:
        gx.e(ve, i)
    gx.e(ix, ve)
    return ix


def _range(gx: _GenCtx, lo: str, hi: str) -> str:
    """`<lo>:<hi>` as a range_expression."""
    r = gx.v("range_expression", "rg")
    gx.e(r, lo)
    gx.e(r, hi)
    return r


def _tuple(gx: _GenCtx, parts: tuple[str, ...]) -> str:
    t = gx.v("tuple_expression", "tup")
    for p in parts:
        gx.e(t, p)
    return t


def _assignment(gx: _GenCtx, lhs: str, rhs: str) -> str:
    """`<lhs> = <rhs>`."""
    asn = gx.v("assignment", "asn")
    gx.e(asn, lhs)
    gx.e(asn, _operator(gx, "="))
    gx.e(asn, rhs)
    return asn


def _macro_call_space(
    gx: _GenCtx, macro_name: str, args: tuple[str, ...]
) -> str:
    """`@<macro_name> arg1 arg2 ...` (space-separated macro args).

    Julia accepts both `@trace(dist, addr)` and `@trace dist addr`. The
    panproto emitter does not consistently round-trip the parenthesised
    form for by-construction schemas (it requires byte-position
    metadata the walker cannot synthesise), so we emit the space-
    separated form. Gen.jl parses both identically.
    """
    mc = gx.v("macrocall_expression", "mc")
    mid = gx.v("macro_identifier", "mid")
    gx.e(mid, _ident(gx, macro_name))
    mal = gx.v("macro_argument_list", "mal")
    for a in args:
        gx.e(mal, a)
    gx.e(mc, mid)
    gx.e(mc, mal)
    return mc


def _macro_call_body(gx: _GenCtx, macro_name: str, body_vid: str) -> str:
    """`@<macro_name> <body>` for a long-form macro (function def, etc.)."""
    return _macro_call_space(gx, macro_name, (body_vid,))


def _function_def(
    gx: _GenCtx, *, name: str, params: tuple[str, ...], body_vid: str
) -> str:
    """`function <name>(<p1>, <p2>, ...) <body> end`."""
    fn = gx.v("function_definition", "fn")
    sig = gx.v("signature", "sig")
    sigcall = gx.v("call_expression", "scall")
    al = gx.v("argument_list", "sargs")
    for p in params:
        gx.e(al, _ident(gx, p))
    gx.e(sigcall, _ident(gx, name))
    gx.e(sigcall, al)
    gx.e(sig, sigcall)
    gx.e(fn, sig)
    gx.e(fn, body_vid)
    return fn


def _for_stmt(
    gx: _GenCtx, *, var: str, lo: str, hi: str, body_stmts: tuple[str, ...]
) -> str:
    """`for <var> in <lo>:<hi> <body> end`."""
    fs = gx.v("for_statement", "fs")
    fb = gx.v("for_binding", "fb")
    gx.e(fb, _ident(gx, var))
    gx.e(fb, _range(gx, lo, hi))
    blk = gx.v("block", "blk")
    for s in body_stmts:
        gx.e(blk, s)
    gx.e(fs, fb)
    gx.e(fs, blk)
    return fs


def _vector_alloc(
    gx: _GenCtx, *, elem_type: str, size_vid: str
) -> str:
    """`Vector{<elem_type>}(undef, <size>)`.

    `elem_type` is a Julia type-name string (e.g. ``"Float64"``,
    ``"Vector{Float64}"``, ``"Matrix{Float64}"``, ``"Int"``).
    """
    callee = _parametrized_type(gx, "Vector", elem_type)
    al = gx.v("argument_list", "al")
    gx.e(al, _ident(gx, "undef"))
    gx.e(al, size_vid)
    call = gx.v("call_expression", "call")
    gx.e(call, callee)
    gx.e(call, al)
    return call


def _parametrized_type(gx: _GenCtx, head: str, inner: str) -> str:
    """Build a `head{inner}` parametrised type expression.

    `inner` is a string spelling another Julia type expression
    (atomic identifier or another parametrised type). Recursive
    parametrisations like ``Vector{Vector{Float64}}`` parse cleanly.
    """
    pt = gx.v("parametrized_type_expression", "pt")
    gx.e(pt, _ident(gx, head))
    curly = gx.v("curly_expression", "cu")
    gx.e(curly, _type_expr(gx, inner))
    gx.e(pt, curly)
    return pt


def _type_expr(gx: _GenCtx, spec: str) -> str:
    """Convert a Julia type-name string into a vertex.

    Supports atomic identifiers and one or more nested parametrised
    type heads (e.g. ``"Vector{Vector{Float64}}"``,
    ``"Matrix{Float64}"``). The grammar is `Head{Inner}` where
    `Inner` is another type-name.
    """
    spec = spec.strip()
    if "{" not in spec:
        return _ident(gx, spec)
    head, rest = spec.split("{", 1)
    if not rest.endswith("}"):
        raise UnsupportedConstruct(
            "qvr-gen", [f"type-expr:{spec}: missing closing brace"]
        )
    inner = rest[:-1]
    return _parametrized_type(gx, head.strip(), inner)


# ---------------------------------------------------------------------------
# Dim-size resolution and element-type tables.
# ---------------------------------------------------------------------------


def _dim_size_vid(gx: _GenCtx, dim: Dim) -> str:
    """Return the vertex id for a dim's size.

    Static dims emit the literal cardinality (`20`). Dynamic dims
    emit `length(<size_name>)` so the generated function reads the
    runtime size from the argument the size was bound to.
    """
    if isinstance(dim, DimStatic):
        return _integer(gx, dim.size)
    if isinstance(dim, DimDynamic):
        # For dynamic dims with a `size_name`, prefer the
        # `length(<name>)` form unless the name is already an integer
        # input the program reads directly.
        return _call(gx, _ident(gx, "length"), (_ident(gx, dim.size_name),))
    raise UnsupportedConstruct(
        "qvr-gen", [f"dim:{type(dim).__name__}"]
    )


# ---------------------------------------------------------------------------
# Per-support storage-type tables.
# ---------------------------------------------------------------------------


def _element_type_for(spec: ConstraintSpec, plate: Plate) -> str:
    """Return the Julia element-type string for one draw of the family.

    Used as the `T` in `Vector{T}(undef, B)` when batch_dims is
    non-empty. Dispatches on the support predicates of
    `notes/transpile-redesign.md` section 2.2.
    """
    del plate
    c = spec.to_constraint()
    if (
        is_real_scalar(c)
        or is_real_positive(c)
        or is_real_unit_interval(c)
    ):
        return "Float64"
    if is_real_vector(c) or is_real_simplex(c) or is_real_one_hot(c):
        return "Vector{Float64}"
    if (
        is_real_matrix(c)
        or is_real_cov_matrix(c)
        or is_real_corr_chol(c)
    ):
        return "Matrix{Float64}"
    if is_int_bit(c) or is_int_category(c) or is_int_count(c):
        return "Int"
    raise UnsupportedConstruct(
        "qvr-gen",
        [f"support:{type(c).__name__}: no Gen.jl storage-type rule"],
    )


# ---------------------------------------------------------------------------
# Loop-variable naming.
#
# A step's loop variable per axis is `m_<AxisName>` by default; when
# the IR contains multiple steps batched on the same axis, the
# renderer disambiguates by appending the step name (`m_Doc_z`).
# ---------------------------------------------------------------------------


def _loop_var_for(gx: _GenCtx, axis_name: str, step_name: str) -> str:
    """Return the loop-variable identifier for `axis_name` in `step_name`.

    Sticks with the bare `m_<Axis>` form unless another previously
    emitted step already used a loop on that axis, in which case the
    step name is appended to disambiguate (`m_Doc_z`).
    """
    base = f"m_{axis_name}"
    if axis_name in gx.used_axes:
        return f"{base}_{step_name}"
    return base


# ---------------------------------------------------------------------------
# IR arg rendering.
# ---------------------------------------------------------------------------


_BINOPS = frozenset({"+", "-", "*", "/", "^"})


def _expected_event_rank(
    family: str, arg_name: str
) -> int:
    """Return the expected `event_dim` of the family's `arg_name`.

    Reads `FAMILY_META[family].distribution_class.arg_constraints`
    and walks it for an `IndependentConstraint(base, n)` constraint
    whose `event_dim` indicates how many vector / matrix axes the
    arg should carry. Returns 0 for scalar args.
    """
    meta = FAMILY_META.get(family)
    if meta is None:
        return 0
    cls_attr = getattr(meta.distribution_class, "arg_constraints", None)
    if not isinstance(cls_attr, dict):
        return 0
    constraint = cls_attr.get(arg_name)
    if constraint is None:
        return 0
    if isinstance(constraint, _constraints._IndependentConstraint):
        return int(constraint.event_dim)
    return 0


def _broadcast_arg_to_event(
    gx: _GenCtx,
    arg_vid: str,
    *,
    arg: IRArg,
    family: str,
    arg_name: str,
    event_dims: tuple[Dim, ...],
    inputs_by_name: dict[str, IRDataInput],
) -> str:
    """Wrap a rendered scalar arg in `fill(...)` when the family arg
    expects a vector / matrix and the user supplied a scalar.

    Triggers when:

    * the IR arg is a literal number; or
    * the IR arg is a reference whose binding is a scalar input.

    The fill shape is the event_dims cardinality.
    """
    rank = _expected_event_rank(family, arg_name)
    if rank == 0:
        return arg_vid
    if isinstance(arg, IRArgBroadcast):
        # Lower already supplied a broadcast wrapper.
        return arg_vid
    needs_fill = False
    if isinstance(arg, IRArgNumber):
        needs_fill = True
    elif isinstance(arg, IRArgRef) and not arg.indices:
        binding = inputs_by_name.get(arg.name)
        if binding is not None and _is_scalar_input(binding):
            needs_fill = True
    if not needs_fill:
        return arg_vid
    sizes: list[str] = []
    for d in event_dims[:rank]:
        sizes.append(_dim_size_vid(gx, d))
    if not sizes:
        return arg_vid
    return _call(gx, _ident(gx, "fill"), (arg_vid, *sizes))


def _is_scalar_input(inp: IRDataInput) -> bool:
    """True iff the data input is a scalar (no batch / event dims)."""
    if inp.plate.batch_dims or inp.plate.event_dims:
        return False
    c = inp.constraint.to_constraint()
    if is_real_vector(c) or is_real_simplex(c) or is_real_one_hot(c):
        return False
    if (
        is_real_matrix(c)
        or is_real_cov_matrix(c)
        or is_real_corr_chol(c)
    ):
        return False
    return True


@dataclasses.dataclass
class _ArgCtx:
    """Per-call rendering context for the args of one distribution call.

    Carries the per-step batch-loop bindings (axis name → loop
    variable identifier), the current step's `via` fibration (if
    any), and a flag indicating whether the inner observe sees the
    fibration semantics so reference threading routes through it.
    """

    batch_loops: dict[str, str] = dataclasses.field(default_factory=dict)
    via_indexer: str | None = None
    via_loop_var: str | None = None


def _render_arg(
    gx: _GenCtx, arg: IRArg, *, arg_ctx: _ArgCtx
) -> str:
    """Render one IR arg into a Julia expression vertex.

    See `_render_arg_ref` for how `arg_ctx` participates in threading
    loop indices through references.
    """
    if isinstance(arg, IRArgNumber):
        return _number(gx, arg.value)
    if isinstance(arg, IRArgRef):
        return _render_arg_ref(gx, arg, arg_ctx=arg_ctx)
    if isinstance(arg, IRArgBroadcast):
        value_vid = _render_arg(gx, arg.value, arg_ctx=arg_ctx)
        return _broadcast_to_shape(gx, value_vid, arg.target_shape)
    if isinstance(arg, IRArgList):
        return _render_list(gx, arg, arg_ctx=arg_ctx)
    if isinstance(arg, IRArgMatrix):
        return _render_matrix(gx, arg, arg_ctx=arg_ctx)
    if isinstance(arg, IRArgFamilyRef):
        raise UnsupportedConstruct(
            "qvr-gen",
            [f"arg:family_ref:{arg.name}: no inline rendering"],
        )
    raise UnsupportedConstruct(
        "qvr-gen", [f"arg:{type(arg).__name__}"]
    )


def _render_arg_ref(
    gx: _GenCtx, ref: IRArgRef, *, arg_ctx: _ArgCtx
) -> str:
    """Render an [`IRArgRef`][quivers.transpile.ir.IRArgRef], threading
    the surrounding loop's index into the reference.

    A reference to a batched declaration acquires one bracket per
    batch axis when the user did not supply explicit indices for
    that reference. The bracket is filled from `arg_ctx.batch_loops`
    when the declaration's axis has a live loop variable, or via the
    `via=` fibration when no direct loop variable matches.

    If the user supplied explicit indices, the renderer treats them
    as the full bracket expression and skips automatic axis
    threading; the inner index args still see the surrounding
    `arg_ctx` and route through it as appropriate.
    """
    base = _ident(gx, ref.name)
    decl_axes = gx.decl_axes.get(ref.name, ())

    if ref.indices:
        rendered = tuple(
            _render_arg(gx, idx, arg_ctx=arg_ctx) for idx in ref.indices
        )
        return _index_into(gx, base, rendered)

    if not decl_axes:
        return base

    leading: list[str] = []
    for axis in decl_axes:
        loop_var = arg_ctx.batch_loops.get(axis.name)
        if loop_var is not None:
            leading.append(_ident(gx, loop_var))
            continue
        # No direct loop variable: route through the via fibration
        # if one is active; otherwise leave the axis unindexed.
        if (
            arg_ctx.via_indexer is not None
            and arg_ctx.via_loop_var is not None
        ):
            via_idx = _index_into(
                gx,
                _ident(gx, arg_ctx.via_indexer),
                (_ident(gx, arg_ctx.via_loop_var),),
            )
            leading.append(via_idx)
            continue

    if not leading:
        return base
    return _index_into(gx, base, tuple(leading))


def _broadcast_to_shape(
    gx: _GenCtx, value_vid: str, target_shape: tuple[int, ...]
) -> str:
    """`fill(<value>, <s0>, <s1>, ...)` per the Gen broadcast idiom."""
    if not target_shape:
        return value_vid
    args: list[str] = [value_vid]
    for s in target_shape:
        args.append(_integer(gx, int(s)))
    return _call(gx, _ident(gx, "fill"), tuple(args))


def _render_list(
    gx: _GenCtx, arg: IRArgList, *, arg_ctx: _ArgCtx
) -> str:
    """`[e0, e1, ...]` as a Julia vector_expression."""
    ve = gx.v("vector_expression", "ve")
    for e in arg.elements:
        gx.e(ve, _render_arg(gx, e, arg_ctx=arg_ctx))
    return ve


def _render_matrix(
    gx: _GenCtx, arg: IRArgMatrix, *, arg_ctx: _ArgCtx
) -> str:
    """`[<r0>; <r1>; ...]` as a Julia matrix_expression of matrix_rows."""
    me = gx.v("matrix_expression", "me")
    for row in arg.rows:
        mr = gx.v("matrix_row", "mr")
        for e in row.elements:
            gx.e(mr, _render_arg(gx, e, arg_ctx=arg_ctx))
        gx.e(me, mr)
    return me


# ---------------------------------------------------------------------------
# Family-name resolution.
# ---------------------------------------------------------------------------


#: Wrapper-family fallback names. Each entry pairs the wrapper QVR
#: family with the Gen.jl call used to render its emitted distribution.
_WRAPPER_TARGET_NAMES: dict[str, str] = {
    "Truncated": "truncated",
}


def _gen_target_name(family: str) -> str:
    """Look up the Gen.jl distribution constructor for a family.

    Consults `FAMILY_META[family].target_names["gen"]` first; falls
    back to the wrapper-family table when the family is a
    composite-distribution wrapper that has no flat Gen.jl name.
    """
    meta = FAMILY_META.get(family)
    if meta is not None and "gen" in meta.target_names:
        return meta.target_names["gen"]
    fallback = _WRAPPER_TARGET_NAMES.get(family)
    if fallback is not None:
        return fallback
    raise UnsupportedConstruct(
        "qvr-gen", [f"family:{family}: no Gen.jl target name"]
    )


# ---------------------------------------------------------------------------
# Wrapper-family rendering.
# ---------------------------------------------------------------------------


def _render_truncated_call(
    gx: _GenCtx,
    *,
    args: tuple[IRArg, ...],
    arg_ctx: _ArgCtx,
) -> str:
    """`truncated(<inner-dist>, <lo>, <hi>)` via Distributions.truncated.

    The wrapped distribution is encoded as an
    [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef] referring
    to a morphism whose `init_family` is the base distribution.
    """
    if not args or not isinstance(args[0], IRArgFamilyRef):
        raise UnsupportedConstruct(
            "qvr-gen",
            ["family:Truncated: first arg must be IRArgFamilyRef"],
        )
    inner_vid = _render_inner_family(
        gx, args[0].name, arg_ctx=arg_ctx,
    )
    bounds = tuple(
        _render_arg(gx, a, arg_ctx=arg_ctx) for a in args[1:]
    )
    return _call(gx, _ident(gx, "truncated"), (inner_vid, *bounds))


def _render_inner_family(
    gx: _GenCtx, morphism_name: str, *, arg_ctx: _ArgCtx
) -> str:
    """Render the distribution call referenced by a morphism name.

    Reads the morphism's `init_family` clause from the carried
    morphism table to recover `(family_name, args)`, then dispatches
    through the family-name lookup as if it were an inline sample
    call.
    """
    decl = gx.morphisms.get(morphism_name)
    if decl is None:
        raise UnsupportedConstruct(
            "qvr-gen",
            [
                f"family_ref:{morphism_name}: morphism not declared "
                f"in transpile context"
            ],
        )
    init = getattr(decl, "init_family", None)
    if init is None or getattr(init, "family", None) is None:
        raise UnsupportedConstruct(
            "qvr-gen",
            [
                f"family_ref:{morphism_name}: morphism has no "
                f"init_family declaration"
            ],
        )
    family = init.family
    raw_args = tuple(init.args or ())
    callee_name = _gen_target_name(family)
    arg_vids = tuple(
        _lift_raw_arg(gx, a, arg_ctx=arg_ctx) for a in raw_args
    )
    return _call(gx, _ident(gx, callee_name), arg_vids)


def _lift_raw_arg(
    gx: _GenCtx, raw: object, *, arg_ctx: _ArgCtx
) -> str:
    """Lift a morphism-table raw arg (str / number / IRArg) to a vertex."""
    if isinstance(raw, IRArg):
        return _render_arg(gx, raw, arg_ctx=arg_ctx)
    if isinstance(raw, (int, float)):
        return _number(gx, float(raw))
    if isinstance(raw, str):
        try:
            value = float(raw)
        except ValueError:
            return _ident(gx, raw)
        return _number(gx, value)
    raise UnsupportedConstruct(
        "qvr-gen", [f"raw-arg:{type(raw).__name__}"]
    )


_WrapperBuilder = Callable[
    [_GenCtx, tuple[IRArg, ...], _ArgCtx], str
]


#: Per-wrapper-family builder dispatch.
_WRAPPER_BUILDERS: dict[str, _WrapperBuilder] = {
    "Truncated": (
        lambda gx, args, arg_ctx: _render_truncated_call(
            gx, args=args, arg_ctx=arg_ctx,
        )
    ),
}


def _build_wrapper_call(
    gx: _GenCtx,
    *,
    family: str,
    args: tuple[IRArg, ...],
    arg_ctx: _ArgCtx,
) -> str:
    builder = _WRAPPER_BUILDERS.get(family)
    if builder is None:
        raise UnsupportedConstruct(
            "qvr-gen", [f"wrapper-family:{family}: no Gen.jl builder"]
        )
    return builder(gx, args, arg_ctx)


# ---------------------------------------------------------------------------
# Trace-call assembly.
# ---------------------------------------------------------------------------


def _trace_address(
    gx: _GenCtx, name: str, loop_indices: tuple[str, ...]
) -> str:
    """Build the trace address: ``:name`` or ``(:name, m_0, m_1, ...)``."""
    if not loop_indices:
        return _quote_sym(gx, name)
    parts = [_quote_sym(gx, name), *[_ident(gx, i) for i in loop_indices]]
    return _tuple(gx, tuple(parts))


def _trace_call(
    gx: _GenCtx,
    *,
    dist_vid: str,
    name: str,
    loop_indices: tuple[str, ...],
) -> str:
    """`@trace(<dist>, <address>)`."""
    addr = _trace_address(gx, name, loop_indices)
    return _macro_call_space(gx, "trace", (dist_vid, addr))


# ---------------------------------------------------------------------------
# The renderer itself.
# ---------------------------------------------------------------------------


class GenRenderer(RendererBase):
    """Render an IR program as a Gen.jl `@gen function` definition.

    Inherits the IR-walk dispatch from
    [`RendererBase`][quivers.transpile.renderers._base.RendererBase].
    Overrides the per-node dispatch in `render` to collect declarations
    plus statements into the `@gen function` envelope; implements
    `declare`, `sample`, `marginalize`, `broadcast` per the Gen.jl
    idiom described in the module docstring.
    """

    target: str = "gen"

    # ------------------------------------------------------------------
    # protocol setup
    # ------------------------------------------------------------------

    def target_protocol(self) -> panproto.Protocol:
        return target_protocol("julia")

    # ------------------------------------------------------------------
    # render: wrap the IR walk in the `@gen function` envelope
    # ------------------------------------------------------------------

    def render(self, ir: IRProgram) -> panproto.Schema:
        assert_no_dangling_refs(ir)
        proto = self.target_protocol()
        sb = proto.schema()
        gx = _GenCtx(sb=sb, cards={}, morphisms={})

        _harvest_cards(ir, gx)

        for inp in ir.inputs:
            gx.params.append(inp.name)
            gx.inputs_by_name[inp.name] = inp
            gx.decl_axes[inp.name] = inp.plate.batch_dims

        # `_RenderCtx` is the inherited carrier; we re-use it for the
        # required dispatch signatures but the per-render scratch
        # lives on `_GenCtx`.
        ctx = _RenderCtx(sb=sb, morphisms={}, lets={})
        self._gx = gx  # noqa: SLF001
        try:
            for node in ir.body:
                self._emit_node(ctx, node)
        finally:
            del self._gx

        blk = gx.v("block", "body")
        for s in gx.body_stmts:
            gx.e(blk, s)
        if gx.return_names:
            ret = gx.v("return_statement", "ret")
            if len(gx.return_names) == 1:
                gx.e(ret, _ident(gx, gx.return_names[0]))
            else:
                tup = gx.v("tuple_expression", "rtup")
                for n in gx.return_names:
                    gx.e(tup, _ident(gx, n))
                gx.e(ret, tup)
            gx.e(blk, ret)

        fn = _function_def(
            gx, name=ir.name, params=tuple(gx.params), body_vid=blk,
        )
        mc = _macro_call_body(gx, "gen", fn)
        src = gx.v("source_file", "src")
        gx.e(src, mc)
        return sb.build()

    # ------------------------------------------------------------------
    # Per-IRNode dispatch
    # ------------------------------------------------------------------

    def _emit_node(self, ctx: _RenderCtx, node: IRNode) -> None:
        if isinstance(node, IRDataInput):
            return
        if isinstance(node, IRSample):
            self._emit_sample(node, observed=False, via=None)
            return
        if isinstance(node, IRObserve):
            self._emit_sample(
                IRSample(
                    name=node.name,
                    family=node.family,
                    args=node.args,
                    arg_names=node.arg_names,
                    constraint=node.constraint,
                    plate=node.plate,
                ),
                observed=True,
                via=node.via,
            )
            return
        if isinstance(node, IRDeterministic):
            self._emit_deterministic(node)
            return
        if isinstance(node, IRScore):
            self._emit_score(ctx, node)
            return
        if isinstance(node, IRMarginalize):
            self._emit_marginalize(ctx, node)
            return
        if isinstance(node, IRReturn):
            self._gx.return_names = tuple(node.names)  # noqa: SLF001
            return
        raise UnsupportedConstruct(
            "qvr-gen", [f"node:{type(node).__name__}"]
        )

    # ------------------------------------------------------------------
    # Sample / observe emission
    # ------------------------------------------------------------------

    def _emit_sample(
        self,
        node: IRSample,
        *,
        observed: bool,
        via: str | None,
    ) -> None:
        """Emit declaration + for-loop nest for a batched sample / observe.

        For the non-batched case (`plate.batch_dims` empty) emits a
        single scalar `name = @trace(<dist>, :name)`. For one-or-more
        batch dims, allocates `Vector{T}(undef, B)` then emits nested
        for-loops that fill the storage element-wise via `@trace`.

        `observed=True` skips the per-step storage allocation (the
        observed value is supplied externally via the function
        signature) and drops the `name[m] = ...` LHS so the trace
        macro emits as a bare statement.
        """
        gx = self._gx  # noqa: SLF001
        if not node.plate.batch_dims:
            self._emit_scalar_sample(node, observed=observed)
            return
        if not observed:
            self._emit_storage_alloc(node)
        # The declaration is now visible to subsequent steps as a
        # batched draw on `node.plate.batch_dims`.
        gx.decl_axes[node.name] = node.plate.batch_dims
        self._emit_loop_nest(node, observed=observed, via=via)

    def _emit_scalar_sample(
        self, node: IRSample, *, observed: bool
    ) -> None:
        gx = self._gx  # noqa: SLF001
        dist_vid = self._build_dist_call(
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            event_dims=node.plate.event_dims,
            arg_ctx=_ArgCtx(),
        )
        trace = _trace_call(
            gx, dist_vid=dist_vid, name=node.name, loop_indices=()
        )
        if observed:
            gx.body_stmts.append(trace)
        else:
            stmt = _assignment(gx, _ident(gx, node.name), trace)
            gx.body_stmts.append(stmt)
            gx.decl_axes[node.name] = ()

    def _emit_storage_alloc(self, node: IRSample) -> None:
        """Pre-allocate `Vector{T}(undef, B0, B1, ...)` for batched draws."""
        gx = self._gx  # noqa: SLF001
        elem_type = _element_type_for(node.constraint, node.plate)
        # For >1 batch dim, nest the storage: Vector{Vector{...{T}...}}.
        for _ in node.plate.batch_dims[1:]:
            elem_type = f"Vector{{{elem_type}}}"
        outer = node.plate.batch_dims[0]
        outer_size = _dim_size_vid(gx, outer)
        alloc = _vector_alloc(
            gx, elem_type=elem_type, size_vid=outer_size,
        )
        stmt = _assignment(gx, _ident(gx, node.name), alloc)
        gx.body_stmts.append(stmt)

    def _emit_loop_nest(
        self,
        node: IRSample,
        *,
        observed: bool,
        via: str | None,
    ) -> None:
        gx = self._gx  # noqa: SLF001
        loop_names = tuple(
            _loop_var_for(gx, dim.name, node.name)
            for dim in node.plate.batch_dims
        )
        # The batch-loop binding for the current step: axis name →
        # loop variable identifier. Each declared ref whose
        # declaration sits on an axis present here picks up the
        # corresponding loop variable as an index.
        batch_loops = {
            dim.name: lv
            for dim, lv in zip(
                node.plate.batch_dims, loop_names, strict=True,
            )
        }
        arg_ctx = _ArgCtx(
            batch_loops=batch_loops,
            via_indexer=via,
            via_loop_var=loop_names[-1] if via else None,
        )
        dist_vid = self._build_dist_call(
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            event_dims=node.plate.event_dims,
            arg_ctx=arg_ctx,
        )
        trace = _trace_call(
            gx,
            dist_vid=dist_vid,
            name=node.name,
            loop_indices=loop_names,
        )
        if observed:
            inner_stmt = trace
        else:
            lhs = self._build_indexed_lhs(node.name, loop_names)
            inner_stmt = _assignment(gx, lhs, trace)

        current_stmts: tuple[str, ...] = (inner_stmt,)
        for dim, loop_var in zip(
            reversed(node.plate.batch_dims),
            reversed(loop_names),
            strict=True,
        ):
            size_vid = _dim_size_vid(gx, dim)
            lo = _integer(gx, 1)
            fs = _for_stmt(
                gx,
                var=loop_var,
                lo=lo,
                hi=size_vid,
                body_stmts=current_stmts,
            )
            current_stmts = (fs,)
        gx.body_stmts.append(current_stmts[0])
        for dim in node.plate.batch_dims:
            gx.used_axes.add(dim.name)

    def _build_indexed_lhs(
        self, name: str, loop_names: tuple[str, ...]
    ) -> str:
        gx = self._gx  # noqa: SLF001
        if not loop_names:
            return _ident(gx, name)
        current = _ident(gx, name)
        for lv in loop_names:
            current = _index_into(gx, current, (_ident(gx, lv),))
        return current

    def _build_dist_call(
        self,
        *,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        event_dims: tuple[Dim, ...],
        arg_ctx: _ArgCtx,
    ) -> str:
        """Build the distribution call vertex for one element draw.

        Wrapper families route through their dedicated builder;
        otherwise the family looks up the Gen.jl callee in
        `FAMILY_META` and renders args inline, threading the
        scalar→vector broadcast for any arg whose constraint expects
        a vector / matrix.
        """
        if family in _WRAPPER_BUILDERS:
            return _build_wrapper_call(
                self._gx,  # noqa: SLF001
                family=family,
                args=args,
                arg_ctx=arg_ctx,
            )
        gx = self._gx  # noqa: SLF001
        callee_name = _gen_target_name(family)
        arg_vids: list[str] = []
        for arg, name in zip(args, arg_names, strict=False):
            rendered = _render_arg(gx, arg, arg_ctx=arg_ctx)
            rendered = _broadcast_arg_to_event(
                gx,
                rendered,
                arg=arg,
                family=family,
                arg_name=name,
                event_dims=event_dims,
                inputs_by_name=gx.inputs_by_name,
            )
            arg_vids.append(rendered)
        return _call(gx, _ident(gx, callee_name), tuple(arg_vids))

    # ------------------------------------------------------------------
    # Deterministic let-bindings
    # ------------------------------------------------------------------

    def _emit_deterministic(self, node: IRDeterministic) -> None:
        gx = self._gx  # noqa: SLF001
        rhs = _render_let_expr_gen(gx, node.expr)
        stmt = _assignment(gx, _ident(gx, node.name), rhs)
        gx.body_stmts.append(stmt)
        gx.decl_axes[node.name] = node.plate.batch_dims

    # ------------------------------------------------------------------
    # Marginalize: lower to IRSample + scope inline
    # ------------------------------------------------------------------

    def _emit_marginalize(
        self, ctx: _RenderCtx, node: IRMarginalize
    ) -> None:
        explicit = self.explicit_latent_scope(node)
        for inner in explicit:
            self._emit_node(ctx, inner)

    # ------------------------------------------------------------------
    # Score: bind value, then `@addlogprob!`
    # ------------------------------------------------------------------

    def _emit_score(self, ctx: _RenderCtx, node: IRScore) -> None:
        gx = self._gx  # noqa: SLF001
        del ctx
        rhs = _render_let_expr_gen(gx, node.expr)
        bind = _assignment(gx, _ident(gx, node.name), rhs)
        gx.body_stmts.append(bind)
        mc = _macro_call_space(
            gx, "addlogprob!", (_ident(gx, node.name),)
        )
        gx.body_stmts.append(mc)

    # ------------------------------------------------------------------
    # Protocol-required dispatch points (used by the inherited walk).
    # The Gen renderer overrides `_emit_node` so these stubs return
    # empty fragments; the work happens in `_emit_*` above.
    # ------------------------------------------------------------------

    def declare(
        self,
        ctx: _RenderCtx,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        del ctx, name, constraint, plate, block
        return ""

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
        del (
            ctx, name, family, args, arg_names, constraint, plate, observed,
        )
        return ""

    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
    ) -> SchemaFragment:
        self._emit_marginalize(ctx, node)
        return ""

    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        del ctx
        gx = self._gx  # noqa: SLF001
        value_vid = _render_arg(gx, value, arg_ctx=_ArgCtx())
        return _broadcast_to_shape(gx, value_vid, target_shape)


# ---------------------------------------------------------------------------
# Let-expression rendering.
# ---------------------------------------------------------------------------


def _render_let_expr_gen(gx: _GenCtx, expr: object) -> str:
    """Render an [`IRExpr`][quivers.transpile.ir.IRExpr] subtree as Julia."""
    if isinstance(expr, LetExprLiteral):
        if isinstance(expr.value, float) or "." in repr(expr.value):
            return gx.vlit("float_literal", str(expr.value), "flt")
        return gx.vlit("integer_literal", str(expr.value), "int")
    if isinstance(expr, LetExprVar):
        return _ident(gx, expr.name)
    if isinstance(expr, LetExprBinOp):
        op = expr.op if expr.op in _BINOPS else "+"
        be = gx.v("binary_expression", "be")
        gx.e(be, _render_let_expr_gen(gx, expr.left))
        gx.e(be, _operator(gx, op))
        gx.e(be, _render_let_expr_gen(gx, expr.right))
        return be
    if isinstance(expr, LetExprUnaryOp):
        ue = gx.v("unary_expression", "ue")
        gx.e(ue, _operator(gx, "-"))
        gx.e(ue, _render_let_expr_gen(gx, expr.operand))
        return ue
    if isinstance(expr, LetExprCall):
        callee = _ident(gx, expr.func)
        args = tuple(_render_let_expr_gen(gx, a) for a in expr.args)
        return _call(gx, callee, args)
    if isinstance(expr, LetExprIndex):
        base = _render_let_expr_gen(gx, expr.array)
        idxs = tuple(_render_let_expr_gen(gx, i) for i in expr.indices)
        return _index_into(gx, base, idxs)
    if isinstance(expr, LetExprList):
        ve = gx.v("vector_expression", "ve")
        for e in expr.elements:
            gx.e(ve, _render_let_expr_gen(gx, e))
        return ve
    raise UnsupportedConstruct(
        "qvr-gen", [f"let-expr:{type(expr).__name__}"]
    )


# ---------------------------------------------------------------------------
# Card harvesting from the IR.
# ---------------------------------------------------------------------------


def _harvest_cards(ir: IRProgram, gx: _GenCtx) -> None:
    """Walk the IR collecting `DimStatic` sizes by name into `gx.cards`."""
    for inp in ir.inputs:
        _harvest_plate(inp.plate, gx)
    for node in ir.body:
        _harvest_node(node, gx)


def _harvest_node(node: IRNode, gx: _GenCtx) -> None:
    if isinstance(node, (IRSample, IRObserve, IRDataInput, IRDeterministic)):
        _harvest_plate(node.plate, gx)
    if isinstance(node, IRMarginalize):
        _harvest_plate(node.plate, gx)
        for inner in node.scope:
            _harvest_node(inner, gx)


def _harvest_plate(plate: Plate, gx: _GenCtx) -> None:
    for dim in (*plate.event_dims, *plate.batch_dims):
        if isinstance(dim, DimStatic):
            gx.cards[dim.name] = dim.size


__all__ = ["GenRenderer"]
