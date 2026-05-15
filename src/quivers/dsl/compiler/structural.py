"""Compiler mixin: structural artifacts.

Handles signature, encoder, decoder, and loss declarations.
"""

from __future__ import annotations
from collections.abc import Callable
import torch
import torch.nn as nn
from quivers.dsl.ast_nodes import (
    DecoderDecl,
    EncoderDecl,
    LossDecl,
    SignatureDecl,
)
from quivers.structural.encoder import (
    Encoder,
    _PerOpFn,
    make_default_op_fn,
    make_default_var_init,
)
from quivers.structural.decoder import Decoder
from quivers.structural.losses import LossEntry, LossRegistry
from quivers.structural.signature import (
    Binder,
    BinderArgSpec,
    BinderVarSpec,
    Constructor,
    EdgeKind,
    Signature,
    Sort,
    SortVocabEntry,
    VertexKind,
)
from quivers.dsl.compiler._prelude import (
    CompileError,
    _decode_vocab_literal,
)


_ENCODER_FACTORY_REGISTRY: dict[str, str] = {
    # Encoder factory name (as users write it in ``using
    # <factory>``) → import path ``module:attribute`` resolved at
    # compile time. Keeping the dispatch table out of the
    # ``quivers.structural.shapes`` namespace lets the DSL key on
    # short identifiers (``rnn_encoder``) without leaking the
    # factory module's full import path into source.
    "rnn_encoder": "quivers.structural.shapes.seq:rnn_encoder",
    "transformer_encoder": "quivers.structural.shapes.seq:transformer_encoder",
    "bow_encoder": "quivers.structural.shapes.seq:bow_encoder",
    "tree_lstm_encoder": "quivers.structural.shapes.tree:tree_lstm_encoder",
    "gnn_encoder": "quivers.structural.shapes.graph:gnn_encoder",
}


def _build_encoder_from_factory(decl: "EncoderDecl", sig) -> "Encoder":
    """Invoke a shipped encoder factory by name.

    Looks up ``decl.factory`` in :data:`_ENCODER_FACTORY_REGISTRY`,
    imports the registered builder, and calls it with
    ``sig=<signature>`` plus any ``[k=v]`` overrides on the
    declaration coerced to int / float / str.

    Raises :class:`CompileError` for unknown factory names or for
    builder kwargs the factory doesn't accept.
    """
    import importlib
    import inspect

    factory_name = decl.factory
    if factory_name not in _ENCODER_FACTORY_REGISTRY:
        raise CompileError(
            f"encoder {decl.name!r}: unknown factory {factory_name!r}; "
            f"available: {', '.join(sorted(_ENCODER_FACTORY_REGISTRY))}",
            decl.line,
            decl.col,
        )
    import_path = _ENCODER_FACTORY_REGISTRY[factory_name]
    module_name, _, attr = import_path.partition(":")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)

    signature_obj = inspect.signature(factory)
    kwargs: dict = {}
    if "sig" in signature_obj.parameters:
        kwargs["sig"] = sig
    for key, raw in dict(decl.factory_options).items():
        if key not in signature_obj.parameters:
            raise CompileError(
                f"encoder {decl.name!r}: factory {factory_name!r} does not "
                f"accept option {key!r}; signature is "
                f"{', '.join(sorted(signature_obj.parameters))}",
                decl.line,
                decl.col,
            )
        # Best-effort coercion: integer-like → int, float-like →
        # float, otherwise leave as the source identifier string.
        try:
            kwargs[key] = int(raw)
        except ValueError:
            try:
                kwargs[key] = float(raw)
            except ValueError:
                kwargs[key] = raw
    return factory(**kwargs)


class _StructuralMixin:
    """Mixin: structural artifact compilation methods."""

    def _compile_signature(self, decl: SignatureDecl) -> None:
        """Register a signature declaration.

        Builds a runtime :class:`quivers.structural.Signature` from
        the AST node, stashes it on ``self._signatures`` keyed by
        name. Performs sort coverage, codomain validity, and binder
        sort-consistency checks.
        """
        if not hasattr(self, "_signatures"):
            self._signatures: dict[str, Signature] = {}

        if decl.name in self._signatures:
            raise CompileError(
                f"signature {decl.name!r} already declared",
                decl.line,
                decl.col,
            )

        # Sort table.
        sorts: dict[str, Sort] = {}
        for s in decl.sorts:
            if s.name in sorts:
                raise CompileError(
                    f"signature {decl.name!r}: duplicate sort {s.name!r}",
                    s.line,
                    s.col,
                )
            if s.vocab and s.kind != "data":
                raise CompileError(
                    f"signature {decl.name!r}: vocab clause is only valid "
                    f"on `data` sorts; sort {s.name!r} has kind {s.kind!r}",
                    s.line,
                    s.col,
                )
            vocab_entries: list[SortVocabEntry] = []
            seen_vals: set = set()
            for lit in s.vocab:
                value = _decode_vocab_literal(decl.name, s.name, lit)
                if value in seen_vals:
                    raise CompileError(
                        f"signature {decl.name!r}: sort {s.name!r} vocabulary "
                        f"contains duplicate entry {value!r}",
                        s.line,
                        s.col,
                    )
                seen_vals.add(value)
                vocab_entries.append(SortVocabEntry(kind=lit.kind, value=value))
            sorts[s.name] = Sort(
                name=s.name,
                kind=s.kind,
                dim=s.dim,
                vocab=tuple(vocab_entries),
            )

        # Vertex / edge kinds (graph-shaped signatures).
        vertex_kinds: dict[str, VertexKind] = {}
        for v in decl.vertex_kinds:
            if v.name in vertex_kinds:
                raise CompileError(
                    f"signature {decl.name!r}: duplicate vertex_kind {v.name!r}",
                    v.line,
                    v.col,
                )
            vertex_kinds[v.name] = VertexKind(name=v.name, kind=v.kind, dim=v.dim)
        edge_kinds: dict[str, EdgeKind] = {}
        for e in decl.edge_kinds:
            if e.name in edge_kinds:
                raise CompileError(
                    f"signature {decl.name!r}: duplicate edge_kind {e.name!r}",
                    e.line,
                    e.col,
                )
            if e.src not in vertex_kinds:
                raise CompileError(
                    f"signature {decl.name!r}: edge_kind {e.name!r} has "
                    f"unknown source vertex_kind {e.src!r}",
                    e.line,
                    e.col,
                )
            if e.tgt not in vertex_kinds:
                raise CompileError(
                    f"signature {decl.name!r}: edge_kind {e.name!r} has "
                    f"unknown target vertex_kind {e.tgt!r}",
                    e.line,
                    e.col,
                )
            edge_kinds[e.name] = EdgeKind(
                name=e.name,
                src=e.src,
                tgt=e.tgt,
                directed=e.directed,
            )

        # Constructors. Every sort mentioned in a constructor must
        # be declared in the signature's `sorts { … }` block —
        # auto-registering an undeclared sort would mask a real
        # declaration error and leave its dim unspecified.
        _RESERVED_OP_NAMES = {"BoundVar", "Data"}
        constructors: dict[str, Constructor] = {}
        for c in decl.constructors:
            if c.name in _RESERVED_OP_NAMES:
                raise CompileError(
                    f"signature {decl.name!r}: constructor name {c.name!r} "
                    f"is reserved by the framework",
                    c.line,
                    c.col,
                )
            if c.name in constructors:
                raise CompileError(
                    f"signature {decl.name!r}: duplicate constructor {c.name!r}",
                    c.line,
                    c.col,
                )
            for s in c.domain:
                if s not in sorts:
                    raise CompileError(
                        f"signature {decl.name!r}: constructor {c.name!r} "
                        f"references undeclared sort {s!r}; declare it in "
                        f"the signature's `sorts {{ … }}` block",
                        c.line,
                        c.col,
                    )
            if c.codomain not in sorts:
                raise CompileError(
                    f"signature {decl.name!r}: constructor {c.name!r} has "
                    f"unknown codomain sort {c.codomain!r}",
                    c.line,
                    c.col,
                )
            constructors[c.name] = Constructor(
                name=c.name,
                domain=c.domain,
                codomain=c.codomain,
            )

        # Binders. Every sort a binder mentions (variable sort,
        # annotation sort, scoped argument sort, codomain) must
        # already be declared in the signature's `sorts { … }` block
        # — binders introduce structural recursion, so silently
        # auto-registering an object sort would mask a real
        # declaration error and produce a sort whose dim the user
        # never specified.
        binders: dict[str, Binder] = {}
        for b in decl.binders:
            if b.name in _RESERVED_OP_NAMES:
                raise CompileError(
                    f"signature {decl.name!r}: binder name {b.name!r} is "
                    f"reserved by the framework",
                    b.line,
                    b.col,
                )
            if b.name in binders or b.name in constructors:
                raise CompileError(
                    f"signature {decl.name!r}: duplicate binder {b.name!r}",
                    b.line,
                    b.col,
                )
            for v in b.binds:
                if v.sort not in sorts:
                    raise CompileError(
                        f"signature {decl.name!r}: binder {b.name!r} introduces "
                        f"variable of undeclared sort {v.sort!r}",
                        b.line,
                        b.col,
                    )
                if v.annot_sort is not None and v.annot_sort not in sorts:
                    raise CompileError(
                        f"signature {decl.name!r}: binder {b.name!r} variable "
                        f"{v.var!r} annotated by undeclared sort "
                        f"{v.annot_sort!r}",
                        b.line,
                        b.col,
                    )
            for a in b.scoped:
                if a.sort not in sorts:
                    raise CompileError(
                        f"signature {decl.name!r}: binder {b.name!r} scoped arg "
                        f"{a.arg!r} has undeclared sort {a.sort!r}",
                        b.line,
                        b.col,
                    )
            if b.codomain not in sorts:
                raise CompileError(
                    f"signature {decl.name!r}: binder {b.name!r} has "
                    f"unknown codomain sort {b.codomain!r}",
                    b.line,
                    b.col,
                )
            binders[b.name] = Binder(
                name=b.name,
                binds=tuple(
                    BinderVarSpec(
                        var=v.var,
                        sort=v.sort,
                        annot_sort=v.annot_sort,
                    )
                    for v in b.binds
                ),
                scoped=tuple(BinderArgSpec(arg=a.arg, sort=a.sort) for a in b.scoped),
                codomain=b.codomain,
            )

        sig = Signature(
            name=decl.name,
            params=decl.params,
            sorts_t=tuple(sorts.values()),
            constructors_t=tuple(constructors.values()),
            binders_t=tuple(binders.values()),
            vertex_kinds_t=tuple(vertex_kinds.values()),
            edge_kinds_t=tuple(edge_kinds.values()),
        )
        self._signatures[decl.name] = sig

    def _resolve_dim(
        self,
        sig: "Signature",
        sort: str,
        overrides: dict[str, int],
        diag_owner: str,
    ) -> int:
        """Resolve a sort's embedding dimension.

        Priority: the per-encoder / per-decoder dim override
        from the DSL block (``dim Term = 64``), then the
        signature's sort declaration. Raises if neither supplies a
        dim — the user must specify one somewhere.
        """
        if sort in overrides:
            return overrides[sort]
        d = sig.sort_dim(sort)
        if d is not None:
            return d
        raise CompileError(
            f"{diag_owner}: sort {sort!r} has no dim — declare it on the "
            f"signature's `sorts {{ … }}` block (e.g. `Term : object dim 64`) "
            f"or override it on the encoder / decoder block "
            f"(e.g. `dim Term = 64`)"
        )

    def _compile_encoder(self, decl: EncoderDecl) -> None:
        """Compile a encoder block into a runtime Encoder module."""
        if not hasattr(self, "_encoders"):
            self._encoders: dict[str, Encoder] = {}
        if not hasattr(self, "_signatures"):
            self._signatures = {}

        if decl.signature not in self._signatures:
            raise CompileError(
                f"encoder {decl.name!r}: unknown signature {decl.signature!r}",
                decl.line,
                decl.col,
            )
        sig = self._signatures[decl.signature]

        if decl.factory:
            encoder = _build_encoder_from_factory(decl, sig)
            self._encoders[decl.name] = encoder
            return

        # Per-sort dim resolution.
        overrides: dict[str, int] = {sd.sort: sd.dim for sd in decl.dims}
        sort_dims: dict[str, int] = {}
        _diag = f"encoder {decl.name!r}"
        for s_name, s in sig.sorts.items():
            sort_dims[s_name] = self._resolve_dim(
                sig,
                s_name,
                overrides,
                _diag,
            )
        for v_name in sig.vertex_kinds:
            sort_dims[v_name] = self._resolve_dim(
                sig,
                v_name,
                overrides,
                _diag,
            )

        # Set the compiler's per-let globals so let-expressions in
        # per-op bodies can reference other module-level morphisms,
        # signatures, encoders, deductions, etc.
        globs = self._lex_globals_for_structural()

        modules_owned: list[nn.Module] = []
        op_fns: dict[str, _PerOpFn] = {}

        for rule in decl.op_rules:
            op = rule.op
            if op in sig.constructors:
                domain = sig.constructors[op].domain
            elif op in sig.binders:
                # `Binder.domain()` already produces the positional
                # sort sequence in the order the per-op function
                # receives children: annotation sorts (one per
                # annotated bound variable, outer-context) followed
                # by scoped argument sorts (extended-context).
                domain = sig.binders[op].domain()
            else:
                raise CompileError(
                    f"encoder {decl.name!r}: op {op!r} is not in signature "
                    f"{sig.name!r}",
                    rule.line,
                    rule.col,
                )
            args = rule.args

            if rule.args and len(rule.args) != len(domain):
                raise CompileError(
                    f"encoder {decl.name!r}: op {op!r} expects "
                    f"{len(domain)} arguments, got {len(rule.args)}",
                    rule.line,
                    rule.col,
                )

            body_fn = self._compile_let_expr(rule.body, globals_=globs)

            def make_call(
                body_fn=body_fn,
                args_=args,
                mode=rule.mode,
                state_var=rule.state_var,
                prefix_var=rule.prefix_var,
            ):
                if mode == "recurrent":
                    # The body sees the named children plus an
                    # alias `state_var` for the recursive child's
                    # already-computed embedding.
                    def call(*children):
                        env = {name: child for name, child in zip(args_, children)}
                        if state_var is not None:
                            # Convention: the recursive child is the
                            # last positional in the surface form
                            # `Cons(head, tail) recurrent state |-> ...`.
                            env[state_var] = children[-1]
                        return body_fn(env)

                    return call
                if mode == "attention":
                    # Children are the non-recursive args followed by
                    # (prefix_list, current_step_state) supplied by
                    # `_compress_attention_chain`.
                    def call(*children_with_extras):
                        non_rec = list(children_with_extras[:-2])
                        prefix_list = children_with_extras[-2]
                        state_arg = children_with_extras[-1]
                        # `args_` names the non-recursive children
                        # plus the recursive arg (as declared in the
                        # source). The recursive arg name is the
                        # last in `args_`; it sees the running step
                        # state, mirroring `recurrent`.
                        env = {name: child for name, child in zip(args_[:-1], non_rec)}
                        if args_:
                            env[args_[-1]] = state_arg
                        if prefix_var is not None:
                            env[prefix_var] = prefix_list
                        return body_fn(env)

                    return call

                def call(*children):
                    env = {name: child for name, child in zip(args_, children)}
                    return body_fn(env)

                return call

            op_fns[op] = _PerOpFn(
                op=op,
                mode=rule.mode,
                args=args,
                fn=make_call(),
                state_var=rule.state_var,
                prefix_var=rule.prefix_var,
            )

        # Scaffold defaults for any constructor / binder not given a
        # rule by the user. We compute the per-argument dim sequence
        # in the exact order the framework passes children to the
        # per-op function.
        for op_name in list(sig.constructors) + list(sig.binders):
            if op_name in op_fns:
                continue
            if op_name in sig.constructors:
                c = sig.constructors[op_name]
                arg_dims = tuple(sort_dims[s] for s in c.domain)
                out_dim = sort_dims[c.codomain]
            else:
                b = sig.binders[op_name]
                arg_dims = tuple(sort_dims[s] for s in b.domain())
                out_dim = sort_dims[b.codomain]
            mod, call = make_default_op_fn(op_name, arg_dims, out_dim)
            modules_owned.append(mod)
            op_fns[op_name] = _PerOpFn(
                op=op_name,
                mode="plain",
                args=(),
                fn=call,
            )

        # var_init functions for binders. We allocate one per
        # (variable_sort, annotation_sort) pair that actually appears
        # in the signature's binders, plus one per unannotated
        # variable sort. Each is a learned 2-layer MLP from the
        # annotation's dim (or zero, for unannotated) to the
        # variable sort's dim.
        var_init_fns: dict = {}
        seen_keys: set = set()
        for b in sig.binders.values():
            for spec in b.binds:
                key: tuple[str, str] | str
                if spec.annot_sort is not None:
                    key = (spec.sort, spec.annot_sort)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    in_dim = sort_dims[spec.annot_sort]
                    out_dim = sort_dims[spec.sort]
                    mod, call = make_default_var_init(in_dim, out_dim)
                    modules_owned.append(mod)
                    var_init_fns[key] = call
                else:
                    key = spec.sort
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    out_dim = sort_dims[spec.sort]
                    init_param = nn.Parameter(torch.randn(out_dim) * 0.1)
                    holder = nn.Module()
                    holder.register_parameter(
                        f"unannot_var_{spec.sort}",
                        init_param,
                    )
                    modules_owned.append(holder)

                    def make_unannot(p=init_param):
                        def call(_annot=None):
                            return p

                        return call

                    var_init_fns[key] = make_unannot()

        # User-supplied per-(var_sort, annot_sort) var_init bodies.
        # Each `var_init <V> from <A> as ty |-> body` declaration
        # overrides the scaffolded default for that exact pair; an
        # omitted `from <A>` clause refers to the unannotated case.
        for vi in decl.var_inits:
            body_fn = self._compile_let_expr(vi.body, globals_=globs)
            if vi.annot_sort is None:
                key: tuple[str, str] | str = vi.var_sort

                def make_call(body_fn=body_fn):
                    def call(_annot=None):
                        return body_fn({})

                    return call

                var_init_fns[key] = make_call()
            else:
                if vi.ty is None:
                    raise CompileError(
                        f"encoder {decl.name!r}: var_init for "
                        f"{vi.var_sort!r} from {vi.annot_sort!r} requires "
                        f"an `as <name>` clause to bind the annotation "
                        f"embedding in the body",
                        vi.line,
                        vi.col,
                    )
                key = (vi.var_sort, vi.annot_sort)

                def make_call(body_fn=body_fn, arg=vi.ty):
                    def call(ty):
                        return body_fn({arg: ty})

                    return call

                var_init_fns[key] = make_call()

        # Data embedders: one learnable table per data sort, keyed by
        # the registered vocabulary (built as encountered).
        data_embedders = self._build_data_embedders(sig, sort_dims, modules_owned)

        # Graph specialization.
        iterations = decl.iterations or 0
        init_fns: dict[str, "Callable"] = {}
        message_fns: dict[str, "Callable"] = {}
        update_fns: dict[str, "Callable"] = {}
        readout = None
        for ir in decl.init_rules:
            ib = self._compile_let_expr(ir.body, globals_=globs)

            def init_call(payload, body_fn=ib, arg=ir.arg):
                return body_fn({arg: payload})

            init_fns[ir.kind] = init_call
        for mr in decl.message_rules:
            mb = self._compile_let_expr(mr.body, globals_=globs)

            def msg_call(s, t, body_fn=mb, sv=mr.src, tv=mr.tgt):
                return body_fn({sv: s, tv: t})

            message_fns[mr.edge_kind] = msg_call
        for ur in decl.update_rules:
            ub = self._compile_let_expr(ur.body, globals_=globs)

            def upd_call(slf, msgs, body_fn=ub, sv=ur.self_var, mv=ur.msgs_var):
                return body_fn({sv: slf, mv: msgs})

            update_fns[ur.vertex_kind] = upd_call
        if decl.readout is not None:
            rb = self._compile_let_expr(decl.readout, globals_=globs)

            def readout_call(embeds, body_fn=rb):
                return body_fn({"embeds": embeds})

            readout = readout_call

        comp = Encoder(
            name=decl.name,
            signature=sig,
            sort_dims=sort_dims,
            op_fns=op_fns,
            var_init_fns=var_init_fns,
            data_embedders=data_embedders,
            modules_owned=modules_owned,
            iterations=iterations,
            init_fns=init_fns,
            message_fns=message_fns,
            update_fns=update_fns,
            readout=readout,
        )
        if decl.name in self._morphisms:
            raise CompileError(
                f"encoder {decl.name!r} name conflicts with existing morphism",
                decl.line,
                decl.col,
            )
        self._encoders[decl.name] = comp
        self._morphisms[decl.name] = comp

    def _compile_decoder(self, decl: DecoderDecl) -> None:
        """Compile a decoder block into a runtime Decoder module.

        Scaffolds, for each missing component, a properly-shaped
        learnable neural network — never a heuristic. The user's
        body overrides take precedence in every slot.
        """
        if not hasattr(self, "_decoders"):
            self._decoders: dict[str, Decoder] = {}
        if not hasattr(self, "_signatures"):
            self._signatures = {}

        if decl.signature not in self._signatures:
            raise CompileError(
                f"decoder {decl.name!r}: unknown signature {decl.signature!r}",
                decl.line,
                decl.col,
            )
        sig: Signature = self._signatures[decl.signature]

        overrides: dict[str, int] = {sd.sort: sd.dim for sd in decl.dims}
        sort_dims: dict[str, int] = {}
        _diag = f"decoder {decl.name!r}"
        for s_name in sig.sorts:
            sort_dims[s_name] = self._resolve_dim(
                sig,
                s_name,
                overrides,
                _diag,
            )

        globs = self._lex_globals_for_structural()
        modules_owned: list[nn.Module] = []

        # ---- structure heads, per sort ----
        # Each object sort needs one structure head emitting logits
        # over its candidate set (constructors + binders +
        # BoundVar). We size each head to the candidate set size.
        structure_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        for s_name, s in sig.sorts.items():
            if s.kind != "object":
                continue
            cands = []
            for c_name, c in sig.constructors.items():
                if c.codomain == s_name:
                    cands.append(c_name)
            for b_name, b in sig.binders.items():
                if b.codomain == s_name:
                    cands.append(b_name)
            # Reserve one extra slot for BoundVar; the runtime
            # always restricts to actually-available candidates.
            n_logits = max(len(cands) + 1, 2)
            head = nn.Linear(sort_dims[s_name], n_logits)
            modules_owned.append(head)

            def _make_struct(head=head):
                def call(v: torch.Tensor) -> torch.Tensor:
                    return head(v.reshape(-1))

                return call

            structure_fns[s_name] = _make_struct()

        # User-supplied structure override.
        if decl.structure is not None and decl.structure_arg is not None:
            sb = self._compile_let_expr(decl.structure, globals_=globs)

            def _struct_override(
                v: torch.Tensor, body_fn=sb, arg=decl.structure_arg
            ) -> torch.Tensor:
                return body_fn({arg: v})

            structure_fns["*"] = _struct_override

        # ---- primitive heads, per data sort ----
        # Each data sort needs a head over its (possibly empty)
        # closed vocabulary. The runtime raises if the vocab is
        # unpopulated; here we only allocate when the vocab is set
        # via the compiler's data_vocab attribute (declared
        # separately if and when needed).
        primitive_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        for s_name, s in sig.sorts.items():
            if s.kind != "data":
                continue
            vocab = self._data_vocab_for(sig).get(s_name, [])
            head = nn.Linear(sort_dims[s_name], max(len(vocab), 1))
            modules_owned.append(head)

            def _make_prim(head=head):
                def call(v: torch.Tensor) -> torch.Tensor:
                    return head(v.reshape(-1))

                return call

            primitive_fns[s_name] = _make_prim()

        if decl.primitive is not None and decl.primitive_arg is not None:
            pb = self._compile_let_expr(decl.primitive, globals_=globs)

            def _prim_override(
                v: torch.Tensor, body_fn=pb, arg=decl.primitive_arg
            ) -> torch.Tensor:
                return body_fn({arg: v})

            primitive_fns["*"] = _prim_override

        # ---- factor functions: per object sort, per arity ----
        # Every arity that occurs in the signature gets a learned
        # linear projection `dim -> n*dim` reshaped to a tuple of
        # n sub-vectors. This is the formally correct child split.
        factor_fns: dict[
            str, dict[int, Callable[[torch.Tensor], tuple[torch.Tensor, ...]]]
        ] = {}
        arities_by_sort: dict[str, set[int]] = {}
        for c in sig.constructors.values():
            if c.arity > 0:
                arities_by_sort.setdefault(c.codomain, set()).add(c.arity)
        for b in sig.binders.values():
            if b.arity > 0:
                arities_by_sort.setdefault(b.codomain, set()).add(b.arity)

        for sort, arities in arities_by_sort.items():
            d = sort_dims[sort]
            per_arity: dict[
                int, Callable[[torch.Tensor], tuple[torch.Tensor, ...]]
            ] = {}
            for n in arities:
                lin = nn.Linear(d, d * n)
                modules_owned.append(lin)

                def _make_factor(lin=lin, n=n, d=d):
                    def call(v: torch.Tensor) -> tuple[torch.Tensor, ...]:
                        out = lin(v.reshape(-1))
                        return tuple(out[i * d : (i + 1) * d] for i in range(n))

                    return call

                per_arity[n] = _make_factor()
            factor_fns[sort] = per_arity

        if decl.factor is not None and decl.factor_arg is not None:
            fb = self._compile_let_expr(decl.factor, globals_=globs)

            # The user-supplied factor body is evaluated with the
            # parent vector bound to `decl.factor_arg` and the arity
            # bound to ``n``. It must return a list / tuple of
            # exactly ``n`` sub-vectors. We close over each arity at
            # install time so the runtime sees a per-(sort, n)
            # function with the canonical (vec) -> tuple shape.
            def _make_factor_at_arity(n: int):
                def call(v: torch.Tensor) -> tuple[torch.Tensor, ...]:
                    result = fb({decl.factor_arg: v, "n": n})
                    if not isinstance(result, (list, tuple)):
                        raise RuntimeError(
                            f"decoder {decl.name!r}: factor body must return "
                            f"a list or tuple of sub-vectors, got "
                            f"{type(result).__name__}"
                        )
                    if len(result) != n:
                        raise RuntimeError(
                            f"decoder {decl.name!r}: factor body at arity "
                            f"{n} returned {len(result)} sub-vectors"
                        )
                    return tuple(result)

                return call

            for sort, per_arity in factor_fns.items():
                for n in list(per_arity):
                    per_arity[n] = _make_factor_at_arity(n)

        # ---- binder_select: scores in-scope variables ----
        # A small bilinear scorer between the parent vector and each
        # in-scope variable's embedding. Required by the runtime
        # whenever a BoundVar choice may fire OR an index-sorted
        # child position is decoded.
        principal_dim = next(iter(sort_dims.values()))
        bs_query = nn.Linear(principal_dim, principal_dim)
        bs_key = nn.Linear(principal_dim, principal_dim)
        modules_owned.extend([bs_query, bs_key])

        def _binder_select_default(
            v: torch.Tensor,
            embeds: list[torch.Tensor],
            q=bs_query,
            k=bs_key,
        ) -> torch.Tensor:
            qv = q(v.reshape(-1))
            keys = torch.stack([k(e.reshape(-1)) for e in embeds], dim=0)
            return keys @ qv

        binder_select_fn: Callable[[torch.Tensor, list[torch.Tensor]], torch.Tensor]
        if decl.binder_select is not None and decl.binder_select_arg is not None:
            bb = self._compile_let_expr(decl.binder_select, globals_=globs)

            def _bs_override(
                v: torch.Tensor,
                embeds: list[torch.Tensor],
                body_fn=bb,
                arg=decl.binder_select_arg,
            ) -> torch.Tensor:
                return body_fn({arg: v, "embeds": embeds})

            binder_select_fn = _bs_override
        else:
            binder_select_fn = _binder_select_default

        dec = Decoder(
            name=decl.name,
            signature=sig,
            sort_dims=sort_dims,
            depth=decl.depth,
            structure_fns=structure_fns,
            primitive_fns=primitive_fns,
            factor_fns=factor_fns,
            binder_select_fn=binder_select_fn,
            data_vocab=self._data_vocab_for(sig),
            modules_owned=modules_owned,
        )
        if decl.name in self._morphisms:
            raise CompileError(
                f"decoder {decl.name!r} name conflicts with existing morphism",
                decl.line,
                decl.col,
            )
        self._decoders[decl.name] = dec
        self._morphisms[decl.name] = dec

    def _compile_loss(self, decl: LossDecl) -> None:
        """Compile a loss declaration into a registry entry."""
        if not hasattr(self, "_loss_registry"):
            self._loss_registry = LossRegistry()

        globs = self._lex_globals_for_structural()
        body_fn = self._compile_let_expr(decl.body, globals_=globs)
        weight_fn = None
        if decl.weight is not None:
            weight_fn = self._compile_let_expr(decl.weight, globals_=globs)
        att = decl.attachment
        self._loss_registry.add(
            LossEntry(
                name=decl.name,
                body=body_fn,
                weight=weight_fn,
                attachment_kind=att.attachment_kind,
                target=att.target,
                rule_deduction=att.rule_deduction,
            )
        )

    def _lex_globals_for_structural(self) -> dict:
        """Build the globals dict visible to encoder/decoder/loss
        let-expression bodies. Includes morphisms, encoders,
        decoders, deductions, signatures."""
        globs: dict = {}
        globs.update(self._morphisms)
        for attr in ("_encoders", "_decoders", "_deductions", "_signatures"):
            d = getattr(self, attr, None)
            if d:
                globs.update(d)
        return globs

    def _build_data_embedders(
        self,
        sig: "Signature",
        sort_dims: dict[str, int],
        modules_owned: list,
    ) -> dict[str, "Callable"]:
        """For each data-sort in the signature, build an open-vocab
        keyed embedding table: each distinct data leaf encountered
        at the sort gets a learnable per-key vector allocated on
        first lookup.

        Dim is sourced strictly from ``sort_dims`` — every data
        sort in the signature must have its dim resolved before this
        runs (which the compiler enforces by calling ``_resolve_dim``
        on every sort up front and raising on missing dims).
        """
        out: dict[str, Callable] = {}
        for s_name, s in sig.sorts.items():
            if s.kind != "data":
                continue
            if s_name not in sort_dims:
                raise CompileError(
                    f"encoder over {sig.name!r}: data sort {s_name!r} "
                    f"has no resolved dim"
                )
            dim = sort_dims[s_name]
            table = nn.ParameterDict()
            modules_owned.append(table)

            def make_embed(table=table, dim=dim):
                def call(key):
                    skey = str(key).replace(".", "_")
                    if skey not in table:
                        p = nn.Parameter(torch.randn(dim) * 0.1)
                        table[skey] = p
                    return table[skey]

                return call

            out[s_name] = make_embed()
        return out

    def _data_vocab_for(self, sig: "Signature") -> dict[str, list]:
        """Return the per-data-sort closed vocabulary for use by the
        decoder's primitive heads.

        Vocabularies are declared inline in the signature's
        ``sorts { … }`` block via the ``vocab { … }`` clause on a
        data sort. The runtime list is the surface declaration's
        Python-decoded values in declaration order; the decoder's
        primitive head and ``log_prob`` use this order to index
        token positions.
        """
        return {
            s.name: list(s.vocab_values) for s in sig.sorts.values() if s.kind == "data"
        }
