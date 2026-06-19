"""Public surface for `quivers.transpile`.

Defines the [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
error raised when a backend cannot represent a QVR construct, and the
support-tier frozensets every backend declares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from quivers.dsl.ast_nodes import Module, Statement


class UnsupportedConstruct(Exception):
    """Raised when a backend cannot transpile one or more QVR constructs.

    Attributes
    ----------
    target
        The backend name (e.g., ``"qvr-stan"``).
    kinds
        The unsupported QVR construct kinds, sorted, deduplicated.
        Each kind is a structured identifier (``"family:Wishart"``,
        ``"node:IRScore"``, ``"declare:vector:event-rank:0"``, ...)
        consumers can match programmatically.

    Notes
    -----
    The exception's `str` is the user-facing message: it translates
    the structured `kinds` list into a plain-English explanation of
    what the user wrote that the backend can't accept and (when
    available) a concrete workaround. The structured `kinds` list is
    preserved as the `kinds` attribute for programmatic dispatch
    (test harnesses, error handlers, downstream tooling).
    """

    def __init__(self, target: str, kinds: list[str]) -> None:
        self.target = target
        self.kinds = sorted(set(kinds))
        super().__init__(
            _user_facing_message(target, self.kinds)
        )


def _backend_display_name(target: str) -> str:
    """Strip the ``qvr-`` prefix the per-backend `target` carries
    internally and return a backend identifier the user typed
    (e.g. ``"qvr-stan"`` -> ``"stan"``)."""
    if target.startswith("qvr-"):
        return target[4:]
    return target


def _user_facing_message(target: str, kinds: list[str]) -> str:
    """Translate the structured `kinds` list into a user-shaped
    explanation. Each kind is parsed by its colon-separated prefix
    and rendered as a one-line summary plus, when applicable, a
    concrete workaround or pointer to the right surface change.

    Multi-kind errors are joined with newlines so the user sees
    one bullet per problem.
    """
    backend = _backend_display_name(target)
    lines = [_render_kind(backend, k) for k in kinds]
    if len(lines) == 1:
        return lines[0]
    bullet = "\n  - "
    return (
        f"{backend} cannot transpile this program:"
        f"{bullet}{bullet.join(lines)}"
    )


def _render_kind(backend: str, kind: str) -> str:
    """One-line user-shaped rendering of a single
    [`UnsupportedConstruct.kinds`][quivers.transpile.UnsupportedConstruct.kinds]
    entry.

    Recognises every prefix the pipeline currently emits; falls
    back to the bare kind string for an unrecognised prefix so a
    new prefix at least surfaces literally rather than being lost.
    """
    head, _, tail = kind.partition(":")
    if head == "family":
        return _render_family_kind(backend, tail)
    if head == "node":
        return _render_node_kind(backend, tail)
    if head == "declare":
        return _render_declare_kind(backend, tail)
    if head == "arg":
        return _render_arg_kind(backend, tail)
    if head == "let-expr":
        return _render_let_expr_kind(backend, tail)
    if head == "let":
        return _render_let_kind(backend, tail)
    if head == "return":
        return _render_return_kind(backend, tail)
    if head == "broadcast":
        return _render_broadcast_kind(backend, tail)
    if head == "step":
        return _render_step_kind(backend, tail)
    if head == "axes":
        return _render_axes_kind(backend, tail)
    if head == "option":
        return _render_option_kind(backend, tail)
    if head == "marginalize":
        return _render_marginalize_kind(backend, tail)
    if head == "dim":
        return _render_dim_kind(backend, tail)
    if head == "ctx":
        return f"internal transpile error in {backend}: {kind!r}"
    if head == "draw-arg":
        return (
            f"{backend} does not yet handle the {tail!r} kind in a "
            f"distribution-argument position"
        )
    if head == "transform":
        return (
            f"{backend} does not support the {tail!r} transform; the "
            f"supported set is `inv_square`, `inv`, `neg`, `log`, `exp`"
        )
    return f"{kind} ({backend}) -- no user-facing translation registered"


_NO_TARGET_HEADS: frozenset[str] = frozenset({
    "no-stan-target",
    "no-bugs-target",
    "no-jags-target",
    "no-target-name",
    "no-webppl-target",
    "no-pymc-target",
    "no-edward2-target",
    "no-numpyro-target",
    "no-pyro-target",
    "no-gen-target",
    "no-turing-target",
    "no-church-target",
})


def _render_family_kind(backend: str, tail: str) -> str:
    """`family:<F>:<detail>` -- the backend has no mapping for the
    distribution family `F`, or the mapping exists but a downstream
    step (sentinel construction, shape derivation) failed.

    Two leading-token shapes are recognised:

    * ``family:<F>`` or ``family:<F>:<detail>`` -- the usual form,
      with `F` as the family name.
    * ``family:<no-*-target>:<F>`` -- the "no native mapping" form
      where the FAMILY_META lookup raised before reaching the
      backend-specific handler; the family name lives in the detail
      slot.
    """
    family, _, detail = tail.partition(":")
    if family in _NO_TARGET_HEADS:
        # Reshape from `family:no-stan-target:TruncatedNormal` to
        # `family:TruncatedNormal:no-stan-target` so the message
        # references the family the user wrote rather than the
        # absence sentinel.
        family, detail = detail, family
    if not detail:
        return (
            f"{backend} has no native `{family}` distribution; the "
            f"FAMILY_META registry has no mapping for it on this "
            f"backend. Pick a family this backend supports, or wrap "
            f"the call site in a backend-specific helper."
        )
    if detail in _NO_TARGET_HEADS or detail.startswith("no-"):
        return (
            f"{backend} has no native `{family}` distribution. Pick a "
            f"family this backend supports, or wrap the call site in "
            f"a backend-specific helper."
        )
    if detail == "sentinel-failed" or detail.startswith("sentinel-failed"):
        sentinel_detail = detail.removeprefix(
            "sentinel-failed:"
        ).removeprefix("sentinel-failed")
        return (
            f"the QVR compiler could not derive the shape sentinel for "
            f"`{family}`. This usually means a positional / keyword arg "
            f"the family requires is missing from the call site. "
            f"Underlying cause: {sentinel_detail.strip(': ') or 'unspecified'}. "
            f"Fix: supply every required argument explicitly at the call "
            f"site rather than relying on sentinel inference."
        )
    if detail == "unknown" or detail.startswith("unknown"):
        return (
            f"`{family}` is not a known distribution family in "
            f"FAMILY_META. Either you typed the name wrong, or the "
            f"family needs to be added to the registry."
        )
    if detail.startswith("arity-mismatch"):
        rest = detail.removeprefix("arity-mismatch:").strip()
        return (
            f"the call to `{family}` has the wrong number of arguments. "
            f"{rest}. Check the family's positional arg order against the "
            f"FAMILY_META registry."
        )
    if detail.startswith("wrapper-inner-unknown"):
        inner = detail.removeprefix("wrapper-inner-unknown:").strip()
        return (
            f"the wrapper family `{family}` references an inner family "
            f"`{inner}` that {backend} does not recognise."
        )
    return (
        f"{backend} cannot transpile `{family}` ({detail})"
    )


def _render_node_kind(backend: str, tail: str) -> str:
    """`node:<NodeKind>[:<detail>]` -- the IR node kind isn't handled
    by the renderer's dispatch."""
    node, _, detail = tail.partition(":")
    detail = detail.lstrip(": ").strip()
    if node == "IRScore":
        rest = detail or (
            f"{backend} has no host-level `target += ...` analogue"
        )
        return (
            f"`score` steps cannot be transpiled to {backend}: {rest}"
        )
    if not detail:
        return f"{backend} renderer does not handle the `{node}` IR node kind"
    return f"{backend} renderer does not handle `{node}`: {detail}"


def _render_declare_kind(backend: str, tail: str) -> str:
    """`declare:<type>:<dimension-issue>` -- the renderer needs more
    event-dim information than the IR currently carries."""
    type_, _, rest = tail.partition(":")
    if "event-rank" in rest:
        return (
            f"the declared shape for a `{type_}` parameter is incompatible "
            f"with {backend}'s grammar: {rest}. The fixture's plate "
            f"decomposition does not annotate the event axes the renderer "
            f"needs. Either add explicit event-axes (`[over=...]`) or use "
            f"a shape-aware family at the call site."
        )
    if type_ == "unsupported-support":
        return (
            f"{backend} has no declaration form for the support kind "
            f"`{rest}`."
        )
    return f"{backend} cannot declare `{type_}` shape: {rest}"


def _render_arg_kind(backend: str, tail: str) -> str:
    """`arg:<shape-detail>` -- the renderer can't accept the given
    argument shape (broadcast, family-ref, literal)."""
    if tail.startswith("broadcast"):
        return (
            f"{backend} cannot inline-broadcast a scalar literal to a "
            f"vector / matrix argument position. Pre-bind the value as "
            f"a vector data input, or supply an already-shaped argument."
        )
    if tail.startswith("family-ref"):
        rest = tail.removeprefix("family-ref:").strip()
        return (
            f"{backend} does not accept a family-reference argument "
            f"({rest or 'wrapper distributions inline via the target idiom'})."
        )
    if tail.startswith("list-literal"):
        return (
            f"{backend} does not accept a list literal `[a, b, c]` in a "
            f"distribution-argument position. Pre-bind the list as a "
            f"data input or use a shape-aware family that takes the "
            f"elements positionally."
        )
    if tail.startswith("matrix-literal"):
        return (
            f"{backend} does not accept a matrix literal `[[a, b], ...]` in "
            f"a distribution-argument position."
        )
    return f"{backend} cannot accept this argument shape: {tail}"


def _render_let_expr_kind(backend: str, tail: str) -> str:
    """`let-expr:<Kind>[:<reason>]` -- a let-expression `Kind` the
    renderer's helper can't unroll."""
    kind, _, rest = tail.partition(":")
    if kind == "LetExprFactor" and rest.startswith("unresolved-binder-size"):
        binder = rest.partition(":")[2]
        return (
            f"{backend} cannot determine the size of the factor binder "
            f"`{binder}`; declare its index type explicitly so the "
            f"renderer can resolve the binder cardinality from the "
            f"cards table."
        )
    if not rest:
        return f"{backend} cannot emit a let-expression of kind `{kind}`"
    return f"{backend} cannot emit `{kind}` ({rest})"


def _render_let_kind(backend: str, tail: str) -> str:
    """`let:<reason>` -- a let-binding the renderer can't resolve."""
    if tail.startswith("composite_expression"):
        rest = tail.partition(":")[2]
        return (
            f"the let-binding resolves to a composite expression that "
            f"{backend} cannot unfold: {rest}. Replace the composition "
            f"with a direct `~ Family(args)` declaration, or split it "
            f"into one `sample` per stochastic step."
        )
    return f"{backend} cannot resolve let-binding: {tail}"


def _render_return_kind(backend: str, tail: str) -> str:
    """`return:undeclared:<name>` -- a `return <name>` whose name
    was never declared, so the renderer cannot derive a shape for
    the generated-quantities alias."""
    if tail.startswith("undeclared"):
        name = tail.removeprefix("undeclared:").strip()
        return (
            f"`return {name}` references a name that was not declared as a "
            f"sample, observe, or let earlier in the program; {backend} "
            f"needs a declared shape to emit the return alias."
        )
    return f"{backend} cannot emit return: {tail}"


def _render_broadcast_kind(backend: str, tail: str) -> str:
    """`broadcast:<reason>` -- the broadcast op cannot represent the
    target shape."""
    return f"{backend} cannot broadcast: {tail}"


def _render_step_kind(backend: str, tail: str) -> str:
    """`step:<kind>` -- the program contains a step kind the
    renderer's body walker does not implement."""
    return f"{backend} does not yet emit the `{tail}` program step"


def _render_axes_kind(backend: str, tail: str) -> str:
    """`axes:<reason>` -- an axes-spec the renderer can't honor."""
    return f"{backend} cannot honor the axes specification: {tail}"


def _render_option_kind(backend: str, tail: str) -> str:
    """`option:<kind>` -- an OptionValue kind the renderer ignores."""
    return f"{backend} does not yet honor the `{tail}` option"


def _render_marginalize_kind(backend: str, tail: str) -> str:
    """`marginalize:<reason>` -- a marginalize block the renderer can
    detect but cannot enumerate."""
    if tail.startswith("non-finite-support"):
        family = tail.partition(":")[2]
        return (
            f"`marginalize` over `{family}` is not finitely enumerable; "
            f"{backend} cannot integrate it out at compile time. Use a "
            f"continuous-relaxation reparameterisation, or move the "
            f"marginalisation into the host-side inference loop."
        )
    if tail.startswith("unknown-cardinality"):
        family = tail.partition(":")[2]
        return (
            f"`marginalize` over `{family}` requires a known support "
            f"cardinality at compile time, which {backend} could not "
            f"derive. Annotate the index type with its cardinality "
            f"explicitly."
        )
    if tail.startswith("scope:IRDeterministic"):
        rest = tail.partition(":")[2]
        return (
            f"{backend} cannot emit a deterministic let inside a "
            f"marginalize scope: {rest}. Hoist the let out of the "
            f"`marginalize` block."
        )
    return f"{backend} cannot marginalize: {tail}"


def _render_dim_kind(backend: str, tail: str) -> str:
    """`dim:<reason>` -- an unsupported Dim variant."""
    return f"{backend} cannot resolve dim: {tail}"


#: Statement kinds every PPL backend accepts: the probabilistic-program
#: surface (declarations + program bodies of sample / observe / let /
#: score / return / marginalize). Excludes categorical-algebra and
#: neural-network declarations.
STAN_LIKE: frozenset[str] = frozenset(
    {
        "object_decl",
        "morphism_decl",
        "let_decl",
        "program_decl",
        "export_decl",
    }
)

#: Categorical-metadata declarations a backend's walker may silently
#: ignore when a `program_decl` is present alongside them. When a
#: module carries ONLY these declarations and no `program_decl`, the
#: walker still raises `UnsupportedConstruct` listing the kinds (so
#: the construct-matrix test continues to verify rejection of
#: standalone categorical declarations).
CATEGORICAL_METADATA_IGNORABLE: frozenset[str] = frozenset(
    {
        "category_decl",
        "schema_decl",
        "composition_decl",
        "bundle_decl",
        "rule_decl",
        "contraction_decl",
        "signature_decl",
        "deduction_decl",
    }
)

#: Adds encoder/decoder declarations for backends with a deep-learning
#: idiom (Pyro modules, NumPyro/Flax modules, Edward2/TF, PyMC custom
#: dists).
PYTHON_DEEP: frozenset[str] = STAN_LIKE | frozenset(
    {"encoder_decl", "decoder_decl"}
)

#: Probabilistic subset; Church/WebPPL realise `marginalize` as a
#: continuation-style `Infer` / `enumerate-query`.
CHURCH_LIKE: frozenset[str] = STAN_LIKE


class Backend(Protocol):
    """The protocol every backend module satisfies.

    Backends register themselves via
    [`didactic.codegen.emitter`][didactic.codegen.emitter] under a
    ``"qvr-<name>"`` key. Quivers' top-level
    [`transpile`][quivers.transpile.transpile] dispatches by looking up
    the registered emitter, then delegates to its
    [`emit_instance`][didactic.codegen.Emitter.emit_instance].

    Attributes
    ----------
    file_extension
        Canonical filename extension (``"stan"``, ``"py"``, ``"jl"``,
        ``"js"``, ``"scm"``).
    grammar
        The tree-sitter grammar name backing this backend, as accepted by
        [`panproto.AstParserRegistry.parse_with_protocol`][panproto.AstParserRegistry.parse_with_protocol].
    support
        The probabilistic-subset support tier accepted by this backend.
    """

    file_extension: str
    grammar: str
    support: frozenset[str]

    def emit_instance(self, module: Module) -> bytes:
        """Transpile a parsed QVR module to bytes."""
        ...


def unsupported_for(
    target: str, module: Module, *, allow: frozenset[str]
) -> None:
    """Raise [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
    if ``module`` contains statement kinds outside ``allow``.

    Walks the module's top-level statements; the ``kind`` field is the
    didactic [`TaggedUnion`][didactic.api.TaggedUnion] discriminator
    (``"program_decl"``, ``"morphism_decl"``, etc.). Any kind not in
    ``allow`` is collected; if the resulting set is non-empty, raises.
    """
    bad: set[str] = set()
    for statement in module.statements:
        kind = cast_kind(statement)
        if kind not in allow:
            bad.add(kind)
    if bad:
        raise UnsupportedConstruct(target, sorted(bad))


def cast_kind(statement: Statement) -> str:
    """Return ``statement.kind`` as a string.

    The didactic [`TaggedUnion`][didactic.api.TaggedUnion] discriminator
    is typed `Literal[...]`; the cast is a single boundary line so the
    caller stays free of literal-narrowing noise.
    """
    return str(getattr(statement, "kind"))
