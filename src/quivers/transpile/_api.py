"""Public surface for `quivers.transpile`.

Defines the [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
error raised when a backend cannot represent a QVR construct, and the
support-tier frozensets every backend declares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from quivers.transpile._diagnostics import (
    RefusedDeclaration,
    user_facing_message,
)

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
    declarations
        The top-level declarations the refusal is about, when it is
        about declarations, so the message can name them the way the
        user wrote them and point at their lines.
    module_has_program
        Whether the refused module declares a probabilistic program at
        all. A module that does not is refused for a further reason
        the message states.

    Notes
    -----
    The exception's `str` is the user-facing message: it translates
    the structured `kinds` list into a plain-English account of what
    the user wrote, where, why the target cannot take it, and what to
    write instead. The structured `kinds` list is preserved as the
    `kinds` attribute for programmatic dispatch (test harnesses, error
    handlers, downstream tooling); see
    [`quivers.transpile._diagnostics`][quivers.transpile._diagnostics]
    for the grammar a kind follows.
    """

    def __init__(
        self,
        target: str,
        kinds: list[str],
        *,
        declarations: tuple[RefusedDeclaration, ...] = (),
        module_has_program: bool = False,
    ) -> None:
        self.target = target
        self.kinds = sorted(set(kinds))
        self.declarations = declarations
        self.module_has_program = module_has_program
        super().__init__(
            user_facing_message(
                target,
                tuple(self.kinds),
                declarations,
                module_has_program,
            )
        )


_NO_TARGET_HEADS: frozenset[str] = frozenset(
    {
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
    }
)


#: Statement kinds every PPL backend accepts: the probabilistic-program
#: surface (declarations + program bodies of sample / observe / let /
#: score / return / marginalize). Excludes categorical-algebra and
#: neural-network declarations.
STAN_LIKE: frozenset[str] = frozenset(
    {
        "object_decl",
        "morphism_decl",
        "define_decl",
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
PYTHON_DEEP: frozenset[str] = STAN_LIKE | frozenset({"encoder_decl", "decoder_decl"})

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


def unsupported_for(target: str, module: Module, *, allow: frozenset[str]) -> None:
    """Raise [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
    if ``module`` contains statement kinds outside ``allow``.

    Walks the module's top-level statements; the ``kind`` field is the
    didactic [`TaggedUnion`][didactic.api.TaggedUnion] discriminator
    (``"program_decl"``, ``"morphism_decl"``, etc.). Any kind not in
    ``allow`` is collected; if the resulting set is non-empty, raises.

    [`CATEGORICAL_METADATA_IGNORABLE`][quivers.transpile.CATEGORICAL_METADATA_IGNORABLE]
    kinds (``composition_decl``, ``category_decl``, ``schema_decl``,
    ``bundle_decl``, ``rule_decl``, ``contraction_decl``,
    ``signature_decl``, ``deduction_decl``) are accepted ALONGSIDE a
    ``program_decl``: when the module has at least one program, the
    walker ignores these metadata declarations and transpiles the
    program. Without a ``program_decl`` they remain in `bad` so the
    contract surfaces "no probabilistic program here to transpile."
    """
    kinds = {cast_kind(s) for s in module.statements}
    has_program = "program_decl" in kinds
    effective_allow = allow | CATEGORICAL_METADATA_IGNORABLE if has_program else allow
    bad: set[str] = set()
    for statement in module.statements:
        kind = cast_kind(statement)
        if kind not in effective_allow:
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
