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
    """

    def __init__(self, target: str, kinds: list[str]) -> None:
        self.target = target
        self.kinds = sorted(set(kinds))
        super().__init__(
            f"backend {target!r} does not support construct kinds: "
            f"{', '.join(self.kinds)}"
        )


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
