"""Transpile a compiled QVR program to other probabilistic-programming
languages, via panproto's tree-sitter emission.

Every backend is a [`didactic.codegen.Emitter`][didactic.codegen.Emitter]
registered via [`@dx.codegen.emitter("qvr-<name>")`][didactic.codegen.emitter]
(e.g. ``"qvr-stan"``). The
[`transpile`][quivers.transpile.transpile] function and the
``qvr transpile`` CLI subcommand are thin sugar over
[`didactic.codegen.lookup_emitter`][didactic.codegen.lookup_emitter].

The pipeline is fixed:
1. [`extract_program_schema`][quivers.dsl.program_theory.extract_program_schema]
   turns the resolved QVR program into a `panproto.Schema`.
2. A backend-specific
   [`SchemaTransform`][quivers.transpile._pipeline.SchemaTransform]
   walks the QVR schema and builds the target schema via
   [`panproto.SchemaBuilder`][panproto.SchemaBuilder].
3. [`panproto.AstParserRegistry.emit_pretty`][panproto.AstParserRegistry.emit_pretty]
   renders the target schema as source bytes.

No backend constructs source via string interpolation; emission flows
through panproto's grammar walker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from didactic.codegen import list_emitters

# `lookup_emitter` is documented and stable but not re-exported from
# `didactic.codegen.__init__`; import directly from the implementation
# module.
from didactic.codegen._emitter import lookup_emitter

from quivers.transpile._api import (
    CHURCH_LIKE,
    PYTHON_DEEP,
    STAN_LIKE,
    Backend,
    UnsupportedConstruct,
    unsupported_for,
)
from quivers.transpile._pipeline import (
    EmitPretty,
    SchemaTransform,
    parser_registry,
    realize,
    target_protocol,
)

# Side-effect: each backend module's import registers itself via
# `@dx.codegen.emitter("qvr-<name>")`.
from quivers.transpile.backends import stan as _stan  # noqa: F401

if TYPE_CHECKING:
    from quivers.dsl.ast_nodes import Module


def transpile(module: Module, *, target: str) -> bytes:
    """Transpile a QVR module to the named ``target`` backend.

    Parameters
    ----------
    module
        The parsed [`Module`][quivers.dsl.ast_nodes.Module] AST.
    target
        A registered backend key (without the ``qvr-`` prefix), e.g.
        ``"stan"``.

    Returns
    -------
    bytes
        The transpiled source bytes.

    Raises
    ------
    LookupError
        If no backend is registered under ``"qvr-<target>"``.
    UnsupportedConstruct
        If the module contains constructs the chosen backend does not
        support.
    """
    name = f"qvr-{target}"
    emitter = lookup_emitter(name)
    if emitter is None:
        msg = (
            f"transpile(target={target!r}): no backend registered. "
            f"Available: {available_targets()}"
        )
        raise LookupError(msg)
    # `Emitter.emit_instance` is typed `(Model) -> bytes`; quivers
    # backends accept a `Module` instead. The Emitter Protocol is
    # `runtime_checkable` and duck-typed; the cast is the static-type
    # boundary.
    return cast("Backend", emitter).emit_instance(module)


def available_targets() -> list[str]:
    """List every registered ``qvr-<name>`` backend, sorted."""
    return sorted(
        n.removeprefix("qvr-")
        for n in list_emitters()
        if n.startswith("qvr-")
    )


__all__ = [
    "CHURCH_LIKE",
    "PYTHON_DEEP",
    "STAN_LIKE",
    "Backend",
    "EmitPretty",
    "SchemaTransform",
    "UnsupportedConstruct",
    "available_targets",
    "parser_registry",
    "realize",
    "target_protocol",
    "transpile",
    "unsupported_for",
]
