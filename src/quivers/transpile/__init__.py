"""Transpile a compiled QVR module to other probabilistic-programming
languages.

The pipeline is a [`didactic.api.Mapping`][didactic.api.Mapping]
composition of three arrows:

    Module --Lower--> IRProgram --Renderer[T]--> panproto.Schema --emit_pretty--> bytes

[`Lower`][quivers.transpile.lower.Lower] is target-independent; it
walks the parsed module, resolves morphism / let references, builds
an [`IRProgram`][quivers.transpile.ir.IRProgram] whose nodes carry
the structural intent (sample, observe, marginalize, ...) plus the
support / plate / argument shape derived from
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] +
[`torch.distributions.Distribution.arg_constraints`][torch.distributions.Distribution.arg_constraints].

Each target `T` has its own
[`Renderer[T]`][quivers.transpile.renderers._base.RendererBase]
subclass in `quivers.transpile.renderers.<target>`; the renderer
consumes the IR and emits a target-specific
[`panproto.Schema`][panproto.Schema]. The renderer's idiom (Stan's
`block` structure, NumPyro's `plate` contexts, BUGS's row-loop
relations) is the only place target-specific vocabulary lives.

Family-level facts (per-target distribution name, argument aliases)
live in `FAMILY_META`; no renderer hardcodes family-name dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quivers.transpile._api import (
    CHURCH_LIKE,
    PYTHON_DEEP,
    STAN_LIKE,
    Backend,
    UnsupportedConstruct,
    unsupported_for,
)
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile._pipeline import (
    EmitPretty,
    SchemaTransform,
    parser_registry,
    realize,
    target_protocol,
)
from quivers.transpile.lower import Lower
from quivers.transpile.renderers._base import RendererBase
from quivers.transpile.renderers.bugs import BUGSRenderer
from quivers.transpile.renderers.church import ChurchRenderer
from quivers.transpile.renderers.edward2 import Edward2Renderer
from quivers.transpile.renderers.gen import GenRenderer
from quivers.transpile.renderers.jags import JAGSRenderer
from quivers.transpile.renderers.numpyro import NumPyroRenderer
from quivers.transpile.renderers.pymc import PyMCRenderer
from quivers.transpile.renderers.pyro import PyroRenderer
from quivers.transpile.renderers.stan import StanRenderer
from quivers.transpile.renderers.turing import TuringRenderer
from quivers.transpile.renderers.webppl import WebPPLRenderer

if TYPE_CHECKING:
    from quivers.dsl.ast_nodes import Module


_RENDERERS: dict[
    str, tuple[type[RendererBase], str, frozenset[str]]
] = {
    "stan":    (StanRenderer,    "stan",       STAN_LIKE),
    "numpyro": (NumPyroRenderer, "python",     PYTHON_DEEP),
    "pyro":    (PyroRenderer,    "python",     PYTHON_DEEP),
    "pymc":    (PyMCRenderer,    "python",     STAN_LIKE),
    "edward2": (Edward2Renderer, "python",     STAN_LIKE),
    "turing":  (TuringRenderer,  "julia",      STAN_LIKE),
    "gen":     (GenRenderer,     "julia",      STAN_LIKE),
    "church":  (ChurchRenderer,  "scheme",     CHURCH_LIKE),
    "webppl":  (WebPPLRenderer,  "javascript", CHURCH_LIKE),
    "bugs":    (BUGSRenderer,    "bugs",       STAN_LIKE),
    "jags":    (JAGSRenderer,    "jags",       STAN_LIKE),
}


def transpile(module: Module, *, target: str) -> bytes:
    """Transpile a QVR module to the named ``target`` backend.

    Parameters
    ----------
    module
        The parsed [`Module`][quivers.dsl.ast_nodes.Module] AST.
    target
        A registered backend key. See
        [`available_targets`][quivers.transpile.available_targets].

    Returns
    -------
    bytes
        The transpiled source bytes.

    Raises
    ------
    UnsupportedConstruct
        If ``target`` is not registered, or if the module contains
        constructs the chosen renderer cannot lower (e.g. a
        non-finite-support marginalize on Stan).
    """
    if target not in _RENDERERS:
        raise UnsupportedConstruct(
            target,
            [
                f"unknown target: {target!r}; available: "
                f"{', '.join(sorted(_RENDERERS))}"
            ],
        )
    renderer_cls, grammar, support_tier = _RENDERERS[target]
    unsupported_for(f"qvr-{target}", module, allow=support_tier)
    expanded = expand_composite_lets(module, target=target)
    ir = Lower().forward(expanded)
    schema = renderer_cls().render(ir)
    return bytes(parser_registry().emit_pretty(grammar, schema))


def available_targets() -> list[str]:
    """List every registered backend, sorted."""
    return sorted(_RENDERERS)


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
