"""JAGS backend: QVR Module → JAGS source under the `jags` tree-sitter
grammar.

JAGS extends BUGS; the surface relevant to QVR's probabilistic core is
identical. We reuse the BUGS family map and walker, only swapping the
``qvr-jags`` registration and the target grammar name.
"""

from __future__ import annotations

import didactic.api as dx
import panproto

from quivers.dsl.ast_nodes import Module
from quivers.transpile._api import STAN_LIKE, unsupported_for
from quivers.transpile._pipeline import SchemaTransform, realize
from quivers.transpile.backends.bugs import _build


class _JagsWalker(SchemaTransform):
    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        return _build(module, "jags")


@dx.codegen.emitter("qvr-jags")
class JagsEmitter:
    file_extension: str = "jags"
    grammar: str = "jags"
    support: frozenset[str] = STAN_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            f"qvr-jags emits instances, not classes; got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-jags", module, allow=STAN_LIKE)
        return realize(module, grammar="jags", transform=_JagsWalker())


__all__ = ["JagsEmitter"]
