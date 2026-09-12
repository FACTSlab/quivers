"""Shared QIEC gate in front of every probabilistic transpiler.

The eleven registered targets consume the probabilistic IR. QIEC static
declarations may accompany an ordinary ``program`` after they have passed the
QIEC checker, but a QIEC computation cannot enter that IR without losing
``perform``, ``handle``, indexed-case evidence, or resumption semantics. This
module makes that boundary one target-independent decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quivers.dsl.qiec_lowering import has_qiec_surface, lower_qvr_to_qiec
from quivers.transpile._api import UnsupportedConstruct, cast_kind

if TYPE_CHECKING:
    from quivers.dsl.ast_nodes import Module
    from quivers.qiec import QiecModule


def check_qiec_transpile_boundary(module: Module, *, target: str) -> QiecModule | None:
    """Check QIEC declarations and refuse non-representable computations.

    Modules without QIEC syntax take no QIEC import or checking path. When the
    surface is present, the public extension lowerer constructs a complete
    :class:`~quivers.qiec.QiecModule` and ``validate_module`` independently
    checks its nominal declarations, static scopes, effect rows, and every
    computation type. Only then may declaration-only QIEC metadata accompany
    the ordinary probabilistic program consumed by the structural IR.
    """
    if not has_qiec_surface(module):
        return None

    from quivers.qiec import validate_module

    qiec_module: QiecModule = lower_qvr_to_qiec(module)
    validate_module(qiec_module)

    if qiec_module.computations:
        names = sorted(computation.name for computation in qiec_module.computations)
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [f"qiec:computation-body:{name}" for name in names],
            module_has_program=any(
                cast_kind(statement) == "program_decl"
                for statement in module.statements
            ),
        )
    return qiec_module


__all__ = ["check_qiec_transpile_boundary"]
