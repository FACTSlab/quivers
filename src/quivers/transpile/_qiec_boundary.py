"""Shared, semantics-preserving QIEC validation and capability boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quivers.dsl.qiec_lowering import has_qiec_surface, lower_qvr_to_qiec

if TYPE_CHECKING:
    from quivers.dsl.ast_nodes import Module
    from quivers.qiec import QiecModule


def check_qiec_transpile_boundary(module: Module, *, target: str) -> QiecModule | None:
    """Check QIEC declarations and return their complete stable module.

    Modules without QIEC syntax take no QIEC import or checking path. When the
    surface is present, the public extension lowerer constructs a complete
    :class:`~quivers.qiec.QiecModule` and ``validate_module`` independently
    checks its nominal declarations, static scopes, effect rows, and every
    computation type. Capability analysis is deliberately separate: lowering
    always retains the complete typed module, and a selected renderer uses
    :func:`quivers.transpile.qiec_ir.analyze_qiec_capabilities` to determine
    whether it can preserve each feature.
    """
    if not has_qiec_surface(module):
        return None

    from quivers.qiec import validate_module

    qiec_module: QiecModule = lower_qvr_to_qiec(module)
    validate_module(qiec_module)

    return qiec_module


__all__ = ["check_qiec_transpile_boundary"]
