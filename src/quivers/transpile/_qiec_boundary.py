"""Shared, semantics-preserving QIEC validation and capability boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import replace

from quivers.dsl.program_elaboration import GAP_CODE
from quivers.dsl.qiec_lowering import (
    QiecDiagnosticError,
    has_qiec_surface,
    lower_qvr_to_qiec,
)
from quivers.qiec import QiecModule, validate_module
from quivers.transpile._api import UnsupportedConstruct

if TYPE_CHECKING:
    from quivers.dsl.ast_nodes import Module


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

    Parameters
    ----------
    module : Module
        The parsed module.
    target : str
        The transpile target, for diagnostics.

    Returns
    -------
    QiecModule | None
        The checked module, or ``None`` when the source has nothing to
        elaborate.

    Raises
    ------
    UnsupportedConstruct
        If a program uses a form the elaboration does not admit; the
        construct is reported the way a renderer reports one it cannot
        emit.
    QiecDiagnosticError
        If a QIEC declaration is rejected.
    """
    if not has_qiec_surface(module):
        return None
    try:
        qiec_module: QiecModule = lower_qvr_to_qiec(module)
    except QiecDiagnosticError as error:
        if error.code == GAP_CODE:
            # A program using a chart or a network morphism has no
            # elaboration yet. The module's other declarations still
            # lower; the program itself reaches the renderer through its
            # plan alone, and the gap is recorded on the plan.
            qiec_module = lower_qvr_to_qiec(module, elaborate_programs=False)
            validate_module(qiec_module)
            return replace(qiec_module, gap=error.message)
        if error.code != "qiec-program" and error.program is None:
            raise
        raise UnsupportedConstruct(f"qvr-{target}", [error.message]) from error
    validate_module(qiec_module)
    return qiec_module


__all__ = ["check_qiec_transpile_boundary"]
