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
from quivers.qiec.module import program_computations
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
    _refuse_deduction_calls(qiec_module, target)
    return qiec_module


def _refuse_deduction_calls(qiec_module: QiecModule, target: str) -> None:
    """Refuse a program that calls a deduction.

    A deduction enumerates its derivations through a search handler that
    no target runtime carries, so a program calling one is refused at
    the boundary, before any target-specific lowering, under the
    capability tag the renderers use.

    Parameters
    ----------
    qiec_module : QiecModule
        The checked module.
    target : str
        The transpile target, for the diagnostic.

    Raises
    ------
    UnsupportedConstruct
        If a program's computation reaches a deduction's computation.
    """
    by_id = {computation.id: computation for computation in qiec_module.computations}
    kinds = [
        f"qiec:capability:search:{by_id[identity].name}"
        for identity in sorted(
            program_computations(qiec_module), key=lambda item: item.digest
        )
        if identity in by_id
        and by_id[identity].origin.structural_path[:1] == ("deductions",)
        and by_id[identity].name.endswith("__run")
    ]
    if kinds:
        raise UnsupportedConstruct(f"qvr-{target}", kinds)


__all__ = ["check_qiec_transpile_boundary"]
