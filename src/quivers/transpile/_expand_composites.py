"""The transpilers' view of composite let expansion.

Expansion lives with the source language in
:mod:`quivers.dsl.composite_lets`; this module turns its failures into
the transpile API's
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct].
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import Module
from quivers.dsl.composite_lets import expand_composite_lets as _expand_composite_lets
from quivers.dsl.step_resolution import StepResolutionError
from quivers.transpile._api import UnsupportedConstruct


def expand_composite_lets(module: Module, *, target: str) -> Module:
    """Expand composite let aliases, reporting failure as unsupported.

    Parameters
    ----------
    module : Module
        The parsed module.
    target : str
        The transpile target, for the diagnostic.

    Returns
    -------
    Module
        The module with every composite draw unfolded into steps.

    Raises
    ------
    UnsupportedConstruct
        If a composite cannot be expanded.
    """
    try:
        return _expand_composite_lets(module, target=target)
    except StepResolutionError as error:
        raise UnsupportedConstruct(error.target, error.kinds) from error


__all__ = ["expand_composite_lets"]
