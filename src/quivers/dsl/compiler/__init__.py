"""quivers.dsl.compiler: AST -> Program compiler.

Re-exports the public surface from the package's submodules.
"""
from quivers.dsl.compiler.core import Compiler
from quivers.dsl.compiler._prelude import (
    CompileError,
    _FAMILY_REGISTRY,
    _FAMILY_EVENT_RANK,
    _QUANTALE_REGISTRY,
    _available_axes_for,
    _family_event_rank,
    _get_family_registry,
    _register_extra_quantales,
    _shape_size,
    _type_factor_names,
    _validate_axis_spec,
)

__all__ = [
    "Compiler",
    "CompileError",
    "_FAMILY_REGISTRY",
    "_FAMILY_EVENT_RANK",
    "_QUANTALE_REGISTRY",
    "_available_axes_for",
    "_family_event_rank",
    "_get_family_registry",
    "_register_extra_quantales",
    "_shape_size",
    "_type_factor_names",
    "_validate_axis_spec",
]
