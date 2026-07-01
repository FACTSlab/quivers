"""Per-(backend, IR-node) emit registry.

Third-party backends and third-party IR nodes plug into the
transpile pipeline through this registry without editing the
in-tree renderers. Every backend registers an emit function per
[`IRNode`][quivers.transpile.ir.IRNode] variant it supports; the
[`BackendRenderer`][quivers.transpile.renderer_registry.BackendRenderer]
base class dispatches through the registry at render time.

Two entry points:

* [`emit_hook(backend_name, node_type)`][quivers.transpile.renderer_registry.emit_hook]
  is a decorator that binds an emit function at import time.
* [`BackendRenderer.emit_node(ctx, node)`][quivers.transpile.renderer_registry.BackendRenderer.emit_node]
  looks up the registered function for the concrete `IRNode`
  subclass and calls it, raising a targeted diagnostic when no
  handler is registered.

An [`IRArg`][quivers.transpile.ir.IRArg] subclass plugs in the
same way: register the argument class's emit through `emit_hook`
under the backend name and it becomes visible to every renderer
that walks the argument tree through the registry.

The registry composes with
[`RendererBase`][quivers.transpile.renderers._base.RendererBase]:
backends can subclass `RendererBase` and hard-code their emit
tables, or drive dispatch through this registry. Both paths are
supported.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Any

from quivers.transpile.ir import IRArg, IRNode


EmitFn = Callable[..., Any]


class RendererLookupError(LookupError):
    """No emit function is registered for the requested
    `(backend_name, node_or_arg_type)` combination.
    """


class RendererDuplicateError(ValueError):
    """A second emit function is being registered for a combination
    that already has one, without `replace=True`. Silently shadowing
    a registered emit is a subtle bug source; the registry rejects
    the second registration unless the caller opts in.
    """


_REGISTRY: dict[str, dict[type, EmitFn]] = {}


def emit_hook(
    backend_name: str,
    node_type: type,
    *,
    replace: bool = False,
) -> Callable[[EmitFn], EmitFn]:
    """Decorator that registers an emit function for a
    `(backend_name, node_type)` combination.

    ``node_type`` is an [`IRNode`][quivers.transpile.ir.IRNode] or
    [`IRArg`][quivers.transpile.ir.IRArg] subclass. Registration
    happens at import time of the module that declares the emit;
    downstream libraries add support for a new backend or a new
    node type by importing their module once.

    ``replace=True`` allows re-registering over an existing entry.
    Left unset, a second registration for the same key raises
    :class:`RendererDuplicateError` so a user does not silently
    shadow a built-in emit.
    """

    def decorator(fn: EmitFn) -> EmitFn:
        backend_table = _REGISTRY.setdefault(backend_name, {})
        if node_type in backend_table and not replace:
            raise RendererDuplicateError(
                f"emit_hook: backend {backend_name!r} already has an "
                f"emit registered for {node_type.__name__!r}; pass "
                "`replace=True` to override."
            )
        backend_table[node_type] = fn
        return fn

    return decorator


def get_emit(backend_name: str, node_type: type) -> EmitFn:
    """Look up the registered emit for a `(backend_name, node_type)`
    combination. Raises :class:`RendererLookupError` with the list
    of registered types for the backend when no emit is present.
    """
    backend_table = _REGISTRY.get(backend_name)
    if backend_table is None:
        raise RendererLookupError(
            f"get_emit: no emits registered for backend "
            f"{backend_name!r}. Registered backends: "
            f"{sorted(_REGISTRY)}"
        )
    if node_type not in backend_table:
        registered = sorted(t.__name__ for t in backend_table)
        raise RendererLookupError(
            f"get_emit: backend {backend_name!r} has no emit for "
            f"{node_type.__name__!r}. Registered: {registered}"
        )
    return backend_table[node_type]


def registered_backends() -> list[str]:
    """List every backend that has at least one registered emit."""
    return sorted(_REGISTRY)


def registered_node_types(backend_name: str) -> list[str]:
    """List the class names of every node type registered for
    ``backend_name``.
    """
    table = _REGISTRY.get(backend_name, {})
    return sorted(t.__name__ for t in table)


def unregister(backend_name: str, node_type: type) -> None:
    """Remove a registration. Used by tests to reset state; not
    typically called in production code.
    """
    if backend_name in _REGISTRY and node_type in _REGISTRY[backend_name]:
        del _REGISTRY[backend_name][node_type]
        if not _REGISTRY[backend_name]:
            del _REGISTRY[backend_name]


class BackendRenderer(ABC):
    """Abstract renderer that dispatches emit through the registry.

    Concrete renderers set the ``backend_name`` class attribute and
    call :meth:`emit_node` to route each
    [`IRNode`][quivers.transpile.ir.IRNode] through the registered
    emit function. Any [`IRArg`][quivers.transpile.ir.IRArg] a
    renderer needs to walk is dispatched through :meth:`emit_arg`.

    Existing renderers under [`quivers.transpile.backends`][quivers.transpile.backends]
    subclass [`RendererBase`][quivers.transpile.backends] instead
    and hard-code their emit tables; both paths coexist. Third-
    party renderers with a small backend surface pick this base
    for the extensibility; large in-tree backends stay on
    `RendererBase` until a migration is warranted.
    """

    backend_name: str

    def emit_node(self, ctx, node: IRNode):
        """Dispatch a top-level IR node through the registry."""
        emit_fn = get_emit(self.backend_name, type(node))
        return emit_fn(self, ctx, node)

    def emit_arg(self, ctx, arg: IRArg):
        """Dispatch an IR argument through the registry."""
        emit_fn = get_emit(self.backend_name, type(arg))
        return emit_fn(self, ctx, arg)


__all__ = [
    "BackendRenderer",
    "EmitFn",
    "RendererDuplicateError",
    "RendererLookupError",
    "emit_hook",
    "get_emit",
    "registered_backends",
    "registered_node_types",
    "unregister",
]
