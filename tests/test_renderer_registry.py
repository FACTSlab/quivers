"""Tests for [`quivers.transpile.renderer_registry`][quivers.transpile.renderer_registry]:
the per-(backend, IR-node) emit registry that lets third-party
backends and third-party IR nodes plug into the transpile pipeline.
"""

from __future__ import annotations

from typing import Literal

import pytest

from quivers.transpile.ir import (
    CSReal,
    IRArgNumber,
    IRArgRef,
    IRDataInput,
    IRNode,
    IRReturn,
    IRSample,
    Plate,
)
from quivers.transpile.renderer_registry import (
    BackendRenderer,
    RendererDuplicateError,
    RendererLookupError,
    emit_hook,
    get_emit,
    registered_backends,
    registered_node_types,
    unregister,
)


class _IRCustomMove(IRNode):
    """User-defined IR node used by the custom-operator dispatch test."""

    name: str
    payload: str
    kind: Literal["ir_custom_move"] = "ir_custom_move"


@pytest.fixture(autouse=True)
def _clear_test_backend():
    """Each test in this file registers under an isolated backend
    name so a failing test does not leave the registry dirty for
    the next one.
    """
    yield
    for backend in list(registered_backends()):
        if backend.startswith("_test_"):
            for cls_name in registered_node_types(backend):
                for op_type in (
                    IRSample,
                    IRDataInput,
                    IRReturn,
                    IRArgNumber,
                    IRArgRef,
                ):
                    if op_type.__name__ == cls_name:
                        unregister(backend, op_type)


def test_emit_hook_registers_and_dispatches() -> None:
    @emit_hook("_test_stan", IRSample)
    def emit_stan_sample(renderer, ctx, node):
        return f"stan-sample:{node.name}"

    fn = get_emit("_test_stan", IRSample)
    node = IRSample(
        name="theta",
        family="Normal",
        args=(IRArgNumber(value=0.0), IRArgNumber(value=1.0)),
        arg_names=("loc", "scale"),
        constraint=CSReal(),
        plate=Plate(event_dims=(), batch_dims=()),
    )
    assert fn(None, None, node) == "stan-sample:theta"


def test_duplicate_registration_rejected() -> None:
    @emit_hook("_test_dupe", IRSample)
    def first(renderer, ctx, node):
        return "first"

    with pytest.raises(RendererDuplicateError, match="already has an emit"):

        @emit_hook("_test_dupe", IRSample)
        def second(renderer, ctx, node):
            return "second"


def test_duplicate_registration_replace_true_overrides() -> None:
    @emit_hook("_test_replace", IRSample)
    def first(renderer, ctx, node):
        return "first"

    @emit_hook("_test_replace", IRSample, replace=True)
    def second(renderer, ctx, node):
        return "second"

    node = IRSample(
        name="x",
        family="Normal",
        args=(),
        arg_names=(),
        constraint=CSReal(),
        plate=Plate(event_dims=(), batch_dims=()),
    )
    assert get_emit("_test_replace", IRSample)(None, None, node) == "second"


def test_missing_registration_raises_with_diagnostic() -> None:
    with pytest.raises(RendererLookupError, match="no emits registered"):
        get_emit("_test_nonexistent_backend", IRSample)


def test_missing_op_lists_registered_ops_in_error() -> None:
    @emit_hook("_test_partial", IRSample)
    def emit(renderer, ctx, node):
        return None

    with pytest.raises(RendererLookupError, match="IRSample"):
        get_emit("_test_partial", IRReturn)


def test_backend_renderer_dispatches_through_registry() -> None:
    @emit_hook("_test_dispatcher", IRSample)
    def emit_sample(renderer, ctx, node):
        assert renderer.backend_name == "_test_dispatcher"
        return {"kind": "sample", "name": node.name}

    class MyRenderer(BackendRenderer):
        backend_name = "_test_dispatcher"

    node = IRSample(
        name="alpha",
        family="Normal",
        args=(),
        arg_names=(),
        constraint=CSReal(),
        plate=Plate(event_dims=(), batch_dims=()),
    )
    result = MyRenderer().emit_node(None, node)
    assert result == {"kind": "sample", "name": "alpha"}


def test_backend_renderer_dispatches_args_through_registry() -> None:
    @emit_hook("_test_argdispatch", IRArgNumber)
    def emit_number(renderer, ctx, arg):
        return f"num:{arg.value}"

    @emit_hook("_test_argdispatch", IRArgRef)
    def emit_ref(renderer, ctx, arg):
        return f"ref:{arg.name}"

    class ArgRenderer(BackendRenderer):
        backend_name = "_test_argdispatch"

    r = ArgRenderer()
    assert r.emit_arg(None, IRArgNumber(value=1.5)) == "num:1.5"
    assert r.emit_arg(None, IRArgRef(name="theta", indices=())) == "ref:theta"


def test_registered_backends_lists_names() -> None:
    @emit_hook("_test_listable", IRSample)
    def emit(renderer, ctx, node):
        return None

    assert "_test_listable" in registered_backends()


def test_registered_node_types_lists_class_names() -> None:
    @emit_hook("_test_listtypes", IRSample)
    def emit_sample(renderer, ctx, node):
        return None

    @emit_hook("_test_listtypes", IRArgNumber)
    def emit_number(renderer, ctx, node):
        return None

    types = registered_node_types("_test_listtypes")
    assert "IRSample" in types
    assert "IRArgNumber" in types


def test_unregister_removes_entry() -> None:
    @emit_hook("_test_unreg", IRSample)
    def emit(renderer, ctx, node):
        return None

    unregister("_test_unreg", IRSample)
    with pytest.raises(RendererLookupError):
        get_emit("_test_unreg", IRSample)


def test_custom_ir_node_plugs_in_end_to_end() -> None:
    """A user-defined `IRNode` subclass plus its emit function
    becomes callable through the registry without editing core.
    """

    @emit_hook("_test_customnode", _IRCustomMove)
    def emit_custom(renderer, ctx, node):
        return f"custom-{node.payload}"

    class Renderer(BackendRenderer):
        backend_name = "_test_customnode"

    node = _IRCustomMove(name="x", payload="hello")
    assert Renderer().emit_node(None, node) == "custom-hello"
    unregister("_test_customnode", _IRCustomMove)
