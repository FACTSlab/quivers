"""Structural target lowering for executable QIEC computations.

The dynamic targets share one small free-computation ABI, expressed in each
host language and grafted into the renderer's Panproto schema.  The three
static graphical languages accept only the analyzer-proven, effect-free scalar
fragment and emit named pure definitions.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Callable

import panproto

from quivers.qiec.identifiers import SiteId
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import parser_registry
from quivers.transpile.ir import IRProgram
from quivers.transpile.qiec_ir import (
    IRQiecAttachmentRef,
    IRQiecBind,
    IRQiecBoolLiteral,
    IRQiecBytesLiteral,
    IRQiecCall,
    IRQiecCase,
    IRQiecComputation,
    IRQiecConstructorValue,
    IRQiecEvidenceValue,
    IRQiecFloatLiteral,
    IRQiecHandle,
    IRQiecIntLiteral,
    IRQiecLiteral,
    IRQiecLiteralValue,
    IRQiecHandlerDef,
    IRQiecIf,
    IRQiecModule,
    IRQiecNamedComputation,
    IRQiecNewInstance,
    IRQiecNullLiteral,
    IRQiecPerform,
    IRQiecPrimitiveApplication,
    IRQiecProjection,
    IRQiecResume,
    IRQiecReturn,
    IRQiecStringLiteral,
    IRQiecTransportValue,
    IRQiecTupleLiteral,
    IRQiecTupleValue,
    IRQiecTypeApplication,
    IRQiecValue,
    IRQiecVar,
    analyze_qiec_capabilities,
)


_HERE = pathlib.Path(__file__).resolve().parent.parent
_RUNTIMES = {
    "python": _HERE / "runtime_qiec.py",
    "julia": _HERE / "runtime_qiec.jl",
    "javascript": _HERE / "runtime_qiec.js",
    "scheme": _HERE / "runtime_qiec.scm",
}
_DYNAMIC_TARGETS = {
    "pyro": "python",
    "numpyro": "python",
    "pymc": "python",
    "edward2": "python",
    "turing": "julia",
    "gen": "julia",
    "webppl": "javascript",
    "church": "scheme",
}
_ROOT_KINDS = {
    "python": "module",
    "julia": "source_file",
    "javascript": "program",
    "scheme": "program",
}


def qiec_target_name(target: str) -> str:
    """Normalize renderer-internal target labels to public registry keys."""
    return target.removeprefix("qvr-")


def has_qiec_computations(ir: IRProgram) -> bool:
    return ir.qiec is not None and bool(ir.qiec.computations)


def graft_qiec_dynamic(
    sb: panproto.SchemaBuilder,
    ir: IRProgram,
    *,
    target: str,
    root: str,
) -> None:
    """Append the shared runtime and named functions to a dynamic target."""
    module = ir.qiec
    if module is None or not module.computations:
        return
    public_target = qiec_target_name(target)
    diagnostics = analyze_qiec_capabilities(module, public_target)
    if diagnostics:
        raise UnsupportedConstruct(
            f"qvr-{public_target}", [diagnostic.kind for diagnostic in diagnostics]
        )
    grammar = _DYNAMIC_TARGETS[public_target]
    runtime = _RUNTIMES[grammar].read_text()
    definitions = _dynamic_definitions(module, grammar)
    _graft_parsed_children(
        sb,
        grammar=grammar,
        source=runtime + "\n" + definitions,
        source_root_kind=_ROOT_KINDS[grammar],
        destination=root,
        prefix=f"qiec_{public_target}",
    )


def graft_qiec_static(
    sb: panproto.SchemaBuilder,
    ir: IRProgram,
    *,
    target: str,
    destination: str,
) -> None:
    """Append analyzer-proven pure scalar definitions to a static target."""
    module = ir.qiec
    if module is None:
        return
    public_target = qiec_target_name(target)
    diagnostics = analyze_qiec_capabilities(module, public_target)
    if diagnostics:
        raise UnsupportedConstruct(
            f"qvr-{public_target}", [diagnostic.kind for diagnostic in diagnostics]
        )
    if not module.computations:
        if public_target not in {"bugs", "jags"}:
            return
        source, source_root_kind = "model {\nqiec_declarations <- 0\n}\n", "model_block"
    else:
        source, source_root_kind = _static_definitions(module, public_target)
    grammar = "bugs" if public_target == "bugs" else public_target
    _graft_parsed_children(
        sb,
        grammar=grammar,
        source=source,
        source_root_kind=source_root_kind,
        destination=destination,
        prefix=f"qiec_{public_target}",
    )


def _graft_parsed_children(
    sb: panproto.SchemaBuilder,
    *,
    grammar: str,
    source: str,
    source_root_kind: str,
    destination: str,
    prefix: str,
) -> None:
    """Parse source and graft one root's children into an existing schema."""
    schema = parser_registry().parse_with_protocol(
        grammar, source.encode(), f"<{prefix}>"
    )
    roots = [vertex.id for vertex in schema.vertices if vertex.kind == source_root_kind]
    if len(roots) != 1:
        raise RuntimeError(
            f"QIEC {grammar} fragment has {len(roots)} {source_root_kind!r} roots"
        )
    source_root = roots[0]
    top_level = [edge.tgt for edge in schema.edges if edge.src == source_root]
    top_level.sort(
        key=lambda vertex: next(
            (
                int(constraint.value)
                for constraint in schema.constraints_for(vertex)
                if constraint.sort == "start-byte"
            ),
            0,
        )
    )
    subtree = _reachable(schema, tuple(top_level))
    id_map = {old: f"{prefix}_{position}" for position, old in enumerate(subtree)}
    kinds = {vertex.id: vertex.kind for vertex in schema.vertices}
    for old, new in id_map.items():
        sb.vertex(new, kinds[old])
        for constraint in schema.constraints_for(old):
            sb.constraint(new, constraint.sort, constraint.value)
    for edge in schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            sb.edge(id_map[edge.src], id_map[edge.tgt], edge.kind)
    for child in top_level:
        sb.edge(destination, id_map[child], "child_of")


def _reachable(schema: panproto.Schema, roots: tuple[str, ...]) -> tuple[str, ...]:
    seen = set(roots)
    ordered = list(roots)
    cursor = 0
    while cursor < len(ordered):
        source = ordered[cursor]
        cursor += 1
        for edge in schema.edges:
            if edge.src == source and edge.tgt not in seen:
                seen.add(edge.tgt)
                ordered.append(edge.tgt)
    return tuple(ordered)


def _safe_name(name: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not candidate or candidate[0].isdigit():
        candidate = "_" + candidate
    return candidate


def _local_name(name: str) -> str:
    """Keep authored locals disjoint from keywords and the generated ABI."""
    return "qv_" + _safe_name(name)


def _function_name(computation: IRQiecNamedComputation) -> str:
    return "qiec_" + _safe_name(computation.name)


def _body_name(computation: IRQiecNamedComputation) -> str:
    """The generated function that builds a computation's body unrun.

    A call needs the callee's computation as data the caller's handlers
    can interpret, not its final value, so every named computation is
    generated twice: this builder, and the public entry that runs it.

    Parameters
    ----------
    computation
        The named computation.

    Returns
    -------
    str
        The builder's host-language name.
    """
    return "qiec_body_" + _safe_name(computation.name)


def _clause_name(handler: IRQiecHandlerDef, operation: str) -> str:
    """The generated function holding one authored clause body.

    Parameters
    ----------
    handler
        The authored handler.
    operation
        The clause's operation, by display name.

    Returns
    -------
    str
        The clause function's host-language name.
    """
    return "qiec_clause_" + _safe_name(handler.name) + "_" + _safe_name(operation)


def _return_clause_name(handler: IRQiecHandlerDef) -> str:
    """The generated function holding an authored return clause body.

    Parameters
    ----------
    handler
        The authored handler.

    Returns
    -------
    str
        The return clause function's host-language name.
    """
    return "qiec_return_" + _safe_name(handler.name)


def _computation_by_id(module: IRQiecModule, callee: str) -> IRQiecNamedComputation:
    """Find a named computation by stable identity.

    Parameters
    ----------
    module
        The module the call is being rendered from.
    callee
        The callee identity, as :attr:`IRQiecId.text`.

    Returns
    -------
    IRQiecNamedComputation
        The computation a call refers to.
    """
    return next(item for item in module.computations if item.id.text == callee)


def _operation_name(module: IRQiecModule, operation: str) -> str:
    """The display name of an operation, by stable identity.

    Parameters
    ----------
    module
        The module declaring the operation's interface.
    operation
        The operation identity, as :attr:`IRQiecId.text`.

    Returns
    -------
    str
        The declared name, or the identity text when no interface in the
        module declares it.
    """
    for effect in module.effects:
        for item in effect.operations:
            if item.id.text == operation:
                return item.name
    return operation


def _ir_data(value: object) -> object:
    """Project typed QIEC IR to lossless host-runtime metadata.

    Didactic's ``model_dump`` is JSON safe, but walking the model itself lets
    us retain stable identifiers in their canonical ``qiec:*`` spelling.
    """
    if hasattr(value, "text") and value.__class__.__name__ == "IRQiecId":
        return value.text  # type: ignore[attr-defined]
    fields = getattr(type(value), "__field_specs__", None)
    if fields is not None:
        return {name: _ir_data(getattr(value, name)) for name in fields}
    if isinstance(value, dict):
        return {_ir_data(key): _ir_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ir_data(item) for item in value]
    return value


def _dynamic_definitions(module: IRQiecModule, grammar: str) -> str:
    generators: dict[str, Callable[[IRQiecNamedComputation, IRQiecModule], str]] = {
        "python": _python_definition,
        "julia": _julia_definition,
        "javascript": _javascript_definition,
        "scheme": _scheme_definition,
    }
    handler_generators: dict[str, Callable[[IRQiecHandlerDef, IRQiecModule], str]] = {
        "python": _python_authored_handler,
        "julia": _julia_authored_handler,
        "javascript": _javascript_authored_handler,
        "scheme": _scheme_authored_handler,
    }
    # Clause bodies are generated before the computations that install
    # them, so a handler is registered by the time any entry point runs.
    authored = [
        handler_generators[grammar](handler, module)
        for handler in module.handlers
        if handler.implementation == "authored"
    ]
    return "\n".join(
        (
            *authored,
            *(generators[grammar](item, module) for item in module.computations),
        )
    )


def _handler_grades(module: IRQiecModule, handler_id: str) -> dict[str, str]:
    handler = _handler(module, handler_id)
    return {clause.operation.text: clause.grade for clause in handler.clauses}


def _handler(module: IRQiecModule, handler_id: str):
    return next(item for item in module.handlers if item.id.text == handler_id)


def _address(request: object) -> tuple[object, ...]:
    origin = request.origin.origin  # type: ignore[attr-defined]
    static = SiteId.derive(
        origin.source_protocol, origin.module, origin.structural_path, origin.role
    ).to_data()
    dynamic = tuple(
        (frame.scope, frame.key)
        for frame in request.origin.dynamic_path  # type: ignore[attr-defined]
    )
    resumptions = tuple(request.origin.resumption_path)  # type: ignore[attr-defined]
    return (static, dynamic, resumptions)


def _runtime_request(node: IRQiecPerform) -> dict[str, object]:
    request = node.request
    return {
        "instance": request.instance.text,
        "effect": _ir_data(request.effect),
        "operation": request.operation.text,
        "static_arguments": _ir_data(request.static_arguments),
        "result_type": _ir_data(request.result_type),
        "origin": _ir_data(request.origin),
        "address": _address(request),
    }


def _free_runtime_capture(
    node: IRQiecComputation,
    bound: frozenset[str] = frozenset(),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return outer locals and attachments captured by ``node``."""
    locals_: set[str] = set()
    attachments: set[str] = set()

    def value(item: IRQiecValue, locally_bound: frozenset[str]) -> None:
        if isinstance(item, IRQiecVar):
            if item.local.name not in locally_bound:
                locals_.add(item.local.name)
        elif isinstance(item, IRQiecConstructorValue):
            for field in item.fields:
                value(field, locally_bound)
        elif isinstance(item, IRQiecAttachmentRef):
            attachments.add(item.attachment.text)
        elif isinstance(item, IRQiecTransportValue):
            value(item.value, locally_bound)
        elif isinstance(item, IRQiecPrimitiveApplication):
            for argument in item.arguments:
                value(argument, locally_bound)
        elif isinstance(item, IRQiecTupleValue):
            for component in item.items:
                value(component, locally_bound)
        elif isinstance(item, IRQiecProjection):
            value(item.value, locally_bound)

    def computation(item: IRQiecComputation, locally_bound: frozenset[str]) -> None:
        if isinstance(item, IRQiecReturn):
            value(item.value, locally_bound)
        elif isinstance(item, IRQiecBind):
            computation(item.first, locally_bound)
            computation(item.then, locally_bound | {item.binder.name})
        elif isinstance(item, IRQiecPerform):
            for argument in item.request.arguments:
                value(argument, locally_bound)
        elif isinstance(item, IRQiecHandle):
            computation(item.computation, locally_bound)
        elif isinstance(item, IRQiecCase):
            value(item.scrutinee, locally_bound)
            for branch in item.branches:
                computation(
                    branch.body,
                    locally_bound | {field.name for field in branch.fields},
                )
        elif isinstance(item, IRQiecIf):
            value(item.condition, locally_bound)
            computation(item.then, locally_bound)
            computation(item.otherwise, locally_bound)
        elif isinstance(item, IRQiecCall):
            for argument in item.arguments:
                value(argument, locally_bound)
        elif isinstance(item, IRQiecResume):
            value(item.value, locally_bound)
        elif isinstance(item, IRQiecNewInstance):
            computation(item.body, locally_bound)

    computation(node, bound)
    return tuple(sorted(locals_)), tuple(sorted(attachments))


def _branch_static(rendered: str, identifier: str) -> str:
    """Point a branch body's static environment at the branch's own.

    Only the bare identifier moves; runtime helper names that contain it,
    such as the static-environment constructor a nested call uses, are
    left alone.

    Parameters
    ----------
    rendered
        The rendered branch body.
    identifier
        The host-language spelling of the static environment name.

    Returns
    -------
    str
        The body with every bare occurrence renamed to the branch's
        environment.
    """
    pattern = r"(?<![\w-])" + re.escape(identifier) + r"(?![\w-])"
    replacement = identifier.replace("static", "branch-static", 1)
    if "_" in identifier:
        replacement = identifier.replace("static", "branch_static", 1)
    return re.sub(pattern, replacement, rendered)


def _case_metadata(node: IRQiecCase) -> dict[str, object]:
    return {
        "motive": _ir_data(node.motive),
        "branches": {
            branch.constructor.text: {
                "static_arguments": _ir_data(branch.static_arguments),
                "fields": _ir_data(branch.fields),
                "scope": branch.scope.text,
            }
            for branch in node.branches
        },
    }


_PYTHON_ABI = "qiec_static, qiec_attachments, qiec_handlers, qiec_operations"


def _python_definition(item: IRQiecNamedComputation, module: IRQiecModule) -> str:
    params = [_local_name(parameter.name) for parameter in item.parameters]
    abi = [
        "qiec_static_arguments=None",
        "qiec_attachments=None",
        "qiec_handlers=None",
        "qiec_operations=None",
    ]
    body = _python_computation(item.body, module)
    return (
        f"def {_body_name(item)}({', '.join((*params, _PYTHON_ABI))}):\n"
        f"    return {body}\n"
        f"def {_function_name(item)}({', '.join((*params, *abi))}):\n"
        f"    qiec_static = _qvr_qiec_static_environment({_ir_data(item.telescope)!r}, qiec_static_arguments)\n"
        "    qiec_attachments = {} if qiec_attachments is None else qiec_attachments\n"
        "    qiec_handlers = {} if qiec_handlers is None else qiec_handlers\n"
        "    qiec_operations = {} if qiec_operations is None else qiec_operations\n"
        f"    return _qvr_qiec_run(lambda: {_body_name(item)}({', '.join((*params, _PYTHON_ABI))}), qiec_operations)\n"
    )


def _python_authored_handler(handler: IRQiecHandlerDef, module: IRQiecModule) -> str:
    """Generate an authored handler's clause bodies and register them.

    Each clause becomes a function with the foreign-clause calling
    convention, so the runtime installs authored and foreign handlers
    alike; the difference is only where the body came from.

    Parameters
    ----------
    handler
        The authored handler declaration.
    module
        The module it belongs to, for resolving names.

    Returns
    -------
    str
        Host-language source defining the clause functions and registering
        the handler under its stable identity.
    """
    lines: list[str] = []
    operations: list[str] = []
    for clause in handler.clauses:
        name = _clause_name(handler, _operation_name(module, clause.operation.text))
        if clause.body is None:
            continue
        params = ", ".join(_local_name(local.name) for local in clause.parameters)
        unpack = f"    ({params},) = qiec_request['arguments']\n" if params else ""
        lines.append(
            f"def {name}(qiec_request, qiec_resume, qiec_context):\n"
            "    qiec_static = qiec_context['static']\n"
            "    qiec_attachments = qiec_context['attachments']\n"
            "    qiec_handlers = qiec_context['handlers']\n"
            "    qiec_operations = qiec_context['operations']\n"
            f"{unpack}"
            f"    return {_python_computation(clause.body, module)}\n"
        )
        operations.append(f"{clause.operation.text!r}: {{'invoke': {name}}}")
    return_entry = "None"
    if handler.return_clause is not None:
        name = _return_clause_name(handler)
        binder = _local_name(handler.return_clause.binder.name)
        lines.append(
            f"def {name}({binder}, qiec_context):\n"
            "    qiec_static = qiec_context['static']\n"
            "    qiec_attachments = qiec_context['attachments']\n"
            "    qiec_handlers = qiec_context['handlers']\n"
            "    qiec_operations = qiec_context['operations']\n"
            f"    return {_python_computation(handler.return_clause.body, module)}\n"
        )
        return_entry = name
    lines.append(
        f"_qvr_qiec_authored[{handler.id.text!r}] = {{"
        f"'operations': {{{', '.join(operations)}}}, "
        f"'return': {return_entry}, 'duplicable_context': True}}\n"
    )
    return "".join(lines)


def _python_computation(
    node: IRQiecComputation, module: IRQiecModule, tail: bool = True
) -> str:
    if isinstance(node, IRQiecReturn):
        return f"_qvr_qiec_pure({_python_value(node.value)})"
    if isinstance(node, IRQiecBind):
        binder = _local_name(node.binder.name)
        capture_names, attachment_ids = _free_runtime_capture(
            node.then, frozenset({node.binder.name})
        )
        captures = (
            "{"
            + ", ".join(f"{name!r}: {_local_name(name)}" for name in capture_names)
            + "}"
        )
        return (
            f"_qvr_qiec_bind({_python_computation(node.first, module, tail=False)}, "
            f"lambda {binder}: {_python_computation(node.then, module)}, "
            f"_qvr_qiec_capture({captures}, {attachment_ids!r}, qiec_attachments))"
        )
    if isinstance(node, IRQiecPerform):
        args = ", ".join(_python_value(value) for value in node.request.arguments)
        tuple_args = f"({args},)" if len(node.request.arguments) == 1 else f"({args})"
        request = _runtime_request(node)
        request["arguments"] = None
        rendered = repr(request).replace(
            "'arguments': None", f"'arguments': {tuple_args}"
        )
        return f"_qvr_qiec_effect({rendered}, qiec_static)"
    if isinstance(node, IRQiecHandle):
        handler = _handler(module, node.handler.text)
        return (
            f"_qvr_qiec_handle(lambda: {_python_computation(node.computation, module)}, "
            f"{node.instance.text!r}, {_ir_data(handler)!r}, "
            f"{_ir_data(node.static_arguments)!r}, qiec_static, qiec_handlers, "
            "qiec_attachments, qiec_operations)"
        )
    if isinstance(node, IRQiecCall):
        callee = _computation_by_id(module, node.callee.text)
        arguments = "".join(
            f"{_python_value(argument)}, " for argument in node.arguments
        )
        return (
            f"_qvr_qiec_enter_call({callee.name!r}, lambda: {_body_name(callee)}("
            f"{arguments}"
            f"_qvr_qiec_static_environment({_ir_data(callee.telescope)!r}, "
            f"_qvr_qiec_specialize({_ir_data(node.static_arguments)!r}, qiec_static)), "
            f"qiec_attachments, qiec_handlers, qiec_operations), {tail!r})"
        )
    if isinstance(node, IRQiecIf):
        return (
            f"_qvr_qiec_if({_python_value(node.condition)}, "
            f"lambda: {_python_computation(node.then, module, tail=tail)}, "
            f"lambda: {_python_computation(node.otherwise, module, tail=tail)})"
        )
    if isinstance(node, IRQiecResume):
        return f"_qvr_qiec_resume(qiec_resume, {_python_value(node.value)})"
    if isinstance(node, IRQiecNewInstance):
        return f"_qvr_qiec_instance(lambda: {_python_computation(node.body, module)})"
    if isinstance(node, IRQiecCase):
        branches = []
        for branch in node.branches:
            params = ", ".join(_local_name(field.name) for field in branch.fields)
            body = _branch_static(
                _python_computation(branch.body, module), "qiec_static"
            )
            separator = ", " if params else ""
            branches.append(
                f"{branch.constructor.text!r}: lambda qiec_branch_static{separator}{params}: {body}"
            )
        return (
            f"_qvr_qiec_case({_python_value(node.scrutinee)}, "
            f"{{{', '.join(branches)}}}, {_case_metadata(node)!r}, qiec_static)"
        )
    raise TypeError(f"unknown QIEC computation {node!r}")


def _python_value(node: IRQiecValue) -> str:
    if isinstance(node, IRQiecVar):
        return f"_qvr_qiec_value({_local_name(node.local.name)})"
    if isinstance(node, IRQiecLiteralValue):
        return _python_literal(node.value)
    if isinstance(node, IRQiecConstructorValue):
        fields = ", ".join(_python_value(field) for field in node.fields)
        fields = f"({fields},)" if len(node.fields) == 1 else f"({fields})"
        return (
            f"_qvr_qiec_constructor({node.constructor.text!r}, "
            f"{_ir_data(node.static_arguments)!r}, {fields}, "
            f"{_ir_data(node.result_type)!r}, qiec_static)"
        )
    if isinstance(node, IRQiecEvidenceValue):
        return f"_qvr_qiec_evidence({_ir_data(node.evidence)!r}, qiec_static)"
    if isinstance(node, IRQiecAttachmentRef):
        return (
            f"_qvr_qiec_attachment(qiec_attachments, {node.attachment.text!r}, "
            f"{_ir_data(node.type)!r}, qiec_static)"
        )
    if isinstance(node, IRQiecTransportValue):
        return (
            f"_qvr_qiec_transport({_ir_data(node.evidence)!r}, "
            f"{_python_value(node.value)}, {_ir_data(node.target_type)!r}, qiec_static)"
        )
    if isinstance(node, IRQiecPrimitiveApplication):
        arguments = ", ".join(_python_value(argument) for argument in node.arguments)
        return f"_qvr_qiec_primitive({node.name!r}, ({arguments},))"
    if isinstance(node, IRQiecTupleValue):
        items = ", ".join(_python_value(item) for item in node.items)
        return f"({items},)" if len(node.items) == 1 else f"({items})"
    if isinstance(node, IRQiecProjection):
        return f"_qvr_qiec_project({_python_value(node.value)}, {node.position})"
    raise TypeError(f"unknown QIEC value {node!r}")


def _python_literal(node: IRQiecLiteral) -> str:
    if isinstance(node, IRQiecNullLiteral):
        return "None"
    if isinstance(node, IRQiecBytesLiteral):
        return f"bytes.fromhex({node.value!r})"
    if isinstance(node, IRQiecTupleLiteral):
        items = ", ".join(_python_literal(item) for item in node.items)
        return f"({items},)" if len(node.items) == 1 else f"({items})"
    return repr(node.value)  # type: ignore[attr-defined]


_JULIA_ABI = "qiec_static, qiec_attachments, qiec_handlers, qiec_operations"


def _julia_definition(item: IRQiecNamedComputation, module: IRQiecModule) -> str:
    params = ", ".join(_local_name(parameter.name) for parameter in item.parameters)
    separator = "; " if params else "; "
    body_params = f"{params}, {_JULIA_ABI}" if params else _JULIA_ABI
    return (
        f"function {_body_name(item)}({body_params})\n"
        f"    return {_julia_computation(item.body, module)}\n"
        "end\n"
        f"function {_function_name(item)}({params}{separator}qiec_attachments=Dict(), "
        "qiec_handlers=Dict(), qiec_operations=Dict(), qiec_static_arguments=nothing)\n"
        f"    qiec_static = _qvr_qiec_static_environment({_julia_data(_ir_data(item.telescope))}, qiec_static_arguments)\n"
        f"    return _qvr_qiec_run(() -> {_body_name(item)}({body_params}), qiec_operations)\n"
        "end\n"
    )


def _julia_authored_handler(handler: IRQiecHandlerDef, module: IRQiecModule) -> str:
    """Generate an authored handler's clause bodies and register them.

    Each clause becomes a function with the foreign-clause calling
    convention, so the runtime installs authored and foreign handlers
    alike; the difference is only where the body came from.

    Parameters
    ----------
    handler
        The authored handler declaration.
    module
        The module it belongs to, for resolving names.

    Returns
    -------
    str
        Host-language source defining the clause functions and registering
        the handler under its stable identity.
    """
    lines: list[str] = []
    operations: list[str] = []
    for clause in handler.clauses:
        if clause.body is None:
            continue
        name = _clause_name(handler, _operation_name(module, clause.operation.text))
        params = [_local_name(local.name) for local in clause.parameters]
        unpack = "".join(
            f'    {param} = qiec_request["arguments"][{index + 1}]\n'
            for index, param in enumerate(params)
        )
        lines.append(
            f"function {name}(qiec_request, qiec_resume, qiec_context)\n"
            '    qiec_static = qiec_context["static"]\n'
            '    qiec_attachments = qiec_context["attachments"]\n'
            '    qiec_handlers = qiec_context["handlers"]\n'
            '    qiec_operations = qiec_context["operations"]\n'
            f"{unpack}"
            f"    return {_julia_computation(clause.body, module)}\n"
            "end\n"
        )
        operations.append(
            f'{_julia_string(clause.operation.text)} => Dict("invoke" => {name})'
        )
    return_entry = "nothing"
    if handler.return_clause is not None:
        name = _return_clause_name(handler)
        binder = _local_name(handler.return_clause.binder.name)
        lines.append(
            f"function {name}({binder}, qiec_context)\n"
            '    qiec_static = qiec_context["static"]\n'
            '    qiec_attachments = qiec_context["attachments"]\n'
            '    qiec_handlers = qiec_context["handlers"]\n'
            '    qiec_operations = qiec_context["operations"]\n'
            f"    return {_julia_computation(handler.return_clause.body, module)}\n"
            "end\n"
        )
        return_entry = name
    lines.append(
        f"_qvr_qiec_authored[{_julia_string(handler.id.text)}] = Dict("
        f'"operations" => Dict{{String, Any}}({", ".join(operations)}), '
        f'"return" => {return_entry}, "duplicable_context" => true)\n'
    )
    return "".join(lines)


def _julia_computation(
    node: IRQiecComputation, module: IRQiecModule, tail: bool = True
) -> str:
    if isinstance(node, IRQiecReturn):
        return f"_qvr_qiec_pure({_julia_value(node.value)})"
    if isinstance(node, IRQiecBind):
        binder = _local_name(node.binder.name)
        capture_names, attachment_ids = _free_runtime_capture(
            node.then, frozenset({node.binder.name})
        )
        captures = _julia_data(
            {name: f"__QVR_LOCAL__{_local_name(name)}" for name in capture_names}
        )
        for name in capture_names:
            captures = captures.replace(
                _julia_string(f"__QVR_LOCAL__{_local_name(name)}"), _local_name(name)
            )
        return f"_qvr_qiec_bind({_julia_computation(node.first, module, tail=False)}, {binder} -> {_julia_computation(node.then, module)}, _qvr_qiec_capture({captures}, {_julia_data(attachment_ids)}, qiec_attachments))"
    if isinstance(node, IRQiecPerform):
        args = ", ".join(_julia_value(value) for value in node.request.arguments)
        request = _runtime_request(node)
        request["arguments"] = "__QVR_ARGUMENTS__"
        rendered = _julia_data(request).replace(
            _julia_string("__QVR_ARGUMENTS__"), f"Any[{args}]"
        )
        return f"_qvr_qiec_effect({rendered}, qiec_static)"
    if isinstance(node, IRQiecHandle):
        handler = _handler(module, node.handler.text)
        return f"_qvr_qiec_handle(() -> {_julia_computation(node.computation, module)}, {_julia_string(node.instance.text)}, {_julia_data(_ir_data(handler))}, {_julia_data(_ir_data(node.static_arguments))}, qiec_static, qiec_handlers, qiec_attachments, qiec_operations)"
    if isinstance(node, IRQiecCall):
        callee = _computation_by_id(module, node.callee.text)
        arguments = "".join(
            f"{_julia_value(argument)}, " for argument in node.arguments
        )
        return (
            f"_qvr_qiec_enter_call({_julia_string(callee.name)}, () -> {_body_name(callee)}("
            f"{arguments}"
            f"_qvr_qiec_static_environment({_julia_data(_ir_data(callee.telescope))}, "
            f"_qvr_qiec_specialize({_julia_data(_ir_data(node.static_arguments))}, qiec_static)), "
            f"qiec_attachments, qiec_handlers, qiec_operations), {'true' if tail else 'false'})"
        )
    if isinstance(node, IRQiecIf):
        return f"_qvr_qiec_if({_julia_value(node.condition)}, () -> {_julia_computation(node.then, module, tail=tail)}, () -> {_julia_computation(node.otherwise, module, tail=tail)})"
    if isinstance(node, IRQiecResume):
        return f"_qvr_qiec_resume(qiec_resume, {_julia_value(node.value)})"
    if isinstance(node, IRQiecNewInstance):
        return f"_qvr_qiec_instance(() -> {_julia_computation(node.body, module)})"
    if isinstance(node, IRQiecCase):
        pairs = []
        for branch in node.branches:
            params = ", ".join(_local_name(field.name) for field in branch.fields)
            body = _branch_static(
                _julia_computation(branch.body, module), "qiec_static"
            )
            branch_params = ", ".join(
                item for item in ("qiec_branch_static", params) if item
            )
            pairs.append(
                f"{_julia_string(branch.constructor.text)} => (({branch_params}) -> {body})"
            )
        return f"_qvr_qiec_case({_julia_value(node.scrutinee)}, Dict({', '.join(pairs)}), {_julia_data(_case_metadata(node))}, qiec_static)"
    raise TypeError(f"unknown QIEC computation {node!r}")


def _julia_value(node: IRQiecValue) -> str:
    if isinstance(node, IRQiecVar):
        return f"_qvr_qiec_value({_local_name(node.local.name)})"
    if isinstance(node, IRQiecLiteralValue):
        return _julia_literal(node.value)
    if isinstance(node, IRQiecConstructorValue):
        return f"_qvr_qiec_constructor({_julia_string(node.constructor.text)}, {_julia_data(_ir_data(node.static_arguments))}, Any[{', '.join(_julia_value(field) for field in node.fields)}], {_julia_data(_ir_data(node.result_type))}, qiec_static)"
    if isinstance(node, IRQiecEvidenceValue):
        return (
            f"_qvr_qiec_evidence({_julia_data(_ir_data(node.evidence))}, qiec_static)"
        )
    if isinstance(node, IRQiecAttachmentRef):
        return f"_qvr_qiec_attachment(qiec_attachments, {_julia_string(node.attachment.text)}, {_julia_data(_ir_data(node.type))}, qiec_static)"
    if isinstance(node, IRQiecTransportValue):
        return f"_qvr_qiec_transport({_julia_data(_ir_data(node.evidence))}, {_julia_value(node.value)}, {_julia_data(_ir_data(node.target_type))}, qiec_static)"
    if isinstance(node, IRQiecPrimitiveApplication):
        arguments = ", ".join(_julia_value(argument) for argument in node.arguments)
        return f"_qvr_qiec_primitive({_julia_string(node.name)}, Any[{arguments}])"
    if isinstance(node, IRQiecTupleValue):
        items = ", ".join(_julia_value(item) for item in node.items)
        return f"({items},)" if len(node.items) == 1 else f"({items})"
    if isinstance(node, IRQiecProjection):
        return f"_qvr_qiec_project({_julia_value(node.value)}, {node.position})"
    raise TypeError(f"unknown QIEC value {node!r}")


def _julia_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _julia_literal(node: IRQiecLiteral) -> str:
    if isinstance(node, IRQiecNullLiteral):
        return "nothing"
    if isinstance(node, IRQiecBoolLiteral):
        return "true" if node.value else "false"
    if isinstance(node, IRQiecBytesLiteral):
        return f"hex2bytes({_julia_string(node.value)})"
    if isinstance(node, IRQiecTupleLiteral):
        items = ", ".join(_julia_literal(item) for item in node.items)
        return f"({items},)" if len(node.items) == 1 else f"({items})"
    if isinstance(node, IRQiecStringLiteral):
        return _julia_string(node.value)
    return repr(node.value)  # type: ignore[attr-defined]


def _julia_data(value: object) -> str:
    if isinstance(value, str):
        return _julia_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "nothing"
    if isinstance(value, dict):
        return (
            "Dict("
            + ", ".join(
                f"{_julia_data(key)} => {_julia_data(item)}"
                for key, item in value.items()
            )
            + ")"
        )
    if isinstance(value, (tuple, list)):
        return "Any[" + ", ".join(_julia_data(item) for item in value) + "]"
    return repr(value)


def _javascript_definition(item: IRQiecNamedComputation, module: IRQiecModule) -> str:
    params = [_local_name(parameter.name) for parameter in item.parameters]
    params.extend(
        (
            "qiec_static_arguments",
            "qiec_attachments",
            "qiec_handlers",
            "qiec_operations",
        )
    )
    body_params = [
        *(_local_name(parameter.name) for parameter in item.parameters),
        "qiec_static",
        "qiec_attachments",
        "qiec_handlers",
        "qiec_operations",
    ]
    return (
        f"var {_body_name(item)} = function({', '.join(body_params)}) {{\n"
        f"  return {_javascript_computation(item.body, module)};\n"
        "};\n"
        f"var {_function_name(item)} = function({', '.join(params)}) {{\n"
        f"  var qiec_static = _qvr_qiec_static_environment({json.dumps(_ir_data(item.telescope), ensure_ascii=False)}, qiec_static_arguments);\n"
        "  qiec_attachments = qiec_attachments || {};\n"
        "  qiec_handlers = qiec_handlers || {};\n"
        "  qiec_operations = qiec_operations || {};\n"
        f"  return _qvr_qiec_run(function() {{ return {_body_name(item)}({', '.join(body_params)}); }}, qiec_operations);\n"
        "};\n"
    )


def _javascript_authored_handler(
    handler: IRQiecHandlerDef, module: IRQiecModule
) -> str:
    """Generate an authored handler's clause bodies and register them.

    Each clause becomes a function with the foreign-clause calling
    convention, so the runtime installs authored and foreign handlers
    alike; the difference is only where the body came from.

    Parameters
    ----------
    handler
        The authored handler declaration.
    module
        The module it belongs to, for resolving names.

    Returns
    -------
    str
        Host-language source defining the clause functions and registering
        the handler under its stable identity.
    """
    lines: list[str] = []
    operations: list[str] = []
    for clause in handler.clauses:
        if clause.body is None:
            continue
        name = _clause_name(handler, _operation_name(module, clause.operation.text))
        params = [_local_name(local.name) for local in clause.parameters]
        unpack = "".join(
            f"  var {param} = qiec_request.arguments[{index}];\n"
            for index, param in enumerate(params)
        )
        lines.append(
            f"var {name} = function(qiec_request, qiec_resume, qiec_context) {{\n"
            "  var qiec_static = qiec_context.static;\n"
            "  var qiec_attachments = qiec_context.attachments;\n"
            "  var qiec_handlers = qiec_context.handlers;\n"
            "  var qiec_operations = qiec_context.operations;\n"
            f"{unpack}"
            f"  return {_javascript_computation(clause.body, module)};\n"
            "};\n"
        )
        operations.append(f"{json.dumps(clause.operation.text)}: {{ invoke: {name} }}")
    return_entry = "null"
    if handler.return_clause is not None:
        name = _return_clause_name(handler)
        binder = _local_name(handler.return_clause.binder.name)
        lines.append(
            f"var {name} = function({binder}, qiec_context) {{\n"
            "  var qiec_static = qiec_context.static;\n"
            "  var qiec_attachments = qiec_context.attachments;\n"
            "  var qiec_handlers = qiec_context.handlers;\n"
            "  var qiec_operations = qiec_context.operations;\n"
            f"  return {_javascript_computation(handler.return_clause.body, module)};\n"
            "};\n"
        )
        return_entry = name
    lines.append(
        f"_qvr_qiec_authored[{json.dumps(handler.id.text)}] = {{ operations: "
        f'{{ {", ".join(operations)} }}, "return": {return_entry}, '
        "duplicable_context: true };\n"
    )
    return "".join(lines)


def _javascript_computation(
    node: IRQiecComputation, module: IRQiecModule, tail: bool = True
) -> str:
    if isinstance(node, IRQiecReturn):
        return f"_qvr_qiec_pure({_javascript_value(node.value)})"
    if isinstance(node, IRQiecBind):
        binder = _local_name(node.binder.name)
        capture_names, attachment_ids = _free_runtime_capture(
            node.then, frozenset({node.binder.name})
        )
        captures = (
            "{"
            + ", ".join(
                f"{json.dumps(name)}: {_local_name(name)}" for name in capture_names
            )
            + "}"
        )
        return f"_qvr_qiec_bind({_javascript_computation(node.first, module, tail=False)}, function({binder}) {{ return {_javascript_computation(node.then, module)}; }}, _qvr_qiec_capture({captures}, {json.dumps(attachment_ids)}, qiec_attachments))"
    if isinstance(node, IRQiecPerform):
        args = ", ".join(_javascript_value(value) for value in node.request.arguments)
        request = _runtime_request(node)
        request["arguments"] = None
        rendered = json.dumps(request, ensure_ascii=False).replace(
            '"arguments": null', f'"arguments": [{args}]'
        )
        return f"_qvr_qiec_effect({rendered}, qiec_static)"
    if isinstance(node, IRQiecHandle):
        handler = _handler(module, node.handler.text)
        return f"_qvr_qiec_handle(function() {{ return {_javascript_computation(node.computation, module)}; }}, {json.dumps(node.instance.text)}, {json.dumps(_ir_data(handler), ensure_ascii=False)}, {json.dumps(_ir_data(node.static_arguments), ensure_ascii=False)}, qiec_static, qiec_handlers, qiec_attachments, qiec_operations)"
    if isinstance(node, IRQiecCall):
        callee = _computation_by_id(module, node.callee.text)
        arguments = "".join(
            f"{_javascript_value(argument)}, " for argument in node.arguments
        )
        return (
            f"_qvr_qiec_enter_call({json.dumps(callee.name)}, function() {{ return {_body_name(callee)}("
            f"{arguments}"
            f"_qvr_qiec_static_environment({json.dumps(_ir_data(callee.telescope), ensure_ascii=False)}, "
            f"_qvr_qiec_specialize({json.dumps(_ir_data(node.static_arguments), ensure_ascii=False)}, qiec_static)), "
            f"qiec_attachments, qiec_handlers, qiec_operations); }}, {'true' if tail else 'false'})"
        )
    if isinstance(node, IRQiecIf):
        return f"_qvr_qiec_if({_javascript_value(node.condition)}, function() {{ return {_javascript_computation(node.then, module, tail=tail)}; }}, function() {{ return {_javascript_computation(node.otherwise, module, tail=tail)}; }})"
    if isinstance(node, IRQiecResume):
        return f"_qvr_qiec_resume(qiec_resume, {_javascript_value(node.value)})"
    if isinstance(node, IRQiecNewInstance):
        return f"_qvr_qiec_instance(function() {{ return {_javascript_computation(node.body, module)}; }})"
    if isinstance(node, IRQiecCase):
        branches = []
        for branch in node.branches:
            params = ", ".join(_local_name(field.name) for field in branch.fields)
            body = _branch_static(
                _javascript_computation(branch.body, module), "qiec_static"
            )
            branches.append(
                f"{json.dumps(branch.constructor.text)}: function(qiec_branch_static{', ' if params else ''}{params}) {{ return {body}; }}"
            )
        return f"_qvr_qiec_case({_javascript_value(node.scrutinee)}, {{{', '.join(branches)}}}, {json.dumps(_case_metadata(node), ensure_ascii=False)}, qiec_static)"
    raise TypeError(f"unknown QIEC computation {node!r}")


def _javascript_value(node: IRQiecValue) -> str:
    if isinstance(node, IRQiecVar):
        return f"_qvr_qiec_value({_local_name(node.local.name)})"
    if isinstance(node, IRQiecLiteralValue):
        return _javascript_literal(node.value)
    if isinstance(node, IRQiecConstructorValue):
        return f"_qvr_qiec_constructor({json.dumps(node.constructor.text)}, {json.dumps(_ir_data(node.static_arguments), ensure_ascii=False)}, [{', '.join(_javascript_value(field) for field in node.fields)}], {json.dumps(_ir_data(node.result_type), ensure_ascii=False)}, qiec_static)"
    if isinstance(node, IRQiecEvidenceValue):
        return f"_qvr_qiec_evidence({json.dumps(_ir_data(node.evidence), ensure_ascii=False)}, qiec_static)"
    if isinstance(node, IRQiecAttachmentRef):
        return f"_qvr_qiec_attachment(qiec_attachments, {json.dumps(node.attachment.text)}, {json.dumps(_ir_data(node.type), ensure_ascii=False)}, qiec_static)"
    if isinstance(node, IRQiecTransportValue):
        return f"_qvr_qiec_transport({json.dumps(_ir_data(node.evidence), ensure_ascii=False)}, {_javascript_value(node.value)}, {json.dumps(_ir_data(node.target_type), ensure_ascii=False)}, qiec_static)"
    if isinstance(node, IRQiecPrimitiveApplication):
        arguments = ", ".join(
            _javascript_value(argument) for argument in node.arguments
        )
        return f"_qvr_qiec_primitive({json.dumps(node.name)}, [{arguments}])"
    if isinstance(node, IRQiecTupleValue):
        items = ", ".join(_javascript_value(item) for item in node.items)
        return f"Object.freeze([{items}])"
    if isinstance(node, IRQiecProjection):
        return f"_qvr_qiec_project({_javascript_value(node.value)}, {node.position})"
    raise TypeError(f"unknown QIEC value {node!r}")


def _javascript_literal(node: IRQiecLiteral) -> str:
    if isinstance(node, IRQiecNullLiteral):
        return "null"
    if isinstance(node, IRQiecBytesLiteral):
        return "[" + ", ".join(str(value) for value in bytes.fromhex(node.value)) + "]"
    if isinstance(node, IRQiecTupleLiteral):
        return (
            "Object.freeze(["
            + ", ".join(_javascript_literal(item) for item in node.items)
            + "])"
        )
    return json.dumps(node.value, ensure_ascii=False)  # type: ignore[attr-defined]


def _scheme_definition(item: IRQiecNamedComputation, module: IRQiecModule) -> str:
    params = " ".join(_local_name(parameter.name) for parameter in item.parameters)
    if params:
        params += " "
    return (
        f"(define ({_body_name(item)} {params}qiec-static qiec-attachments qiec-handlers qiec-operations)\n"
        f"  {_scheme_computation(item.body, module)})\n"
        f"(define ({_function_name(item)} {params}. qiec-abi)\n"
        "  (let* ((qiec-static-arguments (if (pair? qiec-abi) (car qiec-abi) #f))\n"
        f"         (qiec-static (_qvr-qiec-static-environment {_scheme_data(_ir_data(item.telescope))} qiec-static-arguments))\n"
        "         (qiec-attachments (if (and (pair? qiec-abi) (pair? (cdr qiec-abi))) (cadr qiec-abi) '()))\n"
        "         (qiec-handlers (if (and (pair? qiec-abi) (pair? (cdr qiec-abi)) (pair? (cddr qiec-abi))) (caddr qiec-abi) '()))\n"
        "         (qiec-operations (if (and (pair? qiec-abi) (pair? (cdr qiec-abi)) (pair? (cddr qiec-abi)) (pair? (cdddr qiec-abi))) (cadddr qiec-abi) '())))\n"
        f"    (_qvr-qiec-run (lambda () ({_body_name(item)} {params}qiec-static qiec-attachments qiec-handlers qiec-operations)) qiec-operations)))\n"
    )


def _scheme_authored_handler(handler: IRQiecHandlerDef, module: IRQiecModule) -> str:
    """Generate an authored handler's clause bodies and register them.

    Each clause becomes a function with the foreign-clause calling
    convention, so the runtime installs authored and foreign handlers
    alike; the difference is only where the body came from.

    Parameters
    ----------
    handler
        The authored handler declaration.
    module
        The module it belongs to, for resolving names.

    Returns
    -------
    str
        Host-language source defining the clause functions and registering
        the handler under its stable identity.
    """
    lines: list[str] = []
    operations: list[str] = []
    for clause in handler.clauses:
        if clause.body is None:
            continue
        name = _clause_name(handler, _operation_name(module, clause.operation.text))
        params = [_local_name(local.name) for local in clause.parameters]
        bindings = " ".join(
            f'({param} (list-ref (_qvr-qiec-member qiec-request "arguments") {index}))'
            for index, param in enumerate(params)
        )
        lines.append(
            f"(define ({name} qiec-request qiec-resume qiec-context)\n"
            f'  (let* ((qiec-static (_qvr-qiec-member qiec-context "static"))\n'
            f'         (qiec-attachments (_qvr-qiec-member qiec-context "attachments"))\n'
            f'         (qiec-handlers (_qvr-qiec-member qiec-context "handlers"))\n'
            f'         (qiec-operations (_qvr-qiec-member qiec-context "operations"))'
            f"{(' ' + bindings) if bindings else ''})\n"
            f"    {_scheme_computation(clause.body, module)}))\n"
        )
        operations.append(
            f'(cons {_scheme_string(clause.operation.text)} (list (cons "invoke" {name})))'
        )
    return_entry = "#f"
    if handler.return_clause is not None:
        name = _return_clause_name(handler)
        binder = _local_name(handler.return_clause.binder.name)
        lines.append(
            f"(define ({name} {binder} qiec-context)\n"
            f'  (let* ((qiec-static (_qvr-qiec-member qiec-context "static"))\n'
            f'         (qiec-attachments (_qvr-qiec-member qiec-context "attachments"))\n'
            f'         (qiec-handlers (_qvr-qiec-member qiec-context "handlers"))\n'
            f'         (qiec-operations (_qvr-qiec-member qiec-context "operations")))\n'
            f"    {_scheme_computation(handler.return_clause.body, module)}))\n"
        )
        return_entry = name
    lines.append(
        f"(_qvr-qiec-register-authored {_scheme_string(handler.id.text)} "
        f'(list (cons "operations" (list {" ".join(operations)})) '
        f'(cons "return" {return_entry}) (cons "duplicable_context" #t)))\n'
    )
    return "".join(lines)


def _scheme_computation(
    node: IRQiecComputation, module: IRQiecModule, tail: bool = True
) -> str:
    if isinstance(node, IRQiecReturn):
        return f"(_qvr-qiec-pure {_scheme_value(node.value)})"
    if isinstance(node, IRQiecBind):
        binder = _local_name(node.binder.name)
        capture_names, attachment_ids = _free_runtime_capture(
            node.then, frozenset({node.binder.name})
        )
        captures = (
            "(list "
            + " ".join(
                f"(cons {_scheme_string(name)} {_local_name(name)})"
                for name in capture_names
            )
            + ")"
        )
        return f"(_qvr-qiec-bind {_scheme_computation(node.first, module, tail=False)} (lambda ({binder}) {_scheme_computation(node.then, module)}) (_qvr-qiec-capture {captures} {_scheme_data(attachment_ids)} qiec-attachments))"
    if isinstance(node, IRQiecPerform):
        args = " ".join(_scheme_value(value) for value in node.request.arguments)
        request = _runtime_request(node)
        request["arguments"] = "__QVR_ARGUMENTS__"
        rendered = _scheme_data(request).replace(
            _scheme_string("__QVR_ARGUMENTS__"), f"(list {args})"
        )
        return f"(_qvr-qiec-effect {rendered} qiec-static)"
    if isinstance(node, IRQiecHandle):
        handler = _handler(module, node.handler.text)
        return f"(_qvr-qiec-handle (lambda () {_scheme_computation(node.computation, module)}) {_scheme_string(node.instance.text)} {_scheme_data(_ir_data(handler))} {_scheme_data(_ir_data(node.static_arguments))} qiec-static qiec-handlers qiec-attachments qiec-operations)"
    if isinstance(node, IRQiecCall):
        callee = _computation_by_id(module, node.callee.text)
        arguments = "".join(
            f"{_scheme_value(argument)} " for argument in node.arguments
        )
        return (
            f"(_qvr-qiec-enter-call {_scheme_string(callee.name)} (lambda () ({_body_name(callee)} "
            f"{arguments}"
            f"(_qvr-qiec-static-environment {_scheme_data(_ir_data(callee.telescope))} "
            f"(_qvr-qiec-specialize {_scheme_data(_ir_data(node.static_arguments))} qiec-static)) "
            f"qiec-attachments qiec-handlers qiec-operations)) {'#t' if tail else '#f'})"
        )
    if isinstance(node, IRQiecIf):
        return f"(_qvr-qiec-if {_scheme_value(node.condition)} (lambda () {_scheme_computation(node.then, module, tail=tail)}) (lambda () {_scheme_computation(node.otherwise, module, tail=tail)}))"
    if isinstance(node, IRQiecResume):
        return f"(_qvr-qiec-resume qiec-resume {_scheme_value(node.value)})"
    if isinstance(node, IRQiecNewInstance):
        return (
            f"(_qvr-qiec-instance (lambda () {_scheme_computation(node.body, module)}))"
        )
    if isinstance(node, IRQiecCase):
        rendered_branches = []
        for branch in node.branches:
            params = " ".join(_local_name(field.name) for field in branch.fields)
            body = _branch_static(
                _scheme_computation(branch.body, module), "qiec-static"
            )
            rendered_branches.append(
                f"(cons {_scheme_string(branch.constructor.text)} (lambda (qiec-branch-static{' ' if params else ''}{params}) {body}))"
            )
        branches = " ".join(rendered_branches)
        return f"(_qvr-qiec-case {_scheme_value(node.scrutinee)} (list {branches}) {_scheme_data(_case_metadata(node))} qiec-static)"
    raise TypeError(f"unknown QIEC computation {node!r}")


def _scheme_value(node: IRQiecValue) -> str:
    if isinstance(node, IRQiecVar):
        return f"(_qvr-qiec-value {_local_name(node.local.name)})"
    if isinstance(node, IRQiecLiteralValue):
        return _scheme_literal(node.value)
    if isinstance(node, IRQiecConstructorValue):
        return f"(_qvr-qiec-constructor {_scheme_string(node.constructor.text)} {_scheme_data(_ir_data(node.static_arguments))} (list {' '.join(_scheme_value(field) for field in node.fields)}) {_scheme_data(_ir_data(node.result_type))} qiec-static)"
    if isinstance(node, IRQiecEvidenceValue):
        return (
            f"(_qvr-qiec-evidence {_scheme_data(_ir_data(node.evidence))} qiec-static)"
        )
    if isinstance(node, IRQiecAttachmentRef):
        return f"(_qvr-qiec-attachment qiec-attachments {_scheme_string(node.attachment.text)} {_scheme_data(_ir_data(node.type))} qiec-static)"
    if isinstance(node, IRQiecTransportValue):
        return f"(_qvr-qiec-transport {_scheme_data(_ir_data(node.evidence))} {_scheme_value(node.value)} {_scheme_data(_ir_data(node.target_type))} qiec-static)"
    if isinstance(node, IRQiecPrimitiveApplication):
        arguments = " ".join(_scheme_value(argument) for argument in node.arguments)
        return f"(_qvr-qiec-primitive {_scheme_string(node.name)} (list {arguments}))"
    if isinstance(node, IRQiecTupleValue):
        items = " ".join(_scheme_value(item) for item in node.items)
        return f"(_qvr-qiec-tuple (list {items}))"
    if isinstance(node, IRQiecProjection):
        return f"(_qvr-qiec-project {_scheme_value(node.value)} {node.position})"
    raise TypeError(f"unknown QIEC value {node!r}")


def _scheme_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _scheme_literal(node: IRQiecLiteral) -> str:
    if isinstance(node, IRQiecNullLiteral):
        return "'()"
    if isinstance(node, IRQiecBoolLiteral):
        return "#t" if node.value else "#f"
    if isinstance(node, IRQiecBytesLiteral):
        return (
            "(list " + " ".join(str(value) for value in bytes.fromhex(node.value)) + ")"
        )
    if isinstance(node, IRQiecTupleLiteral):
        return (
            "(_qvr-qiec-tuple (list "
            + " ".join(_scheme_literal(item) for item in node.items)
            + "))"
        )
    if isinstance(node, IRQiecStringLiteral):
        return _scheme_string(node.value)
    return repr(node.value)  # type: ignore[attr-defined]


def _scheme_data(value: object) -> str:
    if isinstance(value, str):
        return _scheme_string(value)
    if isinstance(value, bool):
        return "#t" if value else "#f"
    if value is None:
        return "'()"
    if isinstance(value, dict):
        return (
            "(list "
            + " ".join(
                f"(cons {_scheme_data(key)} {_scheme_data(item)})"
                for key, item in value.items()
            )
            + ")"
        )
    if isinstance(value, (tuple, list)):
        return "(list " + " ".join(_scheme_data(item) for item in value) + ")"
    return repr(value)


def _static_definitions(module: IRQiecModule, target: str) -> tuple[str, str]:
    if target == "stan":
        definitions = "\n".join(_stan_function(item) for item in module.computations)
        return f"functions {{\n{definitions}\n}}\n", "functions"
    assignments = []
    for item in module.computations:
        statements, value = _linearize(item.body)
        environment: dict[str, str] = {}
        for position, (binder, bound) in enumerate(statements):
            target_name = (
                f"qiec_{_safe_name(item.name)}__{_safe_name(binder)}_{position}"
            )
            assignments.append(f"{target_name} <- {_static_value(bound, environment)}")
            environment[binder] = target_name
        assignments.append(
            f"{_function_name(item)} <- {_static_value(value, environment)}"
        )
    return "model {\n" + "\n".join(assignments) + "\n}\n", "model_block"


def _linearize(
    node: IRQiecComputation,
) -> tuple[list[tuple[str, IRQiecValue]], IRQiecValue]:
    if isinstance(node, IRQiecReturn):
        return [], node.value
    if isinstance(node, IRQiecBind):
        before, first = _linearize(node.first)
        after, result = _linearize(node.then)
        return [*before, (node.binder.name, first), *after], result
    raise TypeError("the QIEC capability analyzer admitted a non-pure computation")


def _static_value(node: IRQiecValue, environment: dict[str, str] | None = None) -> str:
    if isinstance(node, IRQiecVar):
        return (environment or {}).get(node.local.name, _local_name(node.local.name))
    if not isinstance(node, IRQiecLiteralValue):
        raise TypeError("the QIEC capability analyzer admitted a non-scalar value")
    literal = node.value
    if isinstance(literal, IRQiecBoolLiteral):
        return "1" if literal.value else "0"
    if isinstance(literal, (IRQiecIntLiteral, IRQiecFloatLiteral)):
        return repr(literal.value)
    raise TypeError("the QIEC capability analyzer admitted a non-numeric scalar")


def _stan_type(type_: object) -> str:
    if not isinstance(type_, IRQiecTypeApplication):
        raise TypeError("the QIEC capability analyzer admitted a non-monomorphic type")
    name = type_.constructor.name
    if name in {"Int", "Bool"}:
        return "int"
    if name == "Real":
        return "real"
    raise TypeError(
        f"the QIEC capability analyzer admitted unsupported Stan type {name}"
    )


def _stan_function(item: IRQiecNamedComputation) -> str:
    statements, value = _linearize(item.body)
    params = ", ".join(
        f"{_stan_type(parameter.type)} {_local_name(parameter.name)}"
        for parameter in item.parameters
    )
    lines = [f"{_stan_type(item.type.result)} {_function_name(item)}({params}) {{"]
    for binder, bound in statements:
        local = next(
            candidate.binder
            for candidate in _walk_binds(item.body)
            if candidate.binder.name == binder
        )
        lines.append(
            f"  {_stan_type(local.type)} {_local_name(binder)} = {_static_value(bound)};"
        )
    lines.append(f"  return {_static_value(value)};")
    lines.append("}")
    return "\n".join(lines)


def _walk_binds(node: IRQiecComputation) -> tuple[IRQiecBind, ...]:
    if not isinstance(node, IRQiecBind):
        return ()
    return (node, *_walk_binds(node.first), *_walk_binds(node.then))


__all__ = [
    "graft_qiec_dynamic",
    "graft_qiec_static",
    "has_qiec_computations",
    "qiec_target_name",
]
