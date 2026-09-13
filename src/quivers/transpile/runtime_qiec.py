"""Target-side QIEC runtime embedded in generated Python modules.

Generated functions carry the checked structural ABI with every request and
handler delimiter. Process-local callbacks supply code only; they cannot
override checked coverage, grades, types, or forwarding policy.
"""

import contextvars as _qvr_qiec_contextvars


def _qvr_qiec_pure(value):
    return ("pure", value)


def _qvr_qiec_static_kind(argument):
    kind = argument.get("kind") if isinstance(argument, dict) else None
    if kind in {
        "type-variable",
        "type-application",
        "function-type",
        "equality-type",
    }:
        return "type"
    if kind in {"effect-variable", "effect-ref"}:
        return "effect"
    return "index"


def _qvr_qiec_static_environment(telescope, arguments):
    if arguments is None:
        arguments = []
    if isinstance(arguments, dict):
        expected = {binder["name"] for binder in telescope}
        if set(arguments) != expected:
            raise TypeError("QIEC static arguments do not match the checked telescope")
        ordered = [arguments[binder["name"]] for binder in telescope]
    else:
        ordered = list(arguments)
    if len(ordered) != len(telescope):
        raise TypeError(
            "QIEC static argument arity does not match the checked telescope"
        )
    environment = {}
    for binder, argument in zip(telescope, ordered, strict=True):
        if _qvr_qiec_static_kind(argument) != binder["kind"]:
            raise TypeError(
                "QIEC static argument kind does not match the checked telescope"
            )
        environment[binder["name"]] = argument
    return environment


def _qvr_qiec_specialize(value, environment):
    if isinstance(value, list):
        return [_qvr_qiec_specialize(item, environment) for item in value]
    if isinstance(value, tuple):
        return tuple(_qvr_qiec_specialize(item, environment) for item in value)
    if not isinstance(value, dict):
        return value
    if value.get("kind") in {
        "type-variable",
        "index-variable",
        "effect-variable",
    }:
        key = value.get("identity") or value.get("name")
        if key in environment:
            return environment[key]
    return {key: _qvr_qiec_specialize(item, environment) for key, item in value.items()}


_qvr_qiec_ambient_handlers = _qvr_qiec_contextvars.ContextVar(
    "qvr_qiec_ambient_handlers", default=()
)


def _qvr_qiec_unique(items):
    result = []
    for item in items:
        if not any(candidate is item for candidate in result):
            result.append(item)
    return tuple(result)


def _qvr_qiec_with_ambient(controller, thunk):
    token = _qvr_qiec_ambient_handlers.set(
        (*_qvr_qiec_ambient_handlers.get(), controller)
    )
    try:
        return thunk()
    finally:
        _qvr_qiec_ambient_handlers.reset(token)


def _qvr_qiec_effect(request, static_environment):
    request = _qvr_qiec_specialize(request, static_environment)
    request.setdefault("captures", ())
    controllers = _qvr_qiec_unique(
        (*request.get("handler_captures", ()), *_qvr_qiec_ambient_handlers.get())
    )
    request["handler_captures"] = controllers
    request["handler_states"] = tuple(
        controller["capture"]() for controller in controllers
    )
    return ("effect", request, _qvr_qiec_pure)


def _qvr_qiec_bind(computation, continuation, captures=()):
    if computation[0] == "pure":
        return continuation(computation[1])
    request = dict(computation[1])
    request["captures"] = (*request.get("captures", ()), *captures)
    return (
        "effect",
        request,
        lambda value: _qvr_qiec_bind(computation[2](value), continuation, captures),
    )


def _qvr_qiec_binding(value):
    return (
        isinstance(value, dict) and value.get("qiec") == "binding" and "value" in value
    )


def _qvr_qiec_value(value):
    return value["value"] if _qvr_qiec_binding(value) else value


def _qvr_qiec_constructor(
    constructor, static_arguments, fields, result_type, static_environment
):
    return {
        "qiec": "constructor",
        "constructor": constructor,
        "static_arguments": tuple(
            _qvr_qiec_specialize(static_arguments, static_environment)
        ),
        "fields": tuple(fields),
        "result_type": _qvr_qiec_specialize(result_type, static_environment),
    }


def _qvr_qiec_evidence(evidence, static_environment):
    return {
        "qiec": "evidence",
        "evidence": _qvr_qiec_specialize(evidence, static_environment),
    }


def _qvr_qiec_transport(evidence, value, target_type, static_environment):
    # Equality evidence is proof irrelevant at runtime, but retaining both
    # endpoints in the generated call prevents an erasing backend boundary.
    _qvr_qiec_specialize(evidence, static_environment)
    _qvr_qiec_specialize(target_type, static_environment)
    return value


def _qvr_qiec_attachment(attachments, attachment, expected_type, static_environment):
    if attachment not in attachments:
        raise KeyError("missing QIEC attachment " + attachment)
    binding = attachments[attachment]
    if not _qvr_qiec_binding(binding) or "type" not in binding:
        raise TypeError("QIEC attachment must be a typed binding descriptor")
    expected_type = _qvr_qiec_specialize(expected_type, static_environment)
    if binding["type"] != expected_type:
        raise TypeError("QIEC attachment type disagrees with checked IR")
    return _qvr_qiec_value(
        _qvr_qiec_validate(
            binding.get("validator"), binding, "attachment " + attachment
        )
    )


def _qvr_qiec_case(value, branches, metadata, static_environment):
    metadata = _qvr_qiec_specialize(metadata, static_environment)
    if not isinstance(value, dict) or value.get("qiec") != "constructor":
        raise TypeError("QIEC case scrutinee is not a constructor value")
    constructor = value.get("constructor")
    if constructor not in metadata["branches"]:
        raise ValueError("constructor is not admitted by the checked QIEC case")
    if constructor not in branches:
        raise ValueError("no QIEC case branch for constructor " + str(constructor))
    branch = metadata["branches"][constructor]
    canonical = branch.get("static_arguments", ())
    actual = value.get("static_arguments", ())
    if len(actual) < len(canonical):
        raise TypeError("QIEC constructor static arguments do not match case branch")
    branch_environment = dict(static_environment)
    matched = actual[-len(canonical) :] if canonical else ()
    for variable, argument in zip(canonical, matched, strict=True):
        key = variable.get("identity") or variable.get("name")
        if key is None:
            raise TypeError("QIEC case branch has an unbindable static variable")
        branch_environment[key] = argument
    return branches[constructor](branch_environment, *value.get("fields", ()))


def _qvr_qiec_member(container, key, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _qvr_qiec_as_computation(value):
    if isinstance(value, tuple) and value and value[0] in ("pure", "effect"):
        return value
    return _qvr_qiec_pure(value)


def _qvr_qiec_validate(validator, value, label):
    if validator is None:
        return value
    valid = validator(_qvr_qiec_value(value))
    if valid is False:
        raise TypeError("QIEC runtime value failed " + label)
    return value


def _qvr_qiec_validate_computation(computation, validator, label):
    return _qvr_qiec_bind(
        _qvr_qiec_as_computation(computation),
        lambda value: _qvr_qiec_pure(_qvr_qiec_validate(validator, value, label)),
    )


def _qvr_qiec_capture(locals_, attachment_ids, attachments):
    captured = [
        {"kind": "local", "label": name, "binding": binding}
        for name, binding in locals_.items()
    ]
    for attachment in attachment_ids:
        if attachment not in attachments:
            raise KeyError("missing QIEC attachment " + attachment)
        captured.append(
            {
                "kind": "attachment",
                "label": attachment,
                "binding": attachments[attachment],
            }
        )
    return tuple(captured)


def _qvr_qiec_duplicable(value):
    if _qvr_qiec_binding(value):
        return bool(value.get("duplicable", False))
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return True
    if isinstance(value, tuple):
        return all(_qvr_qiec_duplicable(item) for item in value)
    if isinstance(value, dict) and value.get("qiec") == "constructor":
        return all(_qvr_qiec_duplicable(item) for item in value.get("fields", ()))
    if isinstance(value, dict) and value.get("qiec") == "evidence":
        return True
    return False


def _qvr_qiec_validate_capture(request, current_handler):
    for capture in request.get("captures", ()):
        binding = capture["binding"]
        if not _qvr_qiec_duplicable(binding):
            raise RuntimeError(
                "QIEC unrestricted resumption captures nonduplicable "
                + capture["kind"]
                + " "
                + capture["label"]
            )
        if _qvr_qiec_binding(binding):
            capture["seed"] = binding["value"]
            if binding.get("mutable", False) and not callable(binding.get("fork")):
                raise RuntimeError(
                    "QIEC mutable duplicable binding requires fork " + capture["label"]
                )
    controllers = _qvr_qiec_unique(
        (*request.get("handler_captures", ()), current_handler)
    )
    for controller in controllers:
        controller["snapshot"]()
        handler = controller["get"]()
        if not _qvr_qiec_member(handler, "duplicable_context", False):
            raise RuntimeError(
                "QIEC unrestricted resumption captures nonduplicable handler context"
            )
        if _qvr_qiec_member(handler, "mutable_context", False) and not callable(
            _qvr_qiec_member(handler, "fork_context")
        ):
            raise RuntimeError(
                "QIEC mutable duplicable handler context requires fork_context"
            )


def _qvr_qiec_fork_capture(request, current_handler, shot):
    del shot
    for capture in request.get("captures", ()):
        binding = capture["binding"]
        if _qvr_qiec_binding(binding) and callable(binding.get("fork")):
            binding["value"] = binding["fork"](capture["seed"])
    controllers = _qvr_qiec_unique(
        (*request.get("handler_captures", ()), current_handler)
    )
    for controller in controllers:
        controller["install_fork"]()


def _qvr_qiec_readdress(computation, shot):
    if computation[0] == "pure":
        return computation
    request = dict(computation[1])
    static, dynamic, resumptions = request["address"]
    request["address"] = (static, dynamic, (*resumptions, shot))
    return (
        "effect",
        request,
        lambda value: _qvr_qiec_readdress(computation[2](value), shot),
    )


def _qvr_qiec_lifecycle(handler):
    return {"handler": handler, "finalized": False}


def _qvr_qiec_lifecycle_exit(lifecycle):
    if lifecycle["finalized"]:
        return
    lifecycle["finalized"] = True
    callback = _qvr_qiec_member(lifecycle["handler"], "on_exit")
    if callable(callback):
        callback()


def _qvr_qiec_lifecycle_drop(lifecycle):
    if lifecycle["finalized"]:
        return
    lifecycle["finalized"] = True
    callback = _qvr_qiec_member(lifecycle["handler"], "on_drop")
    if callable(callback):
        callback()


def _qvr_qiec_lifecycle_enter(handler):
    lifecycle = _qvr_qiec_lifecycle(handler)
    try:
        callback = _qvr_qiec_member(handler, "on_enter")
        if callable(callback):
            callback()
    except BaseException:
        _qvr_qiec_lifecycle_drop(lifecycle)
        raise
    return lifecycle


def _qvr_qiec_drop_request(request):
    for lifecycle in request.get("lifecycles", ()):
        _qvr_qiec_lifecycle_drop(lifecycle)


def _qvr_qiec_finalize(computation, lifecycle):
    current = _qvr_qiec_as_computation(computation)
    if current[0] == "pure":
        _qvr_qiec_lifecycle_exit(lifecycle)
        return current
    request = dict(current[1])
    request["lifecycles"] = (*request.get("lifecycles", ()), lifecycle)
    return (
        "effect",
        request,
        lambda value: _qvr_qiec_finalize(current[2](value), lifecycle),
    )


def _qvr_qiec_handler_controller(handler_ref, lifecycle_ref, manifest):
    seed = [handler_ref[0]]

    def install_fork():
        prototype = seed[0]
        fork = _qvr_qiec_member(prototype, "fork_context")
        if not callable(fork):
            return
        clone = fork(prototype)
        try:
            _qvr_qiec_validate_handler(clone, manifest)
        except BaseException:
            callback = _qvr_qiec_member(clone, "on_drop")
            if callable(callback):
                callback()
            raise
        lifecycle = _qvr_qiec_lifecycle_enter(clone)
        handler_ref[0] = clone
        lifecycle_ref[0] = lifecycle

    return {
        "get": lambda: handler_ref[0],
        "seed": lambda: seed[0],
        "snapshot": lambda: seed.__setitem__(0, handler_ref[0]),
        "install_fork": install_fork,
        "capture": lambda: (handler_ref[0], lifecycle_ref[0]),
        "restore": lambda state: (
            handler_ref.__setitem__(0, state[0]),
            lifecycle_ref.__setitem__(0, state[1]),
        ),
    }


def _qvr_qiec_handler_operations(manifest):
    return {clause["operation"] for clause in manifest["clauses"]}


def _qvr_qiec_validate_handler(handler, manifest):
    declared = _qvr_qiec_member(handler, "definition")
    if declared is not None and declared != manifest:
        raise ValueError("QIEC runtime handler definition disagrees with checked IR")
    structural = _qvr_qiec_handler_operations(manifest)
    executable = set(_qvr_qiec_member(handler, "operations", {}))
    if executable != structural:
        raise ValueError("QIEC runtime handler clauses disagree with checked IR")
    if callable(_qvr_qiec_member(handler, "fork_context")) and not _qvr_qiec_member(
        handler, "duplicable_context", False
    ):
        raise ValueError("QIEC handler fork_context requires duplicable_context")
    if (
        _qvr_qiec_member(handler, "mutable_context", False)
        and _qvr_qiec_member(handler, "duplicable_context", False)
        and not callable(_qvr_qiec_member(handler, "fork_context"))
    ):
        raise ValueError("QIEC mutable duplicable handler requires fork_context")


def _qvr_qiec_invoke(entry, *arguments):
    invoke = _qvr_qiec_member(entry, "invoke", entry)
    if not callable(invoke):
        raise TypeError("QIEC runtime entry is not callable")
    return invoke(*arguments)


def _qvr_qiec_handle(
    computation,
    instance,
    manifest,
    static_arguments,
    computation_static_environment,
    handlers,
):
    static_arguments = _qvr_qiec_specialize(
        static_arguments, computation_static_environment
    )
    handler_static_environment = _qvr_qiec_static_environment(
        manifest["telescope"], static_arguments
    )
    manifest = _qvr_qiec_specialize(manifest, handler_static_environment)
    manifest["telescope"] = []
    handler_id = manifest["id"]
    if handler_id not in handlers:
        raise KeyError("missing QIEC handler attachment " + handler_id)
    prototype = handlers[handler_id]
    handler = prototype
    factory = _qvr_qiec_member(prototype, "context_factory")
    if callable(factory):
        handler = factory()
    if handler is prototype and _qvr_qiec_member(prototype, "mutable_context", False):
        raise ValueError("QIEC mutable handler requires context_factory")
    handler_ref = [handler]
    _qvr_qiec_validate_handler(handler_ref[0], manifest)
    lifecycle_ref = [_qvr_qiec_lifecycle_enter(handler)]
    root_lifecycle = lifecycle_ref[0]
    controller = _qvr_qiec_handler_controller(handler_ref, lifecycle_ref, manifest)
    grades = {clause["operation"]: clause["grade"] for clause in manifest["clauses"]}
    context = {
        "handler": handler_id,
        "definition": manifest,
        "static_arguments": static_arguments,
        "resumption_uses": 0,
    }

    def walk(current):
        handler = handler_ref[0]
        if current[0] == "pure":
            value = _qvr_qiec_validate(
                _qvr_qiec_member(handler, "input_validator"),
                current[1],
                "handler input type",
            )
            return_clause = _qvr_qiec_member(handler, "return")
            result = (
                value
                if return_clause is None
                else _qvr_qiec_invoke(return_clause, value, context)
            )
            return _qvr_qiec_finalize(
                _qvr_qiec_validate_computation(
                    result,
                    _qvr_qiec_member(handler, "output_validator"),
                    "handler output type",
                ),
                lifecycle_ref[0],
            )
        request = current[1]
        for captured_controller, state in zip(
            request.get("handler_captures", ()),
            request.get("handler_states", ()),
            strict=True,
        ):
            if captured_controller is controller:
                controller["restore"](state)
                handler = handler_ref[0]
                break

        def continue_forwarded(value):
            handler_ref[0] = forwarded_handler
            lifecycle_ref[0] = forwarded_lifecycle
            return _qvr_qiec_with_ambient(controller, lambda: walk(current[2](value)))

        if request["instance"] != instance:
            forwarded_handler = handler_ref[0]
            forwarded_lifecycle = lifecycle_ref[0]
            forwarded = dict(request)
            forwarded["handler_captures"] = _qvr_qiec_unique(
                (*request.get("handler_captures", ()), controller)
            )
            forwarded["handler_states"] = tuple(
                captured["capture"]() for captured in forwarded["handler_captures"]
            )
            forwarded["lifecycles"] = (
                *request.get("lifecycles", ()),
                lifecycle_ref[0],
            )
            return (
                "effect",
                forwarded,
                continue_forwarded,
            )
        clauses = _qvr_qiec_member(handler, "operations", {})
        operation = request["operation"]
        if request["effect"] != manifest["effect"]:
            raise TypeError("QIEC request and installed handler effect disagree")
        if operation not in clauses:
            if not manifest["total"] or manifest["forwards_unknown"]:
                forwarded_handler = handler_ref[0]
                forwarded_lifecycle = lifecycle_ref[0]
                forwarded = dict(request)
                forwarded["handler_captures"] = _qvr_qiec_unique(
                    (*request.get("handler_captures", ()), controller)
                )
                forwarded["handler_states"] = tuple(
                    captured["capture"]() for captured in forwarded["handler_captures"]
                )
                forwarded["lifecycles"] = (
                    *request.get("lifecycles", ()),
                    lifecycle_ref[0],
                )
                return (
                    "effect",
                    forwarded,
                    continue_forwarded,
                )
            raise KeyError("missing QIEC handler clause " + operation)
        uses = [0]
        grade = grades[operation]
        if grade == "omega":
            _qvr_qiec_validate_capture(request, controller)

        def resume(value):
            shot = uses[0]
            uses[0] += 1
            context["resumption_uses"] = uses[0]
            if grade == "0":
                raise RuntimeError("QIEC zero-grade clause resumed")
            if grade in ("aff", "1") and uses[0] > 1:
                raise RuntimeError("QIEC resumption exceeds grade " + grade)
            if grade == "omega":
                _qvr_qiec_fork_capture(request, controller, shot)
            value = _qvr_qiec_validate(
                _qvr_qiec_member(clauses[operation], "result_validator"),
                value,
                "handler operation result type",
            )
            resumed = _qvr_qiec_with_ambient(
                controller,
                lambda: walk(_qvr_qiec_readdress(current[2](value), shot)),
            )
            return resumed[1] if resumed[0] == "pure" else resumed

        clause_lifecycle = lifecycle_ref[0]
        try:
            result = _qvr_qiec_invoke(clauses[operation], request, resume, context)
        except BaseException:
            _qvr_qiec_drop_request(request)
            _qvr_qiec_lifecycle_drop(clause_lifecycle)
            raise
        if grade == "1" and uses[0] != 1:
            _qvr_qiec_drop_request(request)
            _qvr_qiec_lifecycle_drop(clause_lifecycle)
            raise RuntimeError("QIEC linear clause must resume exactly once")
        if uses[0] == 0:
            _qvr_qiec_drop_request(request)
        if grade == "0" and uses[0] != 0:
            raise RuntimeError("QIEC zero-grade clause cannot resume")
        return _qvr_qiec_finalize(
            _qvr_qiec_validate_computation(
                result,
                _qvr_qiec_member(handler, "output_validator"),
                "handler output type",
            ),
            clause_lifecycle,
        )

    try:
        computation = _qvr_qiec_with_ambient(controller, computation)
        return walk(computation)
    except BaseException:
        _qvr_qiec_lifecycle_drop(lifecycle_ref[0])
        _qvr_qiec_lifecycle_drop(root_lifecycle)
        raise


def _qvr_qiec_operation(operations, request):
    key = (request["instance"], request["operation"])
    entry = operations.get(key)
    if entry is None:
        by_instance = operations.get(request["instance"], {})
        entry = _qvr_qiec_member(by_instance, request["operation"])
    if entry is None:
        raise RuntimeError(
            "unhandled QIEC operation "
            + request["operation"]
            + " on "
            + request["instance"]
        )
    result = _qvr_qiec_invoke(entry, request)
    return _qvr_qiec_validate(
        _qvr_qiec_member(entry, "result_validator"),
        result,
        "operation result type",
    )


def _qvr_qiec_run(computation, operations):
    current = computation
    while current[0] == "effect":
        try:
            current = current[2](_qvr_qiec_operation(operations, current[1]))
        except BaseException:
            _qvr_qiec_drop_request(current[1])
            raise
    return _qvr_qiec_value(current[1])
