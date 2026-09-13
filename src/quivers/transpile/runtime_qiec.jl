abstract type _QvrQiecComputation end
struct _QvrQiecPure <: _QvrQiecComputation
    value
end
struct _QvrQiecEffect <: _QvrQiecComputation
    request
    continuation
end
_qvr_qiec_pure(value) = _QvrQiecPure(value)
function _qvr_qiec_static_kind(argument)
    kind = argument isa AbstractDict ? get(argument, "kind", nothing) : nothing
    kind in ("type-variable", "type-application", "function-type", "equality-type") && return "type"
    kind in ("effect-variable", "effect-ref") && return "effect"
    return "index"
end
function _qvr_qiec_static_environment(telescope, arguments)
    arguments === nothing && (arguments = Any[])
    if arguments isa AbstractDict
        Set(keys(arguments)) == Set(binder["name"] for binder in telescope) || error("QIEC static arguments do not match the checked telescope")
        ordered = Any[arguments[binder["name"]] for binder in telescope]
    else
        ordered = collect(arguments)
    end
    length(ordered) == length(telescope) || error("QIEC static argument arity does not match the checked telescope")
    environment = Dict()
    for (binder, argument) in zip(telescope, ordered)
        _qvr_qiec_static_kind(argument) == binder["kind"] || error("QIEC static argument kind does not match the checked telescope")
        environment[binder["name"]] = argument
    end
    return environment
end
function _qvr_qiec_specialize(value, environment)
    value isa AbstractVector && return Any[_qvr_qiec_specialize(item, environment) for item in value]
    value isa Tuple && return tuple((_qvr_qiec_specialize(item, environment) for item in value)...)
    value isa AbstractDict || return value
    if get(value, "kind", nothing) in ("type-variable", "index-variable", "effect-variable")
        identity = get(value, "identity", nothing)
        static_key = identity === nothing ? get(value, "name", nothing) : identity
        haskey(environment, static_key) && return environment[static_key]
    end
    return Dict(key => _qvr_qiec_specialize(item, environment) for (key, item) in value)
end
const _qvr_qiec_ambient_key = :qvr_qiec_ambient_handlers
_qvr_qiec_ambient_handlers() = get(task_local_storage(), _qvr_qiec_ambient_key, Any[])
function _qvr_qiec_unique(items)
    result = Any[]
    for item in items
        any(candidate -> candidate === item, result) || push!(result, item)
    end
    return result
end
function _qvr_qiec_with_ambient(controller, thunk)
    storage = task_local_storage()
    had_previous = haskey(storage, _qvr_qiec_ambient_key)
    previous = get(storage, _qvr_qiec_ambient_key, Any[])
    storage[_qvr_qiec_ambient_key] = Any[previous..., controller]
    try
        return thunk()
    finally
        had_previous ? (storage[_qvr_qiec_ambient_key] = previous) : delete!(storage, _qvr_qiec_ambient_key)
    end
end
function _qvr_qiec_effect(request, static_environment)
    copy_request = _qvr_qiec_specialize(request, static_environment)
    get!(copy_request, "captures", Any[])
    controllers = _qvr_qiec_unique(Any[get(copy_request, "handler_captures", Any[])..., _qvr_qiec_ambient_handlers()...])
    copy_request["handler_captures"] = controllers
    copy_request["handler_states"] = Any[controller["capture"]() for controller in controllers]
    return _QvrQiecEffect(copy_request, _qvr_qiec_pure)
end
function _qvr_qiec_bind(comp, continuation, captures=Any[])
    comp isa _QvrQiecPure && return continuation(comp.value)
    request = copy(comp.request)
    request["captures"] = Any[get(request, "captures", Any[])..., captures...]
    return _QvrQiecEffect(request, value -> _qvr_qiec_bind(comp.continuation(value), continuation, captures))
end
_qvr_qiec_is_binding(value) = value isa AbstractDict && get(value, "qiec", nothing) == "binding" && haskey(value, "value")
_qvr_qiec_value(value) = _qvr_qiec_is_binding(value) ? value["value"] : value
_qvr_qiec_constructor(constructor, static_arguments, fields, result_type, static_environment) = Dict("qiec" => "constructor", "constructor" => constructor, "static_arguments" => _qvr_qiec_specialize(static_arguments, static_environment), "fields" => fields, "result_type" => _qvr_qiec_specialize(result_type, static_environment))
_qvr_qiec_evidence(evidence, static_environment) = Dict("qiec" => "evidence", "evidence" => _qvr_qiec_specialize(evidence, static_environment))
function _qvr_qiec_transport(evidence, value, target_type, static_environment)
    _qvr_qiec_specialize(evidence, static_environment)
    _qvr_qiec_specialize(target_type, static_environment)
    return value
end
function _qvr_qiec_attachment(attachments, attachment, expected_type, static_environment)
    haskey(attachments, attachment) || error("missing QIEC attachment " * attachment)
    binding = attachments[attachment]
    _qvr_qiec_is_binding(binding) && haskey(binding, "type") || error("QIEC attachment must be a typed binding descriptor")
    expected_type = _qvr_qiec_specialize(expected_type, static_environment)
    binding["type"] == expected_type || error("QIEC attachment type disagrees with checked IR")
    return _qvr_qiec_value(_qvr_qiec_validate(get(binding, "validator", nothing), binding, "attachment " * attachment))
end
function _qvr_qiec_case(value, branches, metadata, static_environment)
    metadata = _qvr_qiec_specialize(metadata, static_environment)
    value isa AbstractDict && get(value, "qiec", nothing) == "constructor" || error("QIEC case scrutinee is not a constructor value")
    constructor = value["constructor"]
    haskey(metadata["branches"], constructor) || error("constructor is not admitted by the checked QIEC case")
    haskey(branches, constructor) || error("no QIEC case branch for " * constructor)
    branch = metadata["branches"][constructor]
    canonical = get(branch, "static_arguments", Any[])
    actual = get(value, "static_arguments", Any[])
    length(actual) >= length(canonical) || error("QIEC constructor static arguments do not match case branch")
    branch_environment = copy(static_environment)
    for (index, variable) in enumerate(canonical)
        identity = get(variable, "identity", nothing)
        static_key = identity === nothing ? get(variable, "name", nothing) : identity
        static_key === nothing && error("QIEC case branch has an unbindable static variable")
        branch_environment[static_key] = actual[length(actual) - length(canonical) + index]
    end
    return branches[constructor](branch_environment, value["fields"]...)
end
_qvr_qiec_as_computation(value) = value isa _QvrQiecComputation ? value : _qvr_qiec_pure(value)
function _qvr_qiec_validate(validator, value, label)
    validator === nothing && return value
    validator(_qvr_qiec_value(value)) === false && error("QIEC runtime value failed " * label)
    return value
end
_qvr_qiec_validate_computation(comp, validator, label) = _qvr_qiec_bind(_qvr_qiec_as_computation(comp), value -> _qvr_qiec_pure(_qvr_qiec_validate(validator, value, label)))
function _qvr_qiec_capture(locals, attachment_ids, attachments)
    captured = Any[Dict("kind" => "local", "label" => name, "binding" => binding) for (name, binding) in locals]
    for attachment in attachment_ids
        haskey(attachments, attachment) || error("missing QIEC attachment " * attachment)
        push!(captured, Dict("kind" => "attachment", "label" => attachment, "binding" => attachments[attachment]))
    end
    return captured
end
function _qvr_qiec_duplicable(value)
    _qvr_qiec_is_binding(value) && return get(value, "duplicable", false)
    (value === nothing || value isa Bool || value isa Number || value isa AbstractString || value isa Tuple) && return !(value isa Tuple) || all(_qvr_qiec_duplicable, value)
    value isa AbstractDict && get(value, "qiec", nothing) == "constructor" && return all(_qvr_qiec_duplicable, value["fields"])
    return value isa AbstractDict && get(value, "qiec", nothing) == "evidence"
end
function _qvr_qiec_validate_capture(request, current_handler)
    for capture in get(request, "captures", Any[])
        binding = capture["binding"]
        _qvr_qiec_duplicable(binding) || error("QIEC unrestricted resumption captures nonduplicable " * capture["kind"] * " " * capture["label"])
        if _qvr_qiec_is_binding(binding)
            capture["seed"] = binding["value"]
            get(binding, "mutable", false) && !(get(binding, "fork", nothing) isa Function) && error("QIEC mutable duplicable binding requires fork " * capture["label"])
        end
    end
    for controller in _qvr_qiec_unique(Any[get(request, "handler_captures", Any[])..., current_handler])
        controller["snapshot"]()
        handler = controller["get"]()
        get(handler, "duplicable_context", false) || error("QIEC unrestricted resumption captures nonduplicable handler context")
        get(handler, "mutable_context", false) && !(get(handler, "fork_context", nothing) isa Function) && error("QIEC mutable duplicable handler context requires fork_context")
    end
end
function _qvr_qiec_fork_capture(request, current_handler, shot)
    for capture in get(request, "captures", Any[])
        binding = capture["binding"]
        if _qvr_qiec_is_binding(binding) && get(binding, "fork", nothing) isa Function
            binding["value"] = binding["fork"](capture["seed"])
        end
    end
    for controller in _qvr_qiec_unique(Any[get(request, "handler_captures", Any[])..., current_handler])
        controller["install_fork"]()
    end
end
function _qvr_qiec_readdress(comp, shot)
    comp isa _QvrQiecPure && return comp
    request = copy(comp.request)
    request["address"] = Any[request["address"][1], request["address"][2], Any[request["address"][3]..., shot]]
    return _QvrQiecEffect(request, value -> _qvr_qiec_readdress(comp.continuation(value), shot))
end
function _qvr_qiec_validate_handler(handler, manifest)
    haskey(handler, "definition") && handler["definition"] != manifest && error("QIEC runtime handler definition disagrees with checked IR")
    structural = Set(clause["operation"] for clause in manifest["clauses"])
    Set(keys(get(handler, "operations", Dict()))) == structural || error("QIEC runtime handler clauses disagree with checked IR")
    get(handler, "fork_context", nothing) isa Function && !get(handler, "duplicable_context", false) && error("QIEC handler fork_context requires duplicable_context")
    get(handler, "mutable_context", false) && get(handler, "duplicable_context", false) && !(get(handler, "fork_context", nothing) isa Function) && error("QIEC mutable duplicable handler requires fork_context")
end
_qvr_qiec_lifecycle(handler) = Dict("handler" => handler, "finalized" => false)
function _qvr_qiec_lifecycle_exit(lifecycle)
    lifecycle["finalized"] && return
    lifecycle["finalized"] = true
    callback = get(lifecycle["handler"], "on_exit", nothing)
    callback isa Function && callback()
end
function _qvr_qiec_lifecycle_drop(lifecycle)
    lifecycle["finalized"] && return
    lifecycle["finalized"] = true
    callback = get(lifecycle["handler"], "on_drop", nothing)
    callback isa Function && callback()
end
function _qvr_qiec_lifecycle_enter(handler)
    lifecycle = _qvr_qiec_lifecycle(handler)
    try
        callback = get(handler, "on_enter", nothing)
        callback isa Function && callback()
    catch
        _qvr_qiec_lifecycle_drop(lifecycle)
        rethrow()
    end
    return lifecycle
end
function _qvr_qiec_drop_request(request)
    foreach(_qvr_qiec_lifecycle_drop, get(request, "lifecycles", Any[]))
end
function _qvr_qiec_finalize(comp, lifecycle)
    current = _qvr_qiec_as_computation(comp)
    if current isa _QvrQiecPure
        _qvr_qiec_lifecycle_exit(lifecycle)
        return current
    end
    request = copy(current.request)
    request["lifecycles"] = Any[get(request, "lifecycles", Any[])..., lifecycle]
    return _QvrQiecEffect(request, value -> _qvr_qiec_finalize(current.continuation(value), lifecycle))
end
_qvr_qiec_invoke(entry, arguments...) = (entry isa AbstractDict ? entry["invoke"] : entry)(arguments...)
function _qvr_qiec_handle(comp, instance, manifest, static_arguments, computation_static_environment, handlers)
    static_arguments = _qvr_qiec_specialize(static_arguments, computation_static_environment)
    handler_static_environment = _qvr_qiec_static_environment(manifest["telescope"], static_arguments)
    manifest = _qvr_qiec_specialize(manifest, handler_static_environment)
    manifest["telescope"] = Any[]
    handler_id = manifest["id"]
    haskey(handlers, handler_id) || error("missing QIEC handler attachment " * handler_id)
    prototype = handlers[handler_id]
    handler = prototype
    get(prototype, "context_factory", nothing) isa Function && (handler = prototype["context_factory"]())
    handler === prototype && get(prototype, "mutable_context", false) && error("QIEC mutable handler requires context_factory")
    handler_ref = Any[handler]
    _qvr_qiec_validate_handler(handler_ref[1], manifest)
    lifecycle_ref = Any[_qvr_qiec_lifecycle_enter(handler)]
    root_lifecycle = lifecycle_ref[1]
    seed_ref = Any[handler]
    controller = Dict(
        "get" => (() -> handler_ref[1]),
        "seed" => (() -> seed_ref[1]),
        "snapshot" => (() -> (seed_ref[1] = handler_ref[1])),
        "capture" => (() -> Any[handler_ref[1], lifecycle_ref[1]]),
        "restore" => (state -> begin
            handler_ref[1] = state[1]
            lifecycle_ref[1] = state[2]
        end),
        "install_fork" => (() -> begin
            seed = seed_ref[1]
            fork = get(seed, "fork_context", nothing)
            fork isa Function || return
            clone = fork(seed)
            try
                _qvr_qiec_validate_handler(clone, manifest)
            catch
                callback = get(clone, "on_drop", nothing)
                callback isa Function && callback()
                rethrow()
            end
            handler_ref[1] = clone
            lifecycle_ref[1] = _qvr_qiec_lifecycle_enter(clone)
        end),
    )
    grades = Dict(clause["operation"] => clause["grade"] for clause in manifest["clauses"])
    context = Dict("handler" => handler_id, "definition" => manifest, "static_arguments" => static_arguments, "resumption_uses" => 0)
    function walk(current)
        handler = handler_ref[1]
        if current isa _QvrQiecPure
            value = _qvr_qiec_validate(get(handler, "input_validator", nothing), current.value, "handler input type")
            return_clause = get(handler, "return", nothing)
            result = return_clause === nothing ? value : _qvr_qiec_invoke(return_clause, value, context)
            return _qvr_qiec_finalize(_qvr_qiec_validate_computation(result, get(handler, "output_validator", nothing), "handler output type"), lifecycle_ref[1])
        end
        request = current.request
        for (captured_controller, state) in zip(get(request, "handler_captures", Any[]), get(request, "handler_states", Any[]))
            if captured_controller === controller
                controller["restore"](state)
                handler = handler_ref[1]
            end
        end
        function forward()
            forwarded_handler = handler_ref[1]
            forwarded_lifecycle = lifecycle_ref[1]
            forwarded = copy(request)
            forwarded["handler_captures"] = _qvr_qiec_unique(Any[get(request, "handler_captures", Any[])..., controller])
            forwarded["handler_states"] = Any[captured["capture"]() for captured in forwarded["handler_captures"]]
            forwarded["lifecycles"] = Any[get(request, "lifecycles", Any[])..., lifecycle_ref[1]]
            return _QvrQiecEffect(forwarded, value -> begin
                handler_ref[1] = forwarded_handler
                lifecycle_ref[1] = forwarded_lifecycle
                _qvr_qiec_with_ambient(controller, () -> walk(current.continuation(value)))
            end)
        end
        request["instance"] != instance && return forward()
        clauses = get(handler, "operations", Dict())
        operation = request["operation"]
        request["effect"] == manifest["effect"] || error("QIEC request and installed handler effect disagree")
        if !haskey(clauses, operation)
            (!manifest["total"] || manifest["forwards_unknown"]) && return forward()
            error("missing QIEC handler clause " * operation)
        end
        uses = Ref(0)
        grade = grades[operation]
        grade == "omega" && _qvr_qiec_validate_capture(request, controller)
        function resume(value)
            shot = uses[]
            uses[] += 1
            context["resumption_uses"] = uses[]
            grade == "0" && error("QIEC zero-grade clause resumed")
            grade in ("aff", "1") && uses[] > 1 && error("QIEC resumption exceeds grade " * grade)
            grade == "omega" && _qvr_qiec_fork_capture(request, controller, shot)
            validator = clause isa AbstractDict ? get(clause, "result_validator", nothing) : nothing
            value = _qvr_qiec_validate(validator, value, "handler operation result type")
            resumed = _qvr_qiec_with_ambient(controller, () -> walk(_qvr_qiec_readdress(current.continuation(value), shot)))
            return resumed isa _QvrQiecPure ? resumed.value : resumed
        end
        clause = clauses[operation]
        clause_lifecycle = lifecycle_ref[1]
        result = try
            _qvr_qiec_invoke(clause, request, resume, context)
        catch
            _qvr_qiec_drop_request(request)
            _qvr_qiec_lifecycle_drop(clause_lifecycle)
            rethrow()
        end
        if grade == "1" && uses[] != 1
            _qvr_qiec_drop_request(request)
            _qvr_qiec_lifecycle_drop(clause_lifecycle)
            error("QIEC linear clause must resume exactly once")
        end
        uses[] == 0 && _qvr_qiec_drop_request(request)
        return _qvr_qiec_finalize(_qvr_qiec_validate_computation(result, get(handler, "output_validator", nothing), "handler output type"), clause_lifecycle)
    end
    try
        comp = _qvr_qiec_with_ambient(controller, comp)
        return walk(comp)
    catch
        _qvr_qiec_lifecycle_drop(lifecycle_ref[1])
        _qvr_qiec_lifecycle_drop(root_lifecycle)
        rethrow()
    end
end
function _qvr_qiec_run(comp, operations)
    current = comp
    while current isa _QvrQiecEffect
        request = current.request
        key = (request["instance"], request["operation"])
        entry = get(operations, key, nothing)
        entry === nothing && haskey(operations, request["instance"]) && (entry = get(operations[request["instance"]], request["operation"], nothing))
        entry === nothing && error("unhandled QIEC operation " * request["operation"] * " on " * request["instance"])
        try
            result = _qvr_qiec_invoke(entry, request)
            validator = entry isa AbstractDict ? get(entry, "result_validator", nothing) : nothing
            current = current.continuation(_qvr_qiec_validate(validator, result, "operation result type"))
        catch
            _qvr_qiec_drop_request(request)
            rethrow()
        end
    end
    return _qvr_qiec_value(current.value)
end
