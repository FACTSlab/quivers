# Shared in-container reshape helper for Julia probes.
#
# Loads `/io/shapes.json` and `/io/dtypes.json` (each absent file
# resolves to an empty dict so legacy probe runs are a no-op),
# rebuilds nested arrays from row-major flat lists, and casts
# leaves to Int / Float64 per dtype.
#
# `reshape_value` is the per-name entry point; `reshape_point`
# takes a JSON3.Object with `params` / `data` sections and returns
# a `Dict{Symbol,Any}` ready for the runtime-specific coercion in
# the calling probe (turing.jl, gen.jl).

using JSON3

function _load_json_dict(path::AbstractString)
    if !isfile(path)
        return Dict{String,Any}()
    end
    raw = JSON3.read(read(path, String))
    # JSON3.Object iterates as (Symbol, value); rebuild with
    # explicitly stringified keys so the downstream
    # `Dict{String,Vector{Int}}` / `Dict{String,String}` typed
    # conversions accept the entries.
    return Dict{String,Any}(String(k) => v for (k, v) in pairs(raw))
end

function load_tables(io::AbstractString)
    shapes_raw = _load_json_dict(joinpath(io, "shapes.json"))
    dtypes_raw = _load_json_dict(joinpath(io, "dtypes.json"))
    shapes = Dict{String,Vector{Int}}(
        k => [Int(x) for x in v] for (k, v) in shapes_raw
    )
    dtypes = Dict{String,String}(k => String(v) for (k, v) in dtypes_raw)
    return shapes, dtypes
end

function _flat_to_nested(flat::AbstractVector, shape::Vector{Int})
    if isempty(shape)
        length(flat) == 1 || error(
            "scalar shape but length(flat)=$(length(flat))"
        )
        return flat[1]
    end
    expected = prod(shape)
    length(flat) == expected || error(
        "flat length $(length(flat)) does not match shape $shape "
        * "(expected $expected)"
    )
    if length(shape) == 1
        return collect(flat)
    end
    stride_ = div(expected, shape[1])
    return [
        _flat_to_nested(flat[(i - 1) * stride_ + 1:i * stride_], shape[2:end])
        for i in 1:shape[1]
    ]
end

function _cast_leaves(value, dtype::String)
    if value isa AbstractArray
        return [_cast_leaves(v, dtype) for v in value]
    end
    return dtype == "int" ? Int(value) : Float64(value)
end

function reshape_value(
    name::String, value, shapes::Dict, dtypes::Dict,
)
    out = value
    if haskey(shapes, name)
        if out isa Real
            out = _flat_to_nested([out], shapes[name])
        elseif out isa AbstractArray
            out = _flat_to_nested(collect(out), shapes[name])
        elseif out isa JSON3.Array
            out = _flat_to_nested([x for x in out], shapes[name])
        end
    end
    if haskey(dtypes, name)
        out = _cast_leaves(out, dtypes[name])
    end
    return out
end

function reshape_point(pt, shapes::Dict, dtypes::Dict)
    out = Dict{String,Dict{Symbol,Any}}()
    for section in ("params", "data")
        sec = haskey(pt, section) ? pt[section] : Dict{Symbol,Any}()
        out[section] = Dict{Symbol,Any}(
            Symbol(k) => reshape_value(String(k), v, shapes, dtypes)
            for (k, v) in sec
        )
    end
    return out
end

# The Julia targets (Turing, Gen) index arrays from 1, but the gallery
# covariates count from 0. `index_input_names` finds every int-dtyped
# name the source subscripts (`[name`); `shift_index_inputs` lifts
# those entries to 1-based after reshape. Count observations and
# response values are never subscripts, so they pass through untouched.
function index_input_names(source::AbstractString, dtypes::Dict)
    names = Set{String}()
    for (name, dt) in dtypes
        if dt == "int" && occursin(
            Regex("\\[\\s*" * name * "(?![0-9A-Za-z_])"), source,
        )
            push!(names, name)
        end
    end
    return names
end

function _offset_leaves(value, offset::Int)
    if value isa AbstractArray
        return [_offset_leaves(v, offset) for v in value]
    end
    return value + offset
end

function shift_index_inputs(point, names, offset::Int = 1)
    out = Dict{String,Dict{Symbol,Any}}()
    for section in ("params", "data")
        sec = point[section]
        out[section] = Dict{Symbol,Any}(
            k => (String(k) in names ? _offset_leaves(v, offset) : v)
            for (k, v) in sec
        )
    end
    return out
end
