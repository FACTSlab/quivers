# In-container Turing.jl probe.
#
# Reads /io/source.jl + /io/points.json; evals the @model source
# and writes /io/result.json with `Turing.logjoint(model, theta)`
# at each point.
#
# Julia world-age note: `Base.eval` introduces a new method (the
# `@model`-expanded `model` factory) at a world age newer than the
# enclosing `main`'s. Calling `model_factory(args...)` directly from
# `main` would dispatch in `main`'s captured world, where the new
# method is not visible ("method too new to be called from this
# world context"). `Base.invokelatest` re-resolves dispatch in the
# latest world so the freshly-eval'd model is callable; we apply it
# to every cross-eval boundary call (factory construction, JSON
# coercions on params/data, `Turing.logjoint`).
using Turing, Distributions, LinearAlgebra, JSON3

include("/io/_reshape.jl")

# Coerce a reshaped value (already at the declared multi-dim shape
# and dtype per `/io/shapes.json`) into the native Julia container
# Distributions.jl / Turing.logjoint expect: `Vector{Float64}` for
# 1D, `Matrix{Float64}` for 2D, etc. The reshape helper returns
# `Array{Any}` for ndims > 1; project to the concrete Float64 array.
function _coerce_value(v)
    if v isa AbstractArray
        if ndims(v) == 1
            if all(x -> x isa Integer, v)
                return [Int(x) for x in v]
            end
            return [Float64(x) for x in v]
        end
        if all(row -> all(x -> x isa Integer, row), v)
            return Array{Int}(reduce(hcat, [[Int(x) for x in row]
                                            for row in v])')
        end
        return Array{Float64}(reduce(hcat, [[Float64(x) for x in row]
                                            for row in v])')
    elseif v isa Integer
        return v
    elseif v isa Real
        return Float64(v)
    else
        return v
    end
end

function _coerce_nt(d)
    pairs = Tuple((Symbol(k), _coerce_value(v)) for (k, v) in d)
    return NamedTuple{Tuple(p[1] for p in pairs)}(Tuple(p[2] for p in pairs))
end

function main()
    source = read("/io/source.jl", String)
    points = JSON3.read(read("/io/points.json", String))
    shapes, dtypes = load_tables("/io")
    # Julia arrays are 1-based; lift every 0-based covariate the model
    # subscripts before it reaches the @model call.
    index_names = index_input_names(source, dtypes)

    # Eval the @model declaration in Main; the macro produces a
    # callable `model` symbol.
    Base.eval(Main, Meta.parse(source))

    log_densities = Float64[]
    for pt in points
        reshaped = shift_index_inputs(
            reshape_point(pt, shapes, dtypes), index_names,
        )
        data = reshaped["data"]
        params = reshaped["params"]
        # Pass observed values as positional args (sorted by name to
        # match the python harness's convention).
        sorted_keys = sort(collect(keys(data)))
        args = Tuple(_coerce_value(data[k]) for k in sorted_keys)
        model_instance = Base.invokelatest(Main.model, args...)
        theta = _coerce_nt(params)
        lp = Base.invokelatest(Turing.logjoint, model_instance, theta)
        push!(log_densities, Float64(lp))
    end

    open("/io/result.json", "w") do io
        JSON3.write(io, (log_densities = log_densities,))
    end
end

main()
