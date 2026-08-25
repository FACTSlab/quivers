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

# `native_array` (from `_reshape.jl`) coerces a reshaped value into
# the native Julia container Distributions.jl / Turing.logjoint expect:
# `Vector{Float64}` for rank 1, `Matrix{Float64}` for rank 2, and an
# `Array{Float64,N}` for anything deeper. `reshape_value` hands back a
# nested vector rather than a multi-dimensional array, whose `ndims` is
# 1 whatever its true rank, so the projection reads the leaves rather
# than dispatching on `ndims`.
function _coerce_nt(d)
    pairs = Tuple((Symbol(k), native_array(v)) for (k, v) in d)
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
        args = Tuple(native_array(data[k]) for k in sorted_keys)
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
