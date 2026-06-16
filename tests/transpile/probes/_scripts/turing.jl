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

# Coerce JSON3 scalar / array values into the native Julia shapes
# Distributions.jl and Turing.logjoint expect: NamedTuple of either
# scalars or `Vector{Float64}`. JSON3.Array does not flow through
# Distributions arithmetic, so we project into a concrete vector.
function _coerce_value(v)
    if v isa JSON3.Array
        return [Float64(x) for x in v]
    elseif v isa AbstractArray
        return [Float64(x) for x in v]
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

    # Eval the @model declaration in Main; the macro produces a
    # callable `model` symbol.
    Base.eval(Main, Meta.parse(source))

    log_densities = Float64[]
    for pt in points
        data = pt.data
        params = pt.params
        # Pass observed values as positional args (sorted by name to
        # match the python harness's convention).
        sorted_keys = sort(collect(keys(data)))
        args = Tuple(_coerce_value(data[Symbol(k)]) for k in sorted_keys)
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
