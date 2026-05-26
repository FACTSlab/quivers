# In-container Turing.jl probe.
#
# Reads /io/source.jl + /io/points.json; evals the @model source
# and writes /io/result.json with `Turing.logjoint(model, theta)`
# at each point.
using Turing, Distributions, LinearAlgebra, JSON3

function main()
    source = read("/io/source.jl", String)
    points = JSON3.read(read("/io/points.json", String))

    # Eval the @model declaration in Main; the macro produces a
    # callable `model` symbol.
    Base.eval(Main, Meta.parse(source))
    model_factory = Main.model

    log_densities = Float64[]
    for pt in points
        data = pt.data
        params = pt.params
        # Pass observed values as positional args (sorted by name to
        # match the python harness's convention).
        args = Tuple(data[Symbol(k)] for k in sort(collect(keys(data))))
        model_instance = model_factory(args...)
        theta = NamedTuple(params)
        lp = Turing.logjoint(model_instance, theta)
        push!(log_densities, Float64(lp))
    end

    open("/io/result.json", "w") do io
        JSON3.write(io, (log_densities = log_densities,))
    end
end

main()
