# In-container Gen.jl probe.
using Gen, Distributions, JSON3

function main()
    source = read("/io/source.jl", String)
    points = JSON3.read(read("/io/points.json", String))

    Base.eval(Main, Meta.parse(source))
    model = Main.model

    log_densities = Float64[]
    for pt in points
        data = pt.data
        params = pt.params
        args = Tuple(data[Symbol(k)] for k in sort(collect(keys(data))))
        constraints = Gen.choicemap()
        for (k, v) in merge(Dict(params), Dict(data))
            Gen.set_value!(constraints, Symbol(k), v)
        end
        weight, _ = Gen.assess(model, args, constraints)
        push!(log_densities, Float64(weight))
    end

    open("/io/result.json", "w") do io
        JSON3.write(io, (log_densities = log_densities,))
    end
end

main()
