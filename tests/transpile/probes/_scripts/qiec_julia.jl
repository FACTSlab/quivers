# In-container probe for QIEC entry points of a generated Julia module.
#
# Reads /io/source.jl and /io/calls.json, a list of [entry, arguments]
# pairs, calls each `qiec_<entry>` with its arguments, and writes the
# results to /io/result.json as plain numbers. The backend in
# /io/backend.txt picks the packages the generated module expects.
using JSON3, Distributions, LinearAlgebra

const BACKEND = strip(read("/io/backend.txt", String))
if BACKEND == "turing"
    using Turing
elseif BACKEND == "gen"
    using Gen
else
    error("unknown backend $(BACKEND)")
end

macro model(expression)
    esc(expression)
end

as_value(value) = value isa AbstractVector ? Tuple(as_value(item) for item in value) : value

Base.include_string(Main, read("/io/source.jl", String))
calls = JSON3.read(read("/io/calls.json", String))
results = Any[]
for (entry, arguments) in calls
    entry_point = getfield(Main, Symbol("qiec_" * String(entry)))
    result = Base.invokelatest(entry_point, (as_value(item) for item in arguments)...)
    push!(results, result)
end
write("/io/result.json", JSON3.write(results))
