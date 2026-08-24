# In-container Gen.jl probe.
#
# Evals the transpiled `@gen function model(...) ... end` source,
# extracts the trace's address names by scanning the source for the
# `@trace ... :NAME` / `@trace ... (:NAME, m_AXIS)` shapes the
# Gen renderer emits, then builds a `Gen.choicemap` that constrains
# every `params` value and every `data` value whose key matches a
# trace-site name (so covariate-only function args are skipped),
# and asks `Gen.assess` for the log-density at each test point.
using Gen, Distributions, JSON3

include("/io/_reshape.jl")

function _coerce(value)
    # JSON3 hands back lazy arrays / numbers; Gen's distribution calls
    # and the model body's broadcast expressions all want concrete
    # Julia scalars or `Vector{Float64}` / `Vector{Int}`. Coerce here
    # so every downstream consumer sees a fully realised value.
    if isa(value, AbstractVector)
        if all(x -> isa(x, Integer), value)
            return Vector{Int}(collect(value))
        end
        return Vector{Float64}(collect(value))
    end
    if isa(value, Integer)
        return Int(value)
    end
    if isa(value, Real)
        return Float64(value)
    end
    return value
end

function _trace_site_names(source::String)
    # The Gen renderer emits trace sites in two forms (and two
    # macro-call shapes -- space-separated and parenthesised):
    #
    #   space, scalar:    `@trace <dist> :name`
    #   space, batched:   `@trace <dist> (:name, m_<axis>...)`
    #   parens, scalar:   `@trace(<dist>, :name)`
    #   parens, batched:  `@trace(<dist>, (:name, m_<axis>...))`
    #
    # Scan for all four shapes; the address symbol always appears
    # after a separator (whitespace or `(`) and is preceded by `:`.
    sites = Set{Symbol}()
    # Scalar address: `:name` not followed by a trailing word
    # character (handled by Julia's regex word-boundary), and not
    # part of a tuple (so we exclude addresses preceded by `(`).
    scalar_re = r"@trace[\s(][^\n]*?[\s,]:([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*,)"
    for m in eachmatch(scalar_re, source)
        push!(sites, Symbol(m.captures[1]))
    end
    # Batched address: `(:name, ...)` — the `:` is preceded by `(`
    # (possibly with whitespace).
    batched_re = r"@trace[\s(][^\n]*?\(\s*:([A-Za-z_][A-Za-z0-9_]*)\s*,"
    for m in eachmatch(batched_re, source)
        push!(sites, Symbol(m.captures[1]))
    end
    return sites
end

function _set_constraint!(constraints, name::Symbol, value)
    # Vector-valued bindings live under per-index addresses
    # `(:name, i)` because the Gen renderer emits one `@trace` per
    # loop iteration. Scalars use the bare `:name` address.
    coerced = _coerce(value)
    if isa(coerced, AbstractVector)
        for i in 1:length(coerced)
            Gen.set_value!(constraints, (name, i), coerced[i])
        end
    else
        Gen.set_value!(constraints, name, coerced)
    end
end

function main()
    source = read("/io/source.jl", String)
    points = JSON3.read(read("/io/points.json", String))
    _, dtypes = load_tables("/io")
    # Julia arrays are 1-based; lift every 0-based covariate the model
    # subscripts before it reaches the @gen call.
    index_names = index_input_names(source, dtypes)

    # Use `include_string` (whole-file evaluation) instead of
    # `Meta.parse` (single expression) so source files with multiple
    # top-level statements -- runtime helper grafts (e.g. the
    # `TruncatedNormalDist` Gen.Distribution lift) followed by the
    # `@gen function model() ... end` -- load correctly. `Meta.parse`
    # only consumes one expression and rejects the rest with
    # "extra token after end of expression".
    Base.include_string(Main, source)
    sites = _trace_site_names(source)

    log_densities = Float64[]
    for pt in points
        data = pt.data
        params = pt.params
        # Pass observed values as positional args sorted by name to
        # match the renderer's alphabetical signature ordering. A
        # covariate the model subscripts is lifted to 1-based.
        args = Tuple(
            (String(k) in index_names ? _coerce(data[Symbol(k)]) .+ 1
                                      : _coerce(data[Symbol(k)]))
            for k in sort(collect(keys(data)))
        )
        constraints = Gen.choicemap()
        for (k, v) in pairs(params)
            sym = Symbol(k)
            sym in sites && _set_constraint!(constraints, sym, v)
        end
        for (k, v) in pairs(data)
            sym = Symbol(k)
            sym in sites && _set_constraint!(constraints, sym, v)
        end
        # `Base.invokelatest` is required because the `@gen` macro
        # registered the model definition in a newer world age than
        # the one this main() began executing in; without it Julia
        # rejects the call with a "method too new" MethodError.
        weight, _ = Base.invokelatest(
            Gen.assess, Main.model, args, constraints,
        )
        push!(log_densities, Float64(weight))
    end

    open("/io/result.json", "w") do io
        JSON3.write(io, (log_densities = log_densities,))
    end
end

main()
