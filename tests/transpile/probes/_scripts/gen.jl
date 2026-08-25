# In-container Gen.jl probe.
#
# Evals the transpiled `@gen function model(...) ... end` source,
# extracts each trace address (its name and how many indices it
# carries) by scanning the source for the `@trace ... :NAME` /
# `@trace ... (:NAME, m_AXIS...)` shapes the Gen renderer emits, then
# builds a `Gen.choicemap` that constrains every `params` value and
# every `data` value whose key matches a trace-site name (so
# covariate-only function args are skipped), and asks `Gen.assess` for
# the log-density at each test point.
#
# Every payload is reshaped to its declared shape first. The address
# arity decides how far into that shape the constraint walk descends:
# a rank-2 latent traced at `(:Z_mat, m_Item, m_LatentDim)` gets one
# scalar per cell, a simplex-valued latent traced at `(:theta, m_Doc)`
# gets the whole row at one address.
using Gen, Distributions, JSON3

include("/io/_reshape.jl")

function _trace_site_arities(source::String)
    # The Gen renderer emits trace sites in two forms (and two
    # macro-call shapes, space-separated and parenthesised):
    #
    #   space, scalar:    `@trace <dist> :name`
    #   space, batched:   `@trace <dist> (:name, m_<axis>...)`
    #   parens, scalar:   `@trace(<dist>, :name)`
    #   parens, batched:  `@trace(<dist>, (:name, m_<axis>...))`
    #
    # The number of index components in the address is the number of
    # plate axes the renderer wrapped a `for` loop around, and it is a
    # contract between the emitted model and the choicemap the probe
    # builds: a rank-2 latent traced at `(:Z_mat, m_Item, m_LatentDim)`
    # needs one scalar constraint per cell, while a simplex-valued
    # latent traced at `(:theta, m_Doc)` needs the whole row at one
    # address. Record the arity per site so the constraint walk knows
    # how far to descend into the reshaped value.
    arities = Dict{Symbol,Int}()
    # Scalar address: `:name` not followed by a trailing word
    # character (handled by Julia's regex word-boundary), and not
    # part of a tuple (so we exclude addresses preceded by `(`).
    scalar_re = r"@trace[\s(][^\n]*?[\s,]:([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*,)"
    for m in eachmatch(scalar_re, source)
        arities[Symbol(m.captures[1])] = 0
    end
    # Batched address: `(:name, i_1, ..., i_k)`, the `:` preceded by
    # `(` (possibly with whitespace). The index components the renderer
    # emits are loop variables, so a plain identifier list suffices.
    batched_re = r"@trace[\s(][^\n]*?\(\s*:([A-Za-z_][A-Za-z0-9_]*)\s*((?:,\s*[A-Za-z_][A-Za-z0-9_]*\s*)+)\)"
    for m in eachmatch(batched_re, source)
        arities[Symbol(m.captures[1])] = count(==(','), m.captures[2])
    end
    return arities
end

function _set_constraint_at!(constraints, name::Symbol, prefix, value, depth::Int)
    # Descend exactly `depth` axes, one per `for` loop the renderer
    # wrapped around the trace site, accumulating the index prefix.
    # Whatever remains at the bottom is the value the site draws, be it
    # a scalar or a simplex row, and it goes in at that one address.
    if depth > 0
        if !(value isa AbstractArray)
            error(
                "gen probe: `$name` is traced under a $depth-index " *
                "address but its value runs out of axes at prefix " *
                "$prefix"
            )
        end
        for (i, elem) in enumerate(value)
            _set_constraint_at!(
                constraints, name, (prefix..., i), elem, depth - 1,
            )
        end
        return constraints
    end
    address = isempty(prefix) ? name : (name, prefix...)
    Gen.set_value!(constraints, address, native_array(value))
    return constraints
end

function _set_constraint!(constraints, name::Symbol, value, arity::Int)
    return _set_constraint_at!(constraints, name, (), value, arity)
end

function main()
    source = read("/io/source.jl", String)
    points = JSON3.read(read("/io/points.json", String))
    shapes, dtypes = load_tables("/io")
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
    arities = _trace_site_arities(source)

    log_densities = Float64[]
    for pt in points
        # Rebuild each flat row-major payload at its declared shape
        # before it reaches either the model signature or the
        # choicemap: a rank-2 latent is traced under one address per
        # cell, so the constraint builder needs the nesting the flat
        # list erases.
        reshaped = shift_index_inputs(
            reshape_point(pt, shapes, dtypes), index_names,
        )
        data = reshaped["data"]
        params = reshaped["params"]
        # Pass observed values as positional args sorted by name to
        # match the renderer's alphabetical signature ordering.
        args = Tuple(
            native_array(data[k]) for k in sort(collect(keys(data)))
        )
        constraints = Gen.choicemap()
        for (k, v) in pairs(params)
            sym = Symbol(k)
            haskey(arities, sym) && _set_constraint!(
                constraints, sym, v, arities[sym],
            )
        end
        for (k, v) in pairs(data)
            sym = Symbol(k)
            haskey(arities, sym) && _set_constraint!(
                constraints, sym, v, arities[sym],
            )
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
