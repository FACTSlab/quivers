# QIEC distribution bridge for generated Turing modules. A QIEC tensor is
# a nested tuple until it reaches a distribution, where it becomes an
# array with the tuple's outermost axis first; a log density is
# Distributions.jl's `logpdf` at the converted value. A family whose
# Distributions.jl support is shifted against the QIEC convention is
# wrapped with its offset, which is added to the scored value.
struct _QvrQiecShifted
    distribution
    offset
end
_qvr_qiec_shifted(distribution, offset) = _QvrQiecShifted(distribution, offset)
function _qvr_qiec_array(value)
    value = _qvr_qiec_value(value)
    if value isa AbstractArray
        return value
    end
    if value isa Tuple
        entries = [_qvr_qiec_array(item) for item in value]
        if !isempty(entries) && all(item -> item isa AbstractArray, entries)
            return stack(entries; dims=1)
        end
        return collect(promote(entries...))
    end
    return value
end
function _qvr_qiec_log_density(distribution, value)
    point = _qvr_qiec_array(value)
    if distribution isa _QvrQiecShifted
        return logpdf(distribution.distribution, point .+ distribution.offset)
    end
    return logpdf(distribution, point)
end
