# QIEC distribution bridge for generated Gen modules. A Gen distribution
# takes its parameters at each draw, so a first-class distribution is the
# distribution object paired with its argument list; a log density
# unpacks the list into `Gen.logpdf`. A QIEC tensor is a nested tuple
# until it reaches a distribution, where it becomes an array with the
# tuple's outermost axis first. A family whose Gen support is shifted
# against the QIEC convention is wrapped with its offset, which is added
# to the scored value.
struct _QvrQiecDistribution
    distribution
    arguments
end
struct _QvrQiecShifted
    distribution
    offset
end
_qvr_qiec_distribution(distribution, arguments...) = _QvrQiecDistribution(distribution, Any[arguments...])
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
        return _qvr_qiec_log_density(distribution.distribution, point .+ distribution.offset)
    end
    return Gen.logpdf(distribution.distribution, point, distribution.arguments...)
end
