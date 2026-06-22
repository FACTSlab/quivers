using Gen
using Distributions

struct TruncatedNormalDist <: Gen.Distribution{Float64} end

const truncated_normal = TruncatedNormalDist()

function Gen.random(::TruncatedNormalDist, loc::Real, scale::Real, low::Real, high::Real)
    return rand(Distributions.truncated(Distributions.Normal(loc, scale), low, high))
end

function Gen.logpdf(::TruncatedNormalDist, x::Real, loc::Real, scale::Real, low::Real, high::Real)
    return Distributions.logpdf(Distributions.truncated(Distributions.Normal(loc, scale), low, high), x)
end

function Gen.logpdf_grad(::TruncatedNormalDist, x::Real, loc::Real, scale::Real, low::Real, high::Real)
    return (nothing, nothing, nothing, nothing, nothing)
end

Gen.has_output_grad(::TruncatedNormalDist) = false

Gen.has_argument_grads(::TruncatedNormalDist) = (false, false, false, false)

(::TruncatedNormalDist)(loc, scale, low, high) = Gen.random(TruncatedNormalDist(), loc, scale, low, high)
