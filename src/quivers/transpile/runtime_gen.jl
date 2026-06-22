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

struct LogisticDist <: Gen.Distribution{Float64} end

const logistic = LogisticDist()

function Gen.random(::LogisticDist, loc::Real, scale::Real)
    return rand(Distributions.Logistic(loc, scale))
end

function Gen.logpdf(::LogisticDist, x::Real, loc::Real, scale::Real)
    return Distributions.logpdf(Distributions.Logistic(loc, scale), x)
end

function Gen.logpdf_grad(::LogisticDist, x::Real, loc::Real, scale::Real)
    return (nothing, nothing, nothing)
end

Gen.has_output_grad(::LogisticDist) = false

Gen.has_argument_grads(::LogisticDist) = (false, false)

(::LogisticDist)(loc, scale) = Gen.random(LogisticDist(), loc, scale)

struct BetaBinomialDist <: Gen.Distribution{Int} end

const beta_binomial = BetaBinomialDist()

function Gen.random(::BetaBinomialDist, total_count::Int, concentration1::Real, concentration0::Real)
    return rand(Distributions.BetaBinomial(total_count, concentration1, concentration0))
end

function Gen.logpdf(::BetaBinomialDist, x::Int, total_count::Int, concentration1::Real, concentration0::Real)
    return Distributions.logpdf(Distributions.BetaBinomial(total_count, concentration1, concentration0), x)
end

function Gen.logpdf_grad(::BetaBinomialDist, x::Int, total_count::Int, concentration1::Real, concentration0::Real)
    return (nothing, nothing, nothing, nothing)
end

Gen.has_output_grad(::BetaBinomialDist) = false

Gen.has_argument_grads(::BetaBinomialDist) = (false, false, false)

(::BetaBinomialDist)(total_count, concentration1, concentration0) = Gen.random(BetaBinomialDist(), total_count, concentration1, concentration0)

struct HalfStudentTDist <: Gen.Distribution{Float64} end

const half_student_t = HalfStudentTDist()

function Gen.random(::HalfStudentTDist, df::Real, scale::Real)
    return abs(rand(Distributions.TDist(df))) * scale
end

function Gen.logpdf(::HalfStudentTDist, x::Real, df::Real, scale::Real)
    if x < 0
        return -Inf
    end
    return Distributions.logpdf(Distributions.TDist(df), x / scale) - log(scale) + log(2)
end

function Gen.logpdf_grad(::HalfStudentTDist, x::Real, df::Real, scale::Real)
    return (nothing, nothing, nothing)
end

Gen.has_output_grad(::HalfStudentTDist) = false

Gen.has_argument_grads(::HalfStudentTDist) = (false, false)

(::HalfStudentTDist)(df, scale) = Gen.random(HalfStudentTDist(), df, scale)

struct KumaraswamyDist <: Gen.Distribution{Float64} end

const kumaraswamy = KumaraswamyDist()

function Gen.random(::KumaraswamyDist, concentration1::Real, concentration0::Real)
    u = rand()
    return (1.0 - (1.0 - u)^(1.0 / concentration0))^(1.0 / concentration1)
end

function Gen.logpdf(::KumaraswamyDist, x::Real, concentration1::Real, concentration0::Real)
    if x <= 0 || x >= 1
        return -Inf
    end
    return log(concentration1) + log(concentration0) +
           (concentration1 - 1) * log(x) +
           (concentration0 - 1) * log1p(-x^concentration1)
end

function Gen.logpdf_grad(::KumaraswamyDist, x::Real, concentration1::Real, concentration0::Real)
    return (nothing, nothing, nothing)
end

Gen.has_output_grad(::KumaraswamyDist) = false

Gen.has_argument_grads(::KumaraswamyDist) = (false, false)

(::KumaraswamyDist)(concentration1, concentration0) = Gen.random(KumaraswamyDist(), concentration1, concentration0)

struct ContinuousBernoulliDist <: Gen.Distribution{Float64} end

const continuous_bernoulli = ContinuousBernoulliDist()

function _cb_log_norm(p::Real)
    if abs(p - 0.5) < 1e-4
        return log(2.0) + (4.0 / 3.0) * (p - 0.5)^2
    end
    return log(abs(2.0 * atanh(1.0 - 2.0 * p))) - log(abs(1.0 - 2.0 * p))
end

function Gen.random(::ContinuousBernoulliDist, probs::Real)
    u = rand()
    if abs(probs - 0.5) < 1e-4
        return u
    end
    return log1p(u * (2.0 * probs - 1.0) / (1.0 - probs)) / log(probs / (1.0 - probs))
end

function Gen.logpdf(::ContinuousBernoulliDist, x::Real, probs::Real)
    if x <= 0 || x >= 1
        return -Inf
    end
    return x * log(probs) + (1.0 - x) * log1p(-probs) + _cb_log_norm(probs)
end

function Gen.logpdf_grad(::ContinuousBernoulliDist, x::Real, probs::Real)
    return (nothing, nothing)
end

Gen.has_output_grad(::ContinuousBernoulliDist) = false

Gen.has_argument_grads(::ContinuousBernoulliDist) = (false,)

(::ContinuousBernoulliDist)(probs) = Gen.random(ContinuousBernoulliDist(), probs)
