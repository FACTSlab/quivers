functions {
  real kumaraswamy_lpdf(real y, real a, real b) {
    return log(a) + log(b) + (a - 1) * log(y) + (b - 1) * log1m(y ^ a);
  }
  real kumaraswamy_rng(real a, real b) {
    real u = uniform_rng(0, 1);
    return (1 - (1 - u) ^ (1 / b)) ^ (1 / a);
  }
  // Continuous Bernoulli (Loaiza-Ganem & Cunningham, NeurIPS 2019,
  // https://arxiv.org/abs/1907.06845). Stan ships no built-in, so
  // the renderer grafts these helpers when an IR sample / observe /
  // marginalize uses the ContinuousBernoulli family.
  //
  // The log normalizing constant C(p) = 2 * atanh(1 - 2p) / (1 - 2p)
  // is unstable near p = 0.5; the implementation mirrors PyTorch's
  // `ContinuousBernoulli._cont_bern_log_norm`: a fourth-order Taylor
  // expansion around 0.5 takes over when |p - 0.5| <= 0.001, and the
  // analytic form is used otherwise. The closed-form ``log_norm``
  // expression below matches the PyTorch one term-for-term.
  real continuous_bernoulli_log_norm(real p) {
    real half_gap = p - 0.5;
    if (abs(half_gap) <= 0.001) {
      real x2 = half_gap * half_gap;
      return log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x2) * x2;
    }
    real abs_log_diff = abs(log1m(p) - log(p));
    if (p <= 0.5) {
      return log(abs_log_diff) - log1m(2.0 * p);
    }
    return log(abs_log_diff) - log(2.0 * p - 1.0);
  }
  real continuous_bernoulli_lpdf(real y, real p) {
    return continuous_bernoulli_log_norm(p)
      + y * log(p) + (1 - y) * log1m(p);
  }
  // Sample via the inverse-CDF. The Taylor branch around p = 0.5
  // degenerates to ``u`` itself (the uniform on (0, 1) is the
  // limiting distribution at p = 0.5); the analytic branch matches
  // PyTorch's `icdf`.
  real continuous_bernoulli_rng(real p) {
    real u = uniform_rng(0, 1);
    if (abs(p - 0.5) <= 0.001) {
      return u;
    }
    return (log1p(-p + u * (2.0 * p - 1.0)) - log1m(p))
      / (log(p) - log1m(p));
  }
}
