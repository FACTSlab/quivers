functions {
  real kumaraswamy_lpdf(real y, real a, real b) {
    return log(a) + log(b) + (a - 1) * log(y) + (b - 1) * log1m(y ^ a);
  }
  real kumaraswamy_rng(real a, real b) {
    real u = uniform_rng(0, 1);
    return (1 - (1 - u) ^ (1 / b)) ^ (1 / a);
  }
}
