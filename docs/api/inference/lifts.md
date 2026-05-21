# Bayesian Lifts

Convert a non-Bayesian model into a proper `MonadicProgram` that
SVI, NUTS, and the rest of `quivers.inference` consume uniformly.
Four lifts cover the standard patterns:

* [`bayesian_lift_parameters`](#quivers.inference.lifts.bayesian_lift_parameters)
  : Normal priors on every learnable parameter, optionally lifting
  intermediate `sample` sites as additional NUTS latents via the
  placeholder-cancellation construction.
* [`lift_to_bayesian_program`](#quivers.inference.lifts.lift_to_bayesian_program)
  : parameter-only morphism plus a user-chosen observation family
  (any `torch.distributions.Distribution` subclass), with a
  `location_fn` callback so the same lift works for `rsample`-style,
  `tensor`-attribute, and `program(x)`-forward shapes.
* [`lift_from_log_prob`](#quivers.inference.lifts.lift_from_log_prob)
  : parameter-only model whose forward is already a
  `log_prob(x, y)` function (e.g. composed Normal kernels' induced
  density).
* [`monte_carlo_log_joint`](#quivers.inference.lifts.monte_carlo_log_joint)
  : single-sample MC estimator of the conditional likelihood
  given a draw from an intermediate latent site. Valid for SVI as
  a stochastic gradient estimator; *not* a substitute for the
  joint-lift route on NUTS.

::: quivers.inference.lifts
