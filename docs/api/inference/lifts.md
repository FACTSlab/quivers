# Bayesian Lifts

These functions convert non-Bayesian models to `MonadicProgram`
instances for inference. They cover four model interfaces:

* [`bayesian_lift_parameters`](#quivers.inference.lifts.bayesian_lift_parameters)
  assigns Normal priors to learnable parameters. It can also lift
  intermediate `sample` sites as NUTS latents through placeholder
  cancellation.
* [`lift_to_bayesian_program`](#quivers.inference.lifts.lift_to_bayesian_program)
  combines a parameter-only morphism with a chosen observation family,
  which may be any `torch.distributions.Distribution` subclass. Its
  `location_fn` callback handles `rsample` outputs, `tensor` attributes,
  and `program(x)` outputs.
* [`lift_from_log_prob`](#quivers.inference.lifts.lift_from_log_prob)
  accepts a parameter-only model whose forward method already computes
  `log_prob(x, y)`, such as the induced density of composed Normal
  kernels.
* [`monte_carlo_log_joint`](#quivers.inference.lifts.monte_carlo_log_joint)
  estimates a conditional likelihood from one draw at an intermediate
  latent site. It is a stochastic-gradient estimator for SVI, not a
  replacement for the joint lift used with NUTS.

::: quivers.inference.lifts
