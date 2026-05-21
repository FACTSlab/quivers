# Bayesian Wrap

`nuts_program_from_deduction` lifts the deduction's learnable
log-weights into a `MonadicProgram` whose joint log-density is
$-\tfrac{1}{2\sigma^2}\lVert \mathbf{w} \rVert^2
+ \sum_n \log Z(s_n; \mathbf{w})$, ready for
[`MCMC`](../../inference/mcmc.md#quivers.inference.mcmc.MCMC).

The sampler targets exactly that joint with a deterministic
log-density and exact gradients. Whether the joint is the
Bayesian posterior $p(\mathbf{w} \mid S)$ depends on the
modelling reading (CRF / globally normalised vs. PCFG / locally
normalised); see the module docstring for the precise statement
and the cancellation condition.

::: quivers.stochastic.deduction.bayes
