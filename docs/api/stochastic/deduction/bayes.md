# Bayesian Wrap

`nuts_program_from_deduction` lifts a deduction's learnable log weights
into a `MonadicProgram` with joint log density
$-\tfrac{1}{2\sigma^2}\lVert \mathbf{w} \rVert^2
+ \sum_n \log Z(s_n; \mathbf{w})$, ready for
[`MCMC`](../../inference/mcmc.md#quivers.inference.mcmc.MCMC).

The sampler targets this joint through a deterministic log density and
exact gradients. Whether that joint is the
Bayesian posterior $p(\mathbf{w} \mid S)$ depends on the
modelling reading (CRF / globally normalised vs. PCFG / locally
normalised). The module docstring states the required cancellation
condition.

::: quivers.stochastic.deduction.bayes
