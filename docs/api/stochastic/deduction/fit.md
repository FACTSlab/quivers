# Point-Estimate Fitting

`adam_fit_deduction` maximises the corpus log-marginal
$\sum_n \log Z(s_n; \mathbf{w})$ under an optional isotropic
Normal prior on the deduction's learnable log-weights. Each
$\log Z$ is computed exactly by the chart's LogProb-semiring
fixed point; autograd through the agenda's semiring operations
gives the exact gradient
$\nabla_\mathbf{w} \log Z(s; \mathbf{w})
= \mathbb{E}_{d \mid s}[\phi(d)]$ (the standard inside-outside
identity).

::: quivers.stochastic.deduction.fit
