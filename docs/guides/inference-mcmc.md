# Variational Inference: MCMC and Hybrid Samplers

This page covers gradient-based MCMC (HMC, NUTS), the hybrid
samplers that combine a variational warm-up with HMC chains, and
posterior predictive sampling from an
[`MCMCResult`](../api/inference/predictive.md). The variational
families and ELBO objectives live in
[the SVI guide](inference-svi.md).

## MCMC: HMC and NUTS

When variational families underfit, fall back to gradient-based
MCMC. The kernel runs on the registry's unconstrained vector;
gradients flow through
[`torch.autograd.grad`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html).

```python
from quivers.inference import NUTSKernel, MCMC

kernel = NUTSKernel(
    target_accept=0.8,
    max_tree_depth=10,
    mass_matrix="diagonal",
)
mcmc = MCMC(
    kernel=kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
)
result = mcmc.run(model, x, observations)

print(result.r_hat)             # per-site split R-hat (Vehtari et al. 2021)
print(result.ess)               # effective sample size
print(result.divergence_counts) # per-chain divergence count
print(result.total_divergences) # sum across chains
samples = result.samples        # dict[str, Tensor] of shape (chains, draws, ...)
```

Both
[`HMCKernel`](../api/inference/predictive.md) and
[`NUTSKernel`](../api/inference/predictive.md) implement
[Nesterov dual-averaging](https://doi.org/10.48550/arXiv.1111.4246)
step-size adaptation and Welford-online mass-matrix adaptation
during warmup. The leapfrog integrator vectorizes `num_chains`
chains as a leading batch axis; warmup runs unvectorised
(adaptation is impure), sampling runs vectorized (kernel is pure).

The [`MCMCResult`](../api/inference/predictive.md) exposes the full
suite of [posterior diagnostics from Vehtari et al.
(2021)](https://doi.org/10.1214/20-BA1221): split `R̂`, bulk and
tail effective sample sizes, energy diagnostic (`E-BFMI`), and
per-chain divergence counts.

## Hybrid samplers

### AutoDAIS

[Differentiable annealed importance
sampling](https://doi.org/10.48550/arXiv.2107.10211) wraps a base
guide with $K$ HMC trajectories along an annealing path between
base and target. The base mean / scale, the step size, and the
inverse temperatures are jointly trained via SVI. Closes the parity
gap with NumPyro / Pyro `AutoDAIS`.

```python
from quivers.inference import AutoNormalGuide, AutoDAIS

base = AutoNormalGuide(model, observed_names={"y"})
guide = AutoDAIS(
    base,
    model=model,
    observations=observations,
    num_steps=8,
    init_step_size=0.05,
    init_temperature=0.1,
)
# Plug into SVI exactly like any other guide.
```

### WarmupThenHMC

Train a variational guide to convergence, then initialise HMC
chains from the guide's posterior mean. [Pareto-dominates
cold-start
HMC](https://doi.org/10.48550/arXiv.2108.03782) on hierarchical models with
skewed prior support.

```python
from quivers.inference import (
    AutoMultivariateNormalGuide, NUTSKernel, WarmupThenHMC
)

sampler = WarmupThenHMC(
    guide=AutoMultivariateNormalGuide(model, observed_names={"y"}),
    kernel=NUTSKernel(),
    svi_steps=1000,
    mcmc_warmup=500,
    mcmc_samples=2000,
)
svi_losses, result = sampler.run(model, x, observations)
```

## Predictive with MCMC

[`Predictive`](../api/inference/predictive.md) consumes either a
[`Guide`](../api/inference/guide.md) or an
[`MCMCResult`](../api/inference/predictive.md). With an
`MCMCResult`, it iterates over posterior samples instead of calling
`guide.rsample`.

```python
from quivers.inference import Predictive

predictive = Predictive(
    model=conditioned.model,
    posterior=result,
    num_samples=500,
)
samples = predictive(x_new)
```

## See also

- [Inference Foundations](inference-foundations.md): the trace,
  conditioning, and `LatentRegistry` primitives every MCMC kernel
  consumes.
- [SVI guide](inference-svi.md): the variational counterpart, and
  the SVI driver wrapping the `AutoDAIS` hybrid sampler.
- [Analysis Pipelines: Fitting and Diagnostics](analysis-fitting-and-diagnostics.md):
  the high-level `fit(...)` surface that wraps the MCMC and SVI
  drivers under one entry point.
