# Multi-output continuous-gate Poisson regression

## Overview

This program is a continuous relaxation inspired by zero-inflated Poisson regression ([Lambert, 1992](https://doi.org/10.2307/1269547)). Each output dimension carries logits for a Poisson-active gate and coefficients for the Poisson rate. Unlike an exact zero-inflated Poisson mixture, the latent `z` is `ContinuousBernoulli` on `(0, 1)`, so the likelihood contains intermediate rates `z * rate` rather than only a point mass at zero and a full-rate Poisson component.

## QVR source

```qvr
# Multi-Output Zero-Inflated Poisson Regression
#
# A multi-output zero-inflated Poisson regression. ZIP models
# count data with an excess of structural zeros relative to a
# plain Poisson likelihood: each cell is a two-component
# mixture of a point mass at zero and Poisson(rate). Each
# output dimension carries its own zero-inflation logits and
# rate coefficients.
#
# Generative structure:
#
#   alpha_zero_d, beta_zero_d ~ Normal(0, 5)
#   alpha_rate_d, beta_rate_d ~ Normal(0, 5)
#   pi_{n, d}                  = sigmoid(alpha_zero_d + beta_zero_d * x_n)
#   rate_{n, d}                = exp(alpha_rate_d + beta_rate_d * x_n)
#   z_{n, d}                   ~ ContinuousBernoulli(pi_{n, d})
#   y_{n, d}                   ~ Poisson(z_{n, d} * rate_{n, d})
#
# The zero-inflation indicator z multiplicatively gates the
# Poisson rate. z near 0 yields Poisson(0) (the zero point
# mass); z near 1 recovers Poisson(rate). The enclosing
# `marginalize z` block integrates z out under the
# ContinuousBernoulli relaxation; the canonical hard form is a
# discrete Bernoulli with logsumexp reduction, recovered as the
# relaxation temperature tightens.
#
# Reference: [Lambert 1992](https://doi.org/10.2307/1269547).

object Item : FinSet 200
object Out : FinSet 2
object Resp : FinSet 400
object Val : Real 1

program zip_regression : Resp -> Val
    sample alpha_zero : Out <- Normal(0.0, 5.0)
    sample beta_zero : Out <- Normal(0.0, 5.0)
    sample alpha_rate : Out <- Normal(0.0, 5.0)
    sample beta_rate : Out <- Normal(0.0, 5.0)

    let az = alpha_zero[out_idx]
    let bz = beta_zero[out_idx]
    let ar = alpha_rate[out_idx]
    let br = beta_rate[out_idx]
    let pi_z = sigmoid(az + bz * x)
    let rate = exp(ar + br * x)

    marginalize z : Resp <- ContinuousBernoulli(pi_z)
        let gated_rate = z * rate
        observe y : Resp <- Poisson(gated_rate)

    return beta_rate

export zip_regression
```

## Walkthrough

Per-output coefficient plates `alpha_zero`, `beta_zero` carry the [logit](https://en.wikipedia.org/wiki/Logit)-link Poisson-active probability `pi_{n,d}`, while `alpha_rate`, `beta_rate` carry the log-link rate. The `ContinuousBernoulli` family has no temperature parameter here, so this source does not approach an exact two-state mixture by "tightening" a temperature. The exact ZIP oracle in the runnable block is thus a comparison distribution, not the likelihood implemented by the QVR program.

The program returns `beta_rate`, an `Out`-indexed plate of real scalars, so the declared codomain is `Val : Real 1`: the per-row value space of the returned coefficients. `Resp` names the flattened `(Item, Out)` plate extent and appears in the signature only as the domain.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/zip_regression.qvr")
model = prog.morphism

N, D = 200, 2
ND = N * D
out_idx = torch.arange(D).repeat(N)
x = torch.randn(ND)

true_alpha_zero = torch.tensor([0.5, 1.0])
true_beta_zero = torch.tensor([0.3, -0.2])
true_alpha_rate = torch.tensor([0.5, 1.0])
true_beta_rate = torch.tensor([1.0, -0.5])
pi_true = torch.sigmoid(true_alpha_zero[out_idx] + true_beta_zero[out_idx] * x)
rate_true = torch.exp(true_alpha_rate[out_idx] + true_beta_rate[out_idx] * x)
z_struct = torch.bernoulli(pi_true)
y = z_struct * torch.poisson(rate_true)

observations = {"x": x, "y": y, "out_idx": out_idx}
x_in = torch.zeros(ND, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

log_p_y0 = torch.log((1 - pi_true) + pi_true * torch.exp(-rate_true))
log_p_yk = torch.log(pi_true) + torch.distributions.Poisson(rate_true).log_prob(y)
oracle_nll = float(-torch.where(y == 0, log_p_y0, log_p_yk).mean())

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "y", "out_idx"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = []
for _ in range(300):
    losses.append(svi.step(x_in, observations))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
print(f"oracle NLL:   {oracle_nll:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical perspective

The model combines a per-cell `ContinuousBernoulli(pi)` kernel on the unit interval with a `Poisson(z * rate)` kernel on the non-negative integers. Because the latent support is continuous rather than binary, it should not be described as a coproduct over two indicator states or as the exact ZIP marginal.


## References

- Diane Lambert. 1992. Zero-inflated Poisson regression, with an application to defects in manufacturing. *Technometrics*, 34(1):1–14.
