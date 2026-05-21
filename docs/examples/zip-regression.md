# Multi-Output Zero-Inflated Poisson Regression

## Overview

The zero-inflated Poisson regression ([Lambert 1992](https://doi.org/10.2307/1269547)) is a two-component mixture of a point mass at zero and a [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) rate. The model fits count data with an excess of structural zeros relative to a plain Poisson likelihood. Each output dimension carries its own zero-inflation logits and rate coefficients, and the per-cell zero-inflation indicator is integrated out by a scoped `marginalize` block.

## QVR Source

```qvr
object Item : FinSet 200
object Out : FinSet 2
object Resp : FinSet 400

program zip_regression : Resp -> Resp
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

Per-output coefficient plates `alpha_zero`, `beta_zero` carry the [logit](https://en.wikipedia.org/wiki/Logit)-link zero-inflation probability `pi_{n, d}`, and `alpha_rate`, `beta_rate` carry the log-link Poisson rate `rate_{n, d}`. The zero-inflation indicator `z` is sampled per cell from a [`ContinuousBernoulli`](https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution) relaxation of the underlying Bernoulli, then integrated out by the enclosing `marginalize z` block: the coordinate is pushed forward through the projection on the trace's `z` axis, integrating out the indicator via reparameterised sampling. The continuous-Bernoulli relaxation gives a closed-form tractable density on `(0, 1)` and lets SVI integrate the coordinate via reparameterised sampling; the canonical logsumexp marginalization over the two integer states is the limiting case as the relaxation temperature tightens.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


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

true_az = torch.tensor([0.5, 1.0])
true_bz = torch.tensor([0.3, -0.2])
true_ar = torch.tensor([0.5, 1.0])
true_br = torch.tensor([1.0, -0.5])
pi_true = torch.sigmoid(true_az[out_idx] + true_bz[out_idx] * x)
rate_true = torch.exp(true_ar[out_idx] + true_br[out_idx] * x)
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


## Categorical Perspective

The model factors as a Kleisli composite of two kernels: a per-cell `ContinuousBernoulli(pi)` kernel on the unit interval and a `Poisson(rate)` kernel on the non-negative integers. The scoped `marginalize` step pushes forward the joint measure on the trace's `z` axis through projection, integrating out the indicator and leaving the marginal Poisson likelihood reweighted by the per-cell mixing weight. Categorically the construction is a coproduct fibration over the binary indicator axis, followed by [logsumexp](https://en.wikipedia.org/wiki/LogSumExp) on the accumulated log-likelihood in the discrete-limit case and reparameterised integration in the relaxed case.
