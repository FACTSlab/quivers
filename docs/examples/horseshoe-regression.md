# Multi-Coefficient Horseshoe Regression

## Overview

A multiple regression with the horseshoe prior ([Carvalho, Polson, and Scott 2010](https://doi.org/10.1093/biomet/asq017)). The horseshoe is a [global-local scale mixture of Normals](https://en.wikipedia.org/wiki/Sparsity-inducing_prior): a single global scale `tau` and per-coordinate local scales `lambda_p` jointly define the coefficient prior, inducing a spike-near-zero / heavy-tail mixture that adaptively shrinks small effects toward zero while leaving large effects nearly unbiased.

## QVR Source

```qvr
object Item : FinSet 200
object Coef : FinSet 4
object Resp : FinSet 800

program horseshoe_regression : Resp -> Resp
    sample tau <- HalfCauchy(1.0)
    sample lambda_local : Coef <- HalfCauchy(1.0)
    sample z_raw : Coef <- Normal(0.0, 1.0)
    sample alpha <- Normal(0.0, 5.0)
    sample sigma <- HalfCauchy(2.0)

    let lam = lambda_local[coef_idx]
    let z = z_raw[coef_idx]
    let beta = tau * lam * z
    let mu = alpha + beta * x

    observe y : Resp <- Normal(mu, sigma)
    return tau

export horseshoe_regression
```

## Walkthrough

The horseshoe has no closed-form marginal density: the right idiom in quivers is the explicit `tau * lambda * z` decomposition built from existing primitives. `tau ~ HalfCauchy(1)` is the global scale; `lambda_local : Coef <- HalfCauchy(1)` is a per-coordinate local scale plate; `z_raw : Coef <- Normal(0, 1)` is the standard-Normal raw draw. The deterministic combination `beta = tau * lam * z` gives the implied coefficient with heavy-tailed marginal density.

The per-coefficient plate-draws scale automatically to any P without rewriting the body. Each observation cell carries a `coef_idx` indicating which coefficient it loads on; the per-cell mean is `alpha + beta[coef_idx] * x`.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/horseshoe_regression.qvr")
model = prog.morphism

N, P = 16, 4
NP = N * P
coef_idx = torch.arange(P).repeat(N)
x = torch.randn(NP)

# Ground truth for every latent site. The horseshoe coefficient is
# the deterministic product beta = tau * lambda_local * z_raw, so we
# fix one decomposition that reproduces the target coefficients: a
# unit global scale and unit local scales, with the raw draws carrying
# the coefficient values themselves. Binding tau / lambda_local / z_raw
# (rather than only the derived beta) pins every free latent.
true_tau = torch.tensor(1.0)
true_lambda_local = torch.ones(P)
true_z_raw = torch.tensor([2.0, 0.0, -1.5, 0.0])
true_alpha = 0.3
true_sigma = 0.5

true_beta = true_tau * true_lambda_local * true_z_raw
mu_true = true_alpha + true_beta[coef_idx] * x
y = torch.distributions.Normal(mu_true, true_sigma).sample()

observations = {"x": x, "y": y, "coef_idx": coef_idx}
x_in = torch.zeros(NP, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.Normal(mu_true, true_sigma).log_prob(y).mean()
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "y", "coef_idx"})
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

N_mcmc, P_mcmc = 8, 4
NP_mcmc = N_mcmc * P_mcmc
coef_idx_mcmc = torch.arange(P_mcmc).repeat(N_mcmc)
x_mcmc = x[:NP_mcmc]
y_mcmc = y[:NP_mcmc]
obs_mcmc = {"x": x_mcmc, "y": y_mcmc, "coef_idx": coef_idx_mcmc}
x_in_mcmc = torch.zeros(NP_mcmc, 1)

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in_mcmc, obs_mcmc)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical Perspective

The model factors as the Kleisli composite of a global-local hyperprior kernel, a deterministic Hadamard product `tau * lambda * z` lifted into the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category as a Dirac kernel, and the per-row Normal likelihood. The plate-draws `lambda_local` and `z_raw` are Kleisli sections of the `Coef`-indexed plate, and `lambda_local[coef_idx]` is the [Kleisli pullback](https://ncatlab.org/nlab/show/Kleisli+category) along the fibration `Resp -> Coef` carried by the runtime index.


## References

- Carlos M. Carvalho, Nicholas G. Polson, and James G. Scott. 2010. The horseshoe estimator for sparse signals. *Biometrika*, 97(2):465–480.
