# Two-Parameter Logistic IRT

## Overview

The 2PL [item response theory](https://en.wikipedia.org/wiki/Item_response_theory) model ([Birnbaum 1968](https://doi.org/10.1007/BF02288467)) for binary item responses `y_{ij}` of respondent i to item j. Each respondent carries a unidimensional ability `theta_i`, each item carries a difficulty `b_j` and a discrimination `a_j` (positive by construction via a [LogNormal](https://en.wikipedia.org/wiki/Log-normal_distribution) prior), and the probability of a correct response is `sigmoid(a_j * (theta_i - b_j))`.

## QVR Source

```qvr
object Person : FinSet 500
object Item : FinSet 30
object Resp : FinSet 15000

program irt_2pl : Resp -> Resp
    sample ability : Person <- Normal(0.0, 1.0)
    sample difficulty : Item <- Normal(0.0, 1.0)
    sample discrim : Item <- LogNormal(0.0, 1.0)

    let theta = ability[person_idx]
    let b = difficulty[item_idx]
    let a = discrim[item_idx]
    let eta = a * (theta - b)
    let p = sigmoid(eta)

    observe y : Resp <- Bernoulli(p)
    return p

export irt_2pl
```

## Walkthrough

`ability`, `difficulty`, and `discrim` are plate-bound on `Person` and `Item`; `discrim` carries a LogNormal prior so it's positive by construction. The runtime supplies `person_idx` and `item_idx` at fit time, and the gather idiom `ability[person_idx]` / `difficulty[item_idx]` / `discrim[item_idx]` realizes the standard cross-classified design. The Bernoulli link `sigmoid(a * (theta - b))` is the canonical [logistic](https://en.wikipedia.org/wiki/Logistic_function) form of the 2PL response function.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/irt_2pl.qvr")
model = prog.morphism

n_person, n_item, n_resp = 8, 8, 64
ability_true = torch.randn(n_person)
difficulty_true = torch.randn(n_item)
discrim_true = torch.randn(n_item).exp()
person_idx = torch.randint(0, n_person, (n_resp,))
item_idx = torch.randint(0, n_item, (n_resp,))
eta_true = discrim_true[item_idx] * (
    ability_true[person_idx] - difficulty_true[item_idx]
)
p_true = torch.sigmoid(eta_true)
y = torch.bernoulli(p_true)

observations = {"person_idx": person_idx, "item_idx": item_idx, "y": y}
x_in = torch.zeros(n_resp, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.Bernoulli(p_true).log_prob(y).mean()
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"y", "person_idx", "item_idx"})
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

n_resp_mcmc = 32
person_idx_mcmc = person_idx[:n_resp_mcmc]
item_idx_mcmc = item_idx[:n_resp_mcmc]
y_mcmc = y[:n_resp_mcmc]
obs_mcmc = {
    "person_idx": person_idx_mcmc,
    "item_idx": item_idx_mcmc,
    "y": y_mcmc,
}
x_in_mcmc = torch.zeros(n_resp_mcmc, 1)

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in_mcmc, obs_mcmc)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical Perspective

The 2PL is a Kleisli morphism over the Person + Item plate structure in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category. The plate-gather operations are pullbacks of indexed kernels along the response-row index maps `person_idx : Resp -> Person` and `item_idx : Resp -> Item`.
