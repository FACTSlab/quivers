# Variational Inference

## The Inference Pipeline

Variational inference estimates a posterior distribution given observations. The quivers pipeline is:

```
Model (MonadicProgram)
    ↓
Trace (record sample sites)
    ↓
Condition (clamp observations)
    ↓
Guide (variational family)
    ↓
ELBO (loss function)
    ↓
SVI (stochastic optimization)
    ↓
Predictive (sample from posterior)
```

## Trace and Sample Sites

A trace records all stochastic operations in a program. Each sample point is a `SampleSite`.

```python
from quivers.inference import trace, Trace, SampleSite

model = ...  # MonadicProgram

# Execute model with tracing
with trace() as tr:
    samples = model.rsample(x, n_samples=10)

# Access sites
sites = tr.sites  # dict[site_name -> SampleSite]

for name, site in sites.items():
    print(f"{name}: {site.log_prob}")
```

A `SampleSite` records:
- `name`: identifier of the sample
- `value`: sampled value
- `log_prob`: log probability of the sample
- `morphism`: the generating distribution

## Conditioning on Observations

The `condition()` function clamps observations, fixing certain variables:

```python
from quivers.inference import condition, Conditioned

model = ...  # MonadicProgram

# Observed values (e.g., from an experiment)
observations = {
    "y_1": torch.tensor(1.5),
    "y_2": torch.tensor(-0.3),
}

# Create conditioned model
conditioned = condition(model, observations)

# Forward pass uses clamped values
log_pjoint = conditioned.log_joint(x, y_obs)
```

The conditioned model is a `Conditioned` instance that wraps the original model and enforces observation constraints.

## Guides: Variational Families

A guide $q_\phi(z | x, y)$ is a variational family approximating the posterior. quivers provides automatic guide construction.

### AutoNormalGuide

A diagonal Gaussian approximation to the posterior:

$$q_\phi(z | x, y) = \prod_i \mathcal{N}(z_i | \mu_i(x, y), \sigma_i(x, y))$$

where $\mu$ and $\sigma$ are learned neural networks.

```python
from quivers.inference import AutoNormalGuide

model = ...  # MonadicProgram
conditioned = condition(model, observations)

guide = AutoNormalGuide(
    model=conditioned,
    hidden_dim=32,
    n_hidden=2,
)

# Sample from guide
z_guide = guide.rsample(x, y_obs, n_samples=100)

# Log probability under guide
log_q = guide.log_prob(x, y_obs, z_guide)
```

### AutoDeltaGuide

A delta (point mass) approximation, i.e. a single best estimate:

$$q_\phi(z | x, y) = \delta_{z^*_\phi(x, y)}(z)$$

```python
from quivers.inference import AutoDeltaGuide

guide = AutoDeltaGuide(
    model=conditioned,
    hidden_dim=64,
)

# Point estimate
z_map = guide.rsample(x, y_obs)  # deterministic

# Delta log probability (0 if equal, -∞ otherwise; clamped)
log_q = guide.log_prob(x, y_obs, z_map)
```

## ELBO: Evidence Lower Bound

The ELBO is the variational objective:

$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z | x, y)} [\log p(y, z | x) - \log q_\phi(z | x, y)]$$

It lower bounds the log marginal likelihood $\log p(y | x)$ and equals it when $q_\phi = p(\cdot | x, y)$.

The `ELBO` class computes this:

```python
from quivers.inference import ELBO

model = ...  # joint p
guide = ...  # variational q

elbo = ELBO(model=model, guide=guide)

# Compute loss
x = torch.randn(5)
y = torch.randn(3)
loss = elbo(x, y, n_samples=10)  # negative ELBO (for minimization)

loss.backward()  # backprop through both model and guide
```

Internally, the ELBO:
1. Samples latent variables $z \sim q_\phi(\cdot | x, y)$
2. Computes $\log p(y, z | x)$ via `model.log_joint()`
3. Computes $\log q_\phi(z | x, y)$ via `guide.log_prob()`
4. Returns $\frac{1}{n}\sum_i [\log q - \log p]$

## SVI: Stochastic Variational Inference

The SVI training loop optimizes both model and guide parameters:

```python
from quivers.inference import SVI
import torch.optim as optim

model = ...
guide = ...

svi = SVI(model=model, guide=guide)

optimizer = optim.Adam(
    list(model.parameters()) + list(guide.parameters()),
    lr=1e-3,
)

# Training loop
for epoch in range(100):
    x = next(data_loader)  # minibatch
    y = x[:, -1]  # observations
    x_input = x[:, :-1]

    loss = svi.step(x_input, y, n_samples=5, optimizer=optimizer)
    print(f"Epoch {epoch}: loss={loss:.4f}")
```

The `step` method:
1. Computes ELBO loss
2. Backpropagates gradients
3. Updates optimizer

## Predictive Sampling

After training, sample from the posterior predictive:

$$p(y_\text{new} | x_\text{new}, \text{observations}) = \int p(y_\text{new} | z, x_\text{new}) p(z | x, y_\text{obs}) dz$$

```python
from quivers.inference import Predictive

predictive = Predictive(
    model=conditioned,
    guide=guide,
    num_samples=1000,
)

# Sample from posterior predictive
x_new = torch.randn(5)
y_new_samples = predictive(x_new)  # shape (1000, 3)

# Posterior mean and credible intervals
y_mean = y_new_samples.mean(dim=0)
y_low = y_new_samples.quantile(0.025, dim=0)
y_high = y_new_samples.quantile(0.975, dim=0)
```

The predictive:
1. Samples latents from the guide: $z \sim q_\phi(\cdot | x, y_\text{obs})$
2. Samples outcomes: $y_\text{new} \sim p(\cdot | z, x_\text{new})$
3. Returns the ensemble

## Full Example: Bayesian Linear Regression

```python
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import Unit
from quivers.inference import (
    condition, AutoNormalGuide, ELBO, SVI, Predictive
)
import torch
import torch.optim as optim

# Model: y = w·x + noise
program = MonadicProgram(
    domain=Euclidean(1),
    codomain=Euclidean(1),
)

# Prior on weight
prior_w = ConditionalNormal(Unit, Euclidean(1))
program.add_morphism("prior_w", prior_w)

# Likelihood
likelihood = ConditionalNormal(Euclidean(1), Euclidean(1))
program.add_morphism("likelihood", likelihood)

# Steps
program.add_draw("w", "prior_w")
program.add_draw("y", "likelihood", args=("w",))
program.add_return("y")

# Observed data
x_obs = torch.randn(100, 1)
y_obs = 2.0 * x_obs + torch.randn(100, 1) * 0.1

# Condition on observations
conditioned = condition(program, {"y": y_obs})

# Variational guide
guide = AutoNormalGuide(
    model=conditioned,
    hidden_dim=16,
    n_hidden=1,
)

# Optimization
svi = SVI(model=conditioned, guide=guide)
optimizer = optim.Adam(
    list(conditioned.parameters()) + list(guide.parameters()),
    lr=1e-2,
)

for epoch in range(100):
    loss = svi.step(x_obs, y_obs, n_samples=10, optimizer=optimizer)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}")

# Posterior predictive on new data
x_new = torch.linspace(-3, 3, 50).view(-1, 1)
predictive = Predictive(model=conditioned, guide=guide, num_samples=500)
y_pred = predictive(x_new)

# Summarize
y_mean = y_pred.mean(dim=0)
y_std = y_pred.std(dim=0)

print(f"Posterior mean of w: {y_mean[0]:.2f} ± {y_std[0]:.2f}")
```

## Advanced: Custom Guides

Implement a custom guide by subclassing `Guide`:

```python
from quivers.inference.guide import Guide

class MyGuide(Guide):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.mu_net = torch.nn.Linear(5, 10)
        self.sigma_net = torch.nn.Linear(5, 10)

    def log_prob(self, x, y, z):
        """Compute log q(z | x, y)"""
        raise NotImplementedError()

    def rsample(self, x, y, n_samples=1):
        """Sample z ~ q(· | x, y)"""
        raise NotImplementedError()
```

## Debugging

Enable tracing to inspect sites and log probabilities:

```python
from quivers.inference import trace

with trace() as tr:
    samples = model.rsample(x, n_samples=1)

for name, site in tr.sites.items():
    print(f"{name}: log_prob={site.log_prob.item():.4f}")
```

Monitor the ELBO during training to detect divergence or poor guide fit.
