# Variational Inference: Guides, Objectives, and SVI

This page covers the variational layer: the `Auto*Guide` family,
the four objectives and four gradient estimators, the SVI training
loop, and predictive sampling from a trained guide. The MCMC layer
(HMC, NUTS, hybrid samplers) lives in [the MCMC
guide](inference-mcmc.md); the trace / conditioning / registry
primitives that every guide consumes live in
[Inference Foundations](inference-foundations.md).

## Guides: variational families

A guide $q_\phi(z \mid x, y)$ is a variational family approximating
the posterior. Nine `Auto*Guide` subclasses cover the standard
zoo, all documented under [Variational
Guides](../api/inference/guide.md); each is constructed from the
model and a set of observed site names.

| Guide | Posterior structure | When to use |
|---|---|---|
| [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) | Diagonal Normal (mean-field) | Default; identifiable posterior, weak correlation |
| [`AutoMultivariateNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoMultivariateNormalGuide) | Full-rank Normal (Cholesky) | Strong posterior correlations; $D \lesssim 1000$ |
| [`AutoLowRankMultivariateNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoLowRankMultivariateNormalGuide) | Low-rank + diagonal | Hierarchical models with localized correlations |
| [`AutoLaplaceApproximation`](../api/inference/guide.md#quivers.inference.guides.AutoLaplaceApproximation) | Gaussian centered at MAP w/ Hessian inverse | Post-hoc; cheap quadratic-around-MAP |
| [`AutoNormalizingFlow`](../api/inference/guide.md#quivers.inference.guides.AutoNormalizingFlow) | Composed bijector over Normal base | Multimodal / heavy-tailed posteriors |
| [`AutoIAFGuide`](../api/inference/guide.md#quivers.inference.guides.AutoIAFGuide) | [Inverse autoregressive flow](https://doi.org/10.48550/arXiv.1606.04934) | Flagship NF default |
| [`AutoNeuralSplineGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNeuralSplineGuide) | Rational-quadratic spline coupling ([Durkan et al. 2019](https://doi.org/10.48550/arXiv.1906.04032)) | Sharper than IAF for bounded support |
| [`AutoMixtureGuide`](../api/inference/guide.md#quivers.inference.guides.AutoMixtureGuide) | K-component mixture of guides | Multimodal posteriors |
| [`AutoDeltaGuide`](../api/inference/guide.md#quivers.inference.guides.AutoDeltaGuide) | Dirac at MAP | Quick MAP; no uncertainty |

Every guide uses `biject_to(support)` per site, so samples always
lie inside the prior's constrained support; `log_prob` carries the
corresponding [log-det
Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant).

### AutoNormalGuide

A diagonal Gaussian approximation to the posterior, with a per-site
bijector that maps unconstrained Normal samples to the prior's
constrained support:

$$q_\phi(z_i \mid x, y) = T_i\bigl(\,\mathcal{N}(\mu_i, \sigma_i)\,\bigr)$$

where $T_i = \mathsf{biject\_to}(\mathrm{support}(p_i))$ is the
bijector for site $i$'s prior support: the identity on the real
line for `Normal`, $\exp$ for `HalfNormal` / `Gamma` /
`Exponential` / `LogNormal`, sigmoid for `Beta` / `Uniform(0, 1)`
/ `LogitNormal`, an affine-shifted sigmoid for arbitrary
`Uniform(low, high)` / `TruncatedNormal`, and
[`StickBreakingTransform`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.StickBreakingTransform)
for `Dirichlet`. The learnable parameters $(\mu_i, \sigma_i)$ live
in the unconstrained space; the constrained sample $v_i = T_i(z_i)$
is always inside the prior's support, so `prior.log_prob(v_i)`
evaluates without raising `Expected value to be within the support
of ...`.

<!-- python: skip -->
```python
from quivers.inference import AutoNormalGuide

model = ...  # MonadicProgram
conditioned = condition(model, observations)

guide = AutoNormalGuide(conditioned.model, observed_names=set(observations))

# Sample latents from guide (each lives in its prior's support)
latents = guide.rsample(x)  # dict {name: tensor}

# Log probability under guide (with Jacobian correction)
log_q = guide.log_prob(x, latents)
```

### AutoDeltaGuide

A delta (point mass) approximation, i.e. a single best estimate.
The point lives in the unconstrained space and is pushed through
`biject_to(support)` at evaluation time, so it always lies inside
the prior's support:

$$q_\phi(z_i \mid x, y) = \delta_{T_i(\zeta_i)}(z_i)$$

where $\zeta_i$ is the learnable unconstrained point and $T_i$ the
same per-site bijector as `AutoNormalGuide`.

<!-- python: skip -->
```python
from quivers.inference import AutoDeltaGuide

guide = AutoDeltaGuide(conditioned.model, observed_names=set(observations))

# Point estimate (deterministic, inside the prior's support)
z_map = guide.rsample(x)

# Delta log probability (zero; the delta term cancels in the ELBO)
log_q = guide.log_prob(x, z_map)
```

## Objectives and gradient estimators

[`SVI`](../api/inference/svi.md) accepts any
[`Objective`](../api/inference/elbo.md) subclass, not just
[`ELBO`](../api/inference/elbo.md#quivers.inference.objectives.ELBO).
Four are shipped:

| Objective | Bound | Use case |
|---|---|---|
| `ELBO(num_particles=K)` | $\mathbb{E}_q[\log p - \log q]$ | Default |
| `IWAEBound(K, estimator=...)` | $\mathbb{E}[\log \tfrac{1}{K}\sum_k (p/q)_k]$ | Tighter than ELBO for $K > 1$ ([Burda et al. 2016](https://doi.org/10.48550/arXiv.1509.00519)) |
| `RenyiBound(alpha, K)` | $\alpha$-divergence bound ([Li-Turner 2016](https://doi.org/10.48550/arXiv.1602.02311)) | $\alpha = 0$ recovers IWAE; $\alpha = 1$ recovers ELBO |
| `VRIWAEBound(alpha, K)` | Variational Rényi-IWAE ([Daudel et al. 2023](https://doi.org/10.48550/arXiv.2210.06226)) | Interpolates cheap-vs-tight regimes |

Each accepts an `estimator=` strategy:

| Estimator | What it does |
|---|---|
| [`Reparameterized`](../api/inference/estimators.md#quivers.inference.estimators.Reparameterized) | Standard pathwise gradient (default) |
| [`StickingTheLanding`](../api/inference/estimators.md#quivers.inference.estimators.StickingTheLanding) | Detaches variational params in `log_q` ([Roeder et al. 2017](https://doi.org/10.48550/arXiv.1703.09194)); variance to 0 as $q \to p^*$ |
| [`DoublyReparameterized`](../api/inference/estimators.md#quivers.inference.estimators.DoublyReparameterized) | DReG for IWAE ([Tucker et al. 2019](https://doi.org/10.48550/arXiv.1810.04152)); kills the K-growing score term |
| [`ScoreFunction`](../api/inference/estimators.md#quivers.inference.estimators.ScoreFunction) | REINFORCE; for non-reparameterizable sites |

<!-- python: skip -->
```python
from quivers.inference import IWAEBound, DoublyReparameterized

iwae = IWAEBound(num_particles=16, estimator=DoublyReparameterized())
loss = iwae(model, guide, x, observations)
```

The Monte Carlo particle dimension is broadcast as a leading torch
axis end-to-end; the inner `model.log_joint` evaluation is a single
fused call against a `(K, batch, ...)`-shaped latent dict.

## ELBO: evidence lower bound

The [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound) is
the variational objective:

$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z \mid x, y)} [\log p(y, z \mid x) - \log q_\phi(z \mid x, y)]$$

It lower bounds the log marginal likelihood $\log p(y \mid x)$ and
equals it when $q_\phi = p(\cdot \mid x, y)$.

Indexed-observe steps (`observe r : N <- F(args)`) read their
response tensors from a runtime
`observations: dict[str, torch.Tensor]` keyed by the
observed-variable name. The dict is threaded through
[`ELBO.forward`](../api/inference/elbo.md#quivers.inference.objectives.ELBO)
and
[`SVI.step`](../api/inference/svi.md#quivers.inference.svi.SVI.step)
via the `observations` kwarg, alongside the domain input.

The `ELBO` class computes:

<!-- python: skip -->
```python
from quivers.inference import ELBO

model = ...  # MonadicProgram (joint p)
guide = ...  # variational q

elbo = ELBO(num_particles=10)

# Compute loss
x = torch.randn(5)
observations = {"y": y_obs}
loss = elbo(model, guide, x, observations)  # negative ELBO (for minimization)

loss.backward()  # backprop through both model and guide
```

Internally, the ELBO:

1. Samples latent variables $z \sim q_\phi(\cdot \mid x, y)$.
2. Computes $\log p(y, z \mid x)$ via
   [`model.log_joint()`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint).
3. Computes $\log q_\phi(z \mid x, y)$ via `guide.log_prob()`.
4. Returns $\frac{1}{n}\sum_i [\log q - \log p]$.

## SVI: stochastic variational inference

The
[`SVI`](../api/inference/svi.md#quivers.inference.svi.SVI) training
loop optimizes both model and guide parameters:

<!-- python: skip -->
```python
from quivers.inference import ELBO, SVI
import torch.optim as optim

model = ...   # MonadicProgram
guide = ...   # Guide
elbo  = ELBO(num_particles=5)

optimizer = optim.Adam(
    list(model.parameters()) + list(guide.parameters()),
    lr=1e-3,
)

svi = SVI(model, guide, optimizer, elbo)

# Training loop
for epoch in range(100):
    x = next(data_loader)  # minibatch
    observations = {"y": x[:, -1]}
    x_input = x[:, :-1]

    loss = svi.step(x_input, observations)
    print(f"Epoch {epoch}: loss={loss:.4f}")
```

The `step` method computes the ELBO loss, backpropagates gradients,
and steps the optimizer.

## Predictive sampling

After training, sample from the posterior predictive:

$$p(y_\text{new} \mid x_\text{new}, \text{observations}) = \int p(y_\text{new} \mid z, x_\text{new}) \, p(z \mid x, y_\text{obs}) \, dz$$

<!-- python: skip -->
```python
from quivers.inference import Predictive

predictive = Predictive(
    model=conditioned.model,
    posterior=guide,
    num_samples=1000,
)

# Sample from posterior predictive
x_new = torch.randn(5)
samples = predictive(x_new)              # dict[str, torch.Tensor]
y_new_samples = samples["y"]             # shape (num_samples, batch, ...)

# Posterior mean and credible intervals
y_mean = y_new_samples.mean(dim=0)
y_low = y_new_samples.quantile(0.025, dim=0)
y_high = y_new_samples.quantile(0.975, dim=0)
```

The [`Predictive`](../api/inference/predictive.md) driver:

1. Samples latents from the guide: $z \sim q_\phi(\cdot \mid x, y_\text{obs})$.
2. Samples outcomes: $y_\text{new} \sim p(\cdot \mid z, x_\text{new})$.
3. Returns the ensemble.

## Full example: Bayesian linear regression

<!-- python: skip -->
```python
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet
from quivers.inference import (
    condition, AutoNormalGuide, ELBO, SVI, Predictive
)
import torch
import torch.optim as optim

# Model: y = w.x + noise
Unit = FinSet(name="Unit", cardinality=1)
R1 = Euclidean(name="R1", dim=1)

prior_w    = ConditionalNormal(Unit, R1)
likelihood = ConditionalNormal(R1, R1)

program = MonadicProgram(
    R1,
    R1,
    steps=[
        (("w",), prior_w, None),
        (("y",), likelihood, ("w",)),
    ],
    return_vars=("y",),
)

# Observed data
x_obs = torch.randn(100, 1)
y_obs = 2.0 * x_obs + torch.randn(100, 1) * 0.1
observations = {"y": y_obs}

# Variational guide built against the unconditioned program
guide = AutoNormalGuide(program, observed_names={"y"})
elbo  = ELBO(num_particles=10)

# Optimization
optimizer = optim.Adam(
    list(program.parameters()) + list(guide.parameters()),
    lr=1e-2,
)
svi = SVI(program, guide, optimizer, elbo)

for epoch in range(100):
    loss = svi.step(x_obs, observations)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}")

# Posterior predictive on new data
x_new = torch.linspace(-3, 3, 50).view(-1, 1)
predictive = Predictive(model=program, posterior=guide, num_samples=500)
samples = predictive(x_new)
y_pred = samples["y"]

# Summarize
y_mean = y_pred.mean(dim=0)
y_std = y_pred.std(dim=0)

print(f"Posterior mean of w: {y_mean[0, 0]:.2f} +/- {y_std[0, 0]:.2f}")
```

## Custom guides

Implement a custom guide by subclassing
[`Guide`](../api/inference/guide.md#quivers.inference.guides.Guide):

<!-- python: skip -->
```python
from quivers.inference.guides import Guide

class MyGuide(Guide):
    def __init__(self, model):
        super().__init__()
        self.mu_net = torch.nn.Linear(5, 10)
        self.sigma_net = torch.nn.Linear(5, 10)

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample latent sites z ~ q(. | x). Returns {site_name: tensor}."""
        raise NotImplementedError()

    def log_prob(
        self, x: torch.Tensor, sites: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute log q(sites | x), summed across latent sites."""
        raise NotImplementedError()
```


## References

- Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. 2019. Neural spline flows. arXiv preprint arXiv:1906.04032.
- Geoffrey Roeder, Yuhuai Wu, and David Duvenaud. 2017. Sticking the landing: Simple, lower-variance gradient estimators for variational inference. arXiv preprint arXiv:1703.09194.
- George Tucker, Dieterich Lawson, Shixiang Gu, and Chris J. Maddison. 2019. Doubly reparameterized gradient estimators for Monte Carlo objectives. arXiv preprint arXiv:1810.04152.
- Kamélia Daudel, Joe Benton, Yuyang Shi, and Arnaud Doucet. 2023. Alpha-divergence variational inference meets importance weighted auto-encoders: Methodology and asymptotics. arXiv preprint arXiv:2210.06226.
- Yingzhen Li and Richard E. Turner. 2016. Rényi divergence variational inference. arXiv preprint arXiv:1602.02311.
- Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. 2016. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519.
