# Tutorial 5: Variational Inference

In this tutorial, you will fit a probabilistic program to observed data using variational inference. You will set up a model, condition it on observations, create a variational guide, define an ELBO loss, run a training loop with SVI, and use Predictive for posterior sampling.

## Concepts

- **Trace**: Record of a program execution with sample sites and log-densities
- **Conditioning**: Fixing observed variables to particular values
- **Guide**: A variational approximation to the posterior distribution
- **ELBO**: Evidence Lower Bound, a loss function for variational inference
- **SVI**: Stochastic Variational Inference, a gradient-based optimization algorithm
- **Predictive**: Posterior sampling using the fitted guide

## Setup

```python
import torch
import torch.optim as optim
from quivers.core.objects import FinSet
from quivers.continuous.spaces import Euclidean
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.programs import MonadicProgram
from quivers.inference import (
    trace,
    condition,
    AutoNormalGuide,
    ELBO,
    SVI,
    Predictive,
)
```

## Building a Model

Create a simple generative model: latent variable z drives observation y.

```python
Unit = FinSet(name="Unit", cardinality=1)
R = Euclidean(name="real", dim=1)

prior = ConditionalNormal(Unit, R)
likelihood = ConditionalNormal(R, R)

model = MonadicProgram(
    Unit, R,
    steps=[
        (("z",), prior, None),           # z <- prior(unit)
        (("y",), likelihood, ("z",)),    # y <- likelihood(z)
    ],
    return_vars=("z", "y"),
)
```

## Simulating Observed Data

Generate synthetic observations from the model (as if from an experiment):

```python
torch.manual_seed(42)

# Sample from the prior
batch = torch.zeros(50, dtype=torch.long)  # 50 samples
samples = model.rsample(batch)

# Tuple returns come back as a dict keyed by variable name.
y_observed = samples["y"]
print(y_observed.shape)  # [50, 1]
print(y_observed.mean(), y_observed.std())
```

In practice, these observations come from real data. Here we simulate for illustration.

## Tracing the Model

A trace records the values and log-densities at each site (random variable):

```python
# Trace the model at a single point
x_single = torch.zeros(1, dtype=torch.long)
tr = trace(model, x_single)

# Inspect sites
print("Sites:", tr.sites.keys())  # {'z': SampleSite, 'y': SampleSite}

# Access a site
z_site = tr.sites["z"]
print("z value:", z_site.value)
print("z log_prob:", z_site.log_prob)
```

Each site has:

- `value`: The sampled value
- `log_prob`: The log-probability under the distribution
- `is_observed`: Whether this site is conditioned (fixed)
- `is_deterministic`: Whether it is a let binding

## Conditioning on Observations

Wrap the model to fix observed variables:

```python
# Fix y to observed values
conditioned_model = condition(model, {"y": y_observed})

# Trace the conditioned model
tr_cond = conditioned_model.trace(x_single)

# Now y is fixed
print(tr_cond.sites["y"].is_observed)  # True
print(tr_cond.sites["y"].value)         # matches y_observed[0]
print(tr_cond.sites["z"].is_observed)  # False (still latent)
```

The conditioned model still allows the latent variable z to vary.

## Creating a Guide

A guide is a variational approximation to the posterior. It has the same interface as the model but is typically simpler (e.g., a mean-field normal distribution).

```python
# Create a guide: assumes posterior over z and y is normal
guide = AutoNormalGuide(model, observed_names={"y"})
```

The `AutoNormalGuide`:

1. Identifies latent (non-observed) variables: just `z` in this case
2. Creates learnable parameters for the mean and log-scale of a normal distribution
3. Provides `rsample(x)` and `log_prob(x, samples)` methods

Sample from the guide:

```python
x = torch.zeros(4, dtype=torch.long)
posterior_samples = guide.rsample(x)
print(posterior_samples)  # dict: {"z": tensor, ...}

posterior_z = posterior_samples["z"]
print(posterior_z.shape)  # [4, 1]
```

Compute the guide's log-probability:

```python
log_q = guide.log_prob(x, posterior_samples)
print(log_q.shape)  # [4]
```

## Passing Observations at Runtime

Programs that use indexed observes (`observe r : N <- F(args)`) read their response tensors from a runtime `observations: dict[str, torch.Tensor]` keyed by the observed-variable name. The dict is forwarded to `MonadicProgram.rsample` (kwarg) and to `ELBO.forward` / `SVI.step` (positional, after the program input):

```python
observations = {"y": y_observed}            # shape matches the program's N
# Once SVI is set up (next section), pass observations into svi.step:
#     loss = svi.step(domain_input, observations)
```

There is no `.qvr`-level data block; observation tensors live in Python at the call site.

## Setting Up Inference

Define the ELBO loss and optimizer:

```python
elbo = ELBO(num_particles=1)

optimizer = optim.Adam(list(model.parameters()) + list(guide.parameters()), lr=0.01)

svi = SVI(model, guide, optimizer, elbo)
```

The [`SVI`](../../api/inference/svi.md) object pairs:

- `model`: the generative `MonadicProgram`. Observations are supplied to each step rather than baked in via [`condition`](../../api/inference/conditioning.md).
- `guide`: the variational posterior.
- `optim`: optimizer over the model and guide parameters.
- `objective`: an [`Objective`](../../api/inference/elbo.md) such as [`ELBO`](../../api/inference/elbo.md).

## Training Loop

Run inference to optimize the guide's parameters:

```python
num_steps = 1000
losses = []

# Prepare observed data
observations = {"y": y_observed}  # 50 observed values
batch_size = 10
n_batches = len(y_observed) // batch_size

for step in range(num_steps):
    # Shuffle and batch observations
    indices = torch.randperm(len(y_observed))
    batch_losses = []

    for i in range(n_batches):
        batch_idx = indices[i*batch_size:(i+1)*batch_size]
        batch_obs = {"y": y_observed[batch_idx]}

        # One SVI step: the observations dict flows through the
        # objective at evaluation time, so the model itself does not
        # need to be re-wrapped per batch.
        loss = svi.step(
            torch.zeros(batch_size, dtype=torch.long), batch_obs
        )

        batch_losses.append(loss)

    epoch_loss = sum(batch_losses) / len(batch_losses)
    losses.append(epoch_loss)

    if step % 100 == 0:
        print(f"Step {step}: Loss {epoch_loss:.4f}")

# Plot loss curve (optional)
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("ELBO Loss")
plt.show()
```

The ELBO loss combines:

1. **Model likelihood**: How well the model explains observations under the guide's samples
2. **KL divergence**: How close the guide is to the prior (regularization)

$$
\text{ELBO} = \mathbb{E}_q[\log p(\text{obs}, z)] - \text{KL}(q \| p)
$$

Minimizing ELBO maximizes the evidence log-likelihood and keeps the guide regularized.

## Posterior Inference

After training, use the guide to make predictions on new data:

```python
# Create a Predictive: posterior samples from the guide
predictive = Predictive(model, guide, num_samples=100)

# Sample posterior at a new point
x_new = torch.zeros(1, dtype=torch.long)
posterior_samples = predictive(x_new)

print(posterior_samples.keys())  # dict with z and y
z_posterior = posterior_samples["z"]
print(z_posterior.shape)  # [100, 1, 1] (num_samples, batch, dim)
```

Analyze the posterior:

```python
# Posterior mean and std
z_mean = z_posterior.mean(dim=0)
z_std = z_posterior.std(dim=0)
print(f"z posterior: mean={z_mean.item():.3f}, std={z_std.item():.3f}")

# Posterior credible interval
z_quantile_lower = z_posterior.quantile(0.025, dim=0)
z_quantile_upper = z_posterior.quantile(0.975, dim=0)
print(f"95% CI: [{z_quantile_lower.item():.3f}, {z_quantile_upper.item():.3f}]")
```

## Evaluating the Guide

Compare the learned guide to the true posterior. Sample from both:

```python
# True posterior: z conditioned on observed y
tr_true = conditioned_model.trace(x_new)
z_true = tr_true.sites["z"].value

# Posterior from guide
z_guide = guide.rsample(x_new)["z"]

print(f"True z: {z_true.item():.3f}")
print(f"Guide z: {z_guide.item():.3f}")
```

Visualize: plot the true posterior density vs. the guide's density (if tractable).

## More Complex Models

The same pattern extends to complex models. For instance, with the PDS model from Tutorial 4:

```python
from quivers.dsl import loads

prog_pds = loads("""
object Entity : 1
object Truth : 2
object Resp : 1

program factivity : Entity -> Truth * Truth * Truth * Resp
    theta_know <- LogitNormal(0.0, 1.0)
    theta_cg <- LogitNormal(0.0, 1.0)
    let cg_complement = 1
    tau_know <- Bernoulli(theta_know)
    cg_matrix <- Bernoulli(theta_cg)
    sigma <- Uniform(0.0, 1.0)
    observe response <- TruncatedNormal(theta_know, sigma, 0.0, 1.0)
    return (tau_know, cg_complement, cg_matrix, response)
""")
model_pds = prog_pds.morphism

# Observed response judgments from a linguistic experiment
observed_responses = torch.tensor([0.8, 0.6, 0.7, 0.9, 0.5])

# Build a guide and an SVI loop. Observations are routed through
# ``svi.step`` rather than wrapped into the model.
guide_pds = AutoNormalGuide(model_pds, observed_names={"response"})

elbo = ELBO(num_particles=1)
optimizer = optim.Adam(
    list(model_pds.parameters()) + list(guide_pds.parameters()),
    lr=0.01,
)
svi = SVI(model_pds, guide_pds, optimizer, elbo)

# Run training... pass {"response": observed_responses} as the
# observations dict on each step.
```

## Summary

You have:

- Built a probabilistic model and simulated observations
- Traced a model to inspect sites and log-probabilities
- Conditioned a model on observed data
- Created an AutoNormalGuide as a posterior approximation
- Set up and ran a variational inference training loop
- Used Predictive for posterior sampling
- Evaluated the inferred posterior

This workflow applies to any quivers probabilistic program, from simple Gaussian models to complex linguistic models like PDS.

## Next

[Tutorial 6](06-first-class-trans.md) covers first-class transformations: `MorphismTransformation` and `AlgebraHomomorphism` as values, the `>>>` composition operator, and change-of-base pipelines.

## Further Reading

- **[Inference Guide](../../guides/inference.md):** Detailed documentation of trace, conditioning, guides, ELBO, and SVI
- **[Continuous Morphisms](../../guides/continuous.md):** More on distributions and spaces
- **[DSL Guide](../../guides/dsl.md):** Writing models in `.qvr` syntax
