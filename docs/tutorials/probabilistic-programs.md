# Tutorial 3: Probabilistic Programs

In this tutorial, you will construct probabilistic programs that mix discrete and continuous random variables. You will build MonadicPrograms by hand using the Python API, create conditional distribution families, sample from programs, and compute log-densities.

## Concepts

- **Continuous spaces**: Euclidean, UnitInterval, Simplex, PositiveReals
- **Conditional distribution families**: Parameterized distributions (ConditionalNormal, ConditionalBernoulli, etc.)
- **MonadicProgram**: A probabilistic computation with draw and let steps
- **Draw step**: Sample a random variable from a distribution conditioned on prior variables
- **Let step**: Deterministic transformation of prior variables
- **Log-probability**: The log of the joint density

## Setup

```python
import torch
from quivers.core.objects import FinSet
from quivers.continuous.spaces import Euclidean, UnitInterval, Simplex
from quivers.continuous.families import ConditionalNormal, ConditionalBernoulli
from quivers.continuous.programs import MonadicProgram
```

## Defining Spaces

Continuous random variables live in continuous spaces. Create a few:

```python
# Euclidean space: R^2
R2 = Euclidean("position", 2)
print(R2.dim)  # 2
print(R2.event_shape)  # (2,)

# Unit interval: [0, 1]
U = UnitInterval("probability")
print(U.dim)  # 1

# Positive reals: (0, inf)
P = PositiveReals("variance", 1)

# Simplex: probability distributions over k categories
S = Simplex("mixture_weights", 3)
print(S.dim)  # 3
```

Each space has methods for sampling and containment checks:

```python
# Sample uniformly from a bounded space
samples = U.sample_uniform(100)
print(samples.shape)  # [100, 1]
assert (samples >= 0.0).all() and (samples <= 1.0).all()
```

## Conditional Distribution Families

A conditional distribution family maps an input space to a parameterized family of distributions. For example, `ConditionalNormal(input_space, output_space)` learns mean and scale as functions of the input.

Create a conditional normal from a discrete input (finite set) to a continuous output:

```python
Unit = FinSet("Unit", 1)
R = Euclidean("latent", 1)

prior_family = ConditionalNormal(Unit, R)
```

The family learns parameters (loc and scale) from the input. Sample from it:

```python
batch = torch.zeros(5, 1, dtype=torch.long)  # batch of 5 unit inputs
samples = prior_family.rsample(batch)
print(samples.shape)  # [5, 1]
log_probs = prior_family.log_prob(batch, samples)
print(log_probs.shape)  # [5]
```

Create a likelihood family from continuous to continuous:

```python
likelihood_family = ConditionalNormal(R, R)
```

Now the input is continuous. Call with an actual value:

```python
z_value = torch.randn(5, 1)  # 5 latent values
y_samples = likelihood_family.rsample(z_value)
print(y_samples.shape)  # [5, 1]
```

Other families available include ConditionalBernoulli, ConditionalBeta, ConditionalLaplace, and many more. They follow the same interface.

## Building a MonadicProgram

A MonadicProgram represents a probabilistic computation as a sequence of steps. Each step is either:

1. **Draw**: Sample a random variable from a conditional distribution
2. **Let**: Compute a deterministic transformation

Construct a simple two-stage program: draw z from a prior, then draw y from a likelihood conditioned on z.

```python
Unit = FinSet("Unit", 1)
R = Euclidean("real", 1)

prior = ConditionalNormal(Unit, R)
likelihood = ConditionalNormal(R, R)

program = MonadicProgram(
    Unit, R,  # input space, output space
    steps=[
        (("z",), prior, None),          # draw z ~ prior(unit)
        (("y",), likelihood, ("z",)),   # draw y ~ likelihood(z)
    ],
    return_vars=("y",),
)
```

The tuple structure:

- `(var_names,)`: Names of variables drawn in this step (a tuple, even if one variable)
- `family`: The conditional distribution, or `None` for a let binding
- `input_vars`: Names of prior variables this step depends on, or `None` for a prior

Sample from the program:

```python
batch = torch.zeros(10, 1, dtype=torch.long)  # 10 samples
output = program.rsample(batch)
print(output.shape)  # [10, 1]
```

Compute the log-joint density:

```python
log_joint = program.log_prob(batch, output)
print(log_joint.shape)  # [10]
```

## Let Bindings

Add a deterministic transformation step. For example, compute w = z^2:

```python
Unit = FinSet("Unit", 1)
R = Euclidean("real", 1)

prior = ConditionalNormal(Unit, R)
likelihood = ConditionalNormal(R, R)

# A let binding: deterministic transformation
def square(env):
    """Compute z^2. env is a dict of prior variables."""
    return env["z"] ** 2

program = MonadicProgram(
    Unit, R,
    steps=[
        (("z",), prior, None),
        (("w",), None, square),           # let w = z^2
        (("y",), likelihood, ("w",)),    # draw y ~ likelihood(w)
    ],
    return_vars=("y",),
)
```

The let binding receives an environment dict with all prior variables:

```python
batch = torch.zeros(5, 1, dtype=torch.long)
output = program.rsample(batch)
# Internally, w was computed as z^2, then y was sampled
```

## Multi-variable Outputs

Return multiple variables:

```python
program = MonadicProgram(
    Unit, R * R,  # output is R^2
    steps=[
        (("z",), prior, None),
        (("y",), likelihood, ("z",)),
    ],
    return_vars=("z", "y"),  # return both
)

samples = program.rsample(batch)
print(samples.shape)  # [10, 2] (concatenated z and y)
```

## Conditional Bernoulli

Use discrete outputs. Create a conditional Bernoulli family:

```python
from quivers.continuous.families import ConditionalBernoulli

R = Euclidean("latent", 2)
Coin = FinSet("Coin", 1)

bernoulli = ConditionalBernoulli(R, Coin)
```

This learns logits as a function of the continuous input, then samples binary values:

```python
z = torch.randn(5, 2)
samples = bernoulli.rsample(z)
print(samples.shape)  # [5, 1]
print((samples == 0) | (samples == 1))  # all True: binary values
```

## Forward and Backward

Once wrapped in a PyTorch nn.Module, the program participates in training:

```python
from quivers.program import Program

prog = Program(program)
optimizer = torch.optim.Adam(prog.parameters(), lr=0.01)

batch = torch.zeros(32, 1, dtype=torch.long)
target = torch.randn(32, 1)

for epoch in range(10):
    optimizer.zero_grad()
    output = prog.rsample(batch)
    loss = ((output - target) ** 2).mean()
    loss.backward()
    optimizer.step()
```

## Complex Programs

Build larger programs with multiple stages and branches. Here is a model for two observations with shared latent:

```python
Unit = FinSet("Unit", 1)
R = Euclidean("space", 1)

# Shared prior
prior = ConditionalNormal(Unit, R)

# Two likelihoods
likelihood1 = ConditionalNormal(R, R)
likelihood2 = ConditionalNormal(R, R)

program = MonadicProgram(
    Unit, R * R,
    steps=[
        (("z",), prior, None),                  # shared latent
        (("y1",), likelihood1, ("z",)),         # observation 1
        (("y2",), likelihood2, ("z",)),         # observation 2
    ],
    return_vars=("y1", "y2"),
)

batch = torch.zeros(20, 1, dtype=torch.long)
samples = program.rsample(batch)
print(samples.shape)  # [20, 2]
```

## Summary

You have:

- Created continuous spaces
- Defined conditional distribution families
- Built MonadicPrograms with draw and let steps
- Sampled from programs and computed log-densities
- Used ConditionalNormal and ConditionalBernoulli
- Integrated programs with PyTorch optimization

Next, see how these concepts apply to linguistic modeling in [Tutorial 4](pds-model.md).
