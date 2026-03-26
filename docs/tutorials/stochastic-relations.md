# Tutorial 2: Stochastic Relations

In this tutorial, you will work with the FinStoch category: Markov kernels on finite sets. These are stochastic morphisms whose entries represent conditional probabilities. You will compose kernels, condition on observations, and compute marginal probabilities and expectations.

## Concepts

- **Markov Quantale**: The quantale where tensor product is multiplication, join is summation
- **StochasticMorphism**: A morphism in FinStoch with row-stochastic tensor (columns sum to 1)
- **DiscretizedFamily**: Continuous distribution discretized into finite bins
- **Conditioning**: Updating a kernel given an observation
- **Marginal and expectation**: Aggregating probabilities and computing expectations

## Setup

```python
import torch
from quivers.core.objects import FinSet
from quivers.core.morphisms import identity, observed
from quivers.stochastic import (
    MARKOV,
    stochastic,
    DiscretizedNormal,
    DiscretizedBeta,
    condition,
    prob,
    marginal_prob,
    expectation,
)
from quivers.program import Program
```

## Creating Stochastic Morphisms

The FinStoch category uses the Markov quantale: composition is Markov kernel composition (matrix multiplication), and the join is summation.

Create two finite sets:

```python
X = FinSet("Latent", 3)
Y = FinSet("Observed", 4)
```

Create a latent (learnable) stochastic morphism from X to Y:

```python
kern = stochastic(X, Y)
print(kern.tensor.shape)  # [3, 4]
```

Verify it is row-stochastic (each row sums to 1):

```python
row_sums = kern.tensor.sum(dim=1)
print(row_sums)  # tensor([1., 1., 1.])
```

The entries represent conditional probabilities:

```python
print(kern.tensor)
# Each row: P(y | x) for a fixed x
```

The morphism is learnable; parameters are adjusted via softmax normalization internally.

Create an observed stochastic morphism from Y to a third set Z:

```python
Z = FinSet("Output", 2)
data = torch.tensor([
    [0.7, 0.3],
    [0.4, 0.6],
    [0.5, 0.5],
    [0.2, 0.8],
])
kern_obs = observed(Y, Z, data)
```

Verify the data is valid:

```python
print(kern_obs.tensor.sum(dim=1))  # all 1.0
```

## Composition of Markov Kernels

Compose stochastic morphisms with `>>`:

```python
composed = kern >> kern_obs
print(composed.tensor.shape)  # [3, 2]
print(composed.tensor.sum(dim=1))  # all 1.0
```

The composition is standard Markov kernel composition (matrix multiplication):

$$
(\kappa_2 \circ \kappa_1)(x, z) = \sum_y \kappa_1(x, y) \cdot \kappa_2(y, z)
$$

Wrap in a Program for training:

```python
program = Program(composed)
output = program()
print(output.shape)  # [3, 2]
```

## Discretized Distributions

For models mixing discrete and continuous variables, discretize continuous distributions into finite bins.

Create a DiscretizedNormal: a normal distribution binned into 10 intervals:

```python
Z_bin = FinSet("DiscretizedValue", 10)
Unit = FinSet("Unit", 1)

# Parameters
loc = 0.5
scale = 0.2

disc_normal = DiscretizedNormal(Unit, Z_bin, loc=loc, scale=scale)
print(disc_normal.tensor.shape)  # [1, 10]
print(disc_normal.tensor.sum(dim=1))  # [1.0]
```

This is a stochastic morphism from the terminal object to a discretized space. Each entry is the probability mass in a bin.

Similarly, create a DiscretizedBeta:

```python
disc_beta = DiscretizedBeta(Unit, Z_bin, alpha=2.0, beta=5.0)
```

These discretized morphisms compose with other stochastic morphisms:

```python
# Map from Unit -> DiscretizedValue -> SomeOutput
combined = disc_normal >> kern_to_output
```

## Observations and Conditioning

Given an observed value (a probability distribution over Y), update a kernel to compute the posterior:

```python
obs_dist = torch.tensor([0.1, 0.4, 0.3, 0.2])  # P(Y)
print(obs_dist.sum())  # 1.0
```

Condition the kernel kern (X -> Y) on this observation:

```python
conditioned = condition(kern, obs_dist)
print(conditioned.tensor.shape)  # [3, 4]
```

Conditioning uses Bayes' rule:

$$
P(\text{domain} \mid \text{observation}) = \frac{P(\text{observation} \mid \text{domain}) \cdot P(\text{domain})}{P(\text{observation})}
$$

This is useful for inverse inference: given an observation, what is the likely latent cause?

## Probabilities and Expectations

Compute marginal probabilities. Start with a prior distribution over X:

```python
prior = torch.tensor([0.5, 0.3, 0.2])  # P(X)
print(prior.sum())  # 1.0
```

Push the prior forward through the kernel:

```python
joint_prob = prob(kern, prior)
print(joint_prob)  # P(X, Y) as a tensor
```

Marginalize over Y to recover P(X):

```python
marginal_x = marginal_prob(joint_prob, n_dom_dims=1, marg_dims=[1])
print(marginal_x.shape)  # [3]
torch.testing.assert_close(marginal_x, prior, atol=1e-5, rtol=0.0)
```

Compute an expectation. Define a value function on Y (e.g., utilities):

```python
values = torch.tensor([10.0, 5.0, 1.0, 0.5])  # values for each outcome
```

Expect the value when prior is pushed through kern:

```python
expected_value = expectation(prob(kern, prior), values, n_dom_dims=1)
print(expected_value.item())  # scalar
```

## Composing with Observations

Chain observations and inference:

```python
# Start with prior over X
prior_x = torch.tensor([0.4, 0.35, 0.25])

# Observe first stage: get joint distribution X, Y
joint_xy = prob(kern, prior_x)

# Condition on Y being in a particular state
obs_y = torch.zeros(4)
obs_y[2] = 1.0  # Certain observation: y = 2

# Posterior over X given observation
posterior_x = marginal_prob(
    prob(kern, prior_x),  # joint X, Y
    n_dom_dims=1,
    marg_dims=[1]
)
print(posterior_x.shape)  # [3]
```

## Multiple Stages

Build a longer chain:

```python
# X -> Y -> Z
kern1 = stochastic(X, Y)
kern2 = stochastic(Y, Z)
composed = kern1 >> kern2

# Prior on X
prior = torch.tensor([0.5, 0.3, 0.2])

# Joint distribution X, Z
joint_xz = prob(composed, prior)
print(joint_xz.shape)  # [3, 2]
```

## Summary

You have:

- Created stochastic morphisms (Markov kernels)
- Verified row-stochasticity
- Composed kernels with `>>`
- Used DiscretizedNormal and DiscretizedBeta to mix discrete and continuous randomness
- Applied conditioning to update kernels given observations
- Computed marginal probabilities and expectations
- Built multi-stage inference chains

Next, learn how to work with continuous spaces and probabilistic programs in [Tutorial 3](probabilistic-programs.md).
