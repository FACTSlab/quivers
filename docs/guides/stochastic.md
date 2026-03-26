# Stochastic Morphisms

## The FinStoch Category

**FinStoch** is the category of finite sets and stochastic maps (Markov kernels):

- **Objects**: finite sets (as in the base category)
- **Morphisms** $A \to B$: stochastic matrices of shape $(|A|, |B|)$ with rows summing to 1
- **Composition**: standard matrix multiplication
- **Identity**: Kronecker delta

FinStoch is the Kleisli category of the Giry monad, restricted to finite sets.

```python
from quivers.stochastic import StochasticMorphism, FinStoch
from quivers.core.objects import FinSet
import torch

X = FinSet("X", 3)
Y = FinSet("Y", 4)

# Create a stochastic morphism (Markov kernel)
f = StochasticMorphism(X, Y)

# Check that rows sum to 1
assert (f.tensor.sum(dim=-1) - 1.0).abs().max() < 1e-5

# Composition preserves row-stochasticity
g = StochasticMorphism(Y, FinSet("Z", 5))
h = f >> g
assert (h.tensor.sum(dim=-1) - 1.0).abs().max() < 1e-5
```

## MarkovQuantale

The enrichment for FinStoch uses the Markov quantale:

$$
\begin{align}
\mathcal{L} &= [0, 1] \\
a \otimes b &= a \cdot b \quad \text{(pointwise product)} \\
\bigvee_i x_i &= \sum_i x_i \quad \text{(sum)} \\
\bigwedge_i x_i &= \min_i x_i \\
I &= 1, \quad \perp = 0
\end{align}
$$

Composition of stochastic matrices uses this structure:

$$(g \circ f)[a, c] = \sum_b f[a, b] \cdot g[b, c]$$

which is standard matrix multiplication.

```python
from quivers.stochastic.quantale import MARKOV

assert MARKOV.name == "Markov"

# Composition uses sum-product (matrix mult)
f_tensor = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
g_tensor = torch.tensor([[0.5, 0.5], [0.8, 0.2]])

composed = MARKOV.compose(f_tensor, g_tensor, n_contract=1)
expected = f_tensor @ g_tensor
assert torch.allclose(composed, expected)
```

## Stochastic Morphisms

A stochastic morphism is a learnable Markov kernel parameterized by an unconstrained weight matrix and a softmax normalization:

```python
from quivers.stochastic import StochasticMorphism
from quivers.core.objects import FinSet

X = FinSet("X", 4)
Y = FinSet("Y", 6)

f = StochasticMorphism(X, Y)

# Tensor is softmax applied to raw weights
# f.tensor has shape (4, 6) with rows summing to 1
assert f.tensor.shape == (4, 6)
assert (f.tensor.sum(dim=-1) - 1.0).abs().max() < 1e-5

# Gradient descent updates via raw weights
module = f.module()
params = list(module.parameters())
assert len(params) == 1  # one weight matrix
```

An alias `CategoricalMorphism` is provided with explicit categorical semantics.

## Discretized Distribution Families

When the codomain is continuous (not a finite set), quivers provides discretized versions of standard distributions. These map a finite input set to a bucketed approximation of the distribution.

```python
from quivers.stochastic.families import (
    DiscretizedNormal,
    DiscretizedLogitNormal,
    DiscretizedBeta,
    DiscretizedTruncatedNormal,
)
from quivers.core.objects import FinSet

X = FinSet("X", 10)

# Discretized normal on 20 bins
normal = DiscretizedNormal(X, n_bins=20)

# Input-conditioned parameters via learned linear maps
# μ, σ computed from input
tensor = normal.tensor  # (10, 20) stochastic matrix

# Discretized beta on (0, 1)
beta = DiscretizedBeta(X, n_bins=15, low=0.0, high=1.0)
```

## Conditioning and Bayesian Updates

Bayesian conditioning via Bayes' rule. Given a joint distribution $p(x, y)$ and observation $\text{obs}(y)$, compute $p(x | \text{obs}(y))$:

$$p(x | \text{obs}) \propto p(x, \text{obs}) = \sum_y p(x, y) \cdot \mathbb{1}_{\text{obs}}(y)$$

```python
from quivers.stochastic import condition, ConditionedMorphism
from quivers.core.morphisms import observed

X = FinSet("X", 3)
Y = FinSet("Y", 4)

# Joint distribution
joint = StochasticMorphism(X * Y, X + Y)

# Observation: Y = 1 (one-hot vector)
obs = torch.zeros(4)
obs[1] = 1.0

# Conditioned on obs
conditioned = condition(joint, Y, obs)
assert isinstance(conditioned, ConditionedMorphism)
assert (conditioned.tensor.sum(dim=-1) - 1.0).abs().max() < 1e-5
```

## Mixture Morphisms

Convex combination of stochastic morphisms:

```python
from quivers.stochastic import mix, MixtureMorphism

f = StochasticMorphism(X, Y)
g = StochasticMorphism(X, Y)

# Mix with weight 0.3
mixture = mix(f, g, weight=0.3)
assert isinstance(mixture, MixtureMorphism)

# Tensor is 0.3 * f + 0.7 * g
expected = 0.3 * f.tensor + 0.7 * g.tensor
assert torch.allclose(mixture.tensor, expected)
```

## Factored Morphisms

Pointwise multiplication (element-wise product):

```python
from quivers.stochastic import factor, FactoredMorphism

f = StochasticMorphism(X, Y)
g = StochasticMorphism(X, Y)

# Pointwise product (likelihood reweighting)
factored = factor(f, g)
assert isinstance(factored, FactoredMorphism)

# Note: result may not be stochastic; use normalize() after
```

## Normalization

Renormalize rows to sum to 1:

```python
from quivers.stochastic import normalize, NormalizedMorphism

unnormalized = ...  # some tensor that doesn't sum to 1
normalized = normalize(unnormalized)
assert (normalized.tensor.sum(dim=-1) - 1.0).abs().max() < 1e-5
```

## Queries: Probability and Expectation

Extract probabilities from stochastic morphisms:

```python
from quivers.stochastic import prob, marginal_prob, expectation

f = StochasticMorphism(X, Y)

# Query: P(x=0, y=2)
p = prob(f, domain_idx=0, codomain_idx=2)
assert 0 <= p <= 1

# Marginalize domain and query codomain
# If f: X × Z → Y, marginalize over X
marginal = marginal_prob(f, domain_component=X)

# Expectation of codomain index under morphism
expect = expectation(f, domain_idx=1)
```

## The Giry Monad and FinStoch

The Giry monad $\mathcal{G}$ sends a space $X$ to its space of probability measures. On finite sets, this becomes:

$$\mathcal{G}(X) = \{\text{probability distributions on } X\}$$

The Kleisli category of $\mathcal{G}$ restricted to finite sets **is** FinStoch.

```python
from quivers.stochastic.giry import GiryMonad, FinStoch

giry = GiryMonad()

# Unit: embed element into point mass
X = FinSet("X", 3)
unit_X = giry.unit(X)  # FinSet("X", 3) → FinSet("X", 3)

# Map (lift functor)
f = morphism(X, Y)
mapped = giry.map(f)

# FinStoch category
fs = FinStoch()
kernel = fs.morphism(X, Y)  # stochastic morphism
```

Stochastic morphisms are composed exactly as in FinStoch (matrix multiplication), not via Kleisli composition. They form a category in their own right.
