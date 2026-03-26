# Quickstart

This guide walks through the core concepts of quivers with concrete examples: creating objects, building morphisms, composing them, and running them as differentiable programs.

## 1. Basic Morphisms

A **morphism** is a $\mathcal{V}$-enriched relation: a function from a pair of objects to a quantale (lattice of truth values). In quivers, morphisms are tensors.

Create finite sets:

```python
from quivers import FinSet, morphism, observed
import torch

X = FinSet("X", 3)
Y = FinSet("Y", 4)
```

Create a **latent** (learnable) morphism from X to Y. Its tensor entries are parameters:

```python
f = morphism(X, Y)
print(f.tensor.shape)  # torch.Size([3, 4])
print(f.domain, f.codomain)  # X -> Y
```

Create an **observed** (fixed) morphism with a fixed tensor:

```python
data = torch.tensor([
    [1.0, 0.5, 0.0, 0.2],
    [0.0, 1.0, 0.8, 0.1],
    [0.3, 0.2, 1.0, 0.9],
])
g = observed(X, Y, data)
print(g.tensor)  # your data tensor
```

Get the underlying `nn.Module` for use in training:

```python
mod = f.module()
params = list(mod.parameters())
print(len(params))  # 1 parameter matrix
```

## 2. Composition

Compose morphisms with the `>>` operator. The domain of the second must match the codomain of the first:

```python
Z = FinSet("Z", 2)
h = morphism(Y, Z)

# Compose f: X -> Y and h: Y -> Z to get X -> Z
composed = f >> h
print(composed.tensor.shape)  # torch.Size([3, 2])
```

Composition is **lazy**: it builds a computation graph. The final tensor is only materialized on evaluation.

## 3. Programs

Wrap a morphism as a differentiable `nn.Module` with `Program`:

```python
from quivers import Program

program = Program(composed)
output = program()
print(output.shape)  # torch.Size([3, 2])
```

Integrate with a training loop:

```python
import torch.optim as optim

optimizer = optim.Adam(program.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    output = program()
    # Define a loss and backpropagate
    loss = output.sum()
    loss.backward()
    optimizer.step()
```

## 4. Monadic Programs (Continuous)

For probabilistic computations with continuous and discrete variables, use **monadic programs**:

```python
from quivers import (
    Euclidean, ConditionalNormal, MonadicProgram
)
import torch

# Define spaces
X = FinSet("X", 3)
R = Euclidean("position", 2)

# Create a conditional distribution: X -> Normal on R
mu_net = torch.nn.Linear(3, 2)
sigma_net = torch.nn.Sequential(
    torch.nn.Linear(3, 2),
    torch.nn.Softplus()
)

family = ConditionalNormal(mu_net, sigma_net)

# Create a monadic program (probabilistic computation)
prog = MonadicProgram(family)

# Sample from the program
x = torch.randn(1, 3)  # batch of inputs
sample = prog.rsample(x)  # reparameterized sample
log_p = prog.log_prob(x, sample)  # log probability
```

## 5. The DSL

Write categorical programs declaratively in `.qvr` files:

```qvr
object X : 3
object Y : 4
object Z : 2

latent f : X -> Y
latent g : Y -> Z
output f >> g
```

Load and run:

```python
from quivers import dsl_load, Program

prog = dsl_load("program.qvr")
output = prog()
```

Or use `loads` for inline strings:

```python
from quivers import dsl_loads

source = """
object X : 3
object Y : 4
object Z : 2
latent f : X -> Y
latent g : Y -> Z
output f >> g
"""

prog = dsl_loads(source)
output = prog()
print(output.shape)  # torch.Size([3, 2])
```

Supported DSL operators:

| Operator | Meaning | Example |
|----------|---------|---------|
| `>>` | composition | `f >> g` |
| `@` | tensor product | `f @ g` |
| `.marginalize(X)` | marginalize over object | `f.marginalize(X)` |
| `identity(X)` | identity morphism | `observed id : X -> X = identity(X)` |

## 6. Stochastic Morphisms

The **FinStoch** category models Markov kernels on finite sets:

```python
from quivers import (
    stochastic, FinSet, DiscretizedNormal,
    condition, ConditionedMorphism
)
import torch

X = FinSet("X", 3)
Y = FinSet("Y", 4)

# Create a stochastic morphism (Markov kernel)
kern = stochastic(X, Y)
print(kern.tensor.shape)  # [3, 4], columns sum to 1
print(kern.tensor.sum(dim=1))  # all 1.0

# Condition on an observation
obs_data = torch.tensor([0.1, 0.8, 0.05, 0.05])  # P(Y)
conditioned = condition(kern, obs_data)
print(conditioned.tensor.shape)  # [3, 4]
```

## 7. Enriched Structures

Work with more abstract categorical structures:

```python
from quivers import (
    FuzzyPowersetMonad, KleisliCategory, FinSet
)

# Create a monad
monad = FuzzyPowersetMonad()

# Work in its Kleisli category
kleisli = KleisliCategory(monad)

X = FinSet("X", 3)
Y = FinSet("Y", 4)

f = morphism(X, Y)
g = morphism(Y, X)

# Kleisli composition
comp = kleisli.compose(f, g)
print(comp.tensor.shape)  # [3, 3]
```

## Next Steps

- **[Architecture](architecture.md)** — learn the package structure and design principles.
- **[API Reference](../api/index.md)** — detailed documentation of all classes and functions.
- **Guides** — core types, morphisms, categorical structures, stochastic and continuous morphisms, the DSL, and variational inference.
- **Tutorials** — end-to-end examples including probabilistic programs and inference.
