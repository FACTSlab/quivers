# Tutorial 1: Your First Quiver

In this tutorial, you will create a simple enriched category and work with morphisms as tensors. A quiver in this context is a directed graph where edges carry values in a lattice (quantale) rather than being abstract. When the quantale is $[0, 1]$ with product t-norm and noisy-OR, morphisms are fuzzy relations: functions from pairs of objects to truth values in $[0, 1]$.

## Concepts

- **Objects**: finite sets (FinSet)
- **Morphisms**: $\mathcal{V}$-relations represented as tensors
- **Latent morphisms**: parameters (learnable)
- **Observed morphisms**: fixed tensors
- **Composition**: tensor contraction according to the quantale's operations

## Setup

```python
import torch
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism, observed
from quivers.core.quantales import PRODUCT_FUZZY
from quivers.program import Program
```

## Creating Objects

Create three finite sets: one for positions (X), one for colors (Y), and one for outcomes (Z):

```python
X = FinSet("Position", 3)
Y = FinSet("Color", 4)
Z = FinSet("Outcome", 2)

print(X.size)   # 3
print(Y.size)   # 4
print(Z.size)   # 2
```

Each object has a shape and size. These define the dimensions of the tensors representing morphisms.

## Creating Morphisms

### Latent Morphism

A latent morphism has learnable tensor entries. Create one from X to Y:

```python
f = morphism(X, Y)
print(f.domain)     # Position
print(f.codomain)   # Color
print(f.tensor)     # shape [3, 4], values in (0, 1)
print(f.tensor.shape)
```

The tensor is initialized with a sigmoid activation, so all entries lie in $(0, 1)$. The entries are PyTorch parameters: they will be updated during backpropagation if you include the morphism in a training loop.

Inspect the underlying module:

```python
mod = f.module()
params = list(mod.parameters())
print(len(params))  # 1
print(params[0].shape)  # torch.Size([3, 4])
```

### Observed Morphism

An observed morphism has a fixed, non-learnable tensor. Create one from Y to Z with explicit data:

```python
data = torch.tensor([
    [1.0,  0.1],
    [0.8,  0.2],
    [0.5,  0.5],
    [0.0,  0.9],
])
g = observed(Y, Z, data)
print(g.tensor)
```

The tensor shape must match $(|Y|, |Z|) = (4, 2)$. If you pass the wrong shape, an error is raised.

Verify it has no learnable parameters:

```python
mod_g = g.module()
params_g = list(mod_g.parameters())
print(len(params_g))  # 0
```

## Composition

Compose two morphisms using the `>>` operator. Composition applies the quantale's operations: the tensor product $\otimes$ (multiplication) and the join $\bigvee$ (noisy-OR).

Create a third latent morphism from Z to a new object:

```python
W = FinSet("Result", 5)
h = morphism(Z, W)
```

Compose f and g: X -> Y -> Z

```python
fg = f >> g
print(fg.domain)  # Position
print(fg.codomain)  # Outcome
print(fg.tensor.shape)  # [3, 2]
```

Composition is lazy: the tensor is not computed until evaluation. Verify this is a ComposedMorphism:

```python
from quivers.core.morphisms import ComposedMorphism
print(isinstance(fg, ComposedMorphism))  # True
```

Compose again: (X -> Y -> Z) -> W

```python
fgh = fg >> h
print(fgh.tensor.shape)  # [3, 5]
```

The composition operation uses the quantale's tensor product (here, pointwise multiplication) and join (noisy-OR):

$$
(g \circ f)(x, z) = \bigvee_y f(x, y) \otimes g(y, z)
$$

In PRODUCT_FUZZY, $\otimes$ is multiplication and $\bigvee$ is $1 - \prod_i (1 - x_i)$.

## Accessing Tensor Values

Once composed, access the materialized tensor:

```python
tensor_value = fg.tensor
print(tensor_value.dtype)  # torch.float32
print(tensor_value.min(), tensor_value.max())  # check range
```

For a relation in PRODUCT_FUZZY, values should lie in $[0, 1]$.

## Working as a Differentiable Module

Wrap a morphism in a `Program` to make it a differentiable `nn.Module`:

```python
prog = Program(f)
output = prog()
print(output.shape)  # [3, 4]
print(output)
```

Integrate with PyTorch training:

```python
import torch.optim as optim

program = Program(fgh)  # a complex composition
optimizer = optim.Adam(program.parameters(), lr=0.01)

for epoch in range(5):
    optimizer.zero_grad()

    # Forward pass
    result = program()

    # Define a loss (e.g., sum for illustration)
    loss = result.sum()

    # Backward and step
    loss.backward()
    optimizer.step()

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

The parameters of all latent morphisms in the composition will be updated.

## Quantales

The examples so far use PRODUCT_FUZZY as the enrichment:

```python
print(PRODUCT_FUZZY.name)  # "ProductFuzzy"
```

Other quantales are available (Boolean, Łukasiewicz, Gödel). The choice of quantale affects:

1. How morphisms are initialized
2. How they compose
3. The semantics of the resulting values

For now, PRODUCT_FUZZY is the natural choice for fuzzy relations.

## Summary

You have:

- Created finite sets (objects)
- Constructed latent (learnable) and observed (fixed) morphisms
- Composed morphisms with `>>`
- Inspected tensor shapes and values
- Wrapped morphisms as differentiable modules
- Learned that composition uses the quantale's operations (tensor product and join)

Next, explore how these ideas extend to stochastic morphisms (Markov kernels) in [Tutorial 2](stochastic-relations.md).
