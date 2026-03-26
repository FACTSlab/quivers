# Morphisms & Composition

## What is a Morphism?

A morphism from domain $A$ to codomain $B$ in a $\mathcal{V}$-enriched category is a tensor $M \in \mathcal{V}^{|A| \times |B|}$, where $\mathcal{V}$ is the enriching quantale. Concretely, it is a multi-dimensional tensor with shape `(*A.shape, *B.shape)` and values in the lattice $\mathcal{L}$ of $\mathcal{V}$.

For a simple `FinSet` morphism $f: X \to Y$ with $|X| = m$ and $|Y| = n$, the tensor is an $m \times n$ matrix.

For a product domain like $X \times Y$, the tensor has shape `(|X|, |Y|, |Z|)`.

## The Morphism Hierarchy

```
Morphism (abstract)
├── ObservedMorphism    — fixed tensor, not learnable
├── LatentMorphism      — learnable parameter with sigmoid output
├── ComposedMorphism    — lazy composition f >> g
├── ProductMorphism     — lazy tensor product f @ g
├── MarginalizedMorphism — lazy marginalization (join reduction)
└── FunctorMorphism     — lazy functor image
```

### ObservedMorphism

A fixed, non-learnable morphism. Use `observed()` to construct:

```python
from quivers.core.objects import FinSet
from quivers.core.morphisms import observed
import torch

X = FinSet("X", 3)
Y = FinSet("Y", 4)
data = torch.tensor([
    [0.9, 0.1, 0.0, 0.0],
    [0.2, 0.6, 0.1, 0.1],
    [0.0, 0.0, 0.8, 0.2],
])

f = observed(X, Y, data)
assert f.tensor.shape == (3, 4)
assert (f.module().parameters() == [])  # no learnable params
```

### LatentMorphism

A learnable morphism parameterized by a weight matrix, with sigmoid output to ensure values in $(0, 1)$. Construct with `morphism()`:

```python
from quivers.core.morphisms import morphism

X = FinSet("X", 3)
Y = FinSet("Y", 4)

f = morphism(X, Y)
print(f.tensor.shape)  # (3, 4)
print((f.tensor > 0).all() and (f.tensor < 1).all())  # True
```

Access the underlying parameter via `f.raw`:

```python
params = list(f.module().parameters())
assert len(params) == 1
assert torch.allclose(f.tensor, torch.sigmoid(f.raw))
```

Initialize with a specific scale:

```python
# Manual initialization of f.raw before calling f.tensor
f = morphism(X, Y)
with torch.no_grad():
    f.raw.fill_(value)  # modify f.raw in place
```

## Composition: The >>  Operator

Composition of two morphisms uses the quantale's operations. If $f: A \to B$ and $g: B \to C$, then $g \circ f: A \to C$ is computed as:

$$(g \circ f)[a, c] = \bigvee_b f[a, b] \otimes g[b, c]$$

where $\bigvee$ is the quantale's join and $\otimes$ is its tensor operation.

```python
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

X = FinSet("X", 3)
Y = FinSet("Y", 4)
Z = FinSet("Z", 2)

f = morphism(X, Y)
g = morphism(Y, Z)

# Compose with >>
h = f >> g
assert h.domain == X
assert h.codomain == Z
assert h.tensor.shape == (3, 2)
```

The composition is lazy: the tensor is not materialized until accessed. Chain compositions:

```python
W = FinSet("W", 5)
k = morphism(Z, W)
pipeline = f >> g >> k
assert pipeline.tensor.shape == (3, 5)
```

Compositions must have compatible quantales:

```python
from quivers.core.quantales import BOOLEAN, GodelQuantale

f_bool = morphism(X, Y, quantale=BOOLEAN)
g_godel = morphism(Y, Z, quantale=GodelQuantale())

# Raises TypeError: incompatible quantales
try:
    _ = f_bool >> g_godel
except TypeError as e:
    print(e)
```

## Tensor Product: The @ Operator

The tensor (or parallel) product $f \otimes g$ combines two morphisms $f: A \to B$ and $g: C \to D$ into a morphism $f \otimes g: A \times C \to B \times D$. The tensor is the outer product via the quantale's $\otimes$:

```python
A = FinSet("A", 2)
B = FinSet("B", 3)
C = FinSet("C", 4)
D = FinSet("D", 5)

f = morphism(A, B)
g = morphism(C, D)

# Tensor product
h = f @ g
assert h.domain == A * C
assert h.codomain == B * D
assert h.tensor.shape == (2, 3, 4, 5)
```

The @ operator works even if domains or codomains already have products; ProductSet automatically flattens:

```python
P = A * B
Q = C * D

f_prod = morphism(P, Q)  # (A × B) → (C × D)
```

## Marginalization

Join-reduce the codomain over specified components:

```python
X = FinSet("X", 3)
Y = FinSet("Y", 4)
Z = FinSet("Z", 5)

f = morphism(X, Y * Z)  # X → Y × Z

# Marginalize over Z (sum/join over Z dimension)
g = f.marginalize(Z)
assert g.domain == X
assert g.codomain == Y
```

The tensor is computed by applying the quantale's join operation over the codomain dimensions corresponding to $Z$.

## Operations Summary

| Operation | Syntax | Input | Output |
|-----------|--------|-------|--------|
| Composition | `f >> g` | $f: A \to B$, $g: B \to C$ | $g \circ f: A \to C$ |
| Tensor | `f @ g` | $f: A \to B$, $g: C \to D$ | $f \otimes g: A \times C \to B \times D$ |
| Marginalize | `f.marginalize(A)` | $f: X \times A \to Y$ | $f\|_X: X \to Y$ (join over $A$) |
| Identity | `identity(X)` | object $X$ | $\text{id}_X: X \to X$ |

## Learning and Gradients

All morphisms expose an `nn.Module` tree via `.module()`:

```python
f = morphism(X, Y)
module = f.module()

# Collect parameters for optimization
optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

# Compute loss and backpropagate
loss = f.tensor.sum()
loss.backward()
optimizer.step()
```

The gradient flows through composition and tensor operations, so you can train entire pipelines end-to-end.
