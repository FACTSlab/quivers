# Tutorial 4: Fuzzy Logic Factorization

This tutorial demonstrates how to use quivers to factorize a fuzzy relation into a composition of learnable morphisms under the product fuzzy logic quantale. This is matrix factorization where the algebra is not standard linear algebra but probabilistic fuzzy logic: product is fuzzy AND and summation is fuzzy OR.

## Background

In standard matrix factorization, we approximate a matrix $R \approx F \cdot G$ using real-valued multiplication and addition. In **fuzzy logic factorization**, we replace these with operations from a quantale:

- **Fuzzy AND** (product t-norm): $a \otimes b = a \cdot b$
- **Fuzzy OR** (probabilistic sum / noisy-OR): $\bigvee_i x_i = 1 - \prod_i (1 - x_i)$

Given an observed fuzzy relation $R: X \to Z$ with values in $[0, 1]$, we seek two learnable fuzzy relations $f: X \to Y$ and $g: Y \to Z$ such that their $\mathcal{V}$-enriched composition approximates $R$:

$$
(g \circ f)(x, z) = \bigvee_{y \in Y} f(x, y) \otimes g(y, z) = 1 - \prod_{y \in Y} \bigl(1 - f(x, y) \cdot g(y, z)\bigr)
$$

Each entry of the composed relation asks: "is there *some* intermediate $y$ through which $x$ relates to $z$?" The "some" is the noisy-OR and the "through" is the product. The latent set $Y$ acts as a bottleneck of fuzzy features.

## Setup

```python
import torch
from quivers import FinSet, morphism, observed, Program
```

Define three finite sets. The latent set $Y$ has smaller cardinality than $X$ and $Z$, forcing the factorization to compress the relation through a low-dimensional bottleneck:

```python
X = FinSet("X", 6)   # domain: 6 elements
Y = FinSet("Y", 3)   # latent: 3 fuzzy features
Z = FinSet("Z", 8)   # codomain: 8 elements
```

## Observed Relation

Create a synthetic fuzzy relation $R: X \to Z$ with values in $[0, 1]$. We build it from two ground-truth factors to ensure it has a low-rank fuzzy structure:

```python
torch.manual_seed(42)

# ground-truth factors (unknown to the model)
f_true = torch.sigmoid(torch.randn(6, 3))
g_true = torch.sigmoid(torch.randn(3, 8))

# compose via noisy-OR of products
product = f_true.unsqueeze(-1) * g_true.unsqueeze(0)  # (6, 3, 8)
R_data = 1.0 - (1.0 - product).prod(dim=1)            # (6, 8)

R = observed(X, Z, R_data)
```

The tensor `R_data` has shape $(6, 8)$ with entries in $[0, 1]$, representing the fuzzy membership of each $(x, z)$ pair.

## Learnable Factorization

Define two **latent morphisms**, learnable $\mathcal{V}$-enriched relations whose parameters are optimized during training:

```python
f = morphism(X, Y)   # learnable: X -> Y
g = morphism(Y, Z)   # learnable: Y -> Z
```

Each latent morphism stores unconstrained real-valued parameters and applies a sigmoid to produce values in $(0, 1)$. The composition `f >> g` automatically uses the `ProductFuzzy` quantale:

```python
h = f >> g            # V-enriched composition: X -> Z
```

Wrap the composition as a trainable `Program` (an `nn.Module`):

```python
model = Program(h)
```

## Training

The `Program` provides a `bce_loss` method that computes binary cross-entropy between the materialized composition and a target tensor. Since both the model output and target are fuzzy membership values in $[0, 1]$, BCE is a natural choice:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(500):
    optimizer.zero_grad()
    loss = model.bce_loss(R_data)
    loss.backward()
    optimizer.step()

    if (step + 1) % 100 == 0:
        print(f"step {step+1:>4d}  loss={loss.item():.4f}")
```

Gradients flow through the noisy-OR composition and the sigmoid parameterization back to the raw parameters of `f` and `g`.

## Inspecting the Result

After training, materialize the composed tensor and compare to the target:

```python
with torch.no_grad():
    R_hat = model()

# element-wise reconstruction error
error = (R_hat - R_data).abs()
print(f"max error:  {error.max().item():.4f}")
print(f"mean error: {error.mean().item():.4f}")
```

Examine the learned factors:

```python
print("f (X -> Y):")
print(f.tensor.detach())

print("\ng (Y -> Z):")
print(g.tensor.detach())
```

Each row of `f.tensor` assigns fuzzy membership over the three latent features for a given element of $X$. Each column of `g.tensor` shows how a latent feature relates to elements of $Z$.

## Why Fuzzy Logic?

Standard (real-valued) matrix factorization uses addition to aggregate contributions from latent dimensions. Fuzzy factorization uses the **noisy-OR**, which has a different semantics:

- In standard factorization: latent contributions *add up*
- In fuzzy factorization: latent contributions provide *independent chances*

The noisy-OR interpretation is natural for modeling scenarios where each latent feature provides an independent "reason" for a relation to hold, and the overall relation holds if *any* reason applies. This is useful for:

- **Knowledge base completion**: an entity pair is related if any latent pattern supports it
- **Recommendation**: a user likes an item if any latent preference dimension matches
- **Feature detection**: a sample belongs to a category if any diagnostic feature is present

## Alternative Quantales

Quivers supports several other quantales that change the meaning of composition:

```python
from quivers import BOOLEAN, LUKASIEWICZ, GODEL, TROPICAL

# Boolean: AND/OR (crisp relations)
f_bool = morphism(X, Y, quantale=BOOLEAN)

# Łukasiewicz: bounded sum (resource-sensitive logic)
f_luk = morphism(X, Y, quantale=LUKASIEWICZ)

# Gödel: min/max (possibilistic, minimax composition)
f_godel = morphism(X, Y, quantale=GODEL)
```

Each quantale gives the same code structure (`morphism`, `>>`, `Program`) but a different algebraic semantics for composition.

## Summary

In this tutorial you:

- Defined an observed fuzzy relation as an `ObservedMorphism`
- Factorized it into two learnable `LatentMorphism` instances
- Composed them with `>>` using the product fuzzy logic quantale
- Trained the factorization with BCE loss via `Program`
- Inspected the learned fuzzy features

Next, learn how to fit probabilistic models to data with variational inference in [Tutorial 5](variational-inference.md).
