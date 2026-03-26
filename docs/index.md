# Quivers

**Quivers** is a Python library for building categorical and probabilistic models as differentiable PyTorch programs. It represents morphisms between finite sets as tensors valued in a quantale (a lattice with a monoidal product), then extends this to stochastic morphisms (Markov kernels), continuous distribution families, monadic probabilistic programs, and variational inference. A built-in functional DSL compiles `.qvr` specifications into trainable `nn.Module` instances.

## Core Concepts

A **quiver** is a directed graph with objects and arrows. In quivers, arrows are $\mathcal{V}$-relations: functions from pairs of objects to a quantale $\mathcal{V}$ (an algebraic structure of truth values). We represent these as tensors, making them differentiable and composable via PyTorch.

The library provides:

- **Core categorical algebra**: finite sets and product constructions as objects; quantales (enrichment lattices) like Boolean, product fuzzy logic, Łukasiewicz, and Gödel; $\mathcal{V}$-enriched relations as parametrized tensors.
- **Categorical structures**: functors, natural transformations, adjunctions, monoidal categories, traced monoidal categories.
- **Monadic and enriched constructs**: monads, comonads, algebras, Kleisli categories, ends/coends, Kan extensions, profunctors, Yoneda, Day convolution, optics.
- **Stochastic morphisms**: the FinStoch category of Markov kernels; discretized families (normal, beta, truncated normal); conditioning and mixing; the Giry monad.
- **Continuous morphisms**: parameterized families of distributions (30+); boundaries (discretize/embed); normalizing flows; monadic programs (probabilistic computations with discrete and continuous random variables).
- **Monadic DSL**: a `.qvr` file format and compiler for writing categorical programs declaratively.
- **Variational inference**: trace-based conditioning, variational guides, ELBO and SVI for posterior inference.

## Quick Start

Install from source:

```bash
pip install torch
git clone https://github.com/aaronstevenwhite/quivers
cd quivers
pip install -e .
```

Create and compose morphisms:

```python
from quivers import FinSet, morphism, observed, identity, Program
import torch

X = FinSet("X", 3)
Y = FinSet("Y", 4)
Z = FinSet("Z", 2)

# Latent (learnable) morphism
f = morphism(X, Y)

# Observed morphism with fixed tensor
g_data = torch.rand(4, 2)
g = observed(Y, Z, g_data)

# V-enriched composition: X -> Y -> Z
h = f >> g

# Wrap as a trainable module
program = Program(h)
output = program()  # shape (3, 2)
```

See [Installation](getting-started/installation.md), [Quickstart](getting-started/quickstart.md), and [Architecture](getting-started/architecture.md) for more.
