# Quivers

**Quivers** is a Python library for building categorical and probabilistic models as differentiable PyTorch programs. It represents morphisms between finite sets as tensors valued in a quantale (a lattice with a monoidal product), then extends this to stochastic morphisms (Markov kernels), continuous distribution families, monadic probabilistic programs, and variational inference. A built-in functional DSL compiles `.qvr` specifications into trainable `nn.Module` instances.

## Core Concepts

A **quiver** is a directed graph with objects and arrows. In quivers, arrows are $\mathcal{V}$-relations: functions from pairs of objects to a quantale $\mathcal{V}$ (an algebraic structure of truth values). We represent these as tensors, making them differentiable and composable via PyTorch.

The library provides:

- **Core categorical algebra**: finite sets and product constructions as objects; eleven shipped quantales (Boolean, product fuzzy, Łukasiewicz, Gödel, tropical min-plus, max-plus / Viterbi, log-prob, Markov, real, probability, counting) plus a homomorphism registry for change-of-base; $\mathcal{V}$-enriched relations as parametrized tensors with full compact-closed surface (`dagger`, `trace`, `cup`, `cap`).
- **Categorical structures**: functors, natural transformations, adjunctions, monoidal categories, traced monoidal categories.
- **Monadic and enriched constructs**: monads, comonads, algebras, Kleisli categories, ends/coends, Kan extensions, profunctors, Yoneda, Day convolution, optics.
- **Stochastic morphisms**: the FinStoch category of Markov kernels; discretized families (normal, beta, truncated normal); conditioning and mixing; the Giry monad.
- **Continuous morphisms**: parameterized families of distributions (30+); boundaries (discretize/embed); normalizing flows; monadic programs (probabilistic computations with discrete and continuous random variables).
- **Monadic DSL**: a `.qvr` file format and compiler for writing categorical programs declaratively.
- **Inference**: a six-layer stack on a shared `LatentRegistry` — nine variational guides (mean-field Normal, Delta, full / low-rank multivariate Normal, Laplace, general normalising flows, IAF, neural-spline coupling, finite mixtures); four objectives (ELBO, IWAE, Rényi, VR-IWAE) × four gradient estimators (reparameterised, sticking-the-landing, doubly-reparameterised, score-function); HMC and NUTS with dual-averaging step-size + Welford mass-matrix adaptation and R-hat / ESS / divergence diagnostics; hybrid `AutoDAIS` and `WarmupThenHMC` samplers; `Predictive` that consumes a `Guide` or an `MCMCResult`.
- **Weighted-deduction framework**: a single agenda-engine runtime subsumes CKY, Earley, Viterbi, semi-naïve Datalog, A* parsing, Knuth's algorithm, and MLTT proof search. Surface `deduction { … }` blocks declare the seven canonical parameters; charts are first-class differentiable values.
- **Structural compression**: `signature { … }`, `encoder { … }`, `decoder { … }`, and `loss { … }` blocks form a uniform F-algebra / F-coalgebra interface for compressing arbitrary structured objects to fixed-length vectors and decoding them back under a learned distribution — realising transformers, tree-LSTMs, graph-NNs, autoregressive LMs, VAEs, and vector inside-outside parsers as instances of one categorical pattern.

## Quick Start

Install from source:

```bash
pip install torch
git clone https://github.com/FACTSlab/quivers
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
