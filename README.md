# Quivers

Computational category theory as differentiable tensor programs.

**Quivers** is a Python library for building categorical and probabilistic models as differentiable PyTorch programs. It represents morphisms between finite sets as tensors valued in a quantale (a lattice with a monoidal product), then extends this to stochastic morphisms (Markov kernels), continuous distribution families, monadic probabilistic programs, and variational inference. A built-in DSL compiles `.qvr` specifications into trainable `nn.Module` instances.

## Features

- **Core categorical algebra**: finite sets and product constructions as objects; quantales (Boolean, fuzzy, Łukasiewicz, Gödel, tropical); $\mathcal{V}$-enriched relations as parametrized tensors.
- **Categorical structures**: functors, natural transformations, adjunctions, monoidal categories, traced monoidal categories.
- **Monadic and enriched constructs**: monads, comonads, algebras, Kleisli categories, ends/coends, Kan extensions, profunctors, Yoneda, Day convolution, optics.
- **Stochastic morphisms**: the FinStoch category of Markov kernels; discretized families (normal, beta, truncated normal); conditioning and mixing; the Giry monad.
- **Continuous morphisms**: parameterized families of distributions (30+); boundaries (discretize/embed); normalizing flows; monadic programs for hybrid discrete-continuous computations.
- **Monadic DSL**: a `.qvr` file format and compiler for writing categorical programs declaratively.
- **Variational inference**: trace-based conditioning, variational guides, ELBO and SVI for posterior inference.

## Installation

```bash
pip install quivers
```

Or install from source:

```bash
git clone https://github.com/aaronstevenwhite/quivers
cd quivers
pip install -e .
```

## Quick Start

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

## Documentation

Full documentation is available at [https://aaronstevenwhite.github.io/quivers](https://aaronstevenwhite.github.io/quivers).

## Requirements

- Python 3.12+
- PyTorch 2.0+

## License

MIT. See [LICENSE](LICENSE) for details.
