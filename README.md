# Quivers

[![CI](https://github.com/FACTSlab/quivers/actions/workflows/ci.yml/badge.svg)](https://github.com/FACTSlab/quivers/actions/workflows/ci.yml)
[![Docs](https://github.com/FACTSlab/quivers/actions/workflows/docs.yml/badge.svg)](https://FACTSlab.github.io/quivers)
[![PyPI](https://img.shields.io/pypi/v/quivers)](https://pypi.org/project/quivers/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Computational category theory as differentiable tensor programs.

**Quivers** is a Python library for building categorical and probabilistic models as differentiable PyTorch programs. It represents morphisms between finite sets as tensors valued in a quantale (a lattice with a monoidal product), then extends this to stochastic morphisms (Markov kernels), continuous distribution families, monadic probabilistic programs, and variational inference. A built-in functional DSL compiles `.qvr` specifications into trainable `nn.Module` instances.

## Features

- **Core categorical algebra**: finite sets, product/coproduct constructions, and free monoids as objects; quantales (Boolean, product fuzzy, Łukasiewicz, Gödel, tropical) as enrichment algebras; $\mathcal{V}$-enriched relations as parametrized tensors with composition via quantale operations.
- **Categorical structures**: functors, natural transformations, adjunctions, monoidal categories, traced monoidal categories, base change between quantales.
- **Enriched category theory**: ends, coends, Kan extensions, weighted limits/colimits, profunctors, Yoneda embedding, Day convolution, optics (lenses, prisms, adapters, grates).
- **Monadic constructs**: monads, comonads, Kleisli/coKleisli categories, algebras, coalgebras, Eilenberg-Moore categories, distributive laws.
- **Stochastic morphisms**: the FinStoch category of Markov kernels; discretized distribution families (normal, logit-normal, beta, truncated normal); conditioning, mixing, and normalization transforms; the Giry monad; query functions (prob, marginal_prob, expectation).
- **Continuous morphisms**: 30+ parameterized conditional distribution families; continuous spaces (Euclidean, simplex, unit interval, positive reals); sampled composition; normalizing flows; discrete-continuous boundaries (discretize/embed).
- **Monadic programs**: probabilistic programs with draw, observe, and return statements; ancestral sampling; log-joint computation; hybrid discrete-continuous random variables.
- **QVR DSL**: a `.qvr` file format with lexer, recursive descent parser, AST, and compiler; supports object/morphism declarations, program blocks, let bindings, type expressions, and grammar-based parsers (PCFG, CCG, Lambek, multimodal type-logical).
- **Variational inference**: execution traces, conditioning, automatic variational guides (normal, delta), ELBO computation, stochastic variational inference (SVI), posterior predictive sampling.

## Installation

```bash
pip install quivers
```

Or install from source:

```bash
git clone https://github.com/FACTSlab/quivers
cd quivers
pip install -e .
```

For development (includes pytest, ruff, pyright):

```bash
pip install -e ".[dev]"
```

## Quick Start

### Discrete morphisms and composition

Define finite sets, create learnable and observed morphisms, and compose them:

```python
from quivers import FinSet, morphism, observed, Program
import torch

X = FinSet("X", 3)
Y = FinSet("Y", 4)
Z = FinSet("Z", 2)

f = morphism(X, Y)                # learnable (sigmoid over raw params)
g = observed(Y, Z, torch.rand(4, 2))  # fixed tensor

h = f >> g                         # V-enriched composition: X -> Z
program = Program(h)
output = program()                 # shape (3, 2), values in [0, 1]
```

Composition uses the product fuzzy quantale by default: AND is multiplication, OR is noisy-OR ($1 - \prod(1 - x_i)$). The result is a differentiable tensor, trainable via `program.parameters()`.

### Stochastic morphisms

Work with Markov kernels in the FinStoch category:

```python
from quivers import FinSet, stochastic, condition, prob

S = FinSet("S", 3)
O = FinSet("O", 5)

transition = stochastic(S, S)   # learnable row-stochastic matrix
emission = stochastic(S, O)     # learnable row-stochastic matrix

# condition on an observation
conditioned = condition(emission, obs_index=2)

# query probabilities
p = prob(transition, domain_idx=0, codomain_idx=1)
```

### The QVR DSL

Write probabilistic programs in `.qvr` syntax and compile to `nn.Module`:

```python
from quivers.dsl import loads

source = """
object Predictor : 1
object Response : 1

program regression : Predictor -> Response
    sigma <- HalfCauchy(2.0)
    beta_0 <- Normal(0.0, 5.0)
    beta_1 <- Normal(0.0, 2.0)
    x <- Normal(0.0, 1.0)
    let mu = beta_0 + beta_1 * x
    observe y ~ Normal(mu, sigma)
    return y

output regression
"""

model = loads(source)
```

## Project Structure

```text
src/quivers/
├── core/           # objects, quantales, morphisms, tensor ops
├── categorical/    # functors, natural transformations, adjunctions, monoidal, traced
├── monadic/        # monads, comonads, algebras, distributive laws
├── enriched/       # ends/coends, Kan extensions, profunctors, Yoneda, Day, optics
├── stochastic/     # Markov kernels, Giry monad, grammar parsers, chart algorithms
├── continuous/     # distribution families, spaces, flows, monadic programs
├── dsl/            # lexer, parser, AST, compiler for .qvr files
├── inference/      # traces, conditioning, guides, ELBO, SVI, predictive
├── program.py      # Program: wraps morphisms as nn.Module
└── giry.py         # GiryMonad, FinStoch
```

## Documentation

Full documentation: [https://FACTSlab.github.io/quivers](https://FACTSlab.github.io/quivers)

- [Installation](https://FACTSlab.github.io/quivers/getting-started/installation/)
- [Quickstart](https://FACTSlab.github.io/quivers/getting-started/quickstart/)
- [Conceptual Guides](https://FACTSlab.github.io/quivers/guides/)
- [Tutorials](https://FACTSlab.github.io/quivers/tutorials/)
- [Examples Gallery](https://FACTSlab.github.io/quivers/examples/)
- [API Reference](https://FACTSlab.github.io/quivers/api/)

## Requirements

- Python 3.13+
- PyTorch 2.0+

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style conventions, and the git workflow.

## License

MIT. See [LICENSE](LICENSE) for details.
