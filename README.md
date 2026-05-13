# Quivers

[![CI](https://github.com/FACTSlab/quivers/actions/workflows/ci.yml/badge.svg)](https://github.com/FACTSlab/quivers/actions/workflows/ci.yml)
[![Docs](https://github.com/FACTSlab/quivers/actions/workflows/docs.yml/badge.svg)](https://FACTSlab.github.io/quivers)
[![PyPI](https://img.shields.io/pypi/v/quivers)](https://pypi.org/project/quivers/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A probabilistic programming language with a categorical implementation.

Quivers is a Python library for writing, fitting, and reasoning about probabilistic models. Models are declared in a small typed DSL whose surface looks like Pyro or NumPyro: you write a `program` block with `v <- F(args)` draws, `let` deterministic bindings, and `observe` statements. Compilation produces a trainable `nn.Module`. Inference goes through a stack of nine variational guides, four objectives, HMC, NUTS, and two hybrid samplers.

```qvr
quantale real
object Item : 100

program regression : Item -> Item ! Sample, Score
    sigma  <- HalfNormal(1.0)
    beta_0 <- Normal(0.0, 5.0)
    beta_1 <- Normal(0.0, 2.0)
    x      <- Normal(0.0, 1.0)
    let mu = beta_0 + beta_1 * x
    observe y <- Normal(mu, sigma)
    return y

export regression
```

```python
from quivers.dsl import loads
from quivers.inference import AutoNormalGuide, ELBO, SVI

program = loads(open("regression.qvr").read())
model   = program.morphism
guide   = AutoNormalGuide(model, observed_names={"y"})
svi     = SVI(model, guide, optimizer, ELBO())
for _ in range(2000):
    svi.step({"x": x_data}, {"y": y_data})
```

The rest is the [tutorial](https://FACTSlab.github.io/quivers/tutorials/).

## What's distinctive

Quivers does the things every PPL does: hierarchical models, GLMs, mixtures, sequence models, posterior-predictive checks. It also does a handful of things most PPLs don't:

- **Typed-scope marginalisation.** `marginalize z : K <- Categorical(p) in { ... }` is a first-class block whose body runs once per discrete value, with `logsumexp` aggregation under the prior. Standard Rao-Blackwellisation, but spelt as syntax instead of as a config flag.
- **Exact-likelihood structured families.** Hidden Markov models, Kalman smoothers, and similar structured likelihoods compose like ordinary distribution families; the forward / forward-backward / smoother passes are wrapped.
- **First-class transformations.** Change-of-base transformations (softmax, L1/L2 normalisation, Bayes inversion, quantale homomorphisms) are values: let-bindable, composable with `>>>`, passable into `change_base`.
- **Composition rules beyond quantales.** The `CompositionRule → BilinearForm | Semigroupoid → Quantale` hierarchy supports non-associative and non-unital composition rules with full operadic n-ary contractions via einsum-style wiring.
- **Compile-time effects.** Programs carry an effect signature `! Sample, Score, Marginal, Pure` that the compiler checks against the body. `! Pure` blocks that try to `observe` are rejected with a typed error.
- **A categorical denotational semantics.** Every well-typed QVR phrase has a [formal denotation](https://FACTSlab.github.io/quivers/semantics/) in a $\mathcal{V}$-enriched symmetric monoidal closed category. The compiler implementation is proved adequate against the denotation.
- **A weighted-deduction surface.** Chart algorithms (CKY, Earley, Viterbi, semi-naïve Datalog, A*, Knuth's algorithm) compose with probabilistic programs through a single agenda-engine runtime parameterised by item algebra, rules, semiring, and priority. Charts are first-class differentiable values.
- **A structural-compression surface.** `signature { … } encoder { … } decoder { … } loss { … }` blocks form an F-algebra / F-coalgebra interface for compressing structured objects (sequences, trees, graphs, parse charts) to fixed-length vectors and decoding them back. Realises transformers, tree-LSTMs, graph-NNs, autoregressive LMs, and the vector-inside-outside parser as instances of one pattern.

The implementation rests on enriched category theory ([Kelly, 1982](http://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html)), the categorical foundations of probability ([Giry, 1982](https://doi.org/10.1007/BFb0092872); [Fritz, 2020](https://doi.org/10.1016/j.aim.2020.107239)), and the SVI / HMC inference substrate ([Hoffman, Blei, Wang & Paisley, 2013](https://doi.org/10.5555/2567709.2502622); [Neal, 2011](https://doi.org/10.1201/b10905-6); [Hoffman & Gelman, 2014](https://www.jmlr.org/papers/v15/hoffman14a.html)).

## Installation

```bash
pip install quivers
```

Or install from source:

```bash
git clone https://github.com/FACTSlab/quivers
cd quivers
pip install -e ".[dev]"
```

Requirements: Python 3.14+, PyTorch 2.0+, didactic 0.6.0+, panproto 0.45.0+, panproto-grammars-all 0.45.0+.

## Learning path

Two parallel tracks, depending on what you want:

- **[QVR DSL tutorial](https://FACTSlab.github.io/quivers/tutorials/qvr/01-first-model/)** for probabilistic-programming users. Seven chapters, model development through inference, side-by-side with PyMC / NumPyro / Stan.
- **[Python API tutorial](https://FACTSlab.github.io/quivers/tutorials/python/01-first-quiver/)** for library developers and category-theory-fluent users. Seven chapters covering the typed categorical surface end to end.

Then:

- [Conceptual guides](https://FACTSlab.github.io/quivers/guides/) for feature-area deep dives.
- [Examples gallery](https://FACTSlab.github.io/quivers/examples/) for end-to-end model code.
- [Denotational semantics](https://FACTSlab.github.io/quivers/semantics/) for the formal treatment.
- [API reference](https://FACTSlab.github.io/quivers/api/) for the typed surface.

## Project structure

```text
src/quivers/
├── core/           objects, quantales, morphisms, tensor ops, wiring
├── categorical/    functors, natural transformations, adjunctions, monoidal, traced
├── monadic/        monads, comonads, algebras, distributive laws
├── enriched/       ends/coends, Kan extensions, profunctors, Yoneda, Day, optics
├── stochastic/     Markov kernels, Giry monad, grammar parsers, chart algorithms
├── continuous/     distribution families, spaces, flows, monadic programs
├── dsl/            parser (panproto / tree-sitter), AST (didactic Models),
│                   compiler, resolution lenses, Program Theory
├── inference/      registry, guides, objectives, estimators, MCMC, hybrids
├── program.py      Program: wraps morphisms as nn.Module
└── giry.py         GiryMonad, FinStoch
```

The tree-sitter grammar lives at `grammars/qvr/` and is vendored by [panproto](https://panproto.dev)'s `panproto-grammars-all` distribution.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and the git workflow. Issues and pull requests welcome at [github.com/FACTSlab/quivers](https://github.com/FACTSlab/quivers).

## License

MIT. See [LICENSE](LICENSE) for details.
