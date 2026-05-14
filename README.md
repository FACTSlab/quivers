# Quivers

[![CI](https://github.com/FACTSlab/quivers/actions/workflows/ci.yml/badge.svg)](https://github.com/FACTSlab/quivers/actions/workflows/ci.yml)
[![Docs](https://github.com/FACTSlab/quivers/actions/workflows/docs.yml/badge.svg)](https://FACTSlab.github.io/quivers)
[![PyPI](https://img.shields.io/pypi/v/quivers)](https://pypi.org/project/quivers/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A probabilistic programming language for PyTorch.

Quivers lets you write Bayesian models in a small, readable DSL and fit them with stochastic variational inference (SVI), NUTS, HMC, or any of nine automatic guides. The program surface should look familiar if you have used Pyro, NumPyro, Stan, or PyMC: declare variables with `<-`, score observations with `observe`, integrate out discrete latents with `marginalize`, get a trainable PyTorch module back.

```qvr
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
import torch

program = loads(open("regression.qvr").read())
model   = program.morphism
guide   = AutoNormalGuide(model, observed_names={"y"})
optim   = torch.optim.Adam(guide.parameters(), lr=1e-2)
svi     = SVI(model, guide, optim, ELBO())
for _ in range(2000):
    svi.step({"x": x_data}, {"y": y_data})
```

The full walkthrough is in the [tutorial](https://FACTSlab.github.io/quivers/tutorials/).

## What you get

The everyday PPL features you would expect, on a PyTorch backend:

- **Forty distribution families** (Normal, Beta, Gamma, Dirichlet, MVN, LKJ, MatrixNormal, GP, Horseshoe, mixtures, normalising flows, and more).
- **Nine variational guides** from mean-field through full-rank multivariate normal, low-rank, mixture, IAF, neural-spline flow, and AutoDAIS.
- **Four inference objectives** (ELBO, IWAE, Renyi, VR-IWAE) with reparameterised / score-function / sticking-the-landing / DReG gradient estimators.
- **NUTS and HMC** with dual-averaging step-size adaptation and Welford mass-matrix adaptation, plus a `WarmupThenHMC` hybrid sampler.
- **Marginalised discrete latents** as a first-class block (`marginalize z : K <- Categorical(p) in { ... }`), with `logsumexp` aggregation handled for you.
- **Plates and grouped marginalisation** for hierarchical models with vectorised observations and per-row fibration into shared random effects.
- **A 36-example gallery** covering regression (Bayesian, Beta, Dirichlet, NegBin, horseshoe, ZIP), latent variable (factor analysis, PPCA, LDA, IRT, PMF, BNN, GMM, VAE), state space (HMM discrete and continuous, linear-Gaussian SSM, deep Markov, AR1, stochastic volatility, changepoint, Weibull survival), language models (RNN, LSTM, GRU, bidirectional, transformer), seq2seq with encoder/decoder, and formal grammars (PCFG, CCG, Lambek, multimodal TLG).

## What's distinctive

Most PPLs let you write `observe y ~ Normal(mu, sigma)`. Quivers lets you write the same thing AND a few things ordinary PPLs do not.

- **Typed scoped marginalisation.** `marginalize z : K <- Categorical(p) in { ... }` is a syntactic block whose body runs once per discrete value of `z`, with the per-value scores aggregated by `logsumexp`. This is the standard Rao-Blackwellisation trick, but spelt as a control-flow construct instead of a runtime flag.
- **Axis-role priors on weights.** A weight matrix `latent W : Euclidean(D) -> Euclidean(K)` can carry a structured prior whose covariance is genuinely matrix-valued: `~ MatrixNormal(loc, row_cov, col_cov) over (dom, cod)`. The `over <axes>` clause says which axes the family's joint covariance lives on; the rest are iid. This is the right surface for factor analysis, PPCA, Bayesian neural nets, and other "matrix of weights with prior" models.
- **Exact-likelihood structured families.** HMMs and Kalman smoothers compose like ordinary distributions; the forward / forward-backward / smoother passes are wrapped.
- **Compile-time effects.** Programs carry an effect signature `! Sample, Score, Marginal, Pure` that the compiler checks against the body. A `! Pure` block that contains an `observe` is rejected with a typed error before training begins.
- **Weighted deduction.** Chart algorithms (CKY, Earley, Viterbi, A*, Knuth's algorithm, semi-naive Datalog) are exposed as a `deduction { atoms ... rule ... semiring ... start ... }` block whose chart is a differentiable tensor. Drops in alongside the rest of the language.
- **Structural compression.** A four-block pattern (`signature { ... } encoder { ... } decoder { ... } loss { ... }`) factors out transformers, tree LSTMs, graph NNs, autoregressive LMs, and the vector inside-outside parser as instances of one interface.

## What's under the hood (optional reading)

The DSL is a thin layer over a typed categorical surface in `src/quivers/`. If you want to extend the library, write a new family, prove anything about a model, or read the type errors fluently, the categorical layer is what you read. If you just want to fit models, you can ignore it. The denotational semantics ([docs](https://FACTSlab.github.io/quivers/semantics/)) gives every well-typed program a formal meaning in a $\mathcal{V}$-enriched symmetric monoidal closed category. The implementation rests on enriched category theory ([Kelly, 1982](http://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html)), the categorical foundations of probability ([Giry, 1982](https://doi.org/10.1007/BFb0092872); [Fritz, 2020](https://doi.org/10.1016/j.aim.2020.107239)), and the SVI / HMC inference substrate ([Hoffman, Blei, Wang & Paisley, 2013](https://doi.org/10.5555/2567709.2502622); [Neal, 2011](https://doi.org/10.1201/b10905-6); [Hoffman & Gelman, 2014](https://www.jmlr.org/papers/v15/hoffman14a.html)).

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
