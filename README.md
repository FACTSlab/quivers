<h1 align="center">Quivers</h1>

<p align="center">
  <em>A typed, compositional probabilistic programming language for PyTorch.</em>
</p>

<p align="center">
  <a href="https://github.com/FACTSlab/quivers/actions/workflows/ci.yml"><img src="https://github.com/FACTSlab/quivers/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://FACTSlab.github.io/quivers"><img src="https://github.com/FACTSlab/quivers/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <a href="https://pypi.org/project/quivers/"><img src="https://img.shields.io/pypi/v/quivers" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.14%2B-blue" alt="Python 3.14+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://FACTSlab.github.io/quivers/tutorials/qvr/01-first-model/"><strong>Tutorial</strong></a>
  ·
  <a href="https://FACTSlab.github.io/quivers/examples/"><strong>Examples</strong></a>
  ·
  <a href="https://FACTSlab.github.io/quivers/guides/"><strong>Guides</strong></a>
  ·
  <a href="https://FACTSlab.github.io/quivers/api/"><strong>API</strong></a>
  ·
  <a href="https://FACTSlab.github.io/quivers/semantics/"><strong>Semantics</strong></a>
</p>

---

Quivers is a probabilistic programming language for PyTorch. The surface looks familiar if you have used Pyro, NumPyro, Stan, or PyMC: declare variables with `<-`, score observations with `observe`, integrate out discrete latents with `marginalize`, fit with SVI / NUTS / HMC, get a trainable `nn.Module` back.

Distinguishing features:

- **Algebra-parametric semantics.** The composition and aggregation operations used by `>>`, `marginalize`, and `deduction` blocks are set by an `algebra` keyword at the top of a program. The default `algebra probability` gives ordinary Bayesian inference. `algebra log_prob` runs the same program in log-space for numerical stability on long sequences. `algebra tropical` replaces summation with maximization, so the forward pass over a hidden Markov model produces a Viterbi decode rather than a marginal likelihood. `algebra boolean` reduces scores to truth values; `deduction` blocks then behave like weighted Datalog. Eleven built-in algebras (`probability`, `log_prob`, `boolean`, `tropical`, `max_plus`, `markov`, `product_fuzzy`, `lukasiewicz`, `godel`, `real`, `counting`); homomorphisms between them are values you can transport models along, with the laws checked at compile time.
- **Programs are typed compositional values.** A program has a domain, codomain, algebra, and effect signature (`! Sample, Score, Marginal, Pure`), checked at compile time. Programs compose with `>>`, parallel-compose with `@`, transport across algebras with `change_base`, and marginalize discrete latents with `marginalize z : K <- ... in { ... }`.
- **Shared substrate for inference, deduction, and structural compression.** A CKY parser in a `deduction { atoms ... rule ... }` block, a transformer-as-encoder over a `signature { ... }` block, and a Bayesian regression all compile to morphisms in the same category, with the same composition operators, and can compose with each other.
- **Inference toolkit.** Forty distribution families. SVI with nine automatic guides (mean-field through full-rank multivariate normal, low-rank, mixture, IAF, neural-spline flow, AutoDAIS). Four objectives (ELBO, IWAE, Renyi, VR-IWAE) with reparameterized, score-function, sticking-the-landing, and DReG gradient estimators. NUTS and HMC with dual-averaging step-size adaptation and Welford mass-matrix adaptation. A brms-style [formula frontend](https://FACTSlab.github.io/quivers/guides/analysis) (`fit("y ~ x + (1|g)", data=df)`) with ArviZ diagnostics and PSIS-LOO model comparison.

## Quick start

```bash
pip install quivers
```

```qvr
object Item : 100

program regression : Item -> Item ! Sample, Score
    sigma  <- HalfNormal(1.0)
    beta_0 <- Normal(0.0, 5.0)
    beta_1 <- Normal(0.0, 2.0)
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
    svi.step(x_data, {"y": y_data})
```

The full walkthrough is in the [tutorial](https://FACTSlab.github.io/quivers/tutorials/).

## Documentation

- [**Tutorial**](https://FACTSlab.github.io/quivers/tutorials/) — two parallel tracks: the QVR DSL tutorial walks probabilistic-programming users from linear regression to inference-algorithm choice with PyMC / NumPyro / Stan equivalents shown side-by-side; the Python API tutorial covers the typed categorical surface.
- [**Examples gallery**](https://FACTSlab.github.io/quivers/examples/) — 36 end-to-end models covering regression, latent-variable, state-space, language models, seq2seq, and formal grammars.
- [**Conceptual guides**](https://FACTSlab.github.io/quivers/guides/) — feature-area deep dives.
- [**API reference**](https://FACTSlab.github.io/quivers/api/) — the typed Python surface.
- [**Denotational semantics**](https://FACTSlab.github.io/quivers/semantics/) — formal meaning of every well-typed program in a $\mathcal{V}$-enriched symmetric monoidal closed category.

## Installation

```bash
pip install quivers
```

From source:

```bash
git clone https://github.com/FACTSlab/quivers
cd quivers
pip install -e ".[dev]"
```

Requirements: Python 3.14+, PyTorch 2.0+, didactic 0.6.0+, panproto 0.45.0+, panproto-grammars-all 0.45.0+.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests welcome at [github.com/FACTSlab/quivers](https://github.com/FACTSlab/quivers).

## License

MIT. See [LICENSE](LICENSE).
