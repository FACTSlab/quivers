<h1 align="center">Quivers</h1>

<p align="center">
  <em>A functional probabilistic programming language for PyTorch.</em>
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

Quivers is a functional probabilistic programming language for PyTorch. The surface will look familiar if you have used Pyro, NumPyro, Stan, or PyMC. But it has a few distinguishing features:

- **Programs are first-class composable typed values.** A program has a domain, codomain, algebra, and effect signature (`! Sample, Score, Marginal, Pure`), checked at compile time. Programs compose with `>>`, parallel-compose with `@`, change base across algebras with `change_base`, and marginalize discrete latents with `marginalize z : K <- ... in { ... }`.
- **Shared substrate for inference, deduction, and structural compression.** A CKY parser in a `deduction { atoms ... rule ... }` block, a transformer-as-encoder over a `signature { ... }` block, and a Bayesian regression all compile to the same underlying semantics, with the same composition operators, and can therefore compose with each other.
- **Algebra-parametric semantics.** Programs can be parameterized by eleven built-in or user-defined algebras. Homomorphisms between algebras are values you can transport models along, with the laws checked at compile time.

It also has some features you are used to from other PPLs:

- **An inference toolkit.** Forty distribution families. SVI with nine automatic guides (mean-field through full-rank multivariate normal, low-rank, mixture, IAF, neural-spline flow, AutoDAIS) and four objectives (ELBO, IWAE, Renyi, VR-IWAE) with reparameterized, score-function, sticking-the-landing, and DReG gradient estimators. NUTS and HMC with dual-averaging step-size adaptation and Welford mass-matrix adaptation.
- **An analysis toolkit.** Static introspection of compiled programs (per-step algebra, chain depth, intermediate shape, source mapping); algebra-aware, saturation-free initialization recipes that adapt to whichever value algebra a program is parameterized over; compile-time diagnostics flagging latents whose default initialization would saturate the active algebra.
- **Diagnostics and model comparison.** ArviZ ecosystem integration: posteriors from any inference method (NUTS, HMC, or SVI) export to ArviZ for trace plots, rank plots, ESS, and $\hat R$. PSIS-LOO (Pareto-smoothed importance-sampling leave-one-out cross-validation) for ranking competing models; posterior-predictive checks against user-defined test statistics; LOO-PIT for calibration.
- **A mixed-effect model API.** A [brms-style formula frontend](https://FACTSlab.github.io/quivers/guides/analysis) for mixed-effect regression compiles formulas to typed QVR programs through a bidirectional lens, with pandas / polars dataframes as the input surface and R-canonical conventions (orthogonal polynomials, R-style transforms in the formula evaluation namespace) as defaults. The emitted QVR is inspectable, so a formula-fitted model is a starting point you can hand-edit rather than a closed black box.
- **Interactive tooling out of the box.** [`qvr repl`](https://FACTSlab.github.io/quivers/guides/repl-and-lsp) is a GHCi-style four-pane Textual TUI with live syntax highlighting, env browser, file-watcher auto-reload, command palette, and meta-commands (`:type`, `:info`, `:browse`, `:edit`, `:save`, `:watch`, …). [`qvr-lsp`](https://FACTSlab.github.io/quivers/guides/repl-and-lsp) is a full LSP 3.17 language server (hover, definition, references, document symbols, semantic tokens, completion, formatting, live diagnostics) that VS Code, Cursor, Zed, and Neovim consume out of the box. A Jupyter kernel (`qvr-kernel install`) drives the same elaborator from notebooks.

## Quick start

```bash
pip install quivers
```

```qvr
object Item : FinSet 100

program regression : Item -> Item [effects=[Sample, Score]]
    sample sigma  <- HalfNormal(1.0)
    sample beta_0 <- Normal(0.0, 5.0)
    sample beta_1 <- Normal(0.0, 2.0)
    let mu = beta_0 + beta_1 * x
    observe y : Item <- Normal(mu, sigma)
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

- [**Tutorial**](https://FACTSlab.github.io/quivers/tutorials/): the QVR DSL tutorial walks probabilistic-programming users from linear regression to inference-algorithm choice with PyMC, NumPyro, and Stan equivalents shown side-by-side, while the Python API tutorial covers the typed categorical surface.
- [**Examples gallery**](https://FACTSlab.github.io/quivers/examples/): 36 end-to-end models covering regression, latent-variable, state-space, language models, seq2seq, and formal grammars.
- [**Conceptual guides**](https://FACTSlab.github.io/quivers/guides/): feature-area deep dives.
- [**API reference**](https://FACTSlab.github.io/quivers/api/): the typed Python surface.
- [**Denotational semantics**](https://FACTSlab.github.io/quivers/semantics/): the meaning of every well-typed program in a $\mathcal{V}$-enriched symmetric monoidal closed category.

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

Requirements: Python 3.14+, PyTorch 2.0+, didactic 0.7.1+, panproto 0.47.3+, panproto-grammars-all 0.47.3+.

Optional extras:

```bash
pip install 'quivers[repl]'    # Textual TUI, prompt_toolkit, rich, ipykernel
pip install 'quivers[lsp]'     # pygls language server
pip install 'quivers[repl,lsp]'  # both
```

After installing `[repl]` you can drop into the interactive type explorer:

```bash
qvr repl path/to/model.qvr
```

After installing `[lsp]` you have `qvr-lsp` on your PATH; the
[`vscode-qvr`](https://github.com/FACTSlab/quivers/tree/main/editors/vscode-qvr)
and
[`zed-extension-qvr`](https://github.com/FACTSlab/quivers/tree/main/editors/zed-extension-qvr)
extensions auto-discover it.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests welcome at [github.com/FACTSlab/quivers](https://github.com/FACTSlab/quivers).

## Acknowledgments

This project was developed by [Aaron Steven White](https://aaronstevenwhite.io/) at the University of Rochester with support from the National Science Foundation (NSF-BCS-2237175 *CAREER: Logical Form Induction*, NSF-BCS-2040831 *Computational Modeling of the Internal Structure of Events*). It was architected and implemented with the assistance of Claude Code.

## License

MIT. See [LICENSE](LICENSE).
