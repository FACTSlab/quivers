<h1 align="center">Quivers</h1>

<p align="center">
  <em>A typed functional probabilistic programming language with a PyTorch runtime.</em>
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

Quivers is a typed functional probabilistic programming language with a PyTorch
runtime and eleven transpilation targets. The surface will look familiar if you
have used Pyro, NumPyro, Stan, or PyMC. But it has a few distinguishing
features:

- **Programs are first-class composable typed values.** A program has a domain, codomain, algebra, and effect signature (`[effects=[Sample, Score, Marginal]]`), checked at compile time. Programs compose with `>>`, parallel-compose with `@`, change base across algebras with `change_base`, and marginalize discrete latents with a scoped `marginalize z : K <- ...` block.
- **Shared substrate for inference, deduction, and structural compression.** A CKY parser in a `deduction` block (its `atoms`, `rule`, and `lexicon` entries), a transformer-as-encoder over a `signature` block, and a Bayesian regression all compile to the same underlying semantics, with the same composition operators, and can thus compose with each other.
- **Indexed families and algebraic effects have a typed core.** QVR v0.19 adds GADT-style indexed constructors, lexical effect instances, row-polymorphic computations, handlers, and resumption grades. These forms check and lower to the stable [Quivers Indexed Effect Core](https://FACTSlab.github.io/quivers/developer/qiec/). Pyro, NumPyro, PyMC, Edward2, Turing, Gen, WebPPL, and Church emit reachable computations through a common runtime ABI. Stan accepts closed monomorphic effect-free scalar functions; BUGS and JAGS accept their measured program subset. Every target reports a stable capability diagnostic rather than silently dropping unsupported code.
- **Algebra-parametric semantics.** Programs can be parameterized by eleven built-in or user-defined algebras. Homomorphisms between algebras are values along which models can be transported. The compiler checks their source and target types; the algebraic laws remain assumptions of each instance.

The probabilistic-programming surface also includes:

- **An inference toolkit.** More than forty distribution families. SVI with automatic guides from mean-field and full-rank multivariate normals through low-rank, mixture, structured, IAF, neural-spline flow, and AutoDAIS guides; seven objectives (ELBO, IWAE, Renyi, VR-IWAE, ChiVI, RWS, and DReGs); and reparameterized, score-function, sticking-the-landing, and DReG gradient estimators. NUTS and HMC use dual-averaging step-size adaptation and Welford mass-matrix adaptation.
- **An analysis toolkit.** Static introspection of compiled programs (per-step algebra, chain depth, intermediate shape, source mapping); algebra-aware, saturation-free initialization recipes that adapt to whichever value algebra a program is parameterized over; compile-time diagnostics flagging latents whose default initialization would saturate the active algebra.
- **Diagnostics and model comparison.** ArviZ ecosystem integration: posteriors from any inference method (NUTS, HMC, or SVI) export to ArviZ for trace plots, rank plots, ESS, and $\hat R$. PSIS-LOO (Pareto-smoothed importance-sampling leave-one-out cross-validation) for ranking competing models; posterior-predictive checks against user-defined test statistics; LOO-PIT for calibration.
- **A mixed-effect model API.** A [brms-style formula frontend](https://FACTSlab.github.io/quivers/guides/analysis) for mixed-effect regression compiles formulas to typed QVR programs through a bidirectional lens, with pandas / polars dataframes as the input surface and R-canonical conventions (orthogonal polynomials, R-style transforms in the formula evaluation namespace) as defaults. The emitted QVR is inspectable, so a formula-fitted model is a starting point you can hand-edit rather than a closed black box.
- **Interactive tooling.** [`qvr repl`](https://FACTSlab.github.io/quivers/guides/repl-and-lsp) is a GHCi-style four-pane Textual TUI with live syntax highlighting, an environment browser, file-watcher reloads, a command palette, and meta-commands (`:type`, `:info`, `:browse`, `:edit`, `:save`, `:watch`, …). [`qvr-lsp`](https://FACTSlab.github.io/quivers/guides/repl-and-lsp) implements LSP 3.17 features including hover, definition, references, document symbols, semantic tokens, completion, formatting, and live diagnostics for VS Code, Cursor, Zed, and Neovim. A Jupyter kernel (`qvr-kernel install`) drives the same elaborator from notebooks.

## QVR v0.19 at a glance

QVR has one compilation path. Authored `define` computations, probabilistic
`program` blocks, deductions, parsers, and structural encoder/decoder graphs
all elaborate to a checked QIEC module. The command line, Python API, REPL,
language server, and transpilers consume that same module.

| Surface | What v0.19 adds | Start here |
| --- | --- | --- |
| Types | closed indices, indexed families, constructors, motive-checked cases | [Types and indexed families](https://FACTSlab.github.io/quivers/reference/qvr/types-and-indexed-families/) |
| Effects | parameterized interfaces, lexical instances, open rows, handlers, resumption grades | [Effects and handlers](https://FACTSlab.github.io/quivers/reference/qvr/effects-and-handlers/) |
| Computations | `define`, static specialization, recursion, pure/effectful binding, calls | [Computations](https://FACTSlab.github.io/quivers/reference/qvr/computations/) |
| Programs | samples, observations, factors, marginalization, groups, scans | [Probabilistic programs](https://FACTSlab.github.io/quivers/reference/qvr/probabilistic-programs/) |
| Generated graphs | deductions, schema parsers, encoders, decoders, and losses in QIEC | [Generated computations](https://FACTSlab.github.io/quivers/reference/qvr/generated-computations/) |
| Tooling | entry execution, traces, providers, LSP, target checks, and transpilation | [Execution and tooling](https://FACTSlab.github.io/quivers/reference/qvr/execution-and-tooling/) |

## Quick start

```bash
pip install 'quivers[lsp]'
```

Save the following as `robust.qvr`. It declares an effect, an instance, a
total handler, and a pure computation whose effect has been discharged.

<!-- compile: qiec -->
```qvr
effect Robust
    shrink : Real -> Real

instance robust : Robust

handler half_weight for Robust : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    shrink(x : Real) resumes 1 =>
        resume(0.5 * x)

define robustify(x : Real) : Real !{} =
    handle robust with half_weight in
        let adjusted <- perform robust.shrink(x)
        return adjusted
```

Parse, type-and-effect check, execute, and test a deployment target through the
same checked module:

```bash
qvr check robust.qvr
qvr run robust.qvr robustify 8.0 --json
qvr check --target pyro robust.qvr
qvr transpile --to pyro robust.qvr
```

For a full statistical model, download a gallery source and pass its compiled
morphism to the inference API:

```bash
curl -LO https://raw.githubusercontent.com/FACTSlab/quivers/main/docs/examples/source/bayesian_regression.qvr
```

```python
import torch

from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

program = load("bayesian_regression.qvr")
model = program.morphism

torch.manual_seed(0)
n = 64
x_data = torch.randn(n)
y_data = torch.distributions.Normal(1.5 + 2.0 * x_data, 0.5).sample()
observations = {"x": x_data, "y": y_data}
model_input = torch.zeros(n, 1)

guide = AutoNormalGuide(model, observed_names={"x", "y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2
)
svi = SVI(model, guide, optim, ELBO())
for _ in range(300):
    svi.step(model_input, observations)
```

The [QVR tutorial](https://FACTSlab.github.io/quivers/tutorials/qvr/01-first-model/)
builds from a first model through indexed data, handlers, generated search,
structural attachments, and target-aware release checks. The integrated
[Amortized Bayesian Semantics](https://FACTSlab.github.io/quivers/examples/amortized-bayesian-semantics/)
example uses the new features together in a hierarchical psycholinguistic
model.

## Documentation

- [**Tutorial**](https://FACTSlab.github.io/quivers/tutorials/): the QVR DSL tutorial walks from regression and inference through indexed data, handlers, generated search, structural attachments, and target-aware release checks, while the Python API tutorial covers the typed categorical surface.
- [**QVR language reference**](https://FACTSlab.github.io/quivers/reference/qvr/): v0.19 syntax, type-and-effect rules, generated computation graphs, entry execution, tooling, and grammar ownership.
- [**Examples gallery**](https://FACTSlab.github.io/quivers/examples/): 46 transpilation-measured programs plus an integrated QIEC case study, covering regression, latent-variable, state-space, language, structural, and formal-grammar models.
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

Requirements: Python 3.14+, PyTorch 2.0+, didactic 0.17.1+, panproto 0.74.4+,
and panproto-grammars-all 0.74.4+.

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
[`zed-extension-qvr`](https://github.com/FACTSlab/quivers/tree/main/editors/zed-extension-qvr) extensions
auto-discover it. Both ship the complete v0.19 vocabulary; the LSP adds typed
semantic tokens and live QIEC and target-capability diagnostics.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests welcome at [github.com/FACTSlab/quivers](https://github.com/FACTSlab/quivers).

## Acknowledgments

This project was developed by [Aaron Steven White](https://aaronstevenwhite.io/) at the University of Rochester with support from the National Science Foundation (NSF-BCS-2237175 *CAREER: Logical Form Induction*, NSF-BCS-2040831 *Computational Modeling of the Internal Structure of Events*). It was architected and implemented with the assistance of Claude Code.

## License

MIT. See [LICENSE](LICENSE).
