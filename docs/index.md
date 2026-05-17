# Quivers

A functional probabilistic programming language for PyTorch.

Quivers is a functional probabilistic programming language for PyTorch. The surface will look familiar if you have used Pyro, NumPyro, Stan, or PyMC. But it has a few distinguishing features:

- **Programs are first-class composable typed values.** A program has a domain, codomain, algebra, and effect signature (`! Sample, Score, Marginal, Pure`), checked at compile time. Programs compose with `>>`, parallel-compose with `@`, change base across algebras with `change_base`, and marginalize discrete latents with `marginalize z : K <- ... in { ... }`.
- **Shared substrate for inference, deduction, and structural compression.** A CKY parser in a `deduction { atoms ... rule ... }` block, a transformer-as-encoder over a `signature { ... }` block, and a Bayesian regression all compile to the same underlying semantics, with the same composition operators, and can therefore compose with each other.
- **Algebra-parametric semantics.** Programs can be parameterized by eleven built-in or user-defined algebras. Homomorphisms between algebras are values you can transport models along, with the laws checked at compile time.

It also has some features you are used to from other PPLs:

- **An inference toolkit.** Forty distribution families. SVI with nine automatic guides (mean-field through full-rank multivariate normal, low-rank, mixture, IAF, neural-spline flow, AutoDAIS) and four objectives (ELBO, IWAE, Renyi, VR-IWAE) with reparameterized, score-function, sticking-the-landing, and DReG gradient estimators. NUTS and HMC with dual-averaging step-size adaptation and Welford mass-matrix adaptation.
- **An analysis toolkit.** Static introspection of compiled programs (per-step algebra, chain depth, intermediate shape, source mapping); algebra-aware, saturation-free initialization recipes that adapt to whichever value algebra a program is parameterized over; compile-time diagnostics flagging latents whose default initialization would saturate the active algebra.
- **Diagnostics and model comparison.** ArviZ ecosystem integration: posteriors from any inference method (NUTS, HMC, or SVI) export to ArviZ for trace plots, rank plots, ESS, and $\hat R$. PSIS-LOO (Pareto-smoothed importance-sampling leave-one-out cross-validation) for ranking competing models; posterior-predictive checks against user-defined test statistics; LOO-PIT for calibration.
- **A mixed-effect model API.** A [brms-style formula frontend](https://FACTSlab.github.io/quivers/guides/analysis) for mixed-effect regression compiles formulas to typed QVR programs through a bidirectional lens, with pandas / polars dataframes as the input surface and R-canonical conventions (orthogonal polynomials, R-style transforms in the formula evaluation namespace) as defaults. The emitted QVR is inspectable, so a formula-fitted model is a starting point you can hand-edit rather than a closed black box.
- **Interactive tooling out of the box.** [`qvr repl`](guides/repl-and-lsp.md) is a GHCi-style four-pane Textual TUI with live syntax highlighting, env browser, file-watcher auto-reload, command palette, and meta-commands (`:type`, `:info`, `:browse`, `:edit`, `:save`, `:watch`, …). [`qvr-lsp`](guides/repl-and-lsp.md) is a full LSP 3.17 language server (hover, definition, references, document symbols, semantic tokens, completion, formatting, live diagnostics) that VS Code, Cursor, Zed, and Neovim consume out of the box. A Jupyter kernel (`qvr-kernel install`) drives the same elaborator from notebooks.

## Quick start

```qvr
object Item : 100

# Predictor `x` flows in as exogenous data via the observations
# dict; free variables in `let` expressions resolve from the
# conditioning data at trace time (host-data channel).
program regression : Item -> Item ! Sample, Score
    sigma  <- HalfNormal(1.0)
    beta_0 <- Normal(0.0, 5.0)
    beta_1 <- Normal(0.0, 2.0)
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
    svi.step(torch.zeros(100, 1), {"x": x_data, "y": y_data})
```

The same regression also expresses through a [brms-style formula frontend](guides/analysis-data-and-formulas.md):

```python
from quivers.formulas import fit

result = fit(
    "y ~ x + (1 | g)",
    data=df,                  # pandas or polars
    family="gaussian",
    method="nuts",
)
result.dump_qvr("regression.qvr")   # inspect the emitted QVR program
```

## Where to start

- **[Installation](getting-started/installation.md)** for setup.
- **[Quickstart](getting-started/quickstart.md)** for a working model in five minutes.
- **[QVR tutorial](tutorials/qvr/01-first-model.md)** for probabilistic-programming users: seven chapters from linear regression through hierarchical models, sequence models, and inference-algorithm choice, with Pyro / NumPyro / Stan equivalents side-by-side.
- **[Python API tutorial](tutorials/python/01-first-quiver.md)** for library developers and category-theory-fluent users: seven chapters covering the typed categorical surface.
- **[Examples gallery](examples/index.md)**: 36 end-to-end models grouped by family.
- **[Conceptual guides](guides/index.md)** for feature-area deep dives.
- **[API reference](api/index.md)** for the typed Python surface.
- **[Denotational semantics](semantics/index.md)** for the formal meaning of every well-typed program.

## Architecture

The DSL is a thin layer over a typed categorical surface. If you want to extend the library, write a new distribution family, or prove anything about a model, the categorical layer is what you read. If you just want to fit models, you can ignore it.

The library decomposes into eight layers. Each is consumable in isolation; each builds on those below it:

```mermaid
flowchart TB
    L8["Layer 8: QVR DSL<br/>.qvr files, tree-sitter grammar, panproto AST"]
    L7["Layer 7: Structural compression<br/>signature, encoder, decoder, loss blocks"]
    L6["Layer 6: Inference<br/>guides, objectives, MCMC kernels, hybrid samplers"]
    L5["Layer 5: Continuous probabilistic programs<br/>MonadicProgram, distribution families, flows, plates"]
    L4["Layer 4: Stochastic morphisms<br/>Markov kernels, Giry monad, chart deduction surface"]
    L3["Layer 3: Monadic and enriched constructs<br/>monads, algebras, ends, coends, Kan, profunctors"]
    L2["Layer 2: Categorical structure<br/>functors, naturals, adjunctions, monoidal, traced"]
    L1["Layer 1: Core V-enriched algebra<br/>FinSet, Morphism, CompositionRule hierarchy, wiring"]
    L8 --> L7 --> L6 --> L5 --> L4 --> L3 --> L2 --> L1
```

The central abstraction is a morphism between finite sets, parameterized by an algebra (a complete lattice with a monoidal product distributing over joins). A morphism `f : A -> B` is a PyTorch tensor of shape `(|A|, |B|)` whose entries take values in the algebra; composition `f >> g` contracts along the shared dimension under the algebra's tensor product and join. Different algebras give different composition semantics: Boolean composes by AND / OR (relational composition), ProductFuzzy by multiplication / noisy-OR, Real by sum-product, Markov by row-stochastic kernel composition, and so on.

The [denotational semantics](semantics/index.md) gives every well-typed QVR phrase a formal meaning in a $\mathcal{V}$-enriched symmetric monoidal closed category. The implementation rests on enriched category theory ([Kelly, 1982](http://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html)), the categorical foundations of probability ([Giry, 1982](https://doi.org/10.1007/BFb0092872); [Fritz, 2020](https://doi.org/10.1016/j.aim.2020.107239)), and the SVI / HMC inference substrate ([Hoffman, Blei, Wang & Paisley, 2013](https://doi.org/10.5555/2567709.2502622); [Neal, 2011](https://doi.org/10.1201/b10905-6); [Hoffman & Gelman, 2014](https://www.jmlr.org/papers/v15/hoffman14a.html)).
