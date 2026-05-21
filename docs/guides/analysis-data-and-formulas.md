# Analysis: Data and Formulas

This page covers the front-half of the analysis stack: feeding
dataframes into models, declaring a model with a brms-style
formula, and inspecting / emitting the QVR source the formula
compiles to. The back-half (fitting, diagnostics, algebra-guided
training tooling) lives in
[Fitting and Diagnostics](analysis-fitting-and-diagnostics.md).

## Architecture

Four small subpackages, each consumable independently:

```mermaid
flowchart TB
    F["quivers.formulas<br/>brms-style formula to typed AST to QVR program"]
    D["quivers.data<br/>DataFrame to object cardinalities + observations"]
    G["quivers.diagnostics<br/>MCMCResult to ArviZ DataTree, compare, PPC"]
    E["quivers.dsl.emit<br/>Module AST to canonical .qvr source"]
    F --> D
    F --> E
    F --> G
    D --> G
```

Each subpackage is gated behind an optional dependency extra so a
user who only wants the DSL + inference doesn't pull pandas /
polars / arviz / formulae. Install everything together via
`pip install "quivers[analysis]"`.

## Dataframes: `quivers.data`

[`DatasetSchema`](../api/data/schema.md) is a typed
[`didactic.api.Model`](https://didactic.dev/api/Model) that maps
dataframe columns to QVR-program artifacts. It accepts pandas,
polars, or any other
[Narwhals](https://narwhals-dev.github.io/narwhals/)-compatible
dataframe.

<!-- python: skip -->
```python
import pandas as pd
from quivers.data import DatasetSchema, compose

df = pd.DataFrame({
    "verb": ["eat", "drink", "run", "eat", ...],
    "subject": ["s1", "s2", "s1", "s3", ...],
    "rt": [0.31, 0.42, 0.28, 0.55, ...],
    "response": [1, 0, 1, 1, ...],
})

schema = DatasetSchema(
    df=df,
    objects={"verb": "Verb", "subject": "Subject"},
    plate_indices={"verb": "verb_idx", "subject": "subj_idx"},
    covariates={"rt": "rt"},
    observations={"response": "y"},
)

print(schema.declarations())          # object Verb : 17 / object Subject : 50
print(schema.cardinalities)           # {"Verb": 17, "Subject": 50}
obs = schema.observations_dict()      # {"verb_idx": tensor, "subj_idx": tensor, ...}
```

Two artifacts come out:

- [`declarations()`](../api/data/schema.md#quivers.data.schema.DatasetSchema.declarations)
  emits a `.qvr` prelude with one `object X : N` line per declared
  object axis. The cardinality is inferred from
  `df[col].n_unique()`; canonical category ordering is the column's
  sorted unique non-null values so plate indices are reproducible
  across reruns.
- [`observations_dict()`](../api/data/schema.md#quivers.data.schema.DatasetSchema.observations_dict)
  packs the per-row tensors that inference consumes (response,
  plate indices, numeric covariates), ready to pass into
  [`SVI.step`](../api/inference/svi.md) or
  [`MCMC.run`](../api/inference/predictive.md).

The companion
[`compose(qvr_body, schema)`](../api/data/schema.md#quivers.data.schema.compose)
prepends the schema's declarations to a user's `.qvr` body before
compiling, so the user writes only the program body and the
cardinalities come from the data.

Missing-data handling is configurable per schema via
[`MissingPolicy`](../api/data/encoding.md#quivers.data.encoding.MissingPolicy):
`RAISE` (default), `DROP`, `IMPUTE`, or `MASK`.

## Formulas: `quivers.formulas`

The [formula frontend](../api/formulas/index.md) compiles a
[brms](https://paul-buerkner.github.io/brms/) /
[`lme4`](https://cran.r-project.org/package=lme4)-style formula
into a typed QVR [`Module`](../api/dsl/ast_nodes.md) AST. No
source-string concatenation: the translation
[`FormulaToQVRModule`](../api/formulas/compile.md#quivers.formulas.compile.FormulaToQVRModule)
is a [`didactic.api.Lens`](https://didactic.dev/api/Lens) from
`Formula` to `Module`, mirroring the existing resolution-lens
pattern in
[`quivers.dsl.resolution`](../api/dsl/resolution.md). Formula
syntax is parsed by the
[`formulae`](https://bambinos.github.io/formulae/) library (the
[Bambi](https://bambinos.github.io/bambi/) team's pure-Python
brms-style parser), then lifted into a typed `Formula` record.

### Inspect or dump the generated QVR

<!-- python: skip -->
```python
from quivers.formulas import formula_to_qvr

src = formula_to_qvr("y ~ poly(x, 2) + (1 | g)", data=df)
print(src)                                  # canonical .qvr source
```

The emit goes through
[`quivers.dsl.emit.module_to_source`](../api/dsl/emit.md), which
walks the `Module` AST and produces canonical `.qvr` source. The
emitted source re-parses through
[`quivers.dsl.loads`](../api/dsl/parser.md) into a `Module` that
compiles to the same program: the round-trip is exercised on every
formula in the test suite.

### R / brms behaviour, exactly

- **Orthogonal polynomials by default.** `poly(x, k)` produces $k$
  orthonormal centred columns, matching R's
  [`stats::poly`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/poly.html).
  Raw monomials remain available via `I(x**k)`.
- **One coefficient per design-matrix column** (matches brms
  display). `poly(x, 2)` produces two named coefficients
  `beta_poly_x_2_1` and `beta_poly_x_2_2`; `x*z` produces three
  named coefficients (`beta_x`, `beta_z`, `beta_x_z`). The
  per-column data flows in as a free variable via the host-data
  channel (see the
  [conditioning surface](inference-foundations.md#host-data-per-row-covariates-and-index-arrays)).
- **R-style transforms** preloaded into the formulae evaluation
  namespace: `log`, `exp`, `sqrt`, `abs`, `sin`, `cos`, `tan`,
  `log10`, `log2`, `log1p`, `expm1`, `asin`, `acos`, `atan`,
  `sinh`, `cosh`, `tanh`. No registration required.
- **Random-effect groups** `(1 | g)`, `(1 + x | g)`, `(x | g)`,
  `(0 + x | g)` parse identically to brms / lme4. Multiple slopes
  per group emit independent random-effect terms (the lme4
  `(... || g)` uncorrelated semantics); correlated LKJ-prior slopes
  are future scope.
- **Interactions** `x:z` (elementwise product, one coefficient) and
  `x*z` (expands to `x + z + x:z`, three coefficients).

### Family registry

`fit(..., family=...)` accepts a string name or a
[`Family`](../api/formulas/family.md#quivers.formulas.family.Family)
value. The ten brms-canonical families:

| Family | Link (inverse) | Auxiliary parameters |
|---|---|---|
| `gaussian` | identity | `sigma ~ HalfCauchy(2.0)` |
| `bernoulli`, `binomial` | logit (sigmoid) | – |
| `categorical` | softmax | – |
| `poisson` | log (exp) | – |
| `negative_binomial` | log (exp) | `disp ~ Gamma(2.0, 2.0)` |
| `gamma` | log (exp) | `shape ~ Gamma(2.0, 2.0)` |
| `beta` | logit (sigmoid) | `phi ~ HalfCauchy(2.0)` |
| `student_t` | identity | `nu ~ Gamma(2.0, 0.1)`, `sigma ~ HalfCauchy(2.0)` |
| `cumulative` | identity | – |

Custom families are pluggable: subclass
[`Family`](../api/formulas/family.md#quivers.formulas.family.Family)
and register your own observe kernel and link.

### Prior overrides

Prior overrides are keyed by the latent's name in the emitted QVR
program (which `formula_to_qvr` lets you inspect upfront). The
prior template is a brms-style `Family(arg, arg, ...)` call;
numeric args become floats, identifier args stay as references to
other latents in the program. The full call shape lives in
[Fitting and Diagnostics](analysis-fitting-and-diagnostics.md#prior-overrides).

## See also

- [Fitting and Diagnostics](analysis-fitting-and-diagnostics.md):
  the `fit(...)` entry point, diagnostics, and algebra-guided
  training tooling.
- [DSL Overview](dsl-overview.md): the typed DSL the formula
  frontend emits source for.
- [Hierarchical Programs](programs-hierarchical.md): the program
  surface that random-effects formulas compile to.
