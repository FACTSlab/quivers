# Transpilation Correctness: Contract and Evidence

`quivers.transpile.transpile(module, target=...)` supports eleven registered targets: BUGS, Church, Edward2, Gen, JAGS, NumPyro, PyMC, Pyro, Stan, Turing, and WebPPL. This chapter states the **transpile-correctness contract** (TCC) and the finite evidence used to check it. It does not claim a formal proof of cross-language contextual equivalence.

## 1. Pipeline

The production path is:

$$
\mathrm{Module}
\xrightarrow{\text{target-specific composite expansion}}
\mathrm{Module}
\xrightarrow{\mathrm{Lower}}
\mathrm{IRProgram}
\xrightarrow{\mathrm{Renderer}_{T}}
\mathrm{Schema}_{T}
\xrightarrow{\mathrm{emit\_pretty}}
\mathrm{bytes}.
$$

`Lower` resolves program steps into target-neutral IR nodes. A renderer builds a panproto schema for one target grammar, and `emit_pretty` serializes that schema. The target registry in `quivers.transpile._RENDERERS` also associates each backend with a construct-support tier.

The pipeline is deterministic on the fixtures covered by `tests/transpile/test_lens_laws.py`. Those tests establish renderer determinism, equality between direct and composed pipeline invocation, and a parse/re-emit fixed point. They are pipeline-composition tests; they are not bidirectional GetPut/PutGet lens proofs.

## 2. Correctness criterion

For a supported QVR fixture $M$, target $T$, parameter point $\theta$, and observed data $y$, the numeric suite compares

$$
\ell_{\mathrm{QVR}}(\theta;y)
\quad\text{and}\quad
\ell_T(\theta;y).
$$

The comparison admits a target-dependent additive constant because probabilistic libraries may omit normalizers that do not depend on $\theta$. The helper thus checks that

$$
\ell_T(\theta_i;y)-\ell_{\mathrm{QVR}}(\theta_i;y)
$$

is approximately constant over the selected point set. This is the **constant-spread criterion** (CSC).

The CSC is useful but limited. A finite grid cannot establish equality on the full parameter space, and a constant-spread match for one data set does not prove equality for every observation. It also does not compare posterior samplers, random-number streams, gradients, generated quantities, or performance.

## 3. Support is explicit

Correctness is conditional on support. The frontend calls `unsupported_for(...)`, family lowering consults `FAMILY_META` in `src/quivers/transpile/family_meta.py`, and renderers may raise `UnsupportedConstruct` for target-specific gaps. A rejected construct is preferable to an emit that silently changes meaning.

The family and construct matrices are executable inventories. They should be consulted instead of inferring support from a backend's native library. A target may have a native distribution that QVR does not yet map, or QVR may implement a distribution through an equivalent target expression rather than a same-named constructor.

## 4. Distribution parameterization

Family names alone do not establish equivalence. For each supported family, the renderer must preserve:

1. argument meaning and order;
2. event and batch shape;
3. support constraints;
4. any Jacobian term introduced by a change of variables;
5. observation and plate reduction conventions.

`FAMILY_META[F].target_names` and target argument aliases record part of this mapping. Helpers may implement the rest, including scale/rate conversions, truncation, or matrix reconstruction. The per-target pages document important cases, but the source and numeric tests remain authoritative for current behavior.

## 5. Plates, marginalization, and `via`

An indexed `sample` or `observe` denotes repeated sites over a finite axis. A renderer may express that repetition through vectorization, a plate context, or an explicit loop. Correctness requires the same product of per-site factors, with event dimensions excluded from the plate reduction.

Scoped `marginalize` has two principal target strategies:

- finite enumeration with log-sum-exp over the latent support;
- an explicit latent retained for a target inference engine that can handle it.

These strategies are not interchangeable for every inference algorithm. In particular, HMC and NUTS do not directly sample discrete latent variables. A target that keeps a discrete latent needs an inference method capable of discrete state or a separate marginalization strategy.

The `via=` form groups or reindexes observations through a finite map. Correctness requires that each observation contribute to the factor for its mapped group exactly once. `tests/transpile/test_via_fibration_numeric.py` checks selected cases numerically.

## 6. Evidence tiers

The test files use the following practical tiers:

| Tier | Test | What it checks |
|---|---|---|
| 1 | `test_structural.py`, `test_lens_laws.py` | emitted schema shape, renderer determinism, pipeline composition, re-emission |
| 2 | `test_external_syntax.py` | acceptance by a target parser/compiler when available |
| 3 | `test_numeric_equivalence.py` | CSC log-density comparison in target runtimes |
| 4 | `test_construct_matrix.py` | construct-by-backend compatibility |
| 5 | `test_family_matrix.py` | distribution-family support matrix |
| 6 | `test_composition_fixtures.py` | larger composition fixtures across targets |

Additional numeric suites cover closed-form marginals, scoped marginalization, `via` grouping, and documentation-gallery programs.

The main Tier-3 grid currently uses seven composition fixtures. It selects five values per parameter axis and caps the Cartesian grid at sixteen points per fixture. Expected unsupported cells are asserted explicitly. These numbers describe current coverage, not a completeness bound.

## 7. Runtime coverage

Docker images provide pinned runtimes for Stan, NumPyro, Pyro, PyMC, Edward2, Turing, Gen, WebPPL, JAGS, and the BUGS-syntax route through JAGS. Their build and probe contract is documented in `tests/transpile/docker/README.md`.

Church is absent from the shared Docker matrix because the repository does not provision a maintained Church runtime image with the same programmable joint-density interface. Church-specific audit tests provide narrower evidence and should not be reported as the same cross-runtime matrix.

External-syntax checks are also target-specific:

- Stan uses `stanc --info -`;
- Python targets use Python's AST parser;
- Julia targets use `Meta.parse`;
- WebPPL output is checked as JavaScript with `node --check`;
- BUGS and JAGS are compiled through JAGS;
- Church has no canonical external parser in the shared suite.

JavaScript syntax acceptance does not establish WebPPL-language acceptance, and JAGS acceptance of BUGS syntax does not establish behavior in every BUGS implementation.

## 8. Per-target notes

- [Stan](stan.md)
- [NumPyro](numpyro.md)
- [Pyro](pyro.md)
- [PyMC](pymc.md)
- [Edward2](edward2.md)
- [Turing](turing.md)
- [Gen](gen.md)
- [Church](church.md)
- [WebPPL](webppl.md)
- [BUGS](bugs.md)
- [JAGS](jags.md)

Each page describes emitted structure, parameter conversions, known gaps, and the evidence exercised for that target. Claims should remain no stronger than the cited source, target documentation, or repository test.

## 9. What remains unproved

The current suite does not prove:

- semantic equivalence for every point in a continuous parameter space;
- equivalence of posterior inference algorithms;
- preservation of gradients or unconstrained parameterizations across all targets;
- equivalence for constructs or families outside the executable matrices;
- equivalence between BUGS dialects;
- full Church or WebPPL language acceptance from generic parenthesis or JavaScript checks;
- correctness of target-library behavior itself.

The TCC is thus an incremental engineering contract: reject unsupported input, emit deterministic target syntax, and compare supported densities on reproducible cases. New family or construct support should add structural, syntax, and numeric evidence at the same time.
