# Tutorials

Quivers exposes the same model through two surfaces: a `.qvr` DSL aimed at people who write probabilistic programs, and a typed Python API aimed at people who build libraries on top of the category-theoretic core. These tutorials are organized into two parallel tracks accordingly. Pick the one that matches what you're trying to do.

## QVR DSL track

If you have written a model in Pyro, NumPyro, Stan, or PyMC and want to do the same thing in quivers, start here. The `.qvr` syntax is the primary user-facing surface: you declare types, write a `program` block whose steps look like `v <- Normal(0, 1)` or `observe y <- Bernoulli(p)`, and fit it with SVI or NUTS. Category theory is mostly invisible; the categorical machinery is the implementation, not the API.

Each chapter shows the QVR version of a familiar model alongside its Pyro / NumPyro / Stan equivalent, then explains what's different and why.

1. **[Your first model](qvr/01-first-model.md)**. Linear regression. Side-by-side with PyMC. SVI fit, posterior predictive check, point summaries.
2. **[Generalized linear models](qvr/02-glms.md)**. Logistic and Poisson regression. Link functions. Posterior calibration plots.
3. **[Hierarchical models](qvr/03-hierarchical.md)**. Random intercepts; the eight-schools model; centered vs non-centered parameterizations; running NUTS with diagnostics.
4. **[Mixtures and discrete latents](qvr/04-marginalize.md)**. Finite mixtures and HMM-shaped models via the `marginalize` block: QVR's typed-scope marginalization, the feature that distinguishes it most from Pyro/NumPyro.
5. **[Sequence models](qvr/05-time-series.md)**. Plates, `scan`, and the deduction surface for chart-shaped models. State-space models and HMMs.
6. **[Choosing an inference algorithm](qvr/06-inference-zoo.md)**. A decision tree over the nine guides, four objectives, two MCMC kernels, and two hybrid samplers. Which combination matches which model shape.
7. **[Under the hood: the categorical surface](qvr/07-categorical.md)**. *(Optional reading.)* What QVR is doing underneath: quantales as enrichment algebras, `>>` as enriched composition, change-of-base as a functor. Useful when you want to extend the library or read the type errors fluently.

You can read the first six chapters without touching category theory. Chapter 7 is the bridge to the Python API track.

## Python API track

The Python API gives you direct access to the typed categorical surface: `FinSet`, `Morphism`, `Quantale`, `MonadicProgram`, the inference primitives, the structural-compression building blocks. Use this track if you are building tooling on top of quivers, extending the categorical machinery, or want to understand what the DSL compiles into.

1. **[Your first quiver](python/01-first-quiver.md)**. `FinSet` objects, observed and latent morphisms, the `>>` composition operator, `Program`.
2. **[Stochastic relations](python/02-stochastic-relations.md)**. Markov kernels and the FinStoch category. Conditioning, marginalization, expectation queries.
3. **[Probabilistic programs](python/03-probabilistic-programs.md)**. `MonadicProgram` by hand: continuous spaces, conditional families, bind (`<-`) / `let` / `observe` steps, sampling, log-density.
4. **[Fuzzy logic factorization](python/04-fuzzy-factorization.md)**. Factorizing an observed fuzzy relation into a composition of latents, training under product-fuzzy noisy-OR composition.
5. **[Variational inference](python/05-variational-inference.md)**. `Guide` + `Objective` + `SVI` + `Predictive`. Setting up the full inference loop end-to-end.
6. **[First-class transformations](python/06-first-class-trans.md)**. `MorphismTransformation` and `QuantaleHomomorphism` as values: let-binding, the `>>>` composition operator, change-of-base pipelines.
7. **[Composition rules beyond quantales](python/07-composition-rules.md)**. The `CompositionRule → Semigroupoid → Quantale` hierarchy, `BilinearForm`, and the operadic `EinsumWiring` surface for n-ary contractions.

## Prerequisites

For the QVR track:

- Python 3.14+, PyTorch 2.0+, quivers installed ([Installation](../getting-started/installation.md)).
- Comfort with one of the popular probabilistic-programming languages (Pyro, NumPyro, Stan, PyMC). You don't need to know category theory.

For the Python API track:

- Python and PyTorch as above.
- Working knowledge of category theory: objects, morphisms, composition, functors. The denotational [semantics](../semantics/index.md) section assumes Kelly-level enriched category theory; the tutorials don't, but a refresher on quantales as enrichment algebras ([Core Types & Quantales](../guides/core.md)) is recommended before chapter 4.

## How to read

Each chapter:

- States the model or feature in plain English first.
- Shows complete runnable code.
- Calls out what changed from the previous chapter.
- Ends with a "try this" exercise and pointers to the next chapter.

The chapters are independent enough that you can skim or skip, but each builds on the previous one's vocabulary, so out-of-order reading is easier in the second half.
