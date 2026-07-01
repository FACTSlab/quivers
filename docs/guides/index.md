# Conceptual Guides

This section gives feature-area introductions to the mathematical
and computational foundations of quivers. The guides assume
familiarity with basic category theory or a willingness to learn
it alongside the material. For a more discursive treatment, see
the [tutorial](../tutorials/index.md); for the formal denotational
semantics, see [Semantics](../semantics/index.md).

The guides are organized in seven thematic sections; each
section's pages stand alone, but reading them in order gives the
clearest picture of how the pieces fit together.

## Foundations

The categorical primitives every other layer is built on.

1. [Core Types & Algebras](core.md). Finite sets (the SetObject
   hierarchy), algebras as enrichment algebras, and the algebraic
   primitives that underpin all morphism composition.
2. [Transformations & Composition Rules](transformations.md).
   First-class change-of-base transformations, the `>>>`
   composition operator, and the
   `CompositionRule → Semigroupoid → Algebra` hierarchy.
3. [Morphisms & Composition](morphisms.md). What a morphism is as
   a tensor in $\mathcal{V}^{|A| \times |B|}$, the morphism
   hierarchy, composition, tensor product, marginalization, and
   the compact-closed surface (`dagger`, `trace`, `cup`, `cap`).

## DSL

Writing programs in the typed `.qvr` language.

4. [DSL Overview](dsl-overview.md). File format, grammar, doc
   comments, compilation pipeline, error handling.
5. [DSL Declarations](dsl-declarations.md). `composition`,
   `object`, `morphism` (with the `[role=...]` option block
   selecting `latent` / `observed` / `kernel` / `embed` /
   `discretize`), `bundle`, the fan / repeat / stack / scan /
   curry combinators, and `export`.
6. [DSL Programs and Let-Expressions](dsl-programs-and-lets.md).
   `program` blocks, bind / observe / marginalize / let steps, the
   axis-role clause, factor expressions, the let-expression
   primitive surface, and inline distributions.
7. [DSL Contractions](dsl-contractions.md). Operadic n-ary
   contractions, type-driven wiring inference, `share` clause,
   explicit `wiring` escape hatch.

## Probabilistic Programming

The runtime semantics of programs and distributions.

8. [Monadic Programs](programs.md). Probabilistic programming via
   sequential bind, `let`, `observe`, and `return` steps; ancestral
   sampling; log-joint computation.
9. [Hierarchical Programs](programs-hierarchical.md). Parametric
   templates for crossed random intercepts, monotone-spline
   coefficients, and grouped marginalization over fibred discrete
   latents.
10. [Continuous Spaces and Morphisms](continuous-spaces.md). The
    `ContinuousSpace` hierarchy, the
    [`ContinuousMorphism`](../api/continuous/morphisms.md)
    interface, sampled composition, the discrete / continuous
    boundary, normalizing flows.
11. [Continuous Families](continuous-families.md). The 30+
    parameterized distribution registry, event ranks, and the
    structured priors (MatrixNormal, InverseWishart, GP,
    Horseshoe, LKJ) that interact with the axis-role surface.
12. [Stochastic Morphisms](stochastic.md). The FinStoch category:
    Markov kernels, conditioning, queries, the Giry monad.

## Inference

Fitting models to data.

13. [Inference Foundations](inference-foundations.md). The
    six-layer inference stack, trace and sample-site interface,
    conditioning.
14. [Variational Inference: Guides, Objectives, and
    SVI](inference-svi.md). The `Auto*Guide` family, ELBO / IWAEBound /
    Rényi / VR-IWAE objectives, gradient estimators, SVI loop,
    predictive sampling.
15. [Variational Inference: MCMC and Hybrid
    Samplers](inference-mcmc.md). HMC and NUTS kernels, `AutoDAIS`,
    `WarmupThenHMC`, predictive sampling from MCMC chains.

## Categorical Structures

Higher-order categorical machinery the rest of the library is
built on or reuses.

16. [Categorical Structures](categorical.md). Functors, natural
    transformations, adjunctions, monoidal structures, base change.
17. [Monads & Comonads](monadic.md). Monadic abstractions, Kleisli
    / coKleisli categories, algebras, coalgebras, distributive
    laws.
18. [Enriched Category Theory](enriched.md). Ends, coends, Kan
    extensions, weighted limits, profunctors, Yoneda, Day
    convolution, optics.
19. [Compositional Effects](effects.md). Algebraic-effects
    framework over the residuated category universe.

## Structured Prediction

The chart-parser and structural-compression substrate.

20. [Weighted Deduction Systems](deduction.md). Agenda-engine
    runtime, semirings, charts as differentiable values, the seven
    canonical parameters.
21. [Structural Compression: Signatures and
    Encoders](structural-compression-signatures-and-encoders.md).
    `signature` and `encoder` blocks; F-algebra surface for
    compressing structured objects; factory form and
    sequence / graph sugar.
22. [Structural Compression: Decoders and
    Losses](structural-compression-decoders-and-losses.md).
    Kleisli-coalgebra `decoder` blocks, `loss` declarations,
    deduction and Bayesian integration.

## Analysis

Workflow surface around inference.

23. [Analysis: Data and Formulas](analysis-data-and-formulas.md).
    `DatasetSchema`, brms-style formulas, the formula-to-QVR
    compile lens, family registry.
24. [Analysis: Fitting and
    Diagnostics](analysis-fitting-and-diagnostics.md). One-line
    `fit`, ArviZ-based diagnostics, algebra-guided training tooling
    (`ChainShape`, `recommend_init`, `saturation_warnings`),
    autograd-safe morphism transforms.

## Quick navigation

- **Probabilistic programming.** [Core](core.md), [Stochastic](stochastic.md),
  [Monadic Programs](programs.md),
  [Inference Foundations](inference-foundations.md), [SVI](inference-svi.md).
- **Hybrid discrete-continuous models.**
  [Continuous Spaces](continuous-spaces.md),
  [Continuous Families](continuous-families.md),
  [Monadic Programs](programs.md),
  [Inference](inference-foundations.md).
- **Building models declaratively.** [DSL Overview](dsl-overview.md),
  [DSL Declarations](dsl-declarations.md),
  [DSL Programs](dsl-programs-and-lets.md),
  [Transformations](transformations.md),
  [Inference](inference-foundations.md).
- **Category-theoretic extension.**
  [Categorical Structures](categorical.md),
  [Monads & Comonads](monadic.md),
  [Enriched Category Theory](enriched.md).
- **Structured prediction and parsing.**
  [Weighted Deduction Systems](deduction.md),
  [Structural Compression: Signatures](structural-compression-signatures-and-encoders.md),
  [Structural Compression: Decoders](structural-compression-decoders-and-losses.md).
- **Working with fitted models.**
  [Inference Foundations](inference-foundations.md),
  [SVI](inference-svi.md), [MCMC](inference-mcmc.md),
  [Analysis: Data](analysis-data-and-formulas.md),
  [Analysis: Fitting](analysis-fitting-and-diagnostics.md).
