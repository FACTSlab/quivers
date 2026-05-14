# Conceptual Guides

This section gives feature-area introductions to the mathematical and computational foundations of quivers. The guides assume familiarity with basic category theory or a willingness to learn it alongside the material. For a more discursive treatment, see the [tutorial](../tutorials/index.md); for the formal denotational semantics, see [Semantics](../semantics/index.md).

## Reading order

1. **[Core Types & Quantales](core.md).** Finite sets (the SetObject hierarchy), quantales as enrichment algebras, and the algebraic primitives that underpin all morphism composition.
2. **[Transformations & Composition Rules](transformations.md).** First-class change-of-base transformations (`softmax(B)`, `expectation`, etc.), the `>>>` composition operator, and the `CompositionRule → Semigroupoid → Quantale` hierarchy. Also covers operadic n-ary contractions via einsum-style wiring specs.
3. **[Morphisms & Composition](morphisms.md).** What a morphism is as a tensor in $\mathcal{V}^{|A| \times |B|}$. The morphism hierarchy, composition, tensor product, marginalization, and the compact-closed surface (`dagger`, `trace`, `cup`, `cap`).
4. **[Categorical Structures](categorical.md).** Higher-order constructions: functors, natural transformations, adjunctions, monoidal structures, base change.
5. **[Monads & Comonads](monadic.md).** Monadic abstractions, Kleisli / coKleisli categories, algebras, coalgebras, distributive laws.
6. **[Enriched Category Theory](enriched.md).** Advanced structures specific to $\mathcal{V}$-enrichment: ends, coends, Kan extensions, weighted limits, profunctors, Yoneda, Day convolution, optics.
7. **[Stochastic Morphisms](stochastic.md).** The FinStoch category: Markov kernels, conditioning, queries, the Giry monad.
8. **[Continuous Distributions](continuous.md).** ContinuousSpace and ContinuousMorphism: parameterized families, sampled composition, normalizing flows.
9. **[Monadic Programs](programs.md).** Probabilistic programming via sequential bind (`<-`), `let`, `observe`, `marginalize`, and `return` steps; ancestral sampling; log-joint computation; parametric program templates.
10. **[The QVR DSL](dsl.md).** Declarative specification: the `.qvr` file format, grammar, and compilation pipeline.
11. **[Variational Inference](inference.md).** The inference stack: `LatentRegistry`, guides, objectives, gradient estimators, MCMC kernels, hybrid samplers, predictive consumption.
12. **[Compositional Effects](effects.md).** Algebraic-effects framework over the residuated category universe.
13. **[Weighted Deduction Systems](deduction.md).** Agenda-engine runtime, semirings, charts as differentiable values, the seven canonical parameters.
14. **[Structural Compression](structural-compression.md).** `signature` / `encoder` / `decoder` / `loss` blocks; F-algebra / F-coalgebra surface for compressing structured objects.

## Quick navigation

- **Probabilistic programming.** Core Types, Stochastic, Monadic Programs, Variational Inference.
- **Hybrid discrete-continuous models.** Continuous Distributions, Monadic Programs, Inference.
- **Building models declaratively.** The QVR DSL, Transformations & Composition Rules, Inference.
- **Category-theoretic extension.** Categorical Structures, Monads & Comonads, Enriched Category Theory.
- **Structured prediction and parsing.** Weighted Deduction Systems, Structural Compression.
