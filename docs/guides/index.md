# Conceptual Guides

This section provides an introduction to the mathematical and computational foundations of quivers. The guides assume familiarity with basic category theory or a willingness to learn it alongside the material.

## Recommended Reading Order

1. **[Core Types & Quantales](core.md)** — Start here. Introduces finite sets (SetObject hierarchy), quantales as enrichment algebras, and the algebraic primitives that underpin all morphism composition.

2. **[Morphisms & Composition](morphisms.md)** — Define what a morphism is as a tensor in $\mathcal{V}^{|A| \times |B|}$. Covers the morphism hierarchy, composition, and algebraic operations.

3. **[Categorical Structures](categorical.md)** — Higher-order constructions: functors, natural transformations, adjunctions, monoidal structures, and base change.

4. **[Monads & Comonads](monadic.md)** — Monadic abstractions (unit, multiply, map), Kleisli categories, algebras, and coalgebras.

5. **[Enriched Category Theory](enriched.md)** — Advanced structures specific to $\mathcal{V}$-enrichment: ends, coends, Kan extensions, weighted limits, profunctors, Yoneda, Day convolution, optics.

6. **[Stochastic Morphisms](stochastic.md)** — The FinStoch category: Markov kernels, conditioned distributions, queries, and the Giry monad.

7. **[Continuous Distributions](continuous.md)** — ContinuousSpace and ContinuousMorphism: parameterized families, sampled composition, normalizing flows.

8. **[Monadic Programs](programs.md)** — Probabilistic programming via sequential draw and let steps, ancestral sampling, and log-joint computation.

9. **[The QVR DSL](dsl.md)** — Declarative specification of categorical networks: the `.qvr` file format, grammar, and compilation pipeline.

10. **[Variational Inference](inference.md)** — Inference pipeline: tracing, conditioning, automatic guides, ELBO, SVI, and predictive sampling.

## Quick Navigation

- **For discrete & finite:** Core Types → Morphisms → Categorical
- **For probabilistic models:** Core Types → Stochastic → Monadic Programs
- **For hybrid discrete-continuous:** Continuous Distributions → Monadic Programs
- **For building models declaratively:** The QVR DSL → Variational Inference
- **For enriched category theory:** Monads & Comonads → Enriched Category Theory
