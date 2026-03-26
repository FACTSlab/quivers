# Architecture

The quivers library is organized into eight subpackages, each providing a layer of categorical structure. This document describes the package hierarchy, dependencies, and key abstractions in each module.

## Package Structure

```
quivers/
├── core/              # foundational types and operations
├── categorical/       # functors, transformations, adjunctions, monoidal
├── monadic/           # monads, comonads, algebras, distributive laws
├── enriched/          # ends/coends, Kan, weighted limits, profunctors, Yoneda, optics
├── stochastic/        # FinStoch (Markov kernels), families, conditioning, Giry monad
├── continuous/        # parameterized families, boundaries, flows, monadic programs
├── dsl/               # lexer, parser, compiler for .qvr files
├── inference/         # trace-based conditioning, variational guides, SVI
├── program.py         # nn.Module wrapper for morphisms
└── giry.py            # backward-compatibility shim
```

## Module Descriptions

### `core/`

Foundational categorical types and operations. Defines the objects and morphisms that all other modules build on.

- **`objects.py`** — Categorical objects: `FinSet` (finite sets), `ProductSet`, `CoproductSet`, `FreeMonoid`, `Unit`.
- **`quantales.py`** — Enrichment lattices: `Quantale` (abstract), `ProductFuzzy`, `BooleanQuantale`, and singletons `PRODUCT_FUZZY`, `BOOLEAN`.
- **`morphisms.py`** — $\mathcal{V}$-enriched relations as tensors: `Morphism` (abstract), `ObservedMorphism` (fixed tensor data), `LatentMorphism` (learnable), `ComposedMorphism`, `ProductMorphism`, `MarginalizedMorphism`, plus factory functions `morphism()`, `observed()`, `identity()`.
- **`tensor_ops.py`** — Tensor operations: `noisy_or_contract`, `noisy_or_reduce`, `noisy_and_reduce`, `componentwise_lift`.
- **`extra_quantales.py`** — Additional quantales: `LukasiewiczQuantale`, `GodelQuantale`, `TropicalQuantale`.

### `categorical/`

Categorical structures built from objects and morphisms.

- **`functors.py`** — Functors and functor composition: `Functor`, `IdentityFunctor`, `ComposedFunctor`, `FreeMonoidFunctor`.
- **`natural_transformations.py`** — Natural transformations: `NaturalTransformation`, `ComponentwiseNT`.
- **`adjunctions.py`** — Adjunctions and free-forgetful pairs: `Adjunction`, `ForgetfulFunctor`, `FreeForgetfulAdjunction`.
- **`monoidal.py`** — Monoidal structures: `MonoidalStructure`, `CartesianMonoidal`, `CoproductMonoidal`, `EmptySet`, `EMPTY`.
- **`base_change.py`** — Base change between quantales: `BaseChange`, `BoolToFuzzy`, `FuzzyToBool`.
- **`traced.py`** — Traced monoidal categories: `TracedMonoidal`, `CartesianTrace`, `IterativeTrace`, functions `trace()`, `partial_trace()`.

### `monadic/`

Monadic structures: monads, comonads, and their algebras.

- **`monads.py`** — Monads and Kleisli categories: `Monad`, `KleisliCategory`, `FuzzyPowersetMonad`, `FreeMonoidMonad`.
- **`comonads.py`** — Comonads and coKleisli categories: `Comonad`, `CoKleisliCategory`, `DiagonalComonad`, `CofreeComonad`.
- **`algebras.py`** — Eilenberg-Moore structure: `Algebra`, `FreeAlgebra`, `ObservedAlgebra`, `Coalgebra`, `CofreeCoalgebra`, `ObservedCoalgebra`, `EilenbergMooreCategory`.
- **`distributive_laws.py`** — Composing monads: `DistributiveLaw`, `FreeMonoidPowersetLaw`.

### `enriched/`

Enriched category theory: the calculus of $\mathcal{V}$-valued hom-functors.

- **`ends_coends.py`** — Limits and colimits in enriched categories: `coend()`, `end()`.
- **`kan_extensions.py`** — Kan extensions: `ObjectMap`, `Projection`, `Inclusion`, `left_kan()`, `right_kan()`.
- **`weighted_limits.py`** — Weighted (co)limits: `Weight`, `Diagram`, `weighted_limit()`, `weighted_colimit()`, `representable_weight()`, `terminal_weight()`.
- **`profunctors.py`** — Profunctors (distributor, bimodules): `Profunctor`.
- **`yoneda.py`** — Yoneda lemma and embedding: `Presheaf`, `representable_profunctor()`, `corepresentable_profunctor()`, `yoneda_embedding()`, `yoneda_lemma()`, `yoneda_density()`, `verify_yoneda_fully_faithful()`.
- **`day_convolution.py`** — Day convolution for monoidal categories: `day_convolution()`, `day_unit()`, `day_convolution_profunctors()`.
- **`optics.py`** — Optics and their composition: `Optic`, `Lens`, `Prism`, `Adapter`, `Grate`, `compose_optics()`.

### `stochastic/`

Stochastic morphisms and the category **FinStoch** of Markov kernels on finite sets.

- **`quantale.py`** — Markov quantale: `MarkovQuantale`, singleton `MARKOV`.
- **`morphisms.py`** — Stochastic morphisms: `StochasticMorphism`, `CategoricalMorphism`, `ConditionedMorphism`, `MixtureMorphism`, `FactoredMorphism`, `NormalizedMorphism`.
- **`families.py`** — Discretized parameterized distributions: `DiscretizedNormal`, `DiscretizedLogitNormal`, `DiscretizedBeta`, `DiscretizedTruncatedNormal`.
- **`transforms.py`** — Transforms on stochastic morphisms: `condition()`, `mix()`, `factor()`, `normalize()`.
- **`queries.py`** — Query functions: `prob()`, `marginal_prob()`, `expectation()`.
- **`giry.py`** — The Giry monad on FinStoch: `GiryMonad`, `FinStoch`.

### `continuous/`

Continuous and hybrid discrete-continuous morphisms and monadic programs.

- **`spaces.py`** — Continuous sample spaces: `ContinuousSpace`, `Euclidean`, `UnitInterval`, `Simplex`, `PositiveReals`, `ProductSpace`.
- **`morphisms.py`** — Continuous morphisms: `ContinuousMorphism`, `SampledComposition`, `ProductContinuousMorphism`, `DiscreteAsContinuous`.
- **`families.py`** — 30+ conditional distribution families: `ConditionalNormal`, `ConditionalLogitNormal`, `ConditionalBeta`, `ConditionalTruncatedNormal`, `ConditionalDirichlet`, `ConditionalCauchy`, `ConditionalLaplace`, `ConditionalGumbel`, `ConditionalLogNormal`, `ConditionalStudentT`, `ConditionalExponential`, `ConditionalGamma`, `ConditionalChi2`, `ConditionalHalfCauchy`, `ConditionalHalfNormal`, `ConditionalInverseGamma`, `ConditionalWeibull`, `ConditionalPareto`, `ConditionalKumaraswamy`, `ConditionalContinuousBernoulli`, `ConditionalFisherSnedecor`, `ConditionalUniform`, `ConditionalMultivariateNormal`, `ConditionalLowRankMVN`, `ConditionalRelaxedBernoulli`, `ConditionalRelaxedOneHotCategorical`, `ConditionalWishart`, `ConditionalBernoulli`, `ConditionalCategorical`.
- **`programs.py`** — Monadic programs: `MonadicProgram` (compositional probabilistic computations).
- **`boundaries.py`** — Boundaries between discrete and continuous: `Discretize`, `Embed`.
- **`flows.py`** — Normalizing flows: `AffineCouplingLayer`, `ConditionalFlow`.

### `dsl/`

Domain-specific language for quiver expressions in `.qvr` files.

- **`tokens.py`** — Token types.
- **`lexer.py`** — Lexical analysis: `Lexer`.
- **`parser.py`** — Syntax analysis: `Parser`.
- **`ast_nodes.py`** — Abstract syntax tree node definitions.
- **`compiler.py`** — Code generation from AST: `Compiler`.

Top-level DSL API: `parse()`, `loads()`, `load()`, plus exceptions `LexError`, `ParseError`, `CompileError`.

### `inference/`

Variational inference for posterior estimation in monadic programs.

- **`trace.py`** — Trace and conditioning: `trace()`, `condition()`.
- **`conditioning.py`** — Conditioning primitives.
- **`guide.py`** — Variational guides: `Guide`.
- **`elbo.py`** — Evidence lower bound: `ELBO`.
- **`svi.py`** — Stochastic variational inference: `SVI`.
- **`predictive.py`** — Predictive posterior: `Predictive`.

### Root-level modules

- **`program.py`** — `Program`: wraps a morphism (discrete or continuous) as a differentiable `nn.Module`.
- **`giry.py`** — Backward-compatibility wrapper; re-exports `GiryMonad` and `FinStoch` from `quivers.stochastic.giry`.

## Dependency Graph

The package has a layered dependency structure:

```
┌─ core (no quivers dependencies)
│
├─ categorical (depends on: core)
│
├─ monadic (depends on: core)
│
├─ enriched (depends on: core, categorical)
│
├─ stochastic (depends on: core, monadic)
│
├─ continuous (depends on: core, categorical, monadic, stochastic)
│
├─ dsl (depends on: core, categorical, monadic, stochastic, continuous)
│
└─ inference (depends on: core, categorical, stochastic, continuous)

program.py (depends on: core, continuous)
giry.py (depends on: stochastic)
```

### Dependency Hierarchy

1. **core** — foundational (no internal dependencies)
2. **categorical**, **monadic** — build on core
3. **enriched** — builds on categorical
4. **stochastic** — builds on core and monadic
5. **continuous** — builds on core, categorical, monadic, and stochastic
6. **dsl** — synthesizes core, categorical, monadic, stochastic, continuous
7. **inference** — builds on core, categorical, stochastic, continuous

This layering ensures that users can work at the level of abstraction appropriate to their task: core algebra only, categorical structures, probabilistic computations, or the full DSL.

## Design Principles

- **Tensors as morphisms**: $\mathcal{V}$-relations are represented as PyTorch tensors, making them amenable to automatic differentiation and GPU acceleration.
- **Lazy composition**: morphism composition creates a DAG; the final tensor is materialized only on evaluation.
- **Type safety**: categorical constraints (domain/codomain compatibility) are enforced statically in Python.
- **Module transparency**: all morphisms expose a `.module()` method returning an `nn.Module`, enabling integration with PyTorch training loops.
- **Quantale flexibility**: the same morphism structure works with any quantale; swapping quantales changes semantics (fuzzy, Boolean, tropical, etc.) without code duplication.
