# API Reference

This section documents the public API by subpackage. Module pages draw their classes, functions, and properties from the source docstrings.

## Core Package

The `core` package defines the categorical objects, algebras, morphisms, and tensor operations used by the other packages.

- **Objects**: Finite and enumerated sets, products, coproducts, free monoids, and free residuated universes
- **Algebras**: Enrichment algebras for $\mathcal{V}$-enriched composition (the eleven built-in cases plus duals and user-defined extensions)
- **Algebra Morphisms**: Homomorphisms between algebras, used for base-change
- **Morphisms**: Base morphism classes and composition operations
- **Tensor Operations**: Vectorized operations on tensors
- **Utilities**: Helper functions for core functionality

## Categorical Package

The `categorical` package implements the library's functors, natural transformations, adjunctions, and monoidal structures.

- **Functors**: Mappings between categories preserving structure
- **Natural Transformations**: Morphisms between functors
- **Adjunctions**: Adjoint functor pairs
- **Monoidal**: Monoidal category structures
- **Base Change**: Change of enriching category
- **Traced**: Traced monoidal categories

## Monadic Package

A typeclass hierarchy plus concrete monad instances, comonads,
algebras, distributive laws, transformers, and algebraic effects.

- **Typeclass Hierarchy**: `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`, `Foldable`, `Traversable`, `MonadTrans` ABCs
- **Stdlib Effect Instances**: `Identity`, `Maybe`, `Alternative_`, `Continuation`, `State`, `Reader`, `Writer`, `List`
- **Monad Transformers**: `StateT`, `ReaderT`, `MaybeT`, `ContT`, `WriterT`
- **Algebraic Effects**: `Operation`, `EffectSignature`, `Handler`, `FreeMonad`
- **Bridges**: `Kleisli` / `ArrowMonad` connecting the monad and arrow towers
- **Typeclass Theories**: panproto-theory mirrors
- **Concrete Monads**: `KleisliCategory`, `FuzzyPowersetMonad`, `FreeMonoidMonad`
- **Comonads**, **Algebras**, **Distributive Laws**

## Arrows Package

Hughes-style arrow tower (parallel to the monad-side hierarchy).

- **Typeclass Hierarchy**: `Category_`, `Arrow`, `ArrowChoice`, `ArrowApply`, `ArrowLoop`, `ArrowZero`, `ArrowPlus`
- **Arrow Theories**: panproto-theory mirrors

## Enriched Package

The `enriched` package implements constructions for categories enriched over a selected algebra.

- **Ends & Coends**: End and coend computations in enriched categories
- **Kan Extensions**: Left and right Kan extensions
- **Weighted Limits**: Limits and colimits weighted by enrichment
- **Profunctors**: Profunctor (bimodule) definitions
- **Yoneda**: Yoneda embeddings and lemmas
- **Day Convolution**: Day convolution product
- **Optics**: Optics and lens constructions

## Stochastic Package

The `stochastic` package provides stochastic morphisms, finite distribution families, and weighted deduction systems.

- **Morphisms**: Stochastic relations and kernels
- **Families**: Parametric families of distributions
- **Transforms**: Operations on stochastic morphisms
- **Queries**: Probabilistic queries and computations
- **Giry Monad**: The Giry monad construction
- **Weighted Deduction**: Rule schemas, chart semirings, and CKY parsers

## Continuous Package

The `continuous` package defines continuous spaces, distribution families, parameter sources, and continuous morphisms.

- **Spaces**: Typed carriers and support constraints for continuous values
- **Morphisms**: Operational kernels with log-density and sampling methods
- **Families**: Families of continuous distributions
- **Programs**: Probabilistic programs in continuous domains
- **Boundaries**: Discretization and embedding between finite and continuous carriers
- **Flows**: Normalizing flows and transformations

## DSL Package

The `dsl` package implements the QVR domain-specific language for quivers. Parsing is delegated to panproto via the `qvr` tree-sitter grammar; AST nodes and value types are didactic Models.

- **Parser**: panproto-driven parser walker (`parse`, `parse_file`, `ParseError`)
- **AST Nodes**: didactic Model definitions for every syntactic node
- **Compiler**: lowering from AST to `Program` (`Compiler`, `CompileError`)
- **Resolution**: bidirectional `dx.Lens` family from `ObjectExpr` to runtime values
- **Program Theory**: `QVR_PROGRAM_PROTOCOL` and `extract_program_schema` for emitting a panproto `Schema` from a compiled program

## Inference Package

The `inference` package provides variational inference, MCMC, and posterior-predictive sampling.

- **Trace**: Program trace data structures
- **Conditioning**: Conditioning and observations
- **Guides**: Variational guide distributions
- **ELBO**: Evidence lower bound computation
- **SVI**: Stochastic variational inference
- **Predictive**: Predictive inference and sampling
- **MCMC**: HMC and NUTS kernels plus the multi-chain runner

## Root Module

- **Program**: Top-level probabilistic program definitions
