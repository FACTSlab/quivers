# API Reference

This section documents the public API of the quivers library, organized by functional subpackages. Each module is documented with its classes, functions, and properties using Sphinx-style docstrings.

## Core Package

The `core` package provides the fundamental categorical and algebraic structures that underpin all other modules.

- **Objects**: Finite sets, products, coproducts, and free monoids
- **Algebras**: Enrichment algebras for $\mathcal{V}$-enriched composition (the eleven built-in cases plus duals and user-defined extensions)
- **Algebra Morphisms**: Homomorphisms between algebras, used for base-change
- **Morphisms**: Base morphism classes and composition operations
- **Tensor Operations**: Vectorized operations on tensors
- **Utilities**: Helper functions for core functionality

## Categorical Package

The `categorical` package implements standard category theory concepts and structures.

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

The `enriched` package extends category theory with enrichment and advanced constructions.

- **Ends & Coends**: End and coend computations in enriched categories
- **Kan Extensions**: Left and right Kan extensions
- **Weighted Limits**: Limits and colimits weighted by enrichment
- **Profunctors**: Profunctor (bimodule) definitions
- **Yoneda**: Yoneda embeddings and lemmas
- **Day Convolution**: Day convolution product
- **Optics**: Optics and lens constructions

## Stochastic Package

The `stochastic` package provides stochastic morphisms and probability distributions.

- **Morphisms**: Stochastic relations and kernels
- **Families**: Parametric families of distributions
- **Transforms**: Operations on stochastic morphisms
- **Queries**: Probabilistic queries and computations
- **Giry Monad**: The Giry monad construction

## Continuous Package

The `continuous` package handles continuous-valued distributions and spaces.

- **Spaces**: Continuous topological spaces
- **Morphisms**: Continuous mappings
- **Families**: Families of continuous distributions
- **Programs**: Probabilistic programs in continuous domains
- **Boundaries**: Boundary conditions and constraints
- **Flows**: Normalizing flows and transformations

## DSL Package

The `dsl` package implements the QVR domain-specific language for quivers. Parsing is delegated to panproto via the `qvr` tree-sitter grammar; AST nodes and value types are didactic Models.

- **Parser**: panproto-driven parser walker (`parse`, `parse_file`, `ParseError`)
- **AST Nodes**: didactic Model definitions for every syntactic node
- **Compiler**: lowering from AST to `Program` (`Compiler`, `CompileError`)
- **Resolution**: bidirectional `dx.Lens` family from `ObjectExpr` to runtime values
- **Program Theory**: `QVR_PROGRAM_PROTOCOL` and `extract_program_schema` for emitting a panproto `Schema` from a compiled program

## Inference Package

The `inference` package provides variational inference capabilities.

- **Trace**: Program trace data structures
- **Conditioning**: Conditioning and observations
- **Guides**: Variational guide distributions
- **ELBO**: Evidence lower bound computation
- **SVI**: Stochastic variational inference
- **Predictive**: Predictive inference and sampling

## Root Module

- **Program**: Top-level probabilistic program definitions
