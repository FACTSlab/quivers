# API Reference

This section documents the public API of the quivers library, organized by functional subpackages. Each module is documented with its classes, functions, and properties using Sphinx-style docstrings.

## Core Package

The `core` package provides the fundamental categorical and quantale structures that underpin all other modules.

- **Objects**: Finite sets, products, coproducts, and free monoids
- **Quantales**: Ordered algebraic structures for morphism composition
- **Extra Quantales**: Specialized quantale implementations
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

The `monadic` package covers monad and comonad theory and their algebraic structures.

- **Monads**: Monad definitions and operations
- **Comonads**: Comonad definitions and operations
- **Algebras**: Monad and comonad algebras and coalgebras
- **Distributive Laws**: Distributive laws between monads

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

- **Quantale**: The [0,1] quantale for probability
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

The `dsl` package implements the QVR domain-specific language for quivers.

- **Compiler**: Compilation from AST to quivers
- **Parser**: Parsing DSL syntax
- **Lexer**: Tokenization of DSL input
- **AST Nodes**: Abstract syntax tree node definitions
- **Tokens**: Token definitions and types

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
