# Changelog

All notable changes to the quivers library are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-03-26

### Added

#### Core Categorical Algebra

- Fundamental category types and morphisms
- Object declarations and morphism composition
- Support for latent and observed morphisms
- Basic categorical operations and abstractions

#### Stochastic Morphisms

- Stochastic morphism declarations and semantics
- Integration with probability theory
- Support for morphism composition in stochastic settings

#### Continuous Distributions (30+ Families)

- Normal distribution and variants (LogitNormal, TruncatedNormal)
- Beta, Dirichlet for probability simplices
- Exponential family: Exponential, Gamma, Chi2
- Heavy-tailed: Cauchy, StudentT, Pareto
- Bounded: Uniform, Kumaraswamy
- Half-variants: HalfCauchy, HalfNormal
- Transformed: LogNormal, Gumbel, Laplace, Weibull
- Multivariate: MultivariateNormal, LowRankMVN, Wishart
- Bernoulli variants: Bernoulli, ContinuousBernoulli, RelaxedBernoulli
- Advanced: RelaxedOneHotCategorical, FisherSnedecor
- Normalized flows: Flow
- Categorical and discrete approximations

#### Monadic Programs

- Draw statements for sampling from morphisms
- Observe statements for conditioning and likelihood
- Return statements with optional labeled outputs
- Variable binding and destructuring in patterns
- Program parameters and composition

#### QVR DSL

- Complete lexer with token recognition for all language constructs
- Recursive descent parser with full grammar support
- Abstract syntax tree (AST) node definitions
- Program block execution with proper scoping
- Let bindings for expression computation
- Built-in let functions: sigmoid, exp, log, abs, softplus
- Comment support (#)
- Type expressions: products (*), coproducts (+)
- Expression operators: composition (>>), tensor product (@), marginalization
- Indentation-aware program body parsing
- Specialized handling for draw/observe arguments

#### Variational Inference Layer

- Inference interface for probabilistic programs
- Support for approximate posterior computation
- Integration with continuous distribution families
