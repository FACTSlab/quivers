# Changelog

All notable changes to the quivers library are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-05-06

### Changed

- Every record-shaped value type (AST nodes, `FinSet`, `ProductSet`, `CoproductSet`, `ContinuousSpace` variants, `Category` variants, `RuleSystem`) is now a `didactic.api.Model`. Recursive sums are `dx.TaggedUnion` roots discriminated by a `kind: Literal[...]` field. JSON round-trips via `model_dump_json` / `model_validate_json` are available on every value type.
- Resolution from `TypeExpr` / `SpaceExpr` AST trees to runtime `SetObject` / `ContinuousSpace` values is expressed as a `dx.Lens` family in `quivers.dsl.resolution`.
- Variadic constructors `ProductSet(A, B, C)` and `CoproductSet(A, B, C)` are replaced by keyword form `ProductSet(components=(A, B, C))` and `CoproductSet(components=(A, B, C))`. The flattening converter preserves the previous flattening behaviour.
- Continuous spaces (`Euclidean`, `Simplex`, `PositiveReals`, `ProductSpace`) expose public `name` and `dim` fields (no longer private with property accessors).
- Minimum supported Python is now 3.14.

### Added

- A tree-sitter grammar for the QVR DSL at `grammars/qvr/`, registered with panproto's `panproto-grammars-all` distribution.
- `quivers.dsl.parser` delegates parsing to panproto via the `qvr` tree-sitter grammar.
- `quivers.dsl.program_theory` defines `QVR_PROGRAM_PROTOCOL` and `extract_program_schema`, lifting every compiled program to a panproto `Schema` for use with `panproto schema diff`, `panproto lens generate`, and related tooling.
- A `Denotational Semantics` documentation section giving a formal, compositional semantics for the DSL across the discrete, stochastic, and continuous strata, plus an adequacy theorem connecting the compiler implementation to the denotation.
- `RuleSystem` carries cross-field axioms (`__axioms__`) ensuring `binary_weights`/`unary_weights` lengths match `binary_rules`/`unary_rules` when supplied.
- `.github/workflows/release.yml` builds an sdist + wheel on tag push and publishes to PyPI via the FACTSlab/quivers OIDC trusted publisher.
- Pull request template under `.github/PULL_REQUEST_TEMPLATE.md` and issue templates under `.github/ISSUE_TEMPLATE/`.

### Removed

- `quivers.dsl.lexer` and `quivers.dsl.tokens`: the hand-written lexer is replaced by panproto's tree-sitter–integrated lexing.
- `LexError`: lexical errors now surface as `ParseError`.

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
