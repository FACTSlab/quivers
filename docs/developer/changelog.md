# Changelog

All notable changes to the quivers library are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — axis-role surface for distribution clauses

Every distribution clause (kernel declaration, latent parameter
prior, sample step, observe step) accepts an optional
`over <axes> [iid over <axes>]` post-clause.  `over` names the
**event axes** of the family; remaining axes are iid (categorically
a product of independent distributions).  Axis names resolve
against the named factors of the surrounding morphism's dom/cod;
`dom`/`cod` are shortcuts when that side is a single unfactored
object.

The axis count must match the family's declared `event_rank`
(0 / 1 / 2 for scalar / vector / matrix families); mismatch is a
compile-time error rather than a silent reinterpretation.  This
preserves the categorical distinction between, e.g., a flat MVN
over `dim(A)·dim(B)` with dense covariance and a `MatrixNormal`
over `(A, B)` with Kronecker structure `V ⊗ U`.

The latent morphism declaration accepts a `~ Family(args)
[options] [over <axes>]` clause that puts a prior on the
morphism's representing tensor.

### Added — new families

- [`ConditionalMatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal):
  Matrix-Normal with Kronecker covariance `V ⊗ U`.  Event rank 2.
  The natural prior for a weight matrix whose row and column
  correlations factor separately.
- [`ConditionalInverseWishart`](../api/continuous/families.md#quivers.continuous.families.ConditionalInverseWishart):
  Conjugate prior for the covariance of a multivariate normal
  ([Gelman et al. 2013](https://doi.org/10.1201/b16018) §3.6).
  Event rank 2.
- [`ConditionalGaussianProcess`](../api/continuous/families.md#quivers.continuous.families.ConditionalGaussianProcess):
  Gaussian-process prior with covariance kernel evaluated at the
  input locations. Supports RBF, Matern 5/2, and linear kernels.
  Registered as `GP` in the DSL; event rank 1
  ([Rasmussen & Williams 2006](http://www.gaussianprocess.org/gpml/)).
- [`ConditionalHorseshoe`](../api/continuous/families.md#quivers.continuous.families.ConditionalHorseshoe):
  Carvalho-Polson-Scott horseshoe prior with global-local shrinkage
  ([Carvalho, Polson & Scott 2010](https://doi.org/10.1093/biomet/asq017)).
  The marginal density (no closed form) is computed via 16-point
  Gauss-Legendre quadrature after a tangent change of variables on
  the auxiliary scale. Event rank 1.

### Changed — keyword unification

The `continuous` and `stochastic` declaration keywords are removed
in favor of a single `kernel` keyword.  A `kernel f : A -> B`
declaration without a `~` clause declares a finite-set lookup-table
kernel (the case formerly written with `stochastic`); with a
`~ Family [options] [axes]` clause it declares a parametric kernel
whose family parameters are produced by a neural parameter network
on the input (the case formerly written with `continuous`).  This
collapses two keywords whose distinction was already in the type
signature into one whose name reads as "Markov kernel" everywhere.

## [0.5.0] - 2026-05-11

### Changed (breaking, pre-1.0 clean cut)

Surface DSL homogenization. The program-block surface is reorganized
around a single Kleisli-bind sigil `<-`, type-annotated indexing on
the binder, scoped marginalization, and `!`-prefixed effect
signatures. See `CHANGELOG.md` for the full migration table.

Headline changes:

- `draw v ~ F` → `v <- F`. The `draw` keyword is retired; `<-` is
  the unique Kleisli-bind sigil.
- `draw v : A -> K ~ F` → `v : A <- F`. Indexed (plate) binds use
  a type annotation on the binder.
- `observe r[n] ~ F for n in N` → `observe r : N <- F`. The
  `for n in N` form is dropped; vectorized observes use the same
  type-annotation shape as plates.
- `marginalize c` → `marginalize c : A <- F in { … }`. Always
  scoped; the integration target is visible at the binding site.
- `posterior P (M) [v]` → `program P(v) : … ! Pure over M`.
- `!` effect signature on programs: `Sample`, `Score`,
  `Marginal`, `Pure`. The compiler verifies actual vs. declared
  effects.
- `output` → `export` (multiple per module allowed).
- Labeled-tuple return form dropped.

### Agenda-based weighted-deduction framework

Also in v0.5: a declarative `deduction { … }` block backed by a
general agenda-engine runtime (`quivers.stochastic.agenda`).

- One engine, seven parameters subsumes CKY, Earley, Viterbi,
  semi-naive Datalog, A* parsing, Knuth's algorithm, depth-first
  MLTT proof search.
- Charts as first-class differentiable values:
  `view.weight(item)`, `view.enumerate(pattern)`,
  `view.goal_weight()` return torch tensors with autograd.
- Pre-registered stdlib deductions in `quivers.stochastic.stdlib`:
  CCG, Lambek, STLC, MLTT, Datalog, Dijkstra, HMM, ViterbiHMM,
  EditDistance.
- panproto integration: `QVR_DEDUCTION_PROTOCOL` +
  `extract_deduction_schema(compiler)` makes deductions
  first-class schemas.

## [0.4.0] - 2026-05-11

### Added

- Hierarchical-Bayesian modeling primitives in `quivers.continuous.plate`, each carrying its categorical denotation in **Kern**:
  - `PlateDraw(index_size, family, domain)`, finite-domain-indexed draw realized as a Kern-morphism `A → B` by the natural isomorphism `Kern(1, B^A) ≅ Kern(A, B)`; subclass of `ContinuousMorphism` so it threads through the existing `MonadicProgram` step machinery.
  - `VectorisedObserve(family, response)`, batched-observation kernel `Φ → G_{≤1}(Φ)` with score `∏_n p_F(r_obs(n); θ(n, φ))`.
  - `marginalize_categorical(log_probs)`, program-level pushforward through `π_{Φ\C}` realized as `log_sum_exp` over the class axis.
  - `LKJCorrelationFactor(K, eta)`, LKJ prior on `CholeskyFactor(K)` via the Lewandowski-Kurowicka-Joe onion method; analytic `log_prob` matches Stan's `lkj_corr_cholesky_lpdf`.
  - `Truncated(base, lower, upper)`, generic interval-truncation combinator (rejection sampling with Monte-Carlo truncation-mass estimation).
  - `cumsum(K)`, `softmax(K)`, deterministic morphisms for monotone splines and simplex projection.
  - `cholesky_quad_form(K)`, covariance reconstruction `Σ = diag(s) L L^T diag(s)`.
  - `CholeskyFactor(K)` `ContinuousSpace`, manifold of K×K lower-triangular factors of correlation matrices.
- Surface syntax for hierarchical-Bayesian models in `.qvr`:
  - `draw v : A -> K ~ Family(args)`, finite-domain-indexed plate draw.
  - `observe r[n] ~ Family(args) for n in N`, vectorized observation.
  - `marginalize c`, program-level discrete-latent marginalization.
  - `arr[idx]`, Kleisli pullback gather expression inside `let`-bodies.
  - `posterior name (model) : domain -> codomain { steps return ... }`, deterministic post-conditioning block whose body consumes posterior latents.
  - Parametric programs: `program name (G : FinSet, scale : Real, prior : Mor[A, B]) : dom -> cod ...`, programs polymorphic over objects, scalars, and morphisms; denote dependent kernels `Π(p:P).Kern(dom(p), cod(p))`. Instantiated at each call site `draw v ~ name(args)` by parameter substitution + α-renaming, so each call contributes fresh latent factors to the caller's joint kernel. Supports the random-effects reuse story without tying latents across call sites.
  - Let-expression builtins: `cumsum`, `softmax`, `cholesky_quad_form` join the existing `sigmoid` / `exp` / `log` / `abs` / `softplus`.
- AST nodes in `quivers.dsl.ast_nodes`: `PlateDrawStep`, `VectorisedObserveStep`, `MarginalizeStep`, `LetExprIndex`, `PosteriorDecl`; each docstring carries the Kern denotation.
- Stan-model port at `docs/examples/source/event_structure.qvr`, a faithful translation of the four-class telicity × durativity latent-class model from `~/Projects/supertelicity/analysis/event-structure-induction/models/event-structure-model.stan`, demonstrating crossed random effects, ordinal monotone splines, vectorized observations, and `marginalize` over the discrete latent class.
- `tests/test_bayesian.py`: 15 tests covering every new primitive and every new AST node's parse / compile round-trip, plus a compile-time smoke test on the Stan-model port.

### Changed

- `_walk_program_step` return type widened from `DrawStep | LetStep` to the `ProgramStep` union root.
- Tree-sitter grammar regenerated (`grammars/qvr/src/parser.c`, `grammar.json`, `node-types.json`) to recognize the new program steps and top-level declarations.

### Added

- Pattern-polymorphic `schema` declarations: `schema r[X, Y : Cat] : (X/Y) * Y -> X`. Subsumes `rule` with explicit parameter types and a unified domain/codomain shape.
- New SetObject variants: `EnumSet(name, elements)` for named-element finite sets, `FreeResiduated(generators, depth, ops)` for residuated category universes. New surface syntax `object Atoms = {NP, S, VP}` and `object Cat = FreeResiduated(Atoms, depth=4, ops=[slash])`.
- `FreeMonoid` surface form: `object Free = FreeMonoid(X, max_length=4)`.
- `TypeSlash` and `TypeEffectApply` `TypeExpr` variants, residuated patterns and effect-typed types are first-class. The `CatPattern` AST family is removed (folded into `TypeExpr`).
- `chart_fold(lex=, binary=, unary=, start=, depth=, effect_depth=, handlers=)` primitive expression, desugared form of `parser(rules=...)`. `unary=` is wired through the inside algorithm's reflexive-transitive unary-rule closure. `handlers=` post-composes effect handlers on the parser output as log-space transition morphisms.
- `.curry_right` / `.curry_left` postfix combinators witnessing the residuation isomorphisms; backed by `quivers.core.morphisms.CurriedMorphism`.
- Typeclass tower in `quivers.monadic.typeclasses`: `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`, `Foldable`, `Traversable`, `MonadTrans`. Concrete monads (`FuzzyPowersetMonad`, `FreeMonoidMonad`, `GiryMonad`) subclass `Monad` directly.
- Stdlib effect instances in `quivers.monadic.instances`: `Identity`, `Maybe`, `Alternative_`, `Continuation`, `State`, `Reader`, `Writer`, `List`. All operations (`pure`, `fmap`, `apply`, `join`, `bind`, `lift_a2`, `empty`, `alt`, `foldr`, `traverse`) are concrete V-relation realizations; function-space-dependent operations encode `[A → B]` as a finite `FinSet` of cardinality `|B|^|A|`. Monad transformers in `quivers.monadic.transformers`: `StateT`, `ReaderT`, `MaybeT`, `ContT`, `WriterT`.
- Algebraic effects + handlers in `quivers.monadic.algebraic`: `Operation`, `EffectSignature`, `Handler`, `FreeMonad`. `FreeMonad` carrier is the bounded-depth signature-tree set realized as a flat `FinSet` with structural decomposition via `_decompose_carrier_index` / `_compose_carrier_index`; `pure`, `fmap`, `join`, `bind`, `lift_a2` satisfy the monad laws up to truncation. `Handler.run` is the post-order tree fold interpreting each leaf through `return_clause` and each operation node through `operation_clauses`. `EffectSignature.to_theory()` and `Handler.as_theory_morphism()` realize the panproto-side theory and theory morphism.
- `quivers.monadic.bridges`: `Kleisli`, `ArrowMonad`, `CoKleisli`, `kleisli`, `arrow_monad`, `cokleisli` connecting the monad and arrow towers. `Kleisli.compose` is fmap-then-join with structural recovery of the underlying B; `Kleisli.first` is realized via the canonical monad strength `σ = (pure × id) >> lift_a2(id_{A⊗B})`; `Kleisli.app` routes through the Applicative apply. `ArrowMonad` provides `fmap/pure/apply/join/bind/lift_a2` via the underlying arrow's `arr`/`id_arr`/`app`/`compose`. `CoKleisli` is registered as `Category_`; promoting to `Arrow` requires an explicit comonad costrength supplied via `first_via_costrength(f, C, costrength)`.
- `quivers.arrows` package, Hughes-style arrow hierarchy (`Category_`, `Arrow`, `ArrowChoice`, `ArrowApply`, `ArrowLoop`, `ArrowZero`, `ArrowPlus`) with panproto-theory mirrors. New `quivers.arrows.instances` with `VRel`, `Function`, `Stochastic` arrow instances; `loop_arr` realized via the V-algebra iterative trace (Joyal-Street-Verity 1996, §3).
- `quivers.stochastic.effect_lifts.class_directed_lifts`: class-driven schema lifting for effect-typed parsers. `make_swap_schema` / `swap_rule_set` emit `swap_TU` schemas from registered `DistributiveLaw` instances for commutation firings.
- `quivers.core._factories` module, concrete morphism constructors `inj`, `case`, `pi`, `pair`, `parallel`, `terminal`, `constant`, `distrib_right`, `coproduct_map`. The algebra on which the stdlib monads, arrows, and algebraic-effects layer are built.
- New tree-sitter grammar at `grammars/qvr/` with regenerated parser; the unified `_type_expr` family subsumes the prior `_cat_pattern` productions.
- Local-grammar override at `quivers.dsl._dev_grammar` (activated by `QVR_USE_LOCAL_GRAMMAR=1`) using panproto 0.47's first-class `AstParserRegistry.override_grammar()` API. Compiles the in-tree grammar and installs it into the standard registry when the upstream `panproto-grammars-all` bundle hasn't yet vendored the latest grammar source.
- `docs/guides/effects.md` and `docs/semantics/effects.md`, user guide and formal denotational layer for the typeclass + algebraic-effects framework.
- `quantifier_scope.qvr` example demonstrating Charlow-style scope-taking via `Continuation`.

### Changed

- The `Monad` ABC is the typeclass-tower one (`quivers.monadic.typeclasses.Monad`); the previous parallel ABC in `quivers.monadic.monads` is removed. Concrete monads provide both the typeclass operations (`pure`, `apply`, `join`) and the Eilenberg–Moore aliases (`unit`, `multiply`).
- `RuleDecl` premises and conclusion are typed at `TypeExpr` (previously `CatPattern`).
- `ObjectDecl` admits both `: type_expr` and `= initializer` forms.
- `parser(...)` infers category atoms from a uniquely-declared `FreeResiduated` in scope when no `categories=` argument is supplied.
- `QVR_PROGRAM_PROTOCOL` extended with `enum_set`, `free_residuated`, `schema_decl` vertex kinds.
- `InsideAlgorithm` accepts an optional `unary` morphism; the chart fills with reflexive-transitive unary-rule closure at each cell.
- `chart_fold(effect_depth>0)` no longer raises `CompileError`; the parameter flows through as informational metadata and the caller-supplied `binary` morphism (typically built via `lift_rule_set` over declared effects) provides the lifted firings. `handlers=` are post-composed via `_ChartHandlerComposite` log-space transitions.
- Denotational-semantics docs: corrected marginalization formulas (proper handling of residual input `Y`), Kleisli composition ordering in `programs.md` (`s_1 ⋄ ⋯ ⋄ s_n ⋄ ret`, not the reverse), scan formula typing in `expressions.md`, profunctor typing in `grammar.md`, the row-stochastic/`column-stochastic` distinction in `morphisms.md`, the `arrow_monad ∘ kleisli ≅ id` natural isomorphism in `effects.md`.

### Fixed

- `FreeMonad` carrier no longer collapses under `CoproductSet` auto-flattening when the leaf type is itself a coproduct; the flat-`FinSet` representation preserves the recursive leaf-vs-operation structure.
- `FreeMonad.lift_a2` is the correct free-monad applicative recursion (bi-depth `(d_a, d_b, d_c)` tracking with proper continuation re-encoding), replacing a prior block-identity rule that misrepresented op-summand handling.
- `FreeMonad.join` splices outer trees correctly through `_carrier_op_offset`, replacing a prior block-identity that mapped op-summand indices without accounting for the differing inner/outer continuation cardinalities.
- `CoKleisli.first` was type-incorrect (`W(A) × C → B × C` vs the required `W(A × C) → B × C`); now registered as `Category_` only, with `first_via_costrength` for promotion to `Arrow` when a comonad costrength is supplied.
- `List.fmap_obj` accepts non-`FinSet` inputs by re-encoding via cardinality; `List(List(A))` now type-checks and the monad laws hold on its own image.
- `TypeCoproduct` was already correctly handled at `resolution.py:119`; documented as such.
- Bridge round-trip claim in `effects.md` §7 corrected from `=` to `≅` (Hughes 2000 proves natural isomorphism via the `1 ⊗ A ≅ A` unitor, not equality).

### Upstream

- panproto/panproto#89 closed and shipped in panproto 0.47.0, first-class runtime grammar override via `AstParserRegistry.override_grammar()`.
- panproto/didactic#38 closed and shipped in didactic 0.7.0, `tuple[Model, ...]` field types accept any `dx.Model` element directly (the workaround tuple-of-`TaggedUnion`-roots is no longer needed).
- panproto/didactic#39 closed and shipped in didactic 0.7.0, `dx.field(opaque=True)` for fields typed at typeclass ABCs (`Monad`, `ArrowApply`, etc.); opaque fields preserve in-process identity through `with_` but drop to `None` on JSON round-trip.

### Dependencies

- `panproto >= 0.47.0` (was `>= 0.45.0`).
- `panproto-grammars-all >= 0.47.0` (was `>= 0.45.0`).
- `didactic >= 0.7.1` (was `>= 0.6.0`).

## [0.2.0] - 2026-05-06

### Changed

- Every record-shaped value type (AST nodes, `FinSet`, `ProductSet`, `CoproductSet`, `ContinuousSpace` variants, `Category` variants, `RuleSystem`) is now a `didactic.api.Model`. Recursive sums are `dx.TaggedUnion` roots discriminated by a `kind: Literal[...]` field. JSON round-trips via `model_dump_json` / `model_validate_json` are available on every value type.
- Resolution from `TypeExpr` / `SpaceExpr` AST trees to runtime `SetObject` / `ContinuousSpace` values is expressed as a `dx.Lens` family in `quivers.dsl.resolution`.
- Variadic constructors `ProductSet(A, B, C)` and `CoproductSet(A, B, C)` are replaced by keyword form `ProductSet(components=(A, B, C))` and `CoproductSet(components=(A, B, C))`. The flattening converter preserves the previous flattening behavior.
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
