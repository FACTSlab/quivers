# Architecture

The quivers library is organized into nineteen subpackages plus two
top-level modules. The categorical core (`core/`, `categorical/`,
`monadic/`, `arrows/`, `enriched/`, `stochastic/`, `continuous/`) is
surrounded by a DSL layer (`dsl/`), an inference layer (`inference/`),
a data layer (`data/`, `diagnostics/`, `formulas/`), structural and
analysis utilities (`structural/`, `analysis/`), effect handlers
(`effects/`), transpilers (`transpile/`), and the interactive surfaces
(`cli/`, `kernel/`, `lsp/`). This document describes the
package hierarchy, dependencies, and key abstractions in each module.

## Package Structure

```
quivers/
├── core/              # foundational types and operations
├── categorical/       # functors, transformations, adjunctions, monoidal
├── monadic/           # typeclass tower (Functor → Applicative →
│                      # Monad → MonadPlus, plus Traversable /
│                      # MonadTrans), stdlib effect instances,
│                      # transformers, algebraic effects,
│                      # Kleisli/ArrowMonad/CoKleisli bridges,
│                      # panproto theory stubs
├── arrows/            # Hughes-style arrow tower (Category_ → Arrow →
│                      # ArrowApply / ArrowChoice / ArrowLoop /
│                      # ArrowZero / ArrowPlus) plus VRel / Function /
│                      # Stochastic instances and panproto theories
├── enriched/          # ends/coends, Kan, weighted limits, profunctors,
│                      # Yoneda, Day convolution, optics
├── stochastic/        # FinStoch (Markov kernels), families, transforms
│                      # (condition / mix / factor / normalize), Giry
│                      # monad, deductive chart parsers (CCG, Lambek),
│                      # semirings, schema, span, effect_lifts
├── continuous/        # parameterized families, spaces, boundaries,
│                      # flows, monadic programs, plates, scans,
│                      # deterministic transforms, inline composition
├── dsl/               # panproto-driven tree-sitter parser package,
│                      # didactic AST node package, compiler package,
│                      # program/deduction theory schemas, Pygments
│                      # lexer, grammar introspection helpers
├── inference/         # trace, conditioning, latent registry,
│                      # variational guides (mean-field, full-rank,
│                      # low-rank, IAF, neural-spline, mixture,
│                      # Laplace, delta), gradient estimators,
│                      # objectives (ELBO / IWAE / Rényi / VR-IWAE /
│                      # ChiVI / RWS / DReGs),
│                      # SVI, predictive, MCMC (HMC, NUTS), DAIS,
│                      # warmup, Bayesian lifts
├── analysis/          # static analysis of programs: scope, chain
│                      # shape, plate graph rendering, initialiser
│                      # specifications, saturation
├── data/              # data-frame encoding helpers and observation
│                      # schemas
├── diagnostics/       # ArviZ-backed trace I/O and predictive checks
├── formulas/          # R-style formula DSL → quivers program
│                      # compilation (formulae integration)
├── structural/        # encoder/decoder shape inference, structural
│                      # signatures, losses
├── effects/           # algebraic effect handlers and reparameterizers
├── transpile/         # target-independent IR and eleven PPL renderers
├── cli/               # the `qvr` command, REPL, language-server
│                      # entry-point, migration driver
├── kernel/            # Jupyter kernel install + runtime
├── lsp/               # pygls-based language server (document model,
│                      # server)
├── program.py         # nn.Module wrapper for morphisms (discrete
│                      # or continuous)
└── giry.py            # backwards-compat re-exports of GiryMonad and
                       # FinStoch from stochastic.giry
```

The QVR tree-sitter grammar lives at the repo root under `grammars/qvr/`; it is vendored by panproto's `panproto-grammars-all` distribution and consumed at runtime by quivers' parser.

## Module Descriptions

### `core/`

Foundational categorical types and operations. Defines the objects and morphisms that all other modules build on.

- **`objects.py`:** Categorical objects as `dx.TaggedUnion` variants: `SetObject` (root), `FinSet`, `ProductSet`, `CoproductSet`, `FreeMonoid`, `EnumSet`, `FreeResiduated`, plus the module-level `Unit: FinSet = FinSet(name="1", cardinality=1)` singleton.
- **`algebras.py`:** Composition rules organised as a three-level hierarchy: `CompositionRule` (abstract) → `Semigroupoid` (composition only) → `Algebra` (composition + tensor unit + bilinear pair). `BilinearForm` sits beside `Semigroupoid` for the tensor-product side. Concrete variants: `ProductFuzzyAlgebra`, `BooleanAlgebra`, `LukasiewiczAlgebra`, `GodelAlgebra`, `TropicalAlgebra`, `MaxPlusAlgebra`, `LogProbAlgebra`, `RealAlgebra`, `ProbabilityAlgebra`, `CountingAlgebra`, `MarkovAlgebra`, plus `DualAlgebra`, `CustomAlgebra`, `CustomSemigroupoid`, `CustomBilinearForm`. Module-level singletons: `PRODUCT_FUZZY`, `BOOLEAN`, `LUKASIEWICZ`, `GODEL`, `TROPICAL`, `MAX_PLUS`, `LOG_PROB`, `REAL`, `PROBABILITY`, `COUNTING`, `MARKOV`, `REICHENBACH`, `BOOLEAN_DUAL`, `DUAL_LUKASIEWICZ`, `DUAL_GODEL`.
- **`algebra_morphisms.py`:** `AlgebraHomomorphism` (abstract) and its concrete variants (`IdentityHom`, `Embedding`, `Expectation`, `LogProb`, `MaxPlus`, `Threshold`, `MaterialImplication`, `ProbabilityClamp`, `CountingFromReal`, `ProbabilityToReal`, `CountingToReal`), plus the `threshold()`, `embedding()`, and `lookup_homomorphism()` helpers.
- **`morphism_transformations.py`:** `MorphismTransformation` (abstract step type) plus the `Softmax`, `L1Normalize`, `L2Normalize`, and `BayesInvert` concrete steps and their `softmax()`/`l1_normalize()`/`l2_normalize()` factory wrappers. These steps drive `Morphism.change_base` and the `TransformedMorphism` carrier.
- **`morphisms.py`:** $\mathcal{V}$-enriched relations as tensors: `Morphism` (abstract), `ObservedMorphism` (fixed tensor data), `LatentMorphism` (learnable), `TransformedMorphism`, `ComposedMorphism`, `ProductMorphism`, `MarginalizedMorphism`, `FunctorMorphism`, `RepeatMorphism`, `CurriedMorphism`, plus the factory functions `morphism()`, `observed()`, `identity()`, `cup()`, `cap()`, and helpers `as_torch_module()`, `extract_morphism()`.
- **`tensor_ops.py`:** Tensor operations: `noisy_or_contract`, `noisy_or_reduce`, `noisy_and_reduce`, `componentwise_lift`.
- **`trans.py`:** `TransSeq`, a composable sequence of `MorphismTransformation` steps.
- **`wiring.py`:** Wiring helpers for tensor-product composition.
- **`_factories.py`, `_util.py`:** Internal factory helpers and numerical utilities (`EPS`, `clamp_probs`, `safe_log1p_neg`).

### `categorical/`

Categorical structures built from objects and morphisms.

- **`functors.py`:** Functors and functor composition: `Functor`, `IdentityFunctor`, `ComposedFunctor`, `FreeMonoidFunctor`.
- **`natural_transformations.py`:** Natural transformations: `NaturalTransformation`, `ComponentwiseNT`.
- **`adjunctions.py`:** Adjunctions and free-forgetful pairs: `Adjunction`, `ForgetfulFunctor`, `FreeForgetfulAdjunction`.
- **`monoidal.py`:** Monoidal structures: `MonoidalStructure`, `CartesianMonoidal`, `CoproductMonoidal`, `EmptySet`, `EMPTY`.
- **`base_change.py`:** Base change between algebras: `BaseChange`, `BoolToFuzzy`, `FuzzyToBool`.
- **`traced.py`:** Traced monoidal categories: `TracedMonoidal`, `CartesianTrace`, `IterativeTrace`, functions `trace()`, `partial_trace()`.

### `monadic/`

A Haskell-style typeclass hierarchy with concrete monad instances
plus comonads, algebras, distributive laws, transformers, and
algebraic effects. Each typeclass is mirrored by a panproto theory.

Modules:

- **`typeclasses.py`:** `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`, `Foldable`, `Traversable`, `MonadTrans` ABCs.
- **`monads.py`:** `FuzzyPowersetMonad`, `FreeMonoidMonad` (concrete Monad instances) plus `KleisliCategory`.
- **`comonads.py`:** `Comonad`, `CoKleisliCategory`, `DiagonalComonad`, `CofreeComonad`.
- **`algebras.py`:** Eilenberg–Moore: `Algebra`, `FreeAlgebra`, `ObservedAlgebra`, `Coalgebra`, `CofreeCoalgebra`, `ObservedCoalgebra`, `EilenbergMooreCategory`.
- **`distributive_laws.py`:** `DistributiveLaw`, `FreeMonoidPowersetLaw`.
- **`instances.py`:** stdlib effect instances: `Identity`, `Maybe`, `Alternative_`, `Continuation`, `State`, `Reader`, `Writer`, `List`. Each registers against the appropriate ABC(s) via `ABC.register(...)`.
- **`transformers.py`:** monad transformers: `StateT`, `ReaderT`, `MaybeT`, `ContT`, `WriterT`.
- **`algebraic.py`:** algebraic effects + handlers: `Operation`, `EffectSignature`, `Handler`, `FreeMonad`.
- **`bridges.py`:** `Kleisli`, `ArrowMonad`, `CoKleisli` plus `kleisli` / `arrow_monad` / `cokleisli` factory helpers connecting the monad, arrow, and comonad towers.
- **`theories.py`:** panproto-theory stubs for each typeclass.
- **`laws.py`:** runtime-checkable typeclass-law scaffolding.

### `arrows/`

Hughes-style arrow tower (parallel to the monad-side hierarchy).

- **`typeclasses.py`:** `Category_`, `Arrow`, `ArrowChoice`, `ArrowApply`, `ArrowLoop`, `ArrowZero`, `ArrowPlus` ABCs.
- **`instances.py`:** concrete arrow carriers: `VRel` (the $\mathcal{V}$-enriched relation arrow), `Function` (the deterministic restriction of `VRel` to 0/1-valued function-graph tensors), `Stochastic` (Markov-kernel arrow).
- **`theories.py`:** panproto-theory stubs for each arrow typeclass.

### `enriched/`

Enriched category theory: the calculus of $\mathcal{V}$-valued hom-functors.

- **`ends_coends.py`:** Limits and colimits in enriched categories: `coend()`, `end()`.
- **`kan_extensions.py`:** Kan extensions: `ObjectMap`, `Projection`, `Inclusion`, `left_kan()`, `right_kan()`.
- **`weighted_limits.py`:** Weighted (co)limits: `Weight`, `Diagram`, `weighted_limit()`, `weighted_colimit()`, `representable_weight()`, `terminal_weight()`.
- **`profunctors.py`:** Profunctors (distributor, bimodules): `Profunctor`.
- **`yoneda.py`:** Yoneda lemma and embedding: `Presheaf`, `representable_profunctor()`, `corepresentable_profunctor()`, `yoneda_embedding()`, `yoneda_lemma()`, `yoneda_density()`, `verify_yoneda_fully_faithful()`.
- **`day_convolution.py`:** Day convolution for monoidal categories: `day_convolution()`, `day_unit()`, `day_convolution_profunctors()`.
- **`optics.py`:** Optics and their composition: `Optic`, `Lens`, `Prism`, `Adapter`, `Grate`, `compose_optics()`.

### `stochastic/`

Stochastic morphisms and the category **FinStoch** of Markov kernels on finite sets.

- **`morphisms.py`:** Row-stochastic learnable Markov kernels: `StochasticMorphism` and its alias `CategoricalMorphism`, plus the `stochastic()` factory.
- **`families.py`:** Discretized parameterized distributions: `DiscretizedNormal`, `DiscretizedLogitNormal`, `DiscretizedBeta`, `DiscretizedTruncatedNormal`.
- **`transforms.py`:** Transforms on stochastic morphisms and the carrier classes they produce: `condition()`, `mix()`, `factor()`, `normalize()`, with `ConditionedMorphism`, `MixtureMorphism`, `FactoredMorphism`, `NormalizedMorphism`.
- **`queries.py`:** Query functions: `prob()`, `marginal_prob()`, `expectation()`.
- **`giry.py`:** The Giry monad on FinStoch: `GiryMonad`, `FinStoch`.
- **`categories.py`, `effect_lifts.py`, `inside.py`, `agenda.py`, `span.py`, `semiring.py`:** chart-parser infrastructure: deduction categories, agenda-based search, span representations, semiring weights, the inside algorithm, and effect lifts.
- **`schema.py`:** `RuleSchema`, a functor from `CategorySystem` to `RuleSystem`. Provides the binary/unary schema combinators (`BinaryRuleSchema`, `UnaryRuleSchema`, `UnionSchema`, `WeightedSchema`) and the concrete CCG/Lambek schema instances (`ForwardApplication`, `BackwardApplication`, `ForwardComposition`, `BackwardComposition`, the crossed and commutative variants, `TensorIntroduction`, `LeftUnitElimination`, …). Schemas compose via `|` (union) and `.weighted(w)`.
- **`deduction/`:** deduction-system primitives (atoms, axioms, rules).
- **`_rule_system.py`, `rules.py`, `ccg.py`, `lambek.py`, `parsers.py`:** the `RuleSystem` core, the `ccg_rules` / `lambek_rules` / `custom_rules` libraries, and the `ChartParser` / `CCGParser` / `LambekParser` drivers.
- **`stdlib/`:** packaged grammars and rule sets.

### `continuous/`

Continuous and hybrid discrete-continuous morphisms and monadic programs.

- **`spaces.py`:** Continuous sample spaces as `dx.TaggedUnion` variants: `ContinuousSpace`, `Euclidean`, `UnitInterval` (factory wrapping `Euclidean`), `Simplex`, `PositiveReals`, `CholeskyFactor`, `ProductSpace`, `Sphere`, `Ball`, `Covariance`, `Correlation`, `Orthogonal`, `Stiefel`, `LowerTriangular`, `Diagonal`.
- **`morphisms.py`:** Continuous morphism base classes and combinators.
- **`families.py`:** Conditional distribution families, defined either as explicit subclasses (e.g. `ConditionalNormal`, `ConditionalBeta`, `ConditionalDirichlet`, `ConditionalMultivariateNormal`, `ConditionalLowRankMVN`, `ConditionalWishart`, `ConditionalInverseWishart`, `ConditionalMatrixNormal`, `ConditionalRelaxedBernoulli`, `ConditionalRelaxedOneHotCategorical`, `ConditionalLKJCholesky`, `ConditionalGaussianProcess`, `ConditionalHorseshoe`, `ConditionalMixture`, `ConditionalIndependent`, `ConditionalTransformed`, `ConditionalBernoulli`, `ConditionalCategorical`, `ConditionalBinomial`, `ConditionalLogisticNormal`, `ConditionalOneHotCategorical`) or generated by the `_make_family` factory (e.g. `ConditionalCauchy`, `ConditionalLaplace`, `ConditionalGumbel`, `ConditionalLogNormal`, `ConditionalStudentT`, `ConditionalExponential`, `ConditionalGamma`, `ConditionalChi2`, `ConditionalHalfCauchy`, `ConditionalHalfNormal`, `ConditionalInverseGamma`, `ConditionalWeibull`, `ConditionalPareto`, `ConditionalKumaraswamy`, `ConditionalContinuousBernoulli`, `ConditionalFisherSnedecor`, `ConditionalPoisson`, `ConditionalGeometric`). Also exports the `LKJCorrelationFactor` and `Truncated` helpers.
- **`family_spec.py`:** Declarative specification objects backing `_make_family`.
- **`programs.py`:** Monadic programs: `MonadicProgram` (compositional probabilistic computations) plus the internal `_StepSpec` / `_LetSpec` / `_ScoreSpec` step records.
- **`plate.py`, `scan.py`:** `plate(...)` and `scan(...)` constructs for vectorised and sequential program fragments.
- **`inline.py`:** inline-composition helpers used by the DSL compiler.
- **`deterministic.py`:** deterministic continuous morphisms (pure functions on continuous spaces).
- **`boundaries.py`:** Boundaries between discrete and continuous: `Discretize`, `Embed`.
- **`flows.py`:** Normalizing flows: `AffineCouplingLayer`, `ConditionalFlow`.

### `dsl/`

Domain-specific language for quiver expressions in `.qvr` files. Parsing is delegated to panproto via the `qvr` tree-sitter grammar (located at `grammars/qvr/` in the repo, vendored by `panproto-grammars-all`); quivers does not run a hand-written lexer. The reference `.qvr` programs that drive the test suite and the tree-sitter grammar fixtures live at `docs/examples/source/`, not under the package itself.

- **`parser/`:** A package that walks the panproto-produced parse tree and builds a tree of `dx.Model` AST nodes. The top-level module re-exports `parse()`, `parse_file()`, `ParseError`; submodules `core.py`, `expressions.py`, `options.py`, `program_steps.py`, `statements.py`, `_helpers.py`, and `_registry.py` partition the walk by syntactic category.
- **`ast_nodes/`:** A package whose every node is a `dx.Model`. Recursive sums (`ObjectExpr`, `Expr`, `LetExprNode`, `ProgramStep`, `Statement`) are `dx.TaggedUnion` roots discriminated by a `kind: Literal[...]` field. Submodules split nodes by category: `declarations.py`, `expressions.py`, `let_expressions.py`, `module.py`, `objects.py`, `program_steps.py`, `structural.py`, `_shared.py`.
- **`compiler/`:** A package that walks the AST and produces a `quivers.Program`. `core.py` defines `Compiler` (composed via mixins from `declarations.py`, `deductions.py`, `expressions.py`, `programs.py`, `structural.py`); `resolution.py` provides the `_ResolutionMixin` whose `_resolve_any_space` routes an `ObjectExpr` to either a `SetObject` (discrete) or a `ContinuousSpace`; `_options.py` parses option blocks and `_prelude.py` defines `CompileError` plus shared compiler-state primitives.
- **`program_theory.py`:** Defines `QVR_PROGRAM_PROTOCOL` (a panproto protocol whose vertex kinds enumerate every `SetObject` and `ContinuousSpace` variant plus the QVR declaration variants), `extract_program_schema(compiler)`, and `extract_deduction_schema(compiler)`. These make every `.qvr` program a schema in panproto's sense, diff, migrate, and lens-generation workflows apply.
- **`constraints.py`:** post-parse axiom checks: `check_constraints(module)` returns a list of `Violation` records (used by the LSP, the REPL, and `qvr check`).
- **`emit.py`:** AST → `.qvr` source emission (the print direction of the parser).
- **`pygments_lexer.py`:** A minimal Pygments lexer used to syntax-highlight `.qvr` blocks in this documentation site (registered as the `qvr` lexer entry point).
- **`_grammar_build.py`, `_grammar_data/`, `_grammar_introspection.py`, `_historical_grammar.py`:** compiles the in-tree grammar for the Pygments lexer, pinned per-revision grammar blobs, runtime grammar introspection, and a loader that selects a historical grammar by revision string (used by `qvr migrate`). Parsing itself goes through the `qvr` grammar vendored in `panproto-grammars-all`.

Top-level DSL API (re-exported from `quivers.dsl`): `parse()`, `parse_file()`, `loads()`, `load()`, `Compiler`, `Module`, `QVR_PROGRAM_PROTOCOL`, `extract_program_schema()`, plus exceptions `ParseError`, `CompileError`.

Parametric programs, `program name (G : FinSet, scale : Real, prior : Mor[A, B]) : dom -> cod`, are stored as templates on the compiler (`Compiler._program_templates`) and inlined per call site by substitution + α-renaming; the runtime `MonadicProgram` is built from the inlined step list, so distinct call sites contribute distinct factors to the parent's joint kernel.

Program bodies use a single Kleisli-bind sigil `<-`. The sampling shapes are `sample v <- F(args)` (scalar draw), `sample v : A <- F(args)` (A-indexed plate), `observe v <- F(args)` (scored observation), `observe r : N <- F(args)` (N-indexed batched score), and `marginalize c : A <- F(args) [over = G, reduction = R]` with an indented scope body (scoped marginalization). A program declaration carries an effect signature in its option block, `[effects = [Sample, Score, Marginal, Pure]]`; the compiler verifies that the body's actual effects are a subset of the declared set. A posterior block is an `[effects = [Pure]]` program with an `[over = M]` option consuming the latents of model `M` as data parameters.

### `inference/`

Inference primitives for monadic probabilistic programs: tracing, observation conditioning, automatic variational guides, gradient estimators, variational objectives, Markov-chain Monte Carlo, and posterior-predictive sampling.

- **`trace.py`:** Execution tracing: `SampleSite`, `Trace`, `trace()`.
- **`conditioning.py`:** Observation marking: `Conditioned`, `condition()`.
- **`registry.py`:** `LatentRegistry` and `LatentSite`, the per-site introspection helper every guide and MCMC kernel consumes.
- **`guides/`:** A package of variational guide families. Public classes: `Guide` (base), `AutoNormalGuide`, `AutoMultivariateNormalGuide`, `AutoLowRankMultivariateNormalGuide`, `AutoDeltaGuide`, `AutoLaplaceApproximation`, `AutoNormalizingFlow`, `AutoIAFGuide`, `AutoNeuralSplineGuide`, `AutoMixtureGuide`, `AutoGuideList`, and `AutoStructured`.
- **`estimators.py`:** Gradient estimators: `GradientEstimator`, `Reparameterized`, `StickingTheLanding`, `DoublyReparameterized`, `ScoreFunction`.
- **`objectives.py`:** Variational objectives: `Objective`, `ELBO`, `IWAEBound`, `RenyiBound`, `VRIWAEBound`, `ChiVI`, `RWS`, and `DReGsBound`.
- **`svi.py`:** Stochastic variational inference driver: `SVI`.
- **`predictive.py`:** Posterior-predictive sampling: `Predictive`.
- **`mcmc/`:** Markov-chain Monte Carlo: `MCMCKernel` (base), `HMCKernel`, `NUTSKernel`, plus the `MCMC` driver and `MCMCResult`.
- **`dais.py`:** Differentiable annealed importance sampling: `AutoDAIS`.
- **`warmup.py`:** `WarmupThenHMC`, a warmup-then-sampling composite kernel.
- **`lifts.py`:** Bayesian lifts that turn a deterministic program into a sampled one: `bayesian_lift_parameters`, `lift_to_bayesian_program`, `lift_from_log_prob`, `monte_carlo_log_joint`.
- **`transforms.py`:** Bijective flow transforms used by the normalizing-flow guides: `TransformModule` (base), `AffineCouplingTransform`, `MaskedAutoregressiveTransform`, `InverseAutoregressiveTransform`, `NeuralSplineCouplingTransform`, `LULinearTransform`, `BatchNormTransform`, plus the `MADE` autoregressive net and the `alternating_mask` / `half_mask` mask helpers.

### `analysis/`

Static analysis of compiled programs: `scope.py` (scope-path traversal), `chain_shape.py` (chain-rule shape inference), `plate_graph.py` and `plate_render.py` (plate-diagram extraction and rendering), `init_spec.py` (initialiser specifications), `saturation.py` (saturation checks for the type/effect system).

### `data/`

Data-frame ingestion. `schema.py` defines `DatasetSchema` (a frozen `dx.Model` pairing a program's declared columns with their roles, dtypes, and plate indices) plus the `compose()` helper that splices a `DatasetSchema` into a `.qvr` body. `encoding.py` defines `ColumnRole`, `MissingPolicy`, and `encode_column()`, which convert narwhals / pandas / polars / pyarrow frames into the tensor dictionaries the runtime consumes.

### `diagnostics/`

ArviZ-backed posterior diagnostics. `arviz_io.py` converts an `MCMCResult` into an `xarray.DataTree` (the ArviZ 1.x container) via `az.from_dict`, repackaging sample tensors under the canonical ArviZ group names with user-supplied coords and dims; `comparison.py` and `predictive_checks.py` wrap the standard cross-validation and posterior-predictive-check workflows.

### `formulas/`

R-style formula DSL. `formula.py` defines the `Formula`, `FormulaData`, `FixedColumn`, and `RandomTerm` data classes plus `formula_from_data()`, the entry point from a formula string + data frame to a parsed formula. `compile.py` defines `FormulaToQVRModule` (a `dx.Lens[Formula, Module, FormulaData]`) that lowers a parsed formula into a `quivers.dsl.Module`. `family.py` declares the conjugate response families (`Family`, `Link`, `AuxParam`). `_fit.py` provides the `BayesianFit` orchestration record and the `fit()` convenience entry point (dispatches between SVI and MCMC).

### `structural/`

Structural compression: a uniform algebraic interface for encoding arbitrary structured terms to fixed-length vectors and decoding them back under a learned distribution.

- **`signature.py`:** Term-level data model. `Sort`, `Constructor`, `Binder`, `BinderArgSpec`, `BinderVarSpec`, `Term`, `VertexKind`, `EdgeKind`, `Signature`, `Context`, `EMPTY_CONTEXT`, plus the `make_term` / `bound_var` factories.
- **`encoder.py`:** `Encoder` (`nn.Module`) plus the `make_default_op_fn` / `make_default_var_init` defaults.
- **`decoder.py`:** `Decoder` (`nn.Module`), the inverse direction.
- **`losses.py`:** `LossEntry`, `LossRegistry` for plug-in reconstruction objectives.
- **`shapes/`:** shape backends (`seq.py`, `tree.py`, `graph.py`), one per supported shape category.

### `effects/`

Algebraic handlers for tracing, clamping observations, interventions,
masking, scaling, blocking, replay, lifting, collapsing, and
reparameterization. `interpreter.py` executes a program under the
active handler stack; `reparam/` contains the loc-scale, transform,
NeuTra, and conjugate strategies.

### `transpile/`

Cross-language emission for compiled QVR modules. `lower.py` produces a
target-independent IR, and `renderers/` emits Stan, NumPyro, Pyro,
PyMC, Edward2, Turing, Gen, Church, WebPPL, BUGS, or JAGS source.

### `cli/`

The `qvr` command (`__init__.py`'s `main`) and its subcommands:
`check.py` (validate a program), `lsp.py` (launch the LSP), `migrate.py` and `migrations/` (run grammar/AST migrations), `repl.py` plus the `repl_*` modules (REPL session, prompt, highlighting, Textual TUI, tab completion).

### `kernel/`

Jupyter kernel: `install.py` registers the kernelspec; `quivers_kernel.py` is the in-process kernel; `__main__.py` is the launcher.

### `lsp/`

pygls-based language server: `server.py` is the LSP entry point; `document.py` models open buffers and incremental edits.

### Root-level modules

- **`program.py`:** `Program`: wraps a morphism (discrete or continuous) as a differentiable `nn.Module`.
- **`giry.py`:** Backwards-compat re-export of `GiryMonad` and `FinStoch` from `quivers.stochastic.giry`.

## Dependency Graph

The package has a roughly layered dependency structure. Layers higher
in the list depend only on layers above them, with one documented
exception: `dsl.compiler.declarations` pulls a single helper
(`_algebra_init_spec`) out of `analysis.init_spec`, so the strict
analysis-then-dsl ordering bends at that one call site.

1. **`core/`:** foundational (no internal dependencies).
2. **`categorical/`, `monadic/`, `arrows/`:** build on `core/`.
3. **`enriched/`:** builds on `core/` and `categorical/`.
4. **`stochastic/`:** builds on `core/`, `categorical/`, `monadic/`.
5. **`continuous/`:** builds on `core/`, `categorical/`, `monadic/`, `stochastic/`.
6. **`analysis/`:** builds on `core/` and `dsl.ast_nodes` / `dsl.compiler._prelude` (uses AST types and the compiler's algebra registry for static analysis).
7. **`dsl/`:** synthesises `core/`, `categorical/`, `monadic/`, `enriched/`, `stochastic/`, `continuous/`, plus `analysis.init_spec` for one initialiser-checking helper.
8. **`inference/`:** builds on `core/`, `categorical/`, `stochastic/`, `continuous/`, and `program.py`.
9. **`data/`:** builds on `dsl/`. **`diagnostics/`:** builds on `inference/`.
10. **`formulas/`:** builds on `continuous/`, `dsl/`, and `inference/`.
11. **`structural/`:** standalone PyTorch-only shape inference; no in-package dependencies.
12. **`effects/`:** builds on `continuous/` and provides the handler-aware interpreter used by inference.
13. **`transpile/`:** builds on `dsl/` and lowers compiled modules to target-specific schemas and source.
14. **`cli/`:** consumes `analysis/`, `dsl/`, `kernel/`, `lsp/`, and `transpile/`. **`kernel/`:** uses `cli/` to start the in-process REPL session. **`lsp/`:** uses `cli/` and `dsl/`.
15. **`program.py`:** depends on `core/` and `continuous/`.
16. **`giry.py`:** thin backwards-compat re-export of `stochastic/`.

Users can enter at the layer their task requires: core algebra,
categorical structures, probabilistic computations, or the full DSL.

## Design Principles

- **Tensors as morphisms**: $\mathcal{V}$-relations are represented as PyTorch tensors, making them amenable to automatic differentiation and GPU acceleration.
- **Value types are didactic Models**: every record-shaped value (AST nodes, `SetObject` variants, `ContinuousSpace` variants, deduction categories, `RuleSystem`, `Kleisli` / `ArrowMonad` / `CoKleisli` bridges, `VRel` / `Function` / `Stochastic` arrows) is a frozen `dx.Model`. This buys uniform construction, structural equality, JSON round-trips via `model_dump_json` / `model_validate_json`, and cross-field axioms via `__axioms__`. Recursive sums are `dx.TaggedUnion` roots with a `kind: Literal[...]` discriminator.
- **Tensor-bearing classes stay mutable**: `Presheaf`, `Weight`, `SampleSite`, and `Trace` accumulate `torch.Tensor` fields and are mutated in place during inference; they remain `@dataclass`.
- **Lazy composition**: morphism composition creates a DAG; the final tensor is materialized only on evaluation.
- **Type safety**: categorical constraints (domain/codomain compatibility) are enforced statically in Python.
- **Module transparency**: all morphisms expose a `.module()` method returning an `nn.Module`, enabling integration with PyTorch training loops.
- **Algebra flexibility**: the same morphism structure works with any algebra; swapping algebras changes semantics (fuzzy, Boolean, tropical, etc.) without code duplication.
- **DSL pipeline as a panproto theory morphism**: parsing is delegated to panproto's tree-sitter–driven AST registry; resolution is a compiler mixin that walks `ObjectExpr` trees into `SetObject` or `ContinuousSpace` values; each compiled program extracts to a panproto `Schema` against `QVR_PROGRAM_PROTOCOL`. Diff/migrate/lens-generation tools work on `.qvr` programs out of the box.
