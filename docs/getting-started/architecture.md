# Architecture

The quivers library is organized into eight subpackages, each providing a layer of categorical structure. This document describes the package hierarchy, dependencies, and key abstractions in each module.

## Package Structure

```
quivers/
├── core/              # foundational types and operations
├── categorical/       # functors, transformations, adjunctions, monoidal
├── monadic/           # typeclass tower (Functor → Applicative →
│                      # Monad → MonadPlus, plus Traversable /
│                      # MonadTrans), stdlib effect instances,
│                      # transformers, algebraic effects,
│                      # Kleisli/ArrowMonad bridges, panproto theories
├── arrows/            # Hughes-style arrow tower (Category_ → Arrow →
│                      # ArrowApply / ArrowChoice / ArrowLoop)
├── enriched/          # ends/coends, Kan, weighted limits, profunctors, Yoneda, optics
├── stochastic/        # FinStoch (Markov kernels), families, conditioning,
│                      # Giry monad, chart parsers, effect_lifts
├── continuous/        # parameterized families, boundaries, flows, monadic programs
├── dsl/               # panproto-driven parser, didactic AST nodes,
│                      # compiler, resolution lenses, Program Theory
├── inference/         # trace-based conditioning, variational guides, SVI
├── program.py         # nn.Module wrapper for morphisms
└── giry.py            # re-exports GiryMonad and FinStoch from stochastic.giry
```

The QVR tree-sitter grammar lives at the repo root under `grammars/qvr/`; it is vendored by panproto's `panproto-grammars-all` distribution and consumed at runtime by quivers' parser.

## Module Descriptions

### `core/`

Foundational categorical types and operations. Defines the objects and morphisms that all other modules build on.

- **`objects.py`:** Categorical objects: `FinSet` (finite sets), `ProductSet`, `CoproductSet`, `FreeMonoid`, `Unit`.
- **`quantales.py`:** Enrichment lattices: `Quantale` (abstract), `ProductFuzzy`, `BooleanQuantale`, and singletons `PRODUCT_FUZZY`, `BOOLEAN`.
- **`morphisms.py`:** $\mathcal{V}$-enriched relations as tensors: `Morphism` (abstract), `ObservedMorphism` (fixed tensor data), `LatentMorphism` (learnable), `ComposedMorphism`, `ProductMorphism`, `MarginalizedMorphism`, plus factory functions `morphism()`, `observed()`, `identity()`.
- **`tensor_ops.py`:** Tensor operations: `noisy_or_contract`, `noisy_or_reduce`, `noisy_and_reduce`, `componentwise_lift`.
- **`extra_quantales.py`:** Additional quantales: `LukasiewiczQuantale`, `GodelQuantale`, `TropicalQuantale`.

### `categorical/`

Categorical structures built from objects and morphisms.

- **`functors.py`:** Functors and functor composition: `Functor`, `IdentityFunctor`, `ComposedFunctor`, `FreeMonoidFunctor`.
- **`natural_transformations.py`:** Natural transformations: `NaturalTransformation`, `ComponentwiseNT`.
- **`adjunctions.py`:** Adjunctions and free-forgetful pairs: `Adjunction`, `ForgetfulFunctor`, `FreeForgetfulAdjunction`.
- **`monoidal.py`:** Monoidal structures: `MonoidalStructure`, `CartesianMonoidal`, `CoproductMonoidal`, `EmptySet`, `EMPTY`.
- **`base_change.py`:** Base change between quantales: `BaseChange`, `BoolToFuzzy`, `FuzzyToBool`.
- **`traced.py`:** Traced monoidal categories: `TracedMonoidal`, `CartesianTrace`, `IterativeTrace`, functions `trace()`, `partial_trace()`.

### `monadic/`

A Haskell-style typeclass hierarchy with concrete monad instances
plus comonads, algebras, distributive laws, transformers, and
algebraic effects. Each typeclass is mirrored by a panproto theory.

Modules:

- **`typeclasses.py`:** `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`, `Foldable`, `Traversable`, `MonadTrans` ABCs.
- **`monads.py`:** `KleisliCategory`, `FuzzyPowersetMonad`, `FreeMonoidMonad` (concrete Monad instances).
- **`comonads.py`:** `Comonad`, `CoKleisliCategory`, `DiagonalComonad`, `CofreeComonad`.
- **`algebras.py`:** Eilenberg–Moore: `Algebra`, `FreeAlgebra`, `ObservedAlgebra`, `Coalgebra`, `CofreeCoalgebra`, `ObservedCoalgebra`, `EilenbergMooreCategory`.
- **`distributive_laws.py`:** `DistributiveLaw`, `FreeMonoidPowersetLaw`.
- **`instances.py`:** stdlib effect instances: `Identity`, `Maybe`, `Alternative_`, `Continuation`, `State`, `Reader`, `Writer`, `List`. Each registers against the appropriate ABC(s) via `ABC.register(...)`.
- **`transformers.py`:** monad transformers: `StateT`, `ReaderT`, `MaybeT`, `ContT`, `WriterT`.
- **`algebraic.py`:** algebraic effects + handlers: `Operation`, `EffectSignature`, `Handler`, `FreeMonad`.
- **`bridges.py`:** `Kleisli`, `ArrowMonad` plus `kleisli` / `arrow_monad` factory helpers connecting the monad and arrow towers.
- **`theories.py`:** panproto-theory mirrors of each typeclass (`ThFunctor`, `ThApplicative`, `ThMonad`, `ThAlternative`, `ThMonadPlus`, `ThMonadTrans`, `ThFoldable`, `ThTraversable`).
- **`laws.py`:** runtime-checkable typeclass-law scaffolding.

### `arrows/`

Hughes-style arrow tower (parallel to the monad-side hierarchy).

- **`typeclasses.py`:** `Category_`, `Arrow`, `ArrowChoice`, `ArrowApply`, `ArrowLoop`, `ArrowZero`, `ArrowPlus` ABCs.
- **`theories.py`:** panproto-theory mirrors of each arrow typeclass.

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

- **`quantale.py`:** Markov quantale: `MarkovQuantale`, singleton `MARKOV`.
- **`morphisms.py`:** Stochastic morphisms: `StochasticMorphism`, `CategoricalMorphism`, `ConditionedMorphism`, `MixtureMorphism`, `FactoredMorphism`, `NormalizedMorphism`.
- **`families.py`:** Discretized parameterized distributions: `DiscretizedNormal`, `DiscretizedLogitNormal`, `DiscretizedBeta`, `DiscretizedTruncatedNormal`.
- **`transforms.py`:** Transforms on stochastic morphisms: `condition()`, `mix()`, `factor()`, `normalize()`.
- **`queries.py`:** Query functions: `prob()`, `marginal_prob()`, `expectation()`.
- **`giry.py`:** The Giry monad on FinStoch: `GiryMonad`, `FinStoch`.

### `continuous/`

Continuous and hybrid discrete-continuous morphisms and monadic programs.

- **`spaces.py`:** Continuous sample spaces: `ContinuousSpace`, `Euclidean`, `UnitInterval`, `Simplex`, `PositiveReals`, `ProductSpace`.
- **`morphisms.py`:** Continuous morphisms: `ContinuousMorphism`, `SampledComposition`, `ProductContinuousMorphism`, `DiscreteAsContinuous`.
- **`families.py`:** 30+ conditional distribution families: `ConditionalNormal`, `ConditionalLogitNormal`, `ConditionalBeta`, `ConditionalTruncatedNormal`, `ConditionalDirichlet`, `ConditionalCauchy`, `ConditionalLaplace`, `ConditionalGumbel`, `ConditionalLogNormal`, `ConditionalStudentT`, `ConditionalExponential`, `ConditionalGamma`, `ConditionalChi2`, `ConditionalHalfCauchy`, `ConditionalHalfNormal`, `ConditionalInverseGamma`, `ConditionalWeibull`, `ConditionalPareto`, `ConditionalKumaraswamy`, `ConditionalContinuousBernoulli`, `ConditionalFisherSnedecor`, `ConditionalUniform`, `ConditionalMultivariateNormal`, `ConditionalLowRankMVN`, `ConditionalRelaxedBernoulli`, `ConditionalRelaxedOneHotCategorical`, `ConditionalWishart`, `ConditionalBernoulli`, `ConditionalCategorical`.
- **`programs.py`:** Monadic programs: `MonadicProgram` (compositional probabilistic computations).
- **`boundaries.py`:** Boundaries between discrete and continuous: `Discretize`, `Embed`.
- **`flows.py`:** Normalizing flows: `AffineCouplingLayer`, `ConditionalFlow`.

### `dsl/`

Domain-specific language for quiver expressions in `.qvr` files. Parsing is delegated to panproto via the `qvr` tree-sitter grammar (located at `grammars/qvr/` in the repo, vendored by `panproto-grammars-all`); quivers does not run a hand-written lexer.

- **`parser.py`:** Walks the panproto-produced parse tree and builds a tree of `dx.Model` AST nodes. Public surface: `parse()`, `parse_file()`, `ParseError`.
- **`ast_nodes.py`:** Every AST node is a `dx.Model`. Recursive sums (`TypeExpr`, `CatPattern`, `SpaceExpr`, `Expr`, `LetExprNode`, `ProgramStep`, `Statement`) are `dx.TaggedUnion` roots discriminated by a `kind: Literal[...]` field.
- **`resolution.py`:** Bidirectional resolution as a `dx.Lens` family. `TypeExprToSetObject` maps `TypeExpr` AST trees to `SetObject` values; `SpaceExprToContinuousSpace` maps `SpaceExpr` trees to `ContinuousSpace` (with a `SetObject` fallback for mixed-domain product spaces). The resolution environment (object/space inventory) is carried on the lens instance — that's the dependent-optics shape.
- **`compiler.py`:** Walks the AST, delegating type and space resolution to the lenses in `resolution.py`, and produces a `quivers.Program`. Public surface: `Compiler`, `CompileError`.
- **`program_theory.py`:** Defines `QVR_PROGRAM_PROTOCOL` (a panproto protocol whose vertex kinds enumerate every `SetObject` and `ContinuousSpace` variant plus the QVR declaration variants) and `extract_program_schema(compiler)`, which walks a compiled environment and emits a panproto `Schema`. This makes every `.qvr` program a schema in panproto's sense — diff, migrate, and lens-generation workflows apply.
- **`pygments_lexer.py`:** A minimal Pygments lexer used to syntax-highlight `.qvr` blocks in this documentation site (registered as the `qvr` lexer entry point).
- **`examples/`:** Reference `.qvr` programs that drive the test suite and the tree-sitter grammar fixtures.

Top-level DSL API: `parse()`, `parse_file()`, `loads()`, `load()`, `Compiler`, `Module`, `QVR_PROGRAM_PROTOCOL`, `extract_program_schema()`, plus exceptions `ParseError`, `CompileError`.

Parametric programs — `program name (G : FinSet, scale : Real, prior : Mor[A, B]) : dom -> cod` — are stored as templates on the compiler (`Compiler._program_templates`) and inlined per call site by substitution + α-renaming; the runtime `MonadicProgram` is built from the inlined step list, so distinct call sites contribute distinct factors to the parent's joint kernel.

### `inference/`

Variational inference for posterior estimation in monadic programs.

- **`trace.py`:** Trace and conditioning: `trace()`, `condition()`.
- **`conditioning.py`:** Conditioning primitives.
- **`guide.py`:** Variational guides: `Guide`.
- **`elbo.py`:** Evidence lower bound: `ELBO`.
- **`svi.py`:** Stochastic variational inference: `SVI`.
- **`predictive.py`:** Predictive posterior: `Predictive`.

### Root-level modules

- **`program.py`:** `Program`: wraps a morphism (discrete or continuous) as a differentiable `nn.Module`.
- **`giry.py`:** Convenience re-export of `GiryMonad` and `FinStoch` from `quivers.stochastic.giry`.

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

1. **core:** foundational (no internal dependencies)
2. **categorical**, **monadic:** build on core
3. **enriched:** builds on categorical
4. **stochastic:** builds on core and monadic
5. **continuous:** builds on core, categorical, monadic, and stochastic
6. **dsl:** synthesizes core, categorical, monadic, stochastic, continuous
7. **inference:** builds on core, categorical, stochastic, continuous

This layering ensures that users can work at the level of abstraction appropriate to their task: core algebra only, categorical structures, probabilistic computations, or the full DSL.

## Design Principles

- **Tensors as morphisms**: $\mathcal{V}$-relations are represented as PyTorch tensors, making them amenable to automatic differentiation and GPU acceleration.
- **Value types are didactic Models**: every record-shaped value (AST nodes, `FinSet`, `ProductSet`, `CoproductSet`, `ContinuousSpace` variants, `Category` variants, `RuleSystem`) is a frozen `dx.Model`. This buys uniform construction, structural equality, JSON round-trips via `model_dump_json` / `model_validate_json`, and cross-field axioms via `__axioms__`. Recursive sums are `dx.TaggedUnion` roots with a `kind: Literal[...]` discriminator.
- **Tensor-bearing classes stay mutable**: `Presheaf`, `Weight`, `SampleSite`, and `Trace` accumulate `torch.Tensor` fields and are mutated in place during inference; they remain `@dataclass`.
- **Lazy composition**: morphism composition creates a DAG; the final tensor is materialized only on evaluation.
- **Type safety**: categorical constraints (domain/codomain compatibility) are enforced statically in Python.
- **Module transparency**: all morphisms expose a `.module()` method returning an `nn.Module`, enabling integration with PyTorch training loops.
- **Quantale flexibility**: the same morphism structure works with any quantale; swapping quantales changes semantics (fuzzy, Boolean, tropical, etc.) without code duplication.
- **DSL pipeline as a panproto theory morphism**: parsing is delegated to panproto's tree-sitter–driven AST registry; resolution is a `dx.Lens` family; each compiled program extracts to a panproto `Schema` against `QVR_PROGRAM_PROTOCOL`. Diff/migrate/lens-generation tools work on `.qvr` programs out of the box.
