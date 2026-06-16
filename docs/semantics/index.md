# Denotational Semantics of the QVR DSL

This section gives a formal denotational semantics for the QVR domain-specific language. Each well-typed phrase $\phi$ is assigned a unique mathematical denotation $\llbracket \phi \rrbracket$ in a fixed semantic universe, parameterized by a choice of algebra $\mathcal{V}$.

The development is organized as follows.

1. **[Setting and notation](setting.md).** The semantic universe: $\mathcal{V}$-enriched symmetric monoidal closed categories, finite sets, standard Borel spaces, and the global parameters that distinguish discrete, stochastic, and continuous fragments.
2. **[Algebras and base change](algebras.md).** The algebraic structure underlying $\mathcal{V}$-enriched composition, the eleven built-in algebras, and the base-change functors that mediate between them.
3. **[Composition rules](composition-rules.md).** The class hierarchy $\mathbf{CompositionRule} \supseteq \{\mathbf{BilinearForm},\ \mathbf{Semigroupoid} \supseteq \mathbf{Algebra}\}$; user-defined inline composition rules; operadic n-ary contractions via flat wirings; first-class transformations with sequential composition.
4. **[Types and spaces](types-and-spaces.md).** The denotation of `ObjectExpr` and `SpaceExpr` syntactic categories as objects in $\mathbf{FinSet}$ and $\mathbf{SBor}$ respectively.
5. **[Morphisms](morphisms.md).** $\mathcal{V}$-relations, stochastic kernels, and continuous conditional families as the three morphism strata; their composition, tensor product, marginalization, and trace.
6. **[Expressions](expressions.md).** Compositional semantics of expression-level combinators: `>>`, `>>>`, `@`, `.marginalize`, `.change_base`, `fan`, `repeat`, `stack`, `scan`, and the parser combinators.
7. **[Programs](programs.md).** Monadic semantics of `program` blocks in the (discrete or continuous) Giry monad; the denotation of bind (`<-`), `observe`, `let`, `score`, `marginalize`, and `return`; effect signatures via the option-block `[effects = [Sample, Score, Marginal, Pure]]`; grouped marginalization with multi-observe fibration; parametric program templates and their instantiation.
8. **[Typing](typing.md).** The type system as a proof calculus: kinds, types, families, contexts ($\Gamma$, $\Delta$, $\Phi$), judgments ($\Gamma \vdash \tau : \kappa$, $\Gamma; \Phi \vdash e : A \rightsquigarrow B$, $\Gamma; \Phi \vdash s \dashv \Phi'$, $\Gamma \vdash p : (\Delta) \Rightarrow A \rightsquigarrow B$), the full inference-rule set, the soundness theorem against the denotation, subject reduction for parametric instantiation, and the bidirectional algorithm underlying the compiler's typechecker.
9. **[Weighted deduction fragment](grammar.md).** Item algebras, rule systems as hyperedges in a multicategory, semiring-weighted chart enumeration, axiom injectors, strategy independence, and differentiable charts. Includes learnable per-rule and bindings-keyed weights, hierarchical (`parent=`) and bounded (`bounded`) rule parameterisations, alpha-renamed binder blocks, convergent-cycle (`tolerance=`) chart evaluation, and the chart-access surface (`parse(D, x)`, `compose(D_1, D_2)`, `subst(t, v, w)`) that lifts a chart's goal weight into a `program`'s log-joint via a [`score`](programs.md#27a-score-factor) step.
10. **[Schemas, rules, categories, bundles](schemas.md).** Category atoms; rule declarations as universally-quantified hyperedges; pattern-polymorphic schema declarations; bundles as first-class rule sets; the residuated type formers $/$, $\backslash$, $T(\cdot)$; free residuated category universes; object- and space-level aliases.
11. **[Structural compression](structural.md).** Signatures as generalized algebraic theories; encoders as initial-algebra catamorphisms into a vector carrier; decoders as Kleisli coalgebras of the Giry monad; losses as attached scalar functionals on training-site traces.
12. **[Compositional effects](effects.md).** Typeclass + algebraic-effects framework over a residuated category universe; class-driven schema lifting; joint type-and-effect dispatch in the chart parser; conservativity over the bare deduction fragment.
13. **[The program theory](program-theory.md).** The schema-level semantics: every compiled program lifts to a panproto `Schema` over the protocol $\mathsf{QVR}$, making syntactic equality of denotations decidable up to the protocol's natural equivalence.
14. **[Adequacy](adequacy.md).** A statement and proof sketch of the adequacy theorem: the compiler implementation $\mathcal{C}$ agrees with the denotation $\llbracket\cdot\rrbracket$ on every well-typed module.
15. **[Transpilation architecture](transpile-architecture.md).** The three-stage Mapping pipeline $\mathrm{Module} \to \mathrm{IR} \to \mathrm{Schema} \to \mathrm{bytes}$ implementing the eleven target backends: the intermediate representation (`IRProgram`, `Plate`, `IRArg`, `IRNode`) with support classification built on [`torch.distributions.constraints`](https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.constraints), the `FAMILY_META` registry (single source of truth for per-target distribution names + argument aliases), the `Renderer` protocol with its four dispatch points (`declare`, `sample`, `marginalize`, `broadcast`), and the five structural invariants (no `if family == ...` in any renderer, no silent drops of AST fields, no fallbacks, backend-symmetric abstractions, single source of truth per concept). Includes the LDA end-to-end trace through Stan and NumPyro, and the how-to-add a new family / new backend recipes.
16. **[Transpilation correctness](transpile-correctness.md).** A category-theoretic treatment of the same transpilation: the source category $\mathbf{Mod}^*_{\mathrm{QVR}}$, the target categories $\mathbf{Prog}^*_{\mathsf{T}}$, the transpile functor $\mathsf{T}_{\mathsf{T}}$ as a three-arrow composite in the Kleisli-style category of $\mathsf{dx.Mapping}$, and the natural isomorphism $\eta_{\mathsf{T}} : \mathsf{S}_{\mathrm{QVR}} \xRightarrow{\,\cong\,} \mathsf{S}_{\mathsf{T}} \circ \mathsf{T}_{\mathsf{T}}$ certifying that the target program denotes the same conditional joint distribution as the QVR source, up to the constant absorbed by reparametrization. Proof by structural induction, with the per-step density identity laid out family-by-family against the canonical parameterizations.

## The unified declaration surface

Every QVR declaration shares the same skeleton:

```
KIND NAME : SIGNATURE [k = v, ...] [~ INIT] [BODY]
```

where:

* `KIND` is one of `composition`, `category`, `object`, `morphism`, `bundle`, `program`, `contraction`, `deduction`, `signature`, `encoder`, `decoder`, `loss`, `let`, `export`, `schema`, `rule`. There are no per-kind keyword variants for the *behavior* of a declaration; behavior is selected in the option block.
* `SIGNATURE` is a colon-prefixed phrase (an object value for `object`, a `dom -> cod` arrow for everything that denotes a morphism, an `inputs / codomain` shape for `contraction`, etc.).
* `[k = v, ...]` is the *unified option block*: a single bracketed key-value list that drives every per-declaration knob (`role`, `scale`, `init`, `axes`, `effects`, `over`, `replicate`, `semiring`, `start`, `depth`, `tolerance`, `max_iterations`, `bins`, `path`, `weight`, `parent`, …). Every parser walker reads the same `OptionEntry` tree; every compiler reads the same `get_option_{flag,name,int,float,string,name_list}` accessors. The block is denotationally inert at the categorical level: it selects which functor / kernel / parametric family the declaration denotes, never how composition fires.
* `~ INIT` is the optional *initializer*: either a `Family(args)` clause (sampled stochastic kernel, evaluated through the family registry) or an arbitrary `expr` (deterministic morphism, evaluated through the expression denotation). On a `morphism` declaration the initializer interacts with `role` in the option block: `role=latent` admits only `~ Family(...)`; `role=let` / `role=observed` admits only `~ expr`; `role=kernel` admits both.
* `[BODY]` is an optional indented block: rule list for `composition` / `deduction` / `signature`, statement list for `program`, etc.

Pragmas are top-level *attribute* statements that decorate the *next* declaration (outer form `#[k = v, ...]`) or the *enclosing module* (inner form `#![k = v, ...]`). They carry the same `pragma_entry` shape as the unified option block; the compiler attaches the entries to the decorated declaration or module-level environment.

The deduction fragment adds a parallel form: lexicon entries and individual `rule` lines admit a trailing `#[k = v, ...]` pragma that controls per-entry / per-rule behavior (`learnable`, `bounded`, `parent`, `weight = expr`, etc.; see [Weighted deduction fragment §2.1–§2.3](grammar.md#21-learnable-rule-weights-learnable)).

## Conventions

Throughout, we use the following conventions.

| Symbol | Meaning |
|--------|---------|
| $\mathcal{V}$ | A QVR algebra (see [Algebras §1](algebras.md)). |
| $\otimes,\ \mathbf{1}$ | Monoidal product and unit of $\mathcal{V}$. |
| $\bigoplus$ | The join of $\mathcal{V}$ (the costructure used to marginalize). |
| $\mathbf{FinSet}$ | The category of finite sets. |
| $\mathbf{SBor}$ | The category of standard Borel spaces with measurable maps. |
| $\mathcal{V}\text{-}\mathbf{Rel}$ | The $\mathcal{V}$-enriched category of $\mathcal{V}$-relations on finite sets. |
| $\mathbf{Stoch}$ | The Kleisli category of the (discrete) Giry monad. |
| $\mathbf{Kern}$ | The category of standard Borel spaces and Markov kernels. |
| $\mathcal{G}$ | The Giry monad (discrete or continuous, disambiguated by context). |
| $\llbracket \phi \rrbracket$ | The denotation of phrase $\phi$. |
| $\Gamma \vdash \phi : \tau$ | $\phi$ is well-typed of type $\tau$ under environment $\Gamma$. |
| $\rho$ | A semantic environment (assignment of denotations to free names). |

A *type* in the QVR sense is a finite-set object; a *space* is a standard Borel object. We write $|X|$ for the cardinality of a finite set $X$, and $\dim(S)$ for the dimension of a continuous space $S$.

## Audience

This document is written for users who wish to reason formally about QVR programs: to verify that two `.qvr` files denote the same morphism, to prove that an optimization pass preserves meaning, or to understand the precise relationship between the syntactic AST, the `dx.Model` value-type layer, and the underlying tensor computations. Familiarity with enriched category theory at the level of Kelly's [*Basic Concepts of Enriched Category Theory*](http://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html) and with the categorical foundations of probability ([Giry, 1982](https://doi.org/10.1007/BFb0092872); [Fritz, 2020](https://doi.org/10.1016/j.aim.2020.107239)) is assumed.


## References

- Tobias Fritz. 2020. A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics. *Advances in Mathematics*, 370:107239.
- Michèle Giry. 1982. A categorical approach to probability theory. In Bernhard Banaschewski, editor, *Categorical Aspects of Topology and Analysis*, volume 915 of *Lecture Notes in Mathematics*, pages 68–85. Springer, Berlin, Heidelberg.
- Gregory M. Kelly. 1982. *Basic Concepts of Enriched Category Theory*. Cambridge University Press; reprinted as *Reprints in Theory and Applications of Categories* 10 (2005):1–136.
