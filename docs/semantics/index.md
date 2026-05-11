# Denotational Semantics of the QVR DSL

This section gives a formal denotational semantics for the QVR domain-specific language. Each well-typed phrase $\phi$ is assigned a unique mathematical denotation $\llbracket \phi \rrbracket$ in a fixed semantic universe, parameterised by a choice of quantale $\mathcal{V}$.

The development is organised as follows.

1. **[Setting and notation](setting.md).** The semantic universe: $\mathcal{V}$-enriched symmetric monoidal closed categories, finite sets, standard Borel spaces, and the global parameters that distinguish discrete, stochastic, and continuous fragments.
2. **[Quantales and base change](quantales.md).** The algebraic structure underlying $\mathcal{V}$-enriched composition, the five built-in quantales, and the base-change functors that mediate between them.
3. **[Types and spaces](types-and-spaces.md).** The denotation of `TypeExpr` and `SpaceExpr` syntactic categories as objects in $\mathbf{FinSet}$ and $\mathbf{SBor}$ respectively.
4. **[Morphisms](morphisms.md).** $\mathcal{V}$-relations, stochastic kernels, and continuous conditional families as the three morphism strata; their composition, tensor product, marginalisation, and trace.
5. **[Expressions](expressions.md).** Compositional semantics of expression-level combinators: `>>`, `@`, `.marginalize`, `fan`, `repeat`, `stack`, `scan`, and the parser combinators.
6. **[Programs](programs.md).** Monadic semantics of `program` blocks in the (discrete or continuous) Giry monad; the denotation of `draw`, `observe`, `let`, and `return`.
7. **[Grammar fragment](grammar.md).** Categorical grammars, residuated lattices of category patterns, rule systems, and the chart-parser denotation.
8. **[Compositional effects](effects.md).** Typeclass + algebraic-effects framework over the residuated category universe; class-driven schema lifting; joint type-and-effect dispatch in the chart parser; conservativity over the bare grammar fragment.
9. **[The program theory](program-theory.md).** The schema-level semantics: every compiled program lifts to a panproto `Schema` over the protocol $\mathsf{QVR}$, making syntactic equality of denotations decidable up to the protocol's natural equivalence.
10. **[Adequacy](adequacy.md).** A statement and proof sketch of the adequacy theorem: the compiler implementation $\mathcal{C}$ agrees with the denotation $\llbracket\cdot\rrbracket$ on every well-typed module.

## Conventions

Throughout, we use the following conventions.

| Symbol | Meaning |
|--------|---------|
| $\mathcal{V}$ | A complete commutative quantale. |
| $\otimes,\ \mathbf{1}$ | Monoidal product and unit of $\mathcal{V}$. |
| $\bigoplus$ | Quantale join (the costructure used to marginalise). |
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

This document is written for users who wish to reason formally about QVR programs — to verify that two `.qvr` files denote the same morphism, to prove that an optimisation pass preserves meaning, or to understand the precise relationship between the syntactic AST, the `dx.Model` value-type layer, and the underlying tensor computations. Familiarity with enriched category theory at the level of Kelly's [*Basic Concepts of Enriched Category Theory*](http://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html) and with the categorical foundations of probability ([Lawvere 1973](https://doi.org/10.1007/BF02924844); [Giry 1982](https://doi.org/10.1007/BFb0092872); [Fritz 2020](https://doi.org/10.1016/j.aim.2020.107239)) is assumed.
