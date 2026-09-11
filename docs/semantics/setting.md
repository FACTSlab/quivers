# Setting and Notation

This page fixes the semantic universe in which every QVR phrase will be interpreted.

## 1. Algebras as enrichment bases

The implementation calls any instance of `quivers.core.algebras.Algebra` a *QVR algebra*. This is an API term: the class supplies `tensor_op`, `join`, `meet`, `negate`, `unit`, and `zero`, but it does not prove the corresponding laws. A strict [quantale](https://ncatlab.org/nlab/show/quantale) supplies the stronger structure needed for the usual enriched-category results: a complete lattice $V$, a monoid product $\otimes$, and arbitrary joins $\bigvee$ over which $\otimes$ distributes on both sides:

$$
a \otimes \bigvee_{i \in I} b_i \;=\; \bigvee_{i \in I} (a \otimes b_i),
\qquad
\Bigl(\bigvee_{i \in I} a_i\Bigr) \otimes b \;=\; \bigvee_{i \in I} (a_i \otimes b).
$$

We write $\bigoplus$ for the implementation's reduction operation. It is a lattice join only for the built-ins whose reduction is idempotent. Results that invoke arbitrary joins, bottom elements, or distributivity thus carry an explicit strict-quantale hypothesis.

Strict quantales are standard enrichment bases. QVR's eleven built-ins are cataloged in [Algebras and base change](algebras.md), but several are finite-reduction semirings or t-norm/t-conorm pairs rather than strict quantales. QVR still evaluates their tensor contractions through the shared `Algebra` interface. It does not follow that every categorical equation in this chapter holds for every built-in.

## 2. $\mathcal{V}$-enriched relations

Let $X, Y$ be finite sets. A *$\mathcal{V}$-relation* from $X$ to $Y$ is a function

$$
r : X \times Y \to V.
$$

We denote by $\mathcal{V}\text{-}\mathbf{Rel}$ the category whose objects are finite sets and whose hom-objects $\mathcal{V}\text{-}\mathbf{Rel}(X, Y)$ are $\mathcal{V}^{|X| \times |Y|}$. Composition is the $\mathcal{V}$-matrix product:

$$
(r ; s)(x, z) \;=\; \bigoplus_{y \in Y} r(x, y) \otimes s(y, z),
\qquad r : X \to Y,\ s : Y \to Z.
$$

Identities are the indicator $\mathcal{V}$-relations $1_X(x, x') = \mathbf{1}$ if $x = x'$ and $\bot$ otherwise, where $\bot$ is the bottom of the lattice. The category $\mathcal{V}\text{-}\mathbf{Rel}$ is symmetric monoidal: tensor product is given by

$$
(r \boxtimes s)((x_1, x_2), (y_1, y_2)) \;=\; r(x_1, y_1) \otimes s(x_2, y_2).
$$

## 3. Standard Borel spaces and Markov kernels

A *standard Borel space* is a measurable space $(S, \Sigma)$ isomorphic to a Borel subset of a Polish space. Write $\mathbf{SBor}$ for the category of standard Borel spaces with measurable maps.

A *Markov kernel* from $S$ to $T$ is a function $k : S \times \Sigma_T \to [0, 1]$ such that $k(s, \cdot)$ is a probability measure on $T$ for every $s$, and $k(\cdot, B)$ is measurable for every $B \in \Sigma_T$. The category $\mathbf{Kern}$ has standard Borel spaces as objects and Markov kernels as morphisms, with composition

$$
(k_1 ; k_2)(s, C) \;=\; \int_T k_2(t, C) \, k_1(s, \mathrm{d}t).
$$

It is the Kleisli category of the *Giry monad* $\mathcal{G} : \mathbf{SBor} \to \mathbf{SBor}$, $S \mapsto \mathcal{G}(S) = \{\mu \mid \mu \text{ probability measure on } S\}$.

We write $\mathbf{Stoch}$ for the finite-set restriction: the Kleisli category of the *finitary* Giry monad $\mathcal{G}_{\mathrm{fin}}$ on $\mathbf{FinSet}$, whose hom-sets are stochastic matrices.

## 4. The three semantic strata

QVR phrases inhabit three strata, each interpreted in a distinct ambient category.

| Stratum | Source of morphism declaration | Ambient category |
|---------|--------------------------------|-------------------|
| Discrete $\mathcal{V}$-enriched | `latent`, `observed` (no `~ Family` clause) | $\mathcal{V}\text{-}\mathbf{Rel}$ |
| Stochastic | `kernel` between finite-set types (no `~ Family` clause) | $\mathbf{Stoch}$ |
| Continuous | `morphism ... ~ Family` with a continuous codomain; continuous `object` declarations | $\mathbf{Kern}$ |

The three strata are not independent: the inclusion $\iota : \mathbf{FinSet} \hookrightarrow \mathbf{SBor}$ (every finite set is canonically a standard Borel space with the discrete $\sigma$-algebra) lifts to a faithful embedding $\mathbf{Stoch} \hookrightarrow \mathbf{Kern}$, and the *functional* sub-category $\mathcal{V}_{\mathbb{B}}\text{-}\mathbf{Rel}_{\mathrm{fun}}$ of row-deterministic Boolean relations embeds into $\mathbf{Stoch}$. The `discretize` and `embed` declarations denote the Giry-monad–level transition between strata; see [Morphisms §5](morphisms.md#5-stratum-transitions).

## 5. Environments

A *semantic environment* $\rho$ is a partial function from identifiers to denotations, partitioned into:

- $\rho_{\mathrm{obj}}$: finite-set objects;
- $\rho_{\mathrm{spc}}$: standard Borel spaces;
- $\rho_{\mathrm{mor}}$: morphisms (discrete, stochastic, or continuous);
- $\rho_{\mathrm{cat}}$: category atoms in the grammar fragment;
- $\rho_{\mathrm{rv}}$: random variables bound earlier in a `program` body, each carrying its current Kleisli arrow (see [Programs §1](programs.md#1-the-giry-monad-as-semantic-substrate)).

Following established practice, we write $\rho[x \mapsto v]$ for the environment obtained by extending $\rho$ with the binding $x = v$. The denotation of a phrase $\phi$ in environment $\rho$ is written $\llbracket \phi \rrbracket_{\rho}$; we elide $\rho$ when the binding context is clear.

## 6. Well-typedness

The QVR type system is given by judgments of the form

$$
\Gamma \vdash \phi : \tau,
$$

where $\Gamma$ is a typing context analogous to $\rho$ but tracking only sorts (object / space / morphism with domain–codomain types), and $\tau$ is the assigned sort. The compiler resolves and checks the supported phrase forms before constructing runtime objects. The judgments in these pages specify that behavior; they are not a machine-checked proof calculus.

The [Typing](typing.md) chapter records the declarative judgments and relates them to checks in `quivers.dsl.compiler`. [Correspondence and limitations](adequacy.md) separates equations exercised by tests from broader mathematical assumptions.
