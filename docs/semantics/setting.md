# Setting and Notation

This page fixes the semantic universe in which every QVR phrase will be interpreted.

## 1. Algebras as enrichment bases

A *QVR algebra* $\mathcal{V} = (V, \le, \otimes, \oplus, \mathbf{1})$ is a partially-ordered carrier set $V$ equipped with an associative, commutative monoid product $\otimes : V \times V \to V$ (unit $\mathbf{1}$) and a join operation $\oplus$ used for marginalization. The strongest case is a [*strict quantale*](https://ncatlab.org/nlab/show/quantale) in [Kelly's sense](https://ncatlab.org/nlab/show/enriched+category): $V$ is a complete lattice, $\oplus$ is the arbitrary-join $\bigvee$, and $\otimes$ distributes over arbitrary joins on both sides:

$$
a \otimes \bigvee_{i \in I} b_i \;=\; \bigvee_{i \in I} (a \otimes b_i),
\qquad
\Bigl(\bigvee_{i \in I} a_i\Bigr) \otimes b \;=\; \bigvee_{i \in I} (a_i \otimes b).
$$

We write $\bigoplus$ for the join (binary or indexed) throughout. We assume $\mathcal{V}$ is commutative: $a \otimes b = b \otimes a$.

Strict quantales are the value objects of [$\mathcal{V}$-enriched categories](https://ncatlab.org/nlab/show/enriched+category) in Kelly's sense. The eleven built-in algebras used in QVR are cataloged in [Algebras and base change](algebras.md); not all of them are strict quantales. The idempotent ($\mathcal{V}_{\mathbb{B}}$, $\mathcal{V}_{\mathrm{G}}$), tropical/Viterbi ($\mathcal{V}_{\mathrm{T}}$, $\mathcal{V}_{\mathrm{MP}}$), and log-additive ($\mathcal{V}_{\mathrm{LP}}$, $\mathcal{V}_{\mathbb{R}}$, $\mathcal{V}_{\mathbb{N}}$) cases satisfy the strict quantale-distributivity law exactly. The t-norm pairs $\mathcal{V}_{\mathrm{pf}}$ and $\mathcal{V}_{\mathrm{L}}$ are [commutative residuated lattices](https://ncatlab.org/nlab/show/residuated+lattice) for which the law holds only laxly; the saturated-sum probability algebra $\mathcal{V}_{[0,1]}$ has its own caveat at the saturation boundary. The development of the present page is uniform across all eleven; the lax cases are flagged in [Algebras §2.1](algebras.md#21-a-note-on-the-product-fuzzy-and-lukasiewicz-pairs) where the distinction matters.

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
| Continuous | `kernel ... ~ Family` (with finite or continuous dom and continuous cod); `space` | $\mathbf{Kern}$ |

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

where $\Gamma$ is a typing context analogous to $\rho$ but tracking only sorts (object / space / morphism with domain–codomain types), and $\tau$ is the assigned sort. The denotational semantics is total on well-typed phrases: the denotation function is defined exactly on the derivation trees of $\Gamma \vdash \phi : \tau$.

We do not present the type system in detail here, the implementation in `quivers.dsl.compiler` is its operational realization, and the [Adequacy](adequacy.md) theorem establishes the soundness of typing with respect to the denotation.
