# Setting and Notation

This page fixes the semantic universe in which every QVR phrase will be interpreted.

## 1. Quantales as enrichment bases

A *quantale* $\mathcal{V} = (V, \le, \otimes, \mathbf{1})$ is a complete lattice $(V, \le)$ equipped with an associative, monoid product $\otimes : V \times V \to V$ with unit $\mathbf{1} \in V$, such that $\otimes$ distributes over arbitrary joins on both sides:

$$
a \otimes \bigvee_{i \in I} b_i \;=\; \bigvee_{i \in I} (a \otimes b_i),
\qquad
\Bigl(\bigvee_{i \in I} a_i\Bigr) \otimes b \;=\; \bigvee_{i \in I} (a_i \otimes b).
$$

We write $\bigoplus$ for the binary join and (by abuse) for arbitrary joins. We assume $\mathcal{V}$ is *commutative*: $a \otimes b = b \otimes a$.

A quantale is the object of values of a $\mathcal{V}$-enriched category. The five quantales used in QVR are catalogued in [Quantales and base change](quantales.md).

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
| Discrete $\mathcal{V}$-enriched | `latent`, `observed` | $\mathcal{V}\text{-}\mathbf{Rel}$ |
| Stochastic | `stochastic` | $\mathbf{Stoch}$ |
| Continuous | `continuous`, `space` | $\mathbf{Kern}$ |

The three strata are not independent: the inclusion $\iota : \mathbf{FinSet} \hookrightarrow \mathbf{SBor}$ (every finite set is canonically a standard Borel space with the discrete $\sigma$-algebra) lifts to faithful embeddings $\mathbf{Stoch} \hookrightarrow \mathbf{Kern}$ and (in the Boolean quantale case) $\mathcal{V}\text{-}\mathbf{Rel} \hookrightarrow \mathbf{Stoch}$. The `discretize` and `embed` declarations denote the Giry-monad–level transition between strata; see [Morphisms §5](morphisms.md#5-stratum-transitions).

## 5. Environments

A *semantic environment* $\rho$ is a partial function from identifiers to denotations, partitioned into:

- $\rho_{\mathrm{obj}}$: finite-set objects;
- $\rho_{\mathrm{spc}}$: standard Borel spaces;
- $\rho_{\mathrm{mor}}$: morphisms (discrete, stochastic, or continuous);
- $\rho_{\mathrm{cat}}$: category atoms in the grammar fragment.

Following established practice, we write $\rho[x \mapsto v]$ for the environment obtained by extending $\rho$ with the binding $x = v$. The denotation of a phrase $\phi$ in environment $\rho$ is written $\llbracket \phi \rrbracket_{\rho}$; we elide $\rho$ when the binding context is clear.

## 6. Well-typedness

The QVR type system is given by judgments of the form

$$
\Gamma \vdash \phi : \tau,
$$

where $\Gamma$ is a typing context analogous to $\rho$ but tracking only sorts (object / space / morphism with domain–codomain types), and $\tau$ is the assigned sort. The denotational semantics is total on well-typed phrases: the denotation function is defined exactly on the derivation trees of $\Gamma \vdash \phi : \tau$.

We do not present the type system in detail here, the implementation in `quivers.dsl.compiler` is its operational realization, and the [Adequacy](adequacy.md) theorem establishes the soundness of typing with respect to the denotation.
