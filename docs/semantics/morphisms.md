# Morphisms

This page assigns denotations to the three QVR morphism strata (discrete $\mathcal{V}$-enriched, stochastic, and continuous) and to the structural transitions between them. Throughout, we fix a quantale $\mathcal{V}$ as in [Setting and notation](setting.md).

## 1. Discrete $\mathcal{V}$-enriched morphisms

A discrete morphism declaration

```
latent f : τ₁ -> τ₂
```

denotes a morphism of $\mathcal{V}\text{-}\mathbf{Rel}$:

$$
\llbracket f \rrbracket : \llbracket \tau_1 \rrbracket \to \llbracket \tau_2 \rrbracket,
\qquad \llbracket f \rrbracket \in V^{|{\llbracket \tau_1 \rrbracket}| \times |{\llbracket \tau_2 \rrbracket}|}.
$$

For `latent`, the entries are *free parameters* drawn from $V$; the realisation in PyTorch is a tensor of `requires_grad = True` parameters passed through a constraint map $\sigma : \mathbb{R} \to V$ (the sigmoid for $\mathcal{V}_{\mathrm{pf}}$, the identity for $\mathcal{V}_{\mathrm{T}}$, etc.).

For `observed g : τ₁ -> τ₂ = data`, the entries are *fixed*: $\llbracket g \rrbracket(x, y) = \mathrm{data}[x, y]$.

The implementation does not distinguish the two cases at the categorical level: both produce objects of $\mathcal{V}\text{-}\mathbf{Rel}$. The latent/observed distinction is operational, controlling the gradient-flow during training.

### 1.1 Composition, tensor, identity

Composition $;$, tensor $\boxtimes$, and identity $1_X$ in $\mathcal{V}\text{-}\mathbf{Rel}$ are defined as in [Setting and notation §2](setting.md#2-mathcalv-enriched-relations). The DSL operators correspond directly:

| Syntax | Denotation | Definition |
|--------|-----------|------------|
| `f >> g` | $\llbracket g \rrbracket \circ \llbracket f \rrbracket$ | $(f; g)(x, z) = \bigoplus_y f(x, y) \otimes g(y, z)$ |
| `f @ g` | $\llbracket f \rrbracket \boxtimes \llbracket g \rrbracket$ | $(f \boxtimes g)((x_1, x_2), (y_1, y_2)) = f(x_1, y_1) \otimes g(x_2, y_2)$ |
| `identity(X)` | $1_{\llbracket X \rrbracket}$ | $1_X(x, x') = \mathbf{1}$ if $x = x'$, $\bot$ otherwise |

### 1.2 Marginalisation

For $f : X \otimes Y \to Z$ in $\mathcal{V}\text{-}\mathbf{Rel}$, the expression `f.marginalize(X)` denotes the $\mathcal{V}$-enriched colimit (quantale-join) of $f$ along the $X$-coordinate:

$$
\llbracket f.\mathrm{marginalize}(X) \rrbracket(y, z) \;=\; \bigoplus_{x \in \llbracket X \rrbracket} \llbracket f \rrbracket\bigl((x, y), z\bigr),
$$

a morphism $Y \to Z$. Equivalently, it is the postcomposition with the $\mathcal{V}$-relation $\top_X : X \to \mathbf{1}$ that sends every $x$ to $\mathbf{1}$, after the canonical reassociation $X \otimes Y \cong Y \otimes X$.

## 2. Stochastic morphisms

A `kernel` declaration with finite-set codomain and no `~ Family` clause,

```
kernel kern : τ₁ -> τ₂
```

denotes a morphism of $\mathbf{Stoch}$, i.e.\ a row-stochastic $|{\llbracket \tau_1 \rrbracket}| \times |{\llbracket \tau_2 \rrbracket}|$ matrix (each row $\llbracket \mathrm{kern} \rrbracket(x, \cdot)$ is a probability distribution over $\llbracket \tau_2 \rrbracket$):

$$
\llbracket \mathrm{kern} \rrbracket(x, \cdot) \in \mathcal{G}_{\mathrm{fin}}\bigl(\llbracket \tau_2 \rrbracket\bigr)
\quad \text{for every } x \in \llbracket \tau_1 \rrbracket.
$$

The implementation realises $\llbracket \mathrm{kern} \rrbracket$ via softmax over a parameter tensor; this is the canonical surjection $\mathbb{R}^{|Y|} \twoheadrightarrow \Delta^{|Y|-1}$ from raw logits onto the simplex.

Composition is the Kleisli composition for $\mathcal{G}_{\mathrm{fin}}$:

$$
(k_1; k_2)(x, z) \;=\; \sum_{y} k_1(x, y) \cdot k_2(y, z).
$$

Note that $\mathbf{Stoch}$ is *not* a sub-category of $\mathcal{V}_{\mathrm{pf}}\text{-}\mathbf{Rel}$: their composition operations differ. $\mathcal{V}_{\mathrm{pf}}\text{-}\mathbf{Rel}$ uses noisy-OR aggregation (the t-conorm $\bigoplus a_i = 1 - \prod (1 - a_i)$) while $\mathbf{Stoch}$ uses ordinary summation, which is the correct aggregation for events partitioning the sample space. The two categories share an underlying tensor representation in $[0, 1]^{|X| \times |Y|}$, but the categorical structure is supplied by different binary operations.

### 2.1 Distribution-family morphisms

Continuous distribution families *parameterised* by a finite set, declared with

```
kernel f : τ₁ -> σ ~ Family
```

denote morphisms of $\mathbf{Kern}$:

$$
\llbracket f \rrbracket : \iota\bigl(\llbracket \tau_1 \rrbracket\bigr) \to \llbracket \sigma \rrbracket,
\qquad
\llbracket f \rrbracket(x, B) \;=\; \int_B p_{\mathrm{Family}}(y \,;\, \theta(x))\, \mathrm{d}y,
$$

where $\theta : \llbracket \tau_1 \rrbracket \to \Theta$ is the family's parameter map (typically a neural network), and $p_{\mathrm{Family}}(\cdot \,;\, \theta)$ is the density of the family at parameter $\theta$. The QVR-supplied family registry catalogues the pairs $(\Theta, p)$ for each name (Normal, Beta, Dirichlet, …).

## 3. Continuous morphisms

A `kernel` declaration whose source is itself a space, e.g.

```
kernel g : σ₁ -> σ₂ ~ Family
```

denotes a Markov kernel between standard Borel spaces:

$$
\llbracket g \rrbracket : \llbracket \sigma_1 \rrbracket \to \llbracket \sigma_2 \rrbracket,
\qquad
\llbracket g \rrbracket(s, B) \;=\; \int_B p_{\mathrm{Family}}(t \,;\, \theta(s))\, \mathrm{d}t.
$$

Composition is the Chapman–Kolmogorov integral

$$
(g_1; g_2)(s, C) \;=\; \int_{\llbracket \sigma_2 \rrbracket} g_2(t, C) \, g_1(s, \mathrm{d}t),
$$

realised numerically by Monte-Carlo or sampled-composition approximation in the implementation.

## 4. Tensor product across strata

The `@` combinator extends to all three strata via the canonical monoidal structures:

| Strata of $f$ and $g$ | Ambient category of $f \otimes g$ |
|------------------------|------------------------------------|
| Both discrete | $\mathcal{V}\text{-}\mathbf{Rel}$ |
| Both stochastic | $\mathbf{Stoch}$ |
| Both continuous | $\mathbf{Kern}$ |
| Mixed | The smallest enclosing category, via the canonical embeddings $\mathcal{V}_{\mathbb{B}}\text{-}\mathbf{Rel} \hookrightarrow \mathbf{Stoch} \hookrightarrow \mathbf{Kern}$ of [Setting §4](setting.md#4-the-three-semantic-strata) |

The denotation is the parallel product of kernels:

$$
\llbracket f \otimes g \rrbracket\bigl((x, y), B \times C\bigr) \;=\; \llbracket f \rrbracket(x, B) \cdot \llbracket g \rrbracket(y, C),
$$

extended uniquely (by the standard product-measure construction) to the product $\sigma$-algebra.

## 5. Stratum transitions

The `discretize` and `embed` declarations witness the canonical functors between strata.

### 5.1 Discretisation

```
discretize d : σ -> n
```

denotes the *quotient* kernel induced by a measurable partition of $\llbracket \sigma \rrbracket$ into $n$ Borel cells $B_0, \dots, B_{n-1}$:

$$
\llbracket d \rrbracket(s, \{i\}) \;=\; \mathbf{1}_{B_i}(s),
$$

i.e.\ the deterministic kernel sending each $s$ to the (unique) cell containing it. As a morphism of $\mathbf{Kern}$ it factors through $\iota(\{0, \dots, n-1\})$. The choice of partition is supplied by the family annotation (e.g.\ uniform quantiles, learned thresholds).

### 5.2 Embedding

```
embed e : n -> σ
```

denotes a *section* of a discretisation: a kernel $e : \iota(\{0, \dots, n-1\}) \to \llbracket \sigma \rrbracket$ such that the composite $e ; d = \mathrm{id}$. In the implementation this is typically realised by sampling from a per-cell *embedding family* $p_{\mathrm{Family}}(\cdot \,;\, \theta_i)$.

The pair $(d, e)$ is a *retraction* in the categorical sense; it is not generally an isomorphism (information about the within-cell distribution is lost by $d$).

## 6. Equivariance and naturality

The constructions above are functorial in the chosen quantale (for the discrete stratum, [Quantales §4](quantales.md#4-functoriality-of-the-language)) and in the choice of family registry (for the stochastic and continuous strata). In particular, any uniform substitution of one family for another that preserves the parameter-map signature induces a natural transformation of denotations; this is the formal underpinning of the family-registry lookup tables in [`quivers.continuous.families`](../api/continuous/families.md) and [`quivers.stochastic.families`](../api/stochastic/families.md).
