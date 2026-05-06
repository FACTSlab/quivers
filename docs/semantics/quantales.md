# Quantales and Base Change

The choice of quantale $\mathcal{V}$ in a `quantale` declaration determines the truth-value semantics of every $\mathcal{V}$-relation in the program. This page records the formal definition of each built-in quantale and the base-change functors that mediate between them.

## 1. The five quantales

Each quantale is determined by its underlying lattice and its monoidal product. Joins are taken pointwise on $V$; the unit $\mathbf{1}$ is the maximum element with respect to the order.

### 1.1 Product fuzzy quantale

$$
\mathcal{V}_{\mathrm{pf}} \;=\; \bigl([0, 1],\ \le,\ \cdot,\ 1\bigr),
\qquad a \otimes b = a \cdot b,
\qquad \bigoplus_i a_i = 1 - \prod_i (1 - a_i).
$$

The join is *noisy-OR*. This is the QVR default.

### 1.2 Boolean quantale

$$
\mathcal{V}_{\mathbb{B}} \;=\; \bigl(\{0, 1\},\ \le,\ \wedge,\ 1\bigr),
\qquad \bigoplus_i a_i = \bigvee_i a_i.
$$

The classical two-element lattice; $\mathcal{V}_{\mathbb{B}}\text{-}\mathbf{Rel}$ is the category of ordinary binary relations on finite sets.

### 1.3 Łukasiewicz quantale

$$
\mathcal{V}_{\mathrm{L}} \;=\; \bigl([0, 1],\ \le,\ \otimes_{\mathrm{L}},\ 1\bigr),
\qquad a \otimes_{\mathrm{L}} b = \max(0, a + b - 1),
\qquad \bigoplus_i a_i = \min(1, \textstyle\sum_i a_i).
$$

### 1.4 Gödel quantale

$$
\mathcal{V}_{\mathrm{G}} \;=\; \bigl([0, 1],\ \le,\ \min,\ 1\bigr),
\qquad \bigoplus_i a_i = \max_i a_i.
$$

### 1.5 Tropical quantale

$$
\mathcal{V}_{\mathrm{T}} \;=\; \bigl([0, +\infty],\ \ge,\ +,\ 0\bigr),
\qquad \bigoplus_i a_i = \min_i a_i.
$$

The order is reversed: smaller is "truer". The unit is $0$ (the additive identity), and the bottom of the lattice (the largest element) is $+\infty$. Composition is the *min-plus* matrix product, suitable for shortest-path semantics.

## 2. Order- and structure-preservation

For each quantale we record which elementary properties hold; these determine which categorical constructions transport across base change.

| Quantale | Idempotent? | Integral? | Cancellative? |
|----------|-------------|-----------|---------------|
| $\mathcal{V}_{\mathrm{pf}}$ | No | Yes ($\mathbf{1} = 1$) | No |
| $\mathcal{V}_{\mathbb{B}}$ | Yes | Yes | No |
| $\mathcal{V}_{\mathrm{L}}$ | No | Yes | No |
| $\mathcal{V}_{\mathrm{G}}$ | Yes | Yes | No |
| $\mathcal{V}_{\mathrm{T}}$ | No | Yes ($\mathbf{1} = 0$) | Yes |

A quantale is *integral* if $\mathbf{1}$ is the top element and *idempotent* if $a \otimes a = a$ for all $a$. Cancellativity means $a \otimes b = a \otimes c \Rightarrow b = c$ whenever $a \neq \bot$. Idempotent integral quantales are precisely *frames* (locales).

### 2.1 A note on the product-fuzzy and Łukasiewicz pairs

The product-fuzzy and Łukasiewicz $(\otimes, \oplus)$ pairs are *t-norm / t-conorm* pairs on $[0, 1]$, equipping the unit interval with the structure of a *commutative residuated lattice* (an MV-algebra in the Łukasiewicz case). They are **not** strict quantales: the full quantale-distributivity law

$$
a \otimes \bigoplus_{i} b_i \;=\; \bigoplus_i (a \otimes b_i)
$$

fails in general for these two pairs.

For example, in the product-fuzzy pair with $a = b_1 = b_2 = 1/2$:

$$
a \otimes (b_1 \oplus b_2) \;=\; \tfrac{1}{2}\bigl(1 - \tfrac{1}{4}\bigr) \;=\; \tfrac{3}{8},
\qquad
(a \otimes b_1) \oplus (a \otimes b_2) \;=\; 1 - \tfrac{9}{16} \;=\; \tfrac{7}{16}.
$$

Strict quantale-distributivity holds in the idempotent ($\mathcal{V}_{\mathbb{B}}$ and $\mathcal{V}_{\mathrm{G}}$) and tropical ($\mathcal{V}_{\mathrm{T}}$) cases. For $\mathcal{V}_{\mathrm{pf}}$ and $\mathcal{V}_{\mathrm{L}}$, the categorical apparatus of $\mathcal{V}\text{-}\mathbf{Rel}$ should be read as describing the *Bayesian noisy-OR* (resp.\ *Łukasiewicz-bounded-sum*) aggregation under the multiplicative t-norm, rather than as a strict $\mathcal{V}$-enriched category in Kelly's sense. Composition, tensor, and the equational laws of [Expressions §5](expressions.md#5-coherence-and-equational-laws) hold up to this standard caveat: equations involving $\bigoplus$-distribution over $\otimes$ are exact in the idempotent / tropical cases and approximate in the t-norm cases.

## 3. Base change

A *quantale homomorphism* $h : \mathcal{V} \to \mathcal{W}$ is a join-preserving monoid map: $h(a \otimes_{\mathcal{V}} b) = h(a) \otimes_{\mathcal{W}} h(b)$, $h(\mathbf{1}_{\mathcal{V}}) = \mathbf{1}_{\mathcal{W}}$, and $h$ commutes with arbitrary joins.

Each homomorphism induces a *base-change functor*

$$
h_* : \mathcal{V}\text{-}\mathbf{Rel} \to \mathcal{W}\text{-}\mathbf{Rel},
\qquad (h_* r)(x, y) = h(r(x, y)),
$$

which acts as the identity on objects and pointwise on $\mathcal{V}$-relations. Functoriality is a direct consequence of $h$ preserving $\otimes$ and $\bigoplus$.

The implementation provides two canonical base-change functors:

- $\beta : \mathcal{V}_{\mathbb{B}} \to \mathcal{V}_{\mathrm{pf}}$, the inclusion $\{0, 1\} \hookrightarrow [0, 1]$;
- $\theta : \mathcal{V}_{\mathrm{pf}} \to \mathcal{V}_{\mathbb{B}}$, thresholding at a chosen $\tau \in (0, 1]$.

The pair $(\beta, \theta)$ is *not* an adjoint pair in general; $\theta \circ \beta = \mathrm{id}$ but $\beta \circ \theta$ collapses real values to $\{0, 1\}$.

## 4. Functoriality of the language

The QVR language is *parametric* in $\mathcal{V}$: every syntactic construct interprets uniformly over all five quantales modulo the existence of joins, products, and units. Concretely, for any quantale $\mathcal{V}$ and base-change homomorphism $h : \mathcal{V} \to \mathcal{W}$, the diagram

$$
\begin{array}{c}
\text{QVR phrases} \\
\downarrow \llbracket \cdot \rrbracket_{\mathcal{V}} \\
\mathcal{V}\text{-}\mathbf{Rel}
\end{array}
\quad
\begin{array}{c}
\\
\\
\xrightarrow{\;\;h_*\;\;}
\end{array}
\quad
\begin{array}{c}
\text{QVR phrases} \\
\downarrow \llbracket \cdot \rrbracket_{\mathcal{W}} \\
\mathcal{W}\text{-}\mathbf{Rel}
\end{array}
$$

commutes for every well-typed phrase whose type does not involve `stochastic` or `continuous` declarations. (The stochastic and continuous strata are tied to the additive structure of $[0,1]$ as a $\sigma$-algebra of probabilities, not to the quantale; they ignore the `quantale` declaration.)

This commutation is the formal content of *base-change invariance*: changing the underlying truth-value algebra distributes over composition, tensor, marginalisation, and every other expression-level combinator.
