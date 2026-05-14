# Quantales and Base Change

The choice of quantale $\mathcal{V}$ in a `quantale` declaration determines the truth-value semantics of every $\mathcal{V}$-relation in the program. This page records the formal definition of each built-in quantale and the base-change functors that mediate between them.

## 1. The eleven quantales

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

### 1.6 Max-plus (Viterbi) quantale

$$
\mathcal{V}_{\mathrm{MP}} \;=\; \bigl([-\infty, 0],\ \le,\ +,\ 0\bigr),
\qquad \bigoplus_i a_i = \max_i a_i.
$$

The max-plus / Viterbi semiring used in best-path scoring; arcweight's `MaxWeight` and the standard semiring underlying Viterbi decoding.

### 1.7 Log-prob quantale

$$
\mathcal{V}_{\mathrm{LP}} \;=\; \bigl([-\infty, 0],\ \le,\ +,\ 0\bigr),
\qquad \bigoplus_i a_i = \operatorname{logsumexp}_i a_i.
$$

The log-space analogue of the product-fuzzy / probability quantale; mirrors arcweight's `LogWeight`. Composition is numerically stable log-domain matrix multiplication.

### 1.8 Markov quantale

$$
\mathcal{V}_{\mathrm{M}} \;=\; \bigl(\Delta,\ \le,\ \star,\ \delta\bigr),
$$

with carrier the row-stochastic kernels $X \to \mathcal{P}(Y)$, tensor $\star$ the Markov-kernel composition, and unit the Dirac kernel $\delta$. Powers the `stochastic` declaration's V-Cat surface.

### 1.9 Real quantale

$$
\mathcal{V}_{\mathbb{R}} \;=\; \bigl(\mathbb{R},\ +,\ \cdot,\ 1\bigr),
\qquad \bigoplus_i a_i = \sum_i a_i.
$$

The sum-product semiring on the reals; mirrors arcweight's `RealWeight`. No bottom/top: this is a semiring, not a bounded quantale, and is used for expectation-style aggregation where negative weights and unbounded magnitudes are required.

### 1.10 Probability quantale

$$
\mathcal{V}_{[0, 1]} \;=\; \bigl([0, 1],\ \le,\ \cdot,\ 1\bigr),
\qquad \bigoplus_i a_i = \min\!\bigl(1, \textstyle\sum_i a_i\bigr).
$$

Sum-product on $[0, 1]$ with explicit saturation at $1$ on aggregation; mirrors arcweight's `ProbabilityWeight`. Distinguishes from $\mathcal{V}_{\mathrm{pf}}$ in its choice of join (saturated-sum rather than noisy-OR).

### 1.11 Counting quantale

$$
\mathcal{V}_{\mathbb{N}} \;=\; \bigl(\mathbb{N},\ +,\ \cdot,\ 1\bigr),
\qquad \bigoplus_i a_i = \sum_i a_i.
$$

Sum-product on the non-negative integers; mirrors arcweight's `IntegerWeight`. Used for derivation-counting and unweighted multiplicity tracking. Negation is undefined.

## 2. Order- and structure-preservation

For each quantale we record which elementary properties hold; these determine which categorical constructions transport across base change.

| Quantale | Idempotent? | Integral? | Cancellative? |
|----------|-------------|-----------|---------------|
| $\mathcal{V}_{\mathrm{pf}}$ | No | Yes ($\mathbf{1} = 1$) | No |
| $\mathcal{V}_{\mathbb{B}}$ | Yes | Yes | No |
| $\mathcal{V}_{\mathrm{L}}$ | No | Yes | No |
| $\mathcal{V}_{\mathrm{G}}$ | Yes | Yes | No |
| $\mathcal{V}_{\mathrm{T}}$ | No | Yes ($\mathbf{1} = 0$) | Yes |
| $\mathcal{V}_{\mathrm{MP}}$ | Yes (join) | Yes ($\mathbf{1} = 0$) | Yes |
| $\mathcal{V}_{\mathrm{LP}}$ | No | Yes ($\mathbf{1} = 0$) | Yes |
| $\mathcal{V}_{\mathrm{M}}$ | No | Yes ($\mathbf{1} = \delta$) | No |
| $\mathcal{V}_{\mathbb{R}}$ | No | Yes ($\mathbf{1} = 1$) | Yes |
| $\mathcal{V}_{[0,1]}$ | No | Yes ($\mathbf{1} = 1$) | No |
| $\mathcal{V}_{\mathbb{N}}$ | No | Yes ($\mathbf{1} = 1$) | Yes |

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

The implementation ships a registry of named homomorphisms, including:

- $\beta : \mathcal{V}_{\mathbb{B}} \to \mathcal{V}_{\mathrm{pf}}$, the inclusion $\{0, 1\} \hookrightarrow [0, 1]$ (`Embedding`);
- $\theta : \mathcal{V}_{\mathrm{pf}} \to \mathcal{V}_{\mathbb{B}}$, thresholding at $\tau \in (0, 1]$ (`Threshold`);
- $a \mapsto a \cdot (1 - a)$-style implication and `MaterialImplication` between $\mathcal{V}_{\mathrm{pf}}$ and $\mathcal{V}_{\mathrm{L}}$;
- $\mathcal{V}_{\mathrm{M}} \to \mathcal{V}_{\mathrm{pf}}$ (`Expectation`) and $\mathcal{V}_{\mathrm{pf}} \to \mathcal{V}_{\mathrm{LP}}$ (`LogProb`);
- $\mathcal{V}_{\mathrm{LP}} \to \mathcal{V}_{\mathrm{MP}}$ collapsing log-sum-exp to max (`MaxPlus`);
- $\mathcal{V}_{\mathbb{R}} \rightleftarrows \mathcal{V}_{[0, 1]}$ (`ProbabilityClamp` / `ProbabilityToReal`);
- $\mathcal{V}_{\mathbb{R}} \rightleftarrows \mathcal{V}_{\mathbb{N}}$ (`CountingFromReal` / `CountingToReal`).

`lookup_homomorphism(src, tgt)` retrieves a registered map; programs select one via the `change-of-base` block. Only $\beta \circ \theta = \mathrm{id}_{\mathcal{V}_\mathbb{B}}$ and the sub-quantale inclusions are sections; the rest are lossy lax monoidal poset functors.

## 4. Functoriality of the language

The QVR language is *parametric* in $\mathcal{V}$: every syntactic construct interprets uniformly over all eleven quantales modulo the existence of joins, products, and units. Concretely, for any quantale $\mathcal{V}$ and base-change homomorphism $h : \mathcal{V} \to \mathcal{W}$, the diagram

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

This commutation is the formal content of *base-change invariance*: changing the underlying truth-value algebra distributes over composition, tensor, marginalization, and every other expression-level combinator.
