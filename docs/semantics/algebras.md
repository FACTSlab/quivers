# Algebras and Base Change

The active `Algebra` instance determines `tensor_op`, `join`, `meet`, `negate`, `unit`, and `zero` for discrete weighted morphisms. “Algebra” is the implementation's interface name. It should not be read as a claim that every built-in is a complete lattice or strict [quantale](https://ncatlab.org/nlab/show/quantale).

## 1. The eleven algebras

Each entry below records the operations implemented by the corresponding class. `join` is the reduction used by composition and marginalization; it is not necessarily an order-theoretic join, and `unit` is not necessarily a top element.

### 1.1 Product fuzzy algebra

$$
\mathcal{V}_{\mathrm{pf}} \;=\; \bigl([0, 1],\ \le,\ \cdot,\ 1\bigr),
\qquad a \otimes b = a \cdot b,
\qquad \bigoplus_i a_i = 1 - \prod_i (1 - a_i).
$$

The join is *noisy-OR*. This is the QVR default.

### 1.2 Boolean algebra

$$
\mathcal{V}_{\mathbb{B}} \;=\; \bigl(\{0, 1\},\ \le,\ \wedge,\ 1\bigr),
\qquad \bigoplus_i a_i = \bigvee_i a_i.
$$

The classical two-element lattice; $\mathcal{V}_{\mathbb{B}}\text{-}\mathbf{Rel}$ is the category of ordinary binary relations on finite sets.

### 1.3 Łukasiewicz algebra

$$
\mathcal{V}_{\mathrm{L}} \;=\; \bigl([0, 1],\ \le,\ \otimes_{\mathrm{L}},\ 1\bigr),
\qquad a \otimes_{\mathrm{L}} b = \max(0, a + b - 1),
\qquad \bigoplus_i a_i = \min(1, \textstyle\sum_i a_i).
$$

### 1.4 Gödel algebra

$$
\mathcal{V}_{\mathrm{G}} \;=\; \bigl([0, 1],\ \le,\ \min,\ 1\bigr),
\qquad \bigoplus_i a_i = \max_i a_i.
$$

### 1.5 Tropical algebra

$$
\mathcal{V}_{\mathrm{T}} \;=\; \bigl([0, +\infty],\ \ge,\ +,\ 0\bigr),
\qquad \bigoplus_i a_i = \min_i a_i.
$$

The order is reversed: smaller is "truer". The unit is $0$ (the additive identity), and the bottom of the lattice (the largest element) is $+\infty$. Composition is the *min-plus* matrix product, suitable for shortest-path semantics.

### 1.6 Max-plus (Viterbi) algebra

$$
\mathcal{V}_{\mathrm{MP}} \;=\; \bigl([-\infty, +\infty),\ \le,\ +,\ 0\bigr),
\qquad \bigoplus_i a_i = \max_i a_i.
$$

The max-plus / [Viterbi](https://en.wikipedia.org/wiki/Viterbi_algorithm) semiring used in best-path scoring; the bottom $\bot = -\infty$ is the additive zero, the unit $\mathbf{1} = 0$, and joins are pointwise max. Programs over log-probabilities live in the sub-poset $(-\infty, 0]$.

### 1.7 Log-prob algebra

$$
\mathcal{V}_{\mathrm{LP}} \;=\; \bigl([-\infty, +\infty),\ \le,\ +,\ 0\bigr),
\qquad \bigoplus_i a_i = \operatorname{logsumexp}_i a_i.
$$

The log-space analogue of the product-fuzzy / probability algebra. Carrier, unit, and bottom coincide with $\mathcal{V}_{\mathrm{MP}}$; the algebra differs by replacing $\max$ with $\operatorname{logsumexp}$, the smooth aggregation. Composition is numerically stable log-domain matrix multiplication. Programs over log-probabilities live in the sub-poset $(-\infty, 0]$.

### 1.8 Markov algebra

$$
\mathcal{V}_{\mathrm{M}} \;=\; \bigl(\mathbb{R}_{\ge 0},\ \le,\ \cdot,\ 1\bigr),
\qquad \bigoplus_i a_i = \sum_i a_i.
$$

The sum-product semiring underlying [stochastic-kernel composition](https://en.wikipedia.org/wiki/Stochastic_matrix): a $\mathcal{V}_{\mathrm{M}}$-relation is the per-entry tabulation of a row-stochastic matrix, and matrix multiplication under this algebra is Kleisli composition in $\mathbf{Stoch}$. Like $\mathcal{V}_{\mathbb{R}}$ this is a semiring rather than a bounded lattice; the row-stochasticity constraint lives in the morphism layer, not in the algebra itself.

### 1.9 Real algebra

$$
\mathcal{V}_{\mathbb{R}} \;=\; \bigl(\mathbb{R},\ \le,\ \cdot,\ 1\bigr),
\qquad \bigoplus_i a_i = \sum_i a_i.
$$

The sum-product semiring on the reals. No bottom/top: this is a semiring, not a bounded algebra, and is used for expectation-style aggregation where negative weights and unbounded magnitudes are required.

### 1.10 Probability algebra

$$
\mathcal{V}_{[0, 1]} \;=\; \bigl([0, 1],\ \le,\ \cdot,\ 1\bigr),
\qquad \bigoplus_i a_i = \min\!\bigl(1, \textstyle\sum_i a_i\bigr).
$$

Sum-product on $[0, 1]$ with explicit saturation at $1$ on aggregation. Distinguishes from $\mathcal{V}_{\mathrm{pf}}$ in its choice of join (saturated-sum rather than noisy-OR).

### 1.11 Counting algebra

$$
\mathcal{V}_{\mathbb{N}} \;=\; \bigl(\mathbb{N},\ +,\ \cdot,\ 1\bigr),
\qquad \bigoplus_i a_i = \sum_i a_i.
$$

Sum-product on the non-negative integers. Used for derivation-counting and unweighted multiplicity tracking. Negation is undefined.

## 2. Algebraic scope

Boolean and Gödel use genuine lattice joins; their finite tensor contractions have the familiar relational interpretation. Tropical and max-plus use idempotent semiring addition. Markov, Real, Counting, and LogProb use finite sum or log-sum-exp reductions and are best understood through finite semiring-style tensor contraction. ProductFuzzy, Łukasiewicz, and Probability use noisy or saturating reductions for which distributivity may fail.

This distinction matters. Associativity of matrix-style composition requires the relevant distributivity law, while compact-closed equations require still more structure. The Python class hierarchy admits all eleven at the `Algebra` gate, so callers should not infer those laws from `isinstance(value, Algebra)` alone.

### 2.1 A note on the product-fuzzy and Łukasiewicz pairs

The product-fuzzy and Łukasiewicz $(\otimes, \oplus)$ pairs use a t-norm and a separate t-conorm on $[0,1]$. Those particular pairs are not strict quantales: the distributivity law

$$
a \otimes \bigoplus_{i} b_i \;=\; \bigoplus_i (a \otimes b_i)
$$

fails in general for these two pairs.

For instance, in the product-fuzzy pair with $a = b_1 = b_2 = 1/2$:

$$
a \otimes (b_1 \oplus b_2) \;=\; \tfrac{1}{2}\bigl(1 - \tfrac{1}{4}\bigr) \;=\; \tfrac{3}{8},
\qquad
(a \otimes b_1) \oplus (a \otimes b_2) \;=\; 1 - \tfrac{9}{16} \;=\; \tfrac{7}{16}.
$$

Finite distributivity holds for the Boolean, Gödel, tropical, max-plus, log-probability, Markov, Real, and Counting operations. This makes their finite contractions semiring-like, but it does not by itself supply completeness or arbitrary joins. ProductFuzzy and Łukasiewicz fail distributivity even on small finite examples. Their compositions are still defined computationally; equations that require distributivity simply do not follow.

## 3. Base change

`AlgebraHomomorphism` is the implementation's name for a shape-preserving entry map with declared source and target algebras. The registry includes lossy maps such as thresholding, clamping, flooring, and a log transform. These maps are not all strict monoid-and-join homomorphisms.

When a map preserves the operations required by a calculation, pointwise application induces the usual base-change functor

$$
h_* : \mathcal{V}\text{-}\mathbf{Rel} \to \mathcal{W}\text{-}\mathbf{Rel},
\qquad (h_* r)(x, y) = h(r(x, y)),
$$

which acts as the identity on shapes. For the lossy registry entries, `.change_base` still performs the documented tensor conversion, but functoriality must not be assumed without checking the relevant preservation laws.

The implementation ships a registry of named homomorphisms, including:

- $\beta : \mathcal{V}_{\mathbb{B}} \to \mathcal{V}_{\mathrm{pf}}$, the inclusion $\{0, 1\} \hookrightarrow [0, 1]$ (`Embedding`);
- $\theta : \mathcal{V}_{\mathrm{pf}} \to \mathcal{V}_{\mathbb{B}}$, thresholding at $\tau \in (0, 1]$ (`Threshold`);
- $\mathcal{V}_{\mathrm{pf}} \to \mathcal{V}_{\mathrm{G}}$ by clamping each entry to $[0,1]$ (`MaterialImplication`); the name refers to the target composition convention, not an entrywise binary conditional;
- $\mathcal{V}_{\mathrm{M}} \to \mathcal{V}_{\mathrm{pf}}$ by clamping entries to $[0,1]$ (`Expectation`);
- $\mathcal{V}_{\mathrm{pf}} \to \mathcal{V}_{\mathrm{LP}}$ via $a \mapsto \log a$ (`LogProb`);
- $\mathcal{V}_{\mathrm{pf}} \to \mathcal{V}_{\mathrm{MP}}$ via $a \mapsto \log a$ (`MaxPlus`); the per-entry map matches `LogProb` but the target join is $\max$ rather than $\operatorname{logsumexp}$, realizing Viterbi-MAP aggregation;
- $\mathcal{V}_{\mathbb{R}} \rightleftarrows \mathcal{V}_{[0, 1]}$ (`ProbabilityClamp` / `ProbabilityToReal`);
- $\mathcal{V}_{\mathbb{R}} \rightleftarrows \mathcal{V}_{\mathbb{N}}$ (`CountingFromReal` / `CountingToReal`).

`lookup_homomorphism(src, tgt)` retrieves a registered map. The threshold registry entry uses $\tau=0.5$ and tests `value > tau`; the reverse Boolean embedding preserves 0 and 1. Most other entries are intentionally lossy conversions.

## 4. Functoriality of the language

Discrete morphism operations dispatch through the active algebra's methods. This is implementation-level polymorphism, not a theorem that every construct is invariant under every registered conversion. For a strict homomorphism $h : \mathcal{V} \to \mathcal{W}$, one may ask whether the diagram

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

commutes for the operations that $h$ preserves. It need not commute for thresholding, clamping, flooring, ProductFuzzy-to-LogProb, or ProductFuzzy-to-MaxPlus. The stochastic and continuous strata use probability distributions directly and do not obtain their semantics from the discrete algebra registry.
