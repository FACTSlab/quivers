# Grammar Fragment

The QVR grammar fragment introduces three syntactic categories — `category`, `rule`, and the `cat_pattern` sub-language — together with the parser combinators of [Expressions §4](expressions.md#4-parser-combinators). Together they describe a categorial-grammar-style chart parser whose denotations sit in the Kleisli category $\mathbf{Stoch}$ (or $\mathbf{Kern}$, when continuous score weights are used).

## 1. Category atoms

A declaration `category A, B, C` introduces atoms $A, B, C \in \mathrm{Atom}$. Their denotation is purely *syntactic*: each atom is a distinct generator of the free monoidal-residuated category that follows.

## 2. Category patterns

The grammar of patterns

$$
\pi \;::=\; A \;\big|\; \pi_1 / \pi_2 \;\big|\; \pi_1 \backslash \pi_2 \;\big|\; \pi_1 \star \pi_2 \;\big|\; \mathbf{1}
$$

generates the free *bi-residuated monoidal category* $\mathcal{C}_{\mathrm{Atom}}$ on the atoms. Concretely, $\mathcal{C}_{\mathrm{Atom}}$ has:

- objects: equivalence classes of patterns under the residuated equations,
- morphisms: derivations in the appropriate sequent calculus (Lambek calculus, with optional non-associativity for multimodal extensions).

The two slashes are the residuals of $\star$ on the right and left, respectively, with the standard adjunctions

$$
\pi_1 \star \pi_2 \to \pi_3 \quad \Longleftrightarrow \quad \pi_1 \to \pi_3 / \pi_2 \quad \Longleftrightarrow \quad \pi_2 \to \pi_1 \backslash \pi_3.
$$

These adjunctions, together with the symmetric monoidal structure of $\star$, axiomatise $\mathcal{C}_{\mathrm{Atom}}$ as the free symmetric residuated monoidal category on $\mathrm{Atom}$.

## 3. Rules

A rule declaration

```
rule r(X₁, …, Xₖ) : π₁, …, πₘ => π
```

introduces a sequent-calculus axiom

$$
\frac{\quad \pi_1[\bar X / \bar\pi]\quad \cdots \quad \pi_m[\bar X / \bar\pi]\quad}{\pi[\bar X / \bar\pi]} \quad (r)
$$

universally quantified over substitutions $[\bar X / \bar\pi]$ for the rule's parameter list. Rules of arity 1 (one premise) and arity 2 (two premises) cover all built-in cases.

Each rule denotes a natural transformation

$$
r \,:\, \mathrm{Hom}(\pi_1) \otimes \cdots \otimes \mathrm{Hom}(\pi_m) \to \mathrm{Hom}(\pi)
$$

between hom-functors of $\mathcal{C}_{\mathrm{Atom}}$, weighted by an optional log-score $\beta_r \in \mathbb{R}$ (zero by default). The collection of all declared rules forms a *rule system* $\Sigma$, formally a functor $\Sigma : \mathrm{Atom}^{\mathrm{op}} \times \mathrm{Atom} \to \mathbf{Vect}_{\mathbb{R}}$ assigning to each pair $(\pi, \pi')$ the $\mathbb{R}$-vector space spanned by derivations $\pi \vdash \pi'$.

The DSL realises $\Sigma$ as an instance of [`quivers.stochastic._rule_system.RuleSystem`](../api/stochastic/rules.md), whose `binary_rules` and `unary_rules` fields enumerate the rule schemas and whose `binary_weights` / `unary_weights` carry the (initial) log-scores $\beta_r$.

## 4. The chart-parser denotation

Fix a rule system $\Sigma$, an alphabet $\mathrm{Token}$ of categories assigned to lexical items by the discrete morphism `lex : Token -> Atom`, a start symbol $s \in \mathrm{Atom}$, and a depth bound $d$.

For an input $w = (w_1, \dots, w_n) \in \mathrm{Token}^n$, the *chart parser* assigns to each contiguous substring $w_{i:j}$ and each pattern $\pi$ the *inside score*

$$
\alpha[i, j, \pi] \;=\; \bigoplus_{\text{deriv}\ d : \pi_1, \dots, \pi_k \vdash \pi}
\bigotimes_{\ell=1}^{k} \alpha[i_{\ell}, j_{\ell}, \pi_{\ell}],
$$

the join over all $\Sigma$-derivations of $\pi$ that span $w_{i:j}$, of the product of the children's inside scores. The base case is the lexical insertion $\alpha[i, i+1, \mathrm{lex}(w_i)] = \mathbf{1}$, all other base cases $\bot$.

The denotation of the parser combinator is the inside score of the start symbol:

$$
\llbracket \mathsf{parser}(\Sigma, s, d) \rrbracket(w) \;=\; \alpha[0, |w|, s].
$$

When the underlying $\bigoplus$ is the noisy-OR of $\mathcal{V}_{\mathrm{pf}}$ this is a soft-membership probability. When it is the additive structure of $[0, 1]$ (with row-stochastic rule weights), it is exactly the inside-algorithm probability $P(w \mid \Sigma)$ of a probabilistic context-free / categorial grammar. When it is the Boolean quantale, it is the membership predicate $w \in L(\Sigma, s)$.

## 5. Soundness of the chart algorithm

The chart algorithm computes $\alpha[i, j, \pi]$ by dynamic programming over the lattice of substrings, using the algebraic identity

$$
\bigoplus_{\substack{\pi_1, \pi_2 \\ k \in (i, j)}}
\beta_r \otimes \alpha[i, k, \pi_1] \otimes \alpha[k, j, \pi_2]
\;=\;
\alpha[i, j, \pi]
\quad \text{for each binary rule } r : \pi_1, \pi_2 \vdash \pi.
$$

This is an instance of *Tarski–Knaster* fixed-point computation in the lattice of pattern-indexed quantale-valued functions; the chart algorithm is the standard bottom-up enumeration. Soundness — i.e.\ the chart's value equals the inside score of [§4](#4-the-chart-parser-denotation) — follows from the distributivity of $\otimes$ over $\bigoplus$ in the underlying quantale.

## 6. Special cases

The combinators $\mathsf{ccg}$ and $\mathsf{lambek}$ fix the rule system $\Sigma$:

| Combinator | Rule system $\Sigma$ |
|------------|----------------------|
| $\mathsf{ccg}$ | Forward and backward application; type-raising; harmonic forward composition |
| $\mathsf{lambek}$ | Application + left/right introduction; no type-raising |
| $\mathsf{parser}$ | User-supplied $\Sigma$ via the `rules` argument |

Their denotations are special cases of [§4](#4-the-chart-parser-denotation).
