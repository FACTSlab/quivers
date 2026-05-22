# Denotational Semantics of Compositional Effects

QVR layers monadic / applicative effects over the residuated category
universe through two parallel typeclass towers (monads and arrows)
plus a class-driven schema-lifting machinery. This page assigns
formal denotations to the surface constructs and proves the layering
is a conservative extension of the bare deduction fragment of
[Weighted Deduction Fragment](grammar.md).

## 1. Setting

We work in the V-enriched semantic universe of
[Setting and Notation](setting.md). For an effect-typed parser we fix:

- A residuated category universe $\mathcal{C}$ realized as a finite
  set $\llbracket \mathrm{Cat} \rrbracket$. The carrier is built by
  the `FreeResiduated(generators, depth, ops)` object class: starting
  from a finite alphabet of atomic categories, it freely closes the
  carrier under the constructors `ops` (which include the residuation
  slashes $/$ and $\backslash$, the monoidal product $\otimes$, and
  the diamond modality $\diamondsuit$) up to a finite `depth` bound,
  yielding a finite-set object whose elements are the well-formed
  category expressions of bounded complexity.
- A finite list of declared effects $T_1, \dots, T_n$ each registered
  against one or more of the typeclass ABCs in
  $\mathbf{Functor}, \mathbf{Applicative}, \mathbf{Monad},
  \mathbf{Alternative}, \mathbf{MonadPlus}, \mathbf{Traversable},
  \mathbf{ArrowApply}, \mathbf{ArrowLoop}$.

The *effect-typed category universe* is

$$
\widehat{\mathcal{C}} \;=\; \mathcal{C} \times \mathrm{EffectStack}_{\le d}(T_1, \dots, T_n),
$$

where $\mathrm{EffectStack}_{\le d}$ is the set of finite stacks of
effects of length at most $d$ (the `effect_depth` parameter of
`chart_fold`). Each chart cell at span $(i, j)$ is indexed by an
element of $\widehat{\mathcal{C}}$.

## 2. Typeclass denotations

Each typeclass $\mathfrak{C}$ in
[`quivers.monadic.typeclasses`](../api/index.md#monadic-package) is
modeled by a panproto theory $\mathrm{Th}\mathfrak{C}$ in
[`quivers.monadic.theories`](../api/index.md#monadic-package); class
extension corresponds to theory inclusion via
$\mathrm{colimit}$. The sorts and operations of each theory are:

| $\mathrm{Th}\mathfrak{C}$ | Sorts | Operations | Equations |
|---------------------------|-------|------------|-----------|
| $\mathrm{ThFunctor}$ | $\mathrm{Carrier}$, $\mathrm{Hom}$ | $\mathrm{fmap\_obj}$, $\mathrm{fmap}$ | identity, composition |
| $\mathrm{ThApplicative}$ | (extends Functor) | $\mathrm{pure}$, $\mathrm{apply}$ | identity, homomorphism, interchange, composition |
| $\mathrm{ThMonad}$ | (extends Applicative) | $\mathrm{join}$ | left/right unit, associativity |
| $\mathrm{ThAlternative}$ | (extends Applicative) | $\mathrm{empty}$, $\mathrm{alt}$ | identity, associativity |
| $\mathrm{ThMonadPlus}$ | (extends Monad + Alternative) | (none new) | left zero |
| $\mathrm{ThTraversable}$ | (extends Functor + Foldable) | $\mathrm{traverse}$ | naturality, identity, composition |

An effect instance $T$ that registers against $\mathfrak{C}$ corresponds
to a panproto theory morphism $\mathrm{Th}\mathfrak{C} \to \mathrm{ThQVR}$
whose vertex map sends $\mathrm{Carrier}$ to the runtime carrier sort
of the residuated universe and whose operation map realizes the
class operations as morphisms of $\mathcal{V}\text{-}\mathbf{Rel}$ /
$\mathbf{Stoch}$ / $\mathbf{Kern}$.

## 3. Effect-typed schema denotations

A `SchemaDecl` whose domain or codomain mentions an `ObjectEffectApply`
node $T(X)$ is *parametric* in its free pattern variables. For each
choice of instantiation $X, Y \in \mathcal{C}$ the schema denotes a
morphism

$$
\llbracket r \rrbracket_{X, Y}
\;:\; T(X/Y) \otimes T(Y) \;\to\; T(X)
\quad \text{in } \widehat{\mathcal{C}};
$$

the family $\bigl(\llbracket r \rrbracket_{X, Y}\bigr)_{X, Y}$ is a
natural transformation of bifunctors (an applicative-style lifted
application). The naturality square is the applicative
*composition law*; it holds by construction when $T$ inhabits
$\mathbf{Applicative}$.

For Monad-typed schemas the denotation is the *Kleisli composition*:

$$
\llbracket \mathrm{bind}_T \rrbracket_{X, Y}
\;:\; T(X) \otimes (X \backslash T(Y)) \;\to\; T(Y),
$$

a morphism in the Kleisli category $\mathrm{Kl}(T)$ realising the
canonical bind. The two presentations agree via the
[`kleisli` / `arrow_monad` bridges](../api/index.md#monadic-package).

## 4. Joint type-and-effect dispatch

A chart parser over a residuated universe $\mathcal{C}$ with declared
effects $\bar T = (T_1, \dots, T_n)$ has cells

$$
\mathrm{Chart}[i, j] \;\in\; \mathcal{V}^{|\widehat{\mathcal{C}}|}
$$

: a $\mathcal{V}$-valued vector indexed by effect-typed categories.
Schema firings populate cells according to four rules:

1. **Base firing.** A schema $r : \pi_1 \otimes \pi_2 \to \pi$ fires
   between cells $\mathrm{Chart}[i, k]$ and $\mathrm{Chart}[k, j]$ at
   the type coordinate when the effect-stack coordinates of both
   cells are empty (or when the schema has been auto-lifted to admit
   the cells' effect prefix; see rule 2).
2. **Lift firing.** When child cells carry effect prefixes
   $\bar{S}_1, \bar{S}_2$ that the schema's domain doesn't, the
   class-driven lifts of
   [`class_directed_lifts`](../api/index.md#stochastic-package) emit a
   schema whose domain wraps the base premises in the appropriate
   effect prefixes. The denotation of this lifted firing is exactly
   the natural-transformation composition of [§3](#3-effect-typed-schema-denotations)
   above.
3. **Handler firing.** When a cell holds a $T(\alpha)$ distribution
   and a $\mathrm{Handler}_{T \to S}$ is registered (a panproto theory
   morphism $\mathrm{Th}T \to \mathrm{Th}S$), applying the handler
   produces a fresh cell at $S(\alpha)$. In the special case
   $S = \mathrm{Identity}$, the handler eliminates the effect entirely.
4. **Commutation firing.** When a $\mathrm{DistributiveLaw}(T, U)$ is
   registered, the $\mathrm{swap}_{T,U}$ schema realizes the natural
   transformation $T \circ U \Rightarrow U \circ T$ on cells whose
   effect-stack ends with $T \cdot U$.

The four firings compose freely; the chart's CKY enumeration computes
the inside score over the joint search space.

## 5. Adequacy of the lifting machinery

**Theorem (Lifting Adequacy).** *Let $r$ be a base SchemaDecl and
$T$ an effect instance registered against the typeclasses
$\mathfrak{C}_1, \dots, \mathfrak{C}_k$. Then for every cell
configuration over $\widehat{\mathcal{C}}$ that admits a derivation
under the lifted schemas
$\mathrm{class\_directed\_lifts}(r, T)$, the chart parser's
inside score equals the categorical denotation of the composite
natural transformation through $\mathrm{Th}\mathfrak{C}_1, \dots,
\mathrm{Th}\mathfrak{C}_k$ in the appropriate enriched category.*

*Proof sketch.* By induction on the typeclass tower:

- For $\mathfrak{C} = \mathbf{Applicative}$, the lifted schema is the
  applicative *composition law*; the chart firing realizes the
  natural transformation
  $T(\pi_1) \otimes T(\pi_2) \Rightarrow T(\pi)$ obtained by
  postcomposing the base $r$ with $T$'s `apply`. The applicative
  laws (identity, homomorphism, interchange, composition) imply
  that successive applicative lifts compose to the same denotation
  irrespective of bracketing; the chart's left-associative scheduling
  is therefore one valid realization among many equivalent choices.
- For $\mathfrak{C} = \mathbf{Monad}$, the bind-lift is the Kleisli
  composition of the base schema; the monad laws imply
  $\mathrm{join} \circ \mathrm{fmap}(\mathrm{join}) = \mathrm{join} \circ \mathrm{join}$,
  so iterated bind-lifts compose to the iterated Kleisli of the base
  rule.
- For $\mathfrak{C} = \mathbf{Alternative}$, the alt-lift realizes
  $T \otimes T \Rightarrow T$ via $T$'s `alt`; the alternative laws
  give the equational theory of the resulting non-deterministic
  derivation tree.

The chart algorithm (CKY) is the standard inside-score recurrence and
is sound for arbitrary $\mathcal{V}$-algebra-valued semirings (see
[Weighted Deduction Fragment §7](grammar.md#7-strategy-independence)).
$\square$

## 6. Free-monad denotation

For an effect signature
$\Sigma = \{\mathrm{op}_1 : P_1 \to R_1, \dots, \mathrm{op}_m : P_m \to R_m\}$,
the free monad $\mathrm{FreeMonad}(\Sigma)$ has the denotation of the
*finitely-branched signature trees* over $\Sigma$:

$$
\llbracket \mathrm{FreeMonad}(\Sigma)(A) \rrbracket
\;=\; \coprod_{n \ge 0} \mathrm{Tree}_{\Sigma, n}(A),
$$

where $\mathrm{Tree}_{\Sigma, n}(A)$ is the set of $\Sigma$-shaped
trees of depth $\le n$ whose leaves are values in $A$. The runtime
realization truncates at the `depth` parameter of the enclosing
`deduction { … }` block to keep the carrier finite.

A handler $H : \Sigma \to S$ is a panproto theory morphism
$\mathrm{Th}\Sigma \to \mathrm{Th}S$; the denotation of
$H.\mathrm{run} : \mathrm{FreeMonad}(\Sigma)(A) \to S(A)$ is the
universal map of free monads (the *fold* over the operation tree
that interprets each internal node through the corresponding
clause of $H$).

## 7. Bridges and conservativity

The `kleisli` and `arrow_monad` bridges of
[`quivers.monadic.bridges`](../api/index.md#monadic-package) realize
the canonical theory morphisms:

$$
\mathrm{kleisli} \;:\; \mathrm{ThMonad} \;\to\; \mathrm{ThArrowApply},
\qquad
\mathrm{arrow\_monad} \;:\; \mathrm{ThArrowApply} \;\to\; \mathrm{ThMonad}
$$

with $\mathrm{arrow\_monad} \circ \mathrm{kleisli} \;\cong\; \mathrm{id}_{\mathrm{ThMonad}}$
and $\mathrm{kleisli} \circ \mathrm{arrow\_monad} \;\cong\; \mathrm{id}_{\mathrm{ThArrowApply}}$
naturally isomorphic (not equal) via the canonical
$1 \otimes A \cong A$ unitor that absorbs the
``ArrowMonad(a)(A) = a(1, A)`` boxing
([Hughes 2000](https://doi.org/10.1016/S0167-6423(99)00023-4), Theorem 3.1). Consequently the
arrow-side and monad-side presentations of any effect agree on
denotation up to the canonical isomorphism. When the underlying
monad is also a *monad-fix*, admitting a least-fixed-point
operator, the `chart_fold` runtime's `loop_arr` construction agrees
denotationally with the Kleisli-iteration of the binary step; for
monads without `mfix`, the arrow-side trace is the well-defined
realization and the Kleisli side is undefined.

**Conservativity.** When the declared effects list $\bar T$ is empty,
the joint type-and-effect dispatch of [§4](#4-joint-type-and-effect-dispatch)
collapses to the classical residuated-grammar dispatch of
[Weighted Deduction Fragment §6](grammar.md#6-chart-denotation), and
the inside score is the standard Lambek / CCG / multimodal-TLG
chart-parser denotation. The effects framework is therefore a strict
extension over the bare grammar fragment.

## References

- Bumford, D. and Charlow, S. (2026). [*Effect-Driven Interpretation: Functors for Natural Language Composition*](https://www.cambridge.org/core/elements/abs/effectdriven-interpretation/56671E539160AAA1DACF8555B82A2FE4). Cambridge Elements in Semantics. Cambridge University Press. Online ISBN 9781009285377; preprint [arXiv:2504.00316](https://arxiv.org/abs/2504.00316), draft at [simoncharlow.com/papers/cup-effects.pdf](https://simoncharlow.com/papers/cup-effects.pdf).
- Hughes, J. (2000). [*Generalizing monads to arrows*](https://doi.org/10.1016/S0167-6423(99)00023-4). Science of Computer Programming, 37(1–3), 67–111.
- Plotkin, G. and Power, J. (2003). [*Algebraic operations and generic effects*](https://doi.org/10.1023/A:1023064908962). Applied Categorical Structures, 11(1), 69–94.
- Bauer, A. and Pretnar, M. (2015). [*Programming with algebraic effects and handlers*](https://doi.org/10.1016/j.jlamp.2014.02.001). Journal of Logical and Algebraic Methods in Programming, 84(1), 108–123.
