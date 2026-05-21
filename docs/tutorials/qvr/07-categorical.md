# 7. Under the hood: the categorical surface

You can use QVR productively without category theory; the previous six chapters demonstrate that. But the library was built on a specific categorical foundation, and at some point you'll see a type error like `Source algebra 'ProductFuzzyAlgebra' does not match this morphism's algebra 'Real'` or a method called `change_base` and want to know what's going on. This chapter is a tour of the categorical machinery, oriented towards the four constructs that surface in the type system: *objects*, *morphisms*, *algebras*, and *change-of-base*.

This is optional reading. Skip it if the DSL is doing what you need.

## Why have types and algebras at all?

A typed signature like `program f : A -> B` exists so the compiler can reject programs that don't make sense before they hit a tensor. Two examples:

```text
# A program declared with effects=[Pure], but the body contains `observe`:
program p : Item -> Item [effects=[Pure]]
    observe y : Item <- Normal(0, 1)
    return y
# CompileError: line 2, col 4: program declared effects=[Pure]
#               but uses Score effect on `observe y`
```

```text
# An algebra mismatch under typed composition:
composition product_fuzzy as algebra
let pipeline = f *> g    # *> demands Markov, but module is product_fuzzy
# CompileError: typed composition *> requires both operands
#               in algebra Markov; got ProductFuzzyAlgebra
```

Without the effect and algebra tags, the same programs would either run and silently produce nonsense, or fail somewhere deep in a tensor evaluation with a stack trace pointing at the wrong place. The category-theoretic reading below is *why* those tags compose cleanly; the practical payoff is the error messages.

## The setting in one paragraph

A QVR program denotes a morphism in a [$\mathcal{V}$-enriched category](https://ncatlab.org/nlab/show/enriched+category) in the sense of [Kelly (1982)](http://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html). *Enriched* means the hom-sets aren't sets, they're objects of a fixed algebra $\mathcal{V}$ called the *enrichment* or *base*. For QVR's category $\mathcal{V}\text{-}\mathbf{Rel}$ of $\mathcal{V}$-valued relations on finite sets, a morphism `f : A -> B` is concretely a tensor of shape `(|A|, |B|)` whose entries live in $\mathcal{V}$. The choice of $\mathcal{V}$ determines what those entries mean: probabilities, log-probabilities, real numbers, fuzzy truth values, Booleans, etc. Composition `f >> g` is parameterized by $\mathcal{V}$ via two operations: a binary tensor product (`a ⊗ b`) that combines entries pointwise, and a join (`⋁_i x_i`) that aggregates over the shared dimension. Different choices of $(\otimes, \bigvee)$ give different composition semantics.

## Algebras: the enrichment algebra

A [algebra](https://ncatlab.org/nlab/show/algebra) is a complete lattice equipped with a monoidal product distributing over arbitrary joins. The shipped algebras:

| Name | $\otimes$ | $\bigvee$ | Identity | Reading |
|---|---|---|---|---|
| `Boolean` | AND | OR | true | Classical relations. |
| `ProductFuzzyAlgebra` | $a \cdot b$ | $1 - \prod(1-x_i)$ | 1 | Fuzzy "probabilistic AND / noisy-OR". |
| `Lukasiewicz` | $\max(0, a+b-1)$ | $\min(1, \sum x_i)$ | 1 | Probabilistic-sum t-conorm. |
| `Godel` | $\min(a, b)$ | $\max_i x_i$ | 1 | Min-max fuzzy logic. |
| `Tropical` | $a + b$ | $\min_i x_i$ | 0 | Shortest-path cost algebra. |
| `MaxPlus` | $a + b$ | $\max_i x_i$ | 0 | Viterbi / best-path cost algebra. |
| `LogProb` | $a + b$ | $\mathrm{logsumexp}_i$ | 0 | Numerically stable probability. |
| `Markov` | $a \cdot b$ | $\sum_i x_i$ | 1 | Markov-kernel composition (with row-stochasticity). |
| `Real` | $a \cdot b$ | $\sum_i x_i$ | 1 | Real sum-product. |
| `Probability` | $a \cdot b$ | $\sum_i x_i$ (clamped to [0,1]) | 1 | Bounded sum-product. |
| `Counting` | $a \cdot b$ | $\sum_i x_i$ | 1 | Nonnegative-integer counting. |

`algebra product_fuzzy` at the top of a `.qvr` file sets the enrichment for the module. Every `f >> g` composition uses the corresponding $(\otimes, \bigvee)$.

The hierarchy in `quivers.core.algebras` is:

```
CompositionRule
    ├── BilinearForm     (no associativity promise)
    └── Semigroupoid     (associative ⊗, no identity)
            └── Algebra  (associative ⊗ with identity, plus meet/negate)
```

Operations like `identity(A)`, `cup(A)`, `cap(A)`, `f.dagger`, `f.trace(A)` need the identity element and the compact-closed structure: they live on `Algebra`. If your module declares `semigroupoid material_impl` (Reichenbach-style implication composition, which is associative but lacks an identity), the compiler rejects `identity(A)` with a typed error pointing at the rule's level. Chapter 4 of the Python API track has the full story.

## Composition operators carry the enrichment

The shipped composition operators split into two groups. `>>` and `>=>` defer to the operands' own algebra (the module-level `algebra` declaration); the typed variants pin a specific enrichment and reject operands carrying a different one:

| Operator | Algebra |
|---|---|
| `>>` | operands' shared algebra |
| `>=>` | [Kleisli composition](https://en.wikipedia.org/wiki/Kleisli_category) in operands' shared algebra |
| `*>` | Markov (sum-product, kernel composition) |
| `~>` | LogProb (numerically stable in log-space) |
| `||>` | Gödel (min/max) |
| `?>` | Viterbi (max-plus) |
| `&&>` | Boolean (AND/OR) |
| `+>` | Łukasiewicz |
| `$>` | Real (sum-product) |
| `%>` | Probability |
| `<<` | Reverse of `>>` |

So `f >> g` composes in whatever enrichment both operands carry, and errors on a mismatch; the typed operators (`*>`, `~>`, ...) require both operands to already live in the target algebra and never auto-base-change. To cross enrichments, you [`change_base`](../../api/core/morphisms.md) between segments.

## Change of base

A *algebra homomorphism* $\varphi : \mathcal{V} \to \mathcal{W}$ is a lax monoidal lattice functor: it preserves the order, is lax with respect to $\otimes$ and $\bigvee$, and takes $\mathcal{V}$'s unit to $\mathcal{W}$'s. Functorially it extends to a base-change functor $\mathcal{V}\text{-}\mathbf{Rel} \to \mathcal{W}\text{-}\mathbf{Rel}$ that takes an $\mathcal{V}$-valued morphism to a $\mathcal{W}$-valued morphism by applying $\varphi$ entrywise. `f.change_base(phi)` is the surface for this.

The DSL exposes a catalog of named homomorphisms (`expectation`, `log_prob`, `max_plus`, `material_implication`, `threshold`, `boolean_embedding`, ...) and a small set of *constructors* parameterized by an object or morphism (`softmax(B)`, `l1_normalize(B)`, `l2_normalize(B)`, `bayes_invert(prior)`). Each of these is a first-class transformation value: you can let-bind them, compose them with `>>>`, pass them through `change_base`. The Python API track chapter 6 walks through the full surface; here's the short version:

```qvr
composition product_fuzzy as algebra
object A : FinSet 3
object B : FinSet 4
morphism f : A -> B [role=latent]

let s = softmax(B)
let pipeline = s >>> expectation
let g = f.change_base(pipeline)
```

`softmax(B)` builds a `MorphismTransformation` with `source = ProductFuzzyAlgebra, target = Markov`; `expectation` is a `AlgebraHomomorphism` with `source = Markov, target = ProductFuzzyAlgebra`. The composition `s >>> expectation` round-trips back to ProductFuzzyAlgebra and gives you a morphism whose rows are normalized on the way out.

## The monadic programs

A `program` block compiles to a `MonadicProgram` morphism in the [Kleisli category](https://en.wikipedia.org/wiki/Kleisli_category) of the (discrete or continuous) [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) ([Giry, 1982](https://doi.org/10.1007/BFb0092872)); see [Fritz, 2020](https://doi.org/10.1016/j.aim.2020.107239) for the more recent synthetic theory of [Markov categories](https://ncatlab.org/nlab/show/Markov+category). Each statement is a Kleisli arrow:

- `v <- F(args)` is monadic bind: sample from `F(args)` and extend the joint kernel.
- `observe v <- F(args)` is conditioning: clamp `v` to the observed value and score the likelihood.
- `let v = expr` is a pure morphism: a deterministic computation extending the joint kernel by a delta mass.
- `marginalize v : A <- F(args)` followed by an indented body is the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) along the projection that integrates `v` out of the body's joint.
- `return v` is the unit-of-monad-projection: project the joint kernel onto `v`'s coordinate.

This is the same Kleisli structure that Pyro, NumPyro, and Church use; the difference is that QVR's effects (`Sample`, `Score`, `Marginal`, `Pure`) are tracked as a static side-channel and checked at compile time.

## A worked example: regression as enriched morphism

The chapter-1 regression compiles to a Kern morphism $\mathbf{1} \to \mathcal{G}(\mathbb{R}^{|\text{Item}|})$ in the continuous Kleisli category, parameterized by the latents `beta_0, beta_1, sigma`. SVI replaces the integral over latents with a tractable lower bound; NUTS samples it directly. The categorical surface is the same in both cases: you've defined a morphism in $\mathbf{Kern}$; the inference algorithms are different ways to compute its expectations.

## Further reading

The denotational [semantics](../../semantics/index.md) section gives a rigorous treatment of everything in this chapter and considerably more: the formal grammar, the categorical setting, the compositional semantics of every DSL construct, the adequacy theorem connecting compiler implementation to denotation. It assumes Kelly-level enriched category theory.

The Python API tutorials (chapters 6 and 7) go further on first-class transformations and the composition-rule hierarchy.

The guides under `guides/` cover individual feature areas at the level of "what's the API and how do I use it" (categorical, monadic, enriched, deduction, structural-compression).

## You're done

If you've read all seven chapters, you can write models in QVR, fit them with any of the shipped inference algorithms, marginalize discrete latents, build hierarchical and sequence-shaped models, and read the categorical machinery underneath when you need to. Suggested next stops:

- The [examples gallery](../../examples/index.md) for end-to-end model code (Bayesian regression, mixture models, VAE, transformer, vanilla RNN).
- The [DSL reference guide](../../guides/dsl-overview.md) for the full surface of `.qvr` syntax.
- The [inference benchmark report](../../developer/inference-benchmarks.md) for the empirical truth-table of which algorithm fits which problem.
