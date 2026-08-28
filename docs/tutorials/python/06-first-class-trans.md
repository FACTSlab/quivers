# 6. First-class transformations

A *change-of-base* transformation turns a $\mathcal{V}$-enriched morphism into a $\mathcal{W}$-enriched one. Two flavours coexist in the library:

- [`AlgebraHomomorphism`](../../api/core/algebras.md) values (Rosenthal, 1990, *Quantales and Their Applications*, Pitman Research Notes in Mathematics 234, Longman; ISBN 978-0-582-06423-2) are lax monoidal [lattice](https://en.wikipedia.org/wiki/Lattice_(order)) maps $\varphi : \mathcal{V} \to \mathcal{W}$ that act pointwise: every entry of the morphism's tensor is sent through $\varphi$.
- [`MorphismTransformation`](../../api/core/morphisms.md) values act on the whole tensor, not entry-by-entry. Softmax row-normalization, L1/L2 row-normalization, Bayes inversion under a prior; all consume axis information that pointwise actions don't see.

Both inherit a common interface: a `.source` algebra, a `.target` algebra, and an `apply` that ingests a tensor (plus a morphism for shape resolution). The Python API treats them as first-class values: you bind them to local names, compose them by calling [`compose_trans`](../../api/core/algebras.md) (the DSL surface spells this as `>>>`), and pass either kind into [`Morphism.change_base`](../../api/core/morphisms.md).

### Why a typed transformation surface?

In a PyTorch program you'd write `softmax(logits, dim=-1)` whenever you needed to normalise. That's fine for a one-shot call, but it has no record of *which algebra the result lives in* once normalised. If you then compose with another morphism that lives in a fuzzy algebra, nothing flags the mismatch and the result is mathematically incoherent (you've sum-product-composed a row-stochastic tensor with a noisy-OR one).

A [`MorphismTransformation`](../../api/core/morphisms.md) is a typed function on morphisms: `softmax(B)` doesn't just normalise a tensor, it announces "I take a ProductFuzzyAlgebra morphism in, I emit a Markov morphism out". The compiler can then verify downstream compositions. Concretely, applying `softmax(B)` to any row-stochastic-shape morphism `f` returns a new morphism `f.change_base(softmax(B))` whose algebra tag is `Markov`. Subsequent `>>` composition against another Markov morphism accepts it; `>>` against a fuzzy morphism rejects it, since the two operands carry different algebras.

This is the same idea as PyMC's `pm.Deterministic` wrapping a transformation so that the trace knows about it, but lifted to the algebra layer: the tag isn't just "deterministic", it's "lives in algebra W now".

## The catalog

```python
from quivers.core.algebra_morphisms import (
    EXPECTATION, MATERIAL_IMPLICATION,
    PROBABILITY_CLAMP, PROBABILITY_TO_REAL,
    COUNTING_FROM_REAL, COUNTING_TO_REAL,
    IdentityHom, Threshold, Embedding,
)
from quivers.core.morphism_transformations import (
    softmax, l1_normalize, l2_normalize, bayes_invert,
    Softmax, L1Normalize, L2Normalize, BayesInvert,
)
```

The shipped *singletons* (no arguments needed):

| Name | `.source` | `.target` | What it does |
|---|---|---|---|
| `EXPECTATION` | `Markov` | `ProductFuzzyAlgebra` | Reinterpret a row-stochastic kernel as fuzzy membership. |
| `MATERIAL_IMPLICATION` | `ProductFuzzyAlgebra` | `Godel` | Reichenbach implication lift. |
| `PROBABILITY_CLAMP` | `Real` | `Probability` | Clamp real entries to `[0, 1]`. |
| `PROBABILITY_TO_REAL` | `Probability` | `Real` | Forget the `[0, 1]` constraint. |
| `COUNTING_FROM_REAL` | `Real` | `Counting` | Floor real entries and clamp them to non-negative integers. |
| `COUNTING_TO_REAL` | `Counting` | `Real` | Embed counts as reals. |

The shipped *constructors* (one argument):

| Constructor | Argument | `.source` | `.target` |
|---|---|---|---|
| `softmax(axis)` | `SetObject` | `ProductFuzzyAlgebra` | `Markov` |
| `l1_normalize(axis)` | `SetObject` | `Real` | `Markov` |
| `l2_normalize(axis)` | `SetObject` | `Real` | `Real` |
| `bayes_invert(prior)` | `Morphism` or `Tensor` | `Markov` | `Markov` |

## Applying one transformation

```python
import torch
from quivers import FinSet, observed
from quivers.core.morphism_transformations import softmax

A = FinSet(name="A", cardinality=3)
B = FinSet(name="B", cardinality=4)
f = observed(A, B, torch.rand(3, 4))      # ProductFuzzyAlgebra-enriched

phi = softmax(B)
g = f.change_base(phi)                     # Markov-enriched
print(g.tensor.sum(dim=-1))                # tensor([1., 1., 1.])
print(g.algebra.name)                     # 'MarkovAlgebra'
```

The result is an `ObservedMorphism` over `Markov`. The transformation is applied to the tensor; the domain and codomain object are preserved for shape-aware transformations like softmax (Bayes inversion is the exception: it swaps them).

## Composing transformations

In Python the call site is [`compose_trans`](../../api/core/algebras.md); in `.qvr` source it surfaces as the `>>>` operator. Both wrap two or more [`MorphismTransformation`](../../api/core/morphisms.md) or [`AlgebraHomomorphism`](../../api/core/algebras.md) values into a sequential composition. The compose-time check verifies that `t1.target` matches `t2.source`; otherwise it raises.

```python
from quivers.core.algebra_morphisms import EXPECTATION
from quivers.core.morphism_transformations import softmax
from quivers.core.trans import compose_trans

phi   = softmax(B)                # ProductFuzzyAlgebra -> Markov
psi   = EXPECTATION               # Markov       -> ProductFuzzyAlgebra
pipe  = compose_trans(phi, psi)   # ProductFuzzyAlgebra -> ProductFuzzyAlgebra

g = f.change_base(pipe)
print(g.algebra.name)            # back to 'ProductFuzzyAlgebra'
```

If the seams don't match (e.g. `compose_trans(softmax(B), PROBABILITY_TO_REAL)` would try to go `Markov` to `Probability`), `compose_trans` raises a `TypeError` naming the mismatch.

`compose_trans` returns a [`TransSeq`](../../api/core/algebras.md) value: a flattened sequence of single steps. Calling `change_base` on a `TransSeq` iterates the steps, applying each in turn. Nested compositions flatten so the result is always a flat tuple of steps.

## Three-step pipelines

```python
pipe = compose_trans(
    softmax(B),
    EXPECTATION,
    softmax(B),
)
g = f.change_base(pipe)
```

`compose_trans` takes any number of arguments. Each adjacent pair is type-checked at compose time.

## Bayes inversion

`bayes_invert(prior)` is the one constructor whose argument is a *morphism* rather than an *object*. It builds a `BayesInvert` transformation parameterized by the prior; the transformed kernel is the Bayes-inverted posterior under that prior. Domain and codomain swap.

```python
import torch
from quivers import FinSet, observed, MARKOV
from quivers.core.morphism_transformations import bayes_invert

Unit = FinSet(name="Unit", cardinality=1)
A    = FinSet(name="A", cardinality=3)

prior_tensor = torch.tensor([[0.5, 0.3, 0.2]])
prior        = observed(Unit, A, prior_tensor, algebra=MARKOV)

f = observed(A, A, torch.rand(3, 3), algebra=MARKOV)
g = f.change_base(bayes_invert(prior))
print(g.tensor.sum(dim=-1))      # rows of g sum to 1 (Markov)
```

## From the DSL surface

Inside `.qvr` files, the same machinery is the `change_base(t)` postfix and the `>>>` operator:

```qvr
composition product_fuzzy [level=algebra]
object A : FinSet 3
object B : FinSet 4
morphism f : A -> B [role=latent]

define s    = softmax(B)
define pipe = s >>> expectation
define g    = f.change_base(pipe)
export g
```

`let` for trans-valued RHS lands the binding in the compiler's transformation namespace (disjoint from morphisms). The [QVR categorical tutorial](../qvr/07-categorical.md) covers the user-side; this Python-side surface mirrors it directly.

## Why this matters

Transformations are first-class values in the same sense morphisms are: you bind them to local names, hand them to functions, and compose them into longer pipelines. Anywhere the API accepts a morphism, the analogous slot accepts a transformation; anywhere the API composes two morphisms, the analogous combinator composes two transformations. The composition check enforces that adjacent steps' source and target algebras line up, so a pipeline either type-checks before any tensor work happens or fails fast at compose time.

## Try this

- Build a pipeline that round-trips a `ProductFuzzyAlgebra` morphism through `Markov` and back: `compose_trans(softmax(B), EXPECTATION)`. Verify the source and target match.
- Apply the same pipeline to two different morphisms `f` and `h`; confirm the result has the same algebra for both.
- Construct an invalid pipeline (`compose_trans(softmax(B), PROBABILITY_TO_REAL)`) and observe the `TypeError`.

## Next

[Tutorial 7](07-composition-rules.md) introduces the `CompositionRule → BilinearForm | Semigroupoid → Algebra` hierarchy and the `EinsumWiring` surface for operadic n-ary contractions.
