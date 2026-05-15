# 6. First-class transformations

A *change-of-base* transformation turns a $\mathcal{V}$-enriched morphism into a $\mathcal{W}$-enriched one. Two flavours coexist in the library:

- [`QuantaleHomomorphism`](../../api/core/quantales.md) values ([Rosenthal, 1990](https://doi.org/10.1090/conm/094)) are lax monoidal [lattice](https://en.wikipedia.org/wiki/Lattice_(order)) maps $\varphi : \mathcal{V} \to \mathcal{W}$ that act pointwise: every entry of the morphism's tensor is sent through $\varphi$.
- [`MorphismTransformation`](../../api/core/morphisms.md) values act on the whole tensor, not entry-by-entry. Softmax row-normalization, L1/L2 row-normalization, Bayes inversion under a prior; all consume axis information that pointwise actions don't see.

Both inherit a common interface: a `.source` quantale, a `.target` quantale, and an `apply` that ingests a tensor (plus a morphism for shape resolution). The Python API treats them as first-class values: you bind them to local names, compose them by calling [`compose_trans`](../../api/core/quantales.md) (the DSL surface spells this as `>>>`), and pass either kind into [`Morphism.change_base`](../../api/core/morphisms.md).

## The catalog

```python
from quivers.core.quantale_morphisms import (
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
| `EXPECTATION` | `Markov` | `ProductFuzzy` | Reinterpret a row-stochastic kernel as fuzzy membership. |
| `MATERIAL_IMPLICATION` | `ProductFuzzy` | `Godel` | Reichenbach implication lift. |
| `PROBABILITY_CLAMP` | `Real` | `Probability` | Clamp real entries to `[0, 1]`. |
| `PROBABILITY_TO_REAL` | `Probability` | `Real` | Forget the `[0, 1]` constraint. |
| `COUNTING_FROM_REAL` | `Real` | `Counting` | Round real entries to non-negative integers. |
| `COUNTING_TO_REAL` | `Counting` | `Real` | Embed counts as reals. |

The shipped *constructors* (one argument):

| Constructor | Argument | `.source` | `.target` |
|---|---|---|---|
| `softmax(axis)` | `SetObject` | `ProductFuzzy` | `Markov` |
| `l1_normalize(axis)` | `SetObject` | `Real` | `Markov` |
| `l2_normalize(axis)` | `SetObject` | `Real` | `Real` |
| `bayes_invert(prior)` | `Morphism` or `Tensor` | `Markov` | `Markov` |

## Applying one transformation

```python
import torch
from quivers import FinSet, observed
from quivers.core.morphism_transformations import softmax

A = FinSet("A", 3)
B = FinSet("B", 4)
f = observed(A, B, torch.rand(3, 4))      # ProductFuzzy-enriched

phi = softmax(B)
g = f.change_base(phi)                     # Markov-enriched
print(g.tensor.sum(dim=-1))                # tensor([1., 1., 1.])
print(g.quantale.name)                     # 'MarkovQuantale'
```

The result is an `ObservedMorphism` over `Markov`. The transformation is applied to the tensor; the domain and codomain object are preserved for shape-aware transformations like softmax (Bayes inversion is the exception: it swaps them).

## Composing transformations

In Python the call site is [`compose_trans`](../../api/core/quantales.md); in `.qvr` source it surfaces as the `>>>` operator. Both wrap two or more [`MorphismTransformation`](../../api/core/morphisms.md) or [`QuantaleHomomorphism`](../../api/core/quantales.md) values into a sequential composition. The compose-time check verifies that `t1.target` matches `t2.source`; otherwise it raises.

```python
from quivers.core.quantale_morphisms import EXPECTATION
from quivers.core.morphism_transformations import softmax
from quivers.core.trans import compose_trans

phi   = softmax(B)                # ProductFuzzy -> Markov
psi   = EXPECTATION               # Markov       -> ProductFuzzy
pipe  = compose_trans(phi, psi)   # ProductFuzzy -> ProductFuzzy

g = f.change_base(pipe)
print(g.quantale.name)            # back to 'ProductFuzzy'
```

If the seams don't match (e.g. `compose_trans(softmax(B), PROBABILITY_TO_REAL)` would try to go `Markov` to `Probability`), `compose_trans` raises a `TypeError` naming the mismatch.

`compose_trans` returns a [`TransSeq`](../../api/core/quantales.md) value: a flattened sequence of single steps. Calling `change_base` on a `TransSeq` iterates the steps, applying each in turn. Nested compositions flatten so the result is always a flat tuple of steps.

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

Unit = FinSet("Unit", 1)
A    = FinSet("A", 3)

prior_tensor = torch.tensor([[0.5, 0.3, 0.2]])
prior        = observed(Unit, A, prior_tensor, quantale=MARKOV)

f = observed(A, A, torch.rand(3, 3), quantale=MARKOV)
g = f.change_base(bayes_invert(prior))
print(g.tensor.sum(dim=-1))      # rows of g sum to 1 (Markov)
```

## From the DSL surface

Inside `.qvr` files, the same machinery is the `change_base(t)` postfix and the `>>>` operator:

```qvr
quantale product_fuzzy
object A : 3
object B : 4
latent f : A -> B

let s    = softmax(B)
let pipe = s >>> expectation
let g    = f.change_base(pipe)
export g
```

`let` for trans-valued RHS lands the binding in the compiler's transformation namespace (disjoint from morphisms). The [QVR categorical tutorial](../qvr/07-categorical.md) covers the user-side; this Python-side surface mirrors it directly.

## Why this matters

Transformations are first-class values in the same sense morphisms are: you bind them to local names, hand them to functions, and compose them into longer pipelines. Anywhere the API accepts a morphism, the analogous slot accepts a transformation; anywhere the API composes two morphisms, the analogous combinator composes two transformations. The composition check enforces that adjacent steps' source and target quantales line up, so a pipeline either type-checks before any tensor work happens or fails fast at compose time.

## Try this

- Build a pipeline that round-trips a `ProductFuzzy` morphism through `Markov` and back: `compose_trans(softmax(B), EXPECTATION)`. Verify the source and target match.
- Apply the same pipeline to two different morphisms `f` and `h`; confirm the result has the same quantale for both.
- Construct an invalid pipeline (`compose_trans(softmax(B), PROBABILITY_TO_REAL)`) and observe the `TypeError`.

## Next

[Tutorial 7](07-composition-rules.md) introduces the `CompositionRule → BilinearForm | Semigroupoid → Quantale` hierarchy and the `EinsumWiring` surface for operadic n-ary contractions.
