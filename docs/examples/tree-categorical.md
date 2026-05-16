# Tree-Structured Categorical Prior

## Overview

A finite-class model in which the $K$-way class-probability vector is *not* drawn from a flat [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) but assembled from a binary decision tree. Each leaf class is a structurally different product of internal-node [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) probabilities, and the per-verb / per-class score table is the rank-2 tensor produced by evaluating a joint-additive body once per cell of the Cartesian product `Verb × Class`.

This example is the canonical demonstration of the [`factor`](../guides/dsl-programs-and-lets.md#factor-expressions-assembling-indexed-tensors) expression, the [left adjoint](https://ncatlab.org/nlab/show/adjoint+functor) of indexing. Both surface forms appear in the same program: the pattern-match form builds the tree-shaped leaf-log-probability vector with a `{ ... }` case table, and the multi-binder uniform form builds the rank-2 score tensor by evaluating its body once per cell.

## QVR Source

```qvr
object Verb : 12
object Class : 4
object Resp : 200

program tree_categorical : Resp -> Resp
    p_root  <- Beta(1.0, 1.0)
    p_left  <- Beta(1.0, 1.0)
    p_right <- Beta(1.0, 1.0)

    let leaf_log = factor cls : Class in {
        0 -> log(1.0 - p_root) + log(1.0 - p_left),
        1 -> log(1.0 - p_root) + log(p_left),
        2 -> log(p_root)       + log(1.0 - p_right),
        3 -> log(p_root)       + log(p_right),
    }

    sigma_v <- HalfNormal(1.0)
    delta : Verb  <- Normal(0.0, sigma_v)
    mu    : Class <- Normal(0.0, 1.0)

    let cell_score = factor v : Verb, cls : Class in
        delta[v] + mu[cls] + leaf_log[cls]

    let cell0 = cell_score[0, 0]
    observe y : Resp <- Normal(cell0, 0.5)
    return delta

export tree_categorical
```

## Walkthrough

### Pattern-match factor: tree-shaped leaf probabilities

<!-- compile: false -->
```qvr
let leaf_log = factor cls : Class in {
    0 -> log(1.0 - p_root) + log(1.0 - p_left),
    1 -> log(1.0 - p_root) + log(p_left),
    2 -> log(p_root)       + log(1.0 - p_right),
    3 -> log(p_root)       + log(p_right),
}
```

The pattern-match form `factor cls : I in { 0 -> e_0, ..., n-1 -> e_{n-1} }` denotes a tensor of shape `(|I|, ...)` whose `i`-th cell is `e_i`. Here each leaf class is a structurally different product of internal-node log-probabilities, reflecting the geometry of a binary decision tree: classes 0 and 1 sit beneath the left child of the root (`1 - p_root` × left branch); classes 2 and 3 sit beneath the right (`p_root` × right branch). The compiler enforces label coverage of `{0, ..., |Class|-1}` exactly and rejects gaps, duplicates, or out-of-range labels at compile time.

This is the categorical surface for *structurally heterogeneous* indexed families: distributions over $\mathsf{Class}$ whose cells come from different upstream latents in different ways.

### Multi-binder uniform factor: the joint score tensor

<!-- compile: false -->
```qvr
let cell_score = factor v : Verb, cls : Class in
    delta[v] + mu[cls] + leaf_log[cls]
```

The multi-binder form `factor v_1 : I_1, ..., v_n : I_n in <body>` is the [left adjoint of multi-axis indexing](../guides/dsl-programs-and-lets.md#factor-expressions-assembling-indexed-tensors). Its denotation is the tensor of shape `(|I_1|, ..., |I_n|, *body_shape)` whose `(i_1, ..., i_n)`-th cell is the body evaluated with each binder `v_k := i_k`.

Here the body indexes into three previously-bound objects: the `Verb`-plate `delta`, the `Class`-plate `mu`, and the pattern-match factor `leaf_log` produced two steps earlier. Each `(v, cls)` cell evaluates to a different scalar; the resulting tensor lives on `Verb × Class` and carries the joint per-verb / per-class log-score.

The binder variables `v` and `cls` are integer-valued and visible only inside the body, mirroring the binder-localization of `let` in any functional language.

### Why factor and not a plate

A plate-bound draw `delta : Verb <- Normal(0.0, sigma_v)` and a factor expression `factor v : Verb in <body>` are not interchangeable. The plate draws an `|Verb|`-shape tensor of *independent* samples from the same kernel; the factor evaluates a *deterministic body* once per index and assembles the results into a tensor. The plate's family is exchangeable in its index; the factor's body is allowed to depend on the index in arbitrary structurally-different ways.

This is why no other example in the gallery uses `factor`: existing models all use exchangeable priors (symmetric Dirichlet, plate-bound Normal) where the plate surface is correct. `factor` becomes the right tool when the index axis is structured (a binary tree, a directed acyclic group structure, a heterogeneous mixture of distinct sub-priors) and the cells of that index are different functions of upstream latents.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)

prog = load("docs/examples/source/tree_categorical.qvr")
model = prog.morphism

N = 200
y = torch.randn(N) * 0.5
obs = {"y": y.unsqueeze(0)}

guide = AutoNormalGuide(model, observed_names=set(obs.keys()))
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=2e-2,
)
svi = SVI(model, guide, optim, ELBO())
for _ in range(200):
    loss = svi.step(torch.zeros(1, 1, dtype=torch.long), obs)
print("Tree-categorical loss:", loss)
```

## Categorical Perspective

A `factor` expression is the [left Kan extension](https://ncatlab.org/nlab/show/left+Kan+extension) of its body along the finite-set indexing functor. Where the indexing surface `arr[i]` is the elimination rule for the dependent function space `I → A` (the right adjoint, projection of a constant kernel), `factor v : I in <body>` is the introduction rule: it freely generates the indexed family from a binder-parameterized body. Composing the two recovers the identity, mirroring the unit and counit of the adjunction.

In the multi-binder case, the indexing functor is `I_1 × ... × I_n → A` and `factor` is the corresponding multi-axis introduction. Inside `\mathbf{Kern}` the construction is functorial in each `I_k`: morphisms of finite-set index objects lift to natural transformations between the factor tensors over the corresponding products.

## See Also

- [DSL Guide: Factor expressions](../guides/dsl-programs-and-lets.md#factor-expressions-assembling-indexed-tensors)
- [Mixture Model](mixture-model.md), the exchangeable counterpart: a flat Dirichlet over `Component` rather than a tree-shaped construction.
