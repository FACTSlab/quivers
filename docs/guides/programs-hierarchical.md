# Hierarchical Programs

This page covers the advanced program-composition surface that
makes idiomatic hierarchical Bayesian models expressible:
parametric program templates that share structure across grouping
factors, and the grouped marginalization construct for jointly
identifying a discrete class indicator by multiple heterogeneous
response axes. The basic `MonadicProgram` surface lives in the
[programs guide](programs.md).

## Hierarchical models with parametric templates

A [parametric
program](dsl-programs-and-lets.md#parametric-programs) declares a
reusable kernel template polymorphic over typed parameters
(`FinSet`, `Space`, `Object`, `Real`, `Nat`, or `Mor[A, B]`). Each
call site `v <- template(...)` inlines a fresh α-renamed copy of
the template's body, so call sites contribute distinct latents.

<!-- compile: false -->
```qvr
object Subject : 200
object Verb : 100
object Resp : 5000

program random_intercepts (G : FinSet, scale : Real) : G -> 1
    sigma <- HalfNormal(scale)
    v : G <- Normal(0.0, sigma)
    return v

program crossed : Resp -> Resp
    intercept <- Normal(0.0, 1.0)

    by_subject <- random_intercepts(Subject, 1.0)
    by_verb    <- random_intercepts(Verb,    1.0)

    observe response : Resp <- Bernoulli(intercept)
    return intercept

export crossed
```

Each `random_intercepts` call inlines an independent `sigma` and
per-level plate; the observed response is the runtime tensor
supplied via `observations={"response": response_tensor}`. Monotone
ordinal effects are expressed as `cumsum` of `HalfNormal`
increments (positive support implies monotone partial sums);
discrete latent classes are integrated out with a scoped
`marginalize ... in { ... }` block.

The plate-draw, vectorized-observe, parametric-program, and
`marginalize` constructs compose into the standard hierarchical
Bayesian forms. The pattern above shows crossed random intercepts
on two grouping factors, both reusing a single parametric template.
Each call to `random_intercepts` inlines a fresh `sigma` and a
fresh per-level plate `v` under α-renamed names
(`by_subject$sigma`, `by_subject$v`, ...), so the two grouping
factors share *structure* but not *latents*.

## Grouped marginalization: fibred discrete latents

A scoped `marginalize` block accepts an optional `over G` clause
that declares a *grouping plate* `G`. Inside the body, every
`observe` step carries its own `via <idx>` clause naming the
fibration `idx : Resp_m -> G` from that observe's response plate to
the shared grouping plate. The body's per-axis per-row per-class
log-likelihoods are scatter-summed into a single $(|G|, K)$
accumulator before the log-sum-exp over the class axis:

```
marginalize class : K <- Categorical(probs)
    over G
    in {
        observe r_a : Resp_a via idx_a <- F_a(...)
        observe r_b : Resp_b via idx_b <- F_b(...)
        ...
    }
```

The block contributes

$$
\sum_{g \in G}\ \log\sum_{k=1}^{K}\exp\!\left[\log \pi(g, k) + \sum_{m}\sum_{n:\ \mathrm{idx}_m(n)=g}\ell_m(n, k)\right]
$$

to the program-level log-density. Categorically this is the
[right Kan extension](https://ncatlab.org/nlab/show/Kan+extension)
along the coproduct fibration $\coprod_m r_m : \coprod_m
\mathrm{Resp}_m \to G$ in
[`Kern`](../api/stochastic/categories.md), composed with the
standard categorical-marginal log-sum-exp under the prior $\pi$.

### Degenerate forms

The fibred form is the canonical hierarchical-Bayes likelihood
pattern, equivalent to Stan's per-item
`target += log_mix(probs, sum_m ll_item_m[i])` accumulation. It
degenerates as expected:

- *No grouping plate*: the global mixture form (ungrouped
  `marginalize`).
- *Single observe with identity fibration* (one group per row): the
  per-row mixture.
- *Single observe with coarser fibration*: the per-block
  hierarchical mixture.
- *Multiple observes sharing a per-item class indicator*: the
  joint mixture across heterogeneous response axes.

### Constraints and runtime

The `over` object must be declared. Each observe's `via` name must
be a previously bound plate variable. The categorical family's
first argument is the prior tensor; its shape may be `(K,)` (shared
across groups) or `(|G|, K)` (per-group prior). The runtime
primitive is exposed at
`quivers.continuous.plate.marginalize_grouped(ll, idx, log_prior,
num_groups)` and accepts either a single `(N, K)` tensor
(single-observe case) or a parallel list of `(N_m, K)` tensors with
their per-axis fibrations (multi-observe case).

### Product fibrations

Product fibrations are supported on each observe via
`via product(idx_a, idx_b)`, paired with an `over G * H` product
grouping plate on the marginalize header. The product-fibration
arity must match the grouping plate's arity. See the
[composition-rule semantics](../semantics/composition-rules.md) for
the formal denotation and
`tests/test_grouped_marginalize_combinations.py` for examples.

## End-to-end fit

```python
from quivers.dsl import load
from quivers.inference import ELBO, AutoNormalGuide, SVI

program = load("crossed.qvr")
model = program.morphism  # underlying MonadicProgram
observations = {"response": response_tensor}

guide = AutoNormalGuide(model, observed_names={"response"})
elbo  = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)

for _ in range(2000):
    svi.step(domain_input, observations)
```

## See also

- [Monadic Programs](programs.md): the basic program semantics and
  the `rsample` / `log_joint` contract this page builds on.
- [DSL Programs and Let-Expressions](dsl-programs-and-lets.md#parametric-programs):
  the surface syntax for parametric templates and the axis-role
  clause used in the observe / plate steps.
- [Analysis Pipelines: Data and Formulas](analysis-data-and-formulas.md):
  the brms-style formula entry point that compiles a
  random-intercepts formula into this hierarchical surface
  automatically.
