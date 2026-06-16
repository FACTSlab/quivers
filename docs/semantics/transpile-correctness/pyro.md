# Pyro

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{Pyro}$.

## Semantics

Pyro's denotational semantics is the trace semantics of Bingham,
Chen, Jankowiak, Obermeyer, Pradhan, Karaletsos, Singh, Szerlip,
Horsfall, and Goodman
([2019](http://jmlr.org/papers/v20/18-403.html)) implemented via
[`pyro.poutine`](https://docs.pyro.ai/en/stable/poutine.html)
handlers. The log-density probe is

```python
trace = pyro.poutine.trace(pyro.condition(model, data={**θ, **y})).get_trace(x)
log_p = trace.log_prob_sum()
```

returning the joint log-probability of the trace.

## Unconstrained-space change of variables

Identity at the renderer level (the inference layer may insert
guide-side reparametrizations; not relevant to model-side
denotation).

## Family parameterizations

Pyro consumes the same
[`torch.distributions`](https://pytorch.org/docs/stable/distributions.html)
hierarchy as NumPyro. The QVR ↔ Pyro mapping is identity for every
QVR family that has a corresponding `pyro.distributions.*` class.
$\pi_{F, \mathsf{Pyro}} = \mathrm{id}$ and $c_{F, \mathsf{Pyro}} =
0$.

## Per-construct emit

**Sample / observe.** `pyro.sample("x", <dist>)` for latents;
`pyro.sample("y", <dist>, obs=y_data)` for observations. Each
registers a trace site with the documented log-density
contribution.

**Plate.** Nested `with pyro.plate(name, B):` contexts per
[`pyro.plate`](https://docs.pyro.ai/en/stable/primitives.html#pyro.plate)
(Bingham et al. 2019 §2.2). Semantics is the product measure of
$B$ i.i.d. draws.

**Marginalize.** Explicit-latent rewrite (lower `IRMarginalize` to
`IRSample(latent)` + scope inline). Soundness as for NumPyro.

**Score / let / return.** `pyro.factor("name", expr)` for score;
deterministic assignment for let; native `return`.

## Acceptance

* **Tier 1 structural.** `def model(...) ... return ...` with
  `with pyro.plate(...)` contexts wrapping `pyro.sample` and
  `pyro.factor` calls.
* **Tier 2 lens-laws.** Composition law holds for `Lower >>
  PyroRenderer >> EmitPretty(python)`.
* **Tier 3 external syntax.** `python -m ast` accepts every emit.
* **Tier 4 numeric equivalence.** `trace.log_prob_sum()` agrees
  with the QVR reference within $10^{-6}$; pairwise transitivity
  with NumPyro / Edward2 holds.
