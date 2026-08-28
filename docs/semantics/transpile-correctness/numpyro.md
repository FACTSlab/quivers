# NumPyro

This page discharges the per-target obligations of
[Transpilation correctness](index.md) for $\mathsf{T} = \mathrm{NumPyro}$.

## Semantics

NumPyro's denotational semantics is the trace semantics of
Goodman et al. ([2008](https://arxiv.org/abs/1206.3255)) /
Wingate, Stuhlmüller, Goodman
([2011](http://proceedings.mlr.press/v15/wingate11a.html))
implemented via the effect-handler architecture of Phan, Pradhan,
and Jankowiak
([2019](https://doi.org/10.48550/arXiv.1912.11554)). A program
is a JAX-traced Python function whose `numpyro.sample` invocations
register sites in a trace. The log-density probe is
[`numpyro.infer.util.log_density`](https://num.pyro.ai/en/stable/utilities.html#log-density);
NumPyro returns the joint log-density of the trace.

## Unconstrained-space change of variables

NumPyro applies the same change-of-variables principle as Stan but
with reparametrization handled by per-site
[`TransformReparam`](https://num.pyro.ai/en/stable/_modules/numpyro/infer/reparam.html)
or auto-transformations registered in
[`AutoNormal`](https://num.pyro.ai/en/stable/autoguide.html#autonormal).
For QVR-emitted code at the level the renderer produces, no
explicit reparametrization is requested: $\Psi_{\mathsf{NumPyro}}
= \mathrm{id}$ and $\log|\det J| \equiv 0$. The inference layer
may add reparametrizations at runtime, but those are guide-side
and do not affect the model-side denotation Theorem 6.1 quantifies
over.

## Family parameterizations

NumPyro families track PyTorch / Pyro's
[`Distribution`](https://pytorch.org/docs/stable/distributions.html)
hierarchy. The mapping is identity for every QVR family that has
a corresponding [`numpyro.distributions.*`](https://num.pyro.ai/en/stable/distributions.html)
class:

* `Normal(μ, σ)` ↔ `Normal(loc=μ, scale=σ)`
* `Dirichlet(α)` ↔ `Dirichlet(concentration=α)` (with `jnp.full((K,), α)` broadcast when α is scalar)
* `Categorical(p)` ↔ `Categorical(probs=p)`
* `Bernoulli(p)` ↔ `Bernoulli(probs=p)`
* `LogitNormal` ↔ `LogitNormal(loc=μ, scale=σ)`

In each case $\pi_{F, \mathsf{NumPyro}}$ is the identity and
$c_{F, \mathsf{NumPyro}} = 0$.

## Per-construct emit

**Sample / observe.** `numpyro.sample("x", <dist>)` for latents;
`numpyro.sample("y", <dist>, obs=y_data)` for observations. Each
contributes the documented per-site log-density to the
[`log_density`](https://num.pyro.ai/en/stable/utilities.html#log-density)
probe.

**Plate.** Nested `with numpyro.plate(name, B):` contexts per the
documented [`plate`](https://num.pyro.ai/en/stable/primitives.html#numpyro.primitives.plate)
primitive (Phan, Pradhan, Jankowiak 2019 §3.2). The semantics is
exactly the conditionally-independent product measure of $B$
i.i.d. draws ([plate discussion](index.md#5-plates-marginalization-and-via)).

**Marginalize.** The NumPyro renderer lowers `IRMarginalize` to
`IRSample(latent) + scope body` inline (head
[the marginalization discussion](index.md#5-plates-marginalization-and-via)). The
latent contributes a normal sample site; the scope's observe site
runs with the same trace. Soundness for inference targets that
estimate the $\theta$-posterior is the projection property of
[Fritz 2020](https://doi.org/10.1016/j.aim.2020.107239)
Proposition 5.4.

**Score / let / return.** `numpyro.factor("name", expr)` for
score; deterministic assignment for let; native `return` for the
return clause.

## Acceptance

* **Tier 1 pipeline composition.** Structural checks cover the
  emitted `model` function and the `Lower >> NumPyroRenderer >>
  EmitPretty(python)` pipeline.
* **Tier 2 external syntax.** Python's AST parser accepts the
  generated modules in the external-validation matrix.
* **Tier 3 numeric equivalence.** For selected fixtures,
  `numpyro.infer.util.log_density` is compared with the QVR
  reference on the shared finite grid and tolerance described in
  [the test contract](index.md#6-evidence-tiers).
