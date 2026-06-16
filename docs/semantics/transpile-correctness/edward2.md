# Edward2

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{Edward2}$.

## Semantics

Edward2's denotational semantics is the trace semantics of
Tran, Hoffman, Saurous, Brevdo, Murphy, and Blei
([2018](https://arxiv.org/abs/1810.03958)) instantiated on top of
[TensorFlow Probability](https://www.tensorflow.org/probability)'s
[`tfp.distributions`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions)
hierarchy. A program is a Python function whose
`edward2.<Family>(...)` calls register
[`RandomVariable`](https://www.tensorflow.org/probability/api_docs/python/tfp/edward2/RandomVariable)
sites in an `ed.tape` context; the log-density probe is

```python
with ed.tape() as t:
    model(...)
log_p = sum(rv.distribution.log_prob(rv.value) for rv in t.values())
```

returning the joint log-density of the trace.

## Unconstrained-space change of variables

Identity at the renderer level. TFP handles bijector-based
reparametrizations at inference time;
$\Psi_{\mathsf{Edward2}} = \mathrm{id}$ for the model-side
denotation.

## Family parameterizations

Edward2 families are TFP `Distribution` classes exported via
[`edward2`](https://www.tensorflow.org/probability/api_docs/python/tfp/edward2).
The QVR ↔ Edward2 arg mapping is identity for every family that has
a corresponding TFP class. $\pi_{F, \mathsf{Edward2}} = \mathrm{id}$
and $c_{F, \mathsf{Edward2}} = 0$.

## Per-construct emit

**Sample / observe.** `<name> = edward2.<Family>(<args>,
sample_shape=[B0, B1, ...], name="<name>")` for latents and
observed variables. The `sample_shape` argument carries the
batch-axis shape.

**Plate.** Captured via `sample_shape=[B]`. By TFP's documented
broadcasting semantics, `sample_shape=[B]` produces a $B$-fold
i.i.d. product (TFP user guide on "Shapes and Broadcasting").

**Marginalize.** Explicit-latent rewrite (head
[§5.3.2](index.md#532-explicit-latent-rewrite-under-mcmc)).

**Score / let / return.** Edward2 has no native `factor`
primitive; the renderer raises
`UnsupportedConstruct(["construct:ScoreStep"])` if a score step is
encountered. Deterministic let-bindings emit as Python
assignments. The function returns the named random variables.

## Acceptance

* **Tier 1 structural.** Every emit has a `def model(...)` whose
  body is a sequence of `edward2.<Family>(...)` calls with
  `sample_shape=[...]` and `name=...` parameters.
* **Tier 2 lens-laws.** Composition law holds.
* **Tier 3 external syntax.** `python -m ast` accepts every emit.
* **Tier 4 numeric equivalence.** The `ed.tape`-based probe agrees
  with the QVR reference within $10^{-6}$.

### References

* Dustin Tran, Matthew Hoffman, Rif A. Saurous, Eugene Brevdo,
  Kevin Murphy, and David M. Blei. 2018. Simple, distributed, and
  accelerated probabilistic programming. In *Neural Information
  Processing Systems*, 7598-7609.
  [https://arxiv.org/abs/1810.03958](https://arxiv.org/abs/1810.03958)
