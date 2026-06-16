# WebPPL

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{WebPPL}$.

## Semantics

WebPPL's denotational semantics is the continuation-passing-style
trace semantics of Goodman and Stuhlmüller
([2014](http://dippl.org)). A program is a JavaScript-syntax
WebPPL function whose `sample` and `observe` calls are reified by
the WebPPL compiler into CPS-transformed continuations; the trace
is the sequence of `(address, distribution, value)` triples
visited. The runtime computes the joint log-density per
[`webppl --output`](https://webppl.readthedocs.io/en/master/inference.html#log-density)
when used with `factor`-based importance sampling.

## Unconstrained-space change of variables

Identity. WebPPL distributions are operationally defined; no
automatic reparametrization layer.

## Family parameterizations

WebPPL families use JavaScript-object keyword arguments:
`Dirichlet({alpha: ...})`, `Categorical({ps: ...})`,
`Gaussian({mu: ..., sigma: ...})`, `Bernoulli({p: ...})`, etc.
The QVR ↔ WebPPL arg mapping requires non-trivial aliases that
live in `FAMILY_META.arg_aliases["webppl"]`:

| QVR arg | WebPPL arg |
|---|---|
| `loc` | `mu` |
| `scale` | `sigma` |
| `concentration` (Dirichlet) | `alpha` |
| `probs` | `ps` |

The renamings are syntactic; $c_{F, \mathsf{WebPPL}} = 0$ for every
family with native WebPPL support.

## Per-construct emit

**Sample / observe.** `var <name> = sample(<Family>({args}));`
for latents; `observe(<Family>({args}), <obs>);` for observations.

**Plate.** `repeat(B, function() { return sample(<dist>); })` for
index-independent batches; `mapIndexed(function(m, _) { return
sample(<dist using m>); }, repeat(B, 0))` for index-dependent.
Per Goodman & Stuhlmüller 2014, `repeat` denotes the documented
$B$-fold product measure.

**Marginalize.** Explicit-latent rewrite (head
[§5.3.2](index.md#532-explicit-latent-rewrite-under-mcmc)).
WebPPL natively samples discrete latents via the trace mechanism.

**Score / let / return.** [`factor(expr)`](https://webppl.readthedocs.io/en/master/inference.html#factor)
for score; `var <name> = <expr>;` for let; native `return`.

## Acceptance

* **Tier 1 structural.** Every emit has `var model = function(...)
  { ... };` whose body uses `sample`, `observe`, `factor`,
  `repeat`, and `mapIndexed`.
* **Tier 2 lens-laws.** Composition law holds.
* **Tier 3 external syntax.** [`node --check /dev/stdin`](https://nodejs.org/api/cli.html#--check)
  accepts every emit (WebPPL syntax is a subset of JavaScript +
  the WebPPL primitive functions).
* **Tier 4 numeric equivalence.** WebPPL importance-sampling
  estimates of the log-marginal agree with the QVR reference
  within sampling error (a wider tolerance than the analytic
  Tier-4 bound for the other backends).
