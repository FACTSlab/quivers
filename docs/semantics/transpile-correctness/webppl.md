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
visited. The repository's numeric probe rewrites rendered `sample`
and `observe` sites into calls to each distribution object's
`score(value)` method at clamped parameter and data values. It does
not estimate the density through importance sampling.

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
[the marginalization discussion](index.md#5-plates-marginalization-and-via)).
WebPPL natively samples discrete latents via the trace mechanism.

**Score / let / return.** [`factor(expr)`](https://webppl.readthedocs.io/en/master/inference.html#factor)
for score; `var <name> = <expr>;` for let; native `return`.

## Acceptance

* **Tier 1 structural.** Every emit has `var model = function(...)
  { ... };` whose body uses `sample`, `observe`, `factor`,
  `repeat`, and `mapIndexed`.
* **Tier 1 pipeline composition.** Direct and composed pipeline calls agree.
* **Tier 2 external syntax.** [`node --check /dev/stdin`](https://nodejs.org/api/cli.html#--check)
  accepts the JavaScript syntax. This does not validate WebPPL
  primitive names or runtime semantics.
* **Tier 3 numeric equivalence.** The rewritten `.score(value)`
  probe is compared with the QVR reference on selected fixture grids.
