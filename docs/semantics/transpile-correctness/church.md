# Church

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{Church}$.

## Semantics

Church's denotational semantics is the stochastic lambda calculus
of Goodman, Mansinghka, Roy, Bonawitz, and Tenenbaum
([2008](https://arxiv.org/abs/1206.3255)). A program is a Scheme
expression whose `sample` invocations introduce randomness; the
trace is a sequence of (address, distribution, value) triples. The
joint log-density is the product of the per-site contributions.

Church does not have a single canonical compiler or runtime; the
target language is not pinned to one external implementation. Church
is absent from the shared Docker numeric matrix; Church-specific audit
tests provide narrower evidence.

## Unconstrained-space change of variables

Identity. Church's distributions are operationally defined and do
not carry an automatic reparametrization layer.

## Family parameterizations

Church uses standard distribution names (`gaussian`, `dirichlet`,
`categorical`, `bernoulli`, `beta`, ...). The QVR ↔ Church mapping
is identity for every family that has a Church counterpart;
$c_{F, \mathsf{Church}} = 0$.

## Per-construct emit

**Sample / observe.** `(define <name> (sample (<family>
<args>)))` for latents; `(observe (<family> <args>) <obs>)` for
observations.

**Plate.** `(map (lambda (m) (sample (<family> <args>))) (iota
B))` produces a length-$B$ list of i.i.d. samples; observed plates
use `(for-each (lambda (n) (observe ...)) (iota N))`. By
Goodman et al. (2008) §3.2, `map` over `iota` denotes the product
measure of $B$ independent samples.

**Marginalize.** Explicit-latent rewrite. The Church abstract
machine supports discrete-latent sampling via the same trace
mechanism as continuous samples.

**Score / let / return.** `(factor <expr>)` for score (per
Goodman & Stuhlmüller 2014); `(define <name> <expr>)` for let;
the trailing expression in the `(define (model ...) ...)` body
is the return value.

**Limitations.** Church's renderer raises
`UnsupportedConstruct(["arg:matrix-literal"])` on matrix-literal
arguments (no canonical Scheme matrix form). Wrapper families
that need access to an inner morphism's `init_family` (Truncated
via rejection sampling) are also not supported.

## Acceptance

* **Tier 1 structural.** Every emit has `(define (model ...) ...)`
  with `sample` / `observe` invocations.
* **Tier 1 pipeline composition.** Direct and composed pipeline calls agree.
* **Tier 2 external syntax.** No canonical external Church compiler
  is exercised by the shared syntax suite. Parenthesis balance is a
  smoke check only, not a sufficient syntax criterion.
* **Tier 3 numeric equivalence.** Not part of the shared Docker matrix;
  Church-specific audit tests cover selected emitted programs.
