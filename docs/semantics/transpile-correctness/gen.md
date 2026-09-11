# Gen.jl

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{Gen.jl}$.

## Semantics

Gen.jl's denotational semantics is the
generative-function trace semantics of Cusumano-Towner, Saad,
Lew, and Mansinghka
([2019](https://doi.org/10.1145/3314221.3314642)). A `@gen
function` is a generative function whose `@trace` invocations
record per-address choices in a `ChoiceMap`. The log-density
probe is [`Gen.assess(generator, args, choicemap)[1]`](https://www.gen.dev/docs/stable/ref/inference/#Gen.assess),
where Julia's `[1]` selects the returned log weight.

## Unconstrained-space change of variables

Identity at the renderer level. Gen's MCMC kernels handle their
own unconstrained-space reparametrizations.

## Family parameterizations

Gen consumes
[Distributions.jl](https://juliastats.org/Distributions.jl/) via
its `Gen.Distribution` wrappers. The QVR ↔ Gen mapping is identity
for every family that has a Distributions.jl counterpart.
$\pi_{F, \mathsf{Gen}} = \mathrm{id}$ and $c_{F, \mathsf{Gen}} = 0$.

Wrapped families (`Truncated`, `Mixture`) use Gen's
[Distributions.jl composition primitives](https://juliastats.org/Distributions.jl/stable/truncate/)
via a per-renderer `_WRAPPER_BUILDERS` dispatch table; cf.
[Architecture §10.10](../transpile-architecture.md).

## Per-construct emit

**Sample / observe.** Gen has no native plate; the renderer emits
per-batch-axis `for m in 1:B; <name>[m] = @trace(<dist>, (:name,
m)); end` loops over a pre-allocated `Vector{T}(undef, B)`
storage array. Each `@trace` registers an address in the choice
map.

**Plate.** The for-loop over `1:B` ranges produces $B$
independent `@trace` sites, each with the same conditional
distribution. The product measure denotation matches the head
[the plate discussion](index.md#5-plates-marginalization-and-via)
lemma.

**Marginalize.** Explicit-latent rewrite. Gen's MCMC
infrastructure supports discrete-latent sampling via custom
proposals.

**Score / let / return.** [`@addlogprob!(expr)`](https://www.gen.dev/docs/stable/ref/gfi/#Gen.@addlogprob!)
for score; native `<name> = <expr>` for let; native `return`.

## Acceptance

* **Tier 1 structural.** Every emit has `@gen function model(...)
  ... end` with `@trace` calls and per-batch `for` loops.
* **Tier 1 pipeline composition.** Direct and composed pipeline calls agree.
* **Tier 2 external syntax.** `julia --eval "Meta.parse(read(stdin, String); raise=true)"`
  accepts every emit.
* **Tier 3 numeric equivalence.** `Gen.assess(...)[1]` is compared
  with the QVR reference on the selected fixture grids.
