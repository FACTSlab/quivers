# Turing.jl

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{Turing.jl}$.

## Semantics

Turing.jl's denotational semantics is the trace semantics of Ge,
Xu, and Ghahramani
([2018](https://proceedings.mlr.press/v84/ge18b.html)) implemented
via the `@model` macro that rewrites `~`-statements into joint
log-density accumulator updates. The log-density probe is
[`Turing.logjoint(model, θ)`](https://turinglang.org/Turing.jl/dev/api/Inference/#Turing.logjoint),
returning the joint log-density at a parameter point.

## Unconstrained-space change of variables

Turing applies per-distribution
[Bijectors.jl](https://github.com/TuringLang/Bijectors.jl)
transformations during inference; `logjoint` returns the
constrained-space density. $\Psi_{\mathsf{Turing}} = \mathrm{id}$
at the renderer level.

## Family parameterizations

Turing consumes
[Distributions.jl](https://juliastats.org/Distributions.jl/) which
uses the canonical parameterizations of
[Wikipedia: List of probability distributions](https://en.wikipedia.org/wiki/List_of_probability_distributions).
The QVR ↔ Turing mapping is identity for every family with a
Distributions.jl counterpart. $\pi_{F, \mathsf{Turing}} =
\mathrm{id}$ and $c_{F, \mathsf{Turing}} = 0$.

The Turing renderer composes `HalfNormal` / `HalfCauchy` (which
have no Distributions.jl primitive) as `truncated(Normal(0,
sigma), 0, Inf)` via the documented
[`truncated`](https://juliastats.org/Distributions.jl/stable/truncate/)
wrapper, contributing $c_{\mathrm{HalfNormal}, \mathsf{Turing}} =
\log 2$ to the per-program constant.

## Per-construct emit

**Sample / observe.** `<name> ~ <Family>(<args>)` inside the
`@model function model(...)` body; observations appear as function
parameters that bind to specific values when `logjoint` is called.

**Plate.** [`filldist(D, B)`](https://turinglang.org/Turing.jl/dev/api/Distributions/#Distributions.filldist)
for index-independent batches; [`arraydist([D_i for i in 1:B])`](https://turinglang.org/Turing.jl/dev/api/Distributions/#Distributions.arraydist)
for index-dependent batches. Both denote the documented product
measure (Turing.jl documentation, "Composing distributions").

**Marginalize.** Explicit-latent rewrite. Turing.jl's HMC samplers
natively handle discrete latents.

**Score / let / return.** [`Turing.@addlogprob!(expr)`](https://turinglang.org/Turing.jl/dev/api/Distributions/#Turing.@addlogprob!)
for score; native `<name> = <expr>` for let; native `return` for
return.

## Acceptance

* **Tier 1 structural.** Every emit has `@model function model(...)
  ... end` with `~` statements wrapping `filldist` / `arraydist`
  per plate axis.
* **Tier 2 lens-laws.** Composition law holds.
* **Tier 3 external syntax.** `julia --eval "Meta.parse(read(stdin, String); raise=true)"`
  accepts every emit.
* **Tier 4 numeric equivalence.** `Turing.logjoint` evaluated at
  256 grid points + corners agrees with the QVR reference within
  $10^{-6}$.
