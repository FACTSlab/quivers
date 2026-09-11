# JAGS

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{JAGS}$.

## Semantics

JAGS's denotational semantics is the directed graphical model
semantics of Plummer
([2003](https://www.r-project.org/conferences/DSC-2003/Proceedings/Plummer.pdf)),
inheriting BUGS's `model { ... }` block syntax (Lunn et al. 2009
[BUGS](bugs.md)) with extensions. The runtime is JAGS itself or
the [`pyjags`](https://github.com/michaelnowotny/pyjags) Python
binding; the log-density probe is JAGS's Gibbs sampler log-joint
accumulator.

The renderer targets the BUGS-like subset accepted by JAGS; its
parameter substitutions and score zero trick follow the BUGS route.

## Unconstrained-space change of variables

Identity, as for BUGS. $\Psi_{\mathsf{JAGS}} = \mathrm{id}$.

## Family parameterizations

JAGS shares BUGS's precision-parameterized normal family
(`dnorm(μ, τ)` with $\tau = 1/\sigma^2$), the same `dlnorm`,
`dt`, `dmnorm` precision conventions, and the same calculation
showing $c_{F, \mathsf{JAGS}} = 0$ for every family. See the
[BUGS page](bugs.md) for the full parameterization table; the
JAGS family-name differences are:

| QVR family | JAGS call | Note |
|---|---|---|
| `Dirichlet(α)` | `ddirich(α)` | (BUGS uses `ddirch`) |
| `Gamma(α, β)` | `dgamma(α, β)` | matches BUGS |
| `Bernoulli(p)` | `dbern(p)` | matches BUGS |
| `Categorical(p)` | `dcat(p)` | matches BUGS |
| `Normal(μ, σ)` | `dnorm(μ, 1/σ²)` | precision parameterization |

JAGS also exposes
[`dgen.gamma`](https://justinribarski.github.io/JAGS/), `pow`,
and `inprod` primitives that the BUGS renderer does not target.
The QVR renderer does not depend on these but they remain
available to user-provided extensions.

## Per-construct emit

Same as BUGS. The renderer differs from `BUGSRenderer` only in
the `FAMILY_META.target_names["jags"]` lookup (returning
`ddirich` instead of `ddirch` for Dirichlet, for instance) and in
applying `FAMILY_META.arg_aliases["jags"]` separately from
`arg_aliases["bugs"]` (the entries are identical in current
practice but the registry pattern admits divergence).

**Sample / observe / plate / marginalize / score / let.** Same
shape as BUGS; the only difference is the distribution name.

## Acceptance

* **Tier 1 structural.** Same shape as BUGS.
* **Tier 1 pipeline composition.** Direct and composed pipeline calls agree.
* **Tier 2 external syntax.** The test writes a JAGS command file,
  asks JAGS to load the model, and rejects compiler error output.
* **Tier 3 numeric equivalence.** The `pyjags` probe evaluates
  selected emitted densities under the constant-spread criterion.

### References

* Martyn Plummer. 2003. JAGS: A program for analysis of Bayesian
  graphical models using Gibbs sampling. In *Proceedings of the
  3rd International Workshop on Distributed Statistical Computing
  (DSC)*, 124-125.
  [https://www.r-project.org/conferences/DSC-2003/Proceedings/Plummer.pdf](https://www.r-project.org/conferences/DSC-2003/Proceedings/Plummer.pdf)
