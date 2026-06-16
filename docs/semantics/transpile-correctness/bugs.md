# BUGS

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{BUGS}$.

## Semantics

BUGS's denotational semantics is the directed graphical model
semantics of Lunn, Spiegelhalter, Thomas, and Best
([2009](https://doi.org/10.1002/sim.3680)). A `model { ... }`
block declares a set of stochastic nodes (`~`) and deterministic
nodes (`<-`) over which the joint distribution factors. The
runtime is WinBUGS or MultiBUGS;
the log-density probe is the Gibbs sampler's per-iteration log-
joint accumulator.

## Unconstrained-space change of variables

BUGS works on the constrained parameter space directly; no
automatic reparametrization. $\Psi_{\mathsf{BUGS}} = \mathrm{id}$.

## Family parameterizations

BUGS uses precision-parameterized normal-family distributions:
`dnorm(μ, τ)` where $\tau = 1/\sigma^2$. This is the canonical
**parameterization substitution** worked through in
[head §5.1.1](index.md#51-per-family-density-preservation): the
algebraic equivalence

$$
\sqrt{\tau/(2\pi)} \exp\!\left(-\tau(v-\mu)^2/2\right)
\;=\;
(2\pi\sigma^2)^{-1/2} \exp\!\left(-(v-\mu)^2/(2\sigma^2)\right)
$$

under $\tau = 1/\sigma^2$ certifies $c_{\mathrm{Normal},
\mathrm{BUGS}} = 0$. The same calculation applies to `dlnorm` (log-
normal), `dt` (Student-t), and `dmnorm` (multivariate normal with
precision matrix $\Omega = \Sigma^{-1}$). The renderer applies
the substitution via the
[`FAMILY_META`](../../api/transpile/family_meta.md) arg_aliases
table (`{"scale": "tau"}`) plus a per-alias arithmetic transform
that wraps the scale arg in `1/(<scale>*<scale>)`. Cf.
[Architecture §10.4](../transpile-architecture.md).

Other parameterization differences:

| QVR family | BUGS call | $\pi_{F, \mathsf{BUGS}}$ |
|---|---|---|
| `Normal(μ, σ)` | `dnorm(μ, τ)` | $\sigma \mapsto 1/\sigma^2$ |
| `Cauchy(μ, γ)` | `dt(μ, 1/γ², 1)` | StudentT-with-1df with precision |
| `Laplace(μ, b)` | `ddexp(μ, 1/b)` | rate = $1/b$ |
| `MultivariateNormal(μ, Σ)` | `dmnorm(μ, Σ⁻¹)` | precision matrix |
| `Dirichlet(α)` | `ddirch(α)` | identity |
| `Categorical(p)` | `dcat(p)` | identity |
| `Bernoulli(p)` | `dbern(p)` | identity |
| `LogNormal(μ, σ)` | `dlnorm(μ, 1/σ²)` | precision |
| `StudentT(ν, μ, σ)` | `dt(μ, 1/σ², ν)` | precision |
| `Exponential(λ)` | `dexp(λ)` | identity |
| `Gamma(α, β)` | `dgamma(α, β)` | identity |
| `Beta(α, β)` | `dbeta(α, β)` | identity |
| `Pareto(α, x_m)` | `dpar(α, x_m)` | identity |
| `Weibull(k, λ)` | `dweib(k, 1/λ^k)` | scale to its $k$-th power |
| `Uniform(a, b)` | `dunif(a, b)` | identity |
| `InverseGamma(α, β)` | inverse-transform of `dgamma(α, β)` (composed) | identity on shape, $1/x$ on rate |

Every entry has $c_{F, \mathsf{BUGS}} = 0$ after substitution.

## Per-construct emit

**Sample / observe.** `<name> ~ d<family>(<args>)` inside the
`model { ... }` block. Latent vs observed is inferred from
whether `<name>` appears on the data side: BUGS programs ship
their data as a separate `list(...)` block, and variables not in
the data block are stochastic latents.

**Plate.** `for (m_<axis> in 1:N_<axis>) { <name>[m_<axis>] ~
d<family>(<args>) }`. Per Lunn et al. 2009, the for-loop's
contribution to the joint is the product of per-iteration
factors. Each row's args may index into other plate-shaped nodes
(LDA's `theta[doc[n], 1:K]` form).

**Marginalize.** Explicit-latent rewrite. BUGS Gibbs sampling
natively handles discrete latents.

**Score / let / return.** BUGS has no native `factor` primitive;
the renderer uses the
[zero-trick of Plummer 2003](https://www.r-project.org/conferences/DSC-2003/Proceedings/Plummer.pdf):
`_zero_<name> <- 0; _zero_<name> ~ dpois(-(<name>))` adds
$-\log(- \cdot) - (- \cdot) = -\log(-\mathrm{expr}) + \mathrm{expr}$
to the joint (since `dpois(0 | λ) = e^{-λ}`); the convention is
adjusted at emit time so the contribution is the documented
`<name>` value. Deterministic let-bindings emit as `<name> <-
<expr>`. Return is a model-level concept: BUGS returns the joint
sample of all stochastic nodes; the renderer does not emit a
target-language return statement.

**Limitations.** The BUGS renderer's `IRDeterministic` and
`IRScore` handlers raise `UnsupportedConstruct(["node:IRDeterministic"])`
/ `node:IRScore` for non-trivial expression bodies; the
implementation is staged for a follow-up.

## Acceptance

* **Tier 1 structural.** Every emit has `model { ... }` with `~`
  and `<-` statements wrapped in `for` loops per plate axis.
* **Tier 2 lens-laws.** Composition law holds.
* **Tier 3 external syntax.** `multibugs --check <file>` accepts
  every emit (when MultiBUGS is installed); structural-only
  acceptance otherwise.
* **Tier 4 numeric equivalence.** BUGS does not expose a
  parameterized log-density probe; the Tier-4 check uses
  posterior-summary comparison instead, with a wider tolerance.

### References

* David Lunn, David Spiegelhalter, Andrew Thomas, and Nicky Best.
  2009. The BUGS project: Evolution, critique and future
  directions. *Statistics in Medicine*, 28(25):3049-3067.
  [https://doi.org/10.1002/sim.3680](https://doi.org/10.1002/sim.3680)
* Martyn Plummer. 2003. JAGS: A program for analysis of Bayesian
  graphical models using Gibbs sampling. In *Proceedings of the
  3rd International Workshop on Distributed Statistical Computing
  (DSC)*, 124-125.
  [https://www.r-project.org/conferences/DSC-2003/Proceedings/Plummer.pdf](https://www.r-project.org/conferences/DSC-2003/Proceedings/Plummer.pdf)
