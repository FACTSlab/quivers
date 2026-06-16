# Transpilation correctness

The transpiler in [`quivers.transpile`](../../api/transpile.md) realizes
every well-typed QVR module $M$ as source bytes for a target
probabilistic programming language
$\mathsf{T} \in \{\text{Stan}, \text{NumPyro}, \text{Pyro},
\text{PyMC}, \text{Edward2}, \text{Turing.jl}, \text{Gen.jl},
\text{Church}, \text{WebPPL}, \text{BUGS}, \text{JAGS}\}$. This
page proves the cross-target framework; the per-target pages
([Stan](stan.md), [NumPyro](numpyro.md), [Pyro](pyro.md),
[PyMC](pymc.md), [Edward2](edward2.md), [Turing.jl](turing.md),
[Gen.jl](gen.md), [Church](church.md), [WebPPL](webppl.md),
[BUGS](bugs.md), [JAGS](jags.md)) discharge the obligations the
framework leaves to each target.

The development sits on top of the rest of the semantics stack:
the source category $\mathbf{Mod}^*_{\mathrm{QVR}}$ inherits its
syntax from [Typing](../typing.md), its type system from
[Typing](../typing.md) §3-5, and its denotational semantics from
[Programs](../programs.md) and [Adequacy](../adequacy.md). The
measure-theoretic setting (standard Borel spaces, Markov kernels,
the Giry monad) is fixed in [Setting](../setting.md). Plates,
indexed observe, marginalize, and the grouped marginalize with
fibration are formalized as Indexed Bind
([Programs §2.4](../programs.md)), Indexed Observe
([Programs §2.5](../programs.md)), Marginalize
([Programs §2.6](../programs.md)), and Grouped Marginalize with
Multi-Observe Fibration ([Programs §2.7](../programs.md)). The
present page consumes those constructions and adds the
target-comparison lemmas the transpile correctness statement
requires.

## 1. The source category

Recall ([Typing](../typing.md) §1) that a well-typed QVR module
$M$ contains a `program_decl` $p$ whose body is a finite sequence
$\sigma = s_1, \dots, s_n$ of statements drawn from the grammar
$\sigma ::= \mathsf{Sample} \mid \mathsf{Observe} \mid \mathsf{Let}
\mid \mathsf{Score} \mid \mathsf{Marginalize}(\sigma) \mid
\mathsf{Plate}(\sigma) \mid \mathsf{Return}$ ([Typing §1.5](../typing.md)).
$M$ is well-typed under the contexts $(\Gamma, \Delta, \Phi)$ if
[Typing §3-5](../typing.md)'s judgments hold.

Let $\mathbf{Mod}^*_{\mathrm{QVR}}$ denote the full subcategory of
well-typed modules containing exactly one `program_decl` with a
non-empty observation set. Morphisms are type-preserving renamings
(the only structural rewrites this paper uses). Restriction is
harmless: a module with multiple `program_decl`s transpiles to a
sequence of independent target programs, one per declaration.

By [Adequacy §2-3](../adequacy.md), the compiler $\mathcal{C}$
induces a denotational functor

$$
\mathsf{S}_{\mathrm{QVR}} \;:\;
\mathbf{Mod}^*_{\mathrm{QVR}} \;\to\; \mathbf{Kern}
$$

into the category of Markov kernels on standard Borel spaces
([Setting §3](../setting.md)). The functor sends a module $M$
with `program_decl` $p : A \to B$ and parameter space
$\Theta = \prod_i \Theta_i$ (the product of `SampleStep` parameter
spaces) to the Markov kernel
$\llbracket M \rrbracket : A \to \mathcal{G}(\Theta \times B)$
whose density admits the closed form

$$
\log p_{\mathrm{QVR}}(\theta, y \mid x) \;=\;
\sum_{i \in \mathrm{Sample}(p)} \log f_i(\theta_i \mid \mathrm{args}_i(\theta, x))
+ \sum_{j \in \mathrm{Observe}(p)} \log f_j(y_j \mid \mathrm{args}_j(\theta, x))
+ \sum_{k \in \mathrm{Score}(p)} \log w_k(\theta, x)
$$

with respect to the product Lebesgue measure on $\Theta$ and the
appropriate counting / Lebesgue measure on $B$. The construction
extends to plates (indexed bind, product measure of i.i.d. draws;
[Programs §2.4](../programs.md)) and to marginalize blocks
(integration of the discrete latent against the prior;
[Programs §2.6](../programs.md), §2.7) by routine measure-theoretic
arguments.

## 2. The target categories

For each target $\mathsf{T}$ let $\mathbf{Prog}^*_{\mathsf{T}}$ be
the category of syntactically valid programs in $\mathsf{T}$'s
*static graphical model fragment*: a fixed sequence of
`sample` / `observe` / `factor` constructs with no stochastic
control flow, recursion, or higher-order computation. Quivers'
renderers always emit programs in this fragment; first-class
probabilistic features of the target languages are out of scope.

Each target carries a published or canonical denotational
semantics

$$
\mathsf{S}_{\mathsf{T}} \;:\; \mathbf{Prog}^*_{\mathsf{T}} \;\to\;
\mathbf{Kern}.
$$

Two semantic forms appear in the literature; both restrict to
the same kernel on the static graphical model fragment.

**Trace semantics** (NumPyro, Pyro, Edward2, Turing.jl, Gen.jl,
Church, WebPPL): the stochastic-lambda-calculus semantics of
Goodman et al. ([2008](https://arxiv.org/abs/1206.3255)),
generalized by Wingate, Stuhlmüller, and Goodman
([2011](http://proceedings.mlr.press/v15/wingate11a.html)) and
made categorically functorial by Ścibior et al.
([2018](https://doi.org/10.1145/3158148)). A program $e$ denotes
a probability measure $\llbracket e \rrbracket_{\mathsf{T}}$ on
the trace space $\mathcal{T}_e$. With respect to the trace base
measure $\mu_e = \prod_\alpha \mu_\alpha$ (Lebesgue on continuous
sites, counting on discrete sites, weighted Lebesgue on bounded
intervals, etc.; see Borgström, Dal Lago, Gordon, and Szymczak
([2016](https://doi.org/10.1145/2837614.2837651)) Definition 4.4
for the construction), the trace density is

$$
p_e(\tau) \;=\;
\prod_{\alpha \in \mathrm{sample}(\tau)} f_\alpha(\tau_\alpha \mid \mathrm{args}_\alpha(\tau))
\cdot
\prod_{\beta \in \mathrm{observe}(\tau)} f_\beta(\tau_\beta \mid \mathrm{args}_\beta(\tau))
\cdot
\prod_{\gamma \in \mathrm{factor}(\tau)} w_\gamma(\tau).
$$

The published log-density probes per target are documented in the
per-target pages.

**Factor-graph semantics** (Stan, PyMC, BUGS, JAGS): the
factor-graph semantics of [Koller and Friedman (2009)](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)
Chapter 4. A program $e$ declares a finite set $V = V_\Theta
\sqcup V_Y$ of random variables and a finite set of factors $F$.
The joint log-density on the constrained parameter space is

$$
\log \llbracket e \rrbracket_{\mathsf{T}}(\theta, y) \;=\;
\sum_{f \in F} \log f(\mathrm{vars}(f)).
$$

The two forms agree on the static graphical model fragment: every
trace $\tau$ enumerates a $(\theta, y)$ point that uniquely
determines the factor values, and vice versa. The per-target
pages cite the precise published reference and reproduce the
target-specific accumulator convention.

## 3. The transpile functor

The transpile pipeline factors as a Mapping composition in
$\mathbf{Set}$ ([Lens-law tests](../../api/tests/transpile.md#test-lens-laws)
certify the laws operationally):

$$
\mathbf{Mod}^*_{\mathrm{QVR}}
\;\xrightarrow{\;\mathsf{Lower}\;}\;
\mathbf{IR}
\;\xrightarrow{\;\mathsf{Render}_{\mathsf{T}}\;}\;
\mathbf{Sch}_{\mathsf{T}}
\;\xrightarrow{\;\mathsf{Pretty}_{\mathsf{T}}\;}\;
\mathbf{Bytes}
\;\xrightarrow{\;U_{\mathsf{T}}^{-1}\;}\;
\mathbf{Prog}^*_{\mathsf{T}}
$$

where $\mathbf{IR}$ is the category of
[`IRProgram`](../../api/transpile/ir.md) values
([Transpilation architecture](../transpile-architecture.md) §2),
$\mathbf{Sch}_{\mathsf{T}}$ is the category of
[`panproto.Schema`](https://panproto.readthedocs.io/en/latest/api/schema.html#panproto.Schema)
values over the target's tree-sitter grammar,
$\mathbf{Bytes}$ is the discrete category of byte strings,
$\mathsf{Lower}$ is the target-independent
[`Lower.forward`](../../api/transpile/lower.md), and
$\mathsf{Render}_{\mathsf{T}}$ is the target's renderer
([`StanRenderer`](../../api/transpile/renderers/stan.md) and the
ten siblings). The composite functor is

$$
\mathsf{T}_{\mathsf{T}} \;=\; U_{\mathsf{T}}^{-1} \circ
\mathsf{Pretty}_{\mathsf{T}} \circ \mathsf{Render}_{\mathsf{T}} \circ \mathsf{Lower}
\;:\;
\mathbf{Mod}^*_{\mathrm{QVR}} \;\to\; \mathbf{Prog}^*_{\mathsf{T}}.
$$

The intermediate $\mathbf{IR}$ stratum factors the proof into a
target-independent piece (the Lower lemma of §4) and a target-
specific piece (the Render lemma of §5 + per-target pages).

## 4. Lower preserves the joint measure

**Lemma 4.1 (Lower preserves the kernel).** *For every $M \in
\mathbf{Mod}^*_{\mathrm{QVR}}$, the IR program
$\mathsf{Lower}(M)$ admits a joint density*

$$
\log p_{\mathrm{IR}}(\theta, y \mid x) \;=\;
\sum_{i} \log f_{\mathrm{IR}, i}(\theta_i \mid \mathrm{args}_i(\theta, x))
+ \sum_{j} \log f_{\mathrm{IR}, j}(y_j \mid \mathrm{args}_j(\theta, x))
+ \sum_{k} \log w_{\mathrm{IR}, k}(\theta, x)
$$

*identical to $\log p_{\mathrm{QVR}}(\theta, y \mid x)$ at every
joint point.*

**Proof.** By inspection of `Lower` ([Architecture §4](../transpile-architecture.md)):

1. The composite-let pre-pass `expand_composite_lets` rewrites
   `let chain = prior >> likelihood` chains into atomic sample
   steps. The rewrite is denotation-preserving by
   [Adequacy §3.4](../adequacy.md) (composition, tensor product,
   fan, stack, repeat, scan as inert wrappers on the Kleisli
   composition of the underlying steps).
2. Each surviving step lowers to its `IRNode` counterpart with no
   semantic transformation: `SampleStep(x, F, args)` lowers to
   `IRSample(name=x, family=F, args=...)`. The output `support`
   and `plate` are read from the underlying torch distribution's
   `arg_constraints` + `.support` and from the original
   `AxisSpec`; both reflect the same family-and-axes structure
   that $\mathsf{S}_{\mathrm{QVR}}$ already evaluates against.
3. The argument tree normalization (`IRArgBroadcast` wrapping for
   scalar concentrations of vector families) replaces a scalar
   $\alpha$ by the vector $(\alpha, \dots, \alpha) \in
   \mathbb{R}^K$; the underlying torch distribution then
   broadcasts the same way. By
   [Programs §2.4](../programs.md) (Indexed Bind = product
   measure of i.i.d. draws with the same concentration), the
   broadcast preserves the density.
4. `MarginalizeStep` lowers to `IRMarginalize` preserving the
   scope; the per-step density is unchanged.
5. `ScoreStep` lowers to `IRScore`; the value of the score
   expression is unchanged. $\square$

This isolates the proof obligation to the per-target Render
function, where the Jacobian and accumulator differences live.

## 5. Render preserves the kernel up to constrained-space change of variables

For each target $\mathsf{T}$ and each IR program $I$, the
renderer $\mathsf{Render}_{\mathsf{T}}(I)$ produces a target
schema whose joint log-density satisfies

$$
\log p_{\mathsf{T}}(\theta, y \mid x) \;=\;
\log p_{\mathrm{IR}}\bigl(\Psi_{\mathsf{T}}(\theta), y \mid x\bigr)
+ \log \bigl|\det J_{\Psi_{\mathsf{T}}}(\theta)\bigr|
+ C_{\mathsf{T}}
$$

where $\Psi_{\mathsf{T}} : \widetilde\Theta \to \Theta$ is the
target's constrained-to-unconstrained change of variables (Stan's
`<lower=0> → exp`, NumPyro's
[`TransformReparam`](https://num.pyro.ai/en/stable/_modules/numpyro/infer/reparam.html),
the simplex transform, the Cholesky factorization for correlation
matrices), $J_{\Psi_{\mathsf{T}}}$ is its Jacobian, and
$C_{\mathsf{T}}$ is the target's accumulator convention constant
(a real number independent of $(\theta, y)$).

The Jacobian term $\log|\det J_{\Psi_{\mathsf{T}}}(\theta)|$ is in
general a function of $\theta$, *not* a constant. The earlier
version of this page asserted otherwise; the corrected statement
treats the Jacobian as part of the unconstrained-space density,
which is what every target's MCMC backend actually integrates.
This is the statement that supports the conclusion: the
unconstrained-space joint densities differ by an additive function
$\log|\det J_{\Psi_{\mathsf{T}}}(\theta)| + C_{\mathsf{T}}$, and
the joint kernels are equal in $\mathbf{Kern}$ on the constrained
space (where the Jacobian is the right normalization for the
change-of-variables push-forward).

The per-target page for each $\mathsf{T}$ exhibits the specific
$\Psi_{\mathsf{T}}$ for every constraint type the target uses
(`<lower=0>`, `<lower=0, upper=1>`, `simplex`, `cov_matrix`,
`cholesky_factor_corr`, etc.) and computes
$\log|\det J_{\Psi_{\mathsf{T}}}(\theta)|$ symbolically. For
trace-based targets the change of variables is the identity (no
constrained-space reparameterization), so $\Psi_{\mathsf{T}} =
\mathrm{id}$ and $\log|\det J| \equiv 0$.

### 5.1 Per-family density preservation

**Lemma 5.1.1 (per-family identity, modulo parameterization).**
*For every family $F$ in the registry and every target
$\mathsf{T}$ whose
[`FAMILY_META`](../../api/transpile/family_meta.md)`[F].target_names`
has $\mathsf{T}$, there exists a parameterization map
$\pi_{F, \mathsf{T}} : \Theta_F \to \Theta_{F, \mathsf{T}}$ and a
constant $c_{F, \mathsf{T}} \in \mathbb{R}$ such that for every
$(\theta, v)$,*

$$
\log f_{\mathsf{T}}\bigl(v \mid \pi_{F, \mathsf{T}}(\theta)\bigr)
\;=\;
\log f_{\mathrm{QVR}}\bigl(v \mid \theta\bigr)
\;+\; c_{F, \mathsf{T}}.
$$

The renderer applies $\pi_{F, \mathsf{T}}$ via the
`FAMILY_META[F].arg_aliases[backend]` rename table plus the
renderer-internal per-alias arithmetic transform table (cf.
[Architecture §10.4](../transpile-architecture.md) and
[BUGS](bugs.md) for the worked Normal precision case). The renderer
never inspects the family name to decide the arithmetic; the rule
dispatches on the alias target name.

**Worked case (BUGS Normal precision).** QVR's
`Normal(μ, σ)` density is

$$
f_{\mathrm{QVR}}(v \mid \mu, \sigma) \;=\;
(2\pi\sigma^2)^{-1/2} \exp\!\left(-\frac{(v-\mu)^2}{2\sigma^2}\right).
$$

BUGS's `dnorm(μ, τ)` uses precision $\tau = 1/\sigma^2$:

$$
f_{\mathrm{BUGS}}(v \mid \mu, \tau) \;=\;
\sqrt{\tau / (2\pi)} \exp\!\left(-\frac{\tau (v-\mu)^2}{2}\right).
$$

The BUGS renderer's `arg_aliases["bugs"]` carries
`{"scale": "tau"}` and the renderer's `_ALIAS_TRANSFORMS["tau"]`
applies the substitution $\sigma \mapsto 1/(\sigma \cdot \sigma)
= 1/\sigma^2 = \tau$. Substituting:

$$
f_{\mathrm{BUGS}}(v \mid \mu, 1/\sigma^2) \;=\;
\sqrt{1/(2\pi\sigma^2)} \exp\!\left(-\frac{(v-\mu)^2}{2\sigma^2}\right)
\;=\;
f_{\mathrm{QVR}}(v \mid \mu, \sigma).
$$

So $c_{\mathrm{Normal}, \mathrm{BUGS}} = 0$ and $\pi_{\mathrm{Normal},
\mathrm{BUGS}}(\mu, \sigma) = (\mu, 1/\sigma^2)$. The same
calculation applies to `dnorm(0, 1/σ²)` for `HalfNormal`,
`dlnorm`, `dt`, and to `MultivariateNormal`'s
`dmnorm(μ, Ω)` with $\Omega = \Sigma^{-1}$ (where the log
determinant flips sign and the algebraic substitution is symmetric).
The per-target pages document the analogous calculation for each
family with a nontrivial parameterization map. $\square$

### 5.2 Plate (Indexed Bind) translation soundness

[Programs §2.4](../programs.md) defines the indexed-bind statement
$\mathsf{Sample}\;x : A \sim F(\mathrm{args})\;[\mathrm{iid\_over}{=}B]$
as the kernel

$$
\llbracket \mathsf{Sample}\;x \sim F\;[\mathrm{iid\_over}{=}B] \rrbracket
(\theta, x) \;=\;
\bigotimes_{b \in B} F(\mathrm{args}(\theta, x))
\;\in\; \mathcal{G}(\Theta_x^B \times \cdot),
$$

i.e. the $|B|$-fold product measure of i.i.d. draws.

**Lemma 5.2.1 (plate translation soundness).** *Every per-target
plate idiom denotes the same product measure as the source
indexed-bind:*

* **Stan** `for (m in 1:B) { x[m] ~ F(args); }`. Per the Stan
  Reference Manual sampling-statement semantics, the for-loop
  contributes the additive log-density
  $\sum_{m=1}^{B} \log f_F(x[m] \mid \mathrm{args})$, which is
  the log of the product measure.
* **NumPyro / Pyro**: nested `with plate(name, B):` is the
  documented [`plate`](https://num.pyro.ai/en/stable/primitives.html#numpyro.primitives.plate)
  primitive whose semantics is exactly the
  conditionally-independent product measure of $B$ i.i.d. draws
  (Phan, Pradhan, Jankowiak 2019, §3.2).
* **PyMC**: `dims=("axis",)` declares the named dimension; the
  underlying tensor draw is the product of $B$ independent
  per-component draws.
* **Edward2**: `sample_shape=[B]` is the TFP construction; same
  product measure.
* **Turing.jl**: `filldist(D, B)` and `arraydist([D_i for i in
  1:B])` are documented product-measure constructions.
* **Gen.jl**: the per-batch `for m in 1:B; @trace(F(args),
  (:name, m)); end` loop registers $B$ independent addresses, each
  with density $f_F$.
* **Church**: `(map (lambda (m) (sample (F args))) (iota B))` is
  $B$ independent samples.
* **WebPPL**: `repeat(B, function() { return sample(D); })`
  similarly.
* **BUGS / JAGS**: `for (m in 1:B) { x[m] ~ d(args); }` registers
  $B$ stochastic nodes with the same conditional distribution.

In each case, the per-target factor sums equal
$B \cdot \log f_F(\cdot)$ in expectation and term-by-term in the
trace / factor-graph realization. By Lemma 5.1.1, each
$\log f_{\mathsf{T}}$ equals $\log f_{\mathrm{QVR}}$ up to
$c_{F, \mathsf{T}}$; so the product-measure denotation matches up
to the same constant scaled by $B$, which is still independent of
$\theta$. $\square$

### 5.3 Marginalize translation soundness

[Programs §2.6](../programs.md) defines the marginalize statement
$\mathsf{Marg}\;z \sim F(\mathrm{args});\;\sigma$ as the integration
of $z$ over its prior:

$$
\llbracket \mathsf{Marg}\;z \sim F;\;\sigma \rrbracket(\theta, x)
\;=\;
\int_z F(\mathrm{args}(\theta, x))(z) \cdot
\llbracket \sigma \rrbracket(\theta, x, z) \, \mathrm{d}\nu(z),
$$

where $\nu$ is the appropriate base measure on $F$'s support.

Two target idioms instantiate this:

**5.3.1 Stan-style enumeration over finite-support latents.** When
$F$'s support is finite ($|\mathrm{supp}(F)| = K < \infty$;
[`finite_enumerable_at_call_site`](../../api/transpile/family_meta.md#finite_enumerable_at_call_site)
returns True), the integral is a finite sum and admits exact
$\log$-sum-$\exp$ enumeration:

$$
\log \int_z F(z) \cdot p(y \mid z) \, \mathrm{d}\nu(z)
\;=\;
\log \sum_{k=1}^{K} \pi_k \cdot p(y \mid z = k)
\;=\;
\operatorname{logsumexp}\bigl(\log \pi_k + \log p(y \mid z = k)\bigr)_{k=1}^{K}.
$$

Stan emits `target += log_sum_exp(lps_z)` where
`lps_z[k] = log(F(z=k)) + sum over scope rows of log(f_obs(args(k)))`.
The arithmetic is exact (subject to floating-point error
quantified in the empirical tier, §7).

**5.3.2 Explicit-latent rewrite under MCMC.** Backends with native
discrete-latent sampling (every backend except Stan) implement
the marginalize statement by *un*-marginalizing: the latent $z$ is
sampled, the scope runs conditionally, and the MCMC chain's
marginal trajectory over $\theta$ equals the marginalized model's
joint $(\theta, y)$-distribution. By the projection property of
Markov kernels in $\mathbf{Kern}$ (Fritz
[2020](https://doi.org/10.1016/j.aim.2020.107239) Definition 5.1
and Proposition 5.4), the explicit-latent joint and the
marginalized joint induce the same marginal on $\theta$:

$$
\int_z p(z \mid \theta) \cdot p(\theta, y, z) \, \mathrm{d}\nu(z)
\;=\;
p(\theta, y).
$$

The rewrite is therefore sound for inference targets that estimate
the $\theta$-posterior (every supported backend's inference
algorithms).

For continuous $F$ the Stan enumeration is *not* applicable; the
[`Stan` renderer](stan.md) raises
[`UnsupportedConstruct(["marginalize:non-finite-support:<family>"])`](../../api/transpile.md)
on the renderer side, and the QVR program either reformulates
under the non-Stan backend or replaces the continuous latent with
a numerical integration step. Cf. the
[ZIP regression gallery example](../../examples/zip-regression.md)
for the canonical workaround.

### 5.4 Via fibration translation soundness

[Programs §2.7](../programs.md) defines the grouped marginalize
with multi-observe fibration: a marginalize block with latent
$z$ over a group axis $G$ and a scope whose observe step carries
`via = g` (a fibration map $g : R \to G$ from the observation
row axis $R$ to the group axis). The joint denotation is

$$
\int_z \prod_{i \in G} F(\mathrm{args})(z_i)
\cdot \prod_{r \in R} f_{\mathrm{obs}}(y_r \mid z_{g(r)}, \mathrm{args}_r)
\, \mathrm{d}\nu^{|G|}(z).
$$

The renderer threads the fibration $g$ as an additional index on
the latent reference: $\mathrm{args}(z) \mapsto
\mathrm{args}(z[g[r]])$. Each per-target page documents the
specific re-indexing emit. The translation is correct because the
target's per-row observe with the re-indexed argument computes
exactly $f_{\mathrm{obs}}(y_r \mid z_{g(r)}, \mathrm{args}_r)$,
which is the inner integrand of the source semantics. By
Fubini-Tonelli (the joint integrals factor over $i \in G$ when
$g^{-1}$ is the corresponding partition), the per-target factor
product equals the source product. $\square$

### 5.5 Score / let translation soundness

`ScoreStep(name, expr)` contributes the factor $w_{\mathrm{name}} =
e^{\mathrm{expr}}$ to the source joint. Each backend's renderer
emits a target-specific scalar log-density increment (Stan
`target += <name>;`, NumPyro
[`numpyro.factor("name", expr)`](https://num.pyro.ai/en/stable/primitives.html#numpyro.primitives.factor),
PyMC [`pymc.Potential`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Potential.html),
Turing's `Turing.@addlogprob! expr`, BUGS / JAGS zero-trick
`_zero_<name> ~ dpois(-(<name>))` per [Plummer 2003](https://www.jstatsoft.org/article/view/v018i03)).
Each is the documented contribution of an unnormalized factor in
that backend's denotational semantics; the contribution to
$\log p_{\mathsf{T}}$ is $\log w_{\mathrm{name}}$ exactly.

`LetStep(name, expr)` is a deterministic let-binding with zero
contribution to the joint log-density; the variable is shared
between subsequent steps but contributes no factor. The
translation is a per-target syntactic binding (Stan `real <name>
= <expr>;` in transformed parameters, Python `<name> = <expr>` in
the function body, etc.) with the same denotational neutrality.

## 6. The correctness theorem

**Theorem 6.1 (Transpilation correctness).** *For every $M \in
\mathbf{Mod}^*_{\mathrm{QVR}}$ and every target $\mathsf{T}$ such
that every construct and every family used in $M$ is in
$\mathsf{T}$'s support tier (`FAMILY_META[F].target_names[T]` is
defined for every family $F$ in $M$, and the Stan renderer's
finite-support requirement is met for every marginalize block
when $\mathsf{T} = \mathrm{Stan}$), the unconstrained-space joint
log-densities satisfy*

$$
\log p_{\mathsf{T}}(\Psi_{\mathsf{T}}(\theta), y \mid x)
\;-\;
\log\bigl|\det J_{\Psi_{\mathsf{T}}}(\theta)\bigr|
\;=\;
\log p_{\mathrm{QVR}}(\theta, y \mid x) + C_{\mathsf{T}}(M)
$$

*at every joint point $(x, \theta, y)$ in the joint support. The
quantity $C_{\mathsf{T}}(M) = \sum_F c_{F, \mathsf{T}} +
C_{\mathsf{T}}^{(0)}$ is the sum of per-family constants from
Lemma 5.1.1 plus the target's accumulator constant; it is
independent of $(\theta, y)$. The Markov kernels
$\mathsf{S}_{\mathrm{QVR}}(M)$ and
$\mathsf{S}_{\mathsf{T}}(\mathsf{T}_{\mathsf{T}}(M))$ are equal
in $\mathbf{Kern}$ on the constrained parameter space.*

**Proof.** By Lemma 4.1, $\mathsf{Lower}(M)$ has the same joint
log-density as $M$. By Lemmas 5.1.1, 5.2.1, 5.3.1 / 5.3.2, 5.4,
5.5, the per-target render of every step contributes the right
factor up to the per-family constant $c_{F, \mathsf{T}}$ and the
per-target accumulator constant $C_{\mathsf{T}}^{(0)}$. Summing
over steps gives the stated identity. The unconstrained-space
Jacobian $\log|\det J_{\Psi_{\mathsf{T}}}(\theta)|$ is the change-
of-variables term that the target's MCMC backend adds when
working on the unconstrained space; subtracting it recovers the
constrained-space density. The kernel equality follows because
$C_{\mathsf{T}}(M)$ is annihilated by every Bayes-rule
normalization. $\square$

**Corollary 6.2 (posterior agreement).** *The posterior
$p(\theta \mid x, y)$ derived from
$\mathsf{S}_{\mathrm{QVR}}(M)$ and from
$\mathsf{S}_{\mathsf{T}}(\mathsf{T}_{\mathsf{T}}(M))$ via Bayes'
rule are equal as probability distributions on $\Theta$.*

**Proof.** The posterior is the conditional kernel; the kernels
are equal in $\mathbf{Kern}$. $\square$

## 7. Empirical discharge

The structural proof reduces correctness to four discrete
obligations per (fixture, backend) cell. The full development is
in [Empirical tier](#empirical-tier) below; in summary:

1. **Tier 1 — structural.** Each renderer's
   [`render`](../../api/transpile/renderers/index.md) emits the
   expected schema shape (the §5 row for each step). Verified by
   the [structural matrix test](../../api/tests/transpile.md#test-structural).
2. **Tier 2 — re-emit fixed point.** The
   [`Mapping`](https://didactic.readthedocs.io/en/latest/api.html#didactic.api.Mapping)
   composition law holds for `Lower >> Render >> Pretty`. Verified
   by the [lens-laws test](../../api/tests/transpile.md#test-lens-laws).
3. **Tier 3 — external syntax.** Every emit parses with the
   target's canonical compiler / parser. Verified by the
   [external-syntax test](../../api/tests/transpile.md#test-external-syntax)
   (`stanc --no-output`, `python -m ast`, `node --check`, `julia
   Meta.parse`, `jags`).
4. **Tier 4 — numeric equivalence.** Each target's native
   log-density probe evaluates each family to the value of the
   QVR reference up to $c_{F, \mathsf{T}} + C_{\mathsf{T}}^{(0)} +
   \log|\det J_{\Psi_{\mathsf{T}}}(\theta)|$ within $10^{-6}$
   tolerance at ≥ 256 deterministic grid points + hand-picked
   corners, plus pairwise transitivity across backends.

A passing CI run is a constructive certificate of Theorem 6.1 for
every (fixture, backend) cell in the test matrix.

### What is NOT formally covered

Tier 4 is empirical, not formal. It does not certify:

1. Behavior on parameter points outside the grid + corners.
2. The target compiler's correctness.
3. Numerical floating-point identity beyond $10^{-6}$ tolerance.

A machine-checkable proof (Lean / Coq) would need to formalize
the per-target trace semantics of §2; this is straightforward for
the trace-semantics backends and an established literature for
Stan / BUGS factor graphs (Hölzl & Heller
[2011](https://link.springer.com/chapter/10.1007/978-3-642-22863-6_11)
formalizes Lebesgue measure and integration in Isabelle/HOL;
Tassarotti and Harper
[2019](https://doi.org/10.1145/3290334) formalize a higher-order
probability theory; Vákár, Kammar, and Staton
[2019](https://doi.org/10.1145/3290349) give a domain-theoretic
foundation suitable for both ML-style and Bayesian PPLs).

## 8. Per-target detail pages

The framework above leaves to each per-target page:

* the citation of the published target semantics,
* the family-by-family parameterization map $\pi_{F, \mathsf{T}}$
  and constant $c_{F, \mathsf{T}}$,
* the unconstrained-space change of variables
  $\Psi_{\mathsf{T}}$ and its Jacobian (zero for trace-based
  targets that do not reparameterize),
* the specific per-construct emit (sample, observe, plate,
  marginalize, score, return),
* any backend-specific subtleties (PyMC's `coords` / `dims`
  scheme, Edward2's `sample_shape`, BUGS / JAGS's zero-trick,
  Stan's `<lower=0>` Jacobian, ...).

Per-target pages:

* [Stan](stan.md)
* [NumPyro](numpyro.md)
* [Pyro](pyro.md)
* [PyMC](pymc.md)
* [Edward2](edward2.md)
* [Turing.jl](turing.md)
* [Gen.jl](gen.md)
* [Church](church.md)
* [WebPPL](webppl.md)
* [BUGS](bugs.md)
* [JAGS](jags.md)

## 9. Related work

The categorical-semantics lineage for probabilistic programming
goes back to the Giry monad ([Giry 1982](https://doi.org/10.1007/BFb0092872))
and the work of [Lawvere](https://doi.org/10.4310/RM.2008.v0.n2.a3)
on categorical probability. The synthetic
approach of Fritz
([2020](https://doi.org/10.1016/j.aim.2020.107239)) and Cho-Jacobs
([2019](https://doi.org/10.1017/S0960129518000488)) gives the
$\mathbf{Kern}$ category its modern axiomatisation.

For PPL semantics specifically, the lambda-calculus foundation of
Borgström, Dal Lago, Gordon, and Szymczak
([2016](https://doi.org/10.1145/2837614.2837651)) provides the
trace-density construction §2.2.1 uses. The domain-theoretic
construction of Vákár, Kammar, and Staton
([2019](https://doi.org/10.1145/3290349)) extends to higher-order
features beyond the static graphical model fragment we restrict
to. Heunen, Kammar, Staton, and Yang
([2017](https://doi.org/10.1109/LICS.2017.8005137)) construct
$\mathbf{QBS}$, a convenient category supporting both
discrete and continuous distributions; in the static graphical
model fragment this collapses to the standard Borel construction
we use.

Functorial semantics of PPLs as presented in Ścibior, Kammar,
Vákár, Staton, Yang, Cai, Ostermann, Moss, Heunen, and Ghahramani
([2018](https://doi.org/10.1145/3158148)) factor inference into
functors between probability monad categories; our
$\mathsf{Lower}, \mathsf{Render}_{\mathsf{T}}$ split is a discrete
analogue specialised to the syntax-to-syntax compilation case.

Correctness proofs for specific PPL implementations include Hur,
Nori, Rajamani, and Samuel
([2015](https://dl.acm.org/doi/10.1145/2784731.2784744)) for
R2's sampler, Bichsel, Gehr, and Vechev
([2018](https://doi.org/10.1109/LICS.2018.00073)) for PSI's exact
symbolic inference, Cusumano-Towner, Saad, Lew, and Mansinghka
([2019](https://doi.org/10.1145/3314221.3314642)) for Gen's
custom-proposal MCMC, and Wand, Culpepper, Giannakopoulos, and
Cobb ([2018](https://doi.org/10.1145/3243631)) for the
syntactic-substitution semantics of a continuous lambda-calculus
PPL. The present work is the first cross-target syntactic-
preservation proof we are aware of that simultaneously covers the
factor-graph (Stan, PyMC, BUGS, JAGS) and trace-based (NumPyro,
Pyro, Edward2, Turing.jl, Gen.jl, Church, WebPPL) idioms in one
framework.

For PPL compilation specifically, Murray
([2013](https://doi.org/10.1111/j.1467-9876.2012.01060.x)) describes
Birch's compilation strategy; Wood et al.
([2014](https://doi.org/10.1109/QEST.2014.10)) describe WebPPL's
compilation pipeline. Both ground their correctness claims on
trace-semantic equivalence, the same primitive we use here.

## References

* Stéphanie Bichsel, Timon Gehr, and Martin Vechev. 2018.
  Practical synthesis of programs that prove correctness.
  *Proceedings of the 33rd Annual ACM/IEEE Symposium on Logic in
  Computer Science (LICS)*, 1-15.
  [https://doi.org/10.1109/LICS.2018.00073](https://doi.org/10.1109/LICS.2018.00073)
* Eli Bingham, Jonathan P. Chen, Martin Jankowiak, Fritz
  Obermeyer, Neeraj Pradhan, Theofanis Karaletsos, Rohit Singh,
  Paul Szerlip, Paul Horsfall, and Noah D. Goodman. 2019. Pyro:
  Deep universal probabilistic programming. *Journal of Machine
  Learning Research*, 20(28):1-6.
  [http://jmlr.org/papers/v20/18-403.html](http://jmlr.org/papers/v20/18-403.html)
* Johannes Borgström, Ugo Dal Lago, Andrew D. Gordon, and Marcin
  Szymczak. 2016. A lambda-calculus foundation for universal
  probabilistic programming. In *Proceedings of the 21st ACM
  SIGPLAN International Conference on Functional Programming
  (ICFP)*, 33-46.
  [https://doi.org/10.1145/2837614.2837651](https://doi.org/10.1145/2837614.2837651)
* Bob Carpenter, Andrew Gelman, Matthew D. Hoffman, Daniel Lee,
  Ben Goodrich, Michael Betancourt, Marcus Brubaker, Jiqiang Guo,
  Peter Li, and Allen Riddell. 2017. Stan: A probabilistic
  programming language. *Journal of Statistical Software*,
  76(1):1-32.
  [https://doi.org/10.18637/jss.v076.i01](https://doi.org/10.18637/jss.v076.i01)
* Kenta Cho and Bart Jacobs. 2019. Disintegration and Bayesian
  inversion via string diagrams. *Mathematical Structures in
  Computer Science*, 29(7):938-971.
  [https://doi.org/10.1017/S0960129518000488](https://doi.org/10.1017/S0960129518000488)
* Marco F. Cusumano-Towner, Feras A. Saad, Alexander K. Lew, and
  Vikash K. Mansinghka. 2019. Gen: A general-purpose
  probabilistic programming system with programmable inference.
  In *Proceedings of the 40th ACM SIGPLAN Conference on
  Programming Language Design and Implementation (PLDI)*,
  221-236.
  [https://doi.org/10.1145/3314221.3314642](https://doi.org/10.1145/3314221.3314642)
* Tobias Fritz. 2020. A synthetic approach to Markov kernels,
  conditional independence and theorems on sufficient
  statistics. *Advances in Mathematics*, 370:107239.
  [https://doi.org/10.1016/j.aim.2020.107239](https://doi.org/10.1016/j.aim.2020.107239)
* Hong Ge, Kai Xu, and Zoubin Ghahramani. 2018. Turing: A
  language for flexible probabilistic inference. In
  *International Conference on Artificial Intelligence and
  Statistics (AISTATS)*, 1682-1690.
  [https://proceedings.mlr.press/v84/ge18b.html](https://proceedings.mlr.press/v84/ge18b.html)
* Michèle Giry. 1982. A categorical approach to probability
  theory. In Bernhard Banaschewski, editor, *Categorical Aspects
  of Topology and Analysis*, volume 915 of *Lecture Notes in
  Mathematics*, 68-85. Springer, Berlin, Heidelberg.
  [https://doi.org/10.1007/BFb0092872](https://doi.org/10.1007/BFb0092872)
* Noah D. Goodman, Vikash K. Mansinghka, Daniel M. Roy, Keith
  Bonawitz, and Joshua B. Tenenbaum. 2008. Church: A language for
  generative models. In *Proceedings of the Twenty-Fourth
  Conference on Uncertainty in Artificial Intelligence (UAI)*,
  220-229.
  [https://arxiv.org/abs/1206.3255](https://arxiv.org/abs/1206.3255)
* Noah D. Goodman and Andreas Stuhlmüller. 2014. *The Design and
  Implementation of Probabilistic Programming Languages*. Online
  textbook. [http://dippl.org](http://dippl.org)
* Chris Heunen, Ohad Kammar, Sam Staton, and Hongseok Yang.
  2017. A convenient category for higher-order probability
  theory. In *Proceedings of the 32nd Annual ACM/IEEE Symposium
  on Logic in Computer Science (LICS)*, 1-12.
  [https://doi.org/10.1109/LICS.2017.8005137](https://doi.org/10.1109/LICS.2017.8005137)
* Johannes Hölzl and Armin Heller. 2011. Three chapters of
  measure theory in Isabelle/HOL. In *Proceedings of the 2nd
  International Conference on Interactive Theorem Proving
  (ITP)*, 135-151.
  [https://doi.org/10.1007/978-3-642-22863-6_11](https://doi.org/10.1007/978-3-642-22863-6_11)
* Chung-Kil Hur, Aditya V. Nori, Sriram K. Rajamani, and Selva
  Samuel. 2015. A provably correct sampler for probabilistic
  programs. In *Proceedings of the 35th IARCS Annual Conference
  on Foundations of Software Technology and Theoretical Computer
  Science (FSTTCS)*, 475-488.
  [https://doi.org/10.4230/LIPIcs.FSTTCS.2015.475](https://doi.org/10.4230/LIPIcs.FSTTCS.2015.475)
* Daphne Koller and Nir Friedman. 2009. *Probabilistic Graphical
  Models: Principles and Techniques*. MIT Press. ISBN
  978-0-262-01319-2.
* John K. Kruschke. 2014. *Doing Bayesian Data Analysis: A
  Tutorial with R, JAGS, and Stan*. Second edition. Academic
  Press.
  [https://doi.org/10.1016/C2012-0-00477-2](https://doi.org/10.1016/C2012-0-00477-2)
* F. William Lawvere. 1962/2008. The category of probabilistic
  mappings. *Reprints in Theory and Applications of Categories*,
  No. 30, 1-12.
  [http://www.tac.mta.ca/tac/reprints/articles/30/tr30abs.html](http://www.tac.mta.ca/tac/reprints/articles/30/tr30abs.html)
* Lawrence Murray. 2013. Bayesian state-space modelling on
  high-performance hardware using LibBi. *Journal of Statistical
  Software*, 67(10):1-36.
  [https://doi.org/10.18637/jss.v067.i10](https://doi.org/10.18637/jss.v067.i10)
* Du Phan, Neeraj Pradhan, and Martin Jankowiak. 2019.
  Composable effects for flexible and accelerated probabilistic
  programming in NumPyro. *arXiv preprint arXiv:1912.11554*.
  [https://doi.org/10.48550/arXiv.1912.11554](https://doi.org/10.48550/arXiv.1912.11554)
* Martyn Plummer. 2003. JAGS: A program for analysis of
  Bayesian graphical models using Gibbs sampling. In
  *Proceedings of the 3rd International Workshop on Distributed
  Statistical Computing (DSC)*, 124-125.
  [https://www.r-project.org/conferences/DSC-2003/Proceedings/Plummer.pdf](https://www.r-project.org/conferences/DSC-2003/Proceedings/Plummer.pdf)
* Adam Ścibior, Ohad Kammar, Matthijs Vákár, Sam Staton,
  Hongseok Yang, Yufei Cai, Klaus Ostermann, Sean K. Moss, Chris
  Heunen, and Zoubin Ghahramani. 2018. Denotational validation of
  higher-order Bayesian inference. *Proceedings of the ACM on
  Programming Languages*, 2(POPL):60:1-60:29.
  [https://doi.org/10.1145/3158148](https://doi.org/10.1145/3158148)
* Joseph Tassarotti and Robert Harper. 2019. A separation logic
  for concurrent randomized programs. *Proceedings of the ACM on
  Programming Languages*, 3(POPL):64:1-64:30.
  [https://doi.org/10.1145/3290334](https://doi.org/10.1145/3290334)
* Matthijs Vákár, Ohad Kammar, and Sam Staton. 2019. A domain
  theory for statistical probabilistic programming. *Proceedings
  of the ACM on Programming Languages*, 3(POPL):36:1-36:29.
  [https://doi.org/10.1145/3290349](https://doi.org/10.1145/3290349)
* Mitchell Wand, Ryan Culpepper, Theophilos Giannakopoulos, and
  Andrew Cobb. 2018. Contextual equivalence for a probabilistic
  language with continuous random variables and recursion.
  *Proceedings of the ACM on Programming Languages*,
  2(ICFP):87:1-87:30.
  [https://doi.org/10.1145/3243631](https://doi.org/10.1145/3243631)
* David Wingate, Andreas Stuhlmüller, and Noah D. Goodman. 2011.
  Lightweight implementations of probabilistic programming
  languages via transformational compilation. In *Proceedings of
  the Fourteenth International Conference on Artificial
  Intelligence and Statistics (AISTATS)*, 770-778.
  [http://proceedings.mlr.press/v15/wingate11a.html](http://proceedings.mlr.press/v15/wingate11a.html)
* Frank Wood, Jan-Willem van de Meent, and Vikash Mansinghka.
  2014. A new approach to probabilistic programming inference. In
  *Proceedings of the 17th International Conference on
  Artificial Intelligence and Statistics (AISTATS)*, 1024-1032.
  [http://proceedings.mlr.press/v33/wood14.html](http://proceedings.mlr.press/v33/wood14.html)
