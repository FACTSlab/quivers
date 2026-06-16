# Stan

This page discharges the per-target obligations of
[Transpilation correctness](index.md) for $\mathsf{T} = \mathrm{Stan}$.

## Semantics

Stan's denotational semantics is the factor-graph semantics of
[Koller and Friedman (2009)](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)
Chapter 4 instantiated by the Stan Reference Manual: a program
declares a finite set of random variables and a finite set of
factors, with `~` statements desugaring to `target += <lpdf>`
contributions to the log-density accumulator (Stan Reference
Manual §6.1, "Sampling Statements"). The log-density probe is
[`cmdstanpy.CmdStanModel.log_prob`](https://mc-stan.org/cmdstanpy/api.html#cmdstanpy.CmdStanModel.log_prob),
which evaluates the program's `target` at a parameter point on the
unconstrained scale.

The reference is Carpenter, Gelman, Hoffman, Lee, Goodrich,
Betancourt, Brubaker, Guo, Li, and Riddell
([2017](https://doi.org/10.18637/jss.v076.i01)).

## Unconstrained-space change of variables

Stan is the only target in the supported set that uses a
non-identity $\Psi_{\mathsf{Stan}}$. Constrained-parameter
declarations introduce the following changes of variables and
Jacobians (Stan Reference Manual §10):

| Declaration | $\Psi^{-1}_{\mathsf{Stan}}$ (constraint → unconstrained) | $\log\bigl|\det J_{\Psi_{\mathsf{Stan}}}\bigr|$ |
|---|---|---|
| `real<lower=0>` | $\theta = \exp(\widetilde\theta)$ | $\widetilde\theta$ |
| `real<lower=0, upper=1>` | $\theta = \mathrm{logit}^{-1}(\widetilde\theta)$ | $\widetilde\theta - 2\log(1 + e^{\widetilde\theta})$ |
| `vector<lower=0>[K]` | per-element $\exp$ | $\sum_k \widetilde\theta_k$ |
| `simplex[K]` | stick-breaking | per Stan Reference Manual §10.6 |
| `cov_matrix[K]` | Cholesky factorization + log-Cholesky | per §10.9 |
| `cholesky_factor_corr[K]` | hyperspherical | per §10.10 |
| `corr_matrix[K]` | LKJ stick-breaking | per §10.10 |

The Stan renderer emits the constrained-space declaration; Stan's
runtime adds $\log|\det J_{\Psi_{\mathsf{Stan}}}|$ to the `target`
accumulator automatically (cf. Stan Reference Manual §10.1, "Built-in
Transforms"). The renderer never emits the Jacobian term itself.

By Theorem 6.1 of the [head page](index.md), the equality

$$
\log p_{\mathsf{Stan}}(\Psi_{\mathsf{Stan}}(\theta), y \mid x)
\;-\;
\log\bigl|\det J_{\Psi_{\mathsf{Stan}}}(\theta)\bigr|
\;=\;
\log p_{\mathrm{QVR}}(\theta, y \mid x) + C_{\mathsf{Stan}}(M)
$$

holds at every joint point: the left-hand side is exactly Stan's
constrained-space log-density (Stan's `target` value minus the
Jacobian Stan added), which equals the QVR log-density on the
matching support.

## Family parameterizations

Stan's families use the canonical parameterizations of the
[Stan Functions Reference](https://mc-stan.org/docs/functions-reference/).
Most map identically from QVR's parameterization (Wikipedia
canonical forms). The non-trivial cases:

| QVR family | Stan call | $\pi_{F, \mathsf{Stan}}$ | $c_{F, \mathsf{Stan}}$ |
|---|---|---|---|
| `Normal(μ, σ)` | `normal(μ, σ)` | identity | 0 |
| `HalfNormal(σ)` | `normal(0, σ)` with `<lower=0>` declaration | $\sigma \mapsto (0, \sigma)$ | $\log 2$ (the truncation normalizer) |
| `HalfCauchy(γ)` | `cauchy(0, γ)` with `<lower=0>` | $\gamma \mapsto (0, \gamma)$ | $\log 2$ |
| `MultivariateNormal(μ, Σ)` | `multi_normal(μ, Σ)` | identity | 0 |
| `LKJCorrCholesky(η)` | `lkj_corr_cholesky(η)` | identity | 0 |
| `Dirichlet(α)` | `dirichlet(α)` | identity (scalar broadcast via `rep_vector`) | 0 |
| `Categorical(p)` | `categorical(p)` | identity (1-indexed) | 0 |
| `Bernoulli(p)` | `bernoulli(p)` | identity | 0 |
| `LogitNormal(μ, σ)` | not supported (no native `logit_normal_lpdf`) | — | raises `family:LogitNormal` |
| every other family in the registry | identical Stan name | identity | 0 |

Stan's 1-indexed `categorical` adds 1 to QVR's 0-indexed support;
the `IntegerInterval(1, K)` declaration on Stan's data side
reflects this. The renderer emits the index expression `k`
inside the `for (k in 1:K)` loop directly; no offset is added at
the call site.

The half-distribution support truncation contributes $c_{F,
\mathsf{Stan}} = \log 2$ per call site. This is collected into
the per-program constant $C_{\mathsf{Stan}}(M)$ and absorbed by
Bayes-rule normalization.

## Per-construct emit

**`SampleStep(x, F, args)`.** `top_var_decl_no_assign` of type
$\tau_F$ in `parameters` (with the right constraint per the
family's `.support`); `sampling_statement` $x \sim F_{\mathsf{Stan}}(\mathrm{args})$
in `model`. Soundness: by Stan's `~` desugaring, the contribution
to `target` is exactly $\log f_{F, \mathsf{Stan}}(x \mid
\mathrm{args})$.

**`ObserveStep(y, F, args)`.** $y$ declared in `data`; same `~`
statement in `model` contributing $\log f_{F, \mathsf{Stan}}(y \mid
\mathrm{args})$.

**`Plate` (Indexed Bind / Indexed Observe).** Nested
`for (m_<axis> in 1:N_<axis>) { ... }` loops; the latent or
observation is declared as `array[N_<axis>] <type>`. Per Stan
Reference Manual §6.4, the for-loop's contribution to `target` is
the additive sum of per-iteration `~` contributions. Soundness
follows from [head §5.2](index.md#52-plate-indexed-bind-translation-soundness).

**`MarginalizeStep`.** Stan emits the canonical `log_sum_exp`
enumeration. For a grouped marginalize with latent $z : K$ on
group axis $G$ and scope observe via fibration $g : R \to G$:

```stan
{
  array[|G|] vector[K] lps_z;
  for (gi in 1:|G|)
    for (k in 1:K)
      lps_z[gi, k] = categorical_lpmf(k | F_args);
  for (n in 1:|R|)
    for (k in 1:K)
      lps_z[g[n], k] += <scope_lpdf>(y[n] | args_obs(k));
  for (gi in 1:|G|)
    target += log_sum_exp(lps_z[gi]);
}
```

Soundness: the inner accumulation is exact arithmetic in
log-space; the outer `log_sum_exp` is the documented
`log_sum_exp` primitive that computes $\log \sum_k \exp(\cdot)$
accurately. By [head §5.3.1](index.md#531-stan-style-enumeration-over-finite-support-latents),
the emitted log-density equals the marginalized joint up to the
per-family constants of Lemma 5.1.1.

The Stan renderer raises
[`UnsupportedConstruct(["marginalize:non-finite-support:<family>"])`](../../api/transpile.md)
when [`finite_enumerable_at_call_site`](../../api/transpile/family_meta.md#finite_enumerable_at_call_site)
returns False; cf.
[ZIP regression](../../examples/zip-regression.md) for the
continuous-support workaround.

**`ScoreStep(name, expr)`.** `real <name> = <expr>;` in
`transformed_parameters`, plus `target += <name>;` in `model`.
Contribution: $\log w_{\mathrm{name}} = \mathrm{expr}$.

**`LetStep(name, expr)`.** `real <name> = <expr>;` in
`transformed_parameters`. Denotationally inert.

**`Return(v_1, \dots, v_m)`.** `generated quantities { <type> v_i_value = v_i; }`
for each return variable that is already a parameter (Stan rejects
re-declaration). Denotationally inert.

## Acceptance

* **Tier 1 structural.** Every emit has the expected `program`
  vertex with `data`, `parameters`, `transformed_parameters`,
  `model`, `generated_quantities` children; per-construct
  subgraph shape per [head §5.1](index.md#51-per-family-density-preservation)
  /
  [head §5.5](index.md#55-score--let-translation-soundness).
* **Tier 2 lens-laws.** `Lower >> StanRenderer >> EmitPretty(stan)`
  composition law holds; the re-emit fixed point holds on the
  by-construction schema (the panproto `emit_pretty` invariant).
* **Tier 3 external syntax.** `stanc --info -` accepts every
  emit in the test matrix.
* **Tier 4 numeric equivalence.** `cmdstanpy.log_prob` evaluated
  at 256 deterministic grid points + hand-picked corners agrees
  with the QVR reference up to $C_{\mathsf{Stan}}(M) +
  \log|\det J_{\Psi_{\mathsf{Stan}}}(\theta)|$ within $10^{-6}$;
  pairwise transitivity with the other Tier-4-passing backends
  holds.
