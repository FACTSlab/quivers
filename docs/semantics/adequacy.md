# Adequacy

The implementation in `quivers.dsl.compiler.Compiler` is a *concrete realization* of the denotational semantics. This page states and sketches a proof of the adequacy theorem connecting the two.

## 1. The compiler as a function

The compiler is a function

$$
\mathcal{C} : \mathrm{AST} \to \mathrm{Program} \cup \{\bot\},
$$

producing a `quivers.Program` (an `nn.Module`) on success and raising `CompileError` (denoted $\bot$) on type-error or undefined-name failure. A `quivers.Program` is in turn a *runnable representation* of a kernel: invoking $\mathrm{Program}.\mathrm{forward}$ produces, for each input, a value or a sample distribution.

We define the *operational denotation* of a successfully-compiled module $M$ as

$$
\mathcal{C}\llbracket M \rrbracket : \llbracket \tau_{\mathrm{in}} \rrbracket \to \mathcal{G}\bigl(\llbracket \tau_{\mathrm{out}} \rrbracket\bigr),
\qquad
\mathcal{C}\llbracket M \rrbracket(x) \;=\; \text{distribution induced by } \mathrm{Program}.\mathrm{forward}(x).
$$

For deterministic (purely $\mathcal{V}$-enriched) modules this collapses to a Dirac kernel.

## 2. Statement of adequacy

**Theorem (Adequacy).** Let $M$ be a well-typed QVR module and let $\theta_M$ be the parameter assignment realized in the compiled program. Then

$$
\mathcal{C}\llbracket M \rrbracket \;=\; \llbracket M \rrbracket
\quad \text{in } \mathbf{Kern},
$$

where $\llbracket M \rrbracket$ is the denotation of [Setting](setting.md), [Morphisms](morphisms.md), [Expressions](expressions.md), and [Programs](programs.md), instantiated at the parameters $\theta_M$.

Equality is in $\mathbf{Kern}$, i.e. as Markov kernels: for every input $x$ the two sides agree as probability measures on the $\sigma$-algebra of the codomain. For the deterministic (purely $\mathcal{V}$-enriched) strata this collapses to exact tensor equality up to floating-point precision; for sampled compositions in $\mathbf{Kern}$ the agreement is *in expectation*, witnessed by unbiased Monte-Carlo estimates whose error decays as the sample budget grows. The test suite asserts the deterministic equalities pointwise and the sampled equalities within numerical tolerance.

## 3. Proof sketch

The proof proceeds by structural induction on the module $M$. We outline the induction at each layer.

### 3.1 Types and spaces

For every `TypeExpr` $\tau$ and every well-formed environment $\rho_{\mathrm{obj}}$, the lens `TypeExprToSetObject(env)` satisfies

$$
\mathrm{forward}(\tau) \;=\; \llbracket \tau \rrbracket_{\rho_{\mathrm{obj}}}
$$

by definition: the recursion in `_resolve` is term-by-term equal to the recursion in [Types §2](types-and-spaces.md#2-denotation-of-types). The $\mathrm{backward}$ direction recovers the original AST verbatim, so the GetPut and PutGet laws hold by construction. The same argument applies to `SpaceExprToContinuousSpace`.

### 3.2 Discrete morphisms

A `latent` declaration creates a learnable PyTorch parameter $W \in \mathbb{R}^{|X| \times |Y|}$ and applies the constraint map $\sigma : \mathbb{R} \to V$ pointwise. The denotation $\llbracket f \rrbracket(x, y) = \sigma(W[x, y])$ is exactly the value computed by `LatentMorphism.tensor`. An `observed` declaration's denotation is the user-supplied tensor verbatim.

### 3.3 Composition

The `>>` operator in PyTorch is realized as $V$-algebra matrix multiplication (`noisy_or_contract` for $\mathcal{V}_{\mathrm{pf}}$, ordinary matrix multiplication for $\mathcal{V}_{\mathbb{B}}$ over `BoolTensor`, etc.). The composition of two tensors $(M_1, M_2)$ at indices $(x, z)$ is

$$
\bigoplus_{y} \sigma(M_1[x, y]) \otimes \sigma(M_2[y, z]),
$$

term-by-term equal to the categorical composition of [Morphisms §1.1](morphisms.md#11-composition-tensor-identity).

### 3.4 Tensor product, marginalization, fan, stack, repeat, scan

Each combinator is implemented as a tensor-level operation whose definition is the term-by-term unfolding of its denotation. The proofs are straightforward: `@` is `torch.einsum` of the appropriate shape; `marginalize` is `torch.sum` (or algebra-join) along the marginalized axes; `fan` is `torch.stack`; `stack` is `kron`; `repeat` is repeated composition; `scan` is a Python-level fold realizing the trace of [Expressions §3.4](expressions.md#34-scan).

### 3.5 Lookup-table kernels (finite-set codomain)

A `morphism f : A -> B [role = kernel]` declaration with finite-set codomain and no `~ Family` clause is realized as a softmax-normalized parameter tensor along the codomain axis (the canonical surjection $\mathbb{R}^{|Y|} \to \Delta^{|Y| - 1}$). Composition is ordinary matrix multiplication, which coincides with Kleisli composition in $\mathbf{Stoch}$ ([Morphisms §2](morphisms.md#2-stochastic-morphisms)).

### 3.6 Parametric kernels (continuous codomain)

A `morphism f : σ -> σ' [role = kernel, axis_role_clause] ~ Family` declaration is realized as a PyTorch `Distribution` parameterized by a small network mapping $\llbracket \sigma \rrbracket$ to $\Theta_{\mathrm{Family}}$. Sampling and density evaluation are exposed by `rsample` and `log_prob`. Composition is realized by *sampled-composition* (Monte-Carlo integration) or, for closed-form families, by the appropriate analytical convolution. The optional axis-role clause configures the family's event / batch decomposition according to the contract of [Morphisms §6](morphisms.md#6-axis-role-specifications); matrix-valued families (`MatrixNormal`, `Wishart`, `InverseWishart`, `LKJCholesky`) receive their row / column dimensions from the named axes at compile time.

The equality

$$
\int_{\llbracket \sigma' \rrbracket} g_2(t, C)\, g_1(s, \mathrm{d}t) \;=\; \mathbb{E}_{T \sim g_1(s, \cdot)}\bigl[g_2(T, C)\bigr]
$$

between the Chapman–Kolmogorov composition and its Monte-Carlo realization holds in expectation; the `quivers` runtime uses reparameterisation to obtain unbiased gradient estimates.

### 3.7 Programs

The body-interpreter `_compile_program_body` of [`quivers.dsl.compiler`](../api/dsl/compiler.md) realizes the Kleisli chain of [Programs §2](programs.md#2-statements). Each statement is interpreted as a Python operation that:

- *Draw*: calls `family.rsample(theta(context))` and appends the result to the trace;
- *Observe*: calls `family.log_prob(value, theta(context))` and accumulates the score;
- *Let*: evaluates the arithmetic expression and binds the result;
- *Score*: evaluates the let-expression to a scalar tensor, binds the result to the named coordinate, AND adds the same scalar to the accumulated log-score (the runtime tag is `_ScoreSpec`, distinct from `_LetSpec` for ordinary let bindings; see [Programs §2.7a](programs.md#27a-score-factor));
- *Return*: projects the trace onto the named coordinates.

The categorical equations of [Programs §5](programs.md#5-soundness-of-monadic-semantics) are inherited from PyTorch's distribution / random-variable algebra, which is well-known to satisfy the Giry-monad axioms.

### 3.7a Deduction systems and chart-access let-expressions

The body-interpreter resolves three deduction-related let-expression builtins through the [`compose_deductions`, `parse`, `subst`] dispatch table on `_ProgramsMixin`:

- *parse(D, x)*: runs the registered `DeductionSystem` $D$ on $x$ to the agenda's fixed point, wrapping the resulting `Chart` in a `ChartView`. The view's method-call surface ([Expressions §4.2](expressions.md#42-deduction-system-call-sites)) realizes presheaf evaluation against the chart's $K$-valued tensor entries; gradients flow through the entries' `requires_grad` via the agenda's semiring operations, matching the differentiable-chart denotation of [Weighted Deduction Fragment §8](grammar.md#8-charts-as-first-class-differentiable-values).
- *compose($D_1, D_2$)*: synthesises a new `DeductionSystem` whose `axiom_injector` chains $D_1$'s `goal_items` into $D_2$'s axioms; the composed system carries cross-attached `_axiom_module` / `_rule_module` references so `composed.parameters()` walks both factors' learnable weights ([Weighted Deduction Fragment §10.1](grammar.md#101-composing-deductions)).
- *subst($t, v, w$)*: performs a structural-equality walk over $t$, replacing each subterm equal to $v$ with $w$. Capture-avoidance is guaranteed at the call site by the compile-time alpha-renaming invariant of the lexicon-LF compiler ([Weighted Deduction Fragment §3a](grammar.md#3a-binders-and-alpha-renaming)).

Adequacy for the deduction-fitting surface follows from the agenda's strategy-independence theorem (Goodman 1999) applied to the bindings-keyed rule-weight parameters: each rule firing on bindings $\sigma$ contributes $\theta_{r, \sigma}$ from a per-deduction `nn.ParameterDict` allocated lazily at run time, and the differentiable-chart equations of §8 propagate the gradient back to $\theta_{r, \sigma}$ through the standard semiring-times / semiring-plus operations.

### 3.7b Bayesian lifts of parameter-bearing artefacts

The four lift constructors of [`quivers.inference.lifts`](../api/inference/lifts.md)
realise [Programs §7](programs.md#7-bayesian-lifts). Adequacy
at this stratum is the claim that the lifted [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram)
realises its stated log-density.

* [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) (Programs §7.1): the
  total log-density assembled by the inference layer is
  $\sum_{i} \log \mathcal{N}(\theta_{i}; 0, \sigma_{\theta}^{2}) + \sum_{j} \log \mathcal{N}(\mathbf{z}_{j}; 0, \sigma_{z}^{2}) + S(\theta, \mathbf{z})$
  with $S$ the score step. The score step's subtraction of the
  placeholder log-priors cancels the second term *pointwise*
  (not merely in expectation), so the lifted log-density
  realises $\log p(\theta) + \log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta)$ exactly. The
  parameter-substitution context manager
  ([`_swap_named_parameters`](../api/inference/lifts.md)) preserves the
  inner module's autograd graph through the override tensors,
  so $\nabla_{(\theta, \mathbf{z})} \log p_{\mathrm{lift}}$ on
  the lift agrees with the analytic gradient of the target on
  the unconstrained latent state.
* [`lift_to_bayesian_program`](../api/inference/lifts.md#quivers.inference.lifts.lift_to_bayesian_program) and [`lift_from_log_prob`](../api/inference/lifts.md#quivers.inference.lifts.lift_from_log_prob)
  (Programs §7.2–§7.3): adequacy reduces by construction to
  §7.1, since both wrap their input in a synthetic `log_joint`
  before delegating to [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters).
* [`monte_carlo_log_joint`](../api/inference/lifts.md#quivers.inference.lifts.monte_carlo_log_joint) (Programs §7.4): the
  wrapper realises a single-sample MC estimator
  $\widetilde{\ell}(x, y; \theta) = \log p_{\mathrm{inner}}(y \mid \mathbf{z}_{*}, x, \theta)$,
  $\mathbf{z}_{*} \sim q$. By Jensen's inequality, the
  estimator is downward-biased relative to the marginal
  log-likelihood; the wrapper is therefore an *unsound*
  denotation for any MCMC sampler that demands a deterministic
  per-evaluation energy (NUTS / HMC). For SVI it is a valid
  reparameterised stochastic gradient estimator of the
  marginal log-likelihood's expected gradient. Adequacy at
  this stratum is the conjunction of these two clauses,
  documented at the call site.

### 3.8 Schema extraction

The schema extractor `extract_program_schema` walks the same resolved environment used by the compiler. Its denotation is exactly the schema-level interpretation of [The Program Theory §2](program-theory.md#2-the-extraction-functor); the equality $\mathcal{S}(M) = \mathrm{extract\_program\_schema}(\mathrm{Compiler}(M))$ holds by construction.

## 4. Tests as adequacy witnesses

The test suite at `tests/test_resolution_lenses.py` and `tests/test_program_theory.py` asserts the following propositions, each a special case of the adequacy theorem.

| Test file | Assertion |
|-----------|-----------|
| `test_resolution_lenses.py` | $\mathrm{forward}(\tau) = \llbracket \tau \rrbracket$ and $\mathrm{forward}(\sigma) = \llbracket \sigma \rrbracket$ on every example program. |
| `test_resolution_lenses.py` | GetPut and PutGet laws hold on every example. |
| `test_resolution_lenses.py` | The lens-driven resolution agrees with the compiler's `_resolve_type` dispatch on every example. |
| `test_program_theory.py` | $\mathcal{S}(M)$ validates against $\mathsf{QVR}$ on every example. |
| `test_program_theory.py` | $\mathcal{S}(M_1) \neq \mathcal{S}(M_2)$ when $M_1, M_2$ differ in declarations. |
| `test_program_theory.py` | Idempotence: $\mathcal{S}(M)$ recomputed twice yields equal schemas. |
| `test_model_roundtrips.py` | Every `dx.Model` instance round-trips through `model_dump_json` / `model_validate_json`. |

The tests are not a substitute for a fully formal proof, but they constitute a finite set of adequacy witnesses across the language's surface syntax, verified on every commit.

## 5. Limitations

The adequacy theorem holds *up to floating-point precision* and *modulo the chosen Monte-Carlo seeds* for sampled compositions in $\mathbf{Kern}$. The semantic equality is between *kernels* (probability distributions); the implementation realizes samples and gradient estimates of those kernels, with an error that decays with the number of Monte-Carlo samples and is irrelevant in the limit.

The four tensor-bearing accumulators (`Presheaf`, `Weight`, `SampleSite`, `Trace`) carry mutable runtime state and are not part of the value layer; the adequacy theorem does not assign them denotations. They are scratch space for inference algorithms; their semantics is given operationally by the inference layer and is outside the scope of this document.

## References

- Joyal, A., Street, R., and Verity, D. (1996). [*Traced monoidal categories*](https://doi.org/10.1017/S0305004100074338). Mathematical Proceedings of the Cambridge Philosophical Society, 119(3), 447–468.
- Giry, M. (1982). [*A categorical approach to probability theory*](https://doi.org/10.1007/BFb0092872). In B. Banaschewski (ed.), *Categorical Aspects of Topology and Analysis*, Lecture Notes in Mathematics 915, 68–85. Springer.
- Cho, K. and Jacobs, B. (2019). [*Disintegration and Bayesian inversion via string diagrams*](https://doi.org/10.1017/S0960129518000488). Mathematical Structures in Computer Science, 29(7), 938–971.
- Fritz, T. (2020). [*A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics*](https://doi.org/10.1016/j.aim.2020.107239). Advances in Mathematics, 370, 107239.
- Kelly, G. M. (1982). [*Basic Concepts of Enriched Category Theory*](http://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html). London Mathematical Society Lecture Note Series 64. Cambridge University Press. Republished as Reprints in Theory and Applications of Categories 10 (2005).
- Lawvere, F. W. (1973). [*Metric spaces, generalized logic, and closed categories*](https://doi.org/10.1007/BF02924844). Rendiconti del Seminario Matematico e Fisico di Milano, 43(1), 135–166. Republished as Reprints in Theory and Applications of Categories 1 (2002), [tac:reprints/articles/1/tr1abs.html](http://www.tac.mta.ca/tac/reprints/articles/1/tr1abs.html).
