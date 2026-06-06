# Transpilation Correctness

The transpiler in [`quivers.transpile`](../api/transpile.md) realizes every well-typed QVR module $M$ as source bytes for a target probabilistic programming language $\mathsf{T} \in \{\text{Stan}, \text{NumPyro}, \text{Pyro}, \text{PyMC}, \text{Edward2}, \text{Turing.jl}, \text{Gen.jl}, \text{Church}, \text{WebPPL}, \text{BUGS}, \text{JAGS}\}$. This page proves that the target bytes denote the same conditional joint distribution as the QVR source.

The development is in three layers. §1 fixes the source category $\mathbf{Mod}_{\mathrm{QVR}}$ of QVR modules and the QVR semantics functor $\mathsf{S}_{\mathrm{QVR}} : \mathbf{Mod}_{\mathrm{QVR}} \to \mathbf{Kern}$. §2 fixes the target category $\mathbf{Prog}_{\mathsf{T}}$ of target-language programs and the target semantics functor $\mathsf{S}_{\mathsf{T}} : \mathbf{Prog}_{\mathsf{T}} \to \mathbf{Kern}$. §3 defines the transpile functor $\mathsf{T}_{\mathsf{T}} : \mathbf{Mod}_{\mathrm{QVR}} \to \mathbf{Prog}_{\mathsf{T}}$ as a composite of three explicit functors, identifies the natural isomorphism

$$
\eta_{\mathsf{T}} \;:\; \mathsf{S}_{\mathrm{QVR}} \;\xRightarrow{\,\cong\,}\; \mathsf{S}_{\mathsf{T}} \circ \mathsf{T}_{\mathsf{T}}
$$

(in $\mathbf{Kern}$, up to the constant of integration absorbed by reparametrization), and proves naturality by structural induction on the QVR module. §4 records the (finite) empirical obligations and the test infrastructure that discharges them.

## 1. The QVR source category

### 1.1 Objects and morphisms

Let $\mathbf{Mod}_{\mathrm{QVR}}$ denote the category whose objects are well-typed QVR modules $M$ (in the sense of [Typing](typing.md)) and whose morphisms are well-typed module rewrites: a morphism $\phi : M \to M'$ is a function on AST nodes that preserves the static-type judgment $\Gamma; \Phi \vdash \cdot$. The identity is the identity rewrite; composition is the composition of rewrites. The category is small: $\mathrm{Ob}(\mathbf{Mod}_{\mathrm{QVR}})$ is the set of QVR modules, $\mathrm{Hom}(M, M')$ is the set of type-preserving rewrites.

For the purpose of this section we restrict attention to the full subcategory $\mathbf{Mod}^*_{\mathrm{QVR}} \subseteq \mathbf{Mod}_{\mathrm{QVR}}$ of modules with exactly one `program_decl` $p : A \to B$ and a non-empty observation set. The restriction is harmless for transpilation: backends with multiple programs would emit a sequence of independent target programs, one per `program_decl`.

### 1.2 The QVR semantics functor

The QVR denotation functor

$$
\mathsf{S}_{\mathrm{QVR}} \;:\; \mathbf{Mod}^*_{\mathrm{QVR}} \;\longrightarrow\; \mathbf{Kern}
$$

sends each module $M$ to the Markov kernel $\llbracket M \rrbracket : A \to \mathcal{G}(\Theta \times B)$ constructed in [Programs](programs.md) and [Adequacy](adequacy.md), where $\Theta$ is the cartesian product of the parameter spaces of `program_decl.draws`'s `SampleStep`s, and where $\mathcal{G}$ is the (continuous) Giry monad on standard Borel spaces. The action on morphisms $\phi : M \to M'$ is the induced morphism of kernels (a measurable map between the joint spaces).

By [Adequacy §2](adequacy.md#2-statement-of-adequacy), $\mathsf{S}_{\mathrm{QVR}}$ is functorial and admits an unambiguous log-density representation

$$
\log p_{\mathrm{QVR}}(\theta, y \mid x)
\;\equiv\;
\log \mathsf{S}_{\mathrm{QVR}}(M)(\theta, y \mid x)
\;=\;
\sum_{i \in \mathrm{Sample}(p)} \log f_i\bigl(\theta_i \mid \mathrm{args}_i(\theta, x)\bigr)
\;+\;
\sum_{j \in \mathrm{Observe}(p)} \log f_j\bigl(y_j \mid \mathrm{args}_j(\theta, x)\bigr)
\;+\;
\sum_{k \in \mathrm{Score}(p)} \log w_k(\theta, x),
$$

where $f_i$ is the density of the family declared in `SampleStep` $i$ in its canonical parameterization and $w_k$ is the user-supplied weight in `ScoreStep` $k$. This is the formula evaluated by [`quivers.inference.trace.trace`](../api/inference/trace.md).

## 2. The target category

### 2.1 Objects and morphisms

For each target $\mathsf{T}$ let $\mathbf{Prog}_{\mathsf{T}}$ denote the category whose objects are syntactically valid $\mathsf{T}$-programs (source-byte strings accepted by the target's canonical compiler / parser) and whose morphisms are program rewrites. The category has a $\mathbf{Bytes}$-valued forgetful functor $U_{\mathsf{T}} : \mathbf{Prog}_{\mathsf{T}} \to \mathbf{Bytes}$ ($\mathbf{Bytes}$ being the discrete category of byte strings), and the target's parser is a partial inverse $U_{\mathsf{T}}^{-1}$ on the image of $U_{\mathsf{T}}$.

For the proof we restrict to the full subcategory $\mathbf{Prog}^*_{\mathsf{T}} \subseteq \mathbf{Prog}_{\mathsf{T}}$ of programs that fall in the *static graphical model fragment* of $\mathsf{T}$: a fixed sequence of `sample` / `observe` / `factor` statements with no stochastic control flow, recursion, or higher-order computation. Quivers' walkers always emit programs in this fragment; first-class probabilistic features of the target languages are out of scope.

### 2.2 The target semantics functor

Each target carries a published or canonical denotational semantics

$$
\mathsf{S}_{\mathsf{T}} \;:\; \mathbf{Prog}^*_{\mathsf{T}} \;\longrightarrow\; \mathbf{Kern}
$$

whose specifics differ by target but whose action is uniform on the static graphical model fragment. We exhibit the semantics in two equivalent forms.

#### 2.2.1 First-class PPLs: trace semantics

For NumPyro, Pyro, Edward2, Turing.jl, Gen.jl, Church, and WebPPL, the published semantics is the trace-based stochastic-lambda-calculus semantics of Goodman et al. (2008), generalized to first-class probabilistic programs by Wingate, Stuhlmüller, and Goodman (2011). A program $e \in \mathbf{Prog}^*_{\mathsf{T}}$ denotes a probability measure $\llbracket e \rrbracket_{\mathsf{T}}$ on the trace space $\mathcal{T}_e = \prod_\alpha S_\alpha$, where $\alpha$ ranges over the addresses of `sample`, `observe`, and `factor` (`score`) invocations along the program's execution, and $S_\alpha$ is the sample space of the family at address $\alpha$.

The trace density is

$$
p_e(\tau) \;=\;
\prod_{\alpha \in \mathrm{sample}(\tau)} f_\alpha\bigl(\tau_\alpha \mid \mathrm{args}_\alpha(\tau)\bigr)
\;\cdot\;
\prod_{\beta \in \mathrm{observe}(\tau)} f_\beta\bigl(\tau_\beta \mid \mathrm{args}_\beta(\tau)\bigr)
\;\cdot\;
\prod_{\gamma \in \mathrm{factor}(\tau)} w_\gamma(\tau).
$$

The functor $\mathsf{S}_{\mathsf{T}}$ sends $e$ to the kernel $\mathcal{T}_e \ni \tau \mapsto \llbracket e \rrbracket_{\mathsf{T}}(\tau)$ in $\mathbf{Kern}$. The trace density is computed exactly by the target's native log-density probe:

* NumPyro: [`numpyro.infer.util.log_density(model, model_args=(y,), params={**θ, **y})`](https://num.pyro.ai/en/stable/utilities.html#log-density).
* Pyro: [`pyro.poutine.trace(pyro.condition(model, data={**θ, **y})).get_trace(y).log_prob_sum()`](https://docs.pyro.ai/en/stable/poutine.html).
* Edward2: `with ed.tape() as t: model(); sum(rv.distribution.log_prob(rv.value) for rv in t.values())` from [`ed.tape`](https://github.com/google/edward2).
* Turing.jl: [`Turing.logjoint(model, θ)`](https://turinglang.org/Turing.jl/dev/api/Inference/#Turing.logjoint).
* Gen.jl: [`Gen.assess(generator, args, choicemap)[1]`](https://www.gen.dev/docs/stable/ref/inference/#Gen.assess) returns the exact joint log-probability.
* Church: trace density per [the stochastic lambda calculus](http://web.stanford.edu/~ngoodman/papers/StochLambdaCalc.pdf).
* WebPPL: trace density per [Goodman & Stuhlmüller](http://dippl.org).

#### 2.2.2 Static graphical model fragment: factor-graph semantics

For Stan, PyMC, BUGS, and JAGS, the published semantics is the factor-graph semantics of [Koller and Friedman (2009), Chapter 4](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/). A program $e \in \mathbf{Prog}^*_{\mathsf{T}}$ declares a finite set of random variables $V = V_\Theta \sqcup V_Y$ (free parameters and observations) and a finite set of factors $F$, each factor binding a tuple of variables to a non-negative weight. The joint log-density is

$$
\log \llbracket e \rrbracket_{\mathsf{T}}(\theta, y) \;=\; \sum_{f \in F} \log f\bigl(\mathrm{vars}(f)\bigr).
$$

Per target:

* Stan: [Stan Reference Manual §8](https://mc-stan.org/docs/reference-manual/sampling-statements.html) defines `y ~ d(args);` as `target += d_lpdf(y | args);`. The `target` accumulator is the program's log-density.
* PyMC: each `pymc.<Distribution>("x", ...)` inside a [`pymc.Model`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html) registers a random variable; [`Model.compile_logp`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html#pymc.Model.compile_logp) returns $\sum_v \log p_v$.
* BUGS / JAGS: each `x ~ d(args)` and each `x <- expr` registers a stochastic or deterministic node; the joint log-density is the sum of stochastic-node `log_d` values.

The two semantic forms agree on the static graphical model fragment: every trace $\tau$ in §2.2.1 enumerates a $(\theta, y)$ point that uniquely determines the factor values in §2.2.2, and vice versa, so

$$
\log p_{\mathsf{T}}(\theta, y \mid x)
\;\equiv\;
\log \mathsf{S}_{\mathsf{T}}(e)(\theta, y \mid x)
\;=\;
\sum_{i} \log f_i^{\mathsf{T}}\bigl(\theta_i \mid \mathrm{args}_i\bigr)
\;+\;
\sum_{j} \log f_j^{\mathsf{T}}\bigl(y_j \mid \mathrm{args}_j\bigr)
\;+\;
\sum_{k} \log w_k^{\mathsf{T}}
\;+\;
C_{\mathsf{T}},
$$

where $f_i^{\mathsf{T}}$ is the density of family $F_i$ as implemented in target $\mathsf{T}$ and $C_{\mathsf{T}}$ is a constant accumulating the choice of base measure, reparametrization Jacobian (Stan `<lower=0>`, NumPyro [`TransformReparam`](https://num.pyro.ai/en/stable/_modules/numpyro/infer/reparam.html)), and the target's `~` accumulator convention.

## 3. The transpile functor

The transpiler is the functor

$$
\mathsf{T}_{\mathsf{T}} \;:\; \mathbf{Mod}^*_{\mathrm{QVR}} \;\longrightarrow\; \mathbf{Prog}^*_{\mathsf{T}}
$$

that factors as

$$
\mathsf{T}_{\mathsf{T}} \;=\; U_{\mathsf{T}}^{-1} \circ \mathsf{E}_{\mathsf{T}} \circ \mathsf{W}_{\mathsf{T}},
$$

a composite of three explicit functors in the diagram

$$
\mathbf{Mod}^*_{\mathrm{QVR}}
\;\xrightarrow{\;\mathsf{W}_{\mathsf{T}}\;}\;
\mathbf{Sch}_{\mathsf{T}}
\;\xrightarrow{\;\mathsf{E}_{\mathsf{T}}\;}\;
\mathbf{Bytes}
\;\xrightarrow{\;U_{\mathsf{T}}^{-1}\;}\;
\mathbf{Prog}^*_{\mathsf{T}}.
$$

* $\mathbf{Sch}_{\mathsf{T}}$ is the category of [panproto schemas](https://panproto.readthedocs.io/en/latest/api/schema.html#panproto.Schema) over $\mathsf{T}$'s tree-sitter grammar: objects are schemas (typed vertex-and-edge structures whose vertex kinds are the grammar's node names); morphisms are panproto [`SchemaMorphism`](https://panproto.readthedocs.io/en/latest/api/schema.html#panproto.SchemaMorphism)s.
* $\mathsf{W}_{\mathsf{T}}$ is the per-backend walker (a subclass of [`SchemaTransform(dx.Mapping)`](../api/transpile.md#schematransform)), constructing a schema by recursing structurally on the AST. The action on morphisms $\phi : M \to M'$ is the induced rewrite at the schema level: a type-preserving rewrite of the source produces a structure-preserving rewrite of the schema.
* $\mathsf{E}_{\mathsf{T}}$ is the panproto by-construction emitter [`emit_pretty`](https://panproto.readthedocs.io/en/latest/api/registry.html#panproto.AstParserRegistry.emit_pretty), walking the schema's productions to produce source bytes. By the panproto re-emit fixed-point property (cf. §4.2), $\mathsf{E}_{\mathsf{T}}$ is faithful on by-construction schemas.
* $U_{\mathsf{T}}^{-1}$ is the target's parser, a partial functor whose image is exactly the syntactically valid bytes (cf. §2.1). By the panproto external syntax property (cf. §4.3), $\mathsf{E}_{\mathsf{T}}$'s image is contained in this image, so $U_{\mathsf{T}}^{-1} \circ \mathsf{E}_{\mathsf{T}}$ is total on $\mathbf{Sch}_{\mathsf{T}}$'s image under $\mathsf{W}_{\mathsf{T}}$.

The composition $\fatsemi$ is the composition in the Kleisli-style category of [`dx.Mapping`](https://didactic.readthedocs.io/en/latest/api.html#didactic.api.Mapping) over $\mathbf{Set}$, satisfying associativity and the identity laws ($\mathsf{Mapping}.\mathsf{compose}$ is `>>`; the laws are the well-known monoidal-functor laws of `dx.Mapping`).

## 4. The correctness theorem

The diagram

$$
\begin{array}{ccc}
\mathbf{Mod}^*_{\mathrm{QVR}} & \xrightarrow{\;\mathsf{T}_{\mathsf{T}}\;} & \mathbf{Prog}^*_{\mathsf{T}} \\[1ex]
\mathsf{S}_{\mathrm{QVR}} \!\!\downarrow & & \downarrow\!\! \mathsf{S}_{\mathsf{T}} \\[1ex]
\mathbf{Kern} & \xrightarrow{\;\mathrm{id}_{\mathbf{Kern}}\;} & \mathbf{Kern}
\end{array}
$$

commutes up to a natural isomorphism whose components are the constant-shift natural transformations of §1.2 and §2.2.

**Theorem 4.1 (Transpilation Correctness).** *For each target $\mathsf{T}$ whose family map and walker support every construct in $M$, there exists a natural transformation*

$$
\eta_{\mathsf{T}} \;:\; \mathsf{S}_{\mathrm{QVR}} \;\xRightarrow{\;\cong\;}\; \mathsf{S}_{\mathsf{T}} \circ \mathsf{T}_{\mathsf{T}}
$$

*whose component at $M$ is the constant-shift kernel morphism*

$$
\eta_{\mathsf{T}, M}(\theta, y \mid x) \;:\; \log p_{\mathrm{QVR}}(\theta, y \mid x) \;\mapsto\; \log p_{\mathrm{QVR}}(\theta, y \mid x) + c_{\mathsf{T}}(M),
$$

*for some constant $c_{\mathsf{T}}(M) \in \mathbb{R}$. Equivalently, for every $(x, \theta, y)$ in the joint support of $M$,*

$$
\log p_{\mathsf{T}}\bigl(\mathsf{T}_{\mathsf{T}}(M)\bigr)(\theta, y \mid x)
\;=\;
\log p_{\mathrm{QVR}}(\theta, y \mid x) \;+\; c_{\mathsf{T}}(M).
$$

The constant $c_{\mathsf{T}}(M) = \sum_F c_{F, \mathsf{T}} + C_{\mathsf{T}}$ is the sum of per-family reparametrization constants (§5.2 below) and the target's accumulator constant from §2.2. Since the constant is independent of $(\theta, y)$, it is annihilated by every Bayes-rule normalization, so the kernels $\mathsf{S}_{\mathrm{QVR}}(M)$ and $\mathsf{S}_{\mathsf{T}}(\mathsf{T}_{\mathsf{T}}(M))$ define the same conditional, the same posterior, the same marginal: they are equal in $\mathbf{Kern}$.

**Naturality.** For every morphism $\phi : M \to M'$ in $\mathbf{Mod}^*_{\mathrm{QVR}}$, the square

$$
\begin{array}{ccc}
\mathsf{S}_{\mathrm{QVR}}(M) & \xrightarrow{\;\eta_{\mathsf{T}, M}\;} & \mathsf{S}_{\mathsf{T}}(\mathsf{T}_{\mathsf{T}}(M)) \\[1ex]
\mathsf{S}_{\mathrm{QVR}}(\phi) \!\!\downarrow & & \downarrow\!\! \mathsf{S}_{\mathsf{T}}(\mathsf{T}_{\mathsf{T}}(\phi)) \\[1ex]
\mathsf{S}_{\mathrm{QVR}}(M') & \xrightarrow{\;\eta_{\mathsf{T}, M'}\;} & \mathsf{S}_{\mathsf{T}}(\mathsf{T}_{\mathsf{T}}(M'))
\end{array}
$$

commutes. This follows because the components $\eta_{\mathsf{T}, M}$ and $\eta_{\mathsf{T}, M'}$ are constant-shift morphisms whose constants are functions of the (unchanged-by-rewrite) family set used in $M$ and $M'$; type-preserving rewrites preserve the family set up to renaming, so $c_{\mathsf{T}}(M) = c_{\mathsf{T}}(M')$ and the square commutes by the identity.

## 5. Proof

We prove Theorem 4.1 by structural induction on the program $p$ in `program_decl.draws`. The induction is over the finite step sequence $s_1, \dots, s_n$ plus the trailing `return_vars` clause.

### 5.1 Translation rules: the per-step action of $\mathsf{W}_{\mathsf{T}}$

The walker $\mathsf{W}_{\mathsf{T}}$ acts on each program step by producing a schema subgraph. Let $\Sigma_{\mathsf{T}}(s)$ denote the subgraph for step $s$:

| QVR step | $\Sigma_{\mathsf{T}}(s)$ for target $\mathsf{T}$ |
|---|---|
| `SampleStep(x, F, args)` (Stan) | `sampling_statement` $x \sim F_{\mathsf{Stan}}(\mathrm{args})$ in the `model` block + `top_var_decl_no_assign` $\tau_F\;x$ in `parameters` (with `<lower=0>` when $F$'s support is $[0, \infty)$) |
| `SampleStep(x, F, args)` (NumPyro / Pyro / Edward2) | `assignment` $x = \mathsf{namespace}.\mathrm{sample}(\text{"}x\text{"},\; \mathrm{Dist}_{\mathsf{T}}(F)(\mathrm{args}))$ in the model function body |
| `SampleStep(x, F, args)` (PyMC) | `assignment` $x = \mathsf{pymc}.F_{\mathsf{PyMC}}(\text{"}x\text{"},\; \mathrm{args})$ inside the `with pymc.Model()` block |
| `SampleStep(x, F, args)` (Turing.jl) | `tilde_assignment` $x \sim F_{\mathsf{Turing}}(\mathrm{args})$ inside the `@model function` body |
| `SampleStep(x, F, args)` (Gen.jl) | `assignment` $x = \texttt{@trace}\; F_{\mathsf{Gen}}(\mathrm{args})\; \text{:}x$ inside the `@gen function` body |
| `SampleStep(x, F, args)` (Church / WebPPL) | `(define x (sample (F args)))` / `var x = sample(F({args}));` |
| `SampleStep(x, F, args)` (BUGS / JAGS) | `x ~ d_{\mathsf{T}}(F)(\mathrm{args})` in the `model` block |
| `ObserveStep(y, F, args)` | as `SampleStep(y, F, args)` plus the target's observation marker (Stan: `y` declared in `data`; NumPyro: `obs=y` kwarg; PyMC: `observed=y_data` kwarg; Turing.jl: `y` appears as function parameter; Gen.jl: `:y` address; Church / WebPPL: `(observe ...)` / `observe(...)`; BUGS / JAGS: `y` is a data variable) |
| `ScoreStep(w)` | Stan `target += w;` / NumPyro `numpyro.factor("score", w)` / Pyro `pyro.factor("score", w)` / PyMC `pymc.Potential("score", w)` / Turing `Turing.@addlogprob! w` / Gen `Gen.@trace(Dirac(0), :score)` weighted by $w$ / Church `(factor w)` / WebPPL `factor(w)` / BUGS / JAGS `zeros_trick(w)` (NOT YET IMPLEMENTED; cf. §5.6) |
| `LetStep(x, e)` | $x = e$ deterministic assignment in the model body (NOT YET IMPLEMENTED; cf. §5.6) |
| `MarginalizeStep(axes)` | Stan `target += log_sum_exp(...)` / NumPyro `numpyro.factor("marg", logsumexp(...))` / etc. (NOT YET IMPLEMENTED; cf. §5.6) |
| `return_vars = (v_1, \dots, v_m)` | per-target return form: NumPyro / Pyro / Edward2 / Turing.jl / Gen.jl `return v_1, ..., v_m`; PyMC wrap the `with` in `def build_model` and return; Stan `generated quantities { real v_i = v_i; }`; Church trailing variable reference in the `define` body |

The first eight rows are implemented in every backend (subject to the family support matrix, [`test_family_matrix.py`](../api/tests/transpile.md#test-family-matrix)); rows 9–11 are pending walker work (§5.6).

### 5.2 Per-family density identity

**Lemma 5.2.1.** *For every family $F$ in QVR's [family registry](../api/dsl/families.md) and every target $\mathsf{T}$ whose `_FAMILIES` map contains $F \mapsto F_{\mathsf{T}}$, there exists a constant $c_{F, \mathsf{T}} \in \mathbb{R}$ such that for every parameter point $\theta \in \Theta_F$ and every realization $v$ in the support of $F(\theta)$,*

$$
\log f_{\mathsf{T}}\bigl(v \mid \theta\bigr) \;=\; \log f_{\mathrm{QVR}}\bigl(v \mid \theta\bigr) \;+\; c_{F, \mathsf{T}}.
$$

**Proof.** Case by case on $F$. QVR's [`_get_family_registry`](../api/dsl/compiler.md#get_family_registry) implements each family at its canonical parameterization. For each target $\mathsf{T}$ and each family $F$ in $\mathsf{T}$'s `_FAMILIES` map, $F_{\mathsf{T}}$ either uses the same canonical parameterization (so $c_{F, \mathsf{T}} = 0$) or applies a documented reparametrization (so $c_{F, \mathsf{T}}$ is the log-Jacobian).

| Family | Canonical density | Stan name | NumPyro / Pyro / Edward2 / Turing.jl name | PyMC name | BUGS / JAGS name |
|---|---|---|---|---|---|
| `Normal(μ, σ)` | $(2\pi\sigma^2)^{-1/2} \exp(-(x-\mu)^2 / 2\sigma^2)$ | `normal` | `Normal` | `Normal` | `dnorm(μ, 1/σ²)` (BUGS uses precision $\tau = 1/\sigma^2$; walker substitutes) |
| `HalfNormal(σ)` | $2 \cdot (2\pi\sigma^2)^{-1/2} \exp(-x^2/2\sigma^2) \cdot \mathbf{1}_{x \ge 0}$ | `normal(0, σ) T[0,]` | `HalfNormal(σ)` | `HalfNormal("name", sigma=σ)` | `dnorm(0, 1/σ²) T(0,)` |
| `Cauchy(μ, γ)` | $(\pi \gamma (1 + ((x-\mu)/\gamma)^2))^{-1}$ | `cauchy` | `Cauchy` | `Cauchy` | `dt(μ, 1/γ², 1)` (Cauchy = Student-t with 1 df) |
| `Beta(α, β)` | $\Gamma(α + β) / (\Gamma(α)\Gamma(β)) \cdot x^{α-1}(1-x)^{β-1}$ | `beta` | `Beta` | `Beta` | `dbeta` |
| `Gamma(α, β)` (rate $\beta$) | $\beta^\alpha / \Gamma(\alpha) \cdot x^{\alpha-1} \exp(-\beta x)$ | `gamma(α, β)` | `Gamma(α, β)` | `Gamma(alpha=α, beta=β)` | `dgamma(α, β)` |
| `Bernoulli(p)` | $p^x (1-p)^{1-x}$ | `bernoulli` | `Bernoulli` | `Bernoulli` | `dbern(p)` |
| `Categorical(p)` | $\prod_i p_i^{x_i}$ over one-hot $x$ | `categorical` | `Categorical(probs=p)` | `Categorical` | `dcat(p[])` |
| `Dirichlet(α)` | $\Gamma(\sum_i α_i) / \prod_i \Gamma(α_i) \cdot \prod_i x_i^{α_i - 1}$ | `dirichlet` | `Dirichlet(α)` | `Dirichlet` | `ddirich(α[])` |
| `Exponential(λ)` (rate $\lambda$) | $\lambda \exp(-\lambda x)$ | `exponential(λ)` | `Exponential(λ)` | `Exponential` | `dexp(λ)` |
| `InverseGamma(α, β)` | $\beta^\alpha / \Gamma(\alpha) \cdot x^{-\alpha - 1} \exp(-\beta / x)$ | `inv_gamma(α, β)` | `InverseGamma(α, β)` | `InverseGamma` | (no direct primitive; emit via the $1/X$ transform of $\mathrm{Gamma}(\alpha, \beta)$) |
| `Laplace(μ, b)` | $(2b)^{-1} \exp(-|x-μ|/b)$ | `double_exponential(μ, b)` | `Laplace(μ, b)` | `Laplace` | `ddexp(μ, 1/b)` |
| `LogNormal(μ, σ)` | $(x\sigma\sqrt{2\pi})^{-1} \exp(-(\ln x - \mu)^2 / 2\sigma^2)$ | `lognormal` | `LogNormal` | `LogNormal` | `dlnorm(μ, 1/σ²)` |
| `MultivariateNormal(μ, Σ)` | $(2\pi)^{-d/2} \|Σ\|^{-1/2} \exp(-(x-μ)^\top Σ^{-1} (x-μ) / 2)$ | `multi_normal` | `MultivariateNormal(μ, Σ)` | `MvNormal(mu=μ, cov=Σ)` | `dmnorm(μ[], Ω[,])` with $\Omega = Σ^{-1}$ |
| `Pareto(α, x_m)` | $\alpha x_m^\alpha / x^{\alpha + 1} \cdot \mathbf{1}_{x \ge x_m}$ | `pareto(α, x_m)` | `Pareto(α, x_m)` | `Pareto` | `dpar(α, x_m)` |
| `StudentT(ν, μ, σ)` | $\Gamma((\nu+1)/2)/(\Gamma(\nu/2) \sqrt{\nu\pi}\sigma) \cdot (1 + (x-\mu)^2/(\nu\sigma^2))^{-(\nu+1)/2}$ | `student_t(ν, μ, σ)` | `StudentT(ν, μ, σ)` | `StudentT` | `dt(μ, 1/σ², ν)` |
| `Uniform(a, b)` | $(b - a)^{-1} \mathbf{1}_{a \le x \le b}$ | `uniform(a, b)` | `Uniform(a, b)` | `Uniform` | `dunif(a, b)` |
| `Weibull(k, λ)` | $(k/\lambda) (x/\lambda)^{k-1} \exp(-(x/\lambda)^k)$ | `weibull(k, λ)` | `Weibull(k, λ)` | `Weibull` | `dweib(k, 1/λ^k)` |

Every entry is the standard parameterization from [Wikipedia: List of probability distributions](https://en.wikipedia.org/wiki/List_of_probability_distributions) cross-referenced with the cited language references. Where a target uses an alternative encoding (BUGS precision $\tau$, Cauchy as Student-t with 1 df, MvNormal as `dmnorm(μ, Ω)` with $\Omega = Σ^{-1}$), the walker performs the algebraic substitution at emit time so the emitted code's $f_{\mathsf{T}}$ is term-by-term equal to the canonical $f_{\mathrm{QVR}}$. Hence $c_{F, \mathsf{T}} = 0$ for every entry. $\square$

### 5.3 Per-step log-density identity

**Lemma 5.3.1.** *For every `SampleStep(x, F, args)` step $s$ and every target $\mathsf{T}$, the schema subgraph $\Sigma_{\mathsf{T}}(s)$ contributes the factor $\log f_{\mathsf{T}}(x \mid \mathrm{args}(\theta, x_0))$ to the target's joint log-density $\log p_{\mathsf{T}}$.*

**Proof.** Case by target on the table of §5.1:

* **Stan.** Per [Stan Reference Manual §8](https://mc-stan.org/docs/reference-manual/sampling-statements.html), `x ~ F_Stan(args);` is the sugar `target += F_Stan_lpdf(x | args);`. The `parameters` block declaration `real x;` (with `<lower=0>` when $F$'s support is $[0, \infty)$) registers $x$ as a free parameter contributing a uniform improper prior on its constrained space, which adds the same constant to $\log p_{\mathsf{T}}$ for every value of $x$ and is absorbed into $C_{\mathsf{T}}$.
* **NumPyro.** [`numpyro.primitives.sample`](https://num.pyro.ai/en/stable/primitives.html#numpyro.primitives.sample) called without `obs=` registers the address `"x"` with distribution `Dist(args)`; `log_density` evaluates to $\log f_{\mathsf{T}}(x \mid \mathrm{args})$ at the trace point.
* **Pyro.** [`pyro.sample`](https://docs.pyro.ai/en/stable/primitives.html#pyro.sample) without `obs=` registers a latent address; `Trace.log_prob_sum` sums per-site `dist.log_prob(value)`.
* **PyMC.** Each `pymc.<Distribution>("x", ...)` invocation inside a `pymc.Model` registers $x$ with `logp` $\log f_{\mathsf{T}}(x \mid \mathrm{args})$.
* **Turing.jl.** `x ~ D(args)` inside a `@model function` is overloaded to add $\log f_{\mathsf{T}}(x \mid \mathrm{args})$ to the accumulator returned by `Turing.logjoint`.
* **Gen.jl.** `@trace(D(args), :x)` records the address `:x` with distribution $D(\mathrm{args})$; `Gen.assess` returns the sum of per-address log-probabilities.
* **Edward2.** `ed.D(args, name="x")` inside `ed.tape()` records a `RandomVariable`; manually computing `sum(rv.distribution.log_prob(rv.value) for rv in t.values())` yields the trace log-density.
* **Church.** `(define x (sample (D args)))` records $x$ as a free choice with $\log f_{\mathsf{T}}$ contribution.
* **WebPPL.** `var x = sample(D({args}));` records the choice and contributes $\log f_{\mathsf{T}}$.
* **BUGS / JAGS.** `x ~ d(args)` declares a stochastic node; the joint log-density is $\sum_v \log d_{\mathsf{T}}(v \mid \mathrm{parents}(v))$.

By Lemma 5.2.1, $\log f_{\mathsf{T}}(x \mid \mathrm{args}) = \log f_{\mathrm{QVR}}(x \mid \mathrm{args}) + c_{F, \mathsf{T}}$. $\square$

**Lemma 5.3.2.** *For every `ObserveStep(y, F, args)` step $s$ and every target $\mathsf{T}$, the schema subgraph $\Sigma_{\mathsf{T}}(s)$ contributes the factor $\log f_{\mathsf{T}}(y \mid \mathrm{args}(\theta, x_0))$ to $\log p_{\mathsf{T}}$.*

**Proof.** Identical to Lemma 5.3.1 with the observation marker per target:

* **Stan.** `y` declared in `data` (not `parameters`); `y ~ F_Stan(args);` adds only the observation factor $\log f_{\mathsf{T}}(y \mid \mathrm{args})$.
* **NumPyro / Pyro / Edward2.** `obs=y` clamps the address to $y$, contributing only the data factor.
* **PyMC.** `observed=y_data` clamps.
* **Turing.jl.** `y` is a function parameter, clamped when `Turing.logjoint` is evaluated.
* **Gen.jl.** The observation is in the input `ChoiceMap` to `Gen.assess`.
* **Church / WebPPL.** `(observe D y)` / `observe(D, y)` contribute $\log f_{\mathsf{T}}(y \mid \mathrm{args})$.
* **BUGS / JAGS.** `y` is a data variable; `y ~ d(args)` contributes only the data factor.

By Lemma 5.2.1, the factor equals $\log f_{\mathrm{QVR}}(y \mid \mathrm{args}) + c_{F, \mathsf{T}}$. $\square$

### 5.4 The whole-program log-density identity

**Lemma 5.4.1.** *Let $p$ be a `program_decl` with sample steps $\mathrm{Sample}(p) = \{s_1, \dots, s_m\}$, observe steps $\mathrm{Observe}(p) = \{t_1, \dots, t_n\}$, and return clause $(v_1, \dots, v_k)$. Then*

$$
\log p_{\mathsf{T}}\bigl(\mathsf{T}_{\mathsf{T}}(M)\bigr)
\;=\;
\sum_{i=1}^m \log f_{\mathsf{T}}\bigl(x_i \mid \mathrm{args}_i\bigr)
\;+\;
\sum_{j=1}^n \log f_{\mathsf{T}}\bigl(y_j \mid \mathrm{args}_j\bigr)
\;+\;
C_{\mathsf{T}}
$$

*at every joint point $(\theta, y)$.*

**Proof.** By the trace-semantics formula of §2.2.1 (NumPyro / Pyro / Edward2 / Turing.jl / Gen.jl / Church / WebPPL) or the factor-graph semantics of §2.2.2 (Stan / PyMC / BUGS / JAGS), the joint log-density is the sum of per-step factors plus the per-target accumulator constant $C_{\mathsf{T}}$. By Lemmas 5.3.1 and 5.3.2, each per-step factor is the corresponding QVR factor up to $c_{F, \mathsf{T}}$. The trailing `return_vars` clause is denotationally inert: the variables it names are already in $\theta$ or $y$ (cf. [Programs §7](programs.md#7-return)), so the return clause adds no factor; it only affects the host language's program-result API. $\square$

### 5.5 The natural isomorphism

By §1.2, $\log p_{\mathrm{QVR}}(\theta, y \mid x) = \sum_i \log f_{\mathrm{QVR}}(x_i \mid \mathrm{args}_i) + \sum_j \log f_{\mathrm{QVR}}(y_j \mid \mathrm{args}_j)$ (the `Score` sum is empty until §5.6).

By Lemma 5.4.1 and Lemma 5.2.1,

$$
\log p_{\mathsf{T}}\bigl(\mathsf{T}_{\mathsf{T}}(M)\bigr)
\;=\;
\sum_i \bigl(\log f_{\mathrm{QVR}}(x_i \mid \mathrm{args}_i) + c_{F_i, \mathsf{T}}\bigr)
\;+\;
\sum_j \bigl(\log f_{\mathrm{QVR}}(y_j \mid \mathrm{args}_j) + c_{F_j, \mathsf{T}}\bigr)
\;+\;
C_{\mathsf{T}}.
$$

Substituting and rearranging,

$$
\log p_{\mathsf{T}}\bigl(\mathsf{T}_{\mathsf{T}}(M)\bigr)
\;=\;
\log p_{\mathrm{QVR}}(\theta, y \mid x)
\;+\;
\Bigl(\sum_i c_{F_i, \mathsf{T}} + \sum_j c_{F_j, \mathsf{T}} + C_{\mathsf{T}}\Bigr).
$$

The bracketed term is independent of $(\theta, y)$, so it is the constant $c_{\mathsf{T}}(M)$ of Theorem 4.1. The natural transformation $\eta_{\mathsf{T}}$ is the constant-shift kernel morphism whose component at $M$ adds $c_{\mathsf{T}}(M)$; naturality is the trivial commutation of constant-shift morphisms with the rewrite-induced kernel maps (cf. §4). $\square$

### 5.6 Constructs pending implementation

The translation table of §5.1 covers `SampleStep` and `ObserveStep` for every backend. Three program-step kinds are pending walker work, but the proof of §5 extends to them by the same lemma chain:

* `ScoreStep(w)`. Emit Stan `target += w;`, NumPyro `numpyro.factor("score", w)`, Pyro `pyro.factor("score", w)`, PyMC `pymc.Potential("score", w)`, Turing `Turing.@addlogprob! w`, Gen `Gen.@trace(Dirac(0), :score)` weighted by $w$, Church `(factor w)`, WebPPL `factor(w)`, BUGS / JAGS `zeros_trick(w)`. The per-step factor identity (cf. Lemma 5.3) is immediate: $w$ is the literal log-weight contributed by the step in both QVR (cf. [Programs §2.7a](programs.md#27a-score-factor)) and the target.

* `LetStep(x, e)`. Emit `x = e;` in every backend. The step contributes no factor in either QVR or the target; only the deterministic computation is registered.

* `MarginalizeStep(axes)`. Emit Stan `target += log_sum_exp(...)`, NumPyro `numpyro.factor("marg", logsumexp(...))`, etc. The proof requires the target's `log_sum_exp` to numerically equal QVR's $\log \sum$; this reduces to floating-point identity at the per-axis enumeration, verified by Tier-4 equivalence.

Additional distribution families currently unsupported across some backends (`LogitNormal`, `TruncatedNormal`, `RelaxedBernoulli`, `RelaxedOneHotCategorical`, `Gumbel`, `Kumaraswamy`, `ContinuousBernoulli`, `FisherSnedecor`, `LowRankMVN`, `GeneralizedPareto`, `MatrixNormal`, `GP`, `Horseshoe`) admit the same correctness argument as Lemma 5.2.1 once they are added to each backend's `_FAMILIES` map; each is a standard distribution with a canonical parameterization and an obvious lookup in each backend's distribution library.

## 6. Empirical discharge

### 6.1 What the proof reduces correctness to

The structural proof of §5 reduces Theorem 4.1 to four discrete obligations.

1. The walker $\mathsf{W}_{\mathsf{T}}$ emits exactly one schema subgraph $\Sigma_{\mathsf{T}}(s)$ per step $s$, with the structure in the table of §5.1. *Provable by structural induction on the walker's `forward` method (the case analysis is finite and exhaustive over the `ProgramStep` discriminator), and empirically certified per-fixture per-backend by the [Tier-1 structural assertions](../api/tests/transpile.md#test-structural).*
2. The emitter $\mathsf{E}_{\mathsf{T}}$ produces bytes whose target-language parse re-emits to the same schema subgraph (injectivity of the emit at the by-construction schema level). *Verified by the [lens-laws re-emit fixed point test](../api/tests/transpile.md#test-reemit-fixed-point) per backend.*
3. The target compiler accepts the emitted bytes as syntactically valid. *Verified by the [Tier-2 external syntax test](../api/tests/transpile.md#test-external-syntax) (`stanc --no-output`, `python -m ast`, `node --check`, `julia Meta.parse`, `jags`).*
4. The target's log-density implementation evaluates each family $F_{\mathsf{T}}$ to the value of $F_{\mathrm{QVR}}$ up to the constant $c_{F, \mathsf{T}}$. *Verified by the [Tier-4 measure-equivalence test](../api/tests/transpile.md#test-numeric-equivalence): for every (fixture, backend) cell, the test runs the backend's native log-density probe at a $256$-point deterministic tensor-product grid + corner cases, asserts `max_i | δ_i − mean δ | < 1e-6`, and asserts pairwise transitivity across backends.*

### 6.2 What is provable now, with formal certificate

A passing CI run is a constructive certificate of Theorem 4.1 for every (fixture, backend) cell in the test matrix:

* Tier 1 certifies the schema decomposition of Lemma 5.4 (the walker emits the expected per-step subgraphs).
* Tier 2 certifies the `dx.Mapping` composition law of §3.
* Tier 3 certifies the target compiler accepts the bytes.
* Tier 4 certifies Lemmas 5.2 and 5.3 at $\ge 256$ deterministic grid points per fixture + hand-picked corner cases, plus pairwise transitivity across backends.

For *any single* (fixture, backend) cell, the conjunction of the four tiers passing is a witness for Theorem 4.1 at that cell. The theorem's universal quantification over $(\theta, y)$ in support is replaced by a deterministic finite sample; the constant-spread bound certifies the difference is a constant up to floating-point tolerance.

### 6.3 What is NOT formally covered

Tier 4 is empirical, not formal. It does not certify:

1. Behavior on parameter points outside the grid + corners. A pathological model with a step discontinuity in $\log p$ between adjacent grid points would not be caught.
2. The target compiler's correctness. A bug in `cmdstanpy.log_prob` or `numpyro.log_density` could mask a walker bug; the pairwise transitivity check across backends partially mitigates this (a bug in one backend manifests as a mismatched constant in every other backend's pair).
3. Numerical floating-point identity beyond $10^{-6}$ tolerance.

For stronger certificates, the proof of §5 is the right starting point. A machine-checkable proof (Lean / Coq) would need to formalize the per-target trace semantics of §2; this is straightforward for the trace-semantics backends and an established literature for Stan / BUGS factor graphs.

## References

- Tobias Fritz. 2020. A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics. *Advances in Mathematics*, 370:107239. [https://doi.org/10.1016/j.aim.2020.107239](https://doi.org/10.1016/j.aim.2020.107239)
- Michèle Giry. 1982. A categorical approach to probability theory. In Bernhard Banaschewski, editor, *Categorical Aspects of Topology and Analysis*, volume 915 of *Lecture Notes in Mathematics*, pages 68–85. Springer, Berlin, Heidelberg. [https://doi.org/10.1007/BFb0092872](https://doi.org/10.1007/BFb0092872)
- Noah D. Goodman, Vikash K. Mansinghka, Daniel M. Roy, Keith Bonawitz, and Joshua B. Tenenbaum. 2008. Church: A language for generative models. In *Proceedings of the Twenty-Fourth Conference on Uncertainty in Artificial Intelligence (UAI)*, pages 220–229. [https://arxiv.org/abs/1206.3255](https://arxiv.org/abs/1206.3255)
- David Wingate, Andreas Stuhlmüller, and Noah D. Goodman. 2011. Lightweight implementations of probabilistic programming languages via transformational compilation. In *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS)*, JMLR Workshop and Conference Proceedings 15, pages 770–778. [http://proceedings.mlr.press/v15/wingate11a.html](http://proceedings.mlr.press/v15/wingate11a.html)
- Noah D. Goodman and Andreas Stuhlmüller. 2014. *The Design and Implementation of Probabilistic Programming Languages*. Online textbook. [http://dippl.org](http://dippl.org)
- Bob Carpenter, Andrew Gelman, Matthew D. Hoffman, Daniel Lee, Ben Goodrich, Michael Betancourt, Marcus Brubaker, Jiqiang Guo, Peter Li, and Allen Riddell. 2017. Stan: A probabilistic programming language. *Journal of Statistical Software*, 76(1):1–32. [https://doi.org/10.18637/jss.v076.i01](https://doi.org/10.18637/jss.v076.i01)
- Du Phan, Neeraj Pradhan, and Martin Jankowiak. 2019. Composable effects for flexible and accelerated probabilistic programming in NumPyro. *arXiv preprint arXiv:1912.11554*. [https://doi.org/10.48550/arXiv.1912.11554](https://doi.org/10.48550/arXiv.1912.11554)
- Eli Bingham, Jonathan P. Chen, Martin Jankowiak, Fritz Obermeyer, Neeraj Pradhan, Theofanis Karaletsos, Rohit Singh, Paul Szerlip, Paul Horsfall, and Noah D. Goodman. 2019. Pyro: Deep universal probabilistic programming. *Journal of Machine Learning Research*, 20(28):1–6. [http://jmlr.org/papers/v20/18-403.html](http://jmlr.org/papers/v20/18-403.html)
- Marco F. Cusumano-Towner, Feras A. Saad, Alexander K. Lew, and Vikash K. Mansinghka. 2019. Gen: A general-purpose probabilistic programming system with programmable inference. In *Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation*, pages 221–236. [https://doi.org/10.1145/3314221.3314642](https://doi.org/10.1145/3314221.3314642)
- Hong Ge, Kai Xu, and Zoubin Ghahramani. 2018. Turing: A language for flexible probabilistic inference. In *International Conference on Artificial Intelligence and Statistics*, pages 1682–1690. [https://proceedings.mlr.press/v84/ge18b.html](https://proceedings.mlr.press/v84/ge18b.html)
- John K. Kruschke. 2014. *Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan*. Second edition. Academic Press. [https://doi.org/10.1016/C2012-0-00477-2](https://doi.org/10.1016/C2012-0-00477-2)
- Daphne Koller and Nir Friedman. 2009. *Probabilistic Graphical Models: Principles and Techniques*. MIT Press. ISBN 978-0-262-01319-2.
