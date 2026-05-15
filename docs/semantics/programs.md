# Programs

The `program` block is the monadic sublanguage of QVR. A `program` declaration

```
program P (params) : τ₁ -> τ₂
    s₁
    s₂
    ⋮
    sₙ
    return e
```

denotes a Markov kernel

$$
\llbracket P \rrbracket : \llbracket \tau_1 \rrbracket \to \mathcal{G}\bigl(\llbracket \tau_2 \rrbracket\bigr),
$$

equivalently a morphism in $\mathbf{Kern}$ when $\tau_2$ is continuous, or in $\mathbf{Stoch}$ when both $\tau_1$ and $\tau_2$ are discrete. Here $\mathcal{G}$ denotes the (continuous or discrete) Giry monad as appropriate.

The block body is a sequence of statements interpreted in the *Kleisli* category of $\mathcal{G}$, with `let` and `return` providing the monad's internal language.

## 1. The Giry monad as semantic substrate

Let $\mathcal{G}$ denote the Giry monad on $\mathbf{SBor}$, with unit $\eta_S : S \to \mathcal{G}(S)$ given by $s \mapsto \delta_s$ (Dirac at $s$) and multiplication $\mu_S : \mathcal{G}(\mathcal{G}(S)) \to \mathcal{G}(S)$ given by integration. The Kleisli category $\mathbf{Kern}$ of $\mathcal{G}$ has the same objects as $\mathbf{SBor}$ and morphisms $S \to T$ given by Markov kernels $S \to \mathcal{G}(T)$.

A QVR program is denoted by a single morphism in $\mathbf{Kern}$, built compositionally by interpreting each statement as a *Kleisli arrow* and composing them via Kleisli composition $\diamond$. For $k_1 : S \to \mathcal{G}(T)$ and $k_2 : T \to \mathcal{G}(U)$:

$$
(k_1 \diamond k_2)(s, C) \;=\; \mu_U \bigl( \mathcal{G}(k_2)(k_1(s)) \bigr)(C)
\;=\; \int_T k_2(t, C) \, k_1(s, \mathrm{d}t).
$$

We extend a program-body environment to track *random variables*: for a pre-fixed program domain $\Gamma$ and a current statement-context $\Phi = (X_1, \dots, X_k)$, every random variable bound earlier in the body has a Kleisli arrow

$$
\rho_{\mathrm{rv}}(v) : \Gamma \to \mathcal{G}(\Phi).
$$

The body-level denotation function is

$$
\mathcal{B}\llbracket s_1 \,;\, \cdots \,;\, s_n \,;\, \mathsf{return}\ e \rrbracket : \Gamma \to \mathcal{G}(\llbracket \tau_2 \rrbracket).
$$

## 2. Statements

We give the denotation of each statement form as a Kleisli arrow on the program's accumulated random-variable context $\Phi$. Concretely, the body is interpreted as the Kleisli composite

$$
\mathcal{B}\llbracket s_1; \cdots; s_n; \mathsf{return}\ e \rrbracket
\;=\; \mathcal{S}\llbracket s_1 \rrbracket \diamond \mathcal{S}\llbracket s_2 \rrbracket \diamond \cdots \diamond \mathcal{S}\llbracket s_n \rrbracket \diamond \mathsf{ret}_e,
$$

where each $\mathcal{S}\llbracket s_i \rrbracket : \Phi_{i-1} \to \mathcal{G}(\Phi_i)$ is the Kleisli arrow assigned to statement $s_i$ (with $\Phi_0 = \Gamma$), and $\mathsf{ret}_e : \Phi_n \to \mathcal{G}(\llbracket \tau_2 \rrbracket)$ is the deterministic Kleisli arrow $\eta \circ \pi_e$ projecting onto the components named by the `return` clause.

### 2.1 Bind

A bind statement

```
v <- F(args)
```

denotes the Kleisli arrow extending the context with a fresh random variable distributed according to family $F$:

$$
\mathcal{S}\llbracket v \leftarrow F(\bar a) \rrbracket : \Phi \to \mathcal{G}\bigl(\Phi \times \llbracket \mathsf{cod}(F) \rrbracket\bigr),
$$

defined on measurable rectangles $B \times C$ (with $B \subseteq \Phi$, $C \subseteq \llbracket \mathsf{cod}(F) \rrbracket$) by

$$
\mathcal{S}\llbracket v \leftarrow F(\bar a) \rrbracket(\phi,\, B \times C)
\;=\;
\mathbf{1}_B(\phi) \cdot \int_C p_F\bigl( y \,;\, \theta_F(\bar a, \phi) \bigr)\, \mathrm{d}y,
$$

where $\theta_F$ is the family's parameter map (which may depend on previously-bound variables in $\phi$). In short: keep the current trace $\phi$ and append a fresh sample from $F$ conditioned on it. The induced action on measures over $\Phi$ is $\mu_{\Phi \times \mathsf{cod}(F)} \circ \mathcal{G}\bigl(\mathcal{S}\llbracket \mathsf{bind} \rrbracket\bigr)$.

### 2.2 Observe

An observe statement

```
observe v <- F(args)
```

denotes a *score* update against an externally-supplied observed value $v_{\mathrm{obs}}$. As a Kleisli arrow in the *unnormalized* Giry monad $\mathcal{G}_{\le 1}$ (sub-probability measures),

$$
\mathcal{S}\llbracket \mathsf{observe}\ v \leftarrow F(\bar a) \rrbracket : \Phi \to \mathcal{G}_{\le 1}(\Phi),
\qquad
\mathcal{S}\llbracket \mathsf{observe}\ v \leftarrow F(\bar a) \rrbracket(\phi,\, B) \;=\; \mathbf{1}_B(\phi) \cdot p_F\bigl( v_{\mathrm{obs}} \,;\, \theta_F(\bar a, \phi)\bigr).
$$

The trace context is preserved, but the total mass of the resulting measure is the likelihood of $v_{\mathrm{obs}}$ at $\phi$. Normalization and posterior inference are deferred to the inference layer (see [`quivers.inference`](../api/inference/svi.md)). The categorical setting is the *Markov category with conditioning* of [Cho & Jacobs 2019](https://doi.org/10.1017/S0960129518000488) and [Fritz 2020](https://doi.org/10.1016/j.aim.2020.107239).

### 2.3 Let

A let statement

```
let v = expr
```

denotes a *deterministic* extension of the context. The right-hand side `expr` is an arithmetic / function-application expression over previously-bound names; it denotes a measurable map $h : \Phi \to T$, and the let statement is the Kleisli arrow

$$
\mathcal{S}\llbracket \mathsf{let}\ v = \mathit{expr} \rrbracket : \Phi \to \mathcal{G}(\Phi \times T),
\qquad
\mathcal{S}\llbracket \mathsf{let}\ v = \mathit{expr} \rrbracket(\phi) \;=\; \delta_{(\phi,\, h(\phi))},
$$

a Dirac kernel. Equivalently, on rectangles $B \times C$:

$$
\mathcal{S}\llbracket \mathsf{let}\ v = \mathit{expr} \rrbracket(\phi,\, B \times C)
\;=\; \mathbf{1}_B(\phi) \cdot \mathbf{1}_C\bigl(h(\phi)\bigr),
$$

i.e.\ pushforward by $\mathrm{id}_{\Phi} \times h$ realized through the *strength* of the Giry monad.

The arithmetic sublanguage is interpreted standardly: $\mathbb{R}$-valued and $\mathbb{N}$-valued operators denote the corresponding measurable functions on the relevant space, and built-in functions denote the corresponding total measurable maps.

#### 2.3.1 Built-in primitives

The let-expression call form `f(arg, ...)` resolves first against a fixed table of tensor primitives drawn from `torch.nn.functional` and `torch`. The table is exported as [`_LET_EXPR_BUILTINS`](../api/dsl/compiler.md) for introspection. Each primitive denotes the total measurable map of the same name; reductions take `dim=-1` by convention (reductions over a *named* axis go through the typed [`contraction`](../api/dsl/compiler.md) surface instead).

| Category | Primitives |
| --- | --- |
| ReLU family | `relu`, `relu6`, `leaky_relu`, `prelu`, `rrelu`, `elu`, `selu`, `celu`, `gelu` |
| Smooth gates | `silu` (alias `swish`), `mish`, `hardsigmoid`, `hardswish`, `hardtanh`, `hardshrink`, `softplus`, `softshrink`, `softsign` |
| Sigmoidal | `sigmoid`, `logsigmoid`, `tanh`, `tanhshrink`, `threshold`, `glu` |
| Probability-simplex | `softmax`, `log_softmax`, `softmin`, `normalize` |
| Transcendentals | `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `sqrt`, `rsqrt`, `square`, `abs`, `neg`, `sign`, `reciprocal`, `clamp` |
| Trigonometric / hyperbolic | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh` |
| Rounding | `floor`, `ceil`, `round`, `trunc` |
| Special functions | `erf`, `erfc`, `erfinv`, `lgamma`, `digamma` |
| Reductions (`dim=-1`) | `sum`, `mean`, `var`, `std`, `min`, `max`, `argmin`, `argmax`, `prod`, `amax`, `amin`, `logsumexp`, `norm` |
| Cumulative / ordering | `cumsum`, `cumprod`, `cummax`, `cummin`, `flip`, `sort` |
| Training-mode | `dropout`, `alpha_dropout`, `layer_norm`, `rms_norm` |

<!-- compile: false -->
```qvr
# Illustrative: each name below is a let-expression primitive,
# not a top-level morphism, so this block is not standalone-compilable.
softmax(x)         # softmax over the last axis
gelu(x)            # smooth gate
sum(x)             # dim=-1 reduction
```

Two names overload between the table and the higher-order combinator pool: `logsumexp(a, b, c, ...)` reduces over an explicit stack of scalar/tensor arguments rather than along `dim=-1`; the variadic form wins on dispatch.

#### 2.3.2 User-defined callables

Calls in let bodies also resolve against the program's own [`_morphisms`](../api/dsl/compiler.md), [`_encoders`](../api/dsl/compiler.md), [`_decoders`](../api/dsl/compiler.md), [`_deductions`](../api/dsl/compiler.md), and [`_signatures`](../api/dsl/compiler.md) tables. A deterministic [`program`](programs.md) is a [Dirac](https://en.wikipedia.org/wiki/Dirac_measure) [Kleisli arrow](https://en.wikipedia.org/wiki/Kleisli_category) embedding [Smooth](https://en.wikipedia.org/wiki/Smooth_manifold) into [Kleisli](https://en.wikipedia.org/wiki/Kleisli_category)$(\mathcal{G})$; calling it from an encoder rule body composes the two Smooth pieces and stays in Smooth.

```qvr
signature Seq {
    sorts {
        Seq : object dim 64
        A   : data   dim 64
    }
    constructors {
        Nil  :        -> Seq
        Cons : A, Seq -> Seq
    }
}

encoder C over Seq {
    dim Seq = 64
    Nil                              |-> 0.0
    Cons(head, tail) recurrent state |-> gelu(head + state)
}
```

The dispatcher consults builtins, then user-defined callables, then the user-declared constructor set (for free-term algebra construction). Builtins shadow user-injected names with the same identifier.

#### 2.3.3 Arity and shape checking

At compile time, the let-expression compiler checks the positional arity of every user-defined callable against the call site:

* a [`Morphism`](../api/core/morphisms.md) is unary (takes the domain tensor);
* a [`MonadicProgram`](../api/continuous/programs.md) with named `params` is `len(params)`-ary, otherwise unary;
* any other callable is introspected through [`inspect.signature`](https://docs.python.org/3/library/inspect.html#inspect.signature), counting positional parameters without defaults; `*args` makes the arity unknowable, in which case the check is skipped.

Tensor-shape mismatches inside a user-defined callable surface as `RuntimeError` from PyTorch; the dispatcher wraps these (and any `TypeError`) into a [`CompileError`](../api/dsl/compiler.md) that names the call site, so the diagnostic is `call to 'L' failed: ...` rather than a bare PyTorch trace.

### 2.4 Indexed Bind (Plate)

An indexed bind

```
v : A <- F(args)
```

declares $v$ as an $A$-indexed plate of independent $F$-draws. The per-fiber codomain $K = \mathsf{cod}(F)$ is taken from the family. The natural isomorphism

$$
\mathbf{Kern}(\mathbf{1}, K^A) \;\cong\; \mathbf{Kern}(A, K)
$$

identifies a single $\mathcal{G}(K^A)$-valued draw with an $A$-indexed family of $\mathcal{G}(K)$-valued draws. The statement therefore denotes the context-extending Kleisli arrow

$$
\mathcal{S}\llbracket v : A \leftarrow F(\bar a) \rrbracket : \Phi \to \mathcal{G}\bigl(\Phi \times K^A\bigr),
$$

with density $\prod_{a \in A} p_F\bigl(v(a) \,;\, \theta_F(\bar a, \phi)\bigr)$ on the appended coordinate.

### 2.5 Indexed Observe

An indexed-observe statement

```
observe r : N <- F(args)
```

denotes a sub-probabilistic Kleisli arrow in $\mathcal{G}_{\le 1}$,

$$
\mathcal{S}\llbracket \mathsf{observe}\ r : N \leftarrow F(\bar a) \rrbracket : \Phi \to \mathcal{G}_{\le 1}(\Phi),
\qquad
\phi \;\longmapsto\; \mathbf{1}_{(\cdot)}(\phi) \cdot \prod_{n \in N} p_F\bigl( r_{\mathrm{obs}}(n) \,;\, \theta_F(\bar a, n, \phi) \bigr).
$$

Bracket-indexed family arguments `theta[N]` in $\bar a$ pick out the $N$-section of a previously-bound plate variable. The response buffer $r_{\mathrm{obs}} : N \to \llbracket \mathsf{cod}(F) \rrbracket$ is supplied externally by the inference layer; the trace context is preserved and the total mass of the resulting measure is the batched likelihood.

### 2.6 Marginalize

A scoped marginalize statement

```
marginalize c : A <- F(args) in { s₁; …; sₖ }
```

introduces the coordinate $c$ bound to $F(\bar a)$, optionally $A$-indexed, with $s_1; \ldots; s_k$ as its integration scope. After interpreting the scope body, the accumulated (sub-)probability measure on $\Phi \times C$ is pushed forward through the projection $\pi_{\Phi} : \Phi \times C \to \Phi$:

$$
\mathcal{S}\llbracket \mathsf{marginalize}\ c \rrbracket : \mathcal{G}_{\le 1}(\Phi \times C) \to \mathcal{G}_{\le 1}(\Phi),
\qquad
\nu \;\longmapsto\; \pi_{\Phi *} \nu.
$$

The denotation is the pushforward $\pi_{\Phi *}$ in both the discrete and continuous cases. Operationally, the implementation realises the pushforward by log-sum-exp on the accumulated log-likelihood when $C$ is a finite-set latent, and by fibrewise integration (e.g. by sampling, when the family admits a reparameterised draw) when $C$ is a continuous space. After the scope closes, $c$ falls out of scope.

The four bind variants, scalar, indexed, scored, marginalized, are uniformly a single underlying step with a `mode ∈ {sample, score, marginal}` tag and an optional index `A`. The scalar/plate axis is orthogonal to the full-probability/sub-probability distinction.

### 2.7 Grouped marginalize with multi-observe fibration

The marginalize step admits a *grouping clause* that turns the body into a fibered scoring problem over a shared plate.  The surface is

```
marginalize c : K <- F(args) over G [reduction = R] in {
    observe r_1 : N_1 via idx_1 <- F_1(...)
    observe r_2 : N_2 via idx_2 <- F_2(...)
    …
}
```

with a single `over G` clause on the header (or `over G_1 * G_2 * …` for a product grouping plate) and a `via idx_m` clause on each observe naming a fibration $\iota_m : N_m \to G$ from that observe's response plate into the shared grouping plate.

Categorically: the body declares a coproduct fibration $\coprod_m r_m : \coprod_m N_m \to G \times K$, and the marginalize step is the right Kan extension along $\pi_G : G \times K \to G$ followed by an aggregation $R$ over the $K$ axis:

$$
\Sigma_g \;\mathrm{aggr}_R\!\!\bigl[\log \pi(g, k) + \textstyle\sum_m \sum_{n \,:\, \iota_m(n) = g} \ell_m(n, k)\bigr],
$$

where $\ell_m(n, k) = \log p_{F_m}\bigl(r^{\mathrm{obs}}_m(n);\, \theta_m(n, k, \phi)\bigr)$ is the per-row per-class log-likelihood of observe $m$, $\pi$ is the per-group per-class prior weight, and $\mathrm{aggr}_R \in \{\mathrm{logsumexp}, \mathrm{sum}, \mathrm{mean}\}$ is the reduction selected by the optional `reduction = R` annotation (default `logsumexp`, the canonical mixture-marginalization form).

The product-grouping case `over G_1 * G_2 * …` paired with `via product(idx_1, idx_2, …)` on each observe extends the right-Kan-extension target to a flat plate of cardinality $\prod_i |G_i|$; the surface arity must match.

### 2.8 Effect signatures

A `program` declaration may carry an *effect signature* after `!`:

```
program P (params) : τ₁ -> τ₂ ! E₁, E₂, …
    body
```

where each $E_i$ is one of $\{\mathsf{Sample}, \mathsf{Score}, \mathsf{Marginal}, \mathsf{Pure}\}$. The signature is a *static type* over the program: a subset of an *effect algebra* $\mathcal{E}$ that the body's statements collectively produce.

Each statement form contributes an effect:

| Statement form | Effect produced |
|---|---|
| `v <- F(args)` | $\mathsf{Sample}$ |
| `v : A <- F(args)` | $\mathsf{Sample}$ |
| `observe v <- F(args)` | $\mathsf{Score}$ |
| `observe r : N <- F(args)` | $\mathsf{Score}$ |
| `marginalize c <- F(args) in { … }` | $\mathsf{Marginal}$ (plus the effects of the scope body) |
| `let v = expr` | $\mathsf{Pure}$ |
| `return e` | $\mathsf{Pure}$ |

The compiler computes the *actual* effect set $\mathcal{E}(P)$ of the body and verifies $\mathcal{E}(P) \subseteq \mathcal{E}_{\mathrm{decl}}$. The signature $\{\mathsf{Pure}\}$ in particular rejects any sample / score / marginal statement, restricting the body to `let` (and a `marginalize` whose own scope is itself pure).

Categorically, effects index the codomain monad of the program's denotation: $\mathsf{Pure}$ programs denote ordinary measurable maps $\tau_1 \to \tau_2$; $\mathsf{Sample}$ programs denote Kleisli arrows in $\mathcal{G}$; $\mathsf{Score}$ programs land in $\mathcal{G}_{\le 1}$; $\mathsf{Marginal}$ programs commute with right Kan extensions along discrete fibrations. The effect-set inclusion is therefore a soundness condition on the monad: the actual codomain monad must be a sub-monad of the declared one.

The `over <model>` clause on a program header marks the program as consuming the named model's latents: the consumed coordinates appear as data parameters and the body is restricted to $\mathsf{Pure}$ (a *posterior consumer*, the deterministic Kleisli arrow $\Theta \to \tau_2$ that lifts to $\mathrm{Data} \to \mathcal{G}(\tau_2)$ by post-composition with the model's posterior kernel).

### 2.9 Indexed Gather (Let-Pullback)

A `let` right-hand side of the form `arr[idx]` is the *Kleisli pullback*. For a plate variable $v : A \to \mathcal{G}(B)$ bound earlier in the body, and a finite fibration $\iota : N \to A$ named in the context, the gather $\iota^* v$ is the composite

$$
\iota^* v \;=\; v \circ \iota \;:\; N \to \mathcal{G}(B).
$$

Interpreted as a deterministic measurable map on the accumulated context (because $v$ has already been realized as a tensor $A \to B$ in the trace), the let-step denotes the Dirac extension

$$
\mathcal{S}\llbracket \mathsf{let}\ w = \mathit{arr}[\mathit{idx}] \rrbracket(\phi)
\;=\;
\delta_{(\phi,\, \phi.\mathit{arr}[\phi.\mathit{idx}])}.
$$

### 2.10 Return

A return statement

```
return e
```

closes the body. If $e = (v_1, \dots, v_m)$ is a tuple of bound names, the return clause is the deterministic Kleisli arrow

$$
\mathsf{ret}_e : \Phi_n \to \mathcal{G}\bigl(\llbracket \tau_2 \rrbracket\bigr),
\qquad
\mathsf{ret}_e(\phi) \;=\; \delta_{\pi_{v_1, \dots, v_m}(\phi)},
$$

where $\pi_{v_1, \dots, v_m} : \Phi_n \to \llbracket \tau_2 \rrbracket$ projects the trace onto the named coordinates. Composing with the body chain marginalizes the joint (sub-)probability measure onto those coordinates.

A bare-tuple return `return (x, y)` projects the trace onto the named coordinates; the resulting product space's components are ordered by tuple position.

## 3. Data parameters

A program declared with bare-identifier parameters

```
program P (q₁, …, qₖ) : τ₁ -> τ₂
    body
```

names the components of the domain $\tau_1$: when $\tau_1 = \sigma_1 \times \cdots \times \sigma_k$ is a $k$-fold product, each $q_i$ binds to the projection $\pi_i$ of the input. The denotation is unchanged from the unparameterised form,

$$
\llbracket P \rrbracket : \llbracket \tau_1 \rrbracket \to \mathcal{G}(\llbracket \tau_2 \rrbracket),
$$

i.e.\ a single morphism in $\mathbf{Kern}$; the $q_i$ are syntactic conveniences in the body, not additional dependent parameters. Typed parameters, covered in §3a below, extend this to dependent kernel families.

## 3a. Parametric programs

A program whose parameter list contains *typed* parameters denotes a *dependent* family of Kleisli arrows. With parameters $p_i : P_i$ drawn from the universes

| Parameter kind | Universe $P_i$ |
|---|---|
| `FinSet`, `Space`, `Object` | an object of the relevant subcategory of $\mathbf{Kern}$ |
| `Real`, `Nat` | a hom-object of scalar type (a hyperparameter) |
| `Mor[A, B]` | the hom-set $\mathbf{Kern}(A, B)$ |

the denotation lives in the dependent kernel space

$$
\llbracket P \rrbracket \;:\; \prod_{p_1 : P_1} \cdots \prod_{p_k : P_k} \mathbf{Kern}\bigl(\mathrm{dom}(p), \mathrm{cod}(p)\bigr),
$$

an object of the indexed family of Kleisli arrows over the parameter category. The domain and codomain may themselves mention the formal parameters, so each fiber is a kernel between possibly-different objects of $\mathbf{Kern}$.

### Inline expansion as substitution

A call site `v <- P(a₁, …, aₖ)` inside another program is interpreted by *substitution* on the dependent denotation: the actual arguments $a_i$ are substituted for the formal parameters $p_i$ in the body of $P$, yielding a closed Kleisli arrow which is then inlined as a sequence of statements into the caller's body. Internal latents are α-renamed under a fresh prefix $v\$$, and the return-variable is renamed to $v$ directly; the result is a well-typed sequence of caller-level Kleisli arrows.

This is sound by a standard substitution lemma: because each formal parameter is bound at the top of the body and the body interprets to a Kleisli arrow built compositionally from its statements, substitution commutes with the body's denotation function $\mathcal{B}\llbracket \cdot \rrbracket$. The α-renaming step is sound because the body's denotation depends only on the multiset of bound-variable types, not on the names. Two call sites of the same template therefore contribute *distinct* factors to the caller's joint kernel, fresh latents per use, recovering the standard "plate-of-plates" semantics for hierarchical models.

## 4. Composition of programs

Two programs $P : X \to Y$ and $Q : Y \to Z$ compose by Kleisli composition:

$$
\llbracket P \mathbin{>\!\!>} Q \rrbracket(x, C) \;=\; \int_Y \llbracket Q \rrbracket(y, C) \, \llbracket P \rrbracket(x, \mathrm{d}y).
$$

The DSL exposes this through a top-level `let` binding the composition and an `export` declaration naming the composite:

<!-- compile: false -->
```qvr
let pq = p >> q
export pq
```

`export` is the public-binding form: any number of `export` declarations per module are allowed, each producing a separate compiled program output.

## 5. Soundness of monadic semantics

The interpretations above satisfy the standard monadic equations:

| Equation | Statement |
|---------|-----------|
| Left unit | $\eta \diamond k = k$ |
| Right unit | $k \diamond \eta = k$ |
| Associativity | $(k_1 \diamond k_2) \diamond k_3 = k_1 \diamond (k_2 \diamond k_3)$ |
| Strength coherence | $\mathrm{str} \circ \mathcal{G}(\sigma) = \sigma' \circ \mathrm{str}$ |

These are valid statements about denotations of QVR programs; in particular, the order in which independent draws are listed in the body is irrelevant to the denotation, by the symmetry of the product measure.

## 6. Inference and conditioning

The denotation of a program is a *kernel*, not yet a posterior. Conditioning on observed data, normalization, and approximate posterior inference are *external* operations on the denotation, supplied by the [`quivers.inference`](../api/inference/svi.md) module. The categorical apparatus is that of *Markov categories with conditionals* ([Cho & Jacobs 2019](https://doi.org/10.1017/S0960129518000488); [Fritz 2020](https://doi.org/10.1016/j.aim.2020.107239)); the implementation realizes trace-based conditioning and stochastic variational inference as concrete instances of that theory.
