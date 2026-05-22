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

denotes a Markov kernel (the typing rules for statements and the program declaration live in [Typing §6–§7](typing.md#6-inference-rules-for-statements))

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

#### 2.1.1 Destructuring bind

The bind statement admits a *tuple pattern* on the left-hand side:

```
(v_1, …, v_m) <- F(args)
```

with denotation identical to the scalar bind above except that the codomain $\llbracket \mathsf{cod}(F) \rrbracket = K_1 \times \cdots \times K_m$ must be an $m$-fold product and the trace is extended with $m$ named coordinates rather than a single one. Subsequent statements may reference each $v_i$ as if it had been bound separately by $v_i \leftarrow \pi_i \circ F(\bar a)$. The two forms have the same denotation when the family's codomain is a product type; the destructuring form gives the components names.

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

```text
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
observe r : N <- F(args) [via = idx]
```

denotes a sub-probabilistic Kleisli arrow in $\mathcal{G}_{\le 1}$,

$$
\mathcal{S}\llbracket \mathsf{observe}\ r : N \leftarrow F(\bar a) \rrbracket : \Phi \to \mathcal{G}_{\le 1}(\Phi),
\qquad
\phi \;\longmapsto\; \mathbf{1}_{(\cdot)}(\phi) \cdot \prod_{n \in N} p_F\bigl( r_{\mathrm{obs}}(n) \,;\, \theta_F(\bar a, n, \phi) \bigr).
$$

Bracket-indexed family arguments `theta[N]` in $\bar a$ pick out the $N$-section of a previously-bound plate variable. The response buffer $r_{\mathrm{obs}} : N \to \llbracket \mathsf{cod}(F) \rrbracket$ is supplied externally by the inference layer; the trace context is preserved and the total mass of the resulting measure is the batched likelihood.

The optional `via = idx` entry in the unified option block names a fibration into a grouping plate ([§2.7](#27-grouped-marginalize-with-multi-observe-fibration)); the bare-statement form omits it.

### 2.6 Marginalize

A scoped marginalize statement

```
marginalize c : A <- F(args)
    s₁
    s₂
    ⋮
    sₖ
```

introduces the coordinate $c$ bound to $F(\bar a)$, optionally $A$-indexed, with $s_1; \ldots; s_k$ as its integration scope. After interpreting the scope body, the accumulated (sub-)probability measure on $\Phi \times C$ is pushed forward through the projection $\pi_{\Phi} : \Phi \times C \to \Phi$:

$$
\mathcal{S}\llbracket \mathsf{marginalize}\ c \rrbracket : \mathcal{G}_{\le 1}(\Phi \times C) \to \mathcal{G}_{\le 1}(\Phi),
\qquad
\nu \;\longmapsto\; \pi_{\Phi *} \nu.
$$

The denotation is the pushforward $\pi_{\Phi *}$ in both the discrete and continuous cases. Operationally, the implementation realizes the pushforward by log-sum-exp on the accumulated log-likelihood when $C$ is a finite-set latent, and by fibrewise integration (e.g. by sampling, when the family admits a reparameterised draw) when $C$ is a continuous space. After the scope closes, $c$ falls out of scope.

The four bind variants, scalar, indexed, scored, marginalized, are uniformly a single underlying step with a `mode ∈ {sample, score, marginal}` tag and an optional index `A`. The scalar/plate axis is orthogonal to the full-probability/sub-probability distinction.

### 2.7 Grouped marginalize with multi-observe fibration

The marginalize step admits a *grouping clause* that turns the body into a fibered scoring problem over a shared plate. The clause lives in the marginalize step's option block, alongside an optional reduction selector; the body's observe steps each carry their own `via = idx` fibration in their option blocks:

```
marginalize c : K <- F(args) [over = G, reduction = R]
    observe r_1 : N_1 <- F_1(...) [via = idx_1]
    observe r_2 : N_2 <- F_2(...) [via = idx_2]
    ⋮
```

with a single `over = G` entry on the marginalize step (or `over = product(G_1, G_2, …)` / `over = [G_1, G_2, …]` for a product grouping plate) and a `via = idx_m` entry on each observe naming a fibration $\iota_m : N_m \to G$ from that observe's response plate into the shared grouping plate.

Categorically: the body declares a coproduct fibration $\coprod_m r_m : \coprod_m N_m \to G \times K$, and the marginalize step is the right Kan extension along $\pi_G : G \times K \to G$ followed by an aggregation $R$ over the $K$ axis:

$$
\Sigma_g \;\mathrm{aggr}_R\!\!\bigl[\log \pi(g, k) + \textstyle\sum_m \sum_{n \,:\, \iota_m(n) = g} \ell_m(n, k)\bigr],
$$

where $\ell_m(n, k) = \log p_{F_m}\bigl(r^{\mathrm{obs}}_m(n);\, \theta_m(n, k, \phi)\bigr)$ is the per-row per-class log-likelihood of observe $m$, $\pi$ is the per-group per-class prior weight, and $\mathrm{aggr}_R \in \{\mathrm{logsumexp}, \mathrm{sum}, \mathrm{mean}\}$ is the reduction selected by the optional `reduction = R` annotation (default `logsumexp`, the canonical mixture-marginalization form).

The product-grouping case `over G_1 * G_2 * …` paired with `via product(idx_1, idx_2, …)` on each observe extends the right-Kan-extension target to a flat plate of cardinality $\prod_i |G_i|$; the surface arity must match.

### 2.8 Effect signatures

A `program` declaration may carry an *effect signature* via the `effects` entry of its option block:

```
program P (params) : τ₁ -> τ₂ [effects = [E₁, E₂, …]]
    body
```

where each $E_i$ is one of $\{\mathsf{Sample}, \mathsf{Score}, \mathsf{Marginal}, \mathsf{Pure}\}$. The signature is a *static type* over the program: a subset of an *effect algebra* $\mathcal{E}$ that the body's statements collectively produce.

Each statement form contributes an effect:

| Statement form | Effect produced |
|---|---|
| `sample v <- F(args)` | $\mathsf{Sample}$ |
| `sample v : A <- F(args)` | $\mathsf{Sample}$ |
| `observe v <- F(args)` | $\mathsf{Score}$ |
| `observe r : N <- F(args)` | $\mathsf{Score}$ |
| `marginalize c <- F(args) [over = G, reduction = R]` (with indented scope body) | $\mathsf{Marginal}$ (plus the effects of the scope body) |
| `let v = expr` | $\mathsf{Pure}$ |
| `score v = expr` | $\mathsf{Score}$ |
| `return e` | $\mathsf{Pure}$ |

The compiler computes the *actual* effect set $\mathcal{E}(P)$ of the body and verifies $\mathcal{E}(P) \subseteq \mathcal{E}_{\mathrm{decl}}$. The signature $\{\mathsf{Pure}\}$ in particular rejects any sample / score / marginal statement, restricting the body to `let` (and a `marginalize` whose own scope is itself pure).

Categorically, effects index the codomain monad of the program's denotation: $\mathsf{Pure}$ programs denote ordinary measurable maps $\tau_1 \to \tau_2$; $\mathsf{Sample}$ programs denote Kleisli arrows in $\mathcal{G}$; $\mathsf{Score}$ programs land in $\mathcal{G}_{\le 1}$; $\mathsf{Marginal}$ programs commute with right Kan extensions along discrete fibrations. The effect-set inclusion is therefore a soundness condition on the monad: the actual codomain monad must be a sub-monad of the declared one.

The `over = <model>` entry in a program's option block marks the program as consuming the named model's latents: the consumed coordinates appear as data parameters and the body is restricted to $\mathsf{Pure}$ (a *posterior consumer*, the deterministic Kleisli arrow $\Theta \to \tau_2$ that lifts to $\mathrm{Data} \to \mathcal{G}(\tau_2)$ by post-composition with the model's posterior kernel).

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

### 2.7a Score (factor)

A score statement

```
score v = expr
```

denotes a Kleisli arrow that simultaneously (i) binds the value of `expr` to a fresh name `v` in the program trace, and (ii) adds that value to the program's running log-joint. As a sub-probabilistic Kleisli arrow in $\mathcal{G}_{\le 1}$,

$$
\mathcal{S}\llbracket \mathsf{score}\ v = \mathit{expr} \rrbracket : \Phi \to \mathcal{G}_{\le 1}(\Phi \times T),
\qquad
\mathcal{S}\llbracket \mathsf{score}\ v = \mathit{expr} \rrbracket(\phi,\, B \times C) \;=\; \mathbf{1}_B(\phi) \cdot \mathbf{1}_C\bigl(h(\phi)\bigr) \cdot \exp\bigl(h(\phi)\bigr),
$$

where $h : \Phi \to \mathbb{R}$ is the measurable map denoted by `expr` (which must denote a scalar log-density). The trace context is extended with the named coordinate, and the total mass of the resulting sub-probability measure is multiplied by $\exp(h(\phi))$.

Score is the Kleisli pendant of [observe](#22-observe), with the log-density supplied directly by an expression rather than via a family's `log_prob` against an externally-supplied observed value. The canonical use is to lift an arbitrary differentiable tensor expression (typically a deduction's chart goal weight, see [Weighted Deduction Fragment §9](grammar.md#9-chart-access-from-program-bodies)) into the program's log-joint. In particular, with $\mathit{expr} = \mathit{chart}.\mathsf{goal\_weight}()$ the program's log-joint matches the sentence's inside log-marginal under the referenced deduction.

The effect contributed by a score step is $\mathsf{Score}$ (the same as `observe`), and the soundness condition $\mathcal{E}(P) \subseteq \mathcal{E}_{\mathrm{decl}}$ in [§2.8](#28-effect-signatures) applies unchanged.

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

`export` is the public-binding form: any number of `export` declarations per module are allowed, each producing a separate compiled program output. A top-level `let N = e [where N_1 = e_1 ...]` declaration is the corresponding *private* binding: it extends the morphism environment with $N \mapsto \llbracket e \rrbracket_\rho$, but $N$ is not part of the module's compiled output unless additionally `export`ed. The optional `where` clause is processed *before* the outer expression and contributes its bindings ($N_1 \mapsto \llbracket e_1 \rrbracket$, …) to the same module-level environment; despite the surface nesting, the `where` clause is not a local scope; its names persist globally after the let-declaration completes. Formally, for a module $M$ with $\mathrm{export}\ E_1, \dots, \mathrm{export}\ E_k$ declarations,

$$
\llbracket M \rrbracket_{\mathrm{exports}} \;=\; \bigl(\llbracket E_1 \rrbracket,\, \dots,\, \llbracket E_k \rrbracket\bigr),
$$

a tuple of compiled morphisms / posteriors / deductions. The expression $E_i$ may be any value-level expression: a top-level morphism name, a program name, a deduction name, an encoder / decoder name, or a let-bound composite. The denotation of each export is the denotation of the underlying expression; `export` itself is a marker for the module-output protocol, not a categorical operation.


## 5. Soundness of monadic semantics

The interpretations above satisfy the standard monadic equations:

| Equation | Statement |
|---------|-----------|
| Left unit | $\eta \diamond k = k$ |
| Right unit | $k \diamond \eta = k$ |
| Associativity | $(k_1 \diamond k_2) \diamond k_3 = k_1 \diamond (k_2 \diamond k_3)$ |
| Strength coherence | $\mathrm{str} \circ \mathcal{G}(\sigma) = \sigma' \circ \mathrm{str}$ |

These are valid statements about denotations of QVR programs; in particular, the order in which independent draws are listed in the body is irrelevant to the denotation, by the symmetry of the product measure.

**Theorem (Monad laws for QVR programs).** *The Kleisli composition $\diamond$ of [Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels) on the kernels produced by the bind / observe / let / score / marginalize / return clauses of §[2](#2-statements) satisfies the four equations above.*

**Proof.** The four equations are the standard monad laws for the Giry monad $\mathcal{G}$, established by [Giry 1982, Theorem 5](https://doi.org/10.1007/BFb0092872): $\mathcal{G}$ is a monad on $\mathbf{SBor}$ with unit $\eta_S(s) = \delta_s$ (the Dirac at $s$) and multiplication $\mu_S(M)(B) = \int_{\mathcal{G}(S)} m(B)\, M(\mathrm{d}m)$ (integration of measures over measures). Kleisli composition $k_1 \diamond k_2 = \mu \circ \mathcal{G}(k_2) \circ k_1$ inherits the monad's universal property. Concretely:

* **Left unit.** $\eta_S \diamond k = \mu_T \circ \mathcal{G}(k) \circ \eta_S = \mu_T \circ \eta_{\mathcal{G}(T)} \circ k = k$, where the second equality is the naturality of $\eta$ and the third is the monad's left-unit law $\mu \circ \eta_{\mathcal{G}} = \mathrm{id}$.
* **Right unit.** $k \diamond \eta_T = \mu_T \circ \mathcal{G}(\eta_T) \circ k = \mathrm{id}_{\mathcal{G}(T)} \circ k = k$, by the monad's right-unit law $\mu \circ \mathcal{G}(\eta) = \mathrm{id}$.
* **Associativity.** A direct calculation: $((k_1 \diamond k_2) \diamond k_3)(s) = \mu \circ \mathcal{G}(k_3) \circ (\mu \circ \mathcal{G}(k_2) \circ k_1)(s)$. Naturality of $\mu$ rewrites this as $\mu \circ \mu_{\mathcal{G}} \circ \mathcal{G}(\mathcal{G}(k_3)) \circ \mathcal{G}(k_2) \circ k_1$, and the monad's associativity law $\mu \circ \mu_{\mathcal{G}} = \mu \circ \mathcal{G}(\mu)$ rewrites it to $\mu \circ \mathcal{G}(\mu) \circ \mathcal{G}(\mathcal{G}(k_3)) \circ \mathcal{G}(k_2) \circ k_1 = \mu \circ \mathcal{G}(\mu \circ \mathcal{G}(k_3) \circ k_2) \circ k_1 = k_1 \diamond (k_2 \diamond k_3)(s)$.
* **Strength coherence.** $\mathcal{G}$ is a *commutative strong* monad on $\mathbf{SBor}$. The right strength $\mathrm{str}_{S, T} : S \times \mathcal{G}(T) \to \mathcal{G}(S \times T)$ is given on rectangles by $(s, m) \mapsto \delta_s \otimes m$, and the left strength $\mathrm{str}'_{S, T} : \mathcal{G}(S) \times T \to \mathcal{G}(S \times T)$ symmetrically by $(m, t) \mapsto m \otimes \delta_t$; both are measurable and satisfy the four Kock-Linton strength axioms (associativity with the monoidal product, unit / pentagon coherence with $\eta$ and $\mu$, and naturality in $S, T$; see [Fritz 2020, §4](https://doi.org/10.1016/j.aim.2020.107239)). The strength-coherence row of the table refers to the naturality of $\mathrm{str}$ with respect to the symmetric-monoidal swap, which is the second Kock-Linton axiom and holds for every strong monad by a standard diagram chase. The deeper *commutativity* property of $\mathcal{G}$ — the equality, as maps $\mathcal{G}(S) \times \mathcal{G}(T) \to \mathcal{G}(S \times T)$, of the two double-strength composites

$$
\mathrm{dst}_1 \;=\; \mu_{S \times T} \,\circ\, \mathcal{G}(\mathrm{str}_{S, T}) \,\circ\, \mathrm{str}'_{S, \mathcal{G}(T)}
\qquad\text{and}\qquad
\mathrm{dst}_2 \;=\; \mu_{S \times T} \,\circ\, \mathcal{G}(\mathrm{str}'_{S, T}) \,\circ\, \mathrm{str}_{\mathcal{G}(S), T}
$$

— is what licenses reordering of independent draws in a QVR program body without changing the denotation. For $\mathcal{G}$ on $\mathbf{SBor}$, $\mathrm{dst}_1 = \mathrm{dst}_2$ reduces exactly to Fubini–Tonelli on the product $\sigma$-algebra ([Kock 1972](https://doi.org/10.1007/BF01304852)).

By Soundness of typing (Theorem [Typing §9.1](typing.md#91-soundness)), every well-typed QVR program denotes a morphism in $\mathbf{Kern}$ built by these Kleisli operations; the monad laws lift from $\mathbf{Kern}$ to QVR programs because the body's denotation function $\mathcal{B}\llbracket \cdot \rrbracket$ is compositional in the Kleisli composite. $\square$

## 6. Inference and conditioning

The denotation of a program is a *kernel*, not yet a posterior. Conditioning on observed data, normalization, and approximate posterior inference are *external* operations on the denotation, supplied by the [`quivers.inference`](../api/inference/svi.md) module. The categorical apparatus is that of *Markov categories with conditionals* ([Cho & Jacobs 2019](https://doi.org/10.1017/S0960129518000488); [Fritz 2020](https://doi.org/10.1016/j.aim.2020.107239)); the implementation realizes trace-based conditioning and stochastic variational inference as concrete instances of that theory.

## 7. Bayesian lifts

A `program` whose body lacks explicit `sample` priors on its
learnable parameters still has a well-defined kernel denotation
(it is a parameterized Kleisli arrow, with the parameters held
fixed). To pass such a program to the SVI / NUTS layer, which
operates on programs with explicit priors, the
[`quivers.inference.lifts`](../api/inference/lifts.md) module exposes four lifts. Each lifts a
parameter-bearing artefact into a [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram) of the standard
sample-then-score shape and admits a precise semantic statement.

Throughout this section let
$\theta \in \mathbb{R}^{D}$ denote the inner artefact's
learnable parameters (the flattened, concatenated tensor running
over [`nn.Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Parameter.html) leaves), let $\sigma_{\theta} > 0$ be a fixed
prior scale, and let $\mathbf{z}$ name an optional collection of
intermediate latents lifted into the program by the caller.

### 7.1 Parameter-only Bayesian lift

[`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) takes a parameter-bearing
[`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) `inner_model` exposing
$\log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta)$ via a
`log_joint(x, observations)` method, plus an optional
`additional_latents` map naming intermediate latents
$\mathbf{z}$ together with their shapes. It returns a lifted
program whose log-joint equals

$$
\log \pi_{\mathrm{lift}}(\theta, \mathbf{z})
\;=\;
\log \mathcal{N}(\theta;\, 0,\, \sigma_{\theta}^{2} I_{D})
\;+\;
\log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta).
$$

**Proposition (parameter lift, exactness of placeholder cancellation).**
Let $\sigma_{z} > 0$ be the placeholder scale supplied via
`latent_placeholder_scale`. The lifted program declares one
sample site $\theta_{i} \sim \mathcal{N}(0, \sigma_{\theta}^{2})$
per parameter and one sample site
$\mathbf{z}_{j} \sim \mathcal{N}(0, \sigma_{z}^{2} I)$ per
declared additional latent. Its score step computes

$$
S(\theta, \mathbf{z})
\;=\;
\log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta)
\;-\;
\sum_{j} \log \mathcal{N}\bigl(\mathbf{z}_{j};\, 0,\, \sigma_{z}^{2} I\bigr).
$$

The total log-density assembled by the inference layer (sum of
all sample-site log-priors plus the score step) is

$$
\log \pi_{\mathrm{lift}}(\theta, \mathbf{z})
\;=\;
\underbrace{\log \mathcal{N}(\theta; 0, \sigma_{\theta}^{2} I_{D})}_{\text{parameter prior}}
\;+\;
\underbrace{\sum_{j} \log \mathcal{N}(\mathbf{z}_{j}; 0, \sigma_{z}^{2} I)}_{\text{placeholder priors}}
\;+\;
S(\theta, \mathbf{z}),
$$

which by inspection cancels the placeholder priors and leaves
$\log \mathcal{N}(\theta; 0, \sigma_{\theta}^{2} I_{D}) + \log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta)$ pointwise.

*Proof.* The score step subtracts exactly the sum of placeholder
log-priors that the sample-site declarations add, so the algebra
is an identity. Gradient flow back to $\theta$ is realized by
the [`_swap_named_parameters`](../api/inference/lifts.md) context manager, which
temporarily writes the override tensor into the parent module's
`_parameters` dict so that downstream attribute reads return the
override; because the override is a non-Parameter tensor with
its own autograd graph, gradients propagate back through it to
the surrounding NUTS / SVI latents. $\square$

**Corollary (target distribution).** Under the bijection
$\theta \mapsto \theta$, $\mathbf{z} \mapsto \mathbf{z}$
([`NUTS`](../api/inference/mcmc.md) operates on the unconstrained
$\mathbb{R}^{D + \sum_{j} \dim \mathbf{z}_{j}}$),

$$
\pi_{\mathrm{lift}}(\theta, \mathbf{z})
\;\propto\;
p(\theta) \, p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta),
$$

the [Bayesian posterior](https://en.wikipedia.org/wiki/Posterior_probability)
$p(\theta, \mathbf{z} \mid x, y)$ under the
$\mathcal{N}(0, \sigma_{\theta}^{2} I)$ prior on $\theta$. The
placeholder scale $\sigma_{z}$ is denotationally irrelevant (it
cancels exactly) and operationally affects only NUTS
mass-matrix adaptation during warmup.

### 7.2 Deterministic-morphism lift via an observation family

[`lift_to_bayesian_program`](../api/inference/lifts.md#quivers.inference.lifts.lift_to_bayesian_program) takes a parameter-bearing
deterministic [`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), a
location-extraction callable
$\mu : \mathcal{X} \to \mathcal{M}$, an observation family
$\mathcal{F} \subset \mathbf{Dist}$ (any
[`torch.distributions.Distribution`](https://docs.pytorch.org/docs/stable/distributions.html) subclass), and a key
$k$ naming the observed coordinate in the observations dict.
The lift composes (i) the parameter-only Bayesian lift of §7.1
against (ii) a synthetic `log_joint` that scores the data under
$\mathcal{F}\bigl(\mu(x);\, \boldsymbol{\eta}\bigr)$ at the
observed value $y = \mathrm{observations}[k]$, where
$\boldsymbol{\eta}$ collects the family's remaining parameters
(scale, concentration, total_count, etc.) from the caller's
`observation_kwargs`.

**Proposition (observation-family lift).** The lifted program's
log-joint equals

$$
\log \pi_{\mathrm{lift}}(\theta)
\;=\;
\log \mathcal{N}(\theta;\, 0,\, \sigma_{\theta}^{2} I_{D})
\;+\;
\log p_{\mathcal{F}}\bigl(y \,\big|\, \mu(x;\, \theta),\, \boldsymbol{\eta}\bigr),
$$

i.e. the [Bayesian posterior](https://en.wikipedia.org/wiki/Posterior_probability)
of $\theta$ under the chosen
likelihood, modulo the
$\theta$-independent normalizer of $\mathcal{F}$.

*Proof.* The lift is implemented as a delegating wrapper whose
`log_joint(x, obs)` returns $\log p_{\mathcal{F}}(y \mid \mu(x), \boldsymbol{\eta})$ (reduced over event axes), then
delegates to [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) without additional latents.
By §7.1, the lifted log-density equals
$\log p(\theta) + \log p_{\mathrm{wrapper}}(y \mid x, \theta)$,
and the wrapper's `log_joint` is exactly the family's
log-density. $\square$

### 7.3 Direct log-prob lift

[`lift_from_log_prob`](../api/inference/lifts.md#quivers.inference.lifts.lift_from_log_prob) is the variant of §7.2 for
modules that already expose
$\log p_{\mathrm{inner}}(y \mid x, \theta)$ directly (without
going through a family / location split). Given a
parameter-bearing module and a callable
$\ell : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ that
returns the conditional log-density, the lift produces a program
whose log-joint equals

$$
\log \pi_{\mathrm{lift}}(\theta)
\;=\;
\log \mathcal{N}(\theta;\, 0,\, \sigma_{\theta}^{2} I_{D})
\;+\;
\ell(x, y;\, \theta).
$$

The proof is the §7.1 argument with the wrapper's `log_joint`
defined as $\ell$ directly. The lift exists so callers that have
already written a `log_prob`-style method (a [`SampledComposition`](../api/continuous/morphisms.md)
over a Normal kernel, an encoder-decoder composition with a
closed-form ELBO term, etc.) do not have to repackage it through
a family interface.

### 7.4 Monte-Carlo conditional-likelihood wrapper

[`monte_carlo_log_joint`](../api/inference/lifts.md#quivers.inference.lifts.monte_carlo_log_joint) wraps a `MonadicProgram` whose
body contains one or more named intermediate `sample` steps
$\mathbf{z}$ that the caller does not want to enumerate as
NUTS latents. The wrapper's `log_joint(x, observations)`
forward-draws each $\mathbf{z}_{j}$ from its declared family at
$x$, merges the draw into the observations dict, calls the
inner's `log_joint`, and subtracts the drawn latents' prior
contributions so the residual is the *conditional* data
likelihood.

**Definition.** Let
$q_{j}(\mathbf{z}_{j} \mid x, \theta) = p_{\mathrm{inner}}(\mathbf{z}_{j} \mid x, \theta)$ denote the
sample-step family of the inner program at site $j$. The
wrapper realizes the random map

$$
\widetilde{\ell}(x, y;\, \theta)
\;=\;
\log p_{\mathrm{inner}}\bigl(y \,\big|\, \mathbf{z}_{*},\, x,\, \theta\bigr),
\qquad
\mathbf{z}_{j*} \sim q_{j}(\cdot \mid x, \theta).
$$

**Proposition (single-sample MC bias).** The wrapper's output is
a single-sample [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) estimator of the conditional
log-likelihood, *not* of the marginal log-likelihood
$\log p_{\mathrm{inner}}(y \mid x, \theta) = \log \mathbb{E}_{\mathbf{z} \sim q}[p_{\mathrm{inner}}(y \mid \mathbf{z}, x, \theta)]$. By [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)
applied to the concave logarithm,

$$
\mathbb{E}_{\mathbf{z}_{*} \sim q}\bigl[\widetilde{\ell}(x, y;\, \theta)\bigr]
\;\le\;
\log p_{\mathrm{inner}}(y \mid x, \theta),
$$

with equality only when $p_{\mathrm{inner}}(y \mid \mathbf{z}, x, \theta)$ is constant in $\mathbf{z}$. The
wrapper's output is therefore biased downward as an estimator
of the marginal log-likelihood.

*Proof.* Jensen applied pointwise in $(x, y, \theta)$ to the
random variable $p_{\mathrm{inner}}(y \mid \mathbf{z}, x, \theta)$ under $q$:
$\log \mathbb{E}_{\mathbf{z}}[p(y \mid \mathbf{z})] \ge \mathbb{E}_{\mathbf{z}}[\log p(y \mid \mathbf{z})]$. $\square$

**Soundness for SVI; unsoundness for NUTS.** The reparameterised
families ([`Normal`](https://docs.pytorch.org/docs/stable/distributions.html#normal), [`MultivariateNormal`](https://docs.pytorch.org/docs/stable/distributions.html#multivariatenormal),
[`LowRankMultivariateNormal`](https://docs.pytorch.org/docs/stable/distributions.html#lowrankmultivariatenormal), etc.) yield a pathwise-
differentiable $\mathbf{z}_{*}(\xi; x, \theta)$ in an auxiliary
noise $\xi$. For SVI, the [unbiased reparameterization gradient identity](https://doi.org/10.48550/arXiv.1312.6114)
$\nabla_{\theta} \mathbb{E}_{\mathbf{z}}[\log p(y \mid \mathbf{z}, \theta)] = \mathbb{E}_{\xi}[\nabla_{\theta} \log p(y \mid \mathbf{z}_{*}(\xi; \theta), \theta)]$
holds, and a single MC draw at each SVI step is an unbiased
stochastic gradient of the wrapper's expected objective; under
standard step-size assumptions [SGD on this estimator](https://doi.org/10.1214/aoms/1177729586) converges to a stationary point
of $\theta \mapsto \mathbb{E}_{\mathbf{z}}[\log p(y \mid \mathbf{z}, \theta)]$, a lower bound on the marginal log-likelihood by Jensen.

For NUTS, the wrapper is unsuitable: re-drawing
$\mathbf{z}_{*}$ on every leapfrog evaluation makes the
Hamiltonian energy stochastic, breaks the [symplectic
integrator's energy-preservation guarantee](https://en.wikipedia.org/wiki/Symplectic_integrator), and biases the
chain away from the intended target. The rigorous route for
NUTS is to enrol $\mathbf{z}$ as an additional NUTS latent via
[`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters)
with `additional_latents={'<name>': <shape>}` (§7.1), so that
$(\theta, \mathbf{z})$ are jointly sampled from the exact
posterior under a deterministic per-evaluation log-density.

### 7.5 Bayesian lift for weighted deduction systems

[`nuts_program_from_deduction`](../api/stochastic/deduction/bayes.md#quivers.stochastic.deduction.bayes.nuts_program_from_deduction) is the
deduction-system specialization of §7.1. Given a
[`DeductionSystem`](../api/stochastic/deduction.md#quivers.stochastic.deduction.DeductionSystem)
$D$ with learnable log-weights $\mathbf{w} \in \mathbb{R}^{D}$
and a corpus $\{s_{n}\}_{n=1}^{N}$, the lift produces a program
whose log-joint equals

$$
\log \pi(\mathbf{w})
\;=\;
-\frac{1}{2 \sigma^{2}} \lVert \mathbf{w} \rVert_{2}^{2}
\;+\;
\sum_{n = 1}^{N} \log Z(s_{n};\, \mathbf{w}),
$$

where $Z(s_{n}; \mathbf{w})$ is the chart's goal weight (the
sentence's inside log-partition under $D$). Whether this joint
*is* the Bayesian posterior $p(\mathbf{w} \mid \{s_{n}\})$
depends on the modeling reading:

* **Undirected / globally normalized** (CRF / log-linear /
  energy-based [Lafferty et al. 2001](https://repository.upenn.edu/cis_papers/159/)). Here
  $p(s_{n} \mid \mathbf{w}) = Z(s_{n}; \mathbf{w}) / \sum_{s'} Z(s'; \mathbf{w})$, and the sentence-level normalizer
  $\sum_{s'} Z(s'; \mathbf{w})$ is a constant in $\mathbf{w}$
  only when the deduction is a [partition function](https://en.wikipedia.org/wiki/Partition_function_(mathematics))
  closed under the
  algebra's join. In the undirected setting that $\mathbf{w}$-
  dependent normalizer is precisely what $\sum_{n} \log Z$
  captures; $\pi(\mathbf{w})$ is then the posterior up to a
  $\mathbf{w}$-independent constant.

* **Directed / locally normalized PCFG** ([Booth & Thompson 1973](https://doi.org/10.1109/TC.1973.223746)).
  The per-rule weights are constrained to a local simplex
  (each non-terminal's expansions sum to $1$), and the true
  per-sentence likelihood is
  $Z(s; \mathbf{w}) / \sum_{s'} Z(s'; \mathbf{w})$ with the
  global normalizer intractable. Treating $\mathbf{w}$ as a
  free unconstrained vector and adding $\sum_{n} \log Z(s_{n}; \mathbf{w})$ to a Gaussian prior produces a
  *pseudo-posterior* (a [composite likelihood](https://en.wikipedia.org/wiki/Composite_likelihood) in the sense
  of [Varin, Reid & Firth 2011](https://www.jstor.org/stable/24309261)) that differs
  from the true posterior by a factor of
  $\bigl(\sum_{s'} Z(s'; \mathbf{w})\bigr)^{-N}$ that the lift
  does not include. The pseudo-posterior is consistent with
  the score function's gradient direction but has wider
  posterior credible regions than the true posterior. Users
  committed to the directed reading should constrain the rule
  weights to local simplices via a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) +
  [softmax](https://en.wikipedia.org/wiki/Softmax_function) surface (written explicitly in a
  `program` block) instead of using the free-parameter Normal
  lift this function provides.

The denotational claim of §7.1 (that the lifted program's
log-density equals the parameter prior plus the wrapped
log-joint, pointwise and with exact placeholder cancellation)
applies unchanged to [`nuts_program_from_deduction`](../api/stochastic/deduction/bayes.md#quivers.stochastic.deduction.bayes.nuts_program_from_deduction);
the choice of reading enters only when interpreting that
log-density as a posterior.

## References

Taylor L. Booth and Richard A. Thompson. 1973. [Applying probability measures to abstract languages](https://doi.org/10.1109/TC.1973.223746). *IEEE Transactions on Computers*, C-22(5):442–450.

Kenta Cho and Bart Jacobs. 2019. [Disintegration and Bayesian inversion via string diagrams](https://doi.org/10.1017/S0960129518000488). *Mathematical Structures in Computer Science*, 29(7):938–971.

Tobias Fritz. 2020. [A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics](https://doi.org/10.1016/j.aim.2020.107239). *Advances in Mathematics*, 370:107239.

Michèle Giry. 1982. [A categorical approach to probability theory](https://doi.org/10.1007/BFb0092872). In Bernhard Banaschewski, editor, *Categorical Aspects of Topology and Analysis*, volume 915 of *Lecture Notes in Mathematics*, pages 68–85. Springer, Berlin, Heidelberg.

Diederik P. Kingma and Max Welling. 2013. [Auto-Encoding Variational Bayes](https://doi.org/10.48550/arXiv.1312.6114). arXiv preprint arXiv:1312.6114.

Anders Kock. 1972. [Strong functors and monoidal monads](https://doi.org/10.1007/BF01304852). *Archiv der Mathematik*, 23(1):113–120.

John D. Lafferty, Andrew McCallum, and Fernando C. N. Pereira. 2001. [Conditional random fields: Probabilistic models for segmenting and labeling sequence data](https://repository.upenn.edu/cis_papers/159/). In *Proceedings of the Eighteenth International Conference on Machine Learning (ICML 2001)*, pages 282–289, San Francisco, CA, USA. Morgan Kaufmann.

Herbert Robbins and Sutton Monro. 1951. [A stochastic approximation method](https://doi.org/10.1214/aoms/1177729586). *The Annals of Mathematical Statistics*, 22(3):400–407.

Cristiano Varin, Nancy Reid, and David Firth. 2011. [An overview of composite likelihood methods](https://www.jstor.org/stable/24309261). *Statistica Sinica*, 21(1):5–42.
