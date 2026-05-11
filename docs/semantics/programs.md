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

### 2.1 Draw

A draw statement

```
draw v ~ F(args)
```

denotes the Kleisli arrow extending the context with a fresh random variable distributed according to family $F$:

$$
\mathcal{S}\llbracket \mathsf{draw}\ v \sim F(\bar a) \rrbracket : \Phi \to \mathcal{G}\bigl(\Phi \times \llbracket \mathsf{cod}(F) \rrbracket\bigr),
$$

defined on measurable rectangles $B \times C$ (with $B \subseteq \Phi$, $C \subseteq \llbracket \mathsf{cod}(F) \rrbracket$) by

$$
\mathcal{S}\llbracket \mathsf{draw}\ v \sim F(\bar a) \rrbracket(\phi,\, B \times C)
\;=\;
\mathbf{1}_B(\phi) \cdot \int_C p_F\bigl( y \,;\, \theta_F(\bar a, \phi) \bigr)\, \mathrm{d}y,
$$

where $\theta_F$ is the family's parameter map (which may depend on previously-drawn variables in $\phi$). In short: keep the current trace $\phi$ and append a fresh sample from $F$ conditioned on it. The induced action on measures over $\Phi$ is $\mu_{\Phi \times \mathsf{cod}(F)} \circ \mathcal{G}\bigl(\mathcal{S}\llbracket \mathsf{draw} \rrbracket\bigr)$.

### 2.2 Observe

An observe statement

```
observe v ~ F(args)
```

denotes a *score* update against an externally-supplied observed value $v_{\mathrm{obs}}$. As a Kleisli arrow in the *unnormalised* Giry monad $\mathcal{G}_{\le 1}$ (sub-probability measures),

$$
\mathcal{S}\llbracket \mathsf{observe}\ v \sim F(\bar a) \rrbracket : \Phi \to \mathcal{G}_{\le 1}(\Phi),
\qquad
\mathcal{S}\llbracket \mathsf{observe}\ v \sim F(\bar a) \rrbracket(\phi,\, B) \;=\; \mathbf{1}_B(\phi) \cdot p_F\bigl( v_{\mathrm{obs}} \,;\, \theta_F(\bar a, \phi)\bigr).
$$

The trace context is preserved, but the total mass of the resulting measure is the likelihood of $v_{\mathrm{obs}}$ at $\phi$. Normalisation and posterior inference are deferred to the inference layer (see [`quivers.inference`](../api/inference/svi.md)). The categorical setting is the *Markov category with conditioning* of [Cho & Jacobs 2019](https://doi.org/10.1017/S0960129518000488) and [Fritz 2020](https://doi.org/10.1016/j.aim.2020.107239).

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

i.e.\ pushforward by $\mathrm{id}_{\Phi} \times h$ realised through the *strength* of the Giry monad.

The arithmetic sublanguage is interpreted standardly: $\mathbb{R}$-valued and $\mathbb{N}$-valued operators denote the corresponding measurable functions on the relevant space; built-in functions (`sigmoid`, `exp`, `log`, `abs`, `softplus`) denote the corresponding total measurable maps.

### 2.4 Plate-Draw

A plate-draw statement

```
draw v : A -> K ~ F(args)
```

denotes an $A$-indexed plate of independent $F$-draws. The natural isomorphism

$$
\mathbf{Kern}(\mathbf{1}, K^A) \;\cong\; \mathbf{Kern}(A, K)
$$

identifies a single $\mathcal{G}(K^A)$-valued draw with an $A$-indexed family of $\mathcal{G}(K)$-valued draws. The statement therefore denotes the context-extending Kleisli arrow

$$
\mathcal{S}\llbracket \mathsf{draw}\ v : A \to K \sim F(\bar a) \rrbracket : \Phi \to \mathcal{G}\bigl(\Phi \times K^A\bigr),
$$

with density $\prod_{a \in A} p_F\bigl(v(a) \,;\, \theta_F(\bar a, \phi)\bigr)$ on the appended coordinate. When $K$ is a numeric literal it is interpreted as $\mathrm{Euclidean}(K)$.

### 2.5 Vectorised Observe

A vectorised-observe statement

```
observe r[n] ~ F(args) for n in N
```

denotes a sub-probabilistic Kleisli arrow in $\mathcal{G}_{\le 1}$,

$$
\mathcal{S}\llbracket \mathsf{observe}\ r[n] \sim F(\bar a)\ \mathsf{for}\ n \in N \rrbracket : \Phi \to \mathcal{G}_{\le 1}(\Phi),
\qquad
\phi \;\longmapsto\; \mathbf{1}_{(\cdot)}(\phi) \cdot \prod_{n \in N} p_F\bigl( r_{\mathrm{obs}}(n) \,;\, \theta_F(\bar a, n, \phi) \bigr).
$$

The response buffer $r_{\mathrm{obs}} : N \to \llbracket \mathsf{cod}(F) \rrbracket$ is supplied externally by the inference layer; the trace context is preserved and the total mass of the resulting measure is the batched likelihood.

### 2.6 Marginalize

A marginalize statement `marginalize c` pushes the accumulated (sub-)probability measure on $\Phi \times C$ forward through the projection $\pi_{\Phi} : \Phi \times C \to \Phi$:

$$
\mathcal{S}\llbracket \mathsf{marginalize}\ c \rrbracket : \mathcal{G}_{\le 1}(\Phi \times C) \to \mathcal{G}_{\le 1}(\Phi),
\qquad
\nu \;\longmapsto\; \pi_{\Phi *} \nu.
$$

For a discrete latent $C$, the projection is computed by log-sum-exp on the accumulated log-likelihood; for a measurable continuous $C$, it is fibrewise integration. The statement requires $c$ to name a previously bound coordinate of the context.

### 2.7 Indexed Gather (Let-Pullback)

A `let` right-hand side of the form `arr[idx]` is the *Kleisli pullback*. For a plate variable $v : A \to \mathcal{G}(B)$ bound earlier in the body, and a finite fibration $\iota : N \to A$ named in the context, the gather $\iota^* v$ is the composite

$$
\iota^* v \;=\; v \circ \iota \;:\; N \to \mathcal{G}(B).
$$

Interpreted as a deterministic measurable map on the accumulated context (because $v$ has already been realised as a tensor $A \to B$ in the trace), the let-step denotes the Dirac extension

$$
\mathcal{S}\llbracket \mathsf{let}\ w = \mathit{arr}[\mathit{idx}] \rrbracket(\phi)
\;=\;
\delta_{(\phi,\, \phi.\mathit{arr}[\phi.\mathit{idx}])}.
$$

### 2.8 Return

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

where $\pi_{v_1, \dots, v_m} : \Phi_n \to \llbracket \tau_2 \rrbracket$ projects the trace onto the named coordinates. Composing with the body chain marginalises the joint (sub-)probability measure onto those coordinates.

If $e$ contains labels (`return (a: x, b: y)`), the labels rename the coordinates of the resulting product space; this is a purely syntactic rebinding without semantic effect.

## 3. Parameters and closures

A program declared with parameters

```
program P (q₁, …, qₖ) : τ₁ -> τ₂
    body
```

denotes a *parameterised* family of kernels: with $\Theta = \prod_i \Theta_{q_i}$ the parameter space (each $\Theta_{q_i}$ determined by $q_i$'s type),

$$
\llbracket P \rrbracket : \Theta \times \llbracket \tau_1 \rrbracket \to \mathcal{G}(\llbracket \tau_2 \rrbracket),
$$

i.e.\ a morphism in $\mathbf{Kern}$ with extended domain. Concretely, parameter names are added to $\Gamma$ before the body is interpreted; the resulting Kleisli arrow is reused with each fresh parameter assignment by the training loop.

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

an object of the indexed family of Kleisli arrows over the parameter category. The domain and codomain may themselves mention the formal parameters, so each fibre is a kernel between possibly-different objects of $\mathbf{Kern}$.

### Inline expansion as substitution

A call site `draw v ~ P(a₁, …, aₖ)` inside another program is interpreted by *substitution* on the dependent denotation: the actual arguments $a_i$ are substituted for the formal parameters $p_i$ in the body of $P$, yielding a closed Kleisli arrow which is then inlined as a sequence of statements into the caller's body. Internal latents are α-renamed under a fresh prefix $v\$$, and the return-variable is renamed to $v$ directly; the result is a well-typed sequence of caller-level Kleisli arrows.

This is sound by a standard substitution lemma: because each formal parameter is bound at the top of the body and the body interprets to a Kleisli arrow built compositionally from its statements, substitution commutes with the body's denotation function $\mathcal{B}\llbracket \cdot \rrbracket$. The α-renaming step is sound because the body's denotation depends only on the multiset of bound-variable types, not on the names. Two call sites of the same template therefore contribute *distinct* factors to the caller's joint kernel — fresh latents per use — recovering the standard "plate-of-plates" semantics for hierarchical models.

## 4. Composition of programs

Two programs $P : X \to Y$ and $Q : Y \to Z$ compose by Kleisli composition:

$$
\llbracket P \mathbin{>\!\!>} Q \rrbracket(x, C) \;=\; \int_Y \llbracket Q \rrbracket(y, C) \, \llbracket P \rrbracket(x, \mathrm{d}y).
$$

The DSL exposes this through `output` declarations: `output P >> Q` denotes the Kleisli composite of the two program kernels.

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

The denotation of a program is a *kernel* — not yet a posterior. Conditioning on observed data, normalisation, and approximate posterior inference are *external* operations on the denotation, supplied by the [`quivers.inference`](../api/inference/svi.md) module. The categorical apparatus is that of *Markov categories with conditionals* ([Cho & Jacobs 2019](https://doi.org/10.1017/S0960129518000488); [Fritz 2020](https://doi.org/10.1016/j.aim.2020.107239)); the implementation realises trace-based conditioning and stochastic variational inference as concrete instances of that theory.
