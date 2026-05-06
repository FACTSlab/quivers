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

A QVR program is denoted by a single morphism in $\mathbf{Kern}$, built compositionally by interpreting each statement as a *Kleisli arrow* and composing them via Kleisli composition $\diamond$:

$$
(k_1 \diamond k_2)(s) \;=\; \mu_T \bigl( \mathcal{G}(k_2)(k_1(s)) \bigr)
\;=\; \int_T k_2(t) \, k_1(s, \mathrm{d}t).
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

We give the denotation of each statement form as a Kleisli arrow on the program's accumulated random-variable context $\Phi$. Concretely, the body is interpreted as a chain

$$
\Gamma \xrightarrow{\;k_1\;} \mathcal{G}(\Phi_1) \xrightarrow{\;k_2\;} \mathcal{G}(\Phi_2) \xrightarrow{\quad} \cdots \xrightarrow{\;k_n\;} \mathcal{G}(\Phi_n)
\xrightarrow{\;\eta \circ \pi_e\;} \mathcal{G}(\llbracket \tau_2 \rrbracket),
$$

where $\pi_e$ is the projection onto the components named by the `return` clause.

### 2.1 Draw

A draw statement

```
draw v ~ F(args)
```

denotes the Kleisli arrow extending the context with a fresh random variable distributed according to family $F$:

$$
\mathcal{S}\llbracket \mathsf{draw}\ v \sim F(\bar a) \rrbracket : \mathcal{G}(\Phi) \to \mathcal{G}(\Phi \times \llbracket \mathsf{cod}(F) \rrbracket),
$$

defined by

$$
\mathcal{S}\llbracket \mathsf{draw}\ v \sim F(\bar a) \rrbracket(\nu)\ (B \times C) \;=\; \int_{B} p_F\bigl( y \,;\, \theta_F(\bar a, \phi) \bigr)\, \mathrm{d}y \, \cdot\, \nu(\mathrm{d}\phi)\,\mathbf{1}_C(\phi),
$$

where $\theta_F$ is the family's parameter map (which may depend on previously-drawn variables in $\phi$). In short: draw a fresh sample from $F$ conditioned on the current context, and append it to the context.

### 2.2 Observe

An observe statement

```
observe v ~ F(args)
```

denotes a *score* update, multiplying the joint by the likelihood of $v$ under $F$:

$$
\mathcal{S}\llbracket \mathsf{observe}\ v \sim F(\bar a) \rrbracket(\nu) \;=\; p_F\bigl(v \,;\, \theta_F(\bar a, \phi)\bigr) \cdot \nu(\mathrm{d}\phi),
$$

a *sub-probability* measure on $\Phi$. In the Kleisli category of the *unnormalised* Giry monad, this is the natural Kleisli arrow; the standard practice is to defer normalisation to the inference layer (see [`quivers.inference`](../api/inference/svi.md)). The categorical setting is the *Markov category with conditioning* of [Cho–Jacobs 2019] / [Fritz 2020].

### 2.3 Let

A let statement

```
let v = expr
```

denotes a *deterministic* extension of the context. The right-hand side `expr` is an arithmetic / function-application expression over previously-bound names; it denotes a measurable map $h : \Phi \to T$, and the let statement is the Kleisli arrow

$$
\mathcal{S}\llbracket \mathsf{let}\ v = \mathit{expr} \rrbracket(\nu)\ (B \times C) \;=\; \int_C \mathbf{1}_B\bigl(h(\phi)\bigr)\, \nu(\mathrm{d}\phi),
$$

i.e.\ pushforward by $h$ paired with the identity on the existing context (the *strength* of the Giry monad).

The arithmetic sublanguage is interpreted standardly: $\mathbb{R}$-valued and $\mathbb{N}$-valued operators denote the corresponding measurable functions on the relevant space; built-in functions (`sigmoid`, `exp`, `log`, `abs`, `softplus`) denote the corresponding total measurable maps.

### 2.4 Return

A return statement

```
return e
```

closes the body. If $e = (v_1, \dots, v_m)$ is a tuple of bound names, the denotation is the marginalisation of the joint posterior onto the named coordinates:

$$
\mathcal{B}\llbracket \cdots \,;\, \mathsf{return}\ (v_1, \dots, v_m) \rrbracket(\gamma) \;=\; (\pi_{v_1, \dots, v_m})_{*} \, \nu_n(\gamma),
$$

where $\nu_n$ is the joint context measure produced by interpreting the body up to statement $n$, and $\pi_{v_1, \dots, v_m}$ is the projection onto the named coordinates.

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

## 4. Composition of programs

Two programs $P : X \to Y$ and $Q : Y \to Z$ compose by Kleisli composition:

$$
\llbracket Q \mathbin{>\!\!>} P \rrbracket(x) \;=\; \int_Y \llbracket Q \rrbracket(y) \, \llbracket P \rrbracket(x, \mathrm{d}y).
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

The denotation of a program is a *kernel* — not yet a posterior. Conditioning on observed data, normalisation, and approximate posterior inference are *external* operations on the denotation, supplied by the [`quivers.inference`](../api/inference/svi.md) module. The categorical apparatus is that of *Markov categories with conditionals* ([Cho–Jacobs 2019]); the implementation realises trace-based conditioning and stochastic variational inference as concrete instances of that theory.
