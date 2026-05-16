# DSL Programs and Let-Expressions

This page covers the `program` block surface: program declarations
and their effect signatures, the bind / observe / marginalize / let
steps that make up a program body, the axis-role clause that
configures structured priors and likelihoods, the let-expression
language with its full primitive surface, and factor expressions.

The grammar summary lives in the
[DSL overview](dsl-overview.md#grammar); declarations of objects,
morphisms, kernels, and combinators are detailed in
[DSL Declarations](dsl-declarations.md).

## Program declarations

A `program` block defines a probabilistic program. The body is a
sequence of *steps* (bind, observe, let, marginalize) followed by
`return`. Each step is a [Kleisli
arrow](https://ncatlab.org/nlab/show/Kleisli+category) on the
accumulated random-variable context $\Phi$; the program denotes the
composite $\Gamma \to \mathcal{G}(\tau_2)$ in
[`Kern`](../api/stochastic/categories.md).

<!-- compile: false -->
```qvr
program my_prog : X -> Y
    mu <- LogitNormal(0.0, 1.0)
    x <- Normal(mu, 1.0)

    return x

program with_params(a, b) : (X * Z) -> Y
    let w = a

    x <- f(w)
    y <- g(x, b)
    return y
```

The compiled program is a
[`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram);
see the [monadic programs guide](programs.md) for the runtime
contract (`rsample`, `log_joint`, the `observations` dict).

### Effect signatures

A program declaration may carry an effect signature after `!`, a
comma-separated subset of `{Sample, Score, Marginal, Pure}`. The
compiler verifies that the body's actual effects are a subset of
the declared set; `! Pure` rejects any sample, score, or marginal
binds.

<!-- compile: false -->
```qvr
program prior : Unit -> Y ! Sample
    mu <- Normal(0.0, 1.0)
    return mu

program deterministic : X -> X ! Pure
    let y = x
    return y
```

See the [compositional effects guide](effects.md) for the algebraic
basis of the effect surface.

### Kleisli bind syntax

The `<-` operator is the unique sampling-step sigil in a `program`
body:

<!-- compile: false -->
```qvr
x <- Normal(0.0, 1.0)
```

It introduces `x` as a random variable distributed according to the
given family. The same sigil carries every sampling-step variant:
scalar draws, indexed plates, scored observes, and scoped
marginalizations, distinguished by the surrounding shape.

### Indexed bind (plate)

`v : A <- Family(args)` declares `v` as an $A$-indexed family of
independent $F$-distributed draws. Categorically `v : A \to
\mathcal{G}(K)` where `K` is the per-fiber codomain taken from the
family; equivalently a single arrow $\mathbf{1} \to
\mathcal{G}(K^A)$ via the natural isomorphism
$\mathbf{Kern}(\mathbf{1}, K^A) \cong \mathbf{Kern}(A, K)$.

<!-- compile: false -->
```qvr
object Item : 1000

duration_incr : Item <- HalfNormal(1.0)
by_subject    : Subject <- Normal(0.0, sigma)
```

### Indexed observe

`observe r : N <- Family(args)` accumulates a batched
log-likelihood: a sub-probability kernel $\Phi \to
\mathcal{G}_{\le 1}(\Phi)$ with score $\prod_{n \in N} p_F(r_{\mathrm{obs}}(n);
\theta(n, \phi))$. The response buffer `r` is supplied at runtime
via the `observations` dict passed to
[`MonadicProgram.rsample`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.rsample),
[`MonadicProgram.log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint),
or [`ELBO.forward`](../api/inference/elbo.md). Family arguments may
use bracket-indexed sections `theta[N]` to refer to plate variables.

<!-- compile: false -->
```qvr
observe cloze_resp : RespCloze <- Bernoulli(intercept_cloze)
```

### Scoped marginalize

`marginalize c : A <- F(args) in { ... }` introduces a coordinate
`c` bound to a kernel `F(args)`, optionally `A`-indexed, with the
`{ ... }` block as its integration scope. At the end of the scope
the coordinate is pushed forward through the projection $\pi :
\Phi \times C \to \Phi$, integrating it out by
[log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) on the
log-likelihood (discrete) or fibrewise integration (continuous);
`c` then falls out of scope.

<!-- compile: false -->
```qvr
marginalize class : Item <- Categorical(class_logits) in {
    observe r : N <- Bernoulli(theta[class[N]])
}
```

The grouped form with `over G` and per-observe `via <idx>` clauses
is the fibred marginalization construct; see
[hierarchical programs](programs-hierarchical.md#grouped-marginalization-fibred-discrete-latents).

### Indexed gather in `let`

A `let`-expression of the form `arr[idx]` denotes the [Kleisli
pullback](https://ncatlab.org/nlab/show/pullback) of a plate
variable along a finite fibration. For a plate `v : A -> B` and an
index morphism $\iota : N \to A$, the gather $\iota^* v = v \circ
\iota$ is itself a [`Kern`](../api/stochastic/categories.md)-morphism
$N \to B$.

<!-- compile: false -->
```qvr
by_verb : Verb <- Normal(0.0, sigma)
let intercept_for_item = by_verb[verb_of_item]
```

### Parametric programs

A `program` declaration whose parameter list contains *typed*
parameters denotes a dependent family of kernels rather than a
single kernel:

$$
\llbracket P \rrbracket \;:\; \prod_{p_1 : P_1} \cdots \prod_{p_k : P_k} \mathbf{Kern}\bigl(\mathrm{dom}(p), \mathrm{cod}(p)\bigr).
$$

Three parameter universes are available:

| Kind | Universe | Quantifies over |
|---|---|---|
| `FinSet`, `Space`, `Object` | object of the relevant subcategory | the carrier of a plate |
| `Real`, `Nat` | hom-object of scalar type | a hyperparameter value |
| `Mor[A, B]` | the hom-set $\mathbf{Kern}(A, B)$ | a kernel passed in by name |

Parametric programs are *not* compiled to runtime `MonadicProgram`s
in isolation; the compiler stores them as templates and inlines
them at each call site:

<!-- compile: false -->
```qvr
v <- template(arg1, arg2, ...)
```

At each call site the template's body is substituted (formal
parameters to actual arguments) and α-renamed (internal latents are
prefixed by `v$`, the return variable is renamed to `v` directly).
The renamed step list is inlined into the caller, so distinct call
sites contribute distinct factors to the parent's joint kernel:
fresh latents per use, no inadvertent tying.

<!-- compile: false -->
```qvr
# Parametric random-intercepts template: one HalfNormal scale and
# a per-level Normal(0, sigma) plate, polymorphic over the grouping
# object G and the half-normal hyperparameter scale.
program random_intercepts (G : FinSet, scale : Real) : G -> 1
    sigma <- HalfNormal(scale)
    v : G <- Normal(0.0, sigma)
    return v
```

Worked hierarchical examples (crossed random intercepts, grouped
marginalization over discrete classes) live in the
[hierarchical programs guide](programs-hierarchical.md).

### Posterior blocks

A `program name(latents) : domain -> codomain ! Pure over model`
declaration denotes a deterministic post-conditioning kernel. The
`over model` modifier marks the program as consuming the named
model's latents; the consumed latents appear as data parameters in
the parameter list. The `! Pure` effect signature rejects any
sample, score, or marginal binds; the body is restricted to `let`
(and `marginalize` over its own scope). Categorically it is a
[`Kern`](../api/stochastic/categories.md)-morphism $\text{Latents}
\to \tau_{\mathrm{out}}$ that lifts to $\text{Data} \to
\mathcal{G}(\tau_{\mathrm{out}})$ by post-composition with the
model's posterior kernel $q(\theta \mid \mathrm{data})$.

<!-- compile: false -->
```qvr
type Logits4 = Euclidean 4

program scored : Item -> Logits4
    raw_logits <- Normal(0.0, 1.0)
    return raw_logits

program class_probs(raw_logits) : Item -> Logits4 ! Pure over scored
    let probs = softmax(raw_logits)
    return probs
```

The data parameter `raw_logits` names the model latent the body
consumes; the runtime supplies a per-sample snapshot of the model's
trace.

## Axis-role clause: `over` and `iid over`

Every distribution clause (kernel declarations, latent parameter
priors, bind steps, observe steps) accepts an optional
**axis-role clause** of the form:

```
~ Family [options] over <axes> [iid over <axes>]
```

`over <axes>` names the **event axes**: the axes on which the
family's joint structure lives. The axis count must match the
family's declared `event_rank` (0 for scalar families like Normal /
Beta / Gamma; 1 for vector families like
[`MultivariateNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMultivariateNormal),
[`Dirichlet`](../api/continuous/families.md#quivers.continuous.families.ConditionalDirichlet),
[`ConditionalGaussianProcess`](../api/continuous/families.md#quivers.continuous.families.ConditionalGaussianProcess),
or
[`ConditionalHorseshoe`](../api/continuous/families.md#quivers.continuous.families.ConditionalHorseshoe);
2 for matrix families like
[`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal),
[`Wishart`](../api/continuous/families.md#quivers.continuous.families.ConditionalWishart),
[`LKJCholesky`](../api/continuous/families.md#quivers.continuous.families.ConditionalLKJCholesky)).
The positional ordering of `over` axes corresponds positionally to
the family's declared event-axis ordering (for asymmetric families
like `MatrixNormal`, the first axis is the row axis, the second
the column axis). The full event-rank table lives in
[continuous families](continuous-families.md#event-rank-and-the-axis-role-surface).

`iid over <axes>` is an optional readability assertion naming the
batch axes (the complement of `over`). Any axis not in `over` is
batched by default, which categorically is a product of independent
distributions on that axis.

**Axis names.** Names resolve against the named factors of the
surrounding morphism's dom and cod (or the type annotation `: T`
on a sample / observe step). The reserved tokens `dom` and `cod`
are shortcuts when that side is a single unfactored object; for a
product-typed side, every factor must be named explicitly.

**Categorical reading.** The surface preserves the distinction
between joint-on-a-product-space (the family's event with possibly
non-trivial correlation) and product-of-independents (iid batches
across an axis), and between a flat MVN over $\dim(A) \cdot \dim(B)$
with dense covariance versus a
[`MatrixNormal`](https://en.wikipedia.org/wiki/Matrix_normal_distribution)
with [Kronecker structure](https://en.wikipedia.org/wiki/Kronecker_product)
$V \otimes U$. There is no auto-substitution between families with
different event ranks. Renaming or refactoring a morphism's type
invalidates axis references at type-check time rather than silently
rebinding.

**Compile-time errors.** Two diagnostics fire here:

- *"axis count does not match family event_rank"*: the family
  declares event_rank `k` but the `over` clause names `j ≠ k`
  axes. Common when migrating from a flat
  [`MultivariateNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMultivariateNormal)
  (event_rank 1) to a matrix
  [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal)
  (event_rank 2).
- *"axis `A` not in domain / codomain"*: an `over` name is not a
  factor of the morphism's dom or cod. Resolve by renaming, or by
  using `dom` / `cod` if the side is a single unfactored object.

<!-- compile: false -->
```qvr
# Vector prior: 5-dim MVN over the codomain axis.
mu : Euclidean(5) <- MVN(zeros, L) over cod

# Matrix prior on a morphism: Kronecker MatrixNormal.
latent W : Euclidean(32) -> Euclidean(64)
    ~ MatrixNormal(loc, row_scale, col_scale) over (dom, cod)

# Per-row Dirichlet on a transition kernel: each row is a K-dim
# simplex independently, rows are iid.
latent T : Euclidean(K) -> Euclidean(K)
    ~ Dirichlet(alpha) over cod iid over dom

# MVN response per observation row.
observe y : N <- MVN(mu_hat, scale_tril) over cod
```

## Let expressions (arithmetic and primitives)

Inside a `program` block, `let` bindings support full arithmetic
with standard operator precedence, unary negation, and a fixed pool
of built-in tensor primitives:

<!-- compile: false -->
```qvr
# arithmetic: +, -, *, /
let eta = mu + sigma * z_raw + lambda * shared_factor
let adjusted = (1.0 - lapse) * p_raw + 0.5 * lapse
let mean = (x + y + z) / 3.0
let negated = -raw_score

# built-in primitives
let prob = sigmoid(eta)
let positive = softplus(raw)
let log_rate = log(rate)
let magnitude = abs(x - 0.5)
let monotone = cumsum(increments)
let weights = softmax(logits)
let regularized = dropout(layer_norm(features))
```

Each `let`-builtin denotes a deterministic measurable map, lifted
into the Kleisli category as a [Dirac
kernel](https://en.wikipedia.org/wiki/Dirac_delta_function).

### Primitive reference

Reductions and shape-preserving operations on the last axis default
to `dim=-1`, the natural choice for per-row operations in `V-Cat`
morphisms; for contractions over a specific named axis, use the
typed [contraction declaration](dsl-contractions.md).

**Activations** (`torch.nn.functional` surface): `relu`, `relu6`,
`leaky_relu`, `prelu`, `rrelu`, `elu`, `selu`, `celu`, `gelu`,
`silu` (alias `swish`), `mish`, `hardtanh`, `hardshrink`,
`hardsigmoid`, `hardswish`, `softplus`, `softshrink`, `softsign`,
`tanh`, `tanhshrink`, `sigmoid`, `logsigmoid`, `threshold`, `glu`.

**Simplex / normalization maps**: `softmax`, `log_softmax`,
`softmin`, `normalize`.

**Pointwise transcendentals**: `exp`, `expm1`, `log`, `log1p`,
`log2`, `log10`, `sqrt`, `rsqrt`, `square`, `abs`, `neg`, `sign`,
`reciprocal`, `clamp`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`,
`sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `floor`, `ceil`, `round`,
`trunc`, `erf`, `erfc`, `erfinv`, `lgamma`, `digamma`.

**Last-axis reductions**: `sum`, `mean`, `var`, `std`, `min`, `max`,
`argmin`, `argmax`, `prod`, `amax`, `amin`, `logsumexp`, `norm`.

**Last-axis shape-preserving**: `cumsum`, `cumprod`, `cummax`,
`cummin`, `flip`, `sort`.

**Training-mode primitives**: `dropout`, `alpha_dropout`,
`layer_norm`, `rms_norm`.

The compiled implementation is `_LET_EXPR_BUILTINS` in
[`quivers.dsl.compiler.programs`](../api/dsl/compiler.md). Function
calls inside a let body resolve against this builtin table first,
then against module-scope callables (programs, morphisms,
encoders, decoders, deductions); arity is checked at compile time.

### Factor expressions: assembling indexed tensors

The `factor` expression in a `let` body builds a
finite-domain-indexed tensor by evaluating a body once per tuple of
index values. Categorically it is the *left adjoint of indexing*:
while `arr[i, j, ...]` is the elimination rule for `I_1 × ... × I_n
-> body_type` (Kleisli pullback of a plate variable along a finite
fibration), `factor` is the introduction rule.

#### Uniform form

```
factor v_1 : I_1, v_2 : I_2, ..., v_n : I_n in <body>
```

denotes the tensor of shape `(|I_1|, ..., |I_n|, *body_shape)`
whose value at position `(i_1, ..., i_n)` is `<body>` evaluated
with `v_k := i_k`. The binder variables are integer-valued and
visible only inside the body.

<!-- compile: false -->
```qvr
object Verb : 40
object Class : 4

# Per-verb, per-class scoring table: shape (40, 4).
let cell = factor v : Verb, cls : Class in coef[v, cls] * weight[cls]
```

#### Pattern-match form (single-axis)

When each cell of a single-axis factor carries a structurally
different expression, the case form lets you state the per-index
expressions side-by-side:

<!-- compile: false -->
```qvr
let class_probs = factor cls : Class in {
    0 -> (1.0 - prob_dur) * (1.0 - prob_telic_nodur),
    1 -> (1.0 - prob_dur) *        prob_telic_nodur,
    2 ->        prob_dur  * (1.0 - prob_telic_dur),
    3 ->        prob_dur  *        prob_telic_dur,
}
```

The case labels must cover `{0, ..., |Index|-1}` exactly; the
compiler rejects gaps, duplicates, or out-of-range labels at
compile time. Braces and comma separators delimit the cases. This
is the natural surface for structured categorical priors: each
cell of a `Class`-shape probability vector is built from a
different combination of upstream scalar latents.

## Inline distributions

Bind and observe steps support inline distribution construction
with any mix of literal and variable arguments. The 30+ registered
families accept literal-or-variable arguments at any position:

<!-- compile: false -->
```qvr
# all-literal (fixed): Unit -> codomain
x <- Normal(0.0, 1.0)
p <- Beta(2.0, 5.0)

# all-variable (direct): variables -> codomain
y <- Normal(mu, sigma)
b <- Bernoulli(theta)

# mixed literal / variable: any combination works
h_cand <- Normal(reset_hidden, 0.5)
z <- Normal(0.0, learned_scale)
r <- TruncatedNormal(mu, sigma, 0.0, 1.0)

# negative literals
z <- Normal(-1.5, 0.3)
```

A representative subset of the inline-distribution registry (the
full set is documented in
[continuous families](continuous-families.md#family-registry)):

| Family | Parameters | Codomain |
|---|---|---|
| `Normal` | `loc`, `scale` | Euclidean |
| `LogitNormal` | `mu`, `sigma` | UnitInterval |
| `Uniform` | `low`, `high` | UnitInterval / Euclidean |
| `Bernoulli` | `probs` | FinSet(2) |
| `Beta` | `concentration1`, `concentration0` | UnitInterval |
| `Exponential` | `rate` | PositiveReals |
| `HalfCauchy` | `scale` | PositiveReals |
| `HalfNormal` | `scale` | PositiveReals |
| `LogNormal` | `loc`, `scale` | PositiveReals |
| `Gamma` | `concentration`, `rate` | PositiveReals |
| `Dirichlet` | `concentration` | Simplex (codomain `dim` / `cardinality`) |
| `TruncatedNormal` | `mu`, `sigma`, `low`, `high` | Euclidean (bounded) |
| `MultivariateNormal` | `loc`, `scale_tril` | Euclidean |
| `MatrixNormal` | `loc`, `row_scale`, `col_scale` | matrix Euclidean |
| `LKJCholesky` | `concentration` | Cholesky-factor manifold |
| `Wishart`, `InverseWishart` | `df`, `scale_tril` | positive-definite matrices |
| `Horseshoe` | `scale` | Euclidean (sparse-shrinkage prior) |

Every parameter position in every family accepts either a literal
value or a previously-bound variable. When all arguments are
literals, a fixed distribution is created; when any argument is a
variable, the family is resolved at runtime against the current
trace's values.

For conditional distributions (input-conditional, learned-parameter
form), use a [`kernel f : A -> B ~ Family`](dsl-declarations.md#kernel)
declaration instead.

## Examples

### Simple discrete model

<!-- compile: false -->
```qvr
object X : 3
object Y : 4

latent f : X -> Y
latent g : Y -> Y

let fg = f >> g

export fg
```

### Continuous conditional model

<!-- compile: false -->
```qvr
object Cond : 2

space Latent : Euclidean(3)
space Obs : Euclidean(5)

kernel prior : Cond -> Latent ~ Normal
kernel likelihood : Latent -> Obs ~ Normal [scale=0.1]

let posterior = prior >> likelihood

export posterior
```

### Probabilistic program with observations

<!-- compile: false -->
```qvr
object Data : 1

space Y : Euclidean(2)

program regression : Data -> Y
    theta <- LogitNormal(0.0, 1.0)
    y <- Normal(theta, 0.5)

    observe _ <- Normal(y, 0.1)

    return y
```

### Init recipe end-to-end

<!-- compile: false -->
```qvr
object D : 1
type In  = Euclidean 8
type Out = Euclidean 4

# Default: weights initialised from a small-scale randn.
latent W_default : In -> Out [scale=0.1]

# Algebra-guided: init pulled from Algebra.init_spec, applied
# through the active algebra's bijector.  Under product_fuzzy
# (the default), the raw parameter is set to logit(ln(2) / depth);
# samples land near the algebra's neutral element on draw zero.
latent W_auto : In -> Out [init=auto]

export W_auto
```

For more examples, see the
[Examples Gallery](../examples/index.md). For a formal account of
what `.qvr` programs *mean*, see the
[Denotational Semantics](../semantics/index.md).
