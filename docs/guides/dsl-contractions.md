# DSL Contractions

A `contraction` block declares an operadic n-ary morphism under a
named composition rule. The contraction is the explicit surface
for "combine these typed morphisms by joining shared axes," with
the einsum wiring either inferred from the typed signature or
supplied explicitly. The categorical setup is detailed in
[Composition Rules § 4](../semantics/composition-rules.md#4-operadic-contractions);
this page covers the DSL surface.

## Declaration

<!-- compile: false -->
```qvr
contraction op_apply (
    arg1 : A -> B,
    arg2 : A -> C,
    kernel : B -> D
) : A -> D
    rule product_fuzzy
```

The declared name is callable from any expression site as
`op_apply(arg1_morph, arg2_morph, kernel_morph)`. Each call checks
argument count and per-argument shape (by numel) against the
declared signature, then runs the contraction under the named rule.

## Type-driven wiring inference

The compiler infers the einsum wiring from the typed signature
using three rules:

- Every axis name that appears in the output sequence (the
  contraction's declared dom and cod) is a *kept* axis (propagates
  from inputs to output).
- Every axis name that appears in two or more input sequences but
  not in the output is a *contracted* axis (joined via the rule's
  `tensor_op` then `join`).
- Every axis name that appears in exactly one input sequence and
  not in the output is *anomalous*: it would need to be summed out
  by the rule but no other input shares it. The compiler raises
  with a source-keyed diagnostic and directs the user at the
  explicit `wiring` clause.

This covers the common cases (standard composition, n-ary
contraction over a shared latent axis, parallel composition with
output axes that simply propagate) without requiring the user to
spell out an einsum string for every contraction.

## `share`: keep shared axes element-wise

The `share T1, T2, ...` clause keeps the listed axes element-wise
even when they are shared across inputs (rather than contracted):

<!-- compile: false -->
```qvr
contraction broadcast_add (
    f : A -> B,
    g : A -> B
) : A -> B
    rule real
    share B
```

Without `share B`, the axis `B` would be contracted because it
appears in two inputs and the output sequence after flattening is
`A, B`. The `share B` clause forces `B` to be kept across the
contraction, yielding element-wise sum on a kept axis.

The `share` clause is the disambiguator for element-wise
contractions and broadcasting patterns where the default inference
rule would over-contract a propagating axis.

## `wiring`: explicit einsum escape hatch

The `wiring "<einsum>"` clause is the explicit escape hatch for
unusual contractions where the inference rules do not apply.
Common uses:

- **Diagonal extraction**: take the diagonal of a square morphism
  rather than tracing it out.
- **Axis reordering**: route inputs to the contraction in a
  non-standard order.
- **Non-product signatures**: when the contraction signature
  involves coproducts or other non-product `TypeExpr`s that the
  axis-flattening pass cannot handle.

<!-- compile: false -->
```qvr
contraction extract_diag (
    A : I * I -> 1
) : I -> 1
    rule real
    wiring "ii->i"
```

The wiring string follows the standard numpy / torch
[`einsum`](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html)
notation: one letter per *distinct* axis name, joined with commas
across inputs, then `->`, then the kept axes' letters in the
output's declared order.

## Choosing the right composition rule

The `rule` keyword references a previously declared composition
rule (built-in or user-defined). The rule fixes the algebraic
operations used to contract shared axes:

- **`product_fuzzy`** (default): truncated-sum / product on
  `[0, 1]`. Right for compositions of fuzzy-truth-valued morphisms.
- **`real`**: ordinary sum / product on $\mathbb{R}$. Right for
  ordinary linear-algebra contractions.
- **`log_prob`**: log-sum-exp / sum on the log-probability
  semiring. Right for marginalizing discrete latents in a Bayesian
  composition.
- **`tropical`** / **`max_plus`**: max / sum, the [Viterbi
  semiring](https://ncatlab.org/nlab/show/tropical+semiring).
- **`boolean`** / **`godel`** / **`lukasiewicz`** /
  **`probability`**: the other built-in truth algebras and
  probability semirings.

User-defined rules declared at module scope (see
[Algebra](dsl-declarations.md#algebra)) are equally valid `rule`
references; the compiler verifies the rule has the right
algebraic level (`composition_rule` is permissive; weaker rules
restrict which operations are available).

## Worked example: bilinear scoring

A typical use case is bilinear scoring of an embedding against a
weight matrix and a target embedding:

<!-- compile: false -->
```qvr
object Item : 1024
type Embed = Euclidean 64
type Score = Euclidean 1

latent E : Item -> Embed
latent W : Embed -> Embed
latent T : Item -> Embed

# Score = sum_d sum_e E[i, d] * W[d, e] * T[i, e]
# Two shared axes: d (between E and W) and e (between W and T)
# get contracted; axis i is shared between E and T and the output,
# so it must be `share`d to keep it element-wise (per-item score).
contraction score (
    e : Item -> Embed,
    w : Embed -> Embed,
    t : Item -> Embed
) : Item -> Score
    rule real
    share Item

let final = score(E, W, T)
export final
```

Without `share Item`, the axis `Item` would be contracted (it
appears in two inputs and is not in the output's flattened
signature `Item, Score`). With `share Item`, the contraction
returns a per-item score.

## See also

- [Composition Rules § 4](../semantics/composition-rules.md#4-operadic-contractions):
  the categorical semantics of operadic contractions.
- [Algebra](dsl-declarations.md#algebra): declaring the composition
  rule used by the contraction.
- [Transformations and Composition Rules](transformations.md): the
  broader `CompositionRule → Semigroupoid → Algebra` hierarchy and
  first-class change-of-base transformations.
