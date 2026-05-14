# Decoders

`Decoder` is a `torch.nn.Module` that realizes a Kleisli
coalgebra `Vec_D → Kern(T_Σ)`, given an input vector, defines
a distribution over terms of a signature. Two operations:

- `sample(vec, ctx, sort)` draws a single `Term`.
- `log_prob(term, vec, ctx, sort)` scores an observed term
  under the same distribution.

The corecursion over a signature Σ:

1. At each sort position, the decoder produces logits over its
   *choice set*, every constructor and binder whose codomain is
   that sort, plus the built-in `BoundVar` whenever the context
   contains at least one in-scope variable of that sort.
2. For the chosen op, the parent vector is split into per-child
   sub-vectors by the per-(sort, arity) `factor` function, and
   the decoder recurses on each child.
3. Data-sorted children are sampled from a closed vocabulary via
   the per-sort `primitive` head; index-sorted children are
   sampled via `binder_select` over the in-scope variables.
4. Binder ops extend Γ before recursing on their scoped arguments,
   exactly mirroring the encoder.

Termination is depth-bounded at construction. At the budget limit
the choice set is restricted to recursion-terminating ops; if no
such op exists at a sort, the decoder raises with a precise
diagnostic.

::: quivers.structural.decoder
