# Structural Compression: Decoders and Losses

This page covers the decoder dual to the encoder surface (the
Kleisli arrow back from vectors to terms), the loss-declaration
surface, and the integration points with the chart-parser and
Bayesian-program machinery. The signature and encoder side lives
in
[Signatures and Encoders](structural-compression-signatures-and-encoders.md).

## Decoder blocks

A decoder declares a [Kleisli
arrow](https://ncatlab.org/nlab/show/Kleisli+category) $\mathrm{Vec}_D
\to \mathrm{Kern}(T_\Sigma)$. Per-sort `structure` / `primitive` /
`factor` / `binder_select` heads are scaffolded as learnable neural
networks; the corecursion (op choice, factor split, recursive
descent, `BoundVar` fallback to in-scope variables) is supplied by
the framework.

<!-- compile: false -->
```qvr
decoder D over LF depth 8 { body |-> recursive }
```

The two runtime operations:

- `sample(vec) -> Term`: draw a single term.
- `log_prob(term, vec) -> Tensor`: score an observed term.

**Depth-bounded termination.** At the depth limit the choice set is
restricted to ops whose every child sort is `data` or `index`
(never `object`); if no such op exists at a sort, the decoder
raises with a precise diagnostic.

## Loss declarations

Losses are first-class declarations attachable at any site in the
training graph:

<!-- compile: false -->
```qvr
loss reconstruction weight 1.0 on encoder C {
    -D(C(input)).log_prob(input)
}

loss type_coherence weight 0.1 on rule combine in Parse {
    cross_entropy(parent_type_dist, combined_children_type_dist)
}

loss completed weight 0.01 on chart of Parse {
    -chart.goal_weight()
}

loss nli weight 1.0 on program nli_predict {
    bce_with_logits(predicted_logit, true_label)
}
```

Attachment kinds: `global` (no `on` clause), `program <name>`,
`deduction <name>`, `encoder <name>`, `decoder <name>`,
`rule <name> in <D>` (fires on every application of that rule
during chart construction), and `chart of <D>` (fires once on the
completed chart).

`prog.losses.evaluate(env)` sums every loss;
`evaluate_on(kind, target, env)` filters by attachment. Inside a
deduction, the runtime accumulates rule-attached and chart-attached
losses into `ChartView.attached_loss` for the training driver to
read.

## Deduction integration

A `deduction` block may attach an item signature and an encoder;
the chart's `embedding(item)` operation then returns a
differentiable vector computed by the attached encoder's
algebra-homomorphism recursion over the chart-item term.

<!-- compile: false -->
```qvr
signature ChartItem {
    sorts {
        Item : object dim 128
        Idx  : data   dim 8
        Type : object dim 32
    }
    constructors {
        span : Idx, Idx, Type -> Item
    }
}

deduction Parse : Sentence -> Tree {
    atoms { S, NP, span }
    rule combine
        : span(i, k, Fwd(X, Y)), span(k, j, X)
        |- span(i, j, Y)
    semiring  LogProb
    signature ChartItem
    encoder InsideC
}
```

The attention-weighted aggregation that distinguishes the
vector-inside-outside parser from semiring parsing lives entirely
inside the encoder's per-op function, outside the chart's role, so
the semiring abstraction is not broken.

## Bayesian integration

A decoder is a Kleisli arrow into a structured type: a distribution
over terms. So everything the program / posterior machinery does
with distributions over scalars and vectors extends, with no
special-casing, to distributions over structured objects:

- **Structured latents.** `latent_term <- D(prior_vec)` inside a
  program body draws a random term from the decoder.
- **Observations of structured data.**
  `observe known_term <- D(some_vec)` scores under the decoder's
  log-prob.
- **Marginalization over structured latents.**
  `marginalize t : Term <- D(v) in { observe y <- ... }` integrates
  `t` out via the decoder's depth-bounded categorical recursion.
- **Variational decoders.**
  `program q : Sentence -> Term ! Sample, Score over generative`
  defines a variational decoder over LFs given a sentence.

## Stdlib decoders

The same `quivers.structural.shapes` registry that ships the
canonical encoders also ships matching decoders:

- `ar_decoder(sig, dim, vocab, depth)`: autoregressive decoder for
  sequence signatures; `depth` caps the unfolded structure size.
- `tree_decoder(sig, dim, leaf_vocab, label_vocab, depth)`:
  top-down structural decoder for tree signatures.

Both are realized on top of the generic
[`Decoder`](../api/structural/decoder.md) runtime, the same Kleisli
coalgebra pattern, no special-casing.

## See also

- [Signatures and Encoders](structural-compression-signatures-and-encoders.md):
  the encoder side of the autoencoder pair.
- [Weighted Deduction Systems](deduction.md): the chart-parser
  substrate that decoders couple to.
- [Compositional Effects](effects.md): the algebraic-effects
  framework over which the decoder's stochasticity is realized.
