# Structural compression

Quivers exposes a uniform algebraic interface for compressing
arbitrary structured objects to fixed-length vectors and decoding
them back under a learned distribution. The interface is built on
two categorical primitives:

- A **encoder** is a Σ-algebra homomorphism `T_Σ → Vec_D` —
  for each operation `op : s_1 × … × s_n → s` in a multi-sorted
  signature Σ, a parametric function
  `Ĉ_op : Vec_D × … × Vec_D → Vec_D`. The recursion over the term
  tree is supplied by the framework; the analyst supplies only
  the per-operation parametric functions.
- A **decoder** is a Kleisli coalgebra `Vec_D → Kern(T_Σ)` —
  a structure choice, primitive choice, factor split, and binder
  selection together corecursively generate a `Term` (with
  per-term `log_prob` available for observed terms).

Encoder + decoder = stochastic autoencoder on `T_Σ`. The same
interface uniformly realises transformers (sequence encoder +
autoregressive decoder), tree-LSTMs (tree encoder), graph
neural networks (message-passing graph encoder), variational
autoencoders (encoder + decoder pair on `T_Σ` with a KL term),
the vector inside-outside parser of Le-Zuidema / Drozdov / Kim
(chart-item signature + attached encoder), and typed-LF
induction pipelines (binders with annotation sorts threading the
type of each bound variable through a de-Bruijn context).

## Signature blocks

A `signature` declares the sorts, constructors, binders, and (for
graph signatures) vertex / edge kinds of an inductive (or
graph-shaped) algebra.

<!-- compile: cumulative -->
```qvr
signature LF {
    sorts {
        Term : object dim 64
        Type : object dim 32
        Name : data   dim 32 vocab { "dog", "cat", "every", "some" }
    }
    constructors {
        Const : Name      -> Term
        App   : Term, Term -> Term
    }
    binders {
        Lam : binds (x : Term : ty : Type) in (body : Term) -> Term
        All : binds (x : Term : ty : Type) in (body : Term) -> Term
    }
}
```

Three sort kinds:

- `object` — the principal sorts whose terms are compressed /
  decoded recursively.
- `data` — opaque atoms (string / int / float / bytes / bool)
  consumed by per-data-sort embedders. A data sort may declare a
  closed vocabulary inline via `vocab { … }` listing string,
  integer, and / or float literals; the decoder samples and scores
  data-sorted children from exactly these tokens.
- `index` — de-Bruijn index slots.

Reserved op names: `BoundVar` and `Data` — these are
framework-built-in op tags. The compiler rejects them as
user-declared constructor / binder names.

### Binders, de-Bruijn context, and type annotations

A binder declaration introduces one or more typed scoped
variables. `binds (x : Term : ty : Type) in (body : Term) -> Term`
reads: "constructs a value of sort `Term`; in the scope of
argument `body`, introduces a variable `x` of sort `Term`
annotated by a `Type`-sorted term `ty`."

The framework threads a de-Bruijn context Γ through both the
encoder's recursion and the decoder's corecursion. Each entry
of Γ is a triple `(var_sort, embedding, type_term)` — so the
**type of every bound variable is structurally tracked**, not just
the variable's existence. `BoundVar(i)` at any object-sorted
position reads Γ at depth `i`. The encoder's `var_init`
function (see below) mints the variable's vector embedding from
its type annotation.

Strict declaration rules:

- Every sort a constructor or binder mentions (domain, codomain,
  annotation sort) must be declared in the signature's `sorts {
  … }` block. There is no silent auto-registration.
- Every sort with no inline `dim` must have its dim supplied by
  every encoder / decoder over the signature. The compiler
  raises if a sort's dim is unresolved.

## Encoder blocks

An encoder declares the carrier dim per sort plus one
parametric function per constructor / binder. Bodies are
optional — the compiler scaffolds a 2-layer MLP per omitted op
with the correct per-arg dim sequence.

<!-- compile: cumulative -->
```qvr
encoder C over LF {
    dim Term = 64
    dim Type = 32

    Const(n)         |-> name_embed
    App(f, x)        |-> mlp_app([f, x])

    Lam(ty, body)    |-> mlp_lam([ty, body])
    All(ty, body)    |-> mlp_all([ty, body])

    var_init Term from Type as ty |-> mlp_typed_var(ty)
}
```

For binder constructors, the framework's calling convention is:

1. The annotation arguments (one per annotated bound variable)
   are compressed in the **outer** context Γ.
2. Each `var_init <var_sort> from <annot_sort> as <name> |-> body`
   declaration is invoked to mint a fresh variable embedding from
   the annotation embedding; the framework pushes
   `(var_sort, embedding, annot_term)` onto Γ.
3. The scoped arguments are compressed in the **extended**
   context Γ′.
4. The per-op function receives the flat child-embedding list
   `[annot_1, …, annot_k, scoped_1, …, scoped_m]` in declaration
   order.

Multiple `var_init` declarations are allowed per encoder — one
per `(var_sort, annot_sort)` pair the signature's binders
introduce. Unannotated binders (those without an annotation sort)
use a learnable nullary constant keyed by `var_sort` alone.

### Sequence-shaped sugar

For sequence signatures (`Seq[A] = Nil | Cons(A, Seq)`), two
extra body shapes are available:

<!-- compile: false -->
```qvr
encoder RNN over Seq {
    Nil                              |-> 0.0
    Cons(head, tail) recurrent state |-> gru_step(head, state)
}

encoder Tfm over Seq {
    Nil                               |-> 0.0
    Cons(head, tail) attention prefix |-> tfm_step(head, prefix)
}
```

- `recurrent <state>` binds the named state variable to the
  recursive child's already-computed embedding — exactly the
  standard right-fold F-algebra recursion, just with the user's
  chosen name for the running state.
- `attention <prefix>` iteratively walks the chain of recursive
  applications outside-in. At step *i*, the body sees
  `prefix = [head_0_emb, …, head_{i-1}_emb]` (the running list of
  non-recursive children's embeddings collected outside-in). The
  encoder's final embedding is the deepest (innermost) step's
  output.

### Graph-shaped sugar

Graph signatures declare `vertex_kinds` and `edge_kinds` (with
typed endpoints), and the encoder body uses message-passing:

```qvr
signature Mol {
    vertex_kinds { Atom : data dim 32, Bond : data dim 32 }
    edge_kinds   { bonded : Atom -- Atom, in_bond : Atom -> Bond }
}

encoder GNN over Mol {
    iterations 4

    init Atom(a)               |-> atom_embed[a]
    init Bond(b)               |-> bond_embed[b]
    message[bonded](src, tgt)  |-> mlp_msg([src, tgt])
    message[in_bond](src, tgt) |-> mlp_in([src, tgt])
    update[Atom](self, msgs)   |-> gru_update_atom(self, mean(msgs))
    update[Bond](self, msgs)   |-> gru_update_bond(self, mean(msgs))
    readout                    |-> mean_pool
}
```

Undirected edges (`Atom -- Atom`) emit messages in both
directions; directed edges (`Atom -> Bond`) emit only in the
declared direction.

## Decoder blocks

A decoder declares a Kleisli arrow `Vec_D → Kern(T_Σ)`. Per-sort
`structure` / `primitive` / `factor` / `binder_select` heads are
scaffolded as learnable neural networks; the corecursion (op
choice, factor split, recursive descent, BoundVar fallback to
in-scope variables) is supplied by the framework.

<!-- compile: cumulative -->
```qvr
decoder D over LF depth 8 { body |-> recursive }
```

`sample(vec) -> Term` draws a single term; `log_prob(term, vec) ->
Tensor` scores an observed term. Depth-bounded termination: at
the depth limit the choice set is restricted to ops whose every
child sort is data or index (never object); if no such op exists
at a sort, the decoder raises with a precise diagnostic.

## Loss declarations

Losses are first-class declarations attachable at any site in the
training graph:

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

`prog.losses.evaluate(env)` sums every loss; `evaluate_on(kind,
target, env)` filters by attachment. Inside a deduction, the
runtime accumulates rule-attached and chart-attached losses into
`ChartView.attached_loss` for the training driver to read.

## Deduction integration

A `deduction` block may attach an item signature and a
encoder; the chart's `embedding(item)` operation then returns
a differentiable vector computed by the attached encoder's
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
inside the encoder's per-op function — outside the chart's
role — so the semiring abstraction is not broken.

## Bayesian integration

A decoder is a Kleisli arrow into a structured type — *that is*,
a distribution over terms. So everything the program / posterior
machinery does with distributions over scalars and vectors
extends, with no special-casing, to distributions over structured
objects:

- **Structured latents:** `latent_term <- D(prior_vec)` inside a
  program body draws a random term from the decoder.
- **Observations of structured data:**
  `observe known_term <- D(some_vec)` scores under the decoder's
  log-prob.
- **Marginalisation over structured latents:** `marginalize t :
  Term <- D(v) in { observe y <- … }` integrates `t` out via the
  decoder's depth-bounded categorical recursion.
- **Variational decoders:** `program q : Sentence -> Term ! Sample,
  Score over generative` defines a variational decoder over LFs
  given a sentence.

## Stdlib shapes

`quivers.structural.shapes` provides ready-to-use signatures and
encoders / decoders for the three principal compressible
shapes:

- `seq_signature(name, dim)` — `Seq[A] = Nil | Cons(A, Seq)`.
- `rnn_encoder(sig, dim)` — GRU-cell right-fold.
- `transformer_encoder(sig, dim)` — head + tail-projection MLP.
- `bow_encoder(sig, dim)` — order-independent sum.
- `ar_decoder(sig, dim, vocab)` — autoregressive decoder.

- `tree_signature(name, dim)` — `Tree[L, B] = Leaf(L) | Node(B,
  Tree, Tree)`.
- `tree_lstm_encoder(sig, dim)` — child-sum binary-tree LSTM
  (Tai et al. 2015).
- `tree_decoder(sig, dim, leaf_vocab, label_vocab)` — top-down
  structural decoder.

- `graph_signature(name, vertex_kinds, edge_kinds)` —
  vertex / edge-kinded graph signature.
- `gnn_encoder(sig, iterations, dim, readout)` — per-edge-kind
  message MLP, per-vertex-kind GRU update, mean / sum / max
  readout.

All shapes are realised on top of the generic `Encoder` and
`Decoder` runtimes — the same algebra-homomorphism / Kleisli
coalgebra pattern, no special-casing.
