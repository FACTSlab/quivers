# Encoders

`Encoder` is a `torch.nn.Module` that realises an F-algebra
homomorphism `T_Σ → Vec_D` from terms over a signature to
fixed-length vectors. The recursion is supplied by the framework;
the analyst supplies only the per-operation parametric functions
(or omits them and accepts the compiler's scaffolded 2-layer
MLP defaults with correct per-arg dimensions).

For binders, the framework threads a typed de-Bruijn context Γ
through the recursion: each binder's annotations are compressed
in the outer context, fresh variable embeddings are minted via
the encoder's `var_init_fns`, and the scoped arguments are
recursed under the extended context.

Graph signatures are compressed via `forward_graph`: per-vertex-kind
initial embedders, finitely many message-passing rounds with
per-edge-kind message functions and per-vertex-kind update
functions, finally a readout reducing the per-vertex final
embeddings to a single graph-level vector.

::: quivers.structural.encoder
