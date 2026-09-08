# Encoders

`Encoder` is a `torch.nn.Module` that realizes an F-algebra
homomorphism `T_Σ → Vec_D` from terms over a signature to
fixed-length vectors. The framework supplies the recursion. The
analyst may specify the parametric function for each operation or use
the compiler-generated two-layer MLPs, whose input dimensions follow
the operation arguments.

For binders, the framework threads a typed de Bruijn context Γ through
the recursion. It compresses each binder's annotations in the outer
context, creates variable embeddings with `var_init_fns`, and recurses
over the scoped arguments in the extended context.

`forward_graph` compresses graph signatures. It applies an initial
embedder for each vertex kind, a finite number of message-passing
rounds with functions indexed by edge and vertex kind, and a readout
that reduces the final vertex embeddings to one graph-level vector.

::: quivers.structural.encoder
