# Signatures

Multi-sorted algebra signatures with constructors, typed
binders, and (optionally) vertex / edge kinds for
graph-shaped signatures. The runtime carrier for
`signature { … }` DSL declarations.

The de-Bruijn discipline is enforced structurally: a
`BoundVar(i)` term carries an integer index; binders push
a fresh `ContextEntry(var_sort, embedding, type_term)` onto
an implicit context Γ tracked by the encoder / decoder
runtime. Binder variables may carry an annotation sort —
the variable's type is stored alongside its embedding in Γ.

::: quivers.structural.signature
