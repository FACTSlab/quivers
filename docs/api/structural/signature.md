# Signatures

Multi-sorted algebra signatures with constructors, typed binders, and
optional vertex and edge kinds for graph-shaped signatures. These
objects are the runtime carriers of `signature { … }` DSL declarations.

The de Bruijn discipline is enforced structurally: a
`BoundVar(i)` term carries an integer index; binders push
a fresh `ContextEntry(var_sort, embedding, type_term)` onto
an implicit context Γ tracked by the encoder / decoder
runtime. Binder variables may carry an annotation sort:
the variable's type is stored alongside its embedding in Γ.

::: quivers.structural.signature
    options:
      filters: ["!^_", "!^(DataLeaf|TermArg|SortKind|EMPTY_CONTEXT)$"]

## Type aliases and constants

::: quivers.structural
    options:
      members:
        - DataLeaf
        - TermArg
        - SortKind
        - EMPTY_CONTEXT
      show_root_heading: false
      show_root_toc_entry: false
