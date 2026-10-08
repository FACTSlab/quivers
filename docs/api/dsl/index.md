# Overview

Parsing, compiling, and printing QVR source, with the public entry
points: `load` and `loads` compile source to a `Program`, `parse`
produces the AST that `Compiler` lowers and `module_to_source` prints,
and the QIEC lowering checks a module against the kernel. The package
re-exports the public names of its modules; the AST node classes live
in [`quivers.dsl.ast_nodes`](ast_nodes.md).

::: quivers.dsl
