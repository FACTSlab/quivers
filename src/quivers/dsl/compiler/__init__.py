"""quivers.dsl.compiler: AST -> Program compiler.

Re-exports the public surface from the package's submodules.
"""

# `core` is imported before `_prelude`: `_prelude` imports from
# `quivers.transpile`, which imports back into this package, and loading
# `core` first completes that cycle before `_prelude` runs.
from quivers.dsl.compiler.core import Compiler
from quivers.dsl.compiler._prelude import CompileError

__all__ = ["CompileError", "Compiler"]
