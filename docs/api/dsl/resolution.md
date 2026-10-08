# Resolution

The compiler mixin that resolves QVR `ObjectExpr` ASTs to runtime
`SetObject` and `ContinuousSpace` values. A single walk handles both
strata, so a mixed-domain product such as `Real 3 * Token` resolves to
one `ProductSpace`.

!!! note "Internal module"
    This module is an implementation detail of the compiler and its
    tooling. Its `__all__` is empty, so none of the names below are
    part of the public API, and they may change without notice.

::: quivers.dsl.compiler.resolution
