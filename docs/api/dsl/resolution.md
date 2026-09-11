# Resolution Lenses

A `dx.Lens` family resolves QVR `TypeExpr` and `SpaceExpr` ASTs to
runtime `SetObject` and `ContinuousSpace` values, and back again. Each
lens carries its resolution environment, which contains the available
objects and spaces.

::: quivers.dsl.compiler.resolution
