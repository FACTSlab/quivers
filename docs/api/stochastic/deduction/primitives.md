# Deduction Primitives

Abstract building blocks for chart-based weighted deduction:
`Axiom`, `Deduction`, `Goal`, `Schedule`, and the
`DeductiveSystem` protocol. The concrete agenda-based
`DeductionSystem` (in `quivers.stochastic.agenda`) is the
canonical realization used by the DSL compiler; the symbols here
exist for custom-deduction subclasses and the inside-algorithm
framework in [`quivers.stochastic.inside`](../inside.md).

::: quivers.stochastic.deduction.primitives
