# Deduction Primitives

Abstract building blocks for chart-based weighted deduction:
`Axiom`, `Deduction`, `Goal`, `Schedule`, and the
`DeductiveSystem` protocol. The concrete agenda-based
`DeductionSystem` (in `quivers.stochastic.agenda`) is the
implementation used by the DSL compiler. The symbols in this module
also support custom-deduction subclasses and the inside-algorithm
framework in [`quivers.stochastic.inside`](../inside.md).

::: quivers.stochastic.deduction.primitives
