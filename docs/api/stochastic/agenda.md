# Agenda-based Deduction

The concrete chart implementation used for DSL-compiled weighted
deductions. `DeductionSystem` combines an axiom injector, a rule
system, a semiring, and an agenda schedule in an `nn.Module`.
`DeductionSystem.__call__` returns the differentiable `ChartView`
presheaf.

For fitting, sampling, and the NUTS wrapper, see
[`api/stochastic/deduction`](deduction.md).

::: quivers.stochastic.agenda
