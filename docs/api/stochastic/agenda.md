# Agenda-based Deduction

The concrete chart implementation underlying every weighted
deduction in the DSL. `DeductionSystem` ties an axiom injector,
a rule system, a semiring, and an agenda schedule into an
`nn.Module`; `ChartView` is the differentiable presheaf returned
from `DeductionSystem.__call__`.

For the user-facing surface (fit, sample, NUTS wrap), see
[`api/stochastic/deduction`](deduction.md).

::: quivers.stochastic.agenda
