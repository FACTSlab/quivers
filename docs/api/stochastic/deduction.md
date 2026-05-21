# Deduction Systems

The user-facing surface for working with weighted chart
deductions: the model type, its abstract primitives, and three
orthogonal operations on it. All public symbols re-exported from
`quivers.stochastic.deduction`.

| Submodule | Job |
|---|---|
| [`primitives`](deduction/primitives.md) | Abstract building blocks (`Axiom`, `Deduction`, `Goal`, `Schedule`, `DeductiveSystem`). Most users do not touch these directly. |
| [`fit`](deduction/fit.md) | Point-estimate gradient fitting (MAP / MLE) of the deduction's learnable log-weights. |
| [`bayes`](deduction/bayes.md) | Lift the parameters into a Bayesian `MonadicProgram` whose posterior NUTS / SVI can target. |
| [`sample`](deduction/sample.md) | Exact length-conditional forward sampling of yields from the chart's distribution. |

::: quivers.stochastic.deduction
