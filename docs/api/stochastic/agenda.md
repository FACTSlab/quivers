# Agenda-based Deduction

The concrete chart implementation used for DSL-compiled weighted
deductions. `DeductionSystem` combines an axiom injector, a rule
system, a semiring, and an agenda schedule in an `nn.Module`.
`DeductionSystem.__call__` returns the differentiable `ChartView`
presheaf.

For fitting, sampling, and the NUTS wrapper, see
[`api/stochastic/deduction`](deduction.md).

## Item algebra

Items, patterns, and bindings are plain tuples and dictionaries; these
aliases name their roles in the signatures below.

::: quivers.stochastic
    options:
      members:
        - Item
        - Pattern
        - Bindings
      show_root_heading: false
      show_root_toc_entry: false

::: quivers.stochastic.agenda
    options:
      filters: ["!^_", "!^(Item|Pattern|Bindings)$"]
