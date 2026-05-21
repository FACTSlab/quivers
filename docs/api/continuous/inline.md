# Inline Distribution Builders

`quivers.continuous.inline` ships the `_FAMILY_BUILDERS` registry
for distribution families that are usable inline in program
bodies (as `sample x <- Family(args)` site forms) but are not
parametric kernel families. The registry covers the standard
fixed-distribution shapes (`Normal`, `Beta`, `Exponential`,
`Gamma`, `HalfCauchy`, `HalfNormal`, `LogNormal`, etc.) plus the
`FixedDistribution` wrapper used by
[`bayesian_lift_parameters`](../inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters)
to declare prior morphisms.

::: quivers.continuous.inline
