# Inline Distribution Builders

`quivers.continuous.inline` defines distribution families for inline
use in program bodies, as in `sample x <- Family(args)`. These builders
are distinct from parametric kernel families. The module includes
`Normal`, `Beta`, `Exponential`, `Gamma`, `HalfCauchy`, `HalfNormal`,
and `LogNormal`, together with the `FixedDistribution` wrapper used by
[`bayesian_lift_parameters`](../inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters)
to declare prior morphisms.

::: quivers.continuous.inline
