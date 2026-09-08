# Estimators

Gradient-estimator strategies for [Objectives](elbo.md):
`Reparameterized` supplies the default pathwise gradient;
`StickingTheLanding` reduces variance near convergence;
`DoublyReparameterized` supplies DReG for `IWAEBound` at large K; and
`ScoreFunction` supplies REINFORCE for non-reparameterizable sites.

::: quivers.inference.estimators
