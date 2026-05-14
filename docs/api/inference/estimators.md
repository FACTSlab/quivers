# Estimators

Gradient-estimator strategies plugged into [Objectives](elbo.md): `Reparameterised` (default pathwise gradient), `StickingTheLanding` (variance reduction near convergence), `DoublyReparameterised` (DReG for `IWAEBound` at large K), and `ScoreFunction` (REINFORCE, for non-reparameterisable sites).

::: quivers.inference.estimators
