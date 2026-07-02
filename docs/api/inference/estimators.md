# Estimators

Gradient-estimator strategies plugged into [Objectives](elbo.md): `Reparameterized` (default pathwise gradient), `StickingTheLanding` (variance reduction near convergence), `DoublyReparameterized` (DReG for `IWAEBound` at large K), and `ScoreFunction` (REINFORCE, for non-reparameterizable sites).

::: quivers.inference.estimators
