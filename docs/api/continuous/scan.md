# Scan Morphism

`scan(cell)` realises the iterated Kleisli composition of a
per-step cell across a sequence input. `ScanMorphism.rsample(x)`
runs the per-step kernel forward, threading the hidden state;
`ScanMorphism.log_joint(x, h)` returns the per-step log-density
sum and accepts the hidden-state trajectory either as a
positional tensor or as a `{state_key: tensor}` dict, so the
standard inference contract `log_joint(x, observations: dict)`
works without an adapter.

::: quivers.continuous.scan
