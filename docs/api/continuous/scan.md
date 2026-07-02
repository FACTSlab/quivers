# Scan Morphism

`ScanMorphism(cell)` realises the iterated Kleisli composition
of a per-step cell across a sequence input.
`ScanMorphism.rsample(x)` runs the per-step kernel forward,
threading the hidden state; `ScanMorphism.log_joint(x,
hidden_states)` returns the per-step log-density sum and accepts
the hidden-state trajectory either as a positional tensor or as
a `{state_key: tensor}` dict, so a caller passing observations
as a dict keyed by `state_key` (default `"h"`) can invoke
`log_joint` without an adapter.

::: quivers.continuous.scan
