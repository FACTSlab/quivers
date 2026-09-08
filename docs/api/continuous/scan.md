# Scan Morphism

`ScanMorphism(cell)` realizes the iterated Kleisli composition of a
per-step cell over a sequence. `ScanMorphism.rsample(x)` applies the
kernel at each step while threading the hidden state.
`ScanMorphism.log_joint(x, hidden_states)` sums the per-step log
density. It accepts the hidden-state trajectory either as a positional
tensor or as a `{state_key: tensor}` dictionary. The default
`state_key` is `"h"`.

::: quivers.continuous.scan
