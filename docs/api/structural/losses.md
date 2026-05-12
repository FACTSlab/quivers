# Loss registry

`LossRegistry` is the per-compiled-module table mapping
attachment sites (program names, deduction names, encoder /
decoder names, rule names, chart sites, global) to weighted
scalar-loss callables.

The training driver calls `LossRegistry.evaluate(env)` to sum
every registered loss; `evaluate_on(kind, target, env,
rule_deduction)` returns the weighted partial sum filtered to
a single attachment site.

Loss bodies are compiled by the QVR compiler as let-expression
closures of signature `(env) -> Tensor`. The `env` for a global
loss includes the compiled module's program / deduction /
encoder / decoder bindings as top-level names; for
rule-attached losses, the env also carries `"rule"`,
`"deduction"`, `"antecedents"`, `"conclusion"`, and `"weight"`
keys populated by the agenda's rule-firing callback.

::: quivers.structural.losses
