# Rule Systems

Composable structural rule systems for chart parsing. A `RuleSystem`
stores binary and unary inference rules as index tensors. The `+`
operator merges rule systems for hybrid grammars.

A rule system may carry **learnable weights**, one log weight per binary
rule and one per unary rule. Parsing adds these weights to the chart
combination scores. When passed to a `ChartParser`, the weights
initialize its `nn.Parameter` tensors, or fixed buffers when
`learnable_rule_weights=False`. Merging with `+` preserves each rule's
weight; duplicate rules keep the weight from the left operand.

The convenience functions `ccg_rules` and `lambek_rules` instantiate
[rule schema](biclosed.md) presets over a given category system.
For custom grammars, compose the schema primitives directly via `|`.

::: quivers.stochastic.rules
