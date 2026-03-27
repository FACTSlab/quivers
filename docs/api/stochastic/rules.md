# Rule Systems

Composable structural rule systems for chart parsing. A `RuleSystem` encapsulates binary and unary inference rules as index tensors and supports merging (`+`) for hybrid grammars.

Each rule system optionally carries **learnable weights**, one log-weight per binary rule and one per unary rule. These weights are added to the chart combination scores during parsing. When a `RuleSystem` is passed to a `ChartParser`, its weights initialize the parser's `nn.Parameter` tensors (or fixed buffers if `learnable_rule_weights=False`). Weight-preserving merge: when two rule systems are combined with `+`, weights from both systems are preserved for their respective rules; duplicate rules keep the weight from the left operand.

The convenience functions `ccg_rules` and `lambek_rules` instantiate
[rule schema](biclosed.md) presets over a given category system.
For custom grammars, compose the schema primitives directly via `|`.

::: quivers.stochastic.rules
