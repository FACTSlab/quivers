# Chart Parsers

Differentiable CKY chart parsers implemented as weighted deductive systems. `ChartParser` extends `DeductiveSystem`, composing `LexicalAxiom`, `BinarySpanDeduction`, `UnarySpanDeduction`, `SpanGoal`, and `CKYSchedule` into a single `nn.Module`. Supports learnable rule weights and semiring-parameterized scoring (log-probability, Viterbi, boolean, counting).

Use `ChartParser.from_schema(schema, category_system, ...)` to construct a parser from composable `RuleSchema` objects, or `ChartParser.from_category_system(...)` to construct from an explicit `RuleSystem`.

Includes concrete `CCGParser` and `LambekParser` convenience subclasses.

::: quivers.stochastic.parsers

::: quivers.stochastic.ccg

::: quivers.stochastic.lambek
