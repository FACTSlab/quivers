# Chart Parsers

Differentiable CKY chart parsers implemented as weighted deductive
systems. `ChartParser` extends `DeductiveSystem` by composing
`LexicalAxiom`, `BinarySpanDeduction`, `UnarySpanDeduction`, `SpanGoal`,
and `CKYSchedule` in one `nn.Module`. It supports learnable rule weights
and semiring-parameterized scoring for log probability, Viterbi,
Boolean recognition, and counting.

`ChartParser.from_schema(schema, category_system, ...)` constructs a
parser from composable `RuleSchema` objects.
`ChartParser.from_category_system(...)` instead accepts an explicit
`RuleSystem`.

Includes concrete `CCGParser` and `LambekParser` convenience subclasses.

::: quivers.stochastic.parsers

::: quivers.stochastic.ccg

::: quivers.stochastic.lambek
