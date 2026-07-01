# Constraint Solver

`check_constraints(module)` walks a parsed `Module` and reports
well-formedness violations (residuated context, effect-name
convention, bundle-member resolvability) without invoking the full
compiler.

Import from the submodule directly:

```python
from quivers.dsl.constraints import check_constraints, Violation
```

These names are not re-exported from `quivers.dsl`.

::: quivers.dsl.constraints
