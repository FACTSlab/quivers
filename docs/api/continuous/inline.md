# Inline Distribution Builders

The morphisms the DSL compiler emits for an inline family call such as
`sample x <- Normal(loc=mu, scale=1.0)`. A call whose arguments are all
literals becomes a [`FixedDistribution`](#quivers.continuous.inline.FixedDistribution),
built by a `make_fixed_*` factory; a call with at least one bound
argument becomes a
[`MixedInlineDistribution`](#quivers.continuous.inline.MixedInlineDistribution),
which reads the bound arguments from its stacked input and fills in the
literals. Both can be constructed directly to build a program in Python.

::: quivers.continuous.inline
