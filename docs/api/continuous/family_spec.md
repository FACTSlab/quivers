# Family Registry

Every distribution family is described once by a
[`FamilySpec`](#quivers.continuous.family_spec.FamilySpec) and registered
in [`FAMILY_REGISTRY`](#quivers.continuous.family_spec.FAMILY_REGISTRY).
The inline builders and conditional families dispatch on the registered
specification, so registering a family through
[`register_family`](#quivers.continuous.family_spec.register_family)
(and refreshing the inline tables with
[`reload_inline_registry`](inline.md#quivers.continuous.inline.reload_inline_registry))
makes it available to programs.

::: quivers.continuous.family_spec
