# Parameter Sources

`quivers.continuous.param_source` provides the map from a conditional
family's input to its distribution parameters. A morphism declared
`~ Family` over a continuous domain is a Kleisli arrow whose
parameters are produced by a `ParamSource`, so the source is where a
kernel's dependence on its input is computed, and where any
nonlinearity in that dependence lives.

The concrete sources cover the standard architectures: `LinearSource`
is a single `nn.Linear`; `MLPSource` is a multi-layer perceptron with
configurable widths and activation; `AttentionSource` is a
self-attention head; `LookupSource` and `EmbeddingSource` handle
discrete domains; `IdentitySource`, `FunctionSource`, and
`ComposeSource` cover pass-through, a fixed callable, and composition
of two sources.

`make_param_source` is the factory the families call, and
`param_source_from_option` parses the DSL's
`[param_source=<kind>]` morphism option. The default for a continuous
domain is `MLPSource` with two hidden layers of width 64 and tanh
activations; a `SetObject` domain always uses `LookupSource`
regardless of the requested kind. The
[Bayesian Neural Network](../../examples/bnn.md) example selects the
MLP source explicitly and relies on it for its nonlinearity.

::: quivers.continuous.param_source
