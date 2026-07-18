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
`param_source_from_option` parses the DSL's `[param_source=<kind>]`
morphism option. The default for a continuous domain is `LinearSource`,
so a kernel is linear unless it asks for something else; a `SetObject`
domain always uses `LookupSource` regardless of the requested kind. The
[Bayesian Neural Network](../../examples/bnn.md) example selects the
MLP source explicitly and relies on it for its nonlinearity.

The hidden widths come either from the option's arguments or from
`hidden_dim`, one width per hidden layer:

<!-- compile: false -->
```qvr
morphism f : X -> Y [param_source=mlp] ~ Normal                      # (64, 64)
morphism f : X -> Y [param_source=mlp(64, 32)] ~ Normal              # (64, 32)
morphism f : X -> Y [param_source=mlp, hidden_dim=[64, 32]] ~ Normal # (64, 32)
morphism f : X -> Y [param_source=mlp, hidden_dim=64] ~ Normal       # (64,)
```

A width given to a source with no hidden layers to apply it to is an
error rather than a silent no-op, and so is `param_source` on a family
whose parameters do not come from a source at all (`Horseshoe`,
`GaussianProcess`, `Independent`, `Transformed`).

::: quivers.continuous.param_source
