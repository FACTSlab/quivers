# Variational Guides

Variational guide distributions for approximate inference. The guides
live in submodules of `quivers.inference.guides` and share the `Guide`
ABC and `LatentRegistry` introspection layer. They are
`AutoNormalGuide`, `AutoDeltaGuide`, `AutoMultivariateNormalGuide`,
`AutoLowRankMultivariateNormalGuide`, `AutoLaplaceApproximation`,
`AutoNormalizingFlow`, `AutoIAFGuide`, `AutoNeuralSplineGuide`,
`AutoMixtureGuide`, `AutoGuideList`, and `AutoStructured`.

`quivers.inference` re-exports every guide, and `quivers.inference.guides`
adds the Pyro-style short names (`AutoNormal`, `AutoMultivariateNormal`,
`AutoLowRankMVN`, `AutoDelta`, `AutoLaplace`, `AutoIAFNormal`), each bound
to the guide class it names, and the type aliases that configure
`AutoStructured`.

::: quivers.inference.guides

## Latent registry

Every guide and MCMC kernel reads the model's latent sites through a
`LatentRegistry`.

::: quivers.inference.registry

## Flow transforms

The learnable bijections that `AutoNormalizingFlow` and its subclasses
stack. A custom flow subclasses `TransformModule`.

::: quivers.inference.transforms

## Annealed importance sampling

::: quivers.inference.dais
