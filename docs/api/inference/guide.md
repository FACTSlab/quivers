# Variational Guides

Variational guide distributions for approximate inference. The shipped guides (`AutoNormalGuide`, `AutoDeltaGuide`, `AutoMultivariateNormalGuide`, `AutoLowRankMultivariateNormalGuide`, `AutoLaplaceApproximation`, `AutoNormalizingFlow`, `AutoIAFGuide`, `AutoNeuralSplineGuide`, `AutoMixtureGuide`, `AutoGuideList`, `AutoStructured`) live as submodules of `quivers.inference.guides` and share the `Guide` ABC and the `LatentRegistry` introspection layer.

::: quivers.inference.guides
