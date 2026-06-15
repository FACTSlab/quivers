"""Continuous morphisms: the hybrid discrete-continuous architecture.

This subpackage extends quivers with true continuous distributions,
enabling morphisms between continuous measurable spaces alongside
the existing finite tensor infrastructure.

The key abstraction is ContinuousMorphism, which defines a conditional
distribution p(y | x) via two operations:

    log_prob(x, y) — evaluate the log-density/probability
    rsample(x)     — generate reparameterized samples

Composition uses ancestral sampling (exact for discrete intermediates,
Monte Carlo for continuous ones), and the >> and @ operators work
across discrete and continuous morphisms transparently.

Spaces
------
Euclidean, UnitInterval, Simplex, PositiveReals, ProductSpace

Morphisms
---------
ContinuousMorphism, SampledComposition, DiscreteAsContinuous

Parameterized families
----------------------
ConditionalNormal, ConditionalLogitNormal, ConditionalBeta,
ConditionalTruncatedNormal, ConditionalDirichlet,
ConditionalGaussianProcess, ConditionalHorseshoe

Boundary morphisms
------------------
Discretize, Embed

Normalizing flows
-----------------
ConditionalFlow, AffineCouplingLayer
"""

from quivers.continuous.spaces import (
    Ball,
    CholeskyFactor,
    ContinuousSpace,
    Correlation,
    Covariance,
    Diagonal,
    Euclidean,
    LowerTriangular,
    Orthogonal,
    PositiveReals,
    ProductSpace,
    Simplex,
    Sphere,
    Stiefel,
    UnitInterval,
)
from quivers.continuous.morphisms import (
    AnySpace,
    ContinuousMorphism,
    SampledComposition,
    ProductContinuousMorphism,
    DiscreteAsContinuous,
)
from quivers.continuous.families import (
    # hand-written (backward compatible)
    ConditionalNormal,
    ConditionalLogitNormal,
    ConditionalBeta,
    ConditionalTruncatedNormal,
    ConditionalDirichlet,
    # loc-scale family
    ConditionalCauchy,
    ConditionalLaplace,
    ConditionalGumbel,
    ConditionalLogNormal,
    ConditionalStudentT,
    # positive-valued
    ConditionalExponential,
    ConditionalGamma,
    ConditionalChi2,
    ConditionalHalfCauchy,
    ConditionalHalfNormal,
    ConditionalInverseGamma,
    ConditionalWeibull,
    ConditionalPareto,
    # (0, 1)-valued
    ConditionalKumaraswamy,
    ConditionalContinuousBernoulli,
    # two-df
    ConditionalFisherSnedecor,
    # special parameterization
    ConditionalUniform,
    # multivariate
    ConditionalMultivariateNormal,
    ConditionalLowRankMVN,
    # relaxed discrete
    ConditionalRelaxedBernoulli,
    ConditionalRelaxedOneHotCategorical,
    # matrix-valued
    ConditionalWishart,
    ConditionalInverseWishart,
    ConditionalMatrixNormal,
    # non-parametric and shrinkage
    ConditionalGaussianProcess,
    ConditionalHorseshoe,
    # discrete-valued
    ConditionalBernoulli,
    ConditionalCategorical,
    # Phase A: discrete count families
    ConditionalPoisson,
    ConditionalNegativeBinomial,
    ConditionalGeometric,
    ConditionalBinomial,
    # Phase A: circular / simplex / one-hot / correlation
    ConditionalVonMises,
    ConditionalLogisticNormal,
    ConditionalOneHotCategorical,
    ConditionalLKJCholesky,
    # Phase A: distribution wrappers
    ConditionalMixture,
    ConditionalIndependent,
    ConditionalTransformed,
    Truncated,
    LKJCorrelationFactor,
)
from quivers.continuous.programs import (
    MonadicProgram,
)
from quivers.continuous.scan import (
    ScanMorphism,
)
from quivers.continuous.boundaries import (
    Discretize,
    Embed,
)
from quivers.continuous.flows import (
    AffineCouplingLayer,
    ConditionalFlow,
)

# optional: GeneralizedPareto (torch version dependent)
try:
    from quivers.continuous.families import ConditionalGeneralizedPareto as _GPD_cls

    ConditionalGeneralizedPareto = _GPD_cls

except ImportError:
    pass

__all__ = [
    # spaces
    "Ball",
    "CholeskyFactor",
    "ContinuousSpace",
    "Correlation",
    "Covariance",
    "Diagonal",
    "Euclidean",
    "LowerTriangular",
    "Orthogonal",
    "PositiveReals",
    "ProductSpace",
    "Simplex",
    "Sphere",
    "Stiefel",
    "UnitInterval",
    # morphisms
    "AnySpace",
    "ContinuousMorphism",
    "SampledComposition",
    "ProductContinuousMorphism",
    "DiscreteAsContinuous",
    # families — original
    "ConditionalNormal",
    "ConditionalLogitNormal",
    "ConditionalBeta",
    "ConditionalTruncatedNormal",
    "ConditionalDirichlet",
    # families — loc-scale
    "ConditionalCauchy",
    "ConditionalLaplace",
    "ConditionalGumbel",
    "ConditionalLogNormal",
    "ConditionalStudentT",
    # families — positive-valued
    "ConditionalExponential",
    "ConditionalGamma",
    "ConditionalChi2",
    "ConditionalHalfCauchy",
    "ConditionalHalfNormal",
    "ConditionalInverseGamma",
    "ConditionalWeibull",
    "ConditionalPareto",
    # families — (0, 1)-valued
    "ConditionalKumaraswamy",
    "ConditionalContinuousBernoulli",
    # families — two-df
    "ConditionalFisherSnedecor",
    # families — special
    "ConditionalUniform",
    # families — multivariate
    "ConditionalMultivariateNormal",
    "ConditionalLowRankMVN",
    # families — relaxed discrete
    "ConditionalRelaxedBernoulli",
    "ConditionalRelaxedOneHotCategorical",
    # families — matrix-valued
    "ConditionalWishart",
    "ConditionalInverseWishart",
    "ConditionalMatrixNormal",
    # families: non-parametric and shrinkage
    "ConditionalGaussianProcess",
    "ConditionalHorseshoe",
    # discrete-valued conditional distributions
    "ConditionalBernoulli",
    "ConditionalCategorical",
    # Phase A: discrete count families
    "ConditionalPoisson",
    "ConditionalNegativeBinomial",
    "ConditionalGeometric",
    "ConditionalBinomial",
    # Phase A: circular / simplex / one-hot / correlation
    "ConditionalVonMises",
    "ConditionalLogisticNormal",
    "ConditionalOneHotCategorical",
    "ConditionalLKJCholesky",
    # Phase A: distribution wrappers
    "ConditionalMixture",
    "ConditionalIndependent",
    "ConditionalTransformed",
    "Truncated",
    "LKJCorrelationFactor",
    # monadic programs
    "MonadicProgram",
    # scan (temporal recurrence)
    "ScanMorphism",
    # boundaries
    "Discretize",
    "Embed",
    # flows
    "AffineCouplingLayer",
    "ConditionalFlow",
]
