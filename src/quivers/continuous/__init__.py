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
ConditionalTruncatedNormal, ConditionalDirichlet

Boundary morphisms
------------------
Discretize, Embed

Normalizing flows
-----------------
ConditionalFlow, AffineCouplingLayer
"""

from quivers.continuous.spaces import (
    ContinuousSpace,
    Euclidean,
    UnitInterval,
    Simplex,
    PositiveReals,
    ProductSpace,
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
    # discrete-valued
    ConditionalBernoulli,
    ConditionalCategorical,
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
    "ContinuousSpace",
    "Euclidean",
    "UnitInterval",
    "Simplex",
    "PositiveReals",
    "ProductSpace",
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
    # discrete-valued conditional distributions
    "ConditionalBernoulli",
    "ConditionalCategorical",
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
