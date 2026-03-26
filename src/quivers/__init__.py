"""quivers: V-enriched categorical relations as PyTorch tensors.

Provides finite sets as categorical objects, V-enriched relations as
morphisms (tensors with values in a quantale's lattice), and
parameterized composition via quantale enrichment.

Quick start::

    from quivers import FinSet, morphism, observed, identity, Program

    X = FinSet("X", 3)
    Y = FinSet("Y", 4)
    Z = FinSet("Z", 2)

    f = morphism(X, Y)           # latent (learnable)
    g = morphism(Y, Z)           # latent (learnable)
    h = f >> g                   # V-enriched composition X -> Z

    program = Program(h)
    output = program()           # tensor of shape (3, 2)
"""

__version__ = "0.1.0"

from quivers.core.objects import (
    SetObject,
    FinSet,
    ProductSet,
    CoproductSet,
    FreeMonoid,
    Unit,
)
from quivers.core.quantales import (
    Quantale,
    ProductFuzzy,
    BooleanQuantale,
    PRODUCT_FUZZY,
    BOOLEAN,
)
from quivers.core.morphisms import (
    Morphism,
    ObservedMorphism,
    LatentMorphism,
    ComposedMorphism,
    ProductMorphism,
    MarginalizedMorphism,
    FunctorMorphism,
    morphism,
    observed,
    identity,
)
from quivers.categorical.functors import (
    Functor,
    IdentityFunctor,
    ComposedFunctor,
    FreeMonoidFunctor,
    IDENTITY,
)
from quivers.program import Program
from quivers.core.tensor_ops import (
    noisy_or_contract,
    noisy_or_reduce,
    noisy_and_reduce,
    componentwise_lift,
)
from quivers.enriched.ends_coends import coend, end
from quivers.categorical.natural_transformations import (
    NaturalTransformation,
    ComponentwiseNT,
)
from quivers.monadic.monads import (
    Monad,
    KleisliCategory,
    FuzzyPowersetMonad,
    FreeMonoidMonad,
)
from quivers.categorical.adjunctions import (
    Adjunction,
    ForgetfulFunctor,
    FreeForgetfulAdjunction,
)
from quivers.enriched.kan_extensions import (
    ObjectMap,
    Projection,
    Inclusion,
    left_kan,
    right_kan,
)
from quivers.enriched.profunctors import Profunctor
from quivers.categorical.monoidal import (
    MonoidalStructure,
    CartesianMonoidal,
    CoproductMonoidal,
    EmptySet,
    EMPTY,
)
from quivers.categorical.base_change import (
    BaseChange,
    BoolToFuzzy,
    FuzzyToBool,
)
from quivers.monadic.distributive_laws import (
    DistributiveLaw,
    FreeMonoidPowersetLaw,
)

# -- new extensions --

from quivers.monadic.comonads import (
    Comonad,
    CoKleisliCategory,
    DiagonalComonad,
    CofreeComonad,
)
from quivers.monadic.algebras import (
    Algebra,
    FreeAlgebra,
    ObservedAlgebra,
    Coalgebra,
    CofreeCoalgebra,
    ObservedCoalgebra,
    EilenbergMooreCategory,
)
from quivers.enriched.weighted_limits import (
    Weight,
    Diagram,
    weighted_limit,
    weighted_colimit,
    weighted_limit_morphisms,
    weighted_colimit_morphisms,
    representable_weight,
    terminal_weight,
)
from quivers.core.extra_quantales import (
    LukasiewiczQuantale,
    GodelQuantale,
    TropicalQuantale,
    LUKASIEWICZ,
    GODEL,
    TROPICAL,
)
from quivers.enriched.yoneda import (
    Presheaf,
    representable_profunctor,
    corepresentable_profunctor,
    yoneda_embedding,
    yoneda_lemma,
    yoneda_density,
    verify_yoneda_fully_faithful,
)
from quivers.enriched.day_convolution import (
    day_convolution,
    day_unit,
    day_convolution_profunctors,
)
from quivers.enriched.optics import (
    Optic,
    Lens,
    Prism,
    Adapter,
    Grate,
    compose_optics,
)
from quivers.categorical.traced import (
    TracedMonoidal,
    CartesianTrace,
    IterativeTrace,
    trace,
    partial_trace,
)

# stochastic (FinStoch / Markov kernels)
from quivers.stochastic import (
    MarkovQuantale,
    MARKOV,
    StochasticMorphism,
    CategoricalMorphism,
    DiscretizedNormal,
    DiscretizedLogitNormal,
    DiscretizedBeta,
    DiscretizedTruncatedNormal,
    ConditionedMorphism,
    MixtureMorphism,
    FactoredMorphism,
    NormalizedMorphism,
    condition,
    mix,
    factor,
    normalize,
    prob,
    marginal_prob,
    expectation,
    stochastic,
)
from quivers.giry import (
    GiryMonad,
    FinStoch,
)

# continuous (hybrid discrete-continuous architecture)
from quivers.continuous import (
    ContinuousSpace,
    Euclidean,
    UnitInterval,
    Simplex,
    PositiveReals,
    ProductSpace,
    ContinuousMorphism,
    SampledComposition,
    ProductContinuousMorphism,
    DiscreteAsContinuous,
    # families — original
    ConditionalNormal,
    ConditionalLogitNormal,
    ConditionalBeta,
    ConditionalTruncatedNormal,
    ConditionalDirichlet,
    # families — loc-scale
    ConditionalCauchy,
    ConditionalLaplace,
    ConditionalGumbel,
    ConditionalLogNormal,
    ConditionalStudentT,
    # families — positive-valued
    ConditionalExponential,
    ConditionalGamma,
    ConditionalChi2,
    ConditionalHalfCauchy,
    ConditionalHalfNormal,
    ConditionalInverseGamma,
    ConditionalWeibull,
    ConditionalPareto,
    # families — (0, 1)-valued
    ConditionalKumaraswamy,
    ConditionalContinuousBernoulli,
    # families — two-df
    ConditionalFisherSnedecor,
    # families — special
    ConditionalUniform,
    # families — multivariate
    ConditionalMultivariateNormal,
    ConditionalLowRankMVN,
    # families — relaxed discrete
    ConditionalRelaxedBernoulli,
    ConditionalRelaxedOneHotCategorical,
    # families — matrix-valued
    ConditionalWishart,
    # families — discrete-valued
    ConditionalBernoulli,
    ConditionalCategorical,
    # monadic programs
    MonadicProgram,
    # boundaries & flows
    Discretize,
    Embed,
    AffineCouplingLayer,
    ConditionalFlow,
)

# dsl
from quivers.dsl import (
    parse as dsl_parse,
    loads as dsl_loads,
    load as dsl_load,
    LexError,
    ParseError,
    CompileError,
)

__all__ = [
    # objects
    "SetObject",
    "FinSet",
    "ProductSet",
    "CoproductSet",
    "FreeMonoid",
    "Unit",
    "EmptySet",
    "EMPTY",
    # quantales
    "Quantale",
    "ProductFuzzy",
    "BooleanQuantale",
    "PRODUCT_FUZZY",
    "BOOLEAN",
    # extra quantales
    "LukasiewiczQuantale",
    "GodelQuantale",
    "TropicalQuantale",
    "LUKASIEWICZ",
    "GODEL",
    "TROPICAL",
    # morphisms
    "Morphism",
    "ObservedMorphism",
    "LatentMorphism",
    "ComposedMorphism",
    "ProductMorphism",
    "MarginalizedMorphism",
    "FunctorMorphism",
    "morphism",
    "observed",
    "identity",
    # functors
    "Functor",
    "IdentityFunctor",
    "ComposedFunctor",
    "FreeMonoidFunctor",
    "IDENTITY",
    # program
    "Program",
    # tensor ops
    "noisy_or_contract",
    "noisy_or_reduce",
    "noisy_and_reduce",
    "componentwise_lift",
    # ends and coends
    "coend",
    "end",
    # natural transformations
    "NaturalTransformation",
    "ComponentwiseNT",
    # monads
    "Monad",
    "KleisliCategory",
    "FuzzyPowersetMonad",
    "FreeMonoidMonad",
    # comonads
    "Comonad",
    "CoKleisliCategory",
    "DiagonalComonad",
    "CofreeComonad",
    # algebras and coalgebras
    "Algebra",
    "FreeAlgebra",
    "ObservedAlgebra",
    "Coalgebra",
    "CofreeCoalgebra",
    "ObservedCoalgebra",
    "EilenbergMooreCategory",
    # adjunctions
    "Adjunction",
    "ForgetfulFunctor",
    "FreeForgetfulAdjunction",
    # kan extensions
    "ObjectMap",
    "Projection",
    "Inclusion",
    "left_kan",
    "right_kan",
    # profunctors
    "Profunctor",
    # monoidal
    "MonoidalStructure",
    "CartesianMonoidal",
    "CoproductMonoidal",
    # base change
    "BaseChange",
    "BoolToFuzzy",
    "FuzzyToBool",
    # distributive laws
    "DistributiveLaw",
    "FreeMonoidPowersetLaw",
    # weighted limits
    "Weight",
    "Diagram",
    "weighted_limit",
    "weighted_colimit",
    "weighted_limit_morphisms",
    "weighted_colimit_morphisms",
    "representable_weight",
    "terminal_weight",
    # yoneda
    "Presheaf",
    "representable_profunctor",
    "corepresentable_profunctor",
    "yoneda_embedding",
    "yoneda_lemma",
    "yoneda_density",
    "verify_yoneda_fully_faithful",
    # day convolution
    "day_convolution",
    "day_unit",
    "day_convolution_profunctors",
    # optics
    "Optic",
    "Lens",
    "Prism",
    "Adapter",
    "Grate",
    "compose_optics",
    # traced monoidal
    "TracedMonoidal",
    "CartesianTrace",
    "IterativeTrace",
    "trace",
    "partial_trace",
    # stochastic
    "MarkovQuantale",
    "MARKOV",
    "StochasticMorphism",
    "CategoricalMorphism",
    "DiscretizedNormal",
    "DiscretizedLogitNormal",
    "DiscretizedBeta",
    "DiscretizedTruncatedNormal",
    "ConditionedMorphism",
    "MixtureMorphism",
    "FactoredMorphism",
    "NormalizedMorphism",
    "condition",
    "mix",
    "factor",
    "normalize",
    "prob",
    "marginal_prob",
    "expectation",
    "stochastic",
    # giry monad
    "GiryMonad",
    "FinStoch",
    # continuous — spaces
    "ContinuousSpace",
    "Euclidean",
    "UnitInterval",
    "Simplex",
    "PositiveReals",
    "ProductSpace",
    # continuous — morphisms
    "ContinuousMorphism",
    "SampledComposition",
    "ProductContinuousMorphism",
    "DiscreteAsContinuous",
    # continuous — families (original)
    "ConditionalNormal",
    "ConditionalLogitNormal",
    "ConditionalBeta",
    "ConditionalTruncatedNormal",
    "ConditionalDirichlet",
    # continuous — families (loc-scale)
    "ConditionalCauchy",
    "ConditionalLaplace",
    "ConditionalGumbel",
    "ConditionalLogNormal",
    "ConditionalStudentT",
    # continuous — families (positive-valued)
    "ConditionalExponential",
    "ConditionalGamma",
    "ConditionalChi2",
    "ConditionalHalfCauchy",
    "ConditionalHalfNormal",
    "ConditionalInverseGamma",
    "ConditionalWeibull",
    "ConditionalPareto",
    # continuous — families ((0,1)-valued)
    "ConditionalKumaraswamy",
    "ConditionalContinuousBernoulli",
    # continuous — families (two-df)
    "ConditionalFisherSnedecor",
    # continuous — families (special)
    "ConditionalUniform",
    # continuous — families (multivariate)
    "ConditionalMultivariateNormal",
    "ConditionalLowRankMVN",
    # continuous — families (relaxed discrete)
    "ConditionalRelaxedBernoulli",
    "ConditionalRelaxedOneHotCategorical",
    # continuous — families (matrix-valued)
    "ConditionalWishart",
    # continuous — families (discrete-valued)
    "ConditionalBernoulli",
    "ConditionalCategorical",
    # continuous — monadic programs
    "MonadicProgram",
    "Discretize",
    "Embed",
    "AffineCouplingLayer",
    "ConditionalFlow",
    # dsl
    "dsl_parse",
    "dsl_loads",
    "dsl_load",
    "LexError",
    "ParseError",
    "CompileError",
]
