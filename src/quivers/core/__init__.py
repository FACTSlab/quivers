"""Core modules for the quivers package.

Canonical location for all core types and functions:
- objects: SetObject, FinSet, ProductSet, CoproductSet, FreeMonoid
- quantales: Quantale, ProductFuzzy, BooleanQuantale, and singletons
- extra_quantales: LukasiewiczQuantale, GodelQuantale, TropicalQuantale
- morphisms: Morphism hierarchy and factory functions
- tensor_ops: Tensor contraction and lifting operations
"""

from quivers.core._util import EPS, clamp_probs, safe_log1p_neg
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
from quivers.core.extra_quantales import (
    LukasiewiczQuantale,
    GodelQuantale,
    TropicalQuantale,
    LUKASIEWICZ,
    GODEL,
    TROPICAL,
)
from quivers.core.morphisms import (
    Morphism,
    ObservedMorphism,
    LatentMorphism,
    ComposedMorphism,
    ProductMorphism,
    MarginalizedMorphism,
    FunctorMorphism,
    RepeatMorphism,
    morphism,
    observed,
    identity,
)
from quivers.core.tensor_ops import (
    noisy_or_contract,
    noisy_or_reduce,
    noisy_and_reduce,
    componentwise_lift,
)

__all__ = [
    # _util
    "EPS",
    "clamp_probs",
    "safe_log1p_neg",
    # objects
    "SetObject",
    "FinSet",
    "ProductSet",
    "CoproductSet",
    "FreeMonoid",
    "Unit",
    # quantales
    "Quantale",
    "ProductFuzzy",
    "BooleanQuantale",
    "PRODUCT_FUZZY",
    "BOOLEAN",
    # extra_quantales
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
    "RepeatMorphism",
    "morphism",
    "observed",
    "identity",
    # tensor_ops
    "noisy_or_contract",
    "noisy_or_reduce",
    "noisy_and_reduce",
    "componentwise_lift",
]
