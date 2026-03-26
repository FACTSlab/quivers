"""Stochastic morphisms: Markov kernels over finite sets.

This module provides the probabilistic layer for quivers, implementing
the category **FinStoch** of finite sets and stochastic maps (Markov
kernels). This is the Kleisli category of the Giry monad restricted
to finite sets, where:

- Objects are finite sets (same as the base category).
- Morphisms A -> B are stochastic matrices: tensors of shape
  (|A|, |B|) whose rows sum to 1.
- Composition is standard matrix multiplication.
- Identity is the Kronecker delta.

Submodules
----------
quantale : MarkovQuantale enrichment algebra
morphisms : StochasticMorphism, CategoricalMorphism
families : Discretized distribution families
transforms : condition, mix, factor, normalize
queries : prob, marginal_prob, expectation
giry : GiryMonad, FinStoch
deduction : Abstract weighted deductive system framework
schema : Composable rule schemas (functors CategorySystem -> RuleSystem)
span : Span-based CKY components (LexicalAxiom, BinarySpanDeduction, etc.)
parsers : ChartParser (DeductiveSystem subclass)
semiring : Chart semiring abstractions (LogProb, Viterbi, Boolean, Counting)
categories : Category types including atoms, slashes, products, unit, modals
"""

from __future__ import annotations

from quivers.stochastic.quantale import MarkovQuantale, MARKOV
from quivers.stochastic.morphisms import (
    StochasticMorphism,
    CategoricalMorphism,
    stochastic,
)
from quivers.stochastic.families import (
    DiscretizedNormal,
    DiscretizedLogitNormal,
    DiscretizedBeta,
    DiscretizedTruncatedNormal,
)
from quivers.stochastic.transforms import (
    ConditionedMorphism,
    condition,
    MixtureMorphism,
    mix,
    FactoredMorphism,
    factor,
    NormalizedMorphism,
    normalize,
)
from quivers.stochastic.queries import prob, marginal_prob, expectation
from quivers.stochastic.giry import GiryMonad, FinStoch
from quivers.stochastic.inside import InsideAlgorithm
from quivers.stochastic.categories import (
    AtomicCategory,
    SlashCategory,
    ProductCategory,
    UnitCategory,
    ModalCategory,
    CategorySystem,
)
from quivers.stochastic.semiring import (
    ChartSemiring,
    LogProbSemiring,
    ViterbiSemiring,
    BooleanSemiring,
    CountingSemiring,
    LOG_PROB,
    VITERBI,
    BOOLEAN,
    COUNTING,
)

# abstract deduction framework
from quivers.stochastic.deduction import (
    Axiom,
    Deduction,
    Goal,
    Schedule,
    DeductiveSystem,
)

# rule schemas (composable functors CategorySystem -> RuleSystem)
from quivers.stochastic.schema import (
    RuleSchema,
    BinaryRuleSchema,
    UnaryRuleSchema,
    UnionSchema,
    WeightedSchema,
    # atomic binary schemas
    ForwardApplication,
    BackwardApplication,
    ForwardComposition,
    BackwardComposition,
    ForwardCrossedComposition,
    BackwardCrossedComposition,
    CommutativeForwardApplication,
    CommutativeBackwardApplication,
    TensorIntroduction,
    LeftUnitElimination,
    RightUnitElimination,
    ModalApplication,
    # atomic unary schemas
    RightLifting,
    LeftLifting,
    LeftProjection,
    RightProjection,
    UnitCoercion,
    ModalInjection,
    ModalProjection,
    GeneralizedComposition,
    # bundled schemas
    EVALUATION,
    HARMONIC_COMPOSITION,
    CROSSED_COMPOSITION,
    COMMUTATIVE_EVALUATION,
    ADJUNCTION_UNITS,
    TENSOR_INTRODUCTION,
    TENSOR_PROJECTION,
    UNIT_INTRODUCTION,
    UNIT_ELIMINATION,
    MODAL_INTRODUCTION,
    MODAL_ELIMINATION,
    MODAL_APPLICATION,
    # grammar presets
    CCG,
    LAMBEK,
    NL,
    LP,
    SCHEMA_REGISTRY,
)

# span-based CKY components
from quivers.stochastic.span import (
    LexicalAxiom,
    BinarySpanDeduction,
    UnarySpanDeduction,
    SpanGoal,
    CKYSchedule,
)

# rule systems and parsers
from quivers.stochastic._rule_system import RuleSystem
from quivers.stochastic.rules import ccg_rules, lambek_rules, custom_rules
from quivers.stochastic.parsers import ChartParser
from quivers.stochastic.ccg import CCGParser
from quivers.stochastic.lambek import LambekParser

__all__ = [
    "MarkovQuantale",
    "MARKOV",
    "StochasticMorphism",
    "CategoricalMorphism",
    "stochastic",
    "DiscretizedNormal",
    "DiscretizedLogitNormal",
    "DiscretizedBeta",
    "DiscretizedTruncatedNormal",
    "ConditionedMorphism",
    "condition",
    "MixtureMorphism",
    "mix",
    "FactoredMorphism",
    "factor",
    "NormalizedMorphism",
    "normalize",
    "prob",
    "marginal_prob",
    "expectation",
    "GiryMonad",
    "FinStoch",
    "InsideAlgorithm",
    # category types
    "AtomicCategory",
    "SlashCategory",
    "ProductCategory",
    "UnitCategory",
    "ModalCategory",
    "CategorySystem",
    # semirings
    "ChartSemiring",
    "LogProbSemiring",
    "ViterbiSemiring",
    "BooleanSemiring",
    "CountingSemiring",
    "LOG_PROB",
    "VITERBI",
    "BOOLEAN",
    "COUNTING",
    # abstract deduction framework
    "Axiom",
    "Deduction",
    "Goal",
    "Schedule",
    "DeductiveSystem",
    # rule schemas
    "RuleSchema",
    "BinaryRuleSchema",
    "UnaryRuleSchema",
    "UnionSchema",
    "WeightedSchema",
    "ForwardApplication",
    "BackwardApplication",
    "ForwardComposition",
    "BackwardComposition",
    "ForwardCrossedComposition",
    "BackwardCrossedComposition",
    "CommutativeForwardApplication",
    "CommutativeBackwardApplication",
    "TensorIntroduction",
    "LeftUnitElimination",
    "RightUnitElimination",
    "ModalApplication",
    "RightLifting",
    "LeftLifting",
    "LeftProjection",
    "RightProjection",
    "UnitCoercion",
    "ModalInjection",
    "ModalProjection",
    "GeneralizedComposition",
    # bundled schemas
    "EVALUATION",
    "HARMONIC_COMPOSITION",
    "CROSSED_COMPOSITION",
    "COMMUTATIVE_EVALUATION",
    "ADJUNCTION_UNITS",
    "TENSOR_INTRODUCTION",
    "TENSOR_PROJECTION",
    "UNIT_INTRODUCTION",
    "UNIT_ELIMINATION",
    "MODAL_INTRODUCTION",
    "MODAL_ELIMINATION",
    "MODAL_APPLICATION",
    # grammar presets
    "CCG",
    "LAMBEK",
    "NL",
    "LP",
    "SCHEMA_REGISTRY",
    # span-based CKY components
    "LexicalAxiom",
    "BinarySpanDeduction",
    "UnarySpanDeduction",
    "SpanGoal",
    "CKYSchedule",
    # rule systems and parsers
    "RuleSystem",
    "ccg_rules",
    "lambek_rules",
    "custom_rules",
    "ChartParser",
    "CCGParser",
    "LambekParser",
]
