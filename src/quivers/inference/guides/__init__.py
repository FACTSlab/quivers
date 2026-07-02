"""Variational guide families.

Public surface (re-exported by the parent `quivers.inference`
package): one ABC (`Guide`) plus a zoo of concrete
``Auto*Guide`` subclasses spanning the standard variational-family
ladder from mean-field Normal to normalizing-flow stacks and
hierarchical / mixture / structured guides.

Every concrete guide is built against a single
`quivers.inference.registry.LatentRegistry` and obeys the
shape contract documented on `Guide`.

Short-name aliases matching the Pyro `AutoGuide` naming
(`AutoNormal`, `AutoMultivariateNormal`, `AutoLowRankMVN`,
`AutoDelta`, `AutoLaplace`, `AutoIAFNormal`) are re-exported for
users switching over from Pyro. The longer ``*Guide`` names remain
the canonical quivers names.
"""

from __future__ import annotations

from quivers.inference.guides.auto_guide_list import AutoGuideList
from quivers.inference.guides.auto_structured import AutoStructured
from quivers.inference.guides.base import Guide
from quivers.inference.guides.delta import AutoDeltaGuide
from quivers.inference.guides.flow import (
    AutoIAFGuide,
    AutoNeuralSplineGuide,
    AutoNormalizingFlow,
)
from quivers.inference.guides.laplace import AutoLaplaceApproximation
from quivers.inference.guides.mixture import AutoMixtureGuide
from quivers.inference.guides.multivariate_normal import (
    AutoLowRankMultivariateNormalGuide,
    AutoMultivariateNormalGuide,
)
from quivers.inference.guides.normal import AutoNormalGuide

# Pyro-compatible short-name aliases.
AutoNormal = AutoNormalGuide
AutoMultivariateNormal = AutoMultivariateNormalGuide
AutoLowRankMVN = AutoLowRankMultivariateNormalGuide
AutoDelta = AutoDeltaGuide
AutoLaplace = AutoLaplaceApproximation
AutoIAFNormal = AutoIAFGuide

__all__ = [
    "Guide",
    "AutoNormalGuide",
    "AutoMultivariateNormalGuide",
    "AutoLowRankMultivariateNormalGuide",
    "AutoDeltaGuide",
    "AutoLaplaceApproximation",
    "AutoNormalizingFlow",
    "AutoIAFGuide",
    "AutoNeuralSplineGuide",
    "AutoMixtureGuide",
    "AutoGuideList",
    "AutoStructured",
    "AutoNormal",
    "AutoMultivariateNormal",
    "AutoLowRankMVN",
    "AutoDelta",
    "AutoLaplace",
    "AutoIAFNormal",
]
