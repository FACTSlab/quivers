"""Variational guide families.

Public surface (re-exported by the parent :mod:`quivers.inference`
package): one ABC (:class:`Guide`) plus a zoo of concrete
``Auto*Guide`` subclasses spanning the standard variational-family
ladder from mean-field Normal to normalising-flow stacks and
hierarchical / mixture / structured guides.

Every concrete guide is built against a single
:class:`~quivers.inference.registry.LatentRegistry` and obeys the
shape contract documented on :class:`Guide`.
"""

from __future__ import annotations

from quivers.inference.guides.base import Guide
from quivers.inference.guides.delta import AutoDeltaGuide
from quivers.inference.guides.normal import AutoNormalGuide

__all__ = [
    "Guide",
    "AutoNormalGuide",
    "AutoDeltaGuide",
]
