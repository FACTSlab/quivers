"""Re-export shim: ``MarkovQuantale`` and ``MARKOV`` now live in
:mod:`quivers.core.quantales`.

The Markov sum-product quantale moved into the core quantales
module so the core categorical layer (which other quantale-
homomorphism code already depends on) can reference it without
the stochastic subpackage being loaded first. This module stays
behind as a thin re-export so existing ``from quivers.stochastic.quantale
import MARKOV`` lines keep working.
"""

from quivers.core.quantales import MARKOV, MarkovQuantale

__all__ = ["MARKOV", "MarkovQuantale"]
